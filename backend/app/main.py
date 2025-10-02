"""
FastAPI application for Mumbai house price prediction.
Serves both HTML frontend and JSON API endpoints.
"""
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from functools import lru_cache

from app.model_loader import load_model, get_model_version
from app.schemas import PredictionRequest, PredictionResponse
from app.utils import (
    format_inr,
    prepare_features,
    map_locality,
    load_top_localities
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/predictions.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ensure logs directory exists
Path("logs").mkdir(exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="Mumbai House Price Predictor",
    description="Predict Mumbai property prices using ML",
    version="1.0.0"
)

# CORS middleware (tighten for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates
templates = Jinja2Templates(directory="app/templates")

# Global model and metadata
MODEL = None
MODEL_LOADED = False
MODEL_ERROR = None
TOP_LOCALITIES = []


@app.on_event("startup")
async def startup_event():
    """Load model and localities on startup."""
    global MODEL, MODEL_LOADED, MODEL_ERROR, TOP_LOCALITIES
    
    try:
        MODEL = load_model()
        MODEL_LOADED = True
        logger.info("Model loaded successfully")
    except Exception as e:
        MODEL_ERROR = str(e)
        MODEL_LOADED = False
        logger.error(f"Failed to load model: {e}")
    
    try:
        TOP_LOCALITIES = load_top_localities()
        logger.info(f"Loaded {len(TOP_LOCALITIES)} top localities")
    except Exception as e:
        logger.warning(f"Failed to load localities: {e}")
        TOP_LOCALITIES = []


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "ok": MODEL_LOADED,
        "model_loaded": MODEL_LOADED,
        "version": get_model_version(MODEL) if MODEL_LOADED else None,
        "error": MODEL_ERROR if not MODEL_LOADED else None
    }


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the HTML form."""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "localities": TOP_LOCALITIES,
            "model_loaded": MODEL_LOADED
        }
    )


@lru_cache(maxsize=128)
def cached_predict(payload_hash: str, **kwargs):
    """Cached prediction to speed up identical requests."""
    return predict_internal(**kwargs)


def predict_internal(
    area: float,
    total_rooms: int,
    Bedrooms: int,
    Bathrooms: int,
    Balcony: int,
    parking: int,
    Lift: bool,
    furnished_status: str,
    building_type: str,
    locality: str,
    new_or_resale: str,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None
) -> dict:
    """Internal prediction logic."""
    import numpy as np
    
    if not MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check /health endpoint."
        )
    
    # Map locality
    locality_mapped = map_locality(locality, TOP_LOCALITIES)
    
    # Prepare features
    X = prepare_features(
        area=area,
        total_rooms=total_rooms,
        Bedrooms=Bedrooms,
        Bathrooms=Bathrooms,
        Balcony=Balcony,
        parking=parking,
        Lift=1 if Lift else 0,
        furnished_status=furnished_status,
        building_type=building_type,
        locality=locality_mapped,
        new_or_resale=new_or_resale,
        latitude=latitude,
        longitude=longitude,
        model=MODEL
    )
    
    # Log prediction request (masked)
    logger.info(f"Prediction request: area={area}, locality={locality_mapped}, type={building_type}")
    
    try:
        # Make prediction
        pred_log = MODEL.predict(X)[0]
        
        # Handle case where model might return price directly
        if pred_log > 1000:
            logger.warning(f"Prediction value {pred_log} > 1000, assuming direct price")
            predicted_price = float(pred_log)
        else:
            predicted_price = float(np.exp(pred_log))
        
        # Round to nearest 100
        predicted_price = round(predicted_price / 100) * 100
        
        # Calculate price per sqft
        price_per_sqft = round(predicted_price / area, 2) if area > 0 else 0
        
        # Format price string
        predicted_price_str = format_inr(predicted_price)
        
        result = {
            "model_version": get_model_version(MODEL),
            "price_log": float(pred_log),
            "predicted_price": predicted_price,
            "predicted_price_str": predicted_price_str,
            "price_per_sqft": price_per_sqft
        }
        
        logger.info(f"Prediction result: {predicted_price_str}")
        return result
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict_json", response_model=PredictionResponse)
async def predict_json(request: PredictionRequest):
    """JSON API endpoint for predictions."""
    try:
        # Create hash for caching
        payload_dict = request.dict()
        payload_hash = str(hash(frozenset(payload_dict.items())))
        
        result = cached_predict(payload_hash, **payload_dict)
        
        # Add custom header
        response = JSONResponse(content=result)
        response.headers["X-Model-Version"] = result["model_version"]
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/predict", response_class=HTMLResponse)
async def predict_form(
    request: Request,
    area: float = Form(...),
    total_rooms: int = Form(...),
    Bedrooms: int = Form(...),
    Bathrooms: int = Form(...),
    Balcony: int = Form(0),
    parking: int = Form(0),
    Lift: str = Form("No"),
    furnished_status: str = Form(...),
    building_type: str = Form(...),
    locality: str = Form(...),
    new_or_resale: str = Form(...),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None)
):
    """Form POST handler returning HTML response."""
    try:
        # Convert Lift to boolean
        lift_bool = Lift.lower() in ['yes', 'true', '1']
        
        result = predict_internal(
            area=area,
            total_rooms=total_rooms,
            Bedrooms=Bedrooms,
            Bathrooms=Bathrooms,
            Balcony=Balcony,
            parking=parking,
            Lift=lift_bool,
            furnished_status=furnished_status,
            building_type=building_type,
            locality=locality,
            new_or_resale=new_or_resale,
            latitude=latitude,
            longitude=longitude
        )
        
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "localities": TOP_LOCALITIES,
                "model_loaded": MODEL_LOADED,
                "prediction": result
            }
        )
        
    except HTTPException as e:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "localities": TOP_LOCALITIES,
                "model_loaded": MODEL_LOADED,
                "error": e.detail
            }
        )