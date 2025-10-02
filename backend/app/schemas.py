"""
Pydantic models for request/response validation.
"""
from typing import Optional, Literal
from pydantic import BaseModel, Field, validator


class PredictionRequest(BaseModel):
    """Request model for price prediction."""
    
    area: float = Field(..., gt=0, le=50000, description="Property size in sqft")
    total_rooms: int = Field(..., ge=1, le=20, description="Total number of rooms")
    Bedrooms: int = Field(..., ge=0, le=15, description="Number of bedrooms")
    Bathrooms: int = Field(..., ge=0, le=10, description="Number of bathrooms")
    Balcony: int = Field(0, ge=0, le=10, description="Number of balconies")
    parking: int = Field(0, ge=0, le=10, description="Number of parking slots")
    Lift: bool = Field(..., description="Lift available (Yes/No)")
    furnished_status: Literal["Unfurnished", "Semi-Furnished", "Furnished"] = Field(
        ..., description="Furnishing status"
    )
    building_type: Literal["Apartment", "Individual House", "Villa", "Studio", "Other"] = Field(
        ..., description="Type of building"
    )
    locality: str = Field(..., min_length=1, max_length=100, description="Locality name")
    new_or_resale: Literal["New", "Resale"] = Field(..., description="Property condition")
    latitude: Optional[float] = Field(None, ge=-90, le=90, description="Latitude coordinate")
    longitude: Optional[float] = Field(None, ge=-180, le=180, description="Longitude coordinate")
    
    @validator('area')
    def validate_area(cls, v):
        """Validate area is reasonable."""
        if v < 100:
            raise ValueError("Area should be at least 100 sqft")
        return v
    
    @validator('Bedrooms')
    def validate_bedrooms(cls, v, values):
        """Validate bedrooms don't exceed total rooms."""
        if 'total_rooms' in values and v > values['total_rooms']:
            raise ValueError("Bedrooms cannot exceed total rooms")
        return v
    
    @validator('locality')
    def validate_locality(cls, v):
        """Sanitize locality input."""
        return v.strip()
    
    class Config:
        schema_extra = {
            "example": {
                "area": 1000.0,
                "total_rooms": 3,
                "Bedrooms": 2,
                "Bathrooms": 2,
                "Balcony": 1,
                "parking": 1,
                "Lift": True,
                "furnished_status": "Semi-Furnished",
                "building_type": "Apartment",
                "locality": "Mira Road",
                "new_or_resale": "Resale",
                "latitude": 19.2855,
                "longitude": 72.8558
            }
        }


class PredictionResponse(BaseModel):
    """Response model for price prediction."""
    
    model_version: str = Field(..., description="Model version identifier")
    price_log: float = Field(..., description="Log-transformed predicted price")
    predicted_price: float = Field(..., description="Predicted price in INR")
    predicted_price_str: str = Field(..., description="Formatted price string")
    price_per_sqft: float = Field(..., description="Price per square foot")
    
    class Config:
        schema_extra = {
            "example": {
                "model_version": "1.0",
                "price_log": 15.45,
                "predicted_price": 5000000.0,
                "predicted_price_str": "₹ 50,00,000",
                "price_per_sqft": 5000.0
            }
        }


class HealthResponse(BaseModel):
    """Health check response model."""
    
    ok: bool
    model_loaded: bool
    version: Optional[str] = None
    error: Optional[str] = None