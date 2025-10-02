# Mumbai House Price Predictor

A FastAPI web application for predicting Mumbai property prices using a trained machine learning model. Includes a responsive HTML frontend with Tailwind CSS and a JSON API.

## Features

*  Real-time price predictions using a scikit-learn pipeline
*  Modern UI with Tailwind CSS
*  REST API with JSON endpoints
*  Input validation via Pydantic
*  Request logging
*  Comprehensive tests with pytest


## Prerequisites

* Python 3.10+
* Trained scikit-learn pipeline saved as `mumbai_price_model_pipeline.joblib`

## Local Setup

```bash
python -m venv .venv
# Activate environment
source .venv/bin/activate  # Unix/macOS
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

Place your trained model file in the repository root:

```bash
mumbai_price_model_pipeline.joblib
```

Start the server:

```bash
uvicorn backend.app.main:app --reload --port 5000
```

* Web UI: [http://localhost:5000](http://localhost:5000)
* API docs: [http://localhost:5000/docs](http://localhost:5000/docs)
* Health check: [http://localhost:5000/health](http://localhost:5000/health)

## API Usage

**POST /predict_json**

Request Body:

```json
{
  "area": 1000.0,
  "total_rooms": 3,
  "Bedrooms": 2,
  "Bathrooms": 2,
  "Balcony": 1,
  "parking": 1,
  "Lift": true,
  "furnished_status": "Semi-Furnished",
  "building_type": "Apartment",
  "locality": "Mira Road",
  "new_or_resale": "Resale",
  "latitude": 19.2855,
  "longitude": 72.8558
}
```

Response:

```json
{
  "model_version": "1.0",
  "price_log": 15.424948,
  "predicted_price": 5000000.0,
  "predicted_price_str": "₹ 50.00 L",
  "price_per_sqft": 5000.0
}
```

Other endpoints:

* `GET /` - Web UI form
* `POST /predict` - Form submission
* `GET /health` - Health check

## Input Fields

| Field            | Type  | Required | Description                                                 | Example          |
| ---------------- | ----- | -------- | ----------------------------------------------------------- | ---------------- |
| area             | float | Yes      | Property size in sqft                                       | 1000.0           |
| total_rooms      | int   | Yes      | Total rooms                                                 | 3                |
| Bedrooms         | int   | Yes      | Bedrooms                                                    | 2                |
| Bathrooms        | int   | Yes      | Bathrooms                                                   | 2                |
| Balcony          | int   | No       | Number of balconies                                         | 1                |
| parking          | int   | No       | Parking slots                                               | 1                |
| Lift             | bool  | Yes      | Lift available                                              | true             |
| furnished_status | str   | Yes      | "Unfurnished", "Semi-Furnished", "Furnished"                | "Semi-Furnished" |
| building_type    | str   | Yes      | "Apartment", "Individual House", "Villa", "Studio", "Other" | "Apartment"      |
| locality         | str   | Yes      | Mumbai locality                                             | "Mira Road"      |
| new_or_resale    | str   | Yes      | "New" or "Resale"                                           | "Resale"         |
| latitude         | float | No       | -90 to 90                                                   | 19.2855          |
| longitude        | float | No       | -180 to 180                                                 | 72.8558          |

## Model Requirements

* scikit-learn pipeline saved with `joblib.dump()`
* Predicts log(price); app exponentiates output
* Handles numeric and categorical features:

  * Numeric: `area, total_rooms, Bedrooms, Bathrooms, Balcony, parking, Lift`
  * Categorical: `furnished_status, building_type, locality, new_or_resale`
* Optional: `latitude, longitude`
* Preprocessing (one-hot encoding, scaling) included in pipeline

## Logging

* Logs stored in `logs/predictions.log`
* Format:

```
2025-10-02 10:30:45 - backend.app.main - INFO - Prediction request: area=1000, locality=Mira Road, type=Apartment
2025-10-02 10:30:45 - backend.app.main - INFO - Prediction result: ₹ 50.00 L
```

