"""
Utility functions for data processing and formatting.
"""
import json
import logging
from pathlib import Path
from typing import Any, Optional, List
import pandas as pd

logger = logging.getLogger(__name__)


def load_top_localities(path: str = "data/top_localities.json") -> List[str]:
    """
    Load top localities from JSON file.
    
    Args:
        path: Path to localities JSON file
        
    Returns:
        List of locality names
    """
    try:
        with open(path, 'r') as f:
            localities = json.load(f)
        return localities
    except FileNotFoundError:
        logger.warning(f"Localities file not found at {path}")
        return []
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in {path}")
        return []


def map_locality(locality: str, top_localities: List[str]) -> str:
    """
    Map user-provided locality to known locality or __OTHER__.
    
    Args:
        locality: User input locality
        top_localities: List of known localities
        
    Returns:
        Mapped locality name or "__OTHER__"
    """
    if not locality:
        return "__OTHER__"
    
    # Normalize input
    locality_normalized = locality.strip().lower()
    
    # Case-insensitive matching
    for known_locality in top_localities:
        if known_locality.lower() == locality_normalized:
            return known_locality
    
    # Not found in top localities
    logger.debug(f"Locality '{locality}' not in top list, mapping to __OTHER__")
    return "__OTHER__"


def format_inr(amount: float) -> str:
    """
    Format amount in Indian Rupee style with ₹ symbol.
    
    Args:
        amount: Amount to format
        
    Returns:
        Formatted string (e.g., "₹ 1,23,45,678")
    """
    if amount >= 10000000:  # 1 Crore or more
        crores = amount / 10000000
        return f"₹ {crores:.2f} Cr"
    elif amount >= 100000:  # 1 Lakh or more
        lakhs = amount / 100000
        return f"₹ {lakhs:.2f} L"
    else:
        # Standard formatting with commas
        return f"₹ {amount:,.0f}"


def prepare_features(
    area: float,
    total_rooms: int,
    Bedrooms: int,
    Bathrooms: int,
    Balcony: int,
    parking: int,
    Lift: int,
    furnished_status: str,
    building_type: str,
    locality: str,
    new_or_resale: str,
    latitude: Optional[float],
    longitude: Optional[float],
    model: Any
) -> pd.DataFrame:
    """
    Prepare feature DataFrame for model prediction.
    
    Args:
        Various input features
        model: Trained model object
        
    Returns:
        pandas DataFrame ready for prediction
    """
    # Build base feature dictionary
    features = {
        'area': area,
        'total_rooms': total_rooms,
        'Bedrooms': Bedrooms,
        'Bathrooms': Bathrooms,
        'Balcony': Balcony,
        'parking': parking,
        'Lift': Lift,
        'furnished_status': furnished_status,
        'building_type': building_type,
        'locality': locality,
        'new_or_resale': new_or_resale
    }
    
    # Add optional coordinates if provided
    if latitude is not None:
        features['latitude'] = latitude
    if longitude is not None:
        features['longitude'] = longitude
    
    # Create DataFrame
    X = pd.DataFrame([features])
    
    # Check if model expects specific features
    if hasattr(model, 'feature_names_in_'):
        expected_features = list(model.feature_names_in_)
        logger.debug(f"Model expects {len(expected_features)} features")
        
        # Handle potential one-hot encoded categorical variables
        # If model expects one-hot columns (e.g., locality_MiraRoad, locality_Andheri)
        # we need to create them
        
        locality_cols = [f for f in expected_features if f.startswith('locality_')]
        furnished_cols = [f for f in expected_features if f.startswith('furnished_status_')]
        building_cols = [f for f in expected_features if f.startswith('building_type_')]
        new_or_resale_cols = [f for f in expected_features if f.startswith('new_or_resale_')]
        
        # If one-hot encoded columns exist, create them
        if locality_cols:
            for col in locality_cols:
                X[col] = 0
            # Set the matching column to 1
            locality_col = f"locality_{locality.replace(' ', '')}"
            if locality_col in locality_cols:
                X[locality_col] = 1
            # Drop original locality column if it exists
            if 'locality' in X.columns and 'locality' not in expected_features:
                X = X.drop('locality', axis=1)
        
        if furnished_cols:
            for col in furnished_cols:
                X[col] = 0
            furnished_col = f"furnished_status_{furnished_status.replace('-', '')}"
            if furnished_col in furnished_cols:
                X[furnished_col] = 1
            if 'furnished_status' in X.columns and 'furnished_status' not in expected_features:
                X = X.drop('furnished_status', axis=1)
        
        if building_cols:
            for col in building_cols:
                X[col] = 0
            building_col = f"building_type_{building_type.replace(' ', '')}"
            if building_col in building_cols:
                X[building_col] = 1
            if 'building_type' in X.columns and 'building_type' not in expected_features:
                X = X.drop('building_type', axis=1)
        
        if new_or_resale_cols:
            for col in new_or_resale_cols:
                X[col] = 0
            resale_col = f"new_or_resale_{new_or_resale}"
            if resale_col in new_or_resale_cols:
                X[resale_col] = 1
            if 'new_or_resale' in X.columns and 'new_or_resale' not in expected_features:
                X = X.drop('new_or_resale', axis=1)
        
        # Add any missing expected features with defaults
        for feature in expected_features:
            if feature not in X.columns:
                X[feature] = 0
                logger.debug(f"Added missing feature: {feature}")
        
        # Reorder columns to match expected order
        X = X[expected_features]
    
    return X