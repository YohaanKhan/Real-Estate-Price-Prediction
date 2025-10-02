"""
Model loading utilities with error handling and version detection.
"""
import os
import logging
from pathlib import Path
from typing import Any, Optional
import joblib

logger = logging.getLogger(__name__)


def load_model(model_path: Optional[str] = None) -> Any:
    """
    Load the trained model pipeline from disk.
    
    Args:
        model_path: Path to model file. If None, uses MODEL_PATH env var
                   or defaults to /app/mumbai_price_model_pipeline.joblib
    
    Returns:
        Loaded model object
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        Exception: If model loading fails
    """
    if model_path is None:
        model_path = os.getenv(
            'MODEL_PATH',
            'mumbai_price_model_pipeline.joblib'
        )
    
    model_path = Path(model_path)
    
    if not model_path.exists():
        error_msg = f"Model file not found at {model_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    logger.info(f"Loading model from {model_path}")
    
    try:
        model = joblib.load(model_path)
        logger.info(f"Model loaded successfully. Type: {type(model).__name__}")
        
        # Log model metadata if available
        if hasattr(model, 'feature_names_in_'):
            logger.info(f"Model expects {len(model.feature_names_in_)} features")
            logger.debug(f"Expected features: {model.feature_names_in_}")
        
        return model
        
    except Exception as e:
        error_msg = f"Failed to load model: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)


def get_model_version(model: Any) -> str:
    """
    Extract version information from model object.
    
    Args:
        model: Loaded model object
        
    Returns:
        Version string (e.g., "1.0", timestamp, or sha)
    """
    # Try multiple common version attributes
    version_attrs = ['version', 'model_version', '_version', 'meta']
    
    for attr in version_attrs:
        if hasattr(model, attr):
            version = getattr(model, attr)
            if version:
                return str(version)
    
    # Check if model has __dict__ with version info
    if hasattr(model, '__dict__'):
        model_dict = model.__dict__
        for attr in version_attrs:
            if attr in model_dict and model_dict[attr]:
                return str(model_dict[attr])
        
        # Check for 'meta' dict with version
        if 'meta' in model_dict and isinstance(model_dict['meta'], dict):
            if 'version' in model_dict['meta']:
                return str(model_dict['meta']['version'])
    
    # Default version
    return "1.0"


def validate_model(model: Any) -> bool:
    """
    Validate that the model has the expected methods.
    
    Args:
        model: Model object to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_methods = ['predict']
    
    for method in required_methods:
        if not hasattr(model, method):
            logger.error(f"Model missing required method: {method}")
            return False
    
    logger.info("Model validation passed")
    return True