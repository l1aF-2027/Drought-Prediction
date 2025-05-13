import numpy as np
import pandas as pd
import logging
from datetime import datetime
from scipy.interpolate import interp1d
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

def normalize(X_static, X_time, scaler_dict=None, scaler_dict_static=None):
    """
    Normalize time series and static data using the pre-fitted scalers
    
    Args:
        X_static: Static data of shape (batch_size, static_dim)
        X_time: Time series data of shape (batch_size, seq_len, time_dim)
        scaler_dict: Dictionary of scalers for time series data
        scaler_dict_static: Dictionary of scalers for static data
        
    Returns:
        X_static_norm: Normalized static data
        X_time_norm: Normalized time series data
    """
    # Make a copy to avoid modifying the original data
    X_static_norm = X_static.copy()
    X_time_norm = X_time.copy()
    
    # Check if we have scalers available
    if not scaler_dict or not scaler_dict_static:
        logger.warning("Scalers not available. Returning unnormalized data.")
        return X_static_norm, X_time_norm
    
    try:
        # Normalize time series data
        for index in range(X_time_norm.shape[-1]):
            if index in scaler_dict:
                try:
                    X_time_norm[:, :, index] = (
                        scaler_dict[index]
                        .transform(X_time_norm[:, :, index].reshape(-1, 1))
                        .reshape(-1, X_time_norm.shape[-2])
                    )
                except Exception as e:
                    logger.warning(f"Error normalizing time feature at index {index}: {str(e)}")
        
        # Normalize static data
        for index in range(X_static_norm.shape[-1]):
            if index in scaler_dict_static:
                try:
                    X_static_norm[:, index] = (
                        scaler_dict_static[index]
                        .transform(X_static_norm[:, index].reshape(-1, 1))
                        .reshape(1, -1)
                    )
                except Exception as e:
                    logger.warning(f"Error normalizing static feature at index {index}: {str(e)}")
    
    except Exception as e:
        logger.error(f"Normalization error: {str(e)}")
        # Return the original data if normalization fails
        return X_static, X_time
    
    return X_static_norm, X_time_norm

def interpolate_nans(padata, pkind='linear'):
    """
    Interpolate missing values in an array
    
    Args:
        padata: Array with possible NaN values
        pkind: Kind of interpolation ('linear', 'cubic', etc.)
        
    Returns:
        interpolated_data: Array with NaN values interpolated
    """
    try:
        aindexes = np.arange(padata.shape[0])
        agood_indexes, = np.where(np.isfinite(padata))
        
        # If all values are NaN or there's only one good value, return zeros
        if len(agood_indexes) == 0:
            return np.zeros_like(padata)
        elif len(agood_indexes) == 1:
            # If there's only one good value, fill with that value
            result = np.full_like(padata, padata[agood_indexes[0]])
            return result
        
        # Interpolate
        f = interp1d(
            agood_indexes,
            padata[agood_indexes],
            bounds_error=False,
            copy=False,
            fill_value="extrapolate",
            kind=pkind
        )
        
        return f(aindexes)
    except Exception as e:
        logger.error(f"Error interpolating NaNs: {str(e)}")
        # Return zeros as fallback
        return np.zeros_like(padata)

def date_encode(date):
    """
    Encode date as sine and cosine components to capture cyclical patterns
    
    Args:
        date: Date to encode, can be string or datetime object
        
    Returns:
        sin_day: Sine component of day of year
        cos_day: Cosine component of day of year
    """
    try:
        if isinstance(date, str):
            date = datetime.strptime(date, "%Y-%m-%d")
        
        # Get day of year (1-366)
        day_of_year = date.timetuple().tm_yday
        
        # Encode as sine and cosine
        sin_day = np.sin(2 * np.pi * day_of_year / 366)
        cos_day = np.cos(2 * np.pi * day_of_year / 366)
        
        return sin_day, cos_day
    except Exception as e:
        logger.error(f"Error encoding date: {str(e)}")
        # Return default values as fallback
        return 0.0, 1.0