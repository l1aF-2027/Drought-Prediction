from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import pandas as pd
import numpy as np
import torch
import json
import io
import joblib
import os
import sys
import logging
from model import TimeSeriesTransformer
from utils import normalize, date_encode, interpolate_nans
from datetime import datetime
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler_dict, scaler_dict_static, device
    
    try:
        logger.info("Starting application initialization")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Load scalers with safety measures for version compatibility
        try:
            logger.info("Loading scalers")
            scaler_dict = joblib.load(os.path.join(os.path.dirname(__file__), "scaler_dict.joblib"))
            scaler_dict_static = joblib.load(os.path.join(os.path.dirname(__file__), "scaler_dict_static.joblib"))
            logger.info("Scalers loaded successfully")
        except Exception as e:
            logger.error(f"Error loading scalers: {str(e)}")
            # Provide fallback empty dictionaries if loading fails
            scaler_dict = {}
            scaler_dict_static = {}
            logger.warning("Using empty scalers as fallback")

        # Define model params
        logger.info("Initializing model")
        batch_size = 256
        one_cycle = True
        lr = 5e-5
        epochs = 15
        clip = 5
        model_dim=256
        n_head=4
        dropout=0.15
        num_layers=2
        staticfc_dim=16
        hidden_dim=256
        output_size = 6

        model = TimeSeriesTransformer(
            21,
            model_dim,
            175,
            n_head,
            dropout,
            num_layers,
            29, 
            staticfc_dim,
            hidden_dim,
            output_size
        )
        
        try:
            model_path = os.path.join(os.path.dirname(__file__), "best_avg_mae_model.pt")
            logger.info(f"Loading model from {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            logger.info("Model loaded and initialized successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise  # Re-raise to prevent app from starting with broken model
            
        logger.info("Application initialization completed successfully")
        
        yield  # Allow app to run
        
        logger.info("Application shutdown initiated")
    except Exception as e:
        logger.error(f"Critical error during initialization: {str(e)}")
        # Still yield to allow proper error handling
        yield
        logger.info("Application shutdown after initialization error")

app = FastAPI(
    title="Drought Prediction API",
    description="API for predicting drought severity based on weather data",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to Drought Prediction API. Use /predict endpoint to make predictions."}

@app.get("/health")
async def health():
    """Simple health check endpoint"""
    return {"status": "ok", "model_loaded": model is not None}

def parse_static_input(x_static: str):
    """Parse the static input string, handling both JSON and comma-separated formats."""
    try:
        # First try to parse as JSON
        return json.loads(x_static)
    except json.JSONDecodeError:
        # If that fails, try parsing as comma-separated values
        try:
            return [float(x.strip()) for x in x_static.split(',')]
        except ValueError as e:
            logger.error(f"Error parsing x_static as comma-separated values: {str(e)}")
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid x_static format. Must be valid JSON array or comma-separated values. Error: {str(e)}"
            )

@app.post("/predict")
async def predict(
    csv_file: UploadFile = File(...),
    x_static: str = Form(...),
    dsci_file: UploadFile = File(...),
):
    try:
        logger.info("Received prediction request")
        
        # Parse static input with the new helper function
        x_static_list = parse_static_input(x_static)
        x_static_array = np.array([x_static_list], dtype=np.float32)
        logger.info(f"Static data shape: {x_static_array.shape}")

        # Load and process CSV
        dsci_content = await dsci_file.read()
        dsci_df = pd.read_csv(io.StringIO(dsci_content.decode('utf-8')))
        dsci_df = dsci_df[["MapDate", "DSCI"]].copy()
        dsci_df["DSCI"] = dsci_df["DSCI"] / 100.0
        dsci_df["MapDate"] = dsci_df["MapDate"].astype(str).astype(int)

        # Load weather CSV
        weather_content = await csv_file.read()
        
        weather_df = pd.read_csv(io.StringIO(weather_content.decode('utf-8')), skiprows=26)

        # Tạo MapDate từ YEAR và DOY
        weather_df["DOY"] = weather_df["DOY"].astype(str).str.zfill(3)
        weather_df["Date"] = pd.to_datetime(weather_df["YEAR"].astype(str), format='%Y') + \
                            pd.to_timedelta(weather_df["DOY"].astype(int) - 1, unit='D')
        weather_df["MapDate"] = weather_df["Date"].dt.strftime("%Y%m%d").astype(int)

        # Merge với dsci
        merged_df = pd.merge(weather_df, dsci_df, on="MapDate", how="left")
        merged_df.rename(columns={"DSCI": "score"}, inplace=True)
        merged_df["score"] = interpolate_nans(merged_df["score"].values)
        
        logger.info(f"Loaded CSV with shape: {merged_df.shape}")
        
        df = prepare_time_data(merged_df)
        
        logger.info("Time data prepared successfully")

        # Feature extraction
        float_cols = [
            'PRECTOTCORR', 'PS', 'QV2M', 'T2M', 'T2MDEW', 'T2MWET', 'T2M_MAX', 'T2M_MIN', 'T2M_RANGE',
            'TS', 'WS10M', 'WS10M_MAX', 'WS10M_MIN', 'WS10M_RANGE',
            'WS50M', 'WS50M_MAX', 'WS50M_MIN', 'WS50M_RANGE', 'score'
        ]
        features = float_cols + ['sin_day', 'cos_day']
        x_time_array = df[features].to_numpy(dtype=np.float32)
        x_time_array = np.expand_dims(x_time_array, axis=0)
        logger.info(f"Time features shape: {x_time_array.shape}")

        # Normalize
        try:
            x_static_norm, x_time_norm = normalize(
                x_static_array,
                x_time_array,
                scaler_dict=scaler_dict,
                scaler_dict_static=scaler_dict_static
            )
            logger.info("Data normalized successfully")
        except Exception as norm_error:
            logger.error(f"Normalization error: {str(norm_error)}")
            # Fall back to using unnormalized data if normalization fails
            logger.warning("Using unnormalized data as fallback")
            x_static_norm = x_static_array
            x_time_norm = x_time_array

        x_time_tensor = torch.tensor(x_time_norm).float().to(device)
        x_static_tensor = torch.tensor(x_static_norm).float().to(device)
        # Predict
        logger.info("Running prediction")
        with torch.no_grad():
            output = model(x_time_tensor, x_static_tensor)
            output = torch.clamp(output, min=0.0, max=5.0)

        predictions = output.cpu().numpy().tolist()[0]
        logger.info(f"Prediction completed: {predictions}")

        result = {
            "raw_predictions": predictions,
        }

        logger.info("Returning prediction result")
        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


def prepare_time_data(df):
    try:
        if 'YEAR' not in df.columns or 'DOY' not in df.columns:
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df['YEAR'] = df['date'].dt.year
                df['DOY'] = df['date'].dt.dayofyear
            else:
                raise ValueError("Input CSV must contain either 'date' column or both 'YEAR' and 'DOY' columns")
        df["DOY"] = df["DOY"].astype(str).str.zfill(3)
        if 'date' not in df.columns:
            df['date'] = pd.to_datetime(df['YEAR'].astype(str) + df['DOY'].astype(str), format="%Y%j")

        df[['sin_day', 'cos_day']] = df['date'].apply(lambda d: pd.Series(date_encode(d)))

        float_cols = [
            'PRECTOTCORR', 'PS', 'QV2M', 'T2M', 'T2MDEW', 'T2MWET', 'T2M_MAX', 'T2M_MIN', 'T2M_RANGE',
            'TS', 'WS10M', 'WS10M_MAX', 'WS10M_MIN', 'WS10M_RANGE',
            'WS50M', 'WS50M_MAX', 'WS50M_MIN', 'WS50M_RANGE', 'score'
        ]
        for col in float_cols:
            if col in df.columns and df[col].isna().any():
                df[col] = interpolate_nans(df[col].values)

        return df
    except Exception as e:
        logger.error(f"Error preparing time data: {str(e)}")
        raise
    
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7865))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)