# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import time

app = FastAPI(
    title="IRIS Prediction API",
    description="ML API for IRIS species classification",
    version="1.0.0"
)

# Global model variable
MODEL_PATH = "model_iris_bq.joblib"
model = None
model_load_time = None

@app.on_event("startup")
def load_model():
    """Load model on application startup"""
    global model, model_load_time
    start_time = time.time()
    
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        model_load_time = time.time() - start_time
        print(f"✅ Model loaded successfully in {model_load_time:.2f}s")
    else:
        raise FileNotFoundError(
            f"❌ Model file {MODEL_PATH} not found. Run 'dvc pull' first."
        )

class IrisInput(BaseModel):
    """Input schema for prediction"""
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    
    class Config:
        schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }

class IrisPrediction(BaseModel):
    """Output schema for prediction"""
    species: str
    species_id: int
    confidence: float = 1.0

@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "message": "IRIS Prediction API",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health():
    """Health check endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_load_time_seconds": model_load_time
    }

@app.post("/predict", response_model=IrisPrediction)
def predict(input_data: IrisInput):
    """
    Predict IRIS species from measurements
    
    - **sepal_length**: Sepal length in cm
    - **sepal_width**: Sepal width in cm
    - **petal_length**: Petal length in cm
    - **petal_width**: Petal width in cm
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Create DataFrame from input
        df = pd.DataFrame([input_data.dict()])
        
        # Make prediction
        prediction = model.predict(df)[0]
        
        # Map prediction to species name
        species_map = {
            0: "setosa",
            1: "versicolor",
            2: "virginica"
        }
        
        return IrisPrediction(
            species=species_map[prediction],
            species_id=int(prediction),
            confidence=1.0
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
