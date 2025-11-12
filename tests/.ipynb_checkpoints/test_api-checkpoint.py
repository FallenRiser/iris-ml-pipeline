from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import random

app = FastAPI(
    title="Iris Classifier API",
    version="1.0.0"
)

# --- Simulate model loading ---
# In a real app, you might load a model from disk here.
model_loaded = True

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# --- Root Endpoint ---
@app.get("/")
def root():
    return {
        "message": "Welcome to the Iris Classifier API",
        "version": "1.0.0"
    }


# --- Health Check Endpoint ---
@app.get("/health")
def health_check():
    if not model_loaded:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "model_loaded": False}
        )
    return {"status": "healthy", "model_loaded": True}


# --- Predict Endpoint ---
@app.post("/predict")
def predict(iris: IrisInput):
    # In a real model, you'd use the features to predict.
    # Here we simulate a random prediction.
    species_list = ["setosa", "versicolor", "virginica"]
    species_id = random.randint(0, 2)
    species = species_list[species_id]
    
    return {
        "species": species,
        "species_id": species_id
    }
