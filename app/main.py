import logging
from contextlib import asynccontextmanager

import joblib
import pandas as pd

from fastapi import FastAPI

from .schemas import CustomerRequest, PredictionResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    model = joblib.load(
        "model/underwriting_model.pkl"
    )

    logger.info("Model loaded successfully")

    yield

    logger.info("Application shutting down")


app = FastAPI(
    title="Underwriting ML API",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
def health():
    if model is None:
        return {
            "status": "unhealthy",
            "model_loaded": False
        }

    return {
        "status": "healthy",
        "model_loaded": True
    }

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(customer: CustomerRequest):


    logger.info(
    "Prediction request received: employment_type=%s, credit_score=%s",
    customer.employment_type,
    customer.credit_score
    )

    input_data = pd.DataFrame([
        customer.model_dump()
    ])

    probability = model.predict_proba(
        input_data
    )[0, 1]

    prediction = int(
        probability >= 0.379
    )

    if probability < 0.30:
        risk_class = "LOW"

    elif probability < 0.60:
        risk_class = "MEDIUM"

    else:
        risk_class = "HIGH"

    return {
        "default_probability": round(
            float(probability),
            4
        ),
        "risk_class": risk_class,
        "prediction": prediction
    }
