
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_DIR / "models"

logistic_model = joblib.load(
    MODELS_DIR / "logistic_regression.joblib"
)
random_forest_model = joblib.load(
    MODELS_DIR / "random_forest.joblib"
)

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal"
]

app = FastAPI(
    title="Heart Disease Classification API",
    description="Educational ML model prediction API",
    version="1.0.0"
)


class PatientFeatures(BaseModel):
    age: float = Field(ge=1, le=120)
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float


class PredictionRequest(BaseModel):
    model_name: str = Field(
        default="logistic_regression",
        pattern="^(logistic_regression|random_forest)$"
    )
    features: PatientFeatures


@app.get("/")
def home():
    return {
        "message": "Heart Disease Classification API is running",
        "available_models": [
            "logistic_regression",
            "random_forest"
        ]
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        model = (
            logistic_model
            if request.model_name == "logistic_regression"
            else random_forest_model
        )

        input_data = pd.DataFrame(
            [request.features.model_dump()],
            columns=FEATURES
        )

        prediction = int(model.predict(input_data)[0])
        probabilities = model.predict_proba(input_data)[0]

        return {
            "model": request.model_name,
            "prediction": prediction,
            "class_probabilities": {
                str(int(label)): float(probability)
                for label, probability in zip(
                    model.classes_, probabilities
                )
            },
            "interpretation": (
                "Model predicted the positive class"
                if prediction == 1
                else "Model predicted the negative class"
            ),
            "disclaimer": (
                "Educational demonstration only; "
                "not a medical diagnosis."
            )
        }
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(exc)}"
        )
