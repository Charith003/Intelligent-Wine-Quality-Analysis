from pathlib import Path
from typing import Literal
import math
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "wine_quality_et_pipeline.joblib"

app = FastAPI(
    title="WineSense AI API",
    description="Wine quality tier prediction using the notebook's Extra Trees model.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

try:
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
except Exception as exc:
    raise RuntimeError(
        f"Could not load model at {MODEL_PATH}. Run `python train_model.py` from backend/."
    ) from exc

TIER_LABELS = {0: "Low (3-4)", 1: "Average (5-6)", 2: "High (7-9)"}

class WineInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    wine_type: Literal["red", "white"]
    fixed_acidity: float = Field(alias="fixed acidity", ge=0, le=30)
    volatile_acidity: float = Field(alias="volatile acidity", ge=0, le=5)
    citric_acid: float = Field(alias="citric acid", ge=0, le=5)
    residual_sugar: float = Field(alias="residual sugar", ge=0, le=100)
    chlorides: float = Field(ge=0, le=2)
    free_sulfur_dioxide: float = Field(alias="free sulfur dioxide", ge=0, le=500)
    total_sulfur_dioxide: float = Field(alias="total sulfur dioxide", ge=0, le=1000)
    density: float = Field(gt=0.8, le=1.2)
    pH: float = Field(alias="pH", ge=0, le=14)
    sulphates: float = Field(ge=0, le=5)
    alcohol: float = Field(ge=0, le=30)

    def as_features(self) -> pd.DataFrame:
        if self.total_sulfur_dioxide < self.free_sulfur_dioxide:
            raise ValueError("Total sulfur dioxide must be greater than or equal to free sulfur dioxide.")
        row = {
            "fixed acidity": self.fixed_acidity,
            "volatile acidity": self.volatile_acidity,
            "citric acid": self.citric_acid,
            "residual sugar": self.residual_sugar,
            "chlorides": self.chlorides,
            "free sulfur dioxide": self.free_sulfur_dioxide,
            "total sulfur dioxide": self.total_sulfur_dioxide,
            "density": self.density,
            "pH": self.pH,
            "sulphates": self.sulphates,
            "alcohol": self.alcohol,
            "is_white": int(self.wine_type == "white"),
        }
        frame = pd.DataFrame([row])
        frame["bound_sulfur_dioxide"] = np.maximum(
            frame["total sulfur dioxide"] - frame["free sulfur dioxide"], 0.0
        )
        frame["free_to_total_so2_ratio"] = frame["free sulfur dioxide"] / (
            frame["total sulfur dioxide"] + 1e-5
        )
        frame["volatile_to_fixed_acidity"] = frame["volatile acidity"] / (
            frame["fixed acidity"] + 1e-5
        )
        frame["sugar_to_alcohol_ratio"] = frame["residual sugar"] / (
            frame["alcohol"] + 1e-5
        )
        frame["sulphate_to_chloride_ratio"] = frame["sulphates"] / (
            frame["chlorides"] + 1e-5
        )
        return frame[bundle["feature_cols"]]

@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": True, "model": bundle.get("model_name", "Extra Trees")}

@app.get("/api/model-info")
def model_info():
    return {
        "model_name": bundle.get("model_name"),
        "algorithm": "ExtraTreesClassifier",
        "classes": [TIER_LABELS[int(c)] for c in model.classes_],
        "features": len(bundle["feature_cols"]),
        "training_rows": bundle.get("training_rows"),
        "metrics": {
            "test_accuracy": bundle.get("test_accuracy"),
            "test_macro_f1": bundle.get("test_macro_f1"),
            "test_macro_roc_auc": bundle.get("test_macro_roc_auc"),
            "cv_macro_f1": bundle.get("cv_macro_f1"),
        },
    }

@app.post("/api/predict")
def predict(payload: WineInput):
    try:
        features = payload.as_features()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    probabilities = model.predict_proba(features)[0]
    class_ids = [int(c) for c in model.classes_]
    ranked = sorted(zip(class_ids, probabilities), key=lambda pair: pair[1], reverse=True)
    predicted_id, top_probability = ranked[0]
    entropy = -sum(float(p) * math.log(max(float(p), 1e-12)) for p in probabilities)
    probability_map = {
        TIER_LABELS[class_id]: round(float(prob), 6)
        for class_id, prob in zip(class_ids, probabilities)
    }
    return {
        "predicted_class_id": predicted_id,
        "predicted_tier": TIER_LABELS[predicted_id],
        "confidence": round(float(top_probability), 6),
        "top_two_margin": round(float(ranked[0][1] - ranked[1][1]), 6),
        "entropy": round(float(entropy), 6),
        "probabilities": probability_map,
        "model_name": bundle.get("model_name"),
        "notice": "Probability is the model's estimated class probability, not a guarantee or a calibrated measure of certainty.",
    }
