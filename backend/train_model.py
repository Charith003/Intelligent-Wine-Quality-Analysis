"""Train the WineSense AI model using the exact pipeline defined in the notebook.
Run from backend/: python train_model.py
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

red = pd.read_csv(DATA / "winequality-red.csv", sep=";")
white = pd.read_csv(DATA / "winequality-white.csv", sep=";")
red["wine_type"] = "red"
white["wine_type"] = "white"
df = pd.concat([red, white], ignore_index=True).drop_duplicates().reset_index(drop=True)

def to_tier_id(q: int) -> int:
    if q <= 4:
        return 0
    if q <= 6:
        return 1
    return 2

def add_derived_features(data: pd.DataFrame) -> pd.DataFrame:
    d = data.copy()
    if "is_white" not in d.columns and "wine_type" in d.columns:
        d["is_white"] = (d["wine_type"].astype(str).str.lower() == "white").astype(int)
    d["bound_sulfur_dioxide"] = np.maximum(d["total sulfur dioxide"] - d["free sulfur dioxide"], 0.0)
    d["free_to_total_so2_ratio"] = d["free sulfur dioxide"] / (d["total sulfur dioxide"] + 1e-5)
    d["volatile_to_fixed_acidity"] = d["volatile acidity"] / (d["fixed acidity"] + 1e-5)
    d["sugar_to_alcohol_ratio"] = d["residual sugar"] / (d["alcohol"] + 1e-5)
    d["sulphate_to_chloride_ratio"] = d["sulphates"] / (d["chlorides"] + 1e-5)
    return d

base_feature_cols = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol", "is_white",
]
feature_cols = base_feature_cols + [
    "bound_sulfur_dioxide", "free_to_total_so2_ratio",
    "volatile_to_fixed_acidity", "sugar_to_alcohol_ratio",
    "sulphate_to_chloride_ratio",
]
tier_names = ["Low (3-4)", "Average (5-6)", "High (7-9)"]
df["is_white"] = (df["wine_type"] == "white").astype(int)
df["tier_id"] = df["quality"].apply(to_tier_id)
df_fe = add_derived_features(df)
strat_key = df["wine_type"] + "_" + df["tier_id"].astype(str)
X_train, X_test, y_train, y_test = train_test_split(
    df_fe[feature_cols], df["tier_id"], test_size=0.20,
    random_state=42, stratify=strat_key,
)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = ExtraTreesClassifier(
    n_estimators=250, max_depth=16, min_samples_leaf=3,
    class_weight="balanced", random_state=42, n_jobs=-1,
)
cv_f1 = float(cross_val_score(model, X_train, y_train, cv=skf, scoring="f1_macro").mean())
model.fit(X_train, y_train)
pred = model.predict(X_test)
probs = model.predict_proba(X_test)
metrics = {
    "cv_macro_f1": round(cv_f1, 4),
    "test_accuracy": round(float(accuracy_score(y_test, pred)), 4),
    "test_macro_f1": round(float(f1_score(y_test, pred, average="macro")), 4),
    "test_macro_roc_auc": round(float(roc_auc_score(y_test, probs, multi_class="ovr", average="macro")), 4),
}
bundle = {
    "model": model,
    "feature_cols": feature_cols,
    "base_feature_cols": base_feature_cols,
    "tier_names": tier_names,
    **metrics,
    "training_rows": int(len(df)),
    "red_rows_raw": int(len(red)),
    "white_rows_raw": int(len(white)),
    "duplicates_removed": int(len(red) + len(white) - len(df)),
    "model_name": "Extra Trees (Balanced + Engineered Features)",
}
out = MODEL_DIR / "wine_quality_et_pipeline.joblib"
joblib.dump(bundle, out)
print(f"Saved model: {out}")
print({k: v for k, v in bundle.items() if k not in {"model", "feature_cols", "base_feature_cols", "tier_names"}})
