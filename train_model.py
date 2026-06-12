"""
Insurance Premium Prediction — Model Training Script
=====================================================
Replaces the Colab notebook (modelTrain.ipynb) with a reproducible local script.

Trains RandomForestClassifier and GradientBoostingClassifier, compares them,
runs 5-fold cross-validation, prints full metrics, and saves the best model.
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
CSV_PATH = BASE_DIR / "insurance_data_500.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "model.pkl"

# ── 1. Load Data ─────────────────────────────────────────────────────────────
print("=" * 60)
print("Loading data …")
df = pd.read_csv(CSV_PATH)
print(f"  Shape: {df.shape}")
print(f"  Target distribution:\n{df['insurance_premium_category'].value_counts().to_string()}")
print()

# ── 2. Feature Engineering ───────────────────────────────────────────────────
df_feat = df.copy()

# Feature 1: BMI
df_feat["bmi"] = round(df_feat["weight"] / (df_feat["height"] ** 2), 2)


# Feature 2: Age Group
def age_group(age: int) -> str:
    if age < 25:
        return "young"
    elif age < 45:
        return "adult"
    elif age < 60:
        return "middle_aged"
    return "senior"


df_feat["age_group"] = df_feat["age"].apply(age_group)


# Feature 3: Lifestyle Risk  (FIXED — uses AND, not OR, for medium)
def lifestyle_risk(row) -> str:
    if row["smoker"] and row["bmi"] > 30:
        return "high"
    elif row["smoker"] and row["bmi"] > 27:
        return "medium"
    else:
        return "low"


df_feat["lifestyle_risk"] = df_feat.apply(lifestyle_risk, axis=1)

# Feature 4: City Tier
tier_1_cities = [
    "Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune",
]
tier_2_cities = [
    "Jaipur", "Chandigarh", "Indore", "Lucknow", "Patna", "Ranchi",
    "Visakhapatnam", "Coimbatore", "Bhopal", "Nagpur", "Vadodara", "Surat",
    "Rajkot", "Jodhpur", "Raipur", "Amritsar", "Varanasi", "Agra", "Dehradun",
    "Mysore", "Jabalpur", "Guwahati", "Thiruvananthapuram", "Ludhiana",
    "Nashik", "Allahabad", "Udaipur", "Aurangabad", "Hubli", "Belgaum",
    "Salem", "Vijayawada", "Tiruchirappalli", "Bhavnagar", "Gwalior",
    "Dhanbad", "Bareilly", "Aligarh", "Gaya", "Kozhikode", "Warangal",
    "Kolhapur", "Bilaspur", "Jalandhar", "Noida", "Guntur", "Asansol",
    "Siliguri",
]


def city_tier(city: str) -> int:
    if city in tier_1_cities:
        return 1
    elif city in tier_2_cities:
        return 2
    return 3


df_feat["city_tier"] = df_feat["city"].apply(city_tier)

# ── 3. Prepare X / Y ────────────────────────────────────────────────────────
X = df_feat[["bmi", "age_group", "lifestyle_risk", "city_tier", "income_lpa", "occupation"]]
Y = df_feat["insurance_premium_category"]

categorical_features = ["age_group", "lifestyle_risk", "occupation", "city_tier"]
numeric_features = ["bmi", "income_lpa"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features),
    ]
)

# ── 4. Train / Test Split (80/20) ───────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y,
)
print(f"Train size: {len(X_train)}  |  Test size: {len(X_test)}")
print()

# ── 5. Define Candidate Models ──────────────────────────────────────────────
candidates = {
    "RandomForest": RandomForestClassifier(
        n_estimators=200, random_state=42,
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42,
    ),
}

results = {}

for name, clf in candidates.items():
    print("=" * 60)
    print(f"  Model: {name}")
    print("=" * 60)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", clf),
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Predict on test set
    y_pred = pipeline.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1_w = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)

    # 5-fold Cross Validation
    cv_scores = cross_val_score(pipeline, X, Y, cv=5, scoring="accuracy")
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    results[name] = {
        "pipeline": pipeline,
        "accuracy": acc,
        "f1_weighted": f1_w,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
    }

    print(f"\n  Accuracy:       {acc:.4f}")
    print(f"  F1 (weighted):  {f1_w:.4f}")
    print(f"  CV Accuracy:    {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"\n  Classification Report:\n{report}")
    print(f"  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"  Labels: {pipeline.classes_.tolist()}")
    print(f"{cm}")
    print()

# ── 6. Select Best Model ────────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["accuracy"])
best = results[best_name]

print("=" * 60)
print("  BEST MODEL SELECTION")
print("=" * 60)
print(f"\n  Winner:          {best_name}")
print(f"  Test Accuracy:   {best['accuracy']:.4f}")
print(f"  F1 (weighted):   {best['f1_weighted']:.4f}")
print(f"  CV Mean Acc:     {best['cv_mean']:.4f} ± {best['cv_std']:.4f}")
print()

# ── 7. Save Best Model ──────────────────────────────────────────────────────
MODEL_DIR.mkdir(parents=True, exist_ok=True)
with open(MODEL_PATH, "wb") as f:
    pickle.dump(best["pipeline"], f)

print(f"  [OK] Model saved to {MODEL_PATH}")
print("=" * 60)
