from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import file_hash, load_training_data

NUMERIC_COLUMNS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]


def build_pipeline(features: pd.DataFrame) -> Pipeline:
    categorical = [column for column in features.columns if column not in NUMERIC_COLUMNS]
    numeric_transformer = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_transformer = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))])
    preprocess = ColumnTransformer([
        ("numeric", numeric_transformer, NUMERIC_COLUMNS),
        ("categorical", categorical_transformer, categorical),
    ])
    return Pipeline([("preprocess", preprocess), ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42, solver="liblinear"))])


def choose_threshold(y_true: pd.Series, probabilities: np.ndarray, false_negative_cost: int = 500, false_positive_cost: int = 100) -> tuple[float, float]:
    candidates = np.arange(0.05, 0.96, 0.01)
    costs = []
    for threshold in candidates:
        predictions = (probabilities >= threshold).astype(int)
        false_negatives = int(((y_true == 1) & (predictions == 0)).sum())
        false_positives = int(((y_true == 0) & (predictions == 1)).sum())
        costs.append(false_negatives * false_negative_cost + false_positives * false_positive_cost)
    index = int(np.argmin(costs))
    return float(candidates[index]), float(costs[index])


def reference_profile(frame: pd.DataFrame) -> dict:
    profile = {}
    for column in frame.columns:
        series = frame[column]
        if column in NUMERIC_COLUMNS:
            values = pd.to_numeric(series, errors="coerce").dropna()
            edges = np.unique(np.quantile(values, np.linspace(0, 1, 11))).tolist()
            if len(edges) < 2:
                edges = [float(values.min()), float(values.max()) + 1]
            distribution = pd.cut(values, bins=edges, include_lowest=True).value_counts(normalize=True, sort=False)
            profile[column] = {"kind": "numeric", "edges": edges, "distribution": distribution.tolist()}
        else:
            distribution = series.fillna("__missing__").astype(str).value_counts(normalize=True)
            profile[column] = {"kind": "categorical", "distribution": distribution.to_dict()}
    return profile


def train(data_path: str, artifact_path: str) -> dict:
    features, target = load_training_data(data_path)
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, stratify=target, random_state=42)
    search = GridSearchCV(build_pipeline(features), {"model__C": [0.05, 0.1, 0.5, 1.0, 2.0]}, cv=5, scoring="roc_auc", n_jobs=-1)
    search.fit(x_train, y_train)
    probabilities = search.best_estimator_.predict_proba(x_test)[:, 1]
    threshold, expected_cost = choose_threshold(y_test.reset_index(drop=True), probabilities)
    predictions = (probabilities >= threshold).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "threshold": threshold,
        "validation_expected_cost": expected_cost,
        "false_negative_cost": 500,
        "false_positive_cost": 100,
        "best_params": search.best_params_,
    }
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_data_hash": file_hash(data_path),
        "training_rows": int(len(features)),
        "feature_columns": list(features.columns),
        "metrics": metrics,
        "versions": {"python": platform.python_version(), "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "reference_profile": reference_profile(x_train),
    }
    artifact = Path(artifact_path)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    bundle = {"pipeline": search.best_estimator_, "threshold": threshold, "metadata": metadata}
    joblib.dump(bundle, artifact)
    artifact.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/telco_churn.csv")
    parser.add_argument("--artifact", default="artifacts/churn_model.joblib")
    args = parser.parse_args()
    print(json.dumps(train(args.data, args.artifact)["metrics"], indent=2))
