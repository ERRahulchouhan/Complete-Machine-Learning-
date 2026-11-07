from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .data import TARGET, make_regression_frame

LOGGER = logging.getLogger("ml_course")

def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or not isinstance(config.get("target"), str):
        raise ValueError(f"Invalid training configuration: {config_path}")
    return config

def create_preprocessor(frame: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    numeric = frame.select_dtypes(include="number").columns.tolist()
    categorical = frame.select_dtypes(exclude="number").columns.tolist()
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        transformers.append(("numeric", Pipeline(numeric_steps), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))
    if not transformers:
        raise ValueError("Input data must contain at least one feature")
    return ColumnTransformer(transformers, remainder="drop")

def train(config_path: str | Path, estimator: Any, output_dir: str | Path | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    seed = int(config.get("random_state", 42))
    frame = make_regression_frame(int(config.get("n_samples", 600)), seed)
    target = config["target"]
    if target not in frame:
        raise ValueError(f"Target column {target!r} is missing")
    X, y = frame.drop(columns=target), frame[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(config.get("test_size", 0.2)), random_state=seed
    )
    scale_numeric = bool(config.get("scale_numeric", False))
    pipeline = Pipeline([("preprocessor", create_preprocessor(X_train, scale_numeric)), ("model", estimator)])
    pipeline.fit(X_train, y_train)
    train_predictions = pipeline.predict(X_train)
    test_predictions = pipeline.predict(X_test)
    results = {
        "algorithm": config.get("algorithm", "regression"),
        "random_state": seed,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "train": regression_metrics(y_train, train_predictions),
        "test": regression_metrics(y_test, test_predictions),
        "feature_names": X.columns.tolist(),
    }
    destination = Path(output_dir) if output_dir else Path(config_path).parent
    (destination / "models").mkdir(parents=True, exist_ok=True)
    (destination / "reports").mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "feature_names": X.columns.tolist(), "target": target}, destination / "models" / "model.joblib")
    (destination / "reports" / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    LOGGER.info("Training complete: %s", results["test"])
    return results

def regression_metrics(actual: Any, predicted: Any) -> dict[str, float]:
    mse = float(mean_squared_error(actual, predicted))
    return {"mae": float(mean_absolute_error(actual, predicted)), "mse": mse, "rmse": float(np.sqrt(mse)), "r2": float(r2_score(actual, predicted))}

def predict(model_path: str | Path, rows: list[dict[str, Any]]) -> list[float]:
    if not rows:
        raise ValueError("At least one prediction row is required")
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or "pipeline" not in bundle or "feature_names" not in bundle:
        raise ValueError("Model artifact has an invalid format")
    frame = pd.DataFrame(rows)
    expected = bundle["feature_names"]
    missing = sorted(set(expected) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(expected))
    if missing or extra:
        raise ValueError(f"Input schema mismatch; missing={missing}, extra={extra}")
    return [float(value) for value in bundle["pipeline"].predict(frame[expected])]

def evaluate(model_path: str | Path, n_samples: int = 600, random_state: int = 42) -> dict[str, float]:
    bundle = joblib.load(model_path)
    frame = make_regression_frame(n_samples, random_state)
    X, y = frame.drop(columns=bundle["target"]), frame[bundle["target"]]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    return regression_metrics(y_test, bundle["pipeline"].predict(X_test))
