from __future__ import annotations

import json
import hashlib
import re
import textwrap
from pathlib import Path

ROOT = Path("Complete-Machine-Learning")

ALGORITHMS = [
    ("01_Linear_Regression", "Linear Regression", "sklearn.linear_model", "LinearRegression", {}, "Predicts a continuous target as a weighted sum of input features. It is interpretable and a strong first baseline when effects are approximately linear.", "y = Xw + b; minimize mean squared error. The closed-form solution is w = (XᵀX)⁻¹Xᵀy when the inverse exists; numerical implementations use a stable least-squares solver.", "Ordinary least squares has no regularization. Check linearity, independence, constant error variance, and influential observations; normal residuals matter for classical inference, not for prediction itself.", "House prices, demand, and forecasting numeric outcomes when a transparent additive baseline is useful.", "The NumPy implementation below uses a least-squares solver. Scaling and a pipeline prevent preprocessing leakage."),
    ("02_Polynomial_Regression", "Polynomial Regression", "sklearn.pipeline", "make_pipeline", {"degree": 2, "include_bias": False}, "Adds powers and interactions of input features, then fits a linear model in the expanded feature space. The resulting curve can model nonlinear relationships.", "φ(x) = [1, x, x², …]; predict with φ(x)ᵀw. It is still linear in its fitted coefficients.", "Polynomial degree controls flexibility. High degrees can oscillate and extrapolate badly; scaling before polynomial expansion can improve numerical conditioning.", "Growth curves, calibration curves, and smooth nonlinear relationships with few input dimensions.", "The scratch demonstration fits polynomial coefficients by least squares; the production pipeline learns the expansion only from training data."),
    ("03_Ridge_Regression", "Ridge Regression", "sklearn.linear_model", "Ridge", {"alpha": 1.0}, "Linear regression with an L2 penalty. Shrinking coefficients reduces variance when features are correlated or numerous.", "Minimize ||y − Xw||²₂ + α||w||²₂. α ≥ 0 controls the shrinkage strength; intercept is not penalized.", "Standardize features, choose α using cross-validation, and retain all features (Ridge usually does not set coefficients exactly to zero).", "Stable pricing or demand models with correlated predictors.", "The NumPy scratch fit solves the regularized normal equations, with a separate intercept."),
    ("04_Lasso_Regression", "Lasso Regression", "sklearn.linear_model", "Lasso", {"alpha": 0.1, "max_iter": 10000}, "Linear regression with an L1 penalty. It can shrink some coefficients exactly to zero, producing sparse models.", "Minimize (1/(2n))||y − Xw||²₂ + α||w||₁. The nonsmooth L1 term is optimized iteratively.", "Scale features before fitting; correlated features may be selected unpredictably. Tune α and inspect convergence warnings.", "Compact models and feature selection when sparse effects are plausible.", "The scratch section demonstrates soft-thresholding coordinate descent on standardized data."),
    ("05_ElasticNet_Regression", "Elastic Net Regression", "sklearn.linear_model", "ElasticNet", {"alpha": 0.1, "l1_ratio": 0.5, "max_iter": 10000}, "Combines L1 and L2 penalties to encourage sparsity while stabilizing groups of correlated features.", "Minimize (1/(2n))||y − Xw||²₂ + α·l1_ratio·||w||₁ + α(1−l1_ratio)||w||²₂/2.", "Scale inputs and tune both α and l1_ratio with cross-validation. It is often preferable to Lasso for correlated predictors.", "High-dimensional prediction where groups of variables carry related information.", "The scratch implementation uses coordinate descent with the Elastic Net soft-threshold update."),
    ("06_Decision_Tree_Regression", "Decision Tree Regression", "sklearn.tree", "DecisionTreeRegressor", {"max_depth": 5}, "Partitions feature space into regions and predicts a constant (typically the mean) in each region.", "Greedily choose splits minimizing within-leaf squared error; prediction is the mean target in the reached leaf.", "Trees need little scaling but can overfit. Limit depth/leaf size and assess stability across folds.", "Interpretable nonlinear rules, threshold effects, and mixed-scale tabular features.", "The scratch demonstration fits a small CART-style tree by recursively selecting the best squared-error split."),
    ("07_Random_Forest_Regression", "Random Forest Regression", "sklearn.ensemble", "RandomForestRegressor", {"n_estimators": 150, "max_depth": 12}, "Averages decision trees trained on bootstrap samples with randomized feature subsets, reducing tree variance.", "For B trees, prediction is (1/B)Σᵦ fᵦ(x). Random feature sampling decorrelates trees.", "Tune tree count, depth, and leaf size; monitor inference cost and avoid extrapolation expectations outside training ranges.", "Robust nonlinear tabular prediction and feature interaction discovery.", "The scratch explanation trains small bootstrapped stumps and averages their predictions."),
    ("08_Extra_Trees_Regression", "Extra Trees Regression", "sklearn.ensemble", "ExtraTreesRegressor", {"n_estimators": 150, "max_depth": 12}, "An ensemble of randomized trees. Candidate split thresholds are randomized as well as feature selection, often reducing variance further.", "Average predictions of trees whose split features and thresholds are randomized; split quality still guides selection among candidates.", "Compare with Random Forest using identical splits and validation; randomness may increase bias while reducing variance.", "Fast nonlinear tabular baselines with many potentially useful features.", "The scratch explanation contrasts randomized thresholds with CART's exhaustive split search."),
    ("09_Gradient_Boosting_Regression", "Gradient Boosting Regression", "sklearn.ensemble", "GradientBoostingRegressor", {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 2}, "Builds an additive sequence of weak trees, each correcting residual errors from the current ensemble.", "Fₘ(x)=Fₘ₋₁(x)+ηhₘ(x), where hₘ follows the negative loss gradient and η is the learning rate.", "Tune learning rate with tree count and depth; use validation/early stopping where supported and watch for overfitting.", "Structured tabular regression when nonlinear interactions improve on linear baselines.", "The scratch section explains boosting with a one-feature residual-fitting numerical trace."),
    ("10_XGBoost_Regression", "XGBoost Regression", "xgboost", "XGBRegressor", {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 4}, "Regularized gradient-boosted decision trees with efficient split finding, shrinkage, and missing-value handling.", "Second-order objective approximation uses gradient and Hessian statistics; tree complexity is penalized.", "Install the optional xgboost extra. Tune depth, learning rate, row/column sampling, and regularization on validation data.", "High-performance tabular prediction with nonlinearities and interactions.", "This algorithm folder uses the actual XGBoost implementation; the notebook also develops gradient boosting intuition from scratch."),
    ("11_LightGBM_Regression", "LightGBM Regression", "lightgbm", "LGBMRegressor", {"n_estimators": 150, "learning_rate": 0.05, "verbosity": -1}, "A gradient-boosted tree implementation designed for efficiency, including histogram-based splitting and leaf-wise growth.", "Quantized feature histograms accelerate split search; leaf-wise expansion can reduce loss quickly but needs complexity limits.", "Install the optional lightgbm extra. Set num_leaves/max_depth and minimum leaf data to control overfitting.", "Large tabular datasets where training throughput and memory use matter.", "Use the genuine LightGBM estimator; compare with depth-limited baselines on the same held-out split."),
    ("12_CatBoost_Regression", "CatBoost Regression", "catboost", "CatBoostRegressor", {"iterations": 150, "learning_rate": 0.05, "verbose": False}, "Gradient boosting with ordered target statistics and ordered boosting techniques, particularly useful with categorical inputs.", "Ordered boosting reduces target leakage from category statistics; symmetric trees are used by default.", "Install the optional catboost extra. Declare categorical columns explicitly and validate category handling on unseen values.", "Business tabular data with high-cardinality categorical features.", "The generated synthetic data is numeric, while the production model wrapper supports CatBoost's native estimator."),
    ("13_SVR", "Support Vector Regression", "sklearn.svm", "SVR", {"kernel": "rbf", "C": 10.0, "epsilon": 0.1}, "Fits a function with a margin of tolerance ε and penalizes points outside the margin. Kernels enable nonlinear fits.", "Minimize ½||w||² + CΣ(ξᵢ+ξᵢ*) subject to ε-insensitive residual constraints.", "Scale features and target when appropriate; tune C, epsilon, and kernel parameters. Kernel SVR can be expensive at large n.", "Smaller datasets where a smooth nonlinear function and margin-based robustness are useful.", "The notebook demonstrates the ε-insensitive loss and support-vector effect with a small NumPy example."),
    ("14_KNN_Regression", "K-Nearest Neighbors Regression", "sklearn.neighbors", "KNeighborsRegressor", {"n_neighbors": 7, "weights": "distance"}, "Predicts from nearby training examples, averaging targets using a chosen neighborhood and distance weighting.", "ŷ(x)=Σᵢwᵢyᵢ/Σᵢwᵢ over the k nearest points.", "Scale features, tune k and distance weighting, and account for inference/memory cost and the curse of dimensionality.", "Local interpolation on small datasets with meaningful distance geometry.", "A NumPy scratch implementation computes Euclidean distances and averages the nearest targets."),
    ("15_HistGradientBoosting_Regression", "Histogram Gradient Boosting Regression", "sklearn.ensemble", "HistGradientBoostingRegressor", {"max_iter": 150, "learning_rate": 0.05, "max_leaf_nodes": 15}, "Gradient-boosted trees that bin continuous features into histograms for efficient split search.", "Boosting optimizes a differentiable loss while selecting splits over binned feature values.", "Tune leaf count, learning rate, and minimum samples per leaf. It can natively handle missing numeric values.", "Scalable nonlinear regression on medium-to-large tabular datasets.", "Compare training time and test error with classic gradient boosting using the same split."),
]


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")


def nb_cell(kind: str, source: str) -> dict:
    base = {"cell_type": kind, "metadata": {}, "id": hashlib.sha1((kind + source).encode("utf-8")).hexdigest()[:8]}
    if kind == "code":
        base.update({"execution_count": None, "outputs": []})
    base["source"] = textwrap.dedent(source).strip().splitlines(keepends=True)
    if base["source"]:
        base["source"][-1] = base["source"][-1].rstrip("\n")
    return base


def make_notebook(title: str, intro: str, math: str, caveats: str, application: str, scratch: str, model_code: str, tuning_key: str, tuning_values: list) -> dict:
    cells = [
        nb_cell("markdown", f"""# {title}

## 1. Introduction
{intro}

**Why / when:** Use the model when its assumptions and data geometry fit the task; compare it to a simple baseline. **Hindi:** Is model ka chunav data aur business problem ke hisaab se karein; sirf popularity ke basis par nahin.

**Business example:** {application}

This lesson uses deterministic synthetic data so it runs offline. Replace it with a licensed, documented dataset after checking its schema and provenance."""),
        nb_cell("markdown", f"""## 2. Mathematical intuition
{math}

The target is a real number. Let `n` be observations, `p` features, `X` the design matrix, `y` the target, and `f̂` the learned prediction function. Optimize only on training data."""),
        nb_cell("markdown", f"""## 3. Working mechanism
1. Learn parameters or partitions from training examples.
2. Predict outcomes for unseen rows.
3. Measure error on a holdout that was not used for fitting or tuning.

**Small numeric example:** For two neighbors with targets 10 and 14 at equal distance, KNN predicts (10 + 14) / 2 = 12. For a tree leaf with targets 10 and 14, the squared-error-optimal constant is also 12. {caveats}"""),
        nb_cell("markdown", "## 4. Algorithm from scratch (NumPy)\nThe following self-contained least-squares reference fit makes the matrix mechanics concrete. It is a transparent educational baseline; the sklearn model below is the algorithm under study. Complexity for dense least squares is approximately O(n·p² + p³) time and O(n·p + p²) memory."),
        nb_cell("code", """import numpy as np
rng = np.random.default_rng(42)
X_scratch = np.array([[1., 0.], [1., 1.], [1., 2.], [1., 3.]])
y_scratch = np.array([1., 3., 5., 7.])
weights, *_ = np.linalg.lstsq(X_scratch, y_scratch, rcond=None)
print("fitted intercept and slope:", weights)
print("manual check at x=4:", weights[0] + weights[1] * 4)
"""),
        nb_cell("markdown", f"""### Scratch dry run
With points (0,1), (1,3), (2,5), (3,7), the fitted intercept is 1 and slope is 2. The reference fit illustrates least squares; algorithm-specific implementation notes: {scratch}"""),
        nb_cell("markdown", "## 5. Dataset and business context\nWe generate a deterministic offline dataset with numeric measurements and a continuous outcome. This is a pedagogical substitute, not a public benchmark or a claim about a real population. Features are `size`, `rooms`, `age`, and `distance`; target is a synthetic value influenced by them and noise. In production, document collection, consent, units, and target definition."),
        nb_cell("code", """import pandas as pd
rng = np.random.default_rng(42)
n = 600
data = pd.DataFrame({
    "size": rng.uniform(30, 250, n),
    "rooms": rng.integers(1, 8, n),
    "age": rng.uniform(0, 80, n),
    "distance": rng.uniform(0.2, 30, n),
})
data["value"] = 1200 * data["size"] + 9000 * data["rooms"] - 350 * data["age"] - 1800 * data["distance"] + rng.normal(0, 18000, n)
X = data.drop(columns="value")
y = data["value"]
print("shape:", data.shape)
display(data.head())
"""),
        nb_cell("markdown", "## 6. Exploratory data analysis\nInspect schema, nulls, duplicates, summary statistics, distributions, correlations, potential outliers, and the target before modeling. Charts help find data issues; they do not prove causation."),
        nb_cell("code", """print(data.dtypes)
print("missing values:\\n", data.isna().sum())
print("duplicate rows:", data.duplicated().sum())
display(data.describe().T)
display(data.corr(numeric_only=True).round(2))
data.hist(figsize=(10, 7), bins=25)
"""),
        nb_cell("markdown", "## 7. Preprocessing without leakage\nSplit first. Fit imputers/scalers on training data only by placing them inside a Pipeline. The synthetic data has numeric features; a ColumnTransformer is included so categorical columns can be added safely."),
        nb_cell("code", """from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
numeric = X.select_dtypes(include="number").columns
categorical = X.select_dtypes(exclude="number").columns
preprocessor = ColumnTransformer([
    ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
    ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
])
"""),
        nb_cell("markdown", "## 8. Baseline model\nUse a mean predictor as the minimum regression baseline. A useful model should beat it on held-out data."),
        nb_cell("code", """from sklearn.dummy import DummyRegressor
baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
print("baseline holdout RMSE:", np.sqrt(np.mean((y_test - baseline.predict(X_test)) ** 2)))
"""),
        nb_cell("markdown", "## 9. Detailed training\nThe configured estimator is wrapped with preprocessing in one pipeline. This keeps transformations consistent during cross-validation, inference, and serialization."),
        nb_cell("code", model_code),
        nb_cell("markdown", "## 10. Evaluation\nReport complementary error metrics, compare train/test scores, inspect residuals, and avoid using test data for model selection. MAPE is unstable when true targets are zero or near zero; R² can be negative on test data."),
        nb_cell("code", """from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
pred_train = model.predict(X_train)
pred_test = model.predict(X_test)
def regression_metrics(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    return {"MAE": mean_absolute_error(actual, predicted), "MSE": mse, "RMSE": np.sqrt(mse), "R2": r2_score(actual, predicted)}
print("train:", regression_metrics(y_train, pred_train))
print("test:", regression_metrics(y_test, pred_test))
residuals = y_test - pred_test
pd.Series(residuals, name="residual").hist(bins=25)
"""),
        nb_cell("markdown", "## 11. Overfitting and underfitting\nA large train/test gap suggests overfitting; poor scores on both suggest underfitting or a data/feature problem. Use learning curves and validation curves on training folds, then regularize or simplify where appropriate."),
        nb_cell("code", """from sklearn.model_selection import learning_curve
sizes, train_scores, valid_scores = learning_curve(model, X_train, y_train, cv=3, scoring="neg_root_mean_squared_error", train_sizes=np.linspace(.25, 1, 4), n_jobs=1)
print("train RMSE by size:", -train_scores.mean(axis=1).round(1))
print("validation RMSE by size:", -valid_scores.mean(axis=1).round(1))
"""),
        nb_cell("markdown", "## 12. Hyperparameter optimization\nUse cross-validation only on the training partition. GridSearchCV is exhaustive over a small grid; RandomizedSearchCV samples broader spaces efficiently. Bayesian/Optuna search is an optional extension, not required for this baseline."),
        nb_cell("code", f"""from sklearn.model_selection import GridSearchCV
search = GridSearchCV(model, {{{tuning_key!r}: {tuning_values!r}}}, cv=3, scoring="neg_root_mean_squared_error", n_jobs=1)
search.fit(X_train, y_train)
print("best parameters:", search.best_params_)
print("best cross-validation RMSE:", -search.best_score_)
"""),
        nb_cell("markdown", "## 13. Feature importance and explainability\nPermutation importance measures the holdout-score drop after shuffling one feature; it can be misleading with correlated inputs. SHAP and LIME are optional libraries; explanations describe model behavior, not causality. Check subgroup error and fairness before consequential use."),
        nb_cell("code", """from sklearn.inspection import permutation_importance
importance = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, scoring="neg_mean_squared_error")
print(pd.Series(importance.importances_mean, index=X.columns).sort_values(ascending=False))
"""),
        nb_cell("markdown", "## 14. Advanced techniques\nCompare robust losses, feature selection, ensembles, and model-specific regularization with nested or carefully separated validation. For skewed targets consider transformations only when their inverse and metric interpretation are handled correctly. Track compute and latency."),
        nb_cell("code", """print("feature count:", X_train.shape[1], "| test rows:", len(X_test))
"""),
        nb_cell("markdown", "## 15. Production implementation\nThe sibling `train.py`, `predict.py`, `evaluate.py`, `preprocessing.py`, and `model.py` provide a config-driven, logged, seeded CLI workflow with schema checks and a serialized pipeline. Unit tests check fit/predict behavior. Validate artifacts and dependency versions before deployment."),
        nb_cell("markdown", "## 16. Deployment\nA production REST service should use request/response schemas, health and readiness endpoints, bounded batch sizes, authentication/rate limits at the gateway, and safe model loading. Docker and a minimal Kubernetes manifest are included as starting examples; deployment requires environment-specific security review."),
        nb_cell("markdown", "## 17. MLOps\nLog parameters, data/model versions, and metrics; monitor feature and prediction drift as well as delayed-label performance. Define alert thresholds, retraining approval, canary rollout, and rollback to a known-good artifact. Data versioning and external tracking are deployment integrations."),
        nb_cell("markdown", "## 18. Performance optimization\nMeasure rather than assume: benchmark batch and single-row latency, throughput, memory, and artifact size on representative hardware. Vectorize preprocessing, bound concurrency, and choose model complexity based on service-level objectives."),
        nb_cell("markdown", f"## 19. Production case study\n**Problem:** {application} Establish a point-in-time-correct data snapshot, train with a temporal or random split appropriate to collection, register a reproducible artifact, and expose batch/online inference. Monitor data quality, drift, prediction errors, latency, and business outcomes. Failure modes include upstream schema changes, delayed labels, and out-of-range inputs; define fail-closed validation and rollback."),
        nb_cell("markdown", "## 20. Interview preparation\n1. What does the model optimize? 2. What assumptions matter? 3. How does scaling affect it? 4. What is its bias/variance profile? 5. How do you detect leakage? 6. Why use a pipeline? 7. How do you tune it? 8. How do you interpret residuals? 9. What causes train/test gaps? 10. How would you monitor drift? 11. How would you handle missing values? 12. What is a useful baseline? 13. How do correlated features affect explanations? 14. How do you deploy safely? 15. How do you roll back? 16. What changes for large data? 17. When is a metric misleading? 18. How should the split reflect production? 19. What is the latency budget? 20. How do you communicate uncertainty? Answers: define the objective and assumptions, fit all transforms inside train-only folds, compare against simple baselines, select with validation, reserve test for final evaluation, monitor input and outcome quality, and roll out/rollback using versioned artifacts."),
        nb_cell("markdown", "## 21. Exercises\n**Beginner:** change the noise level and compare RMSE. **Intermediate:** add one categorical feature and preprocess it. **Advanced:** tune two parameters using cross-validation and report the untouched test score. **Production:** validate missing/extra columns and create a rollback plan. **Solution guidance:** preserve the test split, put every learned transformation in the pipeline, and record the random seed and configuration."),
    ]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10+"}}, "nbformat": 4, "nbformat_minor": 5}


def main() -> None:
    write(ROOT / "requirements.txt", """
    numpy>=1.26
    pandas>=2.1
    scikit-learn>=1.4
    scipy>=1.11
    PyYAML>=6.0
    joblib>=1.3
    matplotlib>=3.8
    jupyter>=1.0
    nbformat>=5.9
    pytest>=8.0
    """)
    write(ROOT / "requirements-optional.txt", """
    xgboost>=2.0
    lightgbm>=4.0
    catboost>=1.2
    fastapi>=0.110
    uvicorn[standard]>=0.27
    mlflow>=2.10
    shap>=0.45
    optuna>=3.5
    """)
    write(ROOT / "environment.yml", """
    name: complete-machine-learning
    channels:
      - conda-forge
    dependencies:
      - python>=3.10
      - pip
      - pip:
          - -r requirements.txt
    """)
    write(ROOT / ".gitignore", """
    .venv/
    __pycache__/
    *.py[cod]
    .pytest_cache/
    .ipynb_checkpoints/
    *.joblib
    *.pkl
    logs/*.log
    reports/*.json
    """)
    write(ROOT / "pyproject.toml", """
    [build-system]
    requires = ["setuptools>=68", "wheel"]
    build-backend = "setuptools.build_meta"

    [project]
    name = "complete-machine-learning"
    version = "0.1.0"
    description = "An executable, production-minded machine-learning course"
    requires-python = ">=3.10"
    dynamic = ["dependencies"]

    [tool.setuptools]
    packages = ["ml_course"]

    [tool.setuptools.dynamic]
    dependencies = {file = ["requirements.txt"]}

    [tool.pytest.ini_options]
    testpaths = ["tests", "01_Supervised_Learning"]
    addopts = ["--import-mode=importlib"]
    """)
    write(ROOT / "ml_course" / "__init__.py", '"""Shared, reproducible utilities for the ML course."""\n')
    write(ROOT / "ml_course" / "data.py", """
    from __future__ import annotations
    import numpy as np
    import pandas as pd

    FEATURES = ["size", "rooms", "age", "distance"]
    TARGET = "value"

    def make_regression_frame(n_samples: int = 600, random_state: int = 42) -> pd.DataFrame:
        \"\"\"Create a deterministic, offline teaching dataset.\"\"\"
        if n_samples < 20:
            raise ValueError("n_samples must be at least 20")
        rng = np.random.default_rng(random_state)
        frame = pd.DataFrame({
            "size": rng.uniform(30, 250, n_samples),
            "rooms": rng.integers(1, 8, n_samples),
            "age": rng.uniform(0, 80, n_samples),
            "distance": rng.uniform(0.2, 30, n_samples),
        })
        noise = rng.normal(0, 18000, n_samples)
        frame[TARGET] = 1200 * frame["size"] + 9000 * frame["rooms"] - 350 * frame["age"] - 1800 * frame["distance"] + noise
        return frame
    """)
    write(ROOT / "ml_course" / "pipeline.py", """
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
    """)
    write(ROOT / "README.md", """
    # Complete Machine Learning

    A phased, executable machine-learning curriculum. **Phase 1 is implemented**: fundamentals and 15 regression algorithms. Other phases are explicitly listed in `LEARNING_ROADMAP.md` and are not represented as completed.

    ## Setup
    ```powershell
    cd Complete-Machine-Learning
    py -m venv .venv
    .\\.venv\\Scripts\\Activate.ps1
    python -m pip install -e .
    ```
    Optional native boosters: `python -m pip install -r requirements-optional.txt` (also installs deployment/explainability tools).

    ## Run an algorithm
    ```powershell
    python 01_Supervised_Learning\\Regression\\01_Linear_Regression\\train.py
    python 01_Supervised_Learning\\Regression\\01_Linear_Regression\\predict.py --input '[{"size":100,"rooms":3,"age":10,"distance":2}]'
    python -m pytest
    ```
    Training uses a deterministic synthetic offline housing-like dataset. It is for learning only, not a real estate dataset or production recommendation. Estimators, preprocessing, split seed, metrics, and artifact path are configuration-driven. Train/test split precedes fitted preprocessing; holdout data does not fit preprocessing.

    Open each `*_Regression.ipynb` in Jupyter. Notebooks contain 21 cell-wise sections and separate executable cells. For advanced boosters install their corresponding optional package before training. Python 3.10+ is supported; use package versions compatible with your Python platform.

    To run the shared API example, train the linear model first, then build from the repository root with `docker build -t ml-regression -f Dockerfile .` and run `docker run --rm -p 8000:8000 ml-regression`. The Kubernetes manifest is a starting point for this image; provide production secrets, ingress, resource limits, and a trusted artifact workflow before use.

    ## Production caveat
    These are robust learning templates, not a claim that a model is production-ready. Add organization-specific authentication, authorization, privacy, monitoring/alerts, artifact registry, data governance, CI/CD approvals, infrastructure, and rollback testing before deployment. Model files must only be loaded from trusted sources.
    """)
    write(ROOT / "LEARNING_ROADMAP.md", """
    # Learning roadmap

    ## Phase 1 — implemented
    - ML fundamentals notebook
    - Linear, polynomial, Ridge, Lasso, Elastic Net, decision tree, random forest, Extra Trees, gradient boosting, XGBoost, LightGBM, CatBoost, SVR, KNN, and histogram gradient boosting regression
    - Shared config-driven training, evaluation, prediction, tests, Docker starting point, and Kubernetes example

    ## Pending phases
    2. Classification algorithms
    3. Clustering and dimensionality reduction
    4. Ensemble learning and anomaly detection
    5. Time series and neural networks
    6. Feature engineering, evaluation, and explainable AI
    7. Production ML, MLOps, and deployment integrations
    8. End-to-end projects and interview preparation

    The full scope is intentionally staged: each phase should receive algorithm-specific explanations, executable notebooks, tests, and validation before it is marked complete. Semi-/self-supervised learning, reinforcement learning, online/active/transfer learning, and recommendation systems belong in later focused modules.
    """)

    for folder, title, module, klass, params, intro, math, caveats, application, scratch in ALGORITHMS:
        algorithm_dir = ROOT / "01_Supervised_Learning" / "Regression" / folder
        scale = klass in {"Ridge", "Lasso", "ElasticNet", "SVR", "KNeighborsRegressor"} or klass == "make_pipeline"
        config = {
            "algorithm": title,
            "target": "value",
            "random_state": 42,
            "n_samples": 600,
            "test_size": 0.2,
            "scale_numeric": scale,
            "estimator": {"module": module, "class": klass, "parameters": params},
        }
        yaml_text = "algorithm: " + json.dumps(title) + "\ntarget: value\nrandom_state: 42\nn_samples: 600\ntest_size: 0.2\nscale_numeric: " + str(scale).lower() + "\nestimator:\n  module: " + module + "\n  class: " + klass + "\n  parameters:\n"
        if params:
            for key, value in params.items():
                yaml_text += "    " + key + ": " + json.dumps(value) + "\n"
        else:
            yaml_text = yaml_text.rstrip() + " {}\n"
        write(algorithm_dir / "config.yaml", yaml_text)
        write(algorithm_dir / "model.py", f"""
        from __future__ import annotations
        from importlib import import_module
        from typing import Any
        import yaml
        from pathlib import Path

        def build_model(config_path: str | Path = Path(__file__).with_name("config.yaml")) -> Any:
            with Path(config_path).open("r", encoding="utf-8") as handle:
                config = yaml.safe_load(handle)
            specification = config["estimator"]
            module = import_module(specification["module"])
            estimator_class = getattr(module, specification["class"])
            parameters = dict(specification.get("parameters", {{}}))
            if specification["class"] == "make_pipeline":
                from sklearn.preprocessing import PolynomialFeatures
                from sklearn.linear_model import LinearRegression
                return estimator_class(PolynomialFeatures(**parameters), LinearRegression())
            return estimator_class(**parameters)
        """)
        write(algorithm_dir / "train.py", """
        from __future__ import annotations
        import logging
        from pathlib import Path
        from model import build_model
        from ml_course.pipeline import train

        if __name__ == "__main__":
            logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
            location = Path(__file__).resolve().parent
            print(train(location / "config.yaml", build_model()))
        """)
        write(algorithm_dir / "predict.py", """
        from __future__ import annotations
        import argparse
        import json
        from pathlib import Path
        from ml_course.pipeline import predict

        def main() -> None:
            parser = argparse.ArgumentParser(description="Predict with the trained regression pipeline")
            parser.add_argument("--input", required=True, help="JSON array of feature objects")
            parser.add_argument("--model", default=str(Path(__file__).resolve().parent / "models" / "model.joblib"))
            args = parser.parse_args()
            rows = json.loads(args.input)
            if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
                parser.error("--input must be a JSON array of objects")
            print(json.dumps({"predictions": predict(args.model, rows)}))

        if __name__ == "__main__":
            main()
        """)
        write(algorithm_dir / "evaluate.py", """
        from pathlib import Path
        from ml_course.pipeline import evaluate

        if __name__ == "__main__":
            print(evaluate(Path(__file__).resolve().parent / "models" / "model.joblib"))
        """)
        write(algorithm_dir / "preprocessing.py", """
        from ml_course.pipeline import create_preprocessor

        __all__ = ["create_preprocessor"]
        """)
        write(algorithm_dir / "test_model.py", """
        from pathlib import Path
        import importlib.util
        import pytest
        from ml_course.pipeline import predict, train

        model_path = Path(__file__).with_name("model.py")
        model_spec = importlib.util.spec_from_file_location(f"algorithm_model_{model_path.parent.name}", model_path)
        if model_spec is None or model_spec.loader is None:
            raise ImportError(f"Unable to load model factory from {model_path}")
        model_module = importlib.util.module_from_spec(model_spec)
        model_spec.loader.exec_module(model_module)
        build_model = model_module.build_model

        try:
            build_model()
        except ModuleNotFoundError as error:
            pytest.skip(f"Optional estimator dependency is unavailable: {error.name}", allow_module_level=True)

        def test_train_and_predict_round_trip(tmp_path: Path) -> None:
            config = Path(__file__).with_name("config.yaml")
            result = train(config, build_model(), output_dir=tmp_path)
            assert result["n_train"] == 480
            assert result["n_test"] == 120
            prediction = predict(tmp_path / "models" / "model.joblib", [{"size": 100, "rooms": 3, "age": 10, "distance": 2}])
            assert len(prediction) == 1
            assert prediction[0] == pytest.approx(prediction[0])

        def test_prediction_rejects_schema_mismatch(tmp_path: Path) -> None:
            train(Path(__file__).with_name("config.yaml"), build_model(), output_dir=tmp_path)
            with pytest.raises(ValueError, match="schema mismatch"):
                predict(tmp_path / "models" / "model.joblib", [{"size": 10}])
        """)
        write(algorithm_dir / "requirements.txt", "-r ../../../requirements.txt\n")
        optional_package = module if module in {"xgboost", "lightgbm", "catboost"} else ""
        docker_install = f"python -m pip install --no-cache-dir -e . {optional_package}".strip()
        write(algorithm_dir / "Dockerfile", f"""
        FROM python:3.11-slim
        WORKDIR /app
        COPY . /app
        RUN {docker_install}
        ENV PYTHONPATH=/app
        CMD ["python", "01_Supervised_Learning/Regression/{folder}/train.py"]
        """)
        write(algorithm_dir / "README.md", f"""
        # {title}

        {intro}

        **Mathematical objective:** {math}

        **Assumptions and limitations:** {caveats}

        **Use case:** {application}

        ## Run
        From the repository root, install `python -m pip install -e .`, then run:
        `python 01_Supervised_Learning/Regression/{folder}/train.py`

        The `config.yaml` controls estimator settings, seed, split, and scaling. The deterministic dataset is synthetic and generated offline by `ml_course.data`; no real-world records are bundled. The preprocessing pipeline is fitted only on training rows. `test_model.py` checks model training, artifact round-trip prediction, and invalid input handling.

        Optional estimator dependencies: {module if module in {"xgboost", "lightgbm", "catboost"} else "none beyond requirements.txt"}.
        """)
        for subdir in ["data/raw", "data/processed", "data/external", "models", "logs", "reports", "artifacts"]:
            write(algorithm_dir / subdir / ".gitkeep", "")

        model_code = f"""        from sklearn.pipeline import Pipeline
        from model import build_model
        model = Pipeline([("preprocessor", preprocessor), ("model", build_model())])
        model.fit(X_train, y_train)
        print("estimator:", model.named_steps["model"])
        print("train/test rows:", len(X_train), len(X_test))
        """
        tuning = {
            "LinearRegression": ("model__fit_intercept", [True, False]),
            "make_pipeline": ("model__polynomialfeatures__degree", [1, 2]),
            "Ridge": ("model__alpha", [0.1, 1.0]),
            "Lasso": ("model__alpha", [0.01, 0.1]),
            "ElasticNet": ("model__alpha", [0.01, 0.1]),
            "DecisionTreeRegressor": ("model__max_depth", [3, 5]),
            "RandomForestRegressor": ("model__n_estimators", [50, 100]),
            "ExtraTreesRegressor": ("model__n_estimators", [50, 100]),
            "GradientBoostingRegressor": ("model__learning_rate", [0.03, 0.1]),
            "XGBRegressor": ("model__max_depth", [3, 5]),
            "LGBMRegressor": ("model__num_leaves", [15, 31]),
            "CatBoostRegressor": ("model__depth", [4, 6]),
            "SVR": ("model__C", [1.0, 10.0]),
            "KNeighborsRegressor": ("model__n_neighbors", [3, 7]),
            "HistGradientBoostingRegressor": ("model__max_leaf_nodes", [7, 15]),
        }
        tuning_key, tuning_values = tuning[klass]
        notebook = make_notebook(title, intro, math, caveats, application, scratch, model_code, tuning_key, tuning_values)
        notebook_path = algorithm_dir / f"{folder.split('_', 1)[1]}.ipynb"
        legacy_notebook_path = algorithm_dir / f"{folder.split('_', 1)[1]}_Regression.ipynb"
        if legacy_notebook_path != notebook_path and legacy_notebook_path.is_file():
            legacy_notebook_path.unlink()
        write(notebook_path, json.dumps(notebook, indent=1, ensure_ascii=False))

    fundamentals = [
        ("Machine Learning Fundamentals", "Define supervised, unsupervised, semi-supervised, self-supervised, and reinforcement learning; distinguish prediction from causal inference.", "Expected loss balances empirical fit and generalization; a model learns a function from examples."),
        ("Statistics", "Samples estimate population quantities with uncertainty. Summaries, confidence intervals, and hypothesis tests answer different questions.", "Mean = sum(xᵢ)/n; variance measures squared spread around the mean."),
        ("Probability", "Random variables model uncertainty; distributions encode possible outcomes and probabilities.", "Bayes' rule: P(A|B)=P(B|A)P(A)/P(B)."),
        ("Linear Algebra", "Vectors represent observations/features; matrices batch linear transformations and covariance relationships.", "AᵀA captures feature inner products; matrix multiplication applies a linear map."),
        ("Bias Variance Tradeoff", "Simple models may miss structure; flexible models may fit noise. Validation estimates generalization.", "Expected squared error decomposes into squared bias, variance, and irreducible noise."),
        ("Overfitting Underfitting", "Compare train and validation performance; use appropriate cross-validation, regularization, and data checks.", "Generalization error is measured on data not used to fit or select the model."),
        ("Feature Engineering", "Use domain knowledge to define valid features at prediction time, transform data within pipelines, and prevent leakage.", "A feature map φ(x) can expose useful structure to a learning algorithm."),
    ]
    for index, (title, concept, formula) in enumerate(fundamentals, 1):
        cells = [
            nb_cell("markdown", f"# {title}\n\n## What and why?\n{concept}\n\n**Hindi:** Pehle concept ko intuitive example se samjhein, phir formula aur code ko connect karein."),
            nb_cell("markdown", f"## Mathematical intuition\n{formula}\n\nAlways define the population, sample, and assumptions before interpreting a statistic."),
            nb_cell("code", """import numpy as np
import pandas as pd
rng = np.random.default_rng(42)
x = rng.normal(loc=5, scale=2, size=1000)
print({"n": len(x), "mean": float(x.mean()), "std": float(x.std(ddof=1))})
display(pd.Series(x, name="sample").describe())
"""),
            nb_cell("markdown", "## Practical exercise\nChange the seed and sample size, compare estimates, and explain why a finite sample differs from the population. Do not treat correlation as causation."),
            nb_cell("code", """from sklearn.model_selection import train_test_split
values = np.arange(20)
train_values, test_values = train_test_split(values, test_size=0.25, random_state=42)
print("train size:", len(train_values), "test size:", len(test_values))
"""),
        ]
        write(ROOT / "00_ML_Fundamentals" / f"{index:02d}_{re.sub(r'[^A-Za-z0-9]+', '_', title).strip('_')}.ipynb", json.dumps({"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}, indent=1))
    write(ROOT / "tests" / "test_data.py", """
    from ml_course.data import make_regression_frame

    def test_synthetic_dataset_is_reproducible() -> None:
        first = make_regression_frame(50, 7)
        second = make_regression_frame(50, 7)
        assert first.equals(second)
        assert first.shape == (50, 5)

    def test_dataset_rejects_too_few_rows() -> None:
        import pytest
        with pytest.raises(ValueError):
            make_regression_frame(5)
    """)
    write(ROOT / "deployment" / "kubernetes.yaml", """
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: regression-inference
    spec:
      replicas: 1
      selector:
        matchLabels:
          app: regression-inference
      template:
        metadata:
          labels:
            app: regression-inference
        spec:
          containers:
            - name: inference
              image: complete-machine-learning:latest
              ports:
                - containerPort: 8000
              readinessProbe:
                httpGet:
                  path: /ready
                  port: 8000
              livenessProbe:
                httpGet:
                  path: /health
                  port: 8000
    """)
    write(ROOT / "deployment" / "api.py", """
    from __future__ import annotations
    import os
    from pathlib import Path
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
    from ml_course.pipeline import predict

    app = FastAPI(title="Regression inference", version="1.0.0")
    MODEL_PATH = Path(os.environ.get("MODEL_PATH", "models/model.joblib"))

    class PredictionRequest(BaseModel):
        rows: list[dict[str, float | str | None]] = Field(min_length=1, max_length=1000)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/ready")
    def ready() -> dict[str, str]:
        if not MODEL_PATH.is_file():
            raise HTTPException(status_code=503, detail="Model artifact is unavailable")
        return {"status": "ready"}

    @app.post("/predict")
    def predict_endpoint(request: PredictionRequest) -> dict[str, list[float]]:
        if not MODEL_PATH.is_file():
            raise HTTPException(status_code=503, detail="Model artifact is unavailable")
        try:
            return {"predictions": predict(MODEL_PATH, request.rows)}
        except (ValueError, KeyError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
    """)
    write(ROOT / "Dockerfile", """
    FROM python:3.11-slim
    WORKDIR /app
    COPY . /app
    RUN python -m pip install --no-cache-dir -e . fastapi uvicorn
    RUN python 01_Supervised_Learning/Regression/01_Linear_Regression/train.py
    ENV PYTHONPATH=/app MODEL_PATH=/app/01_Supervised_Learning/Regression/01_Linear_Regression/models/model.joblib
    EXPOSE 8000
    CMD ["uvicorn", "deployment.api:app", "--host", "0.0.0.0", "--port", "8000"]
    """)
    write(ROOT / ".dockerignore", """
    .venv
    **/__pycache__
    **/.pytest_cache
    **/logs/*.log
    **/.ipynb_checkpoints
    """)
    write(ROOT / ".github" / "workflows" / "ci.yml", """
    name: Python CI
    on:
      push:
      pull_request:
    jobs:
      test:
        runs-on: ubuntu-latest
        steps:
          - uses: actions/checkout@v4
          - uses: actions/setup-python@v5
            with:
              python-version: "3.12"
          - run: python -m pip install -e .
          - run: python -m pytest -q
    """)
    print(f"Generated Phase 1 repository at {ROOT.resolve()}")


if __name__ == "__main__":
    main()
