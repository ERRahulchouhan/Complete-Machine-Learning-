# End-to-End Machine Learning: Methods, Code, and Interview Preparation

This folder is a practical, interview-focused guide to the end-to-end machine-learning workflow. It combines the topics in [`end_to_end_ml_interview_guide.txt`](./end_to_end_ml_interview_guide.txt) with the detailed methods notebook.

The goal is to make each topic easy to study as:

**Problem → method → code → output → production considerations**

## Files in this folder

| File | What it contains |
|---|---|
| [`EDA_to_Machine_Learning_Python_Methods.ipynb`](./EDA_to_Machine_Learning_Python_Methods.ipynb) | Main learning resource: explanations, method examples, executable code, saved outputs, production notes, optional-library examples, interview questions, and a study checklist. |
| [`end_to_end_ml_interview_guide.txt`](./end_to_end_ml_interview_guide.txt) | Text interview guide: end-to-end lifecycle, core method descriptions, production workflow, and 32 interview questions. |
| [`README.md`](./README.md) | Setup instructions, method index, compact code reference, and study guidance. |

## Open and run the notebook

From this folder, create an environment and install the core dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install pandas numpy scipy scikit-learn matplotlib seaborn pyarrow openpyxl joblib jupyter
```

Open the notebook in VS Code or JupyterLab and run cells from top to bottom. Most core examples use small generated or in-memory datasets. Saved outputs are examples to help with learning; rerun cells to reproduce them in your environment.

Some sections describe production services or optional packages rather than executing a deployment. Install optional packages only when you reach those methods:

```powershell
python -m pip install imbalanced-learn category-encoders optuna scikit-optimize hyperopt xgboost lightgbm catboost shap lime umap-learn hdbscan mlxtend pandera evidently mlflow fastapi flask uvicorn
```

Great Expectations and TensorFlow/Keras are separate optional installs. Docker, Kubernetes, DVC remotes, cloud resources, and hosted MLflow require their own tools and configuration. The notebook marks optional examples and infrastructure templates.

## Production importance key

- **[PRODUCTION CORE]**: expected in most reliable machine-learning workflows.
- **[PRODUCTION WHEN NEEDED]**: use when justified by the problem, data, risk, scale, or architecture.
- **[LEARNING / OPTIONAL]**: useful to know, but not a default production requirement or may need an additional dependency.

## Quick end-to-end code reference

This example shows the central pattern: clean values, split before fitting, impute and encode inside a pipeline, tune with cross-validation, evaluate on a held-out test set, and save the complete fitted pipeline. The notebook has expanded examples and outputs for each step.

```python
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

df = pd.read_csv("customers.csv")

# Normalize inconsistent category values before modeling.
df["city"] = df["city"].replace({"NYC": "New York", "N.Y.": "New York"})
df["plan"] = df["plan"].str.strip().str.lower()

# Example of an explicit category-to-number map.
df["plan_tier"] = df["plan"].map({"basic": 0, "plus": 1, "premium": 2})

X = df.drop(columns=["churn"])
y = df["churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

numeric_columns = ["age", "monthly_spend", "plan_tier"]
categorical_columns = ["city", "plan"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns),
])
model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000)),
])

search = GridSearchCV(
    model_pipeline,
    param_grid={"model__C": [0.1, 1.0, 10.0]},
    cv=5,
    scoring="f1",
    n_jobs=-1,
)
search.fit(X_train, y_train)

predictions = search.predict(X_test)
print("Best parameters:", search.best_params_)
print(classification_report(y_test, predictions))

# Save preprocessing and estimator together to keep serving consistent.
joblib.dump(search.best_estimator_, "churn_pipeline.joblib")
```

**Adapt before running:** replace the file path, target, feature names, category values, and scoring metric with values from your dataset. Ensure the target has enough examples per class for the selected split and cross-validation. `Series.map` returns missing values for unmapped categories; check and handle those explicitly. Load pickle/joblib artifacts only from trusted sources.

## Complete method and topic index

The notebook contains code for the methods below. Core examples generally run locally; methods requiring an optional dependency or real infrastructure are labeled in the notebook and use an availability check or a clearly identified template.

| # | Topic | Methods and code coverage | Priority |
|---:|---|---|---|
| 0 | Reliable ML rules | Define the decision, target, unit of analysis, prediction time, baseline, success metric, constraints, and error costs. Prevent target, train/test, time, and group leakage; fit learned transformations on training data only. | **CORE** |
| 1 | Business understanding | Problem definition, eligible population, feature availability, acceptance criteria, latency, throughput, privacy, explainability, and cost. | **CORE** |
| 2 | Data ingestion and output | `pd.read_csv`, `read_excel`, `read_json`, `read_parquet`, `read_sql`, `to_datetime`, trusted `read_pickle`; `to_csv`, `to_parquet`, SQL output; REST/API, object-store, and streaming considerations. | **CORE** |
| 3 | Dataset inspection and types | `head`, `tail`, `sample`, `shape`, `size`, `len`, `info`, `describe`, `dtypes`, `columns`, `select_dtypes`, `nunique`, `unique`, `value_counts`, `duplicated`, `memory_usage`. | **CORE** |
| 4 | EDA, statistics, and plots | Mean, median, mode, min/max/range, variance, standard deviation, quantiles, skew, kurtosis, correlations, groupby/aggregation, crosstab; histograms, KDE, box/violin, ECDF, Q-Q, scatter, pair plot, heatmap, bar/count/pie/stacked plots, scatter matrix. | **WHEN NEEDED** |
| 5 | Statistical tests and association | Pearson/Spearman/Kendall correlation; independent/paired t-test, one-way ANOVA, Mann–Whitney U, chi-square, mutual information for classification/regression; assumptions and multiple-testing considerations. | **WHEN NEEDED** |
| 6 | Data validation and cleaning | `isna`, `notna`, `isin`, `between`, duplicate/key checks, allowed values and schema; `drop_duplicates`, `drop`, `rename`, `astype`, `to_numeric`, `to_datetime`, string `strip`, `lower`, `upper`, `replace`; Pandera and Great Expectations. | **CORE** |
| 7 | Missing data | `isna`, `isnull`, `notna`, missing percentages, `dropna`, `fillna`, `ffill`, `bfill`; `SimpleImputer` with mean/median/most-frequent/constant, `KNNImputer`, `IterativeImputer`, and `MissingIndicator`. | **CORE** |
| 8 | Outliers | IQR, Z-score, modified Z-score/MAD, percentile limits, box plots, clipping, SciPy winsorization, justified row filtering, `log1p`, `RobustScaler`, `IsolationForest`, `LocalOutlierFactor`, `DBSCAN`, `EllipticEnvelope`. | **WHEN NEEDED** |
| 9 | Data splitting and validation | `train_test_split`, stratification, `KFold`, `StratifiedKFold`, `RepeatedKFold`, `LeaveOneOut`, `GroupKFold`, `GroupShuffleSplit`, `StratifiedGroupKFold`, `TimeSeriesSplit`, `cross_val_score`, `cross_validate`, `cross_val_predict`. | **CORE** |
| 10 | Categorical encoding | `Series.map`, `Series.replace`, category codes, `factorize`, frequency/count encoding, `LabelEncoder` for target labels, `OneHotEncoder`, ordered `OrdinalEncoder`, `get_dummies`; optional Binary, Hashing, Count, Target, WOE, LeaveOneOut, and CatBoost encoders. | **CORE** |
| 11 | Scaling and distribution transforms | `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `MaxAbsScaler`, `Normalizer`, `QuantileTransformer`, `PowerTransformer` (Yeo–Johnson/Box–Cox), `FunctionTransformer`, `log1p`, `KBinsDiscretizer`. | **WHEN NEEDED** |
| 12 | Feature engineering | `assign`, `map`, `apply`, type conversion, `cut`, `qcut`, polynomial/interactions, square-root/cube-root/reciprocal/log transforms, datetime parts, text cleanup, lag/rolling features, group `agg`/`transform`, `CountVectorizer`, `TfidfVectorizer`. | **CORE** |
| 13 | Feature selection and interpretation | `VarianceThreshold`, `SelectKBest` (`f_classif`, `chi2`, `f_regression`), mutual information, correlation filtering, `RFE`, `RFECV`, forward/backward `SequentialFeatureSelector`, Lasso/ElasticNet, `SelectFromModel`, tree importance, `permutation_importance`, optional SHAP. | **WHEN NEEDED** |
| 14 | Dimensionality reduction | `PCA`, `TruncatedSVD`, `KernelPCA`, `LinearDiscriminantAnalysis`, `FastICA`, `TSNE`, `Isomap`, `MDS`, optional UMAP and an autoencoder template. | **WHEN NEEDED** |
| 15 | Regression algorithms | Linear and polynomial regression; Ridge, Lasso, ElasticNet, BayesianRidge, Huber; DecisionTree, RandomForest, ExtraTrees, GradientBoosting, AdaBoost, HistGradientBoosting; KNN, SVR; optional XGBoost, LightGBM, CatBoost regressors. | **CORE** |
| 16 | Classification algorithms | LogisticRegression, SGDClassifier, Perceptron, KNN, Gaussian/Multinomial/Bernoulli Naive Bayes, DecisionTree, RandomForest, ExtraTrees, SVC/LinearSVC, LDA/QDA, AdaBoost, GradientBoosting, HistGradientBoosting, MLP; optional XGBoost, LightGBM, CatBoost classifiers. | **CORE** |
| 17 | Clustering, anomalies, and association | KMeans, Agglomerative, DBSCAN, HDBSCAN, OPTICS, MeanShift, GaussianMixture, SpectralClustering, Birch, AffinityPropagation; IsolationForest, LOF, OneClassSVM, EllipticEnvelope; optional `apriori`, `fpgrowth`, `association_rules`. | **WHEN NEEDED** |
| 18 | Cluster evaluation | `silhouette_score`, `davies_bouldin_score`, `calinski_harabasz_score`, `adjusted_rand_score`, `normalized_mutual_info_score`; interpretation and limitations. | **WHEN NEEDED** |
| 19 | Ensemble learning | Bagging, Random Forest, Extra Trees, AdaBoost, Gradient Boosting, HistGradientBoosting, voting and stacking classifiers/regressors, and optional boosted-tree libraries. | **WHEN NEEDED** |
| 20 | Hyperparameter tuning | `GridSearchCV`, `RandomizedSearchCV`, `HalvingGridSearchCV`, `HalvingRandomSearchCV`; Optuna `create_study`, `suggest_int`, `suggest_float`, `suggest_categorical`, `study.optimize`; optional `BayesSearchCV` and Hyperopt. Inspect `best_params_`, `best_score_`, `best_estimator_`. | **CORE** |
| 21 | Evaluation metrics | Classification: accuracy, precision, recall, F1, confusion matrix/report, ROC-AUC/ROC, PR curve/AP, log loss, MCC, Cohen kappa, balanced accuracy. Regression: MAE, MSE/RMSE, R², explained variance, MAPE, median absolute error, MSLE. | **CORE** |
| 22 | Imbalance and thresholds | `class_weight="balanced"`, random over/under-sampling, SMOTE, ADASYN, SMOTETomek, SMOTEENN; validation-set threshold selection and imbalanced-learn pipelines to resample within training folds. | **WHEN NEEDED** |
| 23 | Diagnostics and error analysis | Residuals/plots, segment-level errors, `learning_curve`, `validation_curve`, `calibration_curve`, bias/variance, over/underfitting, threshold and prediction error analysis. | **CORE** |
| 24 | Pipelines and leakage prevention | `Pipeline`, `make_pipeline`, `ColumnTransformer`, `make_column_transformer`; mixed numeric/categorical preprocessing; estimator parameter names; target/train-test/time/group leakage prevention. | **CORE** |
| 25 | Serialization and reproducibility | `joblib.dump/load`, `pickle.dump/load`, save the fitted preprocessing-plus-model pipeline; record code, data, feature, dependency, and model versions. Trusted artifacts only. | **CORE** |
| 26 | MLflow tracking and registry | Tracking URI, experiment/run, parameter/metric/artifact logging, sklearn model logging, registration and pyfunc loading; secure artifact storage and access. | **WHEN NEEDED** |
| 27 | Data and pipeline versioning | DVC `init`, `add`, `remote add`, `push`, `pull`, `status`, `repro`, `dag`; Git for code and DVC for data/artifacts. | **WHEN NEEDED** |
| 28 | Model serving | FastAPI health/predict routes, Pydantic request models, `HTTPException`, `predict`/`predict_proba`, Uvicorn; Flask alternative; request validation, auth, logs, readiness and errors. | **WHEN NEEDED** |
| 29 | Batch and real-time inference | Chunked reads, batch prediction/output, request validation, fitted preprocessing and prediction; latency, throughput, horizontal/vertical scaling, caching, queues, and rate limits. | **WHEN NEEDED** |
| 30 | Docker | Dockerfile and pinned dependencies; build/run/log/stop/pull/push/Compose commands; non-root users, health checks, immutable tags, image scanning, and secret handling. | **WHEN NEEDED** |
| 31 | Kubernetes | Pod, Deployment, Service, optional Ingress, ConfigMap, Secret, HPA, Namespace, PV/PVC; resource requests/limits, probes, `kubectl apply/get/describe/logs/scale/rollout/undo`. | **WHEN NEEDED** |
| 32 | ML CI/CD | GitHub Actions and Jenkins concepts; tests, schema/data validation, reproducibility, metric/fairness gates, security checks, image build, staging, smoke tests, promotion, and rollback. | **CORE** |
| 33 | Monitoring and drift | Input quality, schema, data drift, delayed-label model quality, service health/SLOs; KS test, Wasserstein distance, PSI; Evidently, Prometheus, Grafana, OpenTelemetry, ELK/OpenSearch, CloudWatch/Azure Monitor. | **CORE** |
| 34 | Retraining and release safety | Data vs. concept drift, evidence-based retraining, champion/challenger, version registry, safe rollout, rollback triggers, and retaining the last known-good model. | **CORE** |
| 35 | Explainability | Permutation importance, SHAP global/local explanations (bar, beeswarm, waterfall), LIME local explanations; importance is not causality. | **WHEN NEEDED** |
| 36 | Production project and interview practice | Customer-churn project structure and delivery sequence; 32 interview questions; recommended study order and production checklist. | **CORE** |

## Essential production rules

1. Define the target, prediction time, unit of analysis, business cost, baseline, and success metric before modeling.
2. Split data before fitting imputers, encoders, scalers, feature selectors, or sampling methods. Fit them only on training data/folds.
3. Keep preprocessing and the estimator in one fitted pipeline and reuse that same artifact for serving.
4. Use group-aware or time-aware splits when random splitting could leak entities or future information.
5. Investigate outliers; do not remove valid rare or high-value cases automatically.
6. Choose evaluation metrics and classification thresholds for the business objective; accuracy alone can mislead.
7. Keep the final test set untouched during model selection.
8. Validate inputs and version the data, code, features, environment, and artifacts. Monitor and rehearse rollback.
9. Never load pickle/joblib artifacts from untrusted sources. Do not put credentials in code or container images.

## Suggested study order

1. Python, Pandas, NumPy, EDA, and statistics.
2. Data quality, cleaning, missing values, and outliers.
3. Splitting, encoding, scaling, and feature engineering.
4. Regression/classification algorithms, ensembles, and model selection.
5. Metrics, cross-validation, imbalance, diagnostics, and leakage.
6. Pipelines, hyperparameter tuning, serialization, and reproducibility.
7. MLflow, DVC, model serving, batch inference, and Docker.
8. CI/CD, Kubernetes/cloud concepts, monitoring, drift, retraining, registry, rollback, and explainability.

For each method, practice explaining **what it does, when to use it, key parameters, assumptions, failure modes, code pattern, and production trade-offs**. The notebook is the primary resource for detailed code and example outputs; the TXT guide is a quick interview-preparation companion.
