# End-to-End Machine Learning Methods

An interview-focused learning guide for practical machine learning: **topic -> method -> runnable code -> saved example output -> production trade-offs**. The Jupyter notebook combines the methods cheat sheet with the end-to-end interview guide and labels production-critical practices.

> **[PRODUCTION CORE]** is expected in most reliable ML systems.  
> **[PRODUCTION WHEN NEEDED]** depends on the data, business risk, scale, and architecture.  
> **[LEARNING / OPTIONAL]** is an additional method or requires an optional package.

## Start here

From the repository root, create a Python environment and install the core packages:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install pandas numpy scipy scikit-learn matplotlib seaborn pyarrow openpyxl joblib jupyter
```

Open [the notebook](./EDA_to_Machine_Learning_Python_Methods.ipynb) in JupyterLab or VS Code and run cells from top to bottom. Core examples use small, deterministic, generated or in-memory data. **Outputs are saved** in the notebook; re-run a cell to reproduce them in your own environment. File paths and infrastructure examples are templates and need your project-specific configuration.

### Optional packages

Install only when studying the matching section:

```powershell
python -m pip install imbalanced-learn category-encoders optuna scikit-optimize hyperopt xgboost lightgbm catboost shap lime umap-learn hdbscan mlxtend pandera evidently mlflow fastapi flask uvicorn
```

Great Expectations and TensorFlow/Keras are separate, larger optional dependencies. Docker, Kubernetes, DVC, MLflow server, cloud storage, and CI/CD examples require their respective tools/services; the notebook shows templates and expected behavior but does not deploy infrastructure. Optional cells print a package-install hint when a dependency is missing.

## Files

| File | Purpose |
|---|---|
| [EDA_to_Machine_Learning_Python_Methods.ipynb](./EDA_to_Machine_Learning_Python_Methods.ipynb) | Authoritative, comprehensive curriculum with topic explanations, methods, code, saved outputs, production highlights, and interview practice. |
| [ml_methods_examples.py](../EDA_MLcode_ALL/ml_methods_examples.py) | Earlier supplemental runnable examples; use the notebook for the broader, updated method catalog. |
| [EDA_to_Machine_Learning_Python_Methods (1).txt](../EDA_MLcode_ALL/EDA_to_Machine_Learning_Python_Methods%20(1).txt) | Original EDA-to-ML method cheat sheet. |
| [end_to_end_ml_interview_guide.txt](./end_to_end_ml_interview_guide.txt) | Source interview guide integrated into the notebook, including production lifecycle and 32 practice questions. |

## Complete notebook index

Every section below is represented in the notebook. The examples distinguish locally executed methods from optional or infrastructure templates.

| # | Topic | Methods and coverage | Production |
|---:|---|---|---|
| 0 | Reliable ML rules | Business decision, unit of analysis, target/prediction time, false-positive/negative cost, latency, throughput, privacy, baseline, success metric; target/train-test/time/group leakage; train-only fitting. | **CORE** |
| 1 | Business framing | Problem brief, eligible population, feature availability, acceptance criteria, deployment constraints. | **CORE** |
| 2 | Data ingestion | `read_csv`, `read_excel`, `read_json`, `read_parquet`, `read_sql`, trusted `read_pickle`; `to_csv`, `to_parquet`, SQL table writing; REST/object-store/streaming considerations, lineage, idempotency, retries, raw-data retention. | **CORE** |
| 3 | Dataset inspection and dtypes | `shape`, `size`, `len`, `head`, `tail`, `sample`, `info`, `describe(include="all")`, `dtypes`, `select_dtypes`, `nunique`, `unique`, `value_counts`, `duplicated`, `memory_usage`; numerical, categorical, integer, float, datetime selectors. | **CORE** |
| 4 | EDA and visualization | Numeric mean/median/mode/min/max/range/variance/std/quantiles/skew/kurtosis; category counts/proportions/cardinality/crosstab/group size/mean/aggregation; histograms, KDE, box/violin, ECDF, Q-Q, scatter, pair plot, heatmap, count/bar/pie/stacked chart, scatter matrix. | **WHEN NEEDED** |
| 5 | Statistics and association | Pearson/Spearman/Kendall correlations and tests; independent/paired t-test, one-way ANOVA, Mann-Whitney U, chi-square, mutual information classification/regression. Assumptions and multiple testing are noted. | **WHEN NEEDED** |
| 6 | Data contracts and cleaning | Null/allowed-value/range/ID/duplicate/type/constant checks; `isin`, `between`, `notna`, `drop_duplicates`, `drop`, `rename`, `astype`, `str.strip/lower/upper/replace`, punctuation/special-character cleanup, `to_numeric`, `to_datetime`; optional Pandera and Great Expectations. | **CORE** |
| 7 | Missingness and imputation | `isna`, `isnull`, `notna`, percentages; `dropna` rows/columns/subset; `fillna` mean/median/mode/constants, `ffill`, `bfill`; `SimpleImputer` mean/median/most_frequent/constant, `KNNImputer`, `IterativeImputer`, `MissingIndicator`. | **CORE** |
| 8 | Outliers | IQR, Z-score, modified Z-score/MAD, percentile limits, box plot, `IsolationForest`, `LocalOutlierFactor`, `DBSCAN`, `EllipticEnvelope`, clipping, SciPy winsorization, justified row filtering, `log1p`, `RobustScaler`. | **WHEN NEEDED** |
| 9 | Splitting and validation | `train_test_split`, `KFold`, `StratifiedKFold`, `RepeatedKFold`, `LeaveOneOut`, `GroupKFold`, `StratifiedGroupKFold`, `TimeSeriesSplit`, `cross_val_score`, `cross_validate`. Guidance for stratified, grouped, temporal, and holdout test strategy. | **CORE** |
| 10 | Categorical encoding | `Series.map`, `Series.replace`, category codes, `factorize`, count/frequency encoding, `LabelEncoder` for target labels, `OneHotEncoder(handle_unknown="ignore")`, explicit-order `OrdinalEncoder`, `get_dummies`; optional Binary/Hashing/Count/Target/WOE/LeaveOneOut/CatBoost encoders and scikit-learn target-encoder cross-fitting. | **CORE** |
| 11 | Scaling and distribution transforms | `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `MaxAbsScaler`, `Normalizer`, `QuantileTransformer`, `PowerTransformer` with Yeo-Johnson/Box-Cox. Fit only on training data; method/model suitability notes. | **WHEN NEEDED** |
| 12 | Feature engineering | `log1p`, `sqrt`, `cbrt`, reciprocal with zero guard, Box-Cox, Yeo-Johnson, polynomial and interaction features, `cut`, `qcut`, `KBinsDiscretizer`, `FunctionTransformer`, `assign`, `map`, `apply`, datetime year/month/day/day-of-week/quarter, lag/rolling mean/std, group `agg`/`transform`, CountVectorizer, TfidfVectorizer. | **CORE** |
| 13 | Feature selection and interpretation | `VarianceThreshold`, `SelectKBest(f_classif/chi2/f_regression)`, `mutual_info_classif`, `mutual_info_regression`, correlation, `RFE`, `RFECV`, forward/backward `SequentialFeatureSelector`, Lasso/ElasticNet, `SelectFromModel`, tree `feature_importances_`, `permutation_importance`, optional SHAP. | **WHEN NEEDED** |
| 14 | Dimensionality reduction | `PCA`, `TruncatedSVD`, `KernelPCA`, supervised `LinearDiscriminantAnalysis`, `FastICA`, `TSNE`, `Isomap`, `MDS`, optional `umap.UMAP`, autoencoder (TensorFlow/Keras design template); scale/constraints and training-only fit cautions. | **WHEN NEEDED** |
| 15 | Regression estimators | Linear/polynomial regression; Ridge, Lasso, ElasticNet, BayesianRidge, Huber; DecisionTree, RandomForest, ExtraTrees, GradientBoosting, AdaBoost, HistGradientBoosting; KNN, SVR; optional XGBRegressor, LGBMRegressor, CatBoostRegressor. Common `fit`, `predict`, `score`, parameter inspection/update. | **CORE** |
| 16 | Classification estimators | LogisticRegression, SGDClassifier, Perceptron, KNN, Gaussian/Multinomial/Bernoulli NB, DecisionTree, RandomForest, ExtraTrees, SVC/LinearSVC, LDA/QDA, AdaBoost, GradientBoosting, HistGradientBoosting, MLP; optional XGBClassifier, LGBMClassifier, CatBoostClassifier. | **CORE** |
| 17 | Clustering and anomaly detection | KMeans, Agglomerative, DBSCAN, HDBSCAN, OPTICS, MeanShift, GaussianMixture, SpectralClustering, Birch, AffinityPropagation; IsolationForest, LOF, OneClassSVM, EllipticEnvelope; optional `apriori`, `fpgrowth`, `association_rules`. | **WHEN NEEDED** |
| 18 | Cluster evaluation | `silhouette_score`, `davies_bouldin_score`, `calinski_harabasz_score`, `adjusted_rand_score`, `normalized_mutual_info_score`; interpretation and limitations. | **WHEN NEEDED** |
| 19 | Ensembles | BaggingClassifier/Regressor, RandomForest, ExtraTrees, AdaBoost, GradientBoosting, HistGradientBoosting, VotingClassifier/Regressor, StackingClassifier/Regressor, and XGBoost/LightGBM/CatBoost counterparts. | **WHEN NEEDED** |
| 20 | Hyperparameter tuning | `GridSearchCV`, `RandomizedSearchCV`, `HalvingGridSearchCV`, `HalvingRandomSearchCV`, Optuna `create_study`/`suggest_int`/`suggest_float`/`suggest_categorical`/`study.optimize`, scikit-optimize `BayesSearchCV`, Hyperopt `fmin`/`hp`/`tpe`; training CV, `best_params_`, `best_score_`, `best_estimator_`, holdout evaluation. | **CORE** |
| 21 | Evaluation metrics | Classification: accuracy, precision, recall, F1, confusion matrix/report, ROC-AUC/ROC, PR curve/AP, log loss, MCC, Cohen kappa, balanced accuracy. Regression: MAE, MSE/RMSE, `root_mean_squared_error` when available, R2, explained variance, MAPE, median absolute error, MSLE. | **CORE** |
| 22 | Imbalance and thresholding | `class_weight="balanced"`, RandomOverSampler, RandomUnderSampler, SMOTE, SMOTETomek, SMOTEENN, ADASYN; validation-set threshold choice; `imblearn.pipeline.Pipeline` to resample within training folds only. | **WHEN NEEDED** |
| 23 | Diagnostics and error analysis | Residuals and residual plots, `learning_curve`, `validation_curve`, `calibration_curve`, group-segmented errors, bias/variance, over/underfit, threshold and prediction errors. | **CORE** |
| 24 | Pipelines and leakage prevention | `Pipeline`, `make_pipeline`, `ColumnTransformer`, `make_column_transformer`, numeric/categorical preprocessing, estimator parameter names, imblearn pipeline; target/train-test/time/group leakage and prevention. | **CORE** |
| 25 | Serialization and reproducibility | `joblib.dump/load`, `pickle.dump/load`, complete fitted pipeline, Python/library/data/code/feature versions and artifact hash; trusted artifacts only. | **CORE** |
| 26 | Experiment tracking and registry | MLflow tracking URI/experiment/run, params/metrics/artifacts, sklearn model logging, registration and pyfunc loading; auth, artifact storage, access control, backups. | **WHEN NEEDED** |
| 27 | Data and pipeline versioning | DVC `init`, `add`, `remote add`, `push`, `pull`, `status`, `repro`, `dag`; Git code versus DVC large artifact versioning. | **WHEN NEEDED** |
| 28 | Model APIs | FastAPI health/predict routes, Pydantic schema, `HTTPException`, `predict`/`predict_proba`, Uvicorn; Flask alternative; authentication, request limits, logs, readiness and errors. Templates plus sample response. | **WHEN NEEDED** |
| 29 | Inference architectures | Chunked pandas batch ingestion and prediction/output; real-time request/validation/pipeline/prediction; latency, throughput, scaling, caching, queue, rate-limit concepts. | **WHEN NEEDED** |
| 30 | Docker | Dockerfile, pinned dependencies, non-root, health check, secret/image safety; `build`, `images`, `run`, `ps`, `logs`, `exec`, `stop`, `pull`, `push`, Compose; registries. | **WHEN NEEDED** |
| 31 | Kubernetes | Pod, Deployment, Service, optional Ingress, ConfigMap, Secret, HPA, Namespace, PV/PVC; resource limits/probes; `kubectl get/apply/describe/logs/scale/rollout status/history/undo`. | **WHEN NEEDED** |
| 32 | ML CI/CD | GitHub Actions workflow concepts; Jenkinsfile/pipeline stages; tests, schema/data validation, metric/fairness gates, security scanning, immutable artifact, staging/smoke/promote/rollback. | **CORE** |
| 33 | Monitoring and drift | Data quality, feature/prediction drift, delayed-label model quality, service health/SLOs; KS test, Wasserstein distance, PSI; Evidently, Prometheus, Grafana, OpenTelemetry, ELK/OpenSearch, CloudWatch/Azure Monitor. | **CORE** |
| 34 | Retraining and release safety | Data vs concept drift, label delays, evidence-based retraining, champion/challenger, shadow/canary/A-B testing, versioned registry, rollout and rollback triggers. | **CORE** |
| 35 | Explainability | Permutation importance, SHAP global/local explanations (bar/beeswarm/waterfall), LIME local explanations; attribution is not causality. | **WHEN NEEDED** |
| 36 | Production project and interview practice | Churn project layout/sequence, 32 experienced-practitioner interview questions with answer anchors, recommended study order and readiness checklist. | **CORE** |

## Production principles highlighted in the notebook

- Define the target, prediction time, unit of analysis, business cost, baseline, and acceptance metric before modeling.
- Split first; fit imputation, encoding, scaling, selection, and sampling only in training folds.
- Keep preprocessing and estimator in the same versioned pipeline for training and serving.
- Use time/group-aware validation when random splitting would leak information.
- Do not delete outliers automatically; investigate and document the choice.
- Do not use accuracy alone for imbalanced classification; choose thresholds on validation data.
- Keep the final test set untouched during model selection.
- Validate inputs, version data/code/features/environment/artifacts, monitor service and model quality, and rehearse rollback.
- Never load pickle/joblib files from untrusted sources; never put secrets in source or container images.

## Scope and outputs

The notebook now covers the methods from both source guides with examples, printed outputs, and production guidance. Methods that need external infrastructure (cloud, database service, DVC remote, Docker/Kubernetes cluster, API server, MLflow server) are safe templates with illustrative expected behavior rather than executed deployments. Optional library examples are marked and skip with an install message when the package is unavailable. Package APIs can vary by version; use the environment's installed documentation if a newer/older signature differs.

The previous `ml_methods_examples.py` remains a supplemental code sample and is not the complete curriculum; use the notebook as the comprehensive source.
