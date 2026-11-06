# Complete Machine Learning

A phased, executable machine-learning curriculum. The fundamentals, 15 regression algorithms, and core unsupervised clustering and dimensionality-reduction algorithms are implemented. Other phases are explicitly listed in `LEARNING_ROADMAP.md`.

## Setup
```powershell
cd Complete-Machine-Learning
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```
Optional native boosters: `python -m pip install -r requirements-optional.txt` (also installs deployment/explainability tools).

## Run an algorithm
```powershell
python 01_Supervised_Learning\Regression\01_Linear_Regression\train.py
python 01_Supervised_Learning\Regression\01_Linear_Regression\predict.py --input '[{"size":100,"rooms":3,"age":10,"distance":2}]'
python -m pytest
```
Training uses a deterministic synthetic offline housing-like dataset. It is for learning only, not a real estate dataset or production recommendation. Estimators, preprocessing, split seed, metrics, and artifact path are configuration-driven. Train/test split precedes fitted preprocessing; holdout data does not fit preprocessing.

Open each `*_Regression.ipynb` in Jupyter. Notebooks contain 21 cell-wise sections and separate executable cells. For advanced boosters install their corresponding optional package before training. Python 3.10+ is supported; use package versions compatible with your Python platform.

## Unsupervised learning
Open the notebooks under `Unsupervised_Learning\Clustering` and `Unsupervised_Learning\Dimensionality_Reduction`. They use deterministic synthetic data or scikit-learn's built-in digits dataset, so no external dataset download is required. Each notebook explains one algorithm, prepares data, fits and evaluates an example, and discusses limitations. Clustering covers K-means, agglomerative clustering, DBSCAN, OPTICS, Gaussian mixtures, Mean Shift, spectral clustering, BIRCH, Affinity Propagation, and HDBSCAN. Dimensionality reduction covers PCA, Incremental PCA, Kernel PCA, Truncated SVD, FastICA, NMF, Factor Analysis, t-SNE, Isomap, and Locally Linear Embedding.

For practical pandas workflows in ML data preparation, see [`pandas.md`](./pandas.md).
For model-evaluation, explainability, and production-monitoring visualizations, see [`chart.md`](./chart.md).
For leakage-safe feature processing, selection methods, and production feature governance, see [`feature_selection_and_feature_processing.md`](./feature_selection_and_feature_processing.md).
For the complete model lifecycle from problem definition through deployment and monitoring, see [`model_develoment_pipeline.md`](./model_develoment_pipeline.md).

## Exploratory data analysis and processing — completed
The EDA notebooks cover dataset inspection, univariate and multivariate analysis, data quality checks, and feature preparation. Use the methods below in order; choose only the methods appropriate for the dataset and model.

| Order | Stage | Method names | Use | Production? |
|---:|---|---|---|---|
| 1 | Load and combine data | `pandas.read_csv()`, `pandas.read_excel()`, `pandas.concat()`, `pandas.merge()`, `DataFrame.join()` | Load files and combine related tables; validate keys and row counts. | **Yes**, with schema, source, and data-quality validation. |
| 2 | Inspect structure and quality | `DataFrame.head()`, `DataFrame.shape`, `DataFrame.info()`, `DataFrame.describe()`, `DataFrame.dtypes`, `DataFrame.isna().sum()`, `DataFrame.duplicated().sum()`, `Series.nunique()`, `Series.value_counts()` | Check types, ranges, missingness, duplicates, frequencies, and category counts. Resolve or document data-quality issues. | **Yes**, as validation and monitoring checks; avoid relying on manual inspection alone. |
| 3 | Analyze individual columns | `Series.describe()`, `Series.hist()`, `seaborn.countplot()`, `seaborn.histplot()`, `seaborn.boxplot()` | Summarize category frequencies and numeric distributions; inspect unusual values. | **Mostly offline** for exploration; production monitoring may track approved summary statistics and drift. |
| 4 | Analyze relationships and outliers | `DataFrame.corr()`, `pandas.crosstab()`, `seaborn.scatterplot()`, `seaborn.heatmap()`; domain thresholds, IQR, `scipy.stats.zscore()`, `scipy.stats.median_abs_deviation()`, percentile bounds/`DataFrame.clip()`, `IsolationForest`, `LocalOutlierFactor` | Explore feature relationships and flag unusual records. Investigate flags before removing or capping; Z-score is sensitive to skew, while MAD/IQR are more robust. | **Conditional**: document thresholds and monitor validated rules. Persist learned detectors if needed at inference. |
| 5 | Split the data | `sklearn.model_selection.train_test_split()`, `GroupShuffleSplit`, `TimeSeriesSplit` | Split before learned preprocessing; use stratified, group, or time-based splits as appropriate. | **Training/evaluation only**, not request-time inference; preserve split strategy for reproducible retraining. |
| 6 | Impute missing values | `sklearn.impute.SimpleImputer(strategy="median")`, `SimpleImputer(strategy="mean")`, `SimpleImputer(strategy="most_frequent")`, `SimpleImputer(strategy="constant")`, `sklearn.impute.KNNImputer()`, `DataFrame.fillna()` | Choose a feature-appropriate strategy; fit learned statistics on training data. Use `fillna()` for explicit fixed rules. | **Yes, conditionally**: persist the fitted imputer in the inference pipeline. Do not recalculate statistics independently on production batches. |
| 7 | Encode categorical data | `sklearn.preprocessing.OneHotEncoder(handle_unknown="ignore")`, `sklearn.preprocessing.OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)`, `pd.get_dummies()`, `Series.map()`, `Series.replace()`, `sklearn.preprocessing.LabelEncoder()`, `category_encoders.TargetEncoder()` | One-hot encode nominal and ordinal-encode ordered features. Use `map`/`replace` for fixed mappings; `LabelEncoder` for target labels only. | **Yes, preferably** with persisted pipeline encoders and unknown-category handling. Use `pd.get_dummies()` mainly for exploration; target encoding requires leakage controls. |
| 8 | Transform numerical data | Domain/unit conversion; `numpy.log()`, `numpy.log1p()`, `numpy.sqrt()`, `numpy.reciprocal()`; `sklearn.preprocessing.PowerTransformer(method="yeo-johnson"|"box-cox")`; `sklearn.preprocessing.QuantileTransformer()`; `sklearn.preprocessing.KBinsDiscretizer()`; `sklearn.preprocessing.PolynomialFeatures()`; `sklearn.preprocessing.SplineTransformer()`; `sklearn.preprocessing.FunctionTransformer()` | Select by feature meaning and model; check input domains. Box-Cox needs positive values; Yeo-Johnson supports zero and negatives. | **Yes, conditionally**: fit learned transforms on training data and persist in the inference pipeline. |
| 9 | Scale numerical data | `sklearn.preprocessing.StandardScaler()`, `sklearn.preprocessing.MinMaxScaler()`, `sklearn.preprocessing.RobustScaler()` | Scale for scale-sensitive models; tree-based models usually do not need it. | **Yes, conditionally**: fit on training data and persist for inference; do not refit per request or batch. |
| 10 | Select features and validate | `sklearn.feature_selection.VarianceThreshold()`, `sklearn.feature_selection.SelectKBest()`, `sklearn.feature_selection.mutual_info_classif()`, `statsmodels.stats.outliers_influence.variance_inflation_factor()` | Select using training data and a target-appropriate score; assess with validation data. VIF is diagnostic. | **Yes, conditionally**: persist fitted selectors and validate performance and feature schema after retraining. |
| 11 | Compose and persist preprocessing | `sklearn.compose.ColumnTransformer()`, `sklearn.pipeline.Pipeline()`, `sklearn.compose.make_column_selector()` | Apply separate transformations to numeric and categorical columns, then pass the result to the estimator in one consistent workflow. | **Recommended**: fit once on training data, persist the full pipeline, and use the same fitted pipeline for inference. |
| 12 | Validate models and tune parameters | `sklearn.model_selection.cross_validate()`, `StratifiedKFold`, `GroupKFold`, `TimeSeriesSplit`, `GridSearchCV`, `RandomizedSearchCV`; task-appropriate metrics | Compare candidates and tune parameters using training/validation data; reserve the test set for final evaluation. | **Training/retraining only**: record folds, metrics, parameters, and data versions; do not tune against production traffic or the held-out test set. |
| 13 | Validate inputs and persist model | `pandera.DataFrameSchema`, `pydantic.BaseModel`, `sklearn.utils.validation.check_is_fitted()`, `joblib.dump()`, `joblib.load()` | Enforce expected columns, types, ranges, and required fields; save and load the fitted preprocessing-plus-model pipeline. | **Recommended**: validate every production request, version model and schema together, and load artifacts only from trusted sources. |

### Feature selection methods: why and when

Feature selection is optional: compare each approach with a no-selection baseline using cross-validation. Remove leakage and unavailable-at-prediction-time fields first. Fit every data-dependent selector inside the training folds or a pipeline; never select features using the test set.

| Method | Why use it | When to use it / cautions |
|---|---|---|
| Domain-based selection | Keeps valid, available, interpretable inputs. | Start here; exclude IDs, leakage, post-outcome data, and prohibited fields based on documented rules. |
| `VarianceThreshold` | Removes constant or near-constant columns cheaply. | Unsupervised first filter; scale-dependent for non-binary data and does not consider the target. |
| Correlation filter | Removes redundant numeric predictors. | Useful before linear models; choose one of highly correlated pairs using domain knowledge. Correlation is not evidence that a feature is useless. |
| `SelectKBest` with `f_classif` / `f_regression` | Fast univariate ranking by linear association. | Good baseline for numeric features; can miss interactions and nonlinear effects. |
| `SelectKBest` with `mutual_info_classif` / `mutual_info_regression` | Detects broader, including nonlinear, univariate dependence. | Use when relationships may be nonlinear; score estimates can vary and ignore feature interactions. |
| `SelectKBest` with `chi2` | Scores non-negative counts/frequencies for classification. | Use only with non-negative input; do not apply directly to centered/scaled values that can be negative. |
| `RFE` / `RFECV` | Recursively removes weaker features using an estimator. | Use when feature count is manageable and the estimator has meaningful coefficients/importances; `RFECV` uses CV but can be computationally expensive. |
| `SelectFromModel` (L1 / tree model) | Selects using a fitted model's coefficients or importances. | Useful for sparse linear models or tree-based models; results depend on model, regularization, and importance bias. |
| `SequentialFeatureSelector` | Searches forward or backward for a subset that scores well with a chosen estimator. | Useful for small/medium feature sets; computationally expensive and CV must respect group/time structure. |
| Permutation importance | Measures validation-score drop when a feature is shuffled. | Use after fitting a credible model to compare predictive contribution; correlated features can mask each other's importance. Compute on held-out validation data, not the test set used for final reporting. |
| VIF | Diagnoses multicollinearity in numeric predictors. | Useful for coefficient stability/interpretation in linear regression; it is a diagnostic rather than a universal selection rule and needs `statsmodels`. |

#### Feature selection code examples

Examples assume `X_train` is numeric and imputed, `y_train` is the target, and the task/scoring match the actual problem. These selectors are alternatives; do not apply all of them automatically.

```python
import pandas as pd
import numpy as np
from functools import partial
from sklearn.feature_selection import (
    RFE,
    RFECV,
    SelectFromModel,
    SelectKBest,
    SequentialFeatureSelector,
    VarianceThreshold,
    f_classif,
    mutual_info_classif,
    f_regression,
    mutual_info_regression,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.svm import LinearSVC

# 1. Remove constant columns; use feature names to inspect what remains.
variance_selector = VarianceThreshold(threshold=0.0)
X_train_nonconstant = variance_selector.fit_transform(X_train)
nonconstant_features = X_train.columns[variance_selector.get_support()]

# 2. Univariate classification ranking: choose one scoring function.
univariate_selector = SelectKBest(score_func=f_classif, k=10)
X_train_kbest = univariate_selector.fit_transform(X_train, y_train)
kbest_features = X_train.columns[univariate_selector.get_support()]

mi_selector = SelectKBest(
    score_func=partial(mutual_info_classif, random_state=42), k=10
)
X_train_mi = mi_selector.fit_transform(X_train, y_train)
mi_features = X_train.columns[mi_selector.get_support()]

# For regression, use SelectKBest(score_func=f_regression, k=10)
# or partial(mutual_info_regression, random_state=42).
# For non-negative count/frequency data, chi2 is another classification option.

# 3. Correlation filter: keep the first column from each highly correlated pair.
corr = X_train.corr(numeric_only=True).abs()
upper_triangle = corr.where(
    np.triu(np.ones(corr.shape, dtype=bool), k=1)
)
correlated_to_drop = [
    column for column in upper_triangle.columns
    if (upper_triangle[column] > 0.95).any()
]
X_train_uncorrelated = X_train.drop(columns=correlated_to_drop)

# 4. Recursive feature elimination; RFECV chooses feature count by CV.
base_model = LogisticRegression(max_iter=1000)
rfe = RFE(estimator=base_model, n_features_to_select=10)
X_train_rfe = rfe.fit_transform(X_train, y_train)
rfe_features = X_train.columns[rfe.get_support()]

rfecv = RFECV(
    estimator=base_model,
    step=1,
    cv=5,
    scoring="roc_auc",
)
rfecv.fit(X_train, y_train)
rfecv_features = X_train.columns[rfecv.get_support()]

# 5. Model-based selection (L1 sparsity or tree feature importances).
l1_selector = SelectFromModel(
    LinearSVC(penalty="l1", dual=False, C=0.1, max_iter=1000)
)
X_train_l1 = l1_selector.fit_transform(X_train, y_train)
l1_features = X_train.columns[l1_selector.get_support()]

tree_selector = SelectFromModel(
    RandomForestClassifier(n_estimators=200, random_state=42),
    threshold="median",
)
X_train_tree = tree_selector.fit_transform(X_train, y_train)
tree_features = X_train.columns[tree_selector.get_support()]

# 6. Sequential selection; expensive for wide feature sets.
sequential_selector = SequentialFeatureSelector(
    base_model,
    n_features_to_select=10,
    direction="forward",
    scoring="roc_auc",
    cv=5,
)
X_train_sequential = sequential_selector.fit_transform(X_train, y_train)
sequential_features = X_train.columns[sequential_selector.get_support()]

# 7. Permutation importance on validation data after fitting a model.
validation_model = RandomForestClassifier(n_estimators=200, random_state=42)
validation_model.fit(X_train, y_train)
importance = permutation_importance(
    validation_model,
    X_valid,
    y_valid,
    scoring="roc_auc",
    n_repeats=10,
    random_state=42,
)
permutation_scores = pd.Series(
    importance.importances_mean,
    index=X_valid.columns,
).sort_values(ascending=False)
```

Use `f_regression` or `partial(mutual_info_regression, random_state=42)` for regression targets instead of classification scoring functions. `chi2` is another option for non-negative classification features. For one-hot encoded or transformed data, inspect `get_feature_names_out()` to map selected columns back to readable feature names.

VIF is an optional diagnostic for numeric predictors and requires `statsmodels`:

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.Series(
    [
        variance_inflation_factor(X_train.to_numpy(), i)
        for i in range(X_train.shape[1])
    ],
    index=X_train.columns,
    name="VIF",
)
```

In the correlation example, the intended operation is to inspect only the upper triangle of the correlation matrix. In production model training, put selectors in the estimator `Pipeline` so cross-validation fits them independently within each fold; persist the fitted selector with the model and monitor the resulting input/output feature schema.

### Production preprocessing pipeline example

Replace `X`, `y`, and the feature column lists with the dataset's features and target. The pipeline learns imputers, scaler, and encoder from the training split only; unknown categories are safely ignored during transform.

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

numeric_features = ["age", "income"]
categorical_features = ["city", "plan"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])
model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000)),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model_pipeline.fit(X_train, y_train)
test_score = model_pipeline.score(X_test, y_test)
predictions = model_pipeline.predict(X_test)
```

### Numerical transformation code examples

These are alternatives, not steps to apply all at once. `X_train` and `X_test` should already be split and have missing values handled. Fit each learned transformation on training data only.

```python
import numpy as np
from sklearn.preprocessing import (
    FunctionTransformer,
    KBinsDiscretizer,
    PolynomialFeatures,
    PowerTransformer,
    QuantileTransformer,
    SplineTransformer,
)

# Fixed domain/unit conversion: use the same conversion at training and inference.
df["distance_km"] = df["distance_m"] / 1000

# Simple transforms: apply only when the feature values meet the stated domain.
log_values = np.log1p(X_train["nonnegative_feature"])  # values must be >= 0
sqrt_values = np.sqrt(X_train["nonnegative_feature"])  # values must be >= 0
reciprocal_values = 1 / X_train["nonzero_feature"]  # values must not be 0

# Yeo-Johnson supports zero and negative values. Box-Cox requires values > 0.
power_transformer = PowerTransformer(method="yeo-johnson")
X_train_power = power_transformer.fit_transform(X_train[numeric_columns])
X_test_power = power_transformer.transform(X_test[numeric_columns])

# For strictly positive columns only:
box_cox = PowerTransformer(method="box-cox")
X_train_box_cox = box_cox.fit_transform(X_train[positive_columns])
X_test_box_cox = box_cox.transform(X_test[positive_columns])

# Map each feature approximately to a normal distribution (nonlinear).
quantile_transformer = QuantileTransformer(
    n_quantiles=min(1000, len(X_train)),
    output_distribution="normal",
    random_state=42,
)
X_train_quantile = quantile_transformer.fit_transform(X_train[numeric_columns])
X_test_quantile = quantile_transformer.transform(X_test[numeric_columns])

# Discretize continuous features into quantile-based bins.
binner = KBinsDiscretizer(n_bins=5, encode="onehot", strategy="quantile")
X_train_binned = binner.fit_transform(X_train[numeric_columns])
X_test_binned = binner.transform(X_test[numeric_columns])

# Add polynomial or spline basis features for models that need nonlinear effects.
polynomial = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = polynomial.fit_transform(X_train[numeric_columns])
X_test_poly = polynomial.transform(X_test[numeric_columns])

spline = SplineTransformer(n_knots=5, degree=3)
X_train_spline = spline.fit_transform(X_train[["one_numeric_feature"]])
X_test_spline = spline.transform(X_test[["one_numeric_feature"]])

# Wrap a deterministic function when it must be composed in a sklearn pipeline.
log_transformer = FunctionTransformer(np.log1p, feature_names_out="one-to-one")
X_train_log = log_transformer.fit_transform(X_train[["nonnegative_feature"]])
X_test_log = log_transformer.transform(X_test[["nonnegative_feature"]])
```

In production, place the selected transformation in the same versioned `Pipeline`/`ColumnTransformer` as the estimator. Do not fit a transformer on each request or the full dataset before splitting. Use domain validation instead of silently clipping invalid values; ensure training and inference use the same feature order and transformation parameters.

### Outlier detection code examples

The examples assume `df` is loaded and `numeric_columns` contains the numeric feature names. For the learned multivariate detectors, `X_train` and `X_test` must already be split and imputed; fit on training data only. Detection flags are prompts to investigate, not automatic instructions to delete rows.

```python
import pandas as pd
from scipy.stats import median_abs_deviation, zscore
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

# 1. Domain/business rule: set limits from valid subject-matter requirements.
domain_flags = (df["age"] < 0) | (df["age"] > 120)

# 2. IQR: learn bounds from training data and flag held-out rows.
q1 = X_train[numeric_columns].quantile(0.25)
q3 = X_train[numeric_columns].quantile(0.75)
iqr = q3 - q1
lower_iqr = q1 - 1.5 * iqr
upper_iqr = q3 + 1.5 * iqr
iqr_flags = (X_test[numeric_columns] < lower_iqr) | (
    X_test[numeric_columns] > upper_iqr
)

# 3. Z-score: fit mean/std on training data; most useful for near-normal features.
train_mean = X_train[numeric_columns].mean()
train_std = X_train[numeric_columns].std().replace(0, float("nan"))
z_scores = (X_test[numeric_columns] - train_mean) / train_std
zscore_flags = z_scores.abs() > 3

# 4. Modified Z-score: median/MAD is more robust to extreme values.
train_median = X_train[numeric_columns].median()
train_mad = X_train[numeric_columns].apply(
    lambda column: median_abs_deviation(column.dropna(), scale=1.0)
).replace(0, float("nan"))
modified_z_scores = 0.6745 * (X_test[numeric_columns] - train_median) / train_mad
mad_flags = modified_z_scores.abs() > 3.5

# 5. Percentile bounds: fit on train; flag or cap test values consistently.
lower = X_train[numeric_columns].quantile(0.01)
upper = X_train[numeric_columns].quantile(0.99)
percentile_flags = (X_test[numeric_columns] < lower) | (
    X_test[numeric_columns] > upper
)
X_train_capped = X_train.copy()
X_test_capped = X_test.copy()
X_train_capped[numeric_columns] = X_train[numeric_columns].clip(lower, upper, axis=1)
X_test_capped[numeric_columns] = X_test[numeric_columns].clip(lower, upper, axis=1)

# 6. Isolation Forest: multivariate; contamination needs validation.
isolation_forest = IsolationForest(contamination=0.02, random_state=42)
isolation_forest.fit(X_train[numeric_columns])
isolation_flags = isolation_forest.predict(X_test[numeric_columns]) == -1

# Scale features before distance/kernel/covariance-based methods.
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X_train[numeric_columns])
X_train_scaled = scaler.transform(X_train[numeric_columns])
X_test_scaled = scaler.transform(X_test[numeric_columns])

# 7. Local Outlier Factor: novelty=True enables scoring unseen rows.
lof = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.02)
lof.fit(X_train_scaled)
lof_flags = lof.predict(X_test_scaled) == -1

# 8. One-Class SVM: useful for novelty detection; tune nu/gamma on validation data.
one_class_svm = OneClassSVM(kernel="rbf", nu=0.02, gamma="scale")
one_class_svm.fit(X_train_scaled)
svm_flags = one_class_svm.predict(X_test_scaled) == -1

# 9. Elliptic Envelope: assumes one dominant, approximately Gaussian cluster.
elliptic = EllipticEnvelope(contamination=0.02, random_state=42)
elliptic.fit(X_train_scaled)
elliptic_flags = elliptic.predict(X_test_scaled) == -1

# 10. DBSCAN: labels training-set noise as -1; eps must be tuned after scaling.
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_train_scaled)
dbscan_train_flags = dbscan_labels == -1

# Summarize flags without automatically deleting valid observations.
outlier_summary = pd.DataFrame({
    "iqr": iqr_flags.any(axis=1),
    "zscore": zscore_flags.any(axis=1),
    "modified_zscore": mad_flags.any(axis=1),
    "percentile": percentile_flags.any(axis=1),
    "isolation_forest": isolation_flags,
    "local_outlier_factor": lof_flags,
    "one_class_svm": svm_flags,
    "elliptic_envelope": elliptic_flags,
}, index=X_test.index)
flag_counts = outlier_summary.sum().sort_values(ascending=False)
```

For DBSCAN, use `dbscan_train_flags` only for the data used to fit it; standard DBSCAN does not provide a general `predict()` for unseen rows. For deployment-time novelty detection, use a detector with a supported prediction API, such as Isolation Forest, LOF with `novelty=True`, One-Class SVM, or Elliptic Envelope. Scale data for distance/kernel methods, and use Elliptic Envelope only when its distribution assumptions are reasonable.

In production, keep scaling, thresholds, and learned detectors in a versioned pipeline, validate numeric input and missing-value handling, and monitor flag rates. Tune thresholds and `contamination` on validation data; do not copy example values blindly. Treat flags as review signals, and do not silently drop production records.

For model workflows, put fitted preprocessing and the estimator in `sklearn.pipeline.Pipeline()` (and use `sklearn.compose.ColumnTransformer()` for different column types). Fit the pipeline on training data only, then use it to transform/evaluate held-out data; this helps prevent data leakage.

The learning notebooks and their method inventory are in [`EDA/`](./EDA/) and [`EDA/eda.md`](./EDA/eda.md). Choose methods based on the data and model rather than applying every method to every dataset.

To run the shared API example, train the linear model first, then build from the repository root with `docker build -t ml-regression -f Dockerfile .` and run `docker run --rm -p 8000:8000 ml-regression`. The Kubernetes manifest is a starting point for this image; provide production secrets, ingress, resource limits, and a trusted artifact workflow before use.

## Production caveat
These are robust learning templates, not a claim that a model is production-ready. Add organization-specific authentication, authorization, privacy, monitoring/alerts, artifact registry, data governance, CI/CD approvals, infrastructure, and rollback testing before deployment. Model files must only be loaded from trusted sources.
