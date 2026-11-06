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
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# 1. Domain/business rule: set limits from valid subject-matter requirements.
domain_flags = (df["age"] < 0) | (df["age"] > 120)

# 2. IQR: univariate; robust to some extreme values.
q1 = df[numeric_columns].quantile(0.25)
q3 = df[numeric_columns].quantile(0.75)
iqr = q3 - q1
iqr_flags = (df[numeric_columns] < q1 - 1.5 * iqr) | (
    df[numeric_columns] > q3 + 1.5 * iqr
)

# 3. Z-score: use mainly for roughly symmetric distributions; threshold is a choice.
z_scores = df[numeric_columns].apply(
    lambda column: zscore(column, nan_policy="omit")
)
zscore_flags = z_scores.abs() > 3

# 4. Modified Z-score: median/MAD is less affected by extreme values.
def modified_zscore_flags(column):
    median = column.median()
    mad = median_abs_deviation(column.dropna(), scale=1.0)
    if mad == 0:
        return pd.Series(False, index=column.index)
    return (0.6745 * (column - median) / mad).abs() > 3.5

mad_flags = df[numeric_columns].apply(modified_zscore_flags)

# 5. Percentile bounds: cap rather than remove; derive bounds from training data.
lower = X_train[numeric_columns].quantile(0.01)
upper = X_train[numeric_columns].quantile(0.99)
X_train_capped = X_train.copy()
X_test_capped = X_test.copy()
X_train_capped[numeric_columns] = X_train[numeric_columns].clip(lower, upper, axis=1)
X_test_capped[numeric_columns] = X_test[numeric_columns].clip(lower, upper, axis=1)

# 6. Isolation Forest: multivariate; contamination must be chosen for the use case.
isolation_forest = IsolationForest(contamination=0.02, random_state=42)
isolation_forest.fit(X_train[numeric_columns])
isolation_flags = isolation_forest.predict(X_test[numeric_columns]) == -1

# 7. Local Outlier Factor: novelty=True supports scoring unseen test/production rows.
lof = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.02)
lof.fit(X_train[numeric_columns])
lof_flags = lof.predict(X_test[numeric_columns]) == -1
```

In production, keep preprocessing and any learned detector in the versioned model pipeline, validate numeric input and missing-value handling, and monitor flag rates. Set `contamination` and thresholds using domain knowledge and validation data rather than copying the example values. Apply capping bounds learned from training data consistently at inference; do not silently drop flagged production records.

For model workflows, put fitted preprocessing and the estimator in `sklearn.pipeline.Pipeline()` (and use `sklearn.compose.ColumnTransformer()` for different column types). Fit the pipeline on training data only, then use it to transform/evaluate held-out data; this helps prevent data leakage.

The learning notebooks and their method inventory are in [`EDA/`](./EDA/) and [`EDA/eda.md`](./EDA/eda.md). Choose methods based on the data and model rather than applying every method to every dataset.

To run the shared API example, train the linear model first, then build from the repository root with `docker build -t ml-regression -f Dockerfile .` and run `docker run --rm -p 8000:8000 ml-regression`. The Kubernetes manifest is a starting point for this image; provide production secrets, ingress, resource limits, and a trusted artifact workflow before use.

## Production caveat
These are robust learning templates, not a claim that a model is production-ready. Add organization-specific authentication, authorization, privacy, monitoring/alerts, artifact registry, data governance, CI/CD approvals, infrastructure, and rollback testing before deployment. Model files must only be loaded from trusted sources.
