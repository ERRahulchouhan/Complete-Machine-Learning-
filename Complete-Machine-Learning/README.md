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
| 1 | Load and combine data | `pandas.read_csv()`, `pandas.read_excel()`, `pandas.concat()`, `pandas.merge()`, `DataFrame.join()` | Load files and combine related tables using the correct keys; validate join cardinality and row counts. | **Yes**, with schema, source, and data-quality validation. |
| 2 | Inspect structure and quality | `DataFrame.head()`, `DataFrame.shape`, `DataFrame.info()`, `DataFrame.describe()`, `DataFrame.dtypes`, `DataFrame.isna().sum()`, `DataFrame.duplicated().sum()`, `Series.nunique()`, `Series.value_counts()` | Check dimensions, types, ranges, missing values, duplicate rows, frequencies, and categorical cardinality. Resolve or document data-quality issues. | **Yes**, as validation and monitoring checks; avoid relying on manual inspection alone. |
| 3 | Analyze individual columns | `Series.describe()`, `Series.hist()`, `seaborn.countplot()`, `seaborn.histplot()`, `seaborn.boxplot()` | Summarize categorical frequencies and numerical distributions; investigate unusual values. | **Mostly offline** for exploration; production monitoring may track approved summary statistics and drift. |
| 4 | Analyze relationships and outliers | `DataFrame.corr()`, `pandas.crosstab()`, `seaborn.scatterplot()`, `seaborn.heatmap()`; domain thresholds, IQR, `scipy.stats.zscore()`, `scipy.stats.median_abs_deviation()`, percentile bounds/`DataFrame.clip()`, `IsolationForest`, `LocalOutlierFactor` | Explore relationships and flag unusual observations. IQR/MAD are robust univariate options; Z-score is sensitive to skew and extreme values; Isolation Forest/LOF can flag multivariate anomalies. Investigate flags and confirm errors before removing or capping values. | **Conditional**: document the method and thresholds; monitor validated numeric rules in production. Fit learned detectors on training data and persist them if inference-time anomaly detection is needed. |
| 5 | Split the data | `sklearn.model_selection.train_test_split()`, `GroupShuffleSplit`, `TimeSeriesSplit` | Split before fitting any data-dependent preprocessing. Use stratification for classification, groups for related observations, or time-based splits for temporal data. | **Training/evaluation only**, not request-time inference; preserve split strategy for reproducible retraining. |
| 6 | Impute missing values | `sklearn.impute.SimpleImputer(strategy="median")`, `SimpleImputer(strategy="mean")`, `SimpleImputer(strategy="most_frequent")`, `SimpleImputer(strategy="constant")`, `sklearn.impute.KNNImputer()`, `DataFrame.fillna()` | Fit imputers on training data. Choose a strategy appropriate to the feature and preserve missingness indicators when useful. `fillna()` is suitable for explicit fixed rules; learned statistics must come from training data. | **Yes, conditionally**: persist the fitted imputer in the inference pipeline. Do not recalculate statistics independently on production batches. |
| 7 | Encode categorical data | `sklearn.preprocessing.OneHotEncoder(handle_unknown="ignore")`, `sklearn.preprocessing.OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)`, `pd.get_dummies()`, `Series.map()`, `Series.replace()`, `sklearn.preprocessing.LabelEncoder()`, `category_encoders.TargetEncoder()` | Use one-hot encoding for nominal input features and ordinal encoding only when category order is real. `Series.map()` or `Series.replace()` works for a fixed, explicit business mapping; handle unmapped/new values. `pd.get_dummies()` is convenient for exploration, but independently encoding train and inference data can produce mismatched columns. `LabelEncoder` is for target labels (`y`), not nominal input features. Target encoding needs leakage-safe cross-fitting and smoothing. | **Yes, preferably** with fitted encoders inside `Pipeline`/`ColumnTransformer`, persisted with the model, and tested for unknown categories. Fixed `map`/`replace` mappings can be production-safe if versioned and have an explicit unknown fallback. `pd.get_dummies()` is **not preferred alone** for production inference; use a persisted, schema-consistent encoder. Use `LabelEncoder` only for target labels. Use target encoding only with leakage controls. |
| 8 | Transform numerical data | `sklearn.preprocessing.PowerTransformer()`, `numpy.log1p()`, `numpy.sqrt()`, `FunctionTransformer()` | Reduce skew when appropriate. Ensure the transform supports the feature values and is invertible/interpretable when required. | **Yes, conditionally**: persist the fitted transformer or exact deterministic function in the pipeline. |
| 9 | Scale numerical data | `sklearn.preprocessing.StandardScaler()`, `sklearn.preprocessing.MinMaxScaler()`, `sklearn.preprocessing.RobustScaler()` | Scale when needed by the estimator (for example, distance-based models, SVMs, or regularized linear models); tree-based models usually do not require scaling. | **Yes, conditionally**: fit on training data and persist for inference; do not refit on each production request or batch. |
| 10 | Select features and validate | `sklearn.feature_selection.VarianceThreshold()`, `sklearn.feature_selection.SelectKBest()`, `sklearn.feature_selection.mutual_info_classif()`, `statsmodels.stats.outliers_influence.variance_inflation_factor()` | Fit feature selection on training data only, use a target-appropriate score, and assess choices with validation data. VIF is a diagnostic, not a general-purpose selector. | **Yes, conditionally**: persist fitted selectors with the model and validate performance and feature schema after retraining. |

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
