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

| Order | Stage | Method names | Use |
|---:|---|---|---|
| 1 | Load and combine data | `pandas.read_csv()`, `pandas.read_excel()`, `pandas.concat()`, `pandas.merge()`, `DataFrame.join()` | Load files and combine related tables using the correct keys. |
| 2 | Inspect structure and quality | `DataFrame.head()`, `DataFrame.shape`, `DataFrame.info()`, `DataFrame.describe()`, `DataFrame.dtypes`, `DataFrame.isna().sum()`, `DataFrame.duplicated().sum()`, `Series.nunique()` | Check dimensions, types, ranges, missing values, duplicate rows, and categorical cardinality. Resolve data-quality issues before analysis. |
| 3 | Analyze individual columns | `Series.value_counts()`, `Series.describe()`, `Series.hist()`, `seaborn.countplot()`, `seaborn.histplot()`, `seaborn.boxplot()` | Summarize categorical frequencies and numerical distributions; investigate unusual values. |
| 4 | Analyze relationships and outliers | `DataFrame.corr()`, `pandas.crosstab()`, `seaborn.scatterplot()`, `seaborn.heatmap()`, IQR rule (`Q1 - 1.5 × IQR`, `Q3 + 1.5 × IQR`) | Explore feature relationships and flag potential outliers. Confirm outliers are errors before removing or capping them. |
| 5 | Split the data | `sklearn.model_selection.train_test_split()` | Create training and test sets before fitting any data-dependent preprocessing. Use stratification for classification when appropriate. |
| 6 | Impute missing values | `sklearn.impute.SimpleImputer(strategy="median")`, `sklearn.impute.SimpleImputer(strategy="most_frequent")`, `sklearn.impute.KNNImputer()` | Fit on training data; use median for skewed numerical data, mean for suitable numerical data, and most-frequent for categorical data. |
| 7 | Encode categorical data | `sklearn.preprocessing.OneHotEncoder()`, `pandas.get_dummies()`, `sklearn.preprocessing.OrdinalEncoder()` | One-hot encode nominal features; use ordinal encoding only when categories have a meaningful order. `LabelEncoder` is for target labels, not nominal input features. |
| 8 | Transform numerical data | `sklearn.preprocessing.PowerTransformer()`, `numpy.log1p()`, `numpy.sqrt()` | Reduce skew when appropriate. Choose a transformation that supports the feature's values and interpretability. |
| 9 | Scale numerical data | `sklearn.preprocessing.StandardScaler()`, `sklearn.preprocessing.MinMaxScaler()` | Scale when needed by the estimator (for example, distance-based models, SVMs, or regularized linear models); tree-based models usually do not require scaling. |
| 10 | Select features and validate | `sklearn.feature_selection.VarianceThreshold()`, `sklearn.feature_selection.SelectKBest()`, `sklearn.feature_selection.mutual_info_classif()`, `statsmodels.stats.outliers_influence.variance_inflation_factor()` | Fit feature selection on training data only, then evaluate the selected features using validation/test data. Use a target-appropriate score function. |

For model workflows, put fitted preprocessing and the estimator in `sklearn.pipeline.Pipeline()` (and use `sklearn.compose.ColumnTransformer()` for different column types). Fit the pipeline on training data only, then use it to transform/evaluate held-out data; this helps prevent data leakage.

The learning notebooks and their method inventory are in [`EDA/`](./EDA/) and [`EDA/eda.md`](./EDA/eda.md). Choose methods based on the data and model rather than applying every method to every dataset.

To run the shared API example, train the linear model first, then build from the repository root with `docker build -t ml-regression -f Dockerfile .` and run `docker run --rm -p 8000:8000 ml-regression`. The Kubernetes manifest is a starting point for this image; provide production secrets, ingress, resource limits, and a trusted artifact workflow before use.

## Production caveat
These are robust learning templates, not a claim that a model is production-ready. Add organization-specific authentication, authorization, privacy, monitoring/alerts, artifact registry, data governance, CI/CD approvals, infrastructure, and rollback testing before deployment. Model files must only be loaded from trusted sources.
