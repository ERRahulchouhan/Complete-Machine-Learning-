# EDA to Machine Learning Methods

A guided collection of Jupyter notebook examples for exploring data, preparing features, training models, tuning hyperparameters, evaluating predictions, and saving a model. The notebook is intended for learning and experimentation; it is not a single production workflow.

## Files

| File | Purpose |
|---|---|
| [`EDA_to_Machine_Learning_Python_Methods.ipynb`](./EDA_to_Machine_Learning_Python_Methods.ipynb) | Main, cell-by-cell learning notebook. |
| [`ml_methods_examples.py`](./ml_methods_examples.py) | Supplemental examples organized as Python functions. Adapt the expected column names and data to your project before calling a function. |
| [`EDA_to_Machine_Learning_Python_Methods (1).txt`](./EDA_to_Machine_Learning_Python_Methods%20(1).txt) | Original methods cheat sheet. |

## Setup and run

From the repository root, create a virtual environment and install the core notebook dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install pandas numpy matplotlib seaborn scipy scikit-learn joblib jupyter
```

Start JupyterLab:

```powershell
jupyter lab
```

Open `EDA_MLcode_ALL/EDA_to_Machine_Learning_Python_Methods.ipynb` and run cells in order. In VS Code, open the notebook and select a Python kernel that has the required packages installed.

### Optional packages

Install optional packages only when you want to run the corresponding examples:

```powershell
python -m pip install imbalanced-learn category-encoders optuna scikit-optimize
```

The original cheat sheet also mentions methods from packages such as XGBoost, LightGBM, CatBoost, UMAP, HDBSCAN, MLflow, and FastAPI. Those are not required for the notebook examples documented below; install their packages separately if you add those methods to your study.

## Notebook contents: methods and APIs

This index reflects the methods actually demonstrated or explicitly shown in the notebook. Items marked **optional** require the corresponding package or are provided as a template rather than a default cell to run.

| # | Topic | Methods demonstrated |
|---:|---|---|
| 1 | Package setup | `pip install` command example; imports for `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy.stats`, and `scipy.stats.zscore`. |
| 2 | Read data | `pandas.read_csv()`, `pandas.read_excel()`, `pandas.read_json()`, `pandas.read_parquet()`. File names are placeholders. |
| 3 | Select columns by type | `DataFrame.select_dtypes()` with `numpy.number`, `object`, `category`, and `bool`. |
| 4 | Inspect a dataset | `DataFrame.shape`, `DataFrame.size`, `len()`, `head()`, `tail()`, `sample()`, `info()`, `describe()`, `dtypes`, `nunique()`, `value_counts()`, `duplicated()`, `memory_usage(deep=True)`. |
| 5 | Missing data | Detect with `isnull()`, `isna()`, and missing percentage; fill with `fillna(mean/median/mode)`, `ffill()`, and `bfill()`; use scikit-learn `SimpleImputer(strategy="median")` and `KNNImputer(n_neighbors=2)`. |
| 6 | Clean data | `drop_duplicates()`, string `str.strip()` and `str.lower()`, `pandas.to_numeric(errors="coerce")`, `pandas.to_datetime(errors="coerce")`, `rename()`. |
| 7 | Categorical EDA | `value_counts()`, `value_counts(normalize=True)`, `pandas.crosstab()`, `groupby().size()`, `groupby()[column].mean()`, Seaborn `countplot()`. |
| 8 | Numerical EDA and plots | `mean()`, `median()`, `mode()`, `min()`, `max()`, `std()`, `skew()`, `kurt()`, `dropna()`; Seaborn `histplot(kde=True)` and `boxplot()`; SciPy `stats.probplot(..., dist="norm")`. |
| 9 | Outliers | IQR limits from `quantile(0.25)` and `quantile(0.75)`, outlier row filtering, absolute Z-scores via `scipy.stats.zscore()`, and capping with `clip()`. |
| 10 | Relationships | Pearson correlation with `corr()`, numeric correlation heatmap with Seaborn `heatmap()`, `scatterplot()`, `boxplot()`, and categorical cross-tabulation with `crosstab()`. |
| 11 | Category-to-number encoding | Manual dictionaries with `Series.map()` and `Series.replace()`; `astype("category").cat.codes`; `pandas.factorize()`; count and frequency encoding using `value_counts()` plus `map()`; `LabelEncoder`; `OneHotEncoder(handle_unknown="ignore")`; `OrdinalEncoder` with explicit low/medium/high order; `pandas.get_dummies(drop_first=True)`. |
| 12 | Additional categorical encoders | **Optional `category-encoders`:** `BinaryEncoder`, `HashingEncoder`, `CountEncoder`, `TargetEncoder`, `WOEEncoder`, `LeaveOneOutEncoder`, and `CatBoostEncoder`. The notebook includes code to construct/fit these encoders and a note/template for scikit-learn `TargetEncoder` cross-fitting. |
| 13 | Scaling and power transform | `StandardScaler`, `MinMaxScaler`, `RobustScaler`, and `PowerTransformer(method="yeo-johnson")`, each demonstrated with `fit_transform()`. |
| 14 | Feature transformation and engineering | `numpy.log1p()`, `numpy.square()`, `pandas.to_datetime()`, datetime `.dt.year`, `.dt.month`, `.dt.dayofweek`, and `PolynomialFeatures(degree=2)`. |
| 15 | Feature selection and dimensionality reduction | `SelectKBest(score_func=mutual_info_classif, k=2)`, `RFE(estimator=LogisticRegression(...))`, and `PCA(n_components=2)` with `fit_transform()`. |
| 16 | Holdout split and cross-validation | `train_test_split(test_size=..., stratify=..., random_state=...)`, `KFold`, `StratifiedKFold`, `TimeSeriesSplit`, and `cross_val_score()`. |
| 17 | Regression estimators | The fit/predict loop demonstrates `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`, `DecisionTreeRegressor`, `RandomForestRegressor`, `KNeighborsRegressor`, and `SVR`. The cell also imports `BayesianRidge`, `HuberRegressor`, `AdaBoostRegressor`, `GradientBoostingRegressor`, and `HistGradientBoostingRegressor` as additional estimator examples. |
| 18 | Classification estimators | The fit/predict loop demonstrates `LogisticRegression`, `KNeighborsClassifier`, `GaussianNB`, `DecisionTreeClassifier`, `RandomForestClassifier`, `SVC`, `LinearSVC`, `LinearDiscriminantAnalysis`, and `QuadraticDiscriminantAnalysis`. The cell also imports `AdaBoostClassifier`, `GradientBoostingClassifier`, and `HistGradientBoostingClassifier` as additional estimator examples. |
| 19 | Clustering | `KMeans(n_clusters=2).fit_predict()`, `AgglomerativeClustering(n_clusters=2).fit_predict()`, `DBSCAN(eps=..., min_samples=...).fit_predict()`, and `silhouette_score()`. |
| 20 | Ensembles | `BaggingClassifier` with a random-forest estimator and `VotingClassifier` combining random forest and logistic regression; both use `fit()` and `predict()`. |
| 21 | Hyperparameter tuning | `GridSearchCV`, `RandomizedSearchCV` with `scipy.stats.randint`, `StratifiedKFold`, and experimental `HalvingGridSearchCV` / `HalvingRandomSearchCV`; inspect `best_params_`, `best_score_`, and holdout `score()`. The notebook also includes **optional templates** for Optuna (`create_study`, `suggest_int`, `study.optimize`) and scikit-optimize `BayesSearchCV` / `Integer`. |
| 22 | Classification metrics and curves | `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`, `classification_report`, `roc_auc_score`, `roc_curve`, and Matplotlib plotting. |
| 23 | Imbalanced classification | **Optional `imbalanced-learn`:** `SMOTE(random_state=42).fit_resample()` and before/after class counts using `numpy.bincount()`. |
| 24 | Save and reload a model | `joblib.dump()` / `joblib.load()` and `pickle.dump()`; the example predicts with the reloaded model. |
| 25 | End-to-end classification pipeline | `ColumnTransformer`, `Pipeline`, `SimpleImputer(strategy="median")` for numeric columns, `StandardScaler`, `SimpleImputer(strategy="most_frequent")` and `OneHotEncoder(handle_unknown="ignore")` for categorical columns, `RandomForestClassifier`, stratified `train_test_split()`, `fit()`, `predict()`, and `classification_report()`. |

### Methods not implemented as notebook examples

The source cheat sheet is broader than the current notebook. For example, the notebook does not currently implement regression metrics, statistical hypothesis tests, date lag/rolling features, text vectorizers, anomaly-detection estimators, dimensionality-reduction methods beyond PCA, a deployment API, or live drift monitoring. The package names and methods in the original cheat sheet may be useful for later study, but they are not included in the method index above as runnable notebook demonstrations.

## Recommended learning workflow

1. Run one topic at a time and inspect the code and output.
2. Begin with the small example data included in each cell. The file-reading examples use placeholder paths; change them to files that exist before running.
3. Identify the prediction target and choose metrics that match the goal.
4. Split data before fitting learned preprocessing, feature selection, or encoders. Prefer a `Pipeline` / `ColumnTransformer` for a real modeling workflow.
5. Tune with cross-validation on training data, then evaluate the selected estimator once on the held-out test data.
6. Use time-aware or group-aware splits when observations are time-dependent or related by entity.
7. Investigate outliers before removing or capping them; unusual values may be valid observations.

## Important notes

- The notebook cells are topic demonstrations, not one uninterrupted production workflow.
- `map()` returns missing values for categories absent from its mapping; `replace()` leaves unmatched values unchanged.
- Integer codes do not imply a meaningful order for nominal categories. Prefer one-hot encoding for nominal input features unless another encoding is deliberately appropriate.
- `LabelEncoder` is generally intended for target labels. For ordinal input features, set the order explicitly with `OrdinalEncoder`.
- Target-based encoders use the target values. Use cross-fitting / out-of-fold encoding within training data to prevent target leakage; never fit them on the full dataset before splitting.
- Fit learned imputers, scalers, selectors, and encoders on training data only, then apply the fitted transformations to validation and test data.
- Successive-halving classes are imported through scikit-learn's experimental enable module. If an API is unavailable in your installed version, check the documentation for that version.
- The optional advanced encoder cell requires `category-encoders`; the SMOTE cell requires `imbalanced-learn`.
