# EDA to Machine Learning Methods

A learning resource with a Jupyter notebook and Python examples covering common exploratory data analysis (EDA), preprocessing, feature engineering, model training, evaluation, and deployment methods.

## Files

| File | Description |
|---|---|
| [`EDA_to_Machine_Learning_Python_Methods.ipynb`](./EDA_to_Machine_Learning_Python_Methods.ipynb) | Main learning notebook, organized into topic-based cells with examples and explanations. |
| [`ml_methods_examples.py`](./ml_methods_examples.py) | Supplemental Python examples grouped into reusable functions. Some functions expect columns such as `age`, `city`, `size`, or `target`; adapt the example data to your dataset. |
| [`EDA_to_Machine_Learning_Python_Methods (1).txt`](./EDA_to_Machine_Learning_Python_Methods%20(1).txt) | Original methods cheat sheet. |

## Open the notebook

From the repository root, create and activate a virtual environment, then install the core packages:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install pandas numpy matplotlib seaborn scipy scikit-learn joblib jupyter
```

Start Jupyter:

```powershell
jupyter lab
```

Open `EDA_MLcode_ALL/EDA_to_Machine_Learning_Python_Methods.ipynb` and run cells in order. In VS Code, open the notebook directly and select a Python kernel with the packages installed.

### Optional packages

Install optional packages only for the corresponding examples:

```powershell
python -m pip install imbalanced-learn category-encoders optuna scikit-optimize
```

Other methods in the cheat sheet may require additional libraries, for example `xgboost`, `lightgbm`, `catboost`, `umap-learn`, `hdbscan`, `mlflow`, or `fastapi`. Install the relevant package only when you reach that method.

## Notebook topics

The notebook provides examples for:

1. Installing and importing common packages
2. Reading CSV, Excel, JSON, and Parquet data
3. Selecting numerical and categorical columns
4. Inspecting data shape, types, summaries, values, duplicates, and memory use
5. Finding and imputing missing values with pandas, `SimpleImputer`, and `KNNImputer`
6. Cleaning values, types, dates, and duplicates
7. Analyzing categorical columns and frequencies
8. Analyzing numerical distributions and summary statistics
9. Detecting and treating outliers
10. Exploring bivariate and multivariate relationships
11. Encoding categories with `map`, `replace`, category codes, `factorize`, count/frequency encoding, one-hot/dummy, ordinal, and label encoding
12. Optional categorical encoders, including binary, hashing, target, WOE, leave-one-out, and CatBoost encoders
13. Scaling and transforming features
14. Feature engineering, including polynomial and date features
15. Feature selection and PCA
16. Train/test splitting and cross-validation
17. Regression and classification algorithms
18. Clustering and unsupervised learning
19. Ensemble methods
20. Hyperparameter tuning with grid search, randomized search, successive halving, and optional Optuna/Bayesian search
21. Model metrics and evaluation
22. Handling imbalanced data
23. Saving and loading models
24. An end-to-end scikit-learn preprocessing and modeling pipeline

## Recommended learning workflow

1. Run one topic at a time and inspect both the code and its output.
2. Start with the small example data included in each cell.
3. Replace example data with your own only after understanding the expected columns and data types.
4. For a real prediction task, identify the target and choose metrics that match the goal.
5. Split data before fitting preprocessing, feature selection, or encoders. Put learned preprocessing inside a scikit-learn `Pipeline` or `ColumnTransformer`.
6. Use cross-validation on training data for model selection and hyperparameter tuning. Keep the final test set untouched until the final evaluation.
7. For time-dependent or grouped records, use time-aware or group-aware splits instead of a random split.
8. Treat outliers as investigation flags; do not remove observations automatically.

## Important notes

- The notebook cells demonstrate methods independently. They are not intended to be run as one uninterrupted production workflow.
- The file-reading cell uses placeholder file names. Replace them with paths to files that exist on your machine before running it.
- The optional advanced encoder cell intentionally raises an installation message if `category-encoders` is unavailable.
- Target-based encoders use the target values. Fit them within training folds using a leakage-safe workflow; never encode the full dataset before splitting.
- `map()` assigns missing values to categories absent from its mapping, while `replace()` leaves unmatched values unchanged.
- Integer codes for nominal categories do not represent a meaningful order. Prefer one-hot encoding for nominal features unless the model or encoding strategy justifies another method.
- API availability and arguments can differ by package version. Check the installed library documentation if an example is not supported in your environment.
