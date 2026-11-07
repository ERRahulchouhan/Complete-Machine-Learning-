# EDA notebook method inventory

Distinct callable names found in the Python code cells of the notebooks are listed below. Receiver prefixes (such as `df.` and `np.`) are omitted. Constructors, library functions, and built-ins are included. Prose and cells that are not valid Python are excluded.

## Contents

- [EDA notebooks](#eda-notebooks)
  - [Sessions 1-3: DataFrame basics and categorical analysis](#eda-sessions-1-3)
  - [Sessions 6-11: Analysis and feature preparation](#eda-sessions-6-11)
  - [Feature selection](#feature-selection)
- [EDA 2024 notebooks](#eda-2024-notebooks)
  - [Sessions 1-5: Checks and univariate analysis](#eda-2024-sessions-1-5)
  - [Sessions 6-10: Analysis and data preparation](#eda-2024-sessions-6-10)

## EDA notebooks

### EDA sessions 1-3

#### Session 1 — DataFrame operations

Notebook: [Session_1_DataFrame_operations.ipynb](./EDA/Session_1_DataFrame_operations.ipynb)

**Names:** `DataFrame`, `drop`, `isnull`, `len`, `list`, `range`, `read_csv`, `read_excel`, `rename`, `to_csv`, `to_excel`, `type`, `zip`

#### Session 2 — Data quick checks

Notebook: [Session_2_Data_Quick_checks.ipynb](./EDA/Session_2_Data_Quick_checks.ipynb)

**Names:** `append`, `copy`, `DataFrame`, `drop_duplicates`, `index`, `isnull`, `items`, `keys`, `list`, `print`, `read_csv`, `rename`, `reset_index`, `select_dtypes`, `shape`, `sum`, `to_dict`, `to_list`, `type`, `values`

#### Session 3 — Categorical column analysis

Notebooks: [Session_3_Categorical_Column_Analysis.ipynb](./EDA/Session_3_Categorical_Column_Analysis.ipynb) and [Session_3_Categorical_Column_Analysis (1).ipynb](<./EDA/Session_3_Categorical_Column_Analysis (1).ipynb>)

**Names in both notebooks:** `append`, `bar`, `DataFrame`, `figure`, `index`, `items`, `keys`, `len`, `nunique`, `print`, `read_csv`, `savefig`, `select_dtypes`, `show`, `title`, `to_csv`, `to_dict`, `to_list`, `unique`, `value_counts`, `xlabel`, `ylabel`, `zip`

### EDA sessions 6-11

#### Session 6 — Bivariate and multivariate analysis

Notebook: [Session_6_Bivariate_Multi_Variate_analysis.ipynb](./EDA/Session_6_Bivariate_Multi_Variate_analysis.ipynb)

**Names:** `append`, `corr`, `countplot`, `crosstab`, `DataFrame`, `figure`, `heatmap`, `len`, `list`, `plot`, `range`, `read_csv`, `scatter`, `select_dtypes`, `sorted`, `subplot`, `title`, `unique`, `value_counts`, `zip`

#### Session 7 — Encoding

Notebook: [Session_7_Encoding.ipynb](./EDA/Session_7_Encoding.ipynb)

**Names:** `fit_transform`, `LabelEncoder`, `len`, `map`, `nunique`, `print`, `range`, `read_csv`, `select_dtypes`, `unique`

#### Session 8 — Scaling

Notebook: [Session_8_Scaling.ipynb](./EDA/Session_8_Scaling.ipynb)

**Names:** `fit_transform`, `hist`, `mean`, `read_csv`, `select_dtypes`, `show`, `StandardScaler`, `std`, `subplot`

#### Session 9 — Missing value analysis

Notebook: [Session_9_Missing_ValueAnalaysis.ipynb](./EDA/Session_9_Missing_ValueAnalaysis.ipynb)

**Names:** `DataFrame`, `dir`, `dropna`, `fillna`, `filterwarnings`, `fit_transform`, `isnull`, `KNNImputer`, `mean`, `mode`, `print`, `sum`, `to_csv`

#### Session 10 — Transformation methods

Notebook: [Session_10_Transformation_methods.ipynb](./EDA/Session_10_Transformation_methods.ipynb)

**Names:** `distplot`, `exponential`, `figure`, `filterwarnings`, `fit_transform`, `hist`, `legend`, `log`, `normal`, `PowerTransformer`, `read_csv`, `reshape`, `select_dtypes`, `show`, `sqrt`, `subplot`

#### Session 11 — Joins

Notebook: [Session_11_Joins.ipynb](./EDA/Session_11_Joins.ipynb)

**Names:** `conact`, `DataFrame`, `join`, `merge`

### Feature selection

Notebook: [Feature_selection_EndtoEnd.ipynb](./EDA/Feature_selection_EndtoEnd.ipynb)

**Names:** `add_constant`, `DataFrame`, `dir`, `drop`, `dropna`, `extend`, `filterwarnings`, `fit`, `get_feature_names_out`, `get_params`, `get_support`, `head`, `len`, `list`, `mutual_info_classif`, `normal`, `OLS`, `plot`, `print`, `randn`, `range`, `read_csv`, `reset_index`, `SelectKBest`, `Series`, `sort_values`, `summary`, `train_test_split`, `type`, `var`, `variance_inflation_factor`, `VarianceThreshold`

## EDA 2024 notebooks

### EDA 2024 sessions 1-5

#### Session 1

Notebook: [EDA_Session-1.ipynb](<./EDA 2024/EDA_Session-1.ipynb>)

**Names:** `append`, `DataFrame`, `dir`, `drop`, `getcwd`, `len`, `randint`, `range`, `read_csv`, `to_csv`, `zip`

#### Session 2 — Basic data checks

Notebook: [EDA_Session-2-Basic_Data_Checks.ipynb](<./EDA 2024/EDA_Session-2-Basic_Data_Checks.ipynb>)

**Names:** `dict`, `head`, `info`, `isna`, `len`, `print`, `read_csv`, `select_dtypes`, `sum`, `tail`, `take`, `to_list`, `type`

#### Session 3 — Univariate categorical data analysis

Notebook: [EDA_Session-3_Univariate-Categorical_data_analysis.ipynb](<./EDA 2024/EDA_Session-3_Univariate-Categorical_data_analysis.ipynb>)

**Names:** `array`, `bar`, `bar_label`, `countplot`, `DataFrame`, `dict`, `figure`, `head`, `keys`, `len`, `list`, `nunique`, `pie`, `plot`, `read_csv`, `savefig`, `show`, `title`, `to_csv`, `to_list`, `type`, `unique`, `value_counts`, `values`, `xlabel`, `ylabel`, `zip`

#### Session 4 — Univariate numerical data analysis

Notebook: [EDA_Session-4_Univariate-Numerical_data_analysis.ipynb](<./EDA 2024/EDA_Session-4_Univariate-Numerical_data_analysis.ipynb>)

**Names:** `bar`, `count`, `DataFrame`, `describe`, `displot`, `head`, `hist`, `len`, `max`, `mean`, `median`, `min`, `percentile`, `quantile`, `read_csv`, `round`, `select_dtypes`, `std`, `subplot`, `to_csv`, `zip`

#### Session 5 — Outlier analysis

Notebook: [EDA_Session-5_Outlier analysis.ipynb](<./EDA 2024/EDA_Session-5_Outlier analysis.ipynb>)

**Names:** `boxplot`, `figure`, `format`, `head`, `hist`, `len`, `outliers`, `print`, `quantile`, `read_csv`, `select_dtypes`, `show`, `subplot`

### EDA 2024 sessions 6-10

#### Session 6 — Bivariate analysis

Notebook: [EDA_Session-6_Bivariate Analysis.ipynb](<./EDA 2024/EDA_Session-6_Bivariate Analysis.ipynb>)

**Names:** `append`, `corr`, `crosstab`, `DataFrame`, `figure`, `head`, `heatmap`, `keys`, `len`, `plot`, `print`, `range`, `read_csv`, `scatter`, `set_index`, `unique`, `value_counts`, `zip`

#### Session 7 — Categorical to numerical conversion

Notebook: [EDA_Session-7_Categorical to Numerical Conversion.ipynb](<./EDA 2024/EDA_Session-7_Categorical to Numerical Conversion.ipynb>)

**Names:** `drop`, `fit_transform`, `get_dummies`, `head`, `inverse_transform`, `LabelEncoder`, `len`, `map`, `print`, `range`, `read_csv`, `select_dtypes`, `unique`, `where`

#### Session 8 — Standardization and normalization

Notebook: [EDA_Session-8_Standadiztion and Normalization.ipynb](<./EDA 2024/EDA_Session-8_Standadiztion and Normalization.ipynb>)

**Names:** `boxplot`, `fit_transform`, `head`, `idxmax`, `idxmin`, `max`, `mean`, `min`, `MinMaxScaler`, `print`, `read_csv`, `show`, `StandardScaler`, `std`, `subplot`

#### Session 9 — Data transformation techniques

Notebook: [EDA_Session-9_Data Transformation Techniques.ipynb](<./EDA 2024/EDA_Session-9_Data Transformation Techniques.ipynb>)

**Names:** `displot`, `exponential`, `figure`, `fit`, `hist`, `legend`, `log`, `normal`, `PowerTransformer`, `print`, `reciprocal`, `show`, `sqrt`, `subplot`, `transform`

#### Session 10 — Missing values analysis

Notebook: [EDA_Session_10-Missing values analysis.ipynb](<./EDA 2024/EDA_Session_10-Missing values analysis.ipynb>)

**Names:** `DataFrame`, `fillna`, `isnull`, `mean`, `mode`, `print`, `sum`, `to_csv`
