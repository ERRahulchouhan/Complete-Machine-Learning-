# Master EDA methods: notebook and production reference

This guide combines the method names listed in [eda.md](./eda.md) with additional methods that are commonly useful in reliable data-analysis workflows. It is organized by EDA topic/session so you can find both the method and its purpose.

- **Notebook** means the name appears in one or more EDA notebooks.
- **Recommended** means an additional production-oriented method or API.
- Names are shown with their library prefix when that helps identify the exact API. For example, `DataFrame.drop` is different from `Path.drop` (which is not a standard method).
- Some names in the original notebooks are attributes rather than methods. They are marked **attribute**. Notebook-derived names are retained even when a spelling is incorrect or an API is outdated; see the notes at the end.

## Session 1 — Load data and manage DataFrames

Use this session to load data with explicit types and settings, inspect its shape, and save reproducible outputs.

| Exact name | Origin | Why use it |
|---|---|---|
| `pandas.read_csv` | Notebook | Read delimited text into a DataFrame. Set options such as `dtype`, `usecols`, and `parse_dates` to control the input schema. |
| `pandas.read_excel` | Notebook | Read spreadsheet data when Excel is the source format. Specify the sheet and expected columns. |
| `pandas.DataFrame` | Notebook | Construct a tabular object from dictionaries, arrays, or other structured data. |
| `DataFrame.rename` | Notebook | Rename columns or index labels explicitly; useful for normalizing inconsistent source names. |
| `DataFrame.drop` | Notebook | Remove selected rows or columns when the removal is intentional and documented. |
| `DataFrame.to_csv` | Notebook | Export a DataFrame to CSV. Set `index=False` when the index is not data. |
| `DataFrame.to_excel` | Notebook | Export a DataFrame to an Excel workbook for spreadsheet consumers. |
| `DataFrame.copy` | Notebook | Create a separate DataFrame before a sequence of transformations. |
| `DataFrame.shape` | Notebook — attribute | Check row and column counts without scanning or printing the whole dataset. |
| `DataFrame.index` | Notebook — attribute | Inspect row labels; useful for verifying index alignment. |
| `pandas.Series` | Notebook | Construct a one-dimensional labeled array when working with a single column or series of values. |
| `DataFrame.columns` | Recommended — attribute | Inspect or validate column names. |
| `DataFrame.dtypes` | Recommended — attribute | Check inferred data types and identify unexpected parsing. |
| `DataFrame.memory_usage` | Recommended | Estimate per-column memory use when diagnosing large datasets. |
| `pandas.read_parquet` | Recommended | Read columnar data efficiently while preserving types better than CSV in many workflows. |
| `DataFrame.to_parquet` | Recommended | Persist typed, compressed data for analytical pipelines. |
| `pandas.read_json` | Recommended | Load JSON records or structured JSON when that is the upstream format. |
| `pandas.read_sql` | Recommended | Load a query result from a database rather than exporting an intermediate file. |

## Session 2 — Basic data checks and data quality

Run these checks before analysis so missing values, duplicates, incorrect types, and unexpected dimensions are visible.

| Exact name | Origin | Why use it |
|---|---|---|
| `DataFrame.head` | Notebook | Preview the first rows and check whether parsing looks reasonable. |
| `DataFrame.tail` | Notebook | Inspect the final rows for truncation, malformed records, or ordering issues. |
| `DataFrame.info` | Notebook | Review column types and non-null counts in one compact summary. |
| `DataFrame.isna` | Notebook | Build a Boolean mask of missing values. Prefer this standard spelling for null checks. |
| `DataFrame.isnull` | Notebook | Alias of `isna`; retained because it appears in the notebooks. |
| `DataFrame.notna` | Recommended | Identify present values, for example before validating a required field. |
| `DataFrame.isna().sum` | Notebook | Count missing values by column when called on the resulting DataFrame. |
| `DataFrame.dropna` | Notebook | Remove rows or columns with missing values when dropping them is justified. |
| `DataFrame.fillna` | Notebook | Fill missing values using an explicit, appropriate value or strategy. |
| `DataFrame.duplicated` | Recommended | Identify repeated rows or repeated records using a selected subset of key columns. |
| `DataFrame.drop_duplicates` | Notebook | Remove duplicate records after choosing the correct business key and retention rule. |
| `DataFrame.select_dtypes` | Notebook | Select columns by type, for example numeric columns for a numeric summary. |
| `DataFrame.describe` | Notebook | Get summary statistics for numeric or categorical columns. |
| `DataFrame.nunique` | Recommended | Count distinct values per column to spot constants or unexpectedly high cardinality. |
| `DataFrame.memory_usage` | Recommended | Find columns that dominate memory use, especially object/string columns. |
| `DataFrame.take` | Notebook | Select rows or columns by positional indices; use `iloc` for general label/position slicing. |
| `DataFrame.to_list` | Notebook | Convert a Series to a Python list when an API specifically requires one. |
| `Series.to_dict` | Notebook | Convert a Series to a mapping, such as category-to-count results. |
| `DataFrame.items` | Notebook | Iterate over `(column_name, Series)` pairs. |
| `DataFrame.keys` | Notebook | Access column labels; for clarity, `DataFrame.columns` is usually more explicit. |
| `DataFrame.values` | Notebook — attribute | Access the underlying values; prefer `to_numpy()` when an ndarray is specifically needed. |
| `DataFrame.to_numpy` | Recommended | Convert to a NumPy array at an explicit library boundary. |
| `DataFrame.reset_index` | Notebook | Move the current index back into columns or create a fresh default index. |
| `DataFrame.index` | Notebook — attribute | Inspect row labels and check whether indexes are unique or aligned. |
| `len` | Notebook — built-in | Get the number of rows in a DataFrame. |
| `type` | Notebook — built-in | Check an object's Python type while debugging or validating a value. |
| `print` | Notebook — built-in | Display a quick local diagnostic; use a logger in production code. |

## Session 3 — Univariate categorical analysis

Use frequency tables and plots to understand category distribution, rare levels, and cardinality.

| Exact name | Origin | Why use it |
|---|---|---|
| `Series.value_counts` | Notebook | Rank categories by frequency; set `normalize=True` for proportions and `dropna=False` to include missing values. |
| `Series.unique` | Notebook | List observed category values, particularly for quick inspection. |
| `Series.nunique` | Notebook | Count distinct categories to quantify cardinality. |
| `Series.to_list` | Notebook | Export category labels to a list for display or a downstream API. |
| `dict` | Notebook — built-in | Store a small mapping such as category-to-count values. |
| `dict.keys` | Notebook | Iterate over mapping keys. |
| `dict.values` | Notebook | Iterate over mapping values. |
| `dict.items` | Notebook | Iterate over key-value pairs together. |
| `matplotlib.pyplot.figure` | Notebook | Create a figure before building a plot. |
| `matplotlib.pyplot.bar` | Notebook | Draw a bar chart for category counts. |
| `matplotlib.axes.Axes.bar_label` | Notebook | Label bars with their values to make a chart easier to read. |
| `seaborn.countplot` | Notebook | Plot category counts directly from observations. |
| `matplotlib.pyplot.pie` | Notebook | Show part-to-whole proportions for a small number of categories; avoid it for many categories. |
| `pandas.Series.plot` | Notebook | Create a quick plot from a Series, such as category counts. |
| `matplotlib.pyplot.title` | Notebook | Add a descriptive chart title. |
| `matplotlib.pyplot.xlabel` | Notebook | Label the horizontal axis. |
| `matplotlib.pyplot.ylabel` | Notebook | Label the vertical axis. |
| `matplotlib.pyplot.savefig` | Notebook | Save a chart to a file; set dimensions and resolution for reports. |
| `matplotlib.pyplot.show` | Notebook | Display a chart interactively; generally omit in non-interactive production jobs. |
| `DataFrame.groupby` | Recommended | Aggregate by one or more category columns using explicit aggregation rules. |
| `DataFrame.groupby().size` | Recommended | Count rows in each group, including groups where a key is missing when configured appropriately. |
| `DataFrame.groupby().agg` | Recommended | Compute several named statistics per group in a readable, repeatable way. |
| `DataFrame.pivot_table` | Recommended | Summarize category combinations as a table for reporting or inspection. |
| `DataFrame.value_counts` | Recommended | Count unique row or column combinations when joint category frequencies matter. |

## Session 4 — Univariate numerical analysis

Use summaries and distributions to understand scale, spread, skew, and unusual values.

| Exact name | Origin | Why use it |
|---|---|---|
| `Series.count` | Notebook | Count non-missing observations in a numeric Series. |
| `Series.mean` | Notebook | Measure the arithmetic average; sensitive to extreme values. |
| `Series.median` | Notebook | Measure the midpoint; more robust than the mean for skewed data. |
| `Series.std` | Notebook | Estimate dispersion around the mean. |
| `Series.min` | Notebook | Find the minimum observed value. |
| `Series.max` | Notebook | Find the maximum observed value. |
| `Series.quantile` | Notebook | Compute selected quantiles, such as quartiles or percentiles. |
| `numpy.percentile` | Notebook | Compute percentiles when working with NumPy arrays. |
| `numpy.quantile` | Notebook | Compute quantiles using probabilities from 0 to 1. |
| `numpy.mean` | Notebook | Compute the mean for array-based numerical analysis. |
| `numpy.median` | Notebook | Compute the median for array-based numerical analysis. |
| `numpy.std` | Notebook | Compute standard deviation for array data; choose `ddof` deliberately. |
| `numpy.min` | Notebook | Find an array's minimum. |
| `numpy.max` | Notebook | Find an array's maximum. |
| `builtins.min` | Notebook | Find the smallest value in a Python iterable. |
| `builtins.max` | Notebook | Find the largest value in a Python iterable. |
| `round` | Notebook — built-in | Round a displayed or exported number; avoid rounding intermediate calculations unnecessarily. |
| `seaborn.displot` | Notebook | Plot a distribution, optionally split or faceted by another variable. |
| `matplotlib.pyplot.hist` | Notebook | Plot a numeric histogram with explicit bins to inspect distribution shape. |
| `matplotlib.pyplot.bar` | Notebook | Compare already-aggregated numeric summaries or discrete values. |
| `matplotlib.pyplot.subplot` | Notebook | Arrange multiple plots in one figure for side-by-side comparison. |
| `DataFrame.describe` | Notebook | Generate a useful baseline summary of numeric columns. |
| `Series.skew` | Recommended | Quantify distribution asymmetry to guide transformations and interpretation. |
| `Series.kurt` | Recommended | Inspect tail heaviness as a complement to quantiles and plots. |
| `numpy.isfinite` | Recommended | Detect infinite or invalid numeric values before statistics or model fitting. |

## Session 5 — Outlier analysis

Identify potentially unusual observations, then investigate their cause before removing or changing them.

| Exact name | Origin | Why use it |
|---|---|---|
| `Series.quantile` | Notebook | Calculate IQR fences or percentile thresholds for candidate outliers. |
| `outliers` | Notebook — not a standard method | The notebook calls this name, but pandas has no general `outliers` method. Define an explicit rule, such as IQR bounds, or use a documented domain threshold. |
| `matplotlib.pyplot.boxplot` | Notebook | Visualize median, quartiles, and points outside whiskers. |
| `matplotlib.pyplot.hist` | Notebook | See whether extreme values create a long tail or multiple modes. |
| `seaborn.boxplot` | Recommended | Compare numeric distributions across categories using a consistent plotting API. |
| `DataFrame.clip` | Recommended | Cap values to explicit lower/upper limits when winsorization is justified and recorded. |
| `numpy.percentile` | Notebook | Calculate percentile-based bounds on arrays. |
| `numpy.quantile` | Notebook | Calculate quantile-based bounds on arrays. |
| `len` | Notebook — built-in | Count flagged rows or total observations to report the size of an outlier set. |
| `format` | Notebook — built-in | Format a number for readable reporting; do not use formatted strings for calculations. |

## Session 6 — Bivariate and multivariate analysis

Compare variables and groups, and inspect associations without treating correlation as causation.

| Exact name | Origin | Why use it |
|---|---|---|
| `pandas.crosstab` | Notebook | Create a contingency table for categorical variables; use `normalize` for row/column proportions. |
| `DataFrame.corr` | Notebook | Calculate pairwise numeric correlations; choose a method and handle missing data deliberately. |
| `DataFrame.corrwith` | Recommended | Compare each selected column's correlation with a target Series or DataFrame. |
| `DataFrame.groupby` | Recommended | Compare numeric aggregates across categories. |
| `DataFrame.groupby().agg` | Recommended | Produce multiple summary statistics per group. |
| `seaborn.heatmap` | Notebook | Visualize a correlation matrix or contingency table. |
| `matplotlib.pyplot.scatter` | Notebook | Inspect the relationship between two numeric variables and identify patterns or outliers. |
| `seaborn.scatterplot` | Recommended | Draw a scatter plot with clearer semantic mappings for hue, style, or size. |
| `seaborn.pairplot` | Recommended | Explore pairwise relationships among a manageable number of numeric columns. |
| `seaborn.countplot` | Notebook | Compare category counts. |
| `pandas.Series.plot` | Notebook | Plot an aggregated Series, such as a group summary. |
| `DataFrame.set_index` | Notebook | Set meaningful key columns as the row index before aligned operations. |
| `DataFrame.keys` | Notebook | Inspect column labels; `DataFrame.columns` is often clearer. |
| `Series.unique` | Notebook | Inspect observed values in a variable. |
| `Series.value_counts` | Notebook | Compare the marginal frequency of categorical values. |
| `matplotlib.pyplot.figure` | Notebook | Start a new figure for a plot or group of subplots. |
| `matplotlib.pyplot.subplot` | Notebook | Place related plots into a grid. |
| `matplotlib.pyplot.plot` | Notebook | Draw a line plot, typically for ordered or time-indexed data. |
| `matplotlib.pyplot.title` | Notebook | Add a descriptive chart title. |
| `range` | Notebook — built-in | Generate integer positions for controlled loops; prefer vectorized operations when possible. |
| `sorted` | Notebook — built-in | Return a sorted Python list, for example for stable category ordering. |

## Session 7 — Categorical encoding

Encode categories only after deciding whether they are nominal, ordinal, or target labels. Learn encodings on training data and reuse them for validation and production data.

| Exact name | Origin | Why use it |
|---|---|---|
| `pandas.get_dummies` | Notebook | Create indicator columns for categories in a simple offline workflow. |
| `sklearn.preprocessing.LabelEncoder` | Notebook | Encode target labels; it is generally not the right encoder for nominal feature columns. |
| `LabelEncoder.fit_transform` | Notebook | Learn target-label classes and encode the training labels. |
| `LabelEncoder.inverse_transform` | Notebook | Convert encoded target labels back to their original labels. |
| `Series.map` | Notebook | Apply an explicit category-to-value mapping; define a policy for unknown values. |
| `numpy.where` | Notebook | Create values conditionally for simple binary rules. |
| `DataFrame.drop` | Notebook | Remove a source column after a deliberate encoding or feature-selection step. |
| `sklearn.preprocessing.OneHotEncoder` | Recommended | Encode nominal features in a fitted pipeline; configure unknown-category handling for inference. |
| `sklearn.preprocessing.OrdinalEncoder` | Recommended | Encode ordered categories with an explicit order and a policy for unknown categories. |
| `sklearn.preprocessing.LabelEncoder` | Notebook | Encode a classification target (`y`), not nominal input features (`X`). |
| `sklearn.compose.ColumnTransformer` | Recommended | Apply appropriate encoders and transformations to different column groups. |
| `sklearn.pipeline.Pipeline` | Recommended | Keep preprocessing and modeling together so training and inference use the same fitted steps. |
| `Series.nunique` | Notebook | Check category cardinality before choosing an encoding strategy. |
| `Series.unique` | Notebook | Inspect categories and catch inconsistent labels before encoding. |

## Session 8 — Scaling and normalization

Scale numeric features when the downstream algorithm is scale-sensitive. Fit the scaler on training data only.

| Exact name | Origin | Why use it |
|---|---|---|
| `sklearn.preprocessing.StandardScaler` | Notebook | Center features and scale them to unit variance; common for linear models, SVMs, and distance-based methods. |
| `StandardScaler.fit_transform` | Notebook | Learn scaling parameters and transform training features in one step. |
| `sklearn.preprocessing.MinMaxScaler` | Notebook | Map features to a chosen range, usually `[0, 1]`; sensitive to outliers. |
| `MinMaxScaler.fit_transform` | Notebook | Learn training-set bounds and scale training features. |
| `sklearn.preprocessing.RobustScaler` | Recommended | Scale using robust statistics when numeric features contain outliers. |
| `sklearn.preprocessing.Normalizer` | Recommended | Normalize each row/vector, useful for some text or direction-based models; distinct from scaling each feature. |
| `sklearn.pipeline.Pipeline` | Recommended | Prevent data leakage by fitting the scaler only within each training fold. |
| `Series.mean` | Notebook | Check the mean before/after scaling to validate expected centering. |
| `Series.std` | Notebook | Check standard deviation before/after scaling. |
| `Series.idxmin` | Notebook | Find the index of the smallest value for a quick range check. |
| `Series.idxmax` | Notebook | Find the index of the largest value for a quick range check. |
| `matplotlib.pyplot.boxplot` | Notebook | Compare feature scales and outlier profiles visually. |
| `matplotlib.pyplot.hist` | Notebook | Inspect feature distributions before choosing a scaler. |

## Session 9 — Missing-value analysis and imputation

Measure missingness, understand why values are absent, and fit imputers on training data only.

| Exact name | Origin | Why use it |
|---|---|---|
| `DataFrame.isna` | Notebook | Mark missing values. |
| `DataFrame.isnull` | Notebook | Alias retained from the notebooks; `isna` is the clearer standard spelling. |
| `DataFrame.isna().sum` | Notebook | Count missing cells per column. |
| `DataFrame.dropna` | Notebook | Remove rows/columns only when the missingness and impact make dropping defensible. |
| `DataFrame.fillna` | Notebook | Fill values directly for simple, documented cases. |
| `Series.mean` | Notebook | Calculate a numeric mean used by simple imputation; sensitive to outliers. |
| `Series.mode` | Notebook | Find the most frequent value, often used for categorical imputation. |
| `sklearn.impute.SimpleImputer` | Recommended | Apply a fitted mean, median, most-frequent, or constant strategy consistently. |
| `sklearn.impute.KNNImputer` | Notebook | Impute based on neighboring rows; can be expensive and requires sensible feature scaling. |
| `KNNImputer.fit_transform` | Notebook | Fit neighbor-based imputation and transform training data; use inside a pipeline for validation. |
| `sklearn.compose.ColumnTransformer` | Recommended | Apply different imputers to numeric and categorical columns. |
| `sklearn.pipeline.Pipeline` | Recommended | Avoid leakage by fitting imputation only on training folds. |
| `DataFrame.to_csv` | Notebook | Save a cleaned snapshot when a file deliverable is needed; retain the raw source separately. |
| `warnings.filterwarnings` | Notebook | Control known warning output only when the warning is understood; do not suppress warnings broadly in production. |

## Session 10 — Transformations and distributions

Transformations can reduce skew or improve model fit, but should be selected based on data meaning and validated after fitting.

| Exact name | Origin | Why use it |
|---|---|---|
| `numpy.log` | Notebook | Log-transform positive, right-skewed values; ordinary log is undefined for zero or negative values. |
| `numpy.sqrt` | Notebook | Square-root-transform non-negative values, often counts or moderately skewed measurements. |
| `numpy.reciprocal` | Notebook | Compute reciprocal values when scientifically meaningful; zero values require explicit handling. |
| `numpy.random.normal` | Notebook | Generate normally distributed synthetic values for demonstrations or tests, not for transforming real observations. |
| `numpy.random.exponential` | Notebook | Generate exponential synthetic values for demonstrations or tests. |
| `numpy.random.randn` | Notebook | Generate standard-normal synthetic arrays for examples or tests. |
| `sklearn.preprocessing.PowerTransformer` | Notebook | Estimate a power transformation to reduce skew; consider positivity and the chosen method. |
| `PowerTransformer.fit` | Notebook | Learn transformation parameters from training data. |
| `PowerTransformer.transform` | Notebook | Apply the already-fitted transformation to validation or production data. |
| `PowerTransformer.fit_transform` | Notebook | Fit and transform training data; use within a pipeline to avoid leakage. |
| `numpy.ndarray.reshape` | Notebook | Change an array's shape when an estimator expects a particular dimensionality. |
| `seaborn.displot` | Notebook | Visualize distributions, including comparisons across groups. |
| `seaborn.histplot` | Recommended | Plot histograms with explicit bins and optional density estimates. |
| `seaborn.distplot` | Notebook — deprecated | Appears in the older notebook; use `seaborn.histplot` or `seaborn.displot` instead. |
| `matplotlib.pyplot.hist` | Notebook | Inspect distribution shape and transformation effects. |
| `matplotlib.pyplot.legend` | Notebook | Identify plotted series when multiple distributions share a chart. |
| `matplotlib.pyplot.figure` | Notebook | Create a figure for distribution plots. |
| `matplotlib.pyplot.subplot` | Notebook | Compare original and transformed distributions in one figure. |
| `matplotlib.pyplot.show` | Notebook | Display plots interactively; not generally needed in scheduled jobs. |

## Session 11 — Joining and combining data

Validate join keys and expected row counts so joins do not silently duplicate or discard records.

| Exact name | Origin | Why use it |
|---|---|---|
| `pandas.merge` | Notebook | Join DataFrames on explicit keys; specify `how` and use `validate` to assert expected cardinality. |
| `DataFrame.join` | Notebook | Join primarily by index; use when index-based alignment is intentional. |
| `pandas.concat` | Recommended | Stack DataFrames along rows or columns when their schemas and meanings align. |
| `DataFrame.merge` | Recommended | Method form of merge; make keys and join type explicit. |
| `DataFrame.set_index` | Notebook | Set validated keys as an index before index-based joining. |
| `DataFrame.reset_index` | Notebook | Restore index values to columns after index-based operations. |
| `DataFrame.duplicated` | Recommended | Check whether join keys are unique before a one-to-one or many-to-one merge. |
| `DataFrame.merge(validate=...)` | Recommended | Fail fast if actual join cardinality violates the expected relationship. |
| `DataFrame.merge(indicator=True)` | Recommended | Identify matched and unmatched records during join diagnostics. |
| `conact` | Notebook — likely typo | No standard pandas API has this name; for combining DataFrames, use `pandas.concat`. |

## Feature selection — supplementary notebook

Use feature selection after splitting data and within a cross-validation pipeline. Fit selectors on training data only.

| Exact name | Origin | Why use it |
|---|---|---|
| `sklearn.feature_selection.VarianceThreshold` | Notebook | Remove features with near-zero variance when they contain little useful variation. |
| `VarianceThreshold.fit` | Notebook | Learn which columns satisfy the variance threshold on training data. |
| `VarianceThreshold.get_support` | Notebook | Retrieve a Boolean mask or indices of retained features. |
| `VarianceThreshold.get_feature_names_out` | Notebook | Get the names of retained features after selection. |
| `VarianceThreshold.get_params` | Notebook | Inspect estimator configuration, such as the variance threshold. |
| `sklearn.feature_selection.SelectKBest` | Notebook | Select a fixed number of features using a scoring function. |
| `SelectKBest.fit` | Notebook | Fit scores and selection on training data. |
| `SelectKBest.get_support` | Notebook | Identify which features were selected. |
| `SelectKBest.get_feature_names_out` | Notebook | Report selected feature names. |
| `SelectKBest.get_params` | Notebook | Inspect selector settings, such as `k` and the scoring function. |
| `sklearn.feature_selection.mutual_info_classif` | Notebook | Score feature dependence for a classification target, including nonlinear relationships. |
| `sklearn.model_selection.train_test_split` | Notebook | Separate evaluation data before fitting selection to avoid test leakage. |
| `statsmodels.api.OLS` | Notebook | Fit ordinary least-squares regression for statistical analysis when assumptions are appropriate. |
| `statsmodels.api.add_constant` | Notebook | Add an intercept column for statsmodels regression when the model does not add it automatically. |
| `statsmodels.OLS.fit` | Recommended | Estimate an OLS model on the training data. |
| `RegressionResults.summary` | Notebook | Review model estimates, uncertainty, and diagnostic statistics. |
| `statsmodels.stats.outliers_influence.variance_inflation_factor` | Notebook | Assess multicollinearity among numeric predictors. |
| `sklearn.pipeline.Pipeline` | Recommended | Keep selection and model fitting together inside cross-validation. |
| `sklearn.model_selection.cross_validate` | Recommended | Evaluate the complete selection/model pipeline across folds. |
| `DataFrame.sort_values` | Notebook | Sort feature scores or diagnostic results for review. |
| `DataFrame.drop` | Notebook | Remove selected or unsuitable columns after documenting the reason. |
| `Series.var` | Notebook | Measure variance for an individual feature or diagnostic. |
| `DataFrame.head` | Notebook | Preview selected features or diagnostic output. |
| `DataFrame.reset_index` | Notebook | Restore a simple index after filtering or selection. |
| `list` | Notebook — built-in | Build a Python list of selected names when an API requires one. |
| `range` | Notebook — built-in | Generate integer positions; vectorized selection is generally simpler. |
| `zip` | Notebook — built-in | Pair corresponding names and values for compact iteration. |
| `extend` | Notebook — list method | Add multiple values to a Python list; prefer clear list construction where practical. |

## Shared Python, NumPy, plotting, and inspection names

These notebook names also appear in examples or supporting code. Use them where appropriate, but prefer vectorized operations and logging in reusable code.

| Exact name | Origin | Why use it |
|---|---|---|
| `list.append` | Notebook | Add one item to a Python list; this is a list method, not a pandas DataFrame method. |
| `list.extend` | Notebook | Add all items from an iterable to a Python list. |
| `len` | Notebook — built-in | Count items or rows. |
| `list` | Notebook — built-in | Construct a list from an iterable. |
| `dict` | Notebook — built-in | Construct a key-value mapping. |
| `type` | Notebook — built-in | Inspect an object's type. |
| `print` | Notebook — built-in | Display a quick diagnostic; use structured logging in production. |
| `range` | Notebook — built-in | Generate integer sequences for loops. |
| `zip` | Notebook — built-in | Iterate over multiple iterables in parallel. |
| `sorted` | Notebook — built-in | Return a sorted copy of an iterable. |
| `dir` | Notebook — built-in | Explore available attributes interactively; avoid using it as production validation. |
| `numpy.array` | Notebook | Construct a NumPy array from values. |
| `numpy.var` | Notebook | Calculate variance of array values. |
| `numpy.random.randint` | Notebook | Generate random integers for examples or controlled tests. |
| `os.getcwd` | Notebook | Inspect the current working directory during local development; production jobs should use configured paths. |
| `matplotlib.pyplot.bar` | Notebook | Draw a bar chart. |
| `matplotlib.pyplot.xlabel` | Notebook | Label the x-axis. |
| `matplotlib.pyplot.ylabel` | Notebook | Label the y-axis. |
| `matplotlib.pyplot.title` | Notebook | Title a chart. |
| `matplotlib.pyplot.savefig` | Notebook | Save a chart to a file. |
| `matplotlib.pyplot.show` | Notebook | Display a chart in an interactive environment. |

## Notebook names that need special care

- **`DataFrame.append`**: this spelling is not listed as a recommended method because pandas removed `DataFrame.append`. Use `pandas.concat` to combine DataFrames. The notebook's `append` name may instead refer to Python's `list.append`.
- **`seaborn.distplot`**: deprecated; use `seaborn.histplot` or `seaborn.displot`.
- **`LabelEncoder`**: use it for target labels, not ordinary nominal feature columns. For feature columns, prefer `OneHotEncoder` or `OrdinalEncoder` with an explicit category order.
- **`values`, `shape`, and `index`**: these are attributes/properties, not methods to call with parentheses.
- **`mean` imputation and `KNNImputer`**: fit imputers on training data only, preferably inside a scikit-learn `Pipeline` and `ColumnTransformer`.
- **`PowerTransformer`, scalers, encoders, and feature selectors**: call `fit`/`fit_transform` on training data only; use the fitted transform on validation/test/inference data.
- **`warnings.filterwarnings`**: do not suppress all warnings to make a notebook look clean. Resolve the cause or narrowly suppress a known, documented warning.
- **`print` and `show`**: appropriate for interactive exploration, but use logging and saved outputs in scheduled or deployed jobs.

## Production checklist

1. Validate input columns, types, key uniqueness, row counts, and missingness before transformation.
2. Keep raw inputs unchanged; write cleaned outputs to a separate, versioned location.
3. Use explicit merge keys and join-cardinality checks; compare row counts before and after each join.
4. Split training and evaluation data before fitting imputers, scalers, encoders, transformations, or feature selectors.
5. Put learned preprocessing steps in a `Pipeline`/`ColumnTransformer` so inference applies the same fitted steps.
6. Handle unknown categories, missing values, non-finite numbers, and empty inputs explicitly.
7. Prefer typed, columnar formats such as Parquet for analytical intermediate data when supported by the workflow.
8. Replace exploratory `print` statements with structured logging and record important assumptions and output metrics.
