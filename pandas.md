# Pandas for Machine Learning Development

A practical reference for using pandas to inspect, clean, transform, and prepare tabular data for machine-learning workflows. Pandas is a data-manipulation tool; model fitting and learned preprocessing should be handled by the ML library and fitted only on training data.

## 1. Import and create data

```python
import numpy as np
import pandas as pd

df = pd.DataFrame({
    "age": [25, 38, 42, 31],
    "income": [42000, 81000, 76000, 55000],
    "city": ["North", "South", "North", "West"],
    "purchased": [0, 1, 1, 0],
})
```

Use a `DataFrame` for named feature columns and a `Series` for a single target:

```python
X = df.drop(columns="purchased")
y = df["purchased"]
```

Keep the target separate from features to reduce accidental target leakage.

## 2. Read and write common formats

```python
df = pd.read_csv("data.csv")
df = pd.read_parquet("data.parquet")
df = pd.read_excel("data.xlsx", sheet_name="Sheet1")

df.to_csv("cleaned_data.csv", index=False)
df.to_parquet("cleaned_data.parquet", index=False)
```

For large files, read only needed columns or process chunks:

```python
df = pd.read_csv("large.csv", usecols=["date", "customer_id", "amount"])

for chunk in pd.read_csv("large.csv", chunksize=100_000):
    print(chunk.shape)
```

Validate file provenance, schema, units, and access permissions before using data. Do not load untrusted serialized Python objects as if they were ordinary data files.

## 3. Inspect the dataset

```python
print(df.head())
print(df.sample(5, random_state=42))
print(df.shape)
print(df.columns.tolist())
print(df.dtypes)
df.info()
print(df.describe(include="all").T)
```

Useful checks before training:

```python
print("missing per column:\n", df.isna().sum())
print("duplicate rows:", df.duplicated().sum())
print("unique values:\n", df.nunique(dropna=False))
```

Inspect the target distribution, class balance for classification, numeric ranges, categorical levels, and whether identifiers or post-outcome fields accidentally appear as features.

## 4. Select rows and columns

```python
features = df[["age", "income", "city"]]
age_column = df["age"]             # Series
age_frame = df[["age"]]            # DataFrame

first_rows = df.iloc[:10]          # by integer position
selected_rows = df.loc[df["age"] >= 30]  # by label/boolean condition
```

Use `.loc` for label-based selection and `.iloc` for positional selection. Prefer explicit column lists over broad slicing when defining model inputs.

## 5. Filter and sort

```python
adults = df.loc[df["age"] >= 18]
high_income = df.loc[df["income"] > 70_000]
selected = df.loc[(df["age"] >= 30) & (df["income"] > 50_000)]
not_north = df.loc[~df["city"].eq("North")]

sorted_df = df.sort_values(["city", "income"], ascending=[True, False])
```

Use parentheses around each condition when combining pandas boolean expressions with `&`, `|`, or `~`.

## 6. Handle missing values

First understand the missingness and its possible meaning:

```python
missing_rate = df.isna().mean().sort_values(ascending=False)
print(missing_rate)
```

Pandas supports descriptive inspection and simple transformations:

```python
df["income_missing"] = df["income"].isna().astype("int8")
df["city"] = df["city"].fillna("Unknown")
```

For predictive workflows, do not calculate imputation statistics on the full dataset before splitting. Use sklearn transformers inside a `Pipeline` so imputers are fit on training folds only:

```python
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

numeric_model = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    LogisticRegression(max_iter=1000),
)
```

Choose imputation based on feature meaning and missingness behavior; a missing-value indicator can help when missingness itself carries signal.

## 7. Correct data types and values

```python
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["income"] = pd.to_numeric(df["income"], errors="coerce")
df["city"] = df["city"].astype("string")
```

`errors="coerce"` converts invalid values to missing values. Inspect the resulting missing values rather than silently treating conversion as successful.

Normalize strings carefully:

```python
df["city"] = df["city"].str.strip().str.lower()
df["city"] = df["city"].replace({"n/a": pd.NA, "unknown": pd.NA})
```

Do not normalize categories without checking whether differences in case or whitespace are meaningful.

## 8. Duplicates and invalid records

```python
duplicate_rows = df.loc[df.duplicated()]
df = df.drop_duplicates()

# Deduplicate using the actual entity key and a documented keep rule.
df = df.sort_values("updated_at").drop_duplicates("customer_id", keep="last")
```

Check constraints and ranges explicitly:

```python
invalid_age = df.loc[~df["age"].between(0, 120)]
invalid_income = df.loc[df["income"] < 0]
```

Do not delete outliers or duplicates automatically; determine whether they are data errors, repeated observations, or valid rare events.

## 9. Grouping and aggregation

`groupby` is useful for data-quality checks, summaries, and training-data analysis:

```python
city_summary = (
    df.groupby("city", dropna=False)
      .agg(
          rows=("age", "size"),
          average_income=("income", "mean"),
          median_age=("age", "median"),
      )
      .reset_index()
)
```

For target summaries, calculate them only within the appropriate training partition. Creating a target-mean feature from the complete dataset leaks target information; use fold-aware target encoding when appropriate.

## 10. Merge and concatenate datasets

```python
combined = pd.concat([train_part, validation_part], ignore_index=True)

joined = customers.merge(
    transactions,
    on="customer_id",
    how="left",
    validate="one_to_many",
    indicator=True,
)
print(joined["_merge"].value_counts())
```

Use `validate` to assert expected join cardinality (`one_to_one`, `one_to_many`, or `many_to_one`). Check row counts before and after joins to catch accidental row multiplication. Avoid joining information that would not be available at prediction time.

## 11. Feature engineering

Pandas can create deterministic, row-wise features:

```python
df["income_per_person"] = df["income"] / df["household_size"].replace(0, np.nan)
df["name_length"] = df["name"].fillna("").str.len()
df["is_weekend"] = df["date"].dt.dayofweek.isin([5, 6]).astype("int8")
```

Be cautious with features computed from aggregates, future information, target values, or the complete dataset. Any feature whose parameters are learned from data (for example, quantile cut points) should be fit from training data and then reused for validation/test/production.

## 12. Categorical data

Inspect categories before encoding:

```python
print(df["city"].value_counts(dropna=False))
print(df["city"].nunique())
```

One-hot encoding for exploration or a simple offline transformation:

```python
encoded = pd.get_dummies(df, columns=["city"], dummy_na=True, dtype="int8")
```

For model training, `OneHotEncoder(handle_unknown="ignore")` inside a sklearn `ColumnTransformer`/`Pipeline` is generally safer. It learns categories from training data and handles new categories consistently.

## 13. Numeric transformations and bins

```python
df["log_income"] = np.log1p(df["income"].clip(lower=0))
df["age_band"] = pd.cut(df["age"], bins=[0, 18, 35, 60, 120], include_lowest=True)
```

Transformations should match the feature’s semantics. If bin boundaries, scaling, or other parameters are selected from data, learn them from training folds only.

## 14. Time-series features and safe splitting

Sort by time before creating lags:

```python
df = df.sort_values(["series_id", "timestamp"])
df["previous_value"] = df.groupby("series_id")["value"].shift(1)
df["rolling_mean_7"] = (
    df.groupby("series_id")["value"]
      .transform(lambda values: values.shift(1).rolling(7, min_periods=3).mean())
)
```

The shift before rolling prevents the current observation from entering its own historical feature. Also ensure each feature uses only information available at prediction time.

Use chronological rather than random splits when production predicts future periods:

```python
cutoff = pd.Timestamp("2025-01-01")
train_df = df.loc[df["timestamp"] < cutoff]
test_df = df.loc[df["timestamp"] >= cutoff]
```

For repeated entities, use group-aware splits when rows from the same entity must not appear in both train and test.

## 15. Split features and target safely

```python
target = "purchased"
X = df.drop(columns=[target])
y = df[target]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

`stratify=y` is useful for classification when class proportions should be preserved. Do not stratify regression targets directly without a justified binning approach. For time series or grouped data, use a matching split strategy instead of random splitting.

## 16. Prepare pandas data for sklearn

Sklearn estimators generally accept DataFrames and preserve feature names in many workflows:

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

numeric_features = X_train.select_dtypes(include="number").columns
categorical_features = X_train.select_dtypes(exclude="number").columns

preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric_features),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical_features),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("estimator", LogisticRegression(max_iter=1000)),
])
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

Keep the original DataFrame column order/schema available for inference. Do not manually encode train and test separately because their resulting columns can differ.

## 17. Evaluation and reporting with pandas

```python
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, predictions))
cm = pd.DataFrame(
    confusion_matrix(y_test, predictions),
    index=["actual_0", "actual_1"],
    columns=["predicted_0", "predicted_1"],
)
print(cm)
```

For regression, put predictions next to the corresponding rows using aligned indices:

```python
prediction_report = X_test.copy()
prediction_report["actual"] = y_test
prediction_report["prediction"] = model.predict(X_test)
prediction_report["residual"] = prediction_report["actual"] - prediction_report["prediction"]
```

Pandas aligns operations by index labels. When combining arrays and Series, verify index alignment and row order; use `.to_numpy()` only when positional alignment is explicitly intended.

## 18. Common pitfalls

- **Data leakage:** fitting imputers, scalers, encoders, feature selection, or target encodings on all data before a split.
- **Index alignment:** assigning a shuffled Series to a DataFrame can reorder values by index.
- **Chained assignment:** prefer `df.loc[mask, "column"] = value` to ambiguous chained updates.
- **Join multiplication:** validate merge keys and join cardinality.
- **Target leakage:** exclude post-outcome fields and features derived from future labels.
- **Train/test schema mismatch:** use a fitted preprocessing pipeline and validate columns at inference.
- **Outliers:** investigate before clipping or deleting.
- **Categorical drift:** monitor unseen or newly frequent values.
- **Silent conversion:** inspect values coerced to missing during type conversion.
- **Large data:** select only required columns, choose sensible dtypes, and use Parquet/chunked processing where appropriate.

## 19. Quick end-to-end data preparation example

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("training_data.csv")
df = df.drop_duplicates()

target = "label"
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
numeric = X_train.select_dtypes(include="number").columns
categorical = X_train.select_dtypes(exclude="number").columns

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), numeric),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical),
])
model = Pipeline([
    ("preprocess", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=300, random_state=42)),
])
model.fit(X_train, y_train)
print("test accuracy:", model.score(X_test, y_test))
```

This example is a template, not a universal recipe. Match splitting, metrics, preprocessing, and model choice to the problem and production setting.

## 20. Useful pandas methods at a glance

| Goal | Common pandas tools |
|---|---|
| Read/write | `read_csv`, `read_parquet`, `to_csv`, `to_parquet` |
| Inspect | `head`, `sample`, `shape`, `info`, `describe`, `dtypes`, `nunique` |
| Select/filter | `loc`, `iloc`, `isin`, `between`, `query` |
| Missing data | `isna`, `notna`, `fillna`, `dropna` |
| Duplicates | `duplicated`, `drop_duplicates` |
| Transform | `assign`, `map`, `replace`, `astype`, `to_datetime` |
| Aggregate | `groupby`, `agg`, `transform`, `pivot_table` |
| Combine | `merge`, `join`, `concat` |
| Time series | `sort_values`, `shift`, `rolling`, `resample`, `.dt` |
| Export for models | `select_dtypes`, `to_numpy` (only when positional array input is needed) |

## 21. Pandas method reference for ML development

This is a broad, categorized reference to the pandas methods and accessors commonly used in ML projects. It is not a list of every pandas API (pandas includes many specialized and version-dependent methods). Examples use `df` for a `DataFrame`, `s` for a `Series`, and `pd` for the pandas package.

### Create, load, and save

| Method | Common use |
|---|---|
| `pd.DataFrame(data)` | Create a table from a dictionary, records, or arrays. |
| `pd.Series(data)` | Create one labeled column. |
| `pd.read_csv(path, usecols=..., dtype=...)` | Read CSV; select columns and types at load time. |
| `pd.read_table(path)` | Read delimited text, commonly TSV. |
| `pd.read_parquet(path, columns=...)` | Read columnar Parquet data. |
| `pd.read_json(path)` | Read JSON records or lines. |
| `pd.read_excel(path, sheet_name=...)` | Read an Excel sheet. |
| `pd.read_sql(query, connection)` | Read query results from a database. |
| `pd.read_fwf(path)` | Read fixed-width text data. |
| `df.to_csv(path, index=False)` | Export a table to CSV. |
| `df.to_parquet(path, index=False)` | Export a table to Parquet. |
| `df.to_json(path, orient=...)` | Export a table to JSON. |
| `df.to_excel(path, index=False)` | Export a table to Excel. |
| `df.to_sql(name, connection, if_exists=...)` | Write rows to a SQL table. |
| `pd.read_csv(..., chunksize=...)` | Iterate through a large CSV in chunks. |

### Inspect shape, schema, and values

| Attribute or method | Common use |
|---|---|
| `df.shape` | Number of rows and columns. |
| `df.size` | Number of cells. |
| `df.empty` | Whether the table has no rows. |
| `df.columns` / `df.index` | Column names and row index. |
| `df.dtypes` | Type of each column. |
| `df.info()` | Concise schema, non-null counts, and memory estimate. |
| `df.head(n)` / `df.tail(n)` | Inspect first/last rows. |
| `df.sample(n=..., random_state=...)` | Reproducible random row sample. |
| `df.describe()` | Numeric summary statistics. |
| `df.describe(include="all")` | Summary of numeric and non-numeric columns. |
| `df.memory_usage(deep=True)` | Estimate memory use by column. |
| `df.nunique(dropna=False)` | Count distinct values per column. |
| `df.value_counts()` | Count value combinations or Series values. |
| `s.unique()` | Return distinct values in a Series. |
| `s.value_counts(dropna=False)` | Frequency table for a Series. |
| `df.dtypes.value_counts()` | Count columns by dtype. |

### Select columns and rows

| Method or pattern | Common use |
|---|---|
| `df["col"]` | Select one column as a Series. |
| `df[["a", "b"]]` | Select columns as a DataFrame. |
| `df.loc[row_selector, column_selector]` | Label-based and boolean selection. |
| `df.iloc[row_positions, column_positions]` | Integer-position selection. |
| `df.at[row_label, column_label]` | Fast scalar access by labels. |
| `df.iat[row_position, column_position]` | Fast scalar access by positions. |
| `df.filter(items=[...])` | Select columns/index labels by exact names. |
| `df.filter(regex="pattern", axis="columns")` | Select columns by name pattern. |
| `df.select_dtypes(include="number")` | Select columns by type. |
| `df.get("col", default)` | Get a column with a default if absent. |
| `df.xs(key, level=...)` | Select a cross-section from a MultiIndex. |

For ML features, prefer explicit columns such as `X = df.loc[:, feature_columns]`; this makes feature order and schema visible.

### Filter, query, and sort

| Method | Common use |
|---|---|
| `df.loc[condition]` | Filter with a boolean mask. |
| `s.isin(values)` | Test membership in a set/list of values. |
| `s.between(left, right)` | Test whether values fall within a range. |
| `df.query("age >= 18")` | Filter with a readable expression. |
| `df.nlargest(n, "score")` | Select rows with the largest values. |
| `df.nsmallest(n, "score")` | Select rows with the smallest values. |
| `df.sort_values(by, ascending=...)` | Sort rows by column values. |
| `df.sort_index()` | Sort by index labels. |
| `s.argsort()` | Positions that would sort a Series. |

Example:

```python
eligible = df.loc[df["age"].between(18, 65) & df["city"].isin(["North", "West"])]
top_rows = eligible.nlargest(10, "income")
```

### Rename, add, remove, and reorder columns

| Method | Common use |
|---|---|
| `df.rename(columns={...})` | Rename columns or index labels. |
| `df.set_axis(names, axis="columns")` | Set all column names. |
| `df.assign(new_col=...)` | Return a new table with added/updated columns. |
| `df.insert(loc, name, values)` | Insert a column at a chosen position. |
| `df.drop(columns=[...])` | Remove feature or target columns. |
| `df.pop("col")` | Remove and return one column. |
| `df.reindex(columns=...)` | Reorder, add, or remove columns to match a schema. |
| `df.set_index("id")` | Make one or more columns the index. |
| `df.reset_index(drop=True)` | Restore a default integer index. |

Use `.copy()` when creating a standalone dataset slice for later mutation:

```python
X = df.loc[:, feature_columns].copy()
```

### Missing values and duplicates

| Method | Common use |
|---|---|
| `df.isna()` / `df.isnull()` | Mark missing cells. |
| `df.notna()` / `df.notnull()` | Mark present cells. |
| `df.isna().sum()` | Count missing values per column. |
| `df.dropna(subset=[...])` | Remove rows missing required fields. |
| `df.fillna(value)` | Fill missing values with explicit values. |
| `df.interpolate()` | Interpolate ordered numeric gaps when appropriate. |
| `df.duplicated(subset=[...])` | Flag repeated rows or entity keys. |
| `df.drop_duplicates(subset=[...], keep="last")` | Deduplicate with a chosen rule. |

For modeling, learned imputation belongs inside a training-fitted pipeline; do not use full-dataset means/medians before validation.

### Convert types and validate values

| Method | Common use |
|---|---|
| `df.astype(dtype)` | Convert columns to specified types. |
| `df.convert_dtypes()` | Infer nullable pandas types. |
| `pd.to_numeric(s, errors="coerce")` | Convert values to numeric; invalid entries become missing. |
| `pd.to_datetime(s, errors="coerce")` | Parse dates; invalid entries become `NaT`. |
| `pd.to_timedelta(s)` | Parse durations. |
| `s.clip(lower=..., upper=...)` | Cap values to bounds (only when justified). |
| `s.between(a, b)` | Validate a numeric range. |
| `s.isin(allowed)` | Check allowed categories or codes. |
| `df.replace(old, new)` | Replace sentinel or invalid values. |

After coercion, inspect newly missing values and log/report invalid records instead of silently discarding them.

### String, categorical, and datetime accessors

Accessors apply specialized methods to each Series value.

| Accessor and method | Common use |
|---|---|
| `s.str.strip()` | Remove surrounding whitespace. |
| `s.str.lower()` / `s.str.upper()` | Normalize case where appropriate. |
| `s.str.contains(pattern, na=False)` | Search text for a pattern. |
| `s.str.replace(pattern, replacement, regex=...)` | Clean or transform text. |
| `s.str.len()` | Compute string length. |
| `s.str.split(sep)` | Split strings into parts. |
| `s.str.extract(pattern)` | Extract regex capture groups. |
| `s.cat.codes` | Read category codes after categorical conversion. |
| `s.cat.categories` | Inspect category levels. |
| `s.dt.year`, `s.dt.month`, `s.dt.day` | Extract date parts. |
| `s.dt.dayofweek` | Day of week (Monday is 0). |
| `s.dt.hour` | Extract hour from datetimes. |
| `s.dt.floor("D")` / `s.dt.ceil("D")` | Round timestamps to a frequency. |
| `s.dt.tz_localize(...)` / `s.dt.tz_convert(...)` | Set or convert time zones. |

Text normalization and date extraction should preserve domain meaning and use only information available at prediction time.

### Map, replace, and vectorized transformations

| Method | Common use |
|---|---|
| `s.map(mapping)` | Map known values to labels or values. |
| `s.replace(mapping)` | Replace one or many values. |
| `s.where(condition, other)` | Keep values meeting a condition, replace others. |
| `s.mask(condition, other)` | Replace values meeting a condition. |
| `df.apply(func, axis=...)` | Apply a function by row/column (often slower). |
| `s.apply(func)` | Apply a function elementwise (often slower than vectorized methods). |
| `s.map(func)` | Map a Series through a function or dictionary. |
| `np.where(condition, a, b)` | Vectorized conditional assignment. |
| `pd.cut(s, bins=...)` | Bin values using fixed boundaries. |
| `pd.qcut(s, q=...)` | Bin by quantiles. |

Prefer vectorized arithmetic, `.str`, `.dt`, `map`, `where`, and NumPy operations to Python loops. If bin edges or mappings are learned from data, derive them from training data and reuse them.

### Group, aggregate, and window operations

| Method | Common use |
|---|---|
| `df.groupby(keys)` | Split rows into groups for analysis. |
| `grouped.agg(...)` | Compute one or more summaries per group. |
| `grouped.transform(func)` | Return group statistics aligned to original rows. |
| `grouped.filter(func)` | Keep/drop groups using a condition. |
| `grouped.size()` / `grouped.count()` | Count rows or non-missing values by group. |
| `grouped.mean()` / `median()` / `sum()` | Common group summaries. |
| `s.rolling(window).mean()` | Rolling-window feature/statistic. |
| `s.expanding().mean()` | Expanding cumulative-window statistic. |
| `s.ewm(span=...).mean()` | Exponentially weighted statistic. |
| `s.shift(periods=1)` | Lag or lead values. |
| `s.diff(periods=1)` | Difference from a previous row. |
| `s.pct_change()` | Fractional change between rows. |
| `df.resample("D")` | Time-based grouping after setting a datetime index. |

Use `groupby(...).transform(...)` when the result must align with original rows. For time-dependent features, sort first and shift before rolling so the current/future target cannot leak into its own features.

### Join, concatenate, and reshape

| Method | Common use |
|---|---|
| `pd.concat([...], axis=0)` | Stack rows; use `ignore_index=True` when appropriate. |
| `pd.concat([...], axis=1)` | Combine aligned columns (indexes must be checked). |
| `df.merge(other, on=..., how=..., validate=...)` | Relational join with explicit keys/cardinality. |
| `df.join(other, on=...)` | Join primarily by index. |
| `df.pivot(index=..., columns=..., values=...)` | Reshape unique index/column pairs to wide form. |
| `df.pivot_table(..., aggfunc=...)` | Aggregate duplicate pairs while reshaping. |
| `df.melt(id_vars=..., value_vars=...)` | Convert wide data to long form. |
| `df.stack()` / `df.unstack()` | Move levels between columns and index. |
| `df.explode("list_col")` | Turn list-like cell entries into multiple rows. |
| `pd.crosstab(a, b)` | Frequency table for categorical combinations. |
| `pd.get_dummies(df, columns=[...])` | One-hot encode for exploratory/simple transformations. |

Check row counts and join keys after every merge. For model pipelines, sklearn encoders are usually safer than `get_dummies` because the fitted encoder handles previously unseen categories.

### Descriptive statistics and correlation

| Method | Common use |
|---|---|
| `s.count()` | Count non-missing values. |
| `s.mean()`, `s.median()`, `s.std()`, `s.var()` | Numeric summaries. |
| `s.min()`, `s.max()`, `s.quantile(q)` | Ranges and quantiles. |
| `s.mode()` | Most frequent value(s). |
| `s.skew()`, `s.kurt()` | Distribution-shape diagnostics. |
| `df.corr(numeric_only=True)` | Pairwise numeric correlations. |
| `df.cov(numeric_only=True)` | Pairwise covariance. |
| `s.corr(other)` | Correlation between two Series. |
| `s.rank()` | Convert values to ranks. |
| `s.nlargest(n)` / `s.nsmallest(n)` | Inspect extreme values. |

Correlation is not causation, and pairwise correlation does not detect every nonlinear relationship. Use summaries to generate questions, not to make unsupported causal claims.

### Index alignment and array conversion

Pandas aligns `Series` values by index labels in assignments and arithmetic. Use `reset_index(drop=True)` when a fresh positional index is intended, and verify ordering before converting to NumPy:

```python
X = df.loc[:, feature_columns]
y = df.loc[X.index, target_column]

# Convert only if the downstream estimator/API requires an unlabeled array.
X_array = X.to_numpy()
y_array = y.to_numpy()
```

Common conversion methods:

| Method | Common use |
|---|---|
| `df.to_numpy()` | Convert a DataFrame to a NumPy array. |
| `s.to_numpy()` | Convert a Series to a NumPy array. |
| `df.values` | Legacy array access; prefer `to_numpy()`. |
| `df.items()` | Iterate over `(column_name, Series)` pairs. |
| `df.iterrows()` | Iterate rows (slow; avoid for vectorized feature engineering). |
| `df.itertuples()` | Faster row iteration than `iterrows`, still usually slower than vectorization. |

### Method chaining

Method chains make cleaning steps easier to review. Break long chains into named steps when it improves debugging:

```python
clean = (
    pd.read_csv("data.csv")
      .rename(columns=str.strip)
      .drop_duplicates()
      .assign(date=lambda frame: pd.to_datetime(frame["date"], errors="coerce"))
      .loc[lambda frame: frame["date"].notna()]
      .sort_values("date")
      .reset_index(drop=True)
)
```

### ML data-preparation checklist

1. Validate columns, data types, units, row counts, and target definition.
2. Inspect missing, duplicate, invalid, and rare values before modifying them.
3. Use explicit feature columns and keep the target separate.
4. Choose random, temporal, or group-based splitting to match deployment.
5. Fit learned preprocessing only on training data with a pipeline.
6. Preserve index/row alignment and validate join cardinality.
7. Test the same schema and transformations at inference time.
8. Document every cleaning rule and verify it does not use future or target information.
