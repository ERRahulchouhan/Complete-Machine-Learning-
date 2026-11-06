# Master EDA methods: notebook and production reference

This guide combines the method names listed in [eda.md](./eda.md) with additional methods that are commonly useful in reliable data-analysis workflows. It is organized by EDA topic/session so you can find both the method and a runnable code example.

- **Notebook** means the name appears in one or more EDA notebooks.
- **Recommended** means an additional production-oriented method or API.
- Names are shown with their library prefix when that helps identify the exact API. For example, `DataFrame.drop` is different from `Path.drop` (which is not a standard method).
- Some names in the original notebooks are attributes rather than methods. They are marked **attribute**. Notebook-derived names are retained even when a spelling is incorrect or an API is outdated; see the notes at the end.

## Shared example setup

Run this setup once before trying the examples. Each **Code example** cell is then a runnable usage example; adjust input files, columns, and values for your own data.

```python
import os
import sqlite3
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import (
    SelectKBest,
    VarianceThreshold,
    mutual_info_classif,
)
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    Normalizer,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)

df = pd.DataFrame(
    {
        "id": range(1, 9),
        "category": ["A", "B"] * 4,
        "value": [10.0, 20.0, 30.0, 40.0, 15.0, 25.0, 35.0, 45.0],
        "target": [0, 1] * 4,
    }
)
other = pd.DataFrame({"id": range(1, 9), "extra": range(100, 900, 100)})
s = df["value"]
X = df[["value"]]
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
feature_names = X.columns.to_numpy()
connection = sqlite3.connect(":memory:")
fig, ax = plt.subplots()
category_counts = df["category"].value_counts()
values_list = []
counts = {"A": 3, "B": 2}
```

Examples involving Excel or Parquet may require an optional pandas file-format engine. `OneHotEncoder(sparse_output=False)` is for recent scikit-learn versions; older versions use `sparse=False`.

## Session 1 — Load data and manage DataFrames

Use this session to load data with explicit types and settings, inspect its shape, and save reproducible outputs.

| Exact name | Origin | Code example |
|---|---|---|
|  `pandas.read_csv`  | Notebook  | `pd.read_csv("data.csv")`  |
|  `pandas.read_excel`  | Notebook  | `pd.read_excel("data.xlsx", sheet_name="Sheet1")`  |
|  `pandas.DataFrame`  | Notebook  | `pd.DataFrame({"category": ["A", "B"], "value": [10, 20]})`  |
|  `DataFrame.rename`  | Notebook  | `df.rename(columns={"category": "group"})`  |
|  `DataFrame.drop`  | Notebook  | `df.drop(columns=["target"])`  |
|  `DataFrame.to_csv`  | Notebook  | `df.to_csv("cleaned.csv", index=False)`  |
|  `DataFrame.to_excel`  | Notebook  | `df.to_excel("report.xlsx", index=False)`  |
|  `DataFrame.copy`  | Notebook  | `clean_df = df.copy()`  |
|  `DataFrame.shape`  | Notebook — attribute  | `row_count, column_count = df.shape`  |
|  `DataFrame.index`  | Notebook — attribute  | `row_labels = df.index`  |
|  `pandas.Series`  | Notebook  | `pd.Series([10, 20, 30], name="value")`  |
|  `DataFrame.columns`  | Recommended — attribute  | `column_names = df.columns.tolist()`  |
|  `DataFrame.dtypes`  | Recommended — attribute  | `column_types = df.dtypes`  |
|  `DataFrame.memory_usage`  | Recommended  | `memory_by_column = df.memory_usage(deep=True)`  |
|  `pandas.read_parquet`  | Recommended  | `pd.read_parquet("data.parquet")`  |
|  `DataFrame.to_parquet`  | Recommended  | `df.to_parquet("cleaned.parquet", index=False)`  |
|  `pandas.read_json`  | Recommended  | `pd.read_json("data.json")`  |
|  `pandas.read_sql`  | Recommended  | `pd.read_sql("SELECT 1 AS value", connection)`  |

## Session 2 — Basic data checks and data quality

Run these checks before analysis so missing values, duplicates, incorrect types, and unexpected dimensions are visible.

| Exact name | Origin | Code example |
|---|---|---|
|  `DataFrame.head`  | Notebook  | `df.head(5)`  |
|  `DataFrame.tail`  | Notebook  | `df.tail(5)`  |
|  `DataFrame.info`  | Notebook  | `df.info()`  |
|  `DataFrame.isna`  | Notebook  | `missing_mask = df.isna()`  |
|  `DataFrame.isnull`  | Notebook  | `missing_mask = df.isnull()`  |
|  `DataFrame.notna`  | Recommended  | `present_mask = df.notna()`  |
|  `DataFrame.isna().sum`  | Notebook  | `missing_per_column = df.isna().sum()`  |
|  `DataFrame.dropna`  | Notebook  | `df.dropna(subset=["value"])`  |
|  `DataFrame.fillna`  | Notebook  | `df.fillna({"value": df["value"].median()})`  |
|  `DataFrame.duplicated`  | Recommended  | `duplicate_keys = df.duplicated(subset=["id"], keep=False)`  |
|  `DataFrame.drop_duplicates`  | Notebook  | `df.drop_duplicates(subset=["id"], keep="first")`  |
|  `DataFrame.select_dtypes`  | Notebook  | `numeric_df = df.select_dtypes(include="number")`  |
|  `DataFrame.describe`  | Notebook  | `df.describe(include="all")`  |
|  `DataFrame.nunique`  | Recommended  | `unique_counts = df.nunique(dropna=False)`  |
|  `DataFrame.memory_usage`  | Recommended  | `memory_by_column = df.memory_usage(deep=True)`  |
|  `DataFrame.take`  | Notebook  | `df.take([0, 1])`  |
|  `DataFrame.to_list`  | Notebook  | `df["category"].to_list()`  |
|  `Series.to_dict`  | Notebook  | `s.value_counts().to_dict()`  |
|  `DataFrame.items`  | Notebook  | `column_types = {name: column.dtype for name, column in df.items()}`  |
|  `DataFrame.keys`  | Notebook  | `column_names = df.keys()`  |
|  `DataFrame.values`  | Notebook — attribute  | `array = df.values`  |
|  `DataFrame.to_numpy`  | Recommended  | `array = df[["value"]].to_numpy()`  |
|  `DataFrame.reset_index`  | Notebook  | `df.reset_index(drop=True)`  |
|  `DataFrame.index`  | Notebook — attribute  | `row_labels = df.index`  |
|  `len`  | Notebook — built-in  | `row_count = len(df)`  |
|  `type`  | Notebook — built-in  | `object_type = type(df)`  |
|  `print`  | Notebook — built-in  | `print(df.head())`  |

## Session 3 — Univariate categorical analysis

Use frequency tables and plots to understand category distribution, rare levels, and cardinality.

| Exact name | Origin | Code example |
|---|---|---|
|  `Series.value_counts`  | Notebook  | `category_counts = df["category"].value_counts(dropna=False)`  |
|  `Series.unique`  | Notebook  | `observed_categories = df["category"].unique()`  |
|  `Series.nunique`  | Notebook  | `distinct_count = df["category"].nunique()`  |
|  `Series.to_list`  | Notebook  | `category_list = df["category"].to_list()`  |
|  `dict`  | Notebook — built-in  | `counts = {"A": 3, "B": 2}`  |
|  `dict.keys`  | Notebook  | `keys = counts.keys()`  |
|  `dict.values`  | Notebook  | `values = counts.values()`  |
|  `dict.items`  | Notebook  | `items = counts.items()`  |
|  `matplotlib.pyplot.figure`  | Notebook  | `fig = plt.figure(figsize=(8, 4))`  |
|  `matplotlib.pyplot.bar`  | Notebook  | `plt.bar(category_counts.index.astype(str), category_counts.values)`  |
|  `matplotlib.axes.Axes.bar_label`  | Notebook  | `bars = ax.bar(["A", "B"], [3, 2]); ax.bar_label(bars)`  |
|  `seaborn.countplot`  | Notebook  | `sns.countplot(data=df, x="category")`  |
|  `matplotlib.pyplot.pie`  | Notebook  | `plt.pie(category_counts.values, labels=category_counts.index.astype(str), autopct="%.1f%%")`  |
|  `pandas.Series.plot`  | Notebook  | `category_counts.plot(kind="bar")`  |
|  `matplotlib.pyplot.title`  | Notebook  | `plt.title("Value by category")`  |
|  `matplotlib.pyplot.xlabel`  | Notebook  | `plt.xlabel("Category")`  |
|  `matplotlib.pyplot.ylabel`  | Notebook  | `plt.ylabel("Count")`  |
|  `matplotlib.pyplot.savefig`  | Notebook  | `plt.savefig("chart.png", dpi=150, bbox_inches="tight")`  |
|  `matplotlib.pyplot.show`  | Notebook  | `plt.show()`  |
|  `DataFrame.groupby`  | Recommended  | `grouped = df.groupby("category", dropna=False)`  |
|  `DataFrame.groupby().size`  | Recommended  | `group_sizes = df.groupby("category", dropna=False).size()`  |
|  `DataFrame.groupby().agg`  | Recommended  | `summary = df.groupby("category").agg(value_mean=("value", "mean"), row_count=("id", "count"))`  |
|  `DataFrame.pivot_table`  | Recommended  | `pivot = df.pivot_table(index="category", values="value", aggfunc="mean")`  |
|  `DataFrame.value_counts`  | Recommended  | `joint_counts = df[["category", "target"]].value_counts(dropna=False)`  |

## Session 4 — Univariate numerical analysis

Use summaries and distributions to understand scale, spread, skew, and unusual values.

| Exact name | Origin | Code example |
|---|---|---|
|  `Series.count`  | Notebook  | `non_missing_count = df["value"].count()`  |
|  `Series.mean`  | Notebook  | `value_mean = df["value"].mean()`  |
|  `Series.median`  | Notebook  | `value_median = df["value"].median()`  |
|  `Series.std`  | Notebook  | `value_std = df["value"].std()`  |
|  `Series.min`  | Notebook  | `value_min = df["value"].min()`  |
|  `Series.max`  | Notebook  | `value_max = df["value"].max()`  |
|  `Series.quantile`  | Notebook  | `q1, q3 = df["value"].quantile([0.25, 0.75])`  |
|  `numpy.percentile`  | Notebook  | `p95 = np.percentile(df["value"], 95)`  |
|  `numpy.quantile`  | Notebook  | `q95 = np.quantile(df["value"], 0.95)`  |
|  `numpy.mean`  | Notebook  | `average = np.mean(df["value"])`  |
|  `numpy.median`  | Notebook  | `middle = np.median(df["value"])`  |
|  `numpy.std`  | Notebook  | `spread = np.std(df["value"], ddof=1)`  |
|  `numpy.min`  | Notebook  | `smallest = np.min(df["value"])`  |
|  `numpy.max`  | Notebook  | `largest = np.max(df["value"])`  |
|  `builtins.min`  | Notebook  | `smallest = min([3, 1, 2])`  |
|  `builtins.max`  | Notebook  | `largest = max([3, 1, 2])`  |
|  `round`  | Notebook — built-in  | `rounded_value = round(3.14159, 2)`  |
|  `seaborn.displot`  | Notebook  | `sns.displot(data=df, x="value", kind="hist")`  |
|  `matplotlib.pyplot.hist`  | Notebook  | `plt.hist(df["value"].dropna(), bins=10)`  |
|  `matplotlib.pyplot.bar`  | Notebook  | `plt.bar(category_counts.index.astype(str), category_counts.values)`  |
|  `matplotlib.pyplot.subplot`  | Notebook  | `ax = plt.subplot(1, 2, 1)`  |
|  `DataFrame.describe`  | Notebook  | `df.describe(include="all")`  |
|  `Series.skew`  | Recommended  | `skewness = df["value"].skew()`  |
|  `Series.kurt`  | Recommended  | `tail_weight = df["value"].kurt()`  |
|  `numpy.isfinite`  | Recommended  | `finite_mask = np.isfinite(df["value"].to_numpy())`  |

## Session 5 — Outlier analysis

Identify potentially unusual observations, then investigate their cause before removing or changing them.

| Exact name | Origin | Code example |
|---|---|---|
|  `Series.quantile`  | Notebook  | `q1, q3 = df["value"].quantile([0.25, 0.75])`  |
|  `outliers`  | Notebook — not a standard method  | `outlier_mask = df["value"] > df["value"].quantile(0.95)  # Example threshold; choose a domain rule.`  |
|  `matplotlib.pyplot.boxplot`  | Notebook  | `plt.boxplot(df["value"].dropna())`  |
|  `matplotlib.pyplot.hist`  | Notebook  | `plt.hist(df["value"].dropna(), bins=10)`  |
|  `seaborn.boxplot`  | Recommended  | `sns.boxplot(data=df, x="category", y="value")`  |
|  `DataFrame.clip`  | Recommended  | `df["value"].clip(lower=0, upper=100)`  |
|  `numpy.percentile`  | Notebook  | `p95 = np.percentile(df["value"], 95)`  |
|  `numpy.quantile`  | Notebook  | `q95 = np.quantile(df["value"], 0.95)`  |
|  `len`  | Notebook — built-in  | `row_count = len(df)`  |
|  `format`  | Notebook — built-in  | `display_value = format(3.14159, ".2f")`  |

## Session 6 — Bivariate and multivariate analysis

Compare variables and groups, and inspect associations without treating correlation as causation.

| Exact name | Origin | Code example |
|---|---|---|
|  `pandas.crosstab`  | Notebook  | `pd.crosstab(df["category"], df["target"], normalize="index")`  |
|  `DataFrame.corr`  | Notebook  | `numeric_df = df.select_dtypes(include="number"); correlation = numeric_df.corr(method="spearman")`  |
|  `DataFrame.corrwith`  | Recommended  | `numeric_df = df.select_dtypes(include="number"); target_corr = numeric_df.corrwith(df["target"])`  |
|  `DataFrame.groupby`  | Recommended  | `grouped = df.groupby("category", dropna=False)`  |
|  `DataFrame.groupby().agg`  | Recommended  | `summary = df.groupby("category").agg(value_mean=("value", "mean"), row_count=("id", "count"))`  |
|  `seaborn.heatmap`  | Notebook  | `sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")`  |
|  `matplotlib.pyplot.scatter`  | Notebook  | `plt.scatter(df["value"], df["target"])`  |
|  `seaborn.scatterplot`  | Recommended  | `sns.scatterplot(data=df, x="value", y="target", hue="category")`  |
|  `seaborn.pairplot`  | Recommended  | `sns.pairplot(df, vars=["value", "target"], hue="category")`  |
|  `seaborn.countplot`  | Notebook  | `sns.countplot(data=df, x="category")`  |
|  `pandas.Series.plot`  | Notebook  | `category_counts.plot(kind="bar")`  |
|  `DataFrame.set_index`  | Notebook  | `indexed_df = df.set_index("id")`  |
|  `DataFrame.keys`  | Notebook  | `column_names = df.keys()`  |
|  `Series.unique`  | Notebook  | `observed_categories = df["category"].unique()`  |
|  `Series.value_counts`  | Notebook  | `category_counts = df["category"].value_counts(dropna=False)`  |
|  `matplotlib.pyplot.figure`  | Notebook  | `fig = plt.figure(figsize=(8, 4))`  |
|  `matplotlib.pyplot.subplot`  | Notebook  | `ax = plt.subplot(1, 2, 1)`  |
|  `matplotlib.pyplot.plot`  | Notebook  | `plt.plot([1, 2, 3], [2, 4, 3], label="series")`  |
|  `matplotlib.pyplot.title`  | Notebook  | `plt.title("Value by category")`  |
|  `range`  | Notebook — built-in  | `positions = range(len(df))`  |
|  `sorted`  | Notebook — built-in  | `ordered_values = sorted([3, 1, 2])`  |

## Session 7 — Categorical encoding

Encode categories only after deciding whether they are nominal, ordinal, or target labels. Learn encodings on training data and reuse them for validation and production data.

| Exact name | Origin | Code example |
|---|---|---|
|  `pandas.get_dummies`  | Notebook  | `encoded = pd.get_dummies(df, columns=["category"], dtype=int)`  |
|  `sklearn.preprocessing.LabelEncoder`  | Notebook  | `encoder = LabelEncoder()`  |
|  `LabelEncoder.fit_transform`  | Notebook  | `encoded_target = encoder.fit_transform(df["category"].astype(str))`  |
|  `LabelEncoder.inverse_transform`  | Notebook  | `labels = encoder.inverse_transform(encoded_target)`  |
|  `Series.map`  | Notebook  | `mapped = df["category"].map({"A": 0, "B": 1})`  |
|  `numpy.where`  | Notebook  | `flag = np.where(df["value"] > 20, "high", "low")`  |
|  `DataFrame.drop`  | Notebook  | `df.drop(columns=["target"])`  |
|  `sklearn.preprocessing.OneHotEncoder`  | Recommended  | `encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)`  |
|  `sklearn.preprocessing.OrdinalEncoder`  | Recommended  | `encoder = OrdinalEncoder(categories=[["low", "medium", "high"]], handle_unknown="use_encoded_value", unknown_value=-1)`  |
|  `sklearn.preprocessing.LabelEncoder`  | Notebook  | `encoder = LabelEncoder()`  |
|  `sklearn.compose.ColumnTransformer`  | Recommended  | `preprocessor = ColumnTransformer([("category", OneHotEncoder(handle_unknown="ignore"), ["category"])], remainder="passthrough")`  |
|  `sklearn.pipeline.Pipeline`  | Recommended  | `pipeline = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression())])`  |
|  `Series.nunique`  | Notebook  | `distinct_count = df["category"].nunique()`  |
|  `Series.unique`  | Notebook  | `observed_categories = df["category"].unique()`  |

## Session 8 — Scaling and normalization

Scale numeric features when the downstream algorithm is scale-sensitive. Fit the scaler on training data only.

| Exact name | Origin | Code example |
|---|---|---|
|  `sklearn.preprocessing.StandardScaler`  | Notebook  | `scaler = StandardScaler()`  |
|  `StandardScaler.fit_transform`  | Notebook  | `X_scaled = StandardScaler().fit_transform(X_train)`  |
|  `sklearn.preprocessing.MinMaxScaler`  | Notebook  | `scaler = MinMaxScaler(feature_range=(0, 1))`  |
|  `MinMaxScaler.fit_transform`  | Notebook  | `X_scaled = MinMaxScaler().fit_transform(X_train)`  |
|  `sklearn.preprocessing.RobustScaler`  | Recommended  | `scaler = RobustScaler()`  |
|  `sklearn.preprocessing.Normalizer`  | Recommended  | `normalizer = Normalizer(norm="l2")`  |
|  `sklearn.pipeline.Pipeline`  | Recommended  | `pipeline = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression())])`  |
|  `Series.mean`  | Notebook  | `value_mean = df["value"].mean()`  |
|  `Series.std`  | Notebook  | `value_std = df["value"].std()`  |
|  `Series.idxmin`  | Notebook  | `minimum_row = df["value"].idxmin()`  |
|  `Series.idxmax`  | Notebook  | `maximum_row = df["value"].idxmax()`  |
|  `matplotlib.pyplot.boxplot`  | Notebook  | `plt.boxplot(df["value"].dropna())`  |
|  `matplotlib.pyplot.hist`  | Notebook  | `plt.hist(df["value"].dropna(), bins=10)`  |

## Session 9 — Missing-value analysis and imputation

Measure missingness, understand why values are absent, and fit imputers on training data only.

| Exact name | Origin | Code example |
|---|---|---|
|  `DataFrame.isna`  | Notebook  | `missing_mask = df.isna()`  |
|  `DataFrame.isnull`  | Notebook  | `missing_mask = df.isnull()`  |
|  `DataFrame.isna().sum`  | Notebook  | `missing_per_column = df.isna().sum()`  |
|  `DataFrame.dropna`  | Notebook  | `df.dropna(subset=["value"])`  |
|  `DataFrame.fillna`  | Notebook  | `df.fillna({"value": df["value"].median()})`  |
|  `Series.mean`  | Notebook  | `value_mean = df["value"].mean()`  |
|  `Series.mode`  | Notebook  | `most_common = df["category"].mode(dropna=True)`  |
|  `sklearn.impute.SimpleImputer`  | Recommended  | `imputer = SimpleImputer(strategy="median")`  |
|  `sklearn.impute.KNNImputer`  | Notebook  | `imputer = KNNImputer(n_neighbors=3)`  |
|  `KNNImputer.fit_transform`  | Notebook  | `X_imputed = KNNImputer(n_neighbors=3).fit_transform(X_train)`  |
|  `sklearn.compose.ColumnTransformer`  | Recommended  | `preprocessor = ColumnTransformer([("category", OneHotEncoder(handle_unknown="ignore"), ["category"])], remainder="passthrough")`  |
|  `sklearn.pipeline.Pipeline`  | Recommended  | `pipeline = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression())])`  |
|  `DataFrame.to_csv`  | Notebook  | `df.to_csv("cleaned.csv", index=False)`  |
|  `warnings.filterwarnings`  | Notebook  | `warnings.filterwarnings("default")  # Keep warnings visible unless a specific warning is understood.`  |

## Session 10 — Transformations and distributions

Transformations can reduce skew or improve model fit, but should be selected based on data meaning and validated after fitting.

| Exact name | Origin | Code example |
|---|---|---|
|  `numpy.log`  | Notebook  | `logged_values = np.log1p(df["value"].clip(lower=0))`  |
|  `numpy.sqrt`  | Notebook  | `root_values = np.sqrt(df["value"].clip(lower=0))`  |
|  `numpy.reciprocal`  | Notebook  | `reciprocal_values = np.reciprocal(df["value"].replace(0, np.nan))`  |
|  `numpy.random.normal`  | Notebook  | `synthetic = np.random.normal(loc=0, scale=1, size=100)`  |
|  `numpy.random.exponential`  | Notebook  | `synthetic = np.random.exponential(scale=1, size=100)`  |
|  `numpy.random.randn`  | Notebook  | `synthetic = np.random.randn(100)`  |
|  `sklearn.preprocessing.PowerTransformer`  | Notebook  | `transformer = PowerTransformer(method="yeo-johnson")`  |
|  `PowerTransformer.fit`  | Notebook  | `transformer = PowerTransformer().fit(X_train)`  |
|  `PowerTransformer.transform`  | Notebook  | `X_transformed = transformer.transform(X_test)`  |
|  `PowerTransformer.fit_transform`  | Notebook  | `X_transformed = PowerTransformer().fit_transform(X_train)`  |
|  `numpy.ndarray.reshape`  | Notebook  | `column_array = np.array([1, 2, 3]).reshape(-1, 1)`  |
|  `seaborn.displot`  | Notebook  | `sns.displot(data=df, x="value", kind="hist")`  |
|  `seaborn.histplot`  | Recommended  | `sns.histplot(data=df, x="value", bins=10, kde=True)`  |
|  `seaborn.distplot`  | Notebook — deprecated  | `sns.histplot(data=df, x="value", bins=10, kde=True)  # Replacement for deprecated distplot.`  |
|  `matplotlib.pyplot.hist`  | Notebook  | `plt.hist(df["value"].dropna(), bins=10)`  |
|  `matplotlib.pyplot.legend`  | Notebook  | `plt.legend()`  |
|  `matplotlib.pyplot.figure`  | Notebook  | `fig = plt.figure(figsize=(8, 4))`  |
|  `matplotlib.pyplot.subplot`  | Notebook  | `ax = plt.subplot(1, 2, 1)`  |
|  `matplotlib.pyplot.show`  | Notebook  | `plt.show()`  |

## Session 11 — Joining and combining data

Validate join keys and expected row counts so joins do not silently duplicate or discard records.

| Exact name | Origin | Code example |
|---|---|---|
|  `pandas.merge`  | Notebook  | `pd.merge(df, other, on="id", how="left", validate="one_to_one")`  |
|  `DataFrame.join`  | Notebook  | `df.set_index("id").join(other.set_index("id"), how="left", rsuffix="_right")`  |
|  `pandas.concat`  | Recommended  | `pd.concat([df, other], ignore_index=True)`  |
|  `DataFrame.merge`  | Recommended  | `df.merge(other, on="id", how="left", validate="one_to_one")`  |
|  `DataFrame.set_index`  | Notebook  | `indexed_df = df.set_index("id")`  |
|  `DataFrame.reset_index`  | Notebook  | `df.reset_index(drop=True)`  |
|  `DataFrame.duplicated`  | Recommended  | `duplicate_keys = df.duplicated(subset=["id"], keep=False)`  |
|  `DataFrame.merge(validate=...)`  | Recommended  | `df.merge(other, on="id", validate="one_to_one")`  |
|  `DataFrame.merge(indicator=True)`  | Recommended  | `df.merge(other, on="id", how="outer", indicator=True)`  |
|  `conact`  | Notebook — likely typo  | `pd.concat([df, other], ignore_index=True)  # Correct API; conact is not a pandas method.`  |

## Feature selection — supplementary notebook

Use feature selection after splitting data and within a cross-validation pipeline. Fit selectors on training data only.

| Exact name | Origin | Code example |
|---|---|---|
|  `sklearn.feature_selection.VarianceThreshold`  | Notebook  | `selector = VarianceThreshold(threshold=0.0)`  |
|  `VarianceThreshold.fit`  | Notebook  | `selector = VarianceThreshold().fit(X_train)`  |
|  `VarianceThreshold.get_support`  | Notebook  | `selected_mask = selector.get_support()`  |
|  `VarianceThreshold.get_feature_names_out`  | Notebook  | `selected_names = selector.get_feature_names_out(feature_names)`  |
|  `VarianceThreshold.get_params`  | Notebook  | `settings = selector.get_params()`  |
|  `sklearn.feature_selection.SelectKBest`  | Notebook  | `selector = SelectKBest(score_func=mutual_info_classif, k=1)`  |
|  `SelectKBest.fit`  | Notebook  | `selector = SelectKBest(score_func=mutual_info_classif, k=1).fit(X_train, y_train)`  |
|  `SelectKBest.get_support`  | Notebook  | `selected_mask = selector.get_support()`  |
|  `SelectKBest.get_feature_names_out`  | Notebook  | `selected_names = selector.get_feature_names_out(feature_names)`  |
|  `SelectKBest.get_params`  | Notebook  | `settings = selector.get_params()`  |
|  `sklearn.feature_selection.mutual_info_classif`  | Notebook  | `scores = mutual_info_classif(X_train, y_train, random_state=0)`  |
|  `sklearn.model_selection.train_test_split`  | Notebook  | `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)`  |
|  `statsmodels.api.OLS`  | Notebook  | `model = sm.OLS(y, sm.add_constant(X)).fit()`  |
|  `statsmodels.api.add_constant`  | Notebook  | `X_with_intercept = sm.add_constant(X)`  |
|  `statsmodels.OLS.fit`  | Recommended  | `results = sm.OLS(y, sm.add_constant(X)).fit()`  |
|  `RegressionResults.summary`  | Notebook  | `print(results.summary())`  |
|  `statsmodels.stats.outliers_influence.variance_inflation_factor`  | Notebook  | `vif = variance_inflation_factor(X_with_intercept.to_numpy(), 1)`  |
|  `sklearn.pipeline.Pipeline`  | Recommended  | `pipeline = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression())])`  |
|  `sklearn.model_selection.cross_validate`  | Recommended  | `cv_results = cross_validate(pipeline, X, y, cv=2, scoring="accuracy")`  |
|  `DataFrame.sort_values`  | Notebook  | `sorted_df = df.sort_values("value", ascending=False)`  |
|  `DataFrame.drop`  | Notebook  | `df.drop(columns=["target"])`  |
|  `Series.var`  | Notebook  | `variance = df["value"].var()`  |
|  `DataFrame.head`  | Notebook  | `df.head(5)`  |
|  `DataFrame.reset_index`  | Notebook  | `df.reset_index(drop=True)`  |
|  `list`  | Notebook — built-in  | `column_list = list(df.columns)`  |
|  `range`  | Notebook — built-in  | `positions = range(len(df))`  |
|  `zip`  | Notebook — built-in  | `pairs = list(zip(["A", "B"], [1, 2]))`  |
|  `extend`  | Notebook — list method  | `values_list.extend([3, 4])`  |

## Shared Python, NumPy, plotting, and inspection names

These notebook names also appear in examples or supporting code. Use them where appropriate, but prefer vectorized operations and logging in reusable code.

| Exact name | Origin | Code example |
|---|---|---|
|  `list.append`  | Notebook  | `values_list.append(3)`  |
|  `list.extend`  | Notebook  | `values_list.extend([3, 4])`  |
|  `len`  | Notebook — built-in  | `row_count = len(df)`  |
|  `list`  | Notebook — built-in  | `column_list = list(df.columns)`  |
|  `dict`  | Notebook — built-in  | `counts = {"A": 3, "B": 2}`  |
|  `type`  | Notebook — built-in  | `object_type = type(df)`  |
|  `print`  | Notebook — built-in  | `print(df.head())`  |
|  `range`  | Notebook — built-in  | `positions = range(len(df))`  |
|  `zip`  | Notebook — built-in  | `pairs = list(zip(["A", "B"], [1, 2]))`  |
|  `sorted`  | Notebook — built-in  | `ordered_values = sorted([3, 1, 2])`  |
|  `dir`  | Notebook — built-in  | `available_names = dir(df)`  |
|  `numpy.array`  | Notebook  | `array = np.array([1, 2, 3])`  |
|  `numpy.var`  | Notebook  | `variance = np.var(df["value"], ddof=1)`  |
|  `numpy.random.randint`  | Notebook  | `random_values = np.random.randint(0, 10, size=5)`  |
|  `os.getcwd`  | Notebook  | `working_directory = os.getcwd()`  |
|  `matplotlib.pyplot.bar`  | Notebook  | `plt.bar(category_counts.index.astype(str), category_counts.values)`  |
|  `matplotlib.pyplot.xlabel`  | Notebook  | `plt.xlabel("Category")`  |
|  `matplotlib.pyplot.ylabel`  | Notebook  | `plt.ylabel("Count")`  |
|  `matplotlib.pyplot.title`  | Notebook  | `plt.title("Value by category")`  |
|  `matplotlib.pyplot.savefig`  | Notebook  | `plt.savefig("chart.png", dpi=150, bbox_inches="tight")`  |
|  `matplotlib.pyplot.show`  | Notebook  | `plt.show()`  |

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
