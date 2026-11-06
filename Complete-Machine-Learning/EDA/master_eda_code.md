# Master EDA methods: notebook and production reference

This guide combines the method names listed in [eda.md](./eda.md) with additional methods that are commonly useful in reliable data-analysis workflows. It is organized by EDA topic/session so you can find both the method and a runnable code example.

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

| Exact name | Code example |
|---|---|
|  `pandas.read_csv`  | `pd.read_csv("data.csv")`  |
|  `pandas.read_excel`  | `pd.read_excel("data.xlsx", sheet_name="Sheet1")`  |
|  `pandas.DataFrame`  | `pd.DataFrame({"category": ["A", "B"], "value": [10, 20]})`  |
|  `DataFrame.rename`  | `df.rename(columns={"category": "group"})`  |
|  `DataFrame.drop`  | `df.drop(columns=["target"])`  |
|  `DataFrame.to_csv`  | `df.to_csv("cleaned.csv", index=False)`  |
|  `DataFrame.to_excel`  | `df.to_excel("report.xlsx", index=False)`  |
|  `DataFrame.copy`  | `clean_df = df.copy()`  |
|  `DataFrame.shape`  | `row_count, column_count = df.shape`  |
|  `DataFrame.index`  | `row_labels = df.index`  |
|  `pandas.Series`  | `pd.Series([10, 20, 30], name="value")`  |
|  `DataFrame.columns`  | `column_names = df.columns.tolist()`  |
|  `DataFrame.dtypes`  | `column_types = df.dtypes`  |
|  `DataFrame.memory_usage`  | `memory_by_column = df.memory_usage(deep=True)`  |
|  `pandas.read_parquet`  | `pd.read_parquet("data.parquet")`  |
|  `DataFrame.to_parquet`  | `df.to_parquet("cleaned.parquet", index=False)`  |
|  `pandas.read_json`  | `pd.read_json("data.json")`  |
|  `pandas.read_sql`  | `pd.read_sql("SELECT 1 AS value", connection)`  |

## Session 2 — Basic data checks and data quality

Run these checks before analysis so missing values, duplicates, incorrect types, and unexpected dimensions are visible.

| Exact name | Code example |
|---|---|
|  `DataFrame.head`  | `df.head(5)`  |
|  `DataFrame.tail`  | `df.tail(5)`  |
|  `DataFrame.info`  | `df.info()`  |
|  `DataFrame.isna`  | `missing_mask = df.isna()`  |
|  `DataFrame.isnull`  | `missing_mask = df.isnull()`  |
|  `DataFrame.notna`  | `present_mask = df.notna()`  |
|  `DataFrame.isna().sum`  | `missing_per_column = df.isna().sum()`  |
|  `DataFrame.dropna`  | `df.dropna(subset=["value"])`  |
|  `DataFrame.fillna`  | `df.fillna({"value": df["value"].median()})`  |
|  `DataFrame.duplicated`  | `duplicate_keys = df.duplicated(subset=["id"], keep=False)`  |
|  `DataFrame.drop_duplicates`  | `df.drop_duplicates(subset=["id"], keep="first")`  |
|  `DataFrame.select_dtypes`  | `numeric_df = df.select_dtypes(include="number")`  |
|  `DataFrame.describe`  | `df.describe(include="all")`  |
|  `DataFrame.nunique`  | `unique_counts = df.nunique(dropna=False)`  |
|  `DataFrame.memory_usage`  | `memory_by_column = df.memory_usage(deep=True)`  |
|  `DataFrame.take`  | `df.take([0, 1])`  |
|  `DataFrame.to_list`  | `df["category"].to_list()`  |
|  `Series.to_dict`  | `s.value_counts().to_dict()`  |
|  `DataFrame.items`  | `column_types = {name: column.dtype for name, column in df.items()}`  |
|  `DataFrame.keys`  | `column_names = df.keys()`  |
|  `DataFrame.values`  | `array = df.values`  |
|  `DataFrame.to_numpy`  | `array = df[["value"]].to_numpy()`  |
|  `DataFrame.reset_index`  | `df.reset_index(drop=True)`  |
|  `DataFrame.index`  | `row_labels = df.index`  |
|  `len`  | `row_count = len(df)`  |
|  `type`  | `object_type = type(df)`  |
|  `print`  | `print(df.head())`  |

## Session 3 — Univariate categorical analysis

Use frequency tables and plots to understand category distribution, rare levels, and cardinality.

| Exact name | Code example |
|---|---|
|  `Series.value_counts`  | `category_counts = df["category"].value_counts(dropna=False)`  |
|  `Series.unique`  | `observed_categories = df["category"].unique()`  |
|  `Series.nunique`  | `distinct_count = df["category"].nunique()`  |
|  `Series.to_list`  | `category_list = df["category"].to_list()`  |
|  `dict`  | `counts = {"A": 3, "B": 2}`  |
|  `dict.keys`  | `keys = counts.keys()`  |
|  `dict.values`  | `values = counts.values()`  |
|  `dict.items`  | `items = counts.items()`  |
|  `matplotlib.pyplot.figure`  | `fig = plt.figure(figsize=(8, 4))`  |
|  `matplotlib.pyplot.bar`  | `plt.bar(category_counts.index.astype(str), category_counts.values)`  |
|  `matplotlib.axes.Axes.bar_label`  | `bars = ax.bar(["A", "B"], [3, 2]); ax.bar_label(bars)`  |
|  `seaborn.countplot`  | `sns.countplot(data=df, x="category")`  |
|  `matplotlib.pyplot.pie`  | `plt.pie(category_counts.values, labels=category_counts.index.astype(str), autopct="%.1f%%")`  |
|  `pandas.Series.plot`  | `category_counts.plot(kind="bar")`  |
|  `matplotlib.pyplot.title`  | `plt.title("Value by category")`  |
|  `matplotlib.pyplot.xlabel`  | `plt.xlabel("Category")`  |
|  `matplotlib.pyplot.ylabel`  | `plt.ylabel("Count")`  |
|  `matplotlib.pyplot.savefig`  | `plt.savefig("chart.png", dpi=150, bbox_inches="tight")`  |
|  `matplotlib.pyplot.show`  | `plt.show()`  |
|  `DataFrame.groupby`  | `grouped = df.groupby("category", dropna=False)`  |
|  `DataFrame.groupby().size`  | `group_sizes = df.groupby("category", dropna=False).size()`  |
|  `DataFrame.groupby().agg`  | `summary = df.groupby("category").agg(value_mean=("value", "mean"), row_count=("id", "count"))`  |
|  `DataFrame.pivot_table`  | `pivot = df.pivot_table(index="category", values="value", aggfunc="mean")`  |
|  `DataFrame.value_counts`  | `joint_counts = df[["category", "target"]].value_counts(dropna=False)`  |

## Session 4 — Univariate numerical analysis

Use summaries and distributions to understand scale, spread, skew, and unusual values.

| Exact name | Code example |
|---|---|
|  `Series.count`  | `non_missing_count = df["value"].count()`  |
|  `Series.mean`  | `value_mean = df["value"].mean()`  |
|  `Series.median`  | `value_median = df["value"].median()`  |
|  `Series.std`  | `value_std = df["value"].std()`  |
|  `Series.min`  | `value_min = df["value"].min()`  |
|  `Series.max`  | `value_max = df["value"].max()`  |
|  `Series.quantile`  | `q1, q3 = df["value"].quantile([0.25, 0.75])`  |
|  `numpy.percentile`  | `p95 = np.percentile(df["value"], 95)`  |
|  `numpy.quantile`  | `q95 = np.quantile(df["value"], 0.95)`  |
|  `numpy.mean`  | `average = np.mean(df["value"])`  |
|  `numpy.median`  | `middle = np.median(df["value"])`  |
|  `numpy.std`  | `spread = np.std(df["value"], ddof=1)`  |
|  `numpy.min`  | `smallest = np.min(df["value"])`  |
|  `numpy.max`  | `largest = np.max(df["value"])`  |
|  `builtins.min`  | `smallest = min([3, 1, 2])`  |
|  `builtins.max`  | `largest = max([3, 1, 2])`  |
|  `round`  | `rounded_value = round(3.14159, 2)`  |
|  `seaborn.displot`  | `sns.displot(data=df, x="value", kind="hist")`  |
|  `matplotlib.pyplot.hist`  | `plt.hist(df["value"].dropna(), bins=10)`  |
|  `matplotlib.pyplot.bar`  | `plt.bar(category_counts.index.astype(str), category_counts.values)`  |
|  `matplotlib.pyplot.subplot`  | `ax = plt.subplot(1, 2, 1)`  |
|  `DataFrame.describe`  | `df.describe(include="all")`  |
|  `Series.skew`  | `skewness = df["value"].skew()`  |
|  `Series.kurt`  | `tail_weight = df["value"].kurt()`  |
|  `numpy.isfinite`  | `finite_mask = np.isfinite(df["value"].to_numpy())`  |

## Session 5 — Outlier analysis

Identify potentially unusual observations, then investigate their cause before removing or changing them.

| Exact name | Code example |
|---|---|
|  `Series.quantile`  | `q1, q3 = df["value"].quantile([0.25, 0.75])`  |
|  `outliers`  | `outlier_mask = df["value"] > df["value"].quantile(0.95)  # Example threshold; choose a domain rule.`  |
|  `matplotlib.pyplot.boxplot`  | `plt.boxplot(df["value"].dropna())`  |
|  `matplotlib.pyplot.hist`  | `plt.hist(df["value"].dropna(), bins=10)`  |
|  `seaborn.boxplot`  | `sns.boxplot(data=df, x="category", y="value")`  |
|  `DataFrame.clip`  | `df["value"].clip(lower=0, upper=100)`  |
|  `numpy.percentile`  | `p95 = np.percentile(df["value"], 95)`  |
|  `numpy.quantile`  | `q95 = np.quantile(df["value"], 0.95)`  |
|  `len`  | `row_count = len(df)`  |
|  `format`  | `display_value = format(3.14159, ".2f")`  |

## Session 6 — Bivariate and multivariate analysis

Compare variables and groups, and inspect associations without treating correlation as causation.

| Exact name | Code example |
|---|---|
|  `pandas.crosstab`  | `pd.crosstab(df["category"], df["target"], normalize="index")`  |
|  `DataFrame.corr`  | `numeric_df = df.select_dtypes(include="number"); correlation = numeric_df.corr(method="spearman")`  |
|  `DataFrame.corrwith`  | `numeric_df = df.select_dtypes(include="number"); target_corr = numeric_df.corrwith(df["target"])`  |
|  `DataFrame.groupby`  | `grouped = df.groupby("category", dropna=False)`  |
|  `DataFrame.groupby().agg`  | `summary = df.groupby("category").agg(value_mean=("value", "mean"), row_count=("id", "count"))`  |
|  `seaborn.heatmap`  | `sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")`  |
|  `matplotlib.pyplot.scatter`  | `plt.scatter(df["value"], df["target"])`  |
|  `seaborn.scatterplot`  | `sns.scatterplot(data=df, x="value", y="target", hue="category")`  |
|  `seaborn.pairplot`  | `sns.pairplot(df, vars=["value", "target"], hue="category")`  |
|  `seaborn.countplot`  | `sns.countplot(data=df, x="category")`  |
|  `pandas.Series.plot`  | `category_counts.plot(kind="bar")`  |
|  `DataFrame.set_index`  | `indexed_df = df.set_index("id")`  |
|  `DataFrame.keys`  | `column_names = df.keys()`  |
|  `Series.unique`  | `observed_categories = df["category"].unique()`  |
|  `Series.value_counts`  | `category_counts = df["category"].value_counts(dropna=False)`  |
|  `matplotlib.pyplot.figure`  | `fig = plt.figure(figsize=(8, 4))`  |
|  `matplotlib.pyplot.subplot`  | `ax = plt.subplot(1, 2, 1)`  |
|  `matplotlib.pyplot.plot`  | `plt.plot([1, 2, 3], [2, 4, 3], label="series")`  |
|  `matplotlib.pyplot.title`  | `plt.title("Value by category")`  |
|  `range`  | `positions = range(len(df))`  |
|  `sorted`  | `ordered_values = sorted([3, 1, 2])`  |

## Session 7 — Categorical encoding

Encode categories only after deciding whether they are nominal, ordinal, or target labels. Learn encodings on training data and reuse them for validation and production data.

| Exact name | Code example |
|---|---|
|  `pandas.get_dummies`  | `encoded = pd.get_dummies(df, columns=["category"], dtype=int)`  |
|  `sklearn.preprocessing.LabelEncoder`  | `encoder = LabelEncoder()`  |
|  `LabelEncoder.fit_transform`  | `encoded_target = encoder.fit_transform(df["category"].astype(str))`  |
|  `LabelEncoder.inverse_transform`  | `labels = encoder.inverse_transform(encoded_target)`  |
|  `Series.map`  | `mapped = df["category"].map({"A": 0, "B": 1})`  |
|  `numpy.where`  | `flag = np.where(df["value"] > 20, "high", "low")`  |
|  `DataFrame.drop`  | `df.drop(columns=["target"])`  |
|  `sklearn.preprocessing.OneHotEncoder`  | `encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)`  |
|  `sklearn.preprocessing.OrdinalEncoder`  | `encoder = OrdinalEncoder(categories=[["low", "medium", "high"]], handle_unknown="use_encoded_value", unknown_value=-1)`  |
|  `sklearn.preprocessing.LabelEncoder`  | `encoder = LabelEncoder()`  |
|  `sklearn.compose.ColumnTransformer`  | `preprocessor = ColumnTransformer([("category", OneHotEncoder(handle_unknown="ignore"), ["category"])], remainder="passthrough")`  |
|  `sklearn.pipeline.Pipeline`  | `pipeline = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression())])`  |
|  `Series.nunique`  | `distinct_count = df["category"].nunique()`  |
|  `Series.unique`  | `observed_categories = df["category"].unique()`  |

## Session 8 — Scaling and normalization

Scale numeric features when the downstream algorithm is scale-sensitive. Fit the scaler on training data only.

| Exact name | Code example |
|---|---|
|  `sklearn.preprocessing.StandardScaler`  | `scaler = StandardScaler()`  |
|  `StandardScaler.fit_transform`  | `X_scaled = StandardScaler().fit_transform(X_train)`  |
|  `sklearn.preprocessing.MinMaxScaler`  | `scaler = MinMaxScaler(feature_range=(0, 1))`  |
|  `MinMaxScaler.fit_transform`  | `X_scaled = MinMaxScaler().fit_transform(X_train)`  |
|  `sklearn.preprocessing.RobustScaler`  | `scaler = RobustScaler()`  |
|  `sklearn.preprocessing.Normalizer`  | `normalizer = Normalizer(norm="l2")`  |
|  `sklearn.pipeline.Pipeline`  | `pipeline = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression())])`  |
|  `Series.mean`  | `value_mean = df["value"].mean()`  |
|  `Series.std`  | `value_std = df["value"].std()`  |
|  `Series.idxmin`  | `minimum_row = df["value"].idxmin()`  |
|  `Series.idxmax`  | `maximum_row = df["value"].idxmax()`  |
|  `matplotlib.pyplot.boxplot`  | `plt.boxplot(df["value"].dropna())`  |
|  `matplotlib.pyplot.hist`  | `plt.hist(df["value"].dropna(), bins=10)`  |

## Session 9 — Missing-value analysis and imputation

Measure missingness, understand why values are absent, and fit imputers on training data only.

| Exact name | Code example |
|---|---|
|  `DataFrame.isna`  | `missing_mask = df.isna()`  |
|  `DataFrame.isnull`  | `missing_mask = df.isnull()`  |
|  `DataFrame.isna().sum`  | `missing_per_column = df.isna().sum()`  |
|  `DataFrame.dropna`  | `df.dropna(subset=["value"])`  |
|  `DataFrame.fillna`  | `df.fillna({"value": df["value"].median()})`  |
|  `Series.mean`  | `value_mean = df["value"].mean()`  |
|  `Series.mode`  | `most_common = df["category"].mode(dropna=True)`  |
|  `sklearn.impute.SimpleImputer`  | `imputer = SimpleImputer(strategy="median")`  |
|  `sklearn.impute.KNNImputer`  | `imputer = KNNImputer(n_neighbors=3)`  |
|  `KNNImputer.fit_transform`  | `X_imputed = KNNImputer(n_neighbors=3).fit_transform(X_train)`  |
|  `sklearn.compose.ColumnTransformer`  | `preprocessor = ColumnTransformer([("category", OneHotEncoder(handle_unknown="ignore"), ["category"])], remainder="passthrough")`  |
|  `sklearn.pipeline.Pipeline`  | `pipeline = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression())])`  |
|  `DataFrame.to_csv`  | `df.to_csv("cleaned.csv", index=False)`  |
|  `warnings.filterwarnings`  | `warnings.filterwarnings("default")  # Keep warnings visible unless a specific warning is understood.`  |

## Session 10 — Transformations and distributions

Transformations can reduce skew or improve model fit, but should be selected based on data meaning and validated after fitting.

| Exact name | Code example |
|---|---|
|  `numpy.log`  | `logged_values = np.log1p(df["value"].clip(lower=0))`  |
|  `numpy.sqrt`  | `root_values = np.sqrt(df["value"].clip(lower=0))`  |
|  `numpy.reciprocal`  | `reciprocal_values = np.reciprocal(df["value"].replace(0, np.nan))`  |
|  `numpy.random.normal`  | `synthetic = np.random.normal(loc=0, scale=1, size=100)`  |
|  `numpy.random.exponential`  | `synthetic = np.random.exponential(scale=1, size=100)`  |
|  `numpy.random.randn`  | `synthetic = np.random.randn(100)`  |
|  `sklearn.preprocessing.PowerTransformer`  | `transformer = PowerTransformer(method="yeo-johnson")`  |
|  `PowerTransformer.fit`  | `transformer = PowerTransformer().fit(X_train)`  |
|  `PowerTransformer.transform`  | `X_transformed = transformer.transform(X_test)`  |
|  `PowerTransformer.fit_transform`  | `X_transformed = PowerTransformer().fit_transform(X_train)`  |
|  `numpy.ndarray.reshape`  | `column_array = np.array([1, 2, 3]).reshape(-1, 1)`  |
|  `seaborn.displot`  | `sns.displot(data=df, x="value", kind="hist")`  |
|  `seaborn.histplot`  | `sns.histplot(data=df, x="value", bins=10, kde=True)`  |
|  `seaborn.distplot`  | `sns.histplot(data=df, x="value", bins=10, kde=True)  # Replacement for deprecated distplot.`  |
|  `matplotlib.pyplot.hist`  | `plt.hist(df["value"].dropna(), bins=10)`  |
|  `matplotlib.pyplot.legend`  | `plt.legend()`  |
|  `matplotlib.pyplot.figure`  | `fig = plt.figure(figsize=(8, 4))`  |
|  `matplotlib.pyplot.subplot`  | `ax = plt.subplot(1, 2, 1)`  |
|  `matplotlib.pyplot.show`  | `plt.show()`  |

## Session 11 — Joining and combining data

Validate join keys and expected row counts so joins do not silently duplicate or discard records.

| Exact name | Code example |
|---|---|
|  `pandas.merge`  | `pd.merge(df, other, on="id", how="left", validate="one_to_one")`  |
|  `DataFrame.join`  | `df.set_index("id").join(other.set_index("id"), how="left", rsuffix="_right")`  |
|  `pandas.concat`  | `pd.concat([df, other], ignore_index=True)`  |
|  `DataFrame.merge`  | `df.merge(other, on="id", how="left", validate="one_to_one")`  |
|  `DataFrame.set_index`  | `indexed_df = df.set_index("id")`  |
|  `DataFrame.reset_index`  | `df.reset_index(drop=True)`  |
|  `DataFrame.duplicated`  | `duplicate_keys = df.duplicated(subset=["id"], keep=False)`  |
|  `DataFrame.merge(validate=...)`  | `df.merge(other, on="id", validate="one_to_one")`  |
|  `DataFrame.merge(indicator=True)`  | `df.merge(other, on="id", how="outer", indicator=True)`  |
|  `conact`  | `pd.concat([df, other], ignore_index=True)  # Correct API; conact is not a pandas method.`  |

## Feature selection — supplementary notebook

Use feature selection after splitting data and within a cross-validation pipeline. Fit selectors on training data only.

| Exact name | Code example |
|---|---|
|  `sklearn.feature_selection.VarianceThreshold`  | `selector = VarianceThreshold(threshold=0.0)`  |
|  `VarianceThreshold.fit`  | `selector = VarianceThreshold().fit(X_train)`  |
|  `VarianceThreshold.get_support`  | `selected_mask = selector.get_support()`  |
|  `VarianceThreshold.get_feature_names_out`  | `selected_names = selector.get_feature_names_out(feature_names)`  |
|  `VarianceThreshold.get_params`  | `settings = selector.get_params()`  |
|  `sklearn.feature_selection.SelectKBest`  | `selector = SelectKBest(score_func=mutual_info_classif, k=1)`  |
|  `SelectKBest.fit`  | `selector = SelectKBest(score_func=mutual_info_classif, k=1).fit(X_train, y_train)`  |
|  `SelectKBest.get_support`  | `selected_mask = selector.get_support()`  |
|  `SelectKBest.get_feature_names_out`  | `selected_names = selector.get_feature_names_out(feature_names)`  |
|  `SelectKBest.get_params`  | `settings = selector.get_params()`  |
|  `sklearn.feature_selection.mutual_info_classif`  | `scores = mutual_info_classif(X_train, y_train, random_state=0)`  |
|  `sklearn.model_selection.train_test_split`  | `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)`  |
|  `statsmodels.api.OLS`  | `model = sm.OLS(y, sm.add_constant(X)).fit()`  |
|  `statsmodels.api.add_constant`  | `X_with_intercept = sm.add_constant(X)`  |
|  `statsmodels.OLS.fit`  | `results = sm.OLS(y, sm.add_constant(X)).fit()`  |
|  `RegressionResults.summary`  | `print(results.summary())`  |
|  `statsmodels.stats.outliers_influence.variance_inflation_factor`  | `vif = variance_inflation_factor(X_with_intercept.to_numpy(), 1)`  |
|  `sklearn.pipeline.Pipeline`  | `pipeline = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression())])`  |
|  `sklearn.model_selection.cross_validate`  | `cv_results = cross_validate(pipeline, X, y, cv=2, scoring="accuracy")`  |
|  `DataFrame.sort_values`  | `sorted_df = df.sort_values("value", ascending=False)`  |
|  `DataFrame.drop`  | `df.drop(columns=["target"])`  |
|  `Series.var`  | `variance = df["value"].var()`  |
|  `DataFrame.head`  | `df.head(5)`  |
|  `DataFrame.reset_index`  | `df.reset_index(drop=True)`  |
|  `list`  | `column_list = list(df.columns)`  |
|  `range`  | `positions = range(len(df))`  |
|  `zip`  | `pairs = list(zip(["A", "B"], [1, 2]))`  |
|  `extend`  | `values_list.extend([3, 4])`  |

## Shared Python, NumPy, plotting, and inspection names

These notebook names also appear in examples or supporting code. Use them where appropriate, but prefer vectorized operations and logging in reusable code.

| Exact name | Code example |
|---|---|
|  `list.append`  | `values_list.append(3)`  |
|  `list.extend`  | `values_list.extend([3, 4])`  |
|  `len`  | `row_count = len(df)`  |
|  `list`  | `column_list = list(df.columns)`  |
|  `dict`  | `counts = {"A": 3, "B": 2}`  |
|  `type`  | `object_type = type(df)`  |
|  `print`  | `print(df.head())`  |
|  `range`  | `positions = range(len(df))`  |
|  `zip`  | `pairs = list(zip(["A", "B"], [1, 2]))`  |
|  `sorted`  | `ordered_values = sorted([3, 1, 2])`  |
|  `dir`  | `available_names = dir(df)`  |
|  `numpy.array`  | `array = np.array([1, 2, 3])`  |
|  `numpy.var`  | `variance = np.var(df["value"], ddof=1)`  |
|  `numpy.random.randint`  | `random_values = np.random.randint(0, 10, size=5)`  |
|  `os.getcwd`  | `working_directory = os.getcwd()`  |
|  `matplotlib.pyplot.bar`  | `plt.bar(category_counts.index.astype(str), category_counts.values)`  |
|  `matplotlib.pyplot.xlabel`  | `plt.xlabel("Category")`  |
|  `matplotlib.pyplot.ylabel`  | `plt.ylabel("Count")`  |
|  `matplotlib.pyplot.title`  | `plt.title("Value by category")`  |
|  `matplotlib.pyplot.savefig`  | `plt.savefig("chart.png", dpi=150, bbox_inches="tight")`  |
|  `matplotlib.pyplot.show`  | `plt.show()`  |

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
