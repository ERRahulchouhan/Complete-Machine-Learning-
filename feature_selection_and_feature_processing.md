# Feature Selection and Feature Processing for Machine Learning

This guide covers a practical end-to-end workflow for deciding which input variables to use and how to transform them for ML. Feature processing creates model-ready representations; feature selection chooses a useful subset. Both must be designed to avoid leakage and validated against the real deployment setting.

## 1. The end-to-end workflow

1. Define the prediction target, prediction time, unit of observation, and business objective.
2. Audit feature definitions, provenance, availability, permissions, and data types.
3. Remove target leakage and features unavailable at prediction time.
4. Split data using a strategy that reflects production (random, group, or temporal).
5. Build preprocessing in a fitted pipeline; fit transformations on training folds only.
6. Establish a baseline with a simple, defensible feature set.
7. Compare feature-processing and feature-selection choices using cross-validation.
8. Select the workflow using validation metrics, stability, latency, and business constraints.
9. Evaluate the selected workflow once on an untouched test set.
10. Persist the complete fitted preprocessing/model pipeline and feature schema.
11. Monitor production input quality, feature drift, missingness, and model performance.

Keep a record of the feature list, rationale, transformations, data version, split design, validation results, and artifact version.

## 2. Feature processing versus feature selection

- **Feature processing** transforms existing input data: imputation, scaling, encoding, text/vector transformations, date extraction, or dimensionality reduction.
- **Feature selection** chooses a subset of original features, such as selecting columns using domain knowledge, univariate scores, regularization, or recursive elimination.
- **Feature engineering** creates new inputs from existing information, such as a ratio, lag, interaction, or date part.

These are related but not interchangeable. Dimensionality reduction creates latent components; those components are not necessarily a selected subset of original features.

## 3. Start with feature and target audits

Create a data dictionary and check:

- Feature meaning, units, source, owner, and refresh frequency.
- Whether a value exists at the intended prediction time.
- Missing, invalid, duplicate, constant, and near-constant values.
- Identifier columns and high-cardinality categories.
- Target definition, target delay, and how labels are produced.
- Sensitive, regulated, or restricted attributes and permitted use.
- Train/serve schema consistency and data freshness.

Example audit:

```python
import pandas as pd

print(df.shape)
print(df.dtypes)
print(df.isna().mean().sort_values(ascending=False))
print("constant or single-value columns:")
print(df.nunique(dropna=False).loc[lambda values: values <= 1])
print("duplicate rows:", int(df.duplicated().sum()))
```

### Leakage checks

Look for:

- Features recorded after the target outcome or decision.
- Target-derived columns, post-event statuses, or future aggregates.
- Preprocessing statistics calculated from all rows before splitting.
- Duplicate entities or near-duplicate rows crossing train and test.
- Group, household, device, or patient information crossing a split when it should not.
- Time-based features that accidentally use the current/future target.

Do not remove a feature merely because it correlates strongly with the target. Verify when and how it is available and whether it is legitimately known at prediction time.

## 4. Split before learning preprocessing

Choose the split to represent deployment:

```python
# Independent and identically distributed rows (example)
from sklearn.model_selection import train_test_split

X = df.drop(columns="target")
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

For classification, `stratify=y` can preserve class proportions when appropriate. For repeated entities, use a group split; for future prediction, use a chronological split. Do not select a split method only because it yields a better score.

Use the training partition for feature discovery, transformation fitting, feature selection, and model selection. Use validation or cross-validation for choices. Reserve the test partition for final evaluation.

## 5. Missing values and invalid values

First determine why values are missing. Missingness may be random, caused by a data-collection process, or meaningful in itself.

Common approaches:

- Numeric: median or domain-informed imputation; sometimes mean for well-behaved distributions.
- Categorical: most-frequent or explicit `"Unknown"` category.
- Missingness indicator: useful when the fact that a value is absent carries information.
- Invalid values: validate against documented ranges and units; do not silently coerce and discard.
- Time series: forward/backward fill only if valid for the prediction-time information set.

Fit imputation in a pipeline so each validation fold learns imputation values from its training fold only.

## 6. Numeric feature processing

### Scaling

| Transformation | Typical use | Notes |
|---|---|---|
| Standardization (`StandardScaler`) | Linear models, SVM, KNN, PCA, neural nets | Centers and scales using training mean/std; sensitive to outliers. |
| Robust scaling (`RobustScaler`) | Numeric features with influential outliers | Uses median and quantiles; does not make bad records valid. |
| Min-max scaling (`MinMaxScaler`) | Bounded inputs, some neural-network workflows | Values can fall outside the training range at inference. |
| Max-absolute scaling (`MaxAbsScaler`) | Sparse data | Preserves zero entries. |
| No scaling | Many tree-based models | Still validate units and outliers; scaling may be needed for upstream steps. |

Tree models often do not need feature scaling, while distance-, margin-, and gradient-based methods often do. Keep scaling inside a pipeline.

### Skew, bounds, and outliers

Possible transforms include `log1p`, power transforms, clipping, or domain-specific unit conversions. Use only if valid for the feature and business meaning. Fit data-dependent thresholds on training data; do not clip rare but valid events simply to improve a metric.

## 7. Categorical feature processing

- **One-hot encoding:** good starting point for nominal categories with manageable cardinality.
- **Ordinal encoding:** use only when the category order has real meaning; arbitrary integer codes introduce a false order for many models.
- **Frequency/count encoding:** maps categories to counts/frequencies; fit mapping on training data and handle unseen values explicitly.
- **Target encoding:** uses target statistics and is leakage-prone; use cross-fitting within training folds and smoothing.
- **Native categorical support:** use only with a documented, correctly configured model implementation.
- **Rare categories:** group only with a training-fitted rule, and document the threshold and fallback.

For standard tabular ML, `OneHotEncoder(handle_unknown="ignore")` inside a `ColumnTransformer` is a reliable baseline. Never independently call `get_dummies` on train and test and assume their columns will match.

## 8. Dates, time series, text, and other data

### Dates and time series

Potential date features include year, month, day of week, hour, elapsed duration, and holiday flags. Use cyclical encodings such as sine/cosine for periodic variables when appropriate. For lag/rolling features:

- Sort by entity and timestamp.
- Shift before computing rolling statistics so the current target cannot leak.
- Split chronologically.
- Ensure historical windows do not cross entity boundaries.

### Text

Start with simple baselines such as word/character counts or TF-IDF. Fit vocabulary and IDF on training data only using a pipeline. For embeddings, record model/version and ensure inference uses the same tokenizer and model.

### Images, audio, and other modalities

Use domain-appropriate transformations and validate them with the same split strategy as the downstream model. Avoid augmentations or normalization that incorporate held-out data or label information.

## 9. Feature engineering

Feature engineering can add ratios, differences, interactions, domain rules, aggregates, date parts, and lag features. Before adding each feature, ask:

1. Is it available at prediction time?
2. Does it encode target or future information?
3. Is the calculation stable for missing, zero, extreme, or unseen inputs?
4. Can it be reproduced identically in training and serving?
5. Does it improve validation performance or decision quality?
6. Is the transformation understandable and monitorable?

Create reusable transformers for learned or complex operations. Avoid one-off notebook logic that cannot be reproduced in production.

## 10. Feature selection methods

### A. Domain and data-quality selection

Remove or review:

- Target leakage and post-outcome features.
- Unavailable, unauthorized, or unstable features.
- Duplicate columns and exact duplicates.
- Constant or near-constant columns.
- Identifiers that do not generalize (unless identity is genuinely predictive and permitted).
- Redundant variables when they harm stability, cost, or interpretability.

Domain-based selection should be documented and validated. Correlation thresholds are not universal removal rules.

### B. Filter methods

Filter methods score features independently of, or before fitting, a final model:

- Variance threshold for near-constant columns.
- Correlation or mutual information screens.
- ANOVA F-score or chi-square for appropriate classification data.
- F-test for regression.
- Univariate rank-based tests when their assumptions suit the data.

Filters are fast but can miss interactions and may keep redundant features. Any supervised filter must be fitted within each training fold, never on the full dataset.

Example selector in a pipeline:

```python
from sklearn.feature_selection import SelectPercentile, mutual_info_classif
from sklearn.pipeline import Pipeline

feature_selector = SelectPercentile(
    score_func=mutual_info_classif,
    percentile=50,
)

workflow = Pipeline([
    ("preprocess", preprocessor),
    ("select", feature_selector),
    ("model", classifier),
])
```

The scoring function must match the task and data representation. For classification with sparse one-hot features, consider selector compatibility and computational cost.

### C. Wrapper methods

- **Sequential Feature Selection:** adds or removes features based on cross-validated model score.
- **Recursive Feature Elimination (RFE):** repeatedly fits a model and removes lower-ranked features.
- **RFECV:** uses cross-validation to choose the number of retained features.

Wrappers can model interactions better than univariate filters but require many fits and can be unstable on small or correlated datasets. Run them inside training folds and budget compute.

### D. Embedded methods

- **L1 regularization (Lasso/logistic L1):** can set coefficients to zero; correlated features may be selected unpredictably.
- **Elastic Net:** combines L1 and L2 penalties and can be more stable for correlated inputs.
- **Tree-based importance:** useful for screening but can favor features with many split points and can spread importance across correlated features.
- **Regularized linear models or boosting:** selection is part of model fitting, but does not automatically produce causal or stable feature rankings.

Tune regularization or estimator parameters with cross-validation. Fit all model-based selection inside the pipeline.

### E. Dimensionality reduction is not feature selection

PCA, Truncated SVD, and other reducers create new component features. Use them when compression, denoising, or representation learning is appropriate. Original feature names are combined into latent dimensions, which can reduce direct interpretability. Fit the reducer inside the training pipeline.

## 11. Feature selection with cross-validation

Use a pipeline so preprocessing and selection are refit within each fold:

```python
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

selection_workflow = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("select", SelectFromModel(
        LogisticRegression(penalty="l1", solver="liblinear", C=0.1)
    )),
    ("model", LogisticRegression(max_iter=1000)),
])

results = cross_validate(
    selection_workflow,
    X_train_numeric,
    y_train,
    cv=5,
    scoring=["accuracy", "f1_macro"],
    return_train_score=True,
)
print("mean validation F1:", results["test_f1_macro"].mean())
```

For mixed numeric/categorical data, put a `ColumnTransformer` before the selector and ensure the selector/model support the transformed sparse/dense matrix. Inspect transformed feature names with `get_feature_names_out()` where available.

## 12. Comparing feature sets fairly

Compare candidate workflows using the same:

- Data split and folds.
- Target definition and metric.
- Preprocessing rules and leakage protections.
- Hyperparameter-search budget.
- Runtime/hardware conditions.

Include a simple baseline, a domain-selected feature set, and one or more selection/regularization alternatives. Prefer a simpler feature set when predictive performance is statistically/practically similar and it improves latency, reliability, privacy, cost, or interpretability.

Do not repeatedly inspect the final test set while choosing features. This turns the test set into part of training and makes the reported score optimistic.

## 13. Correlated and redundant features

Correlation can make coefficient estimates or selected feature lists unstable. It does not always harm prediction. Consider:

- Pairwise and domain-based redundancy review.
- Regularization, especially Elastic Net for correlated linear features.
- Grouping correlated features into meaningful aggregates when justified.
- Stability checks across folds and bootstrap samples.
- Reporting importance at a feature-group level.

Do not remove features solely because two columns are correlated without checking measurement meaning, model behavior, and deployment needs.

## 14. Class imbalance and feature selection

For classification, consider stratified/group/time-aware splits, per-class metrics, and appropriate class weighting or resampling. Any resampling must occur inside training folds, after splitting; never resample the full dataset before creating validation/test partitions. Feature selection must optimize the intended metric, not accuracy alone when classes are imbalanced.

## 15. Production-ready sklearn preprocessing pipeline

```python
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

numeric_columns = X_train.select_dtypes(include="number").columns
categorical_columns = X_train.select_dtypes(exclude="number").columns

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ("scale", StandardScaler()),
])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore")),
])
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns),
])

workflow = Pipeline([
    ("preprocess", preprocessor),
    # Optional: supervised selection happens inside each training fold.
    ("select", SelectPercentile(score_func=f_regression, percentile=80)),
    ("model", Ridge()),
])
workflow.fit(X_train, y_train)
predictions = workflow.predict(X_test)
```

The example is for regression and assumes the selector receives compatible transformed features. For classification use a classification score function; for tree models, scaling may be unnecessary. Tune the selection percentile inside cross-validation rather than choosing it from the test set.

## 16. Serving-time schema and artifact checks

Before deploying, persist the full fitted pipeline, expected input column names/order, package versions, and model metadata. Validate:

- Required fields are present and unexpected fields are handled by an explicit policy.
- Data types, units, nulls, finite values, and category handling match training.
- The same transformations run in batch and online inference.
- Feature order is stable, especially if converting DataFrames to arrays.
- The artifact can be loaded in a clean environment with pinned compatible dependencies.
- Pickle/joblib artifacts are loaded only from trusted sources.
- Predictions remain finite and within any documented output constraints.

## 17. Monitoring and feature governance

Monitor:

- Missing, invalid, out-of-range, and unseen-category rates.
- Feature distributions and drift against a reference population.
- Feature freshness and upstream data availability.
- Model performance when labels arrive, overall and across approved cohorts.
- Latency, throughput, memory, artifact version, and pipeline errors.
- Feature definitions, owners, permissions, and deprecation/change events.

Drift is a signal to investigate, not proof of quality degradation. Stable feature distributions also do not guarantee stable performance.

## 18. Common mistakes

- Selecting features using the full dataset before the split.
- Running feature selection outside cross-validation.
- Treating target correlation or feature importance as causal evidence.
- Using a random split for time-dependent or grouped records.
- Creating future-derived or post-outcome features.
- One-hot encoding train and test independently.
- Treating category integer codes as meaningful order without justification.
- Removing outliers or correlated features automatically.
- Choosing a feature set based only on the final test score.
- Saving only the estimator while omitting its fitted preprocessing.
- Assuming selection improves performance; it may reduce it.
- Ignoring operational cost, privacy, fairness, and data availability.

## 19. Interview and project checklist

Be ready to explain:

1. What is the target and when is a prediction made?
2. Which features are known at that moment, and how was that verified?
3. Which transformations are learned, and where are they fitted?
4. Why was each feature-selection method appropriate for the model/task?
5. How was selection nested inside cross-validation?
6. How stable is the selected set across folds or resamples?
7. What metric and validation split reflect production?
8. What happens when inputs are missing, malformed, or unseen?
9. How are feature changes monitored and governed?
10. Can the complete pipeline be reproduced and rolled back?

