# End-to-End Model Development Pipeline

A practical, reproducible workflow for developing, validating, deploying, and monitoring machine-learning models. The exact tools vary by project, but the principles—valid evaluation, leakage prevention, reproducibility, and operational readiness—apply broadly.

## Pipeline overview

1. Define the problem and success criteria.
2. Audit data and establish prediction-time correctness.
3. Create reproducible train, validation, and test partitions.
4. Build preprocessing and a simple baseline.
5. Train candidate models and tune using training-only validation.
6. Evaluate model quality, robustness, fairness, and operational cost.
7. Freeze the selected workflow and evaluate once on the final test set.
8. Package the complete preprocessing/model pipeline and metadata.
9. Deploy with input validation, monitoring, and rollback.
10. Monitor delayed outcomes and retrain only under a governed process.

## 1. Problem definition

Write down:

- The decision or workflow the model supports.
- The unit of prediction (person, transaction, device, document, time period).
- The target definition and how/when its label becomes available.
- The prediction timestamp and features available at that timestamp.
- The cost of false positives, false negatives, and prediction delays.
- The baseline decision or current process.
- The success metric and minimum acceptable threshold.
- Constraints: privacy, fairness, interpretability, latency, throughput, memory, and budget.

Avoid starting with an algorithm choice. First decide what an operationally useful and safe prediction means.

## 2. Data audit and provenance

Create a data dictionary containing feature name, meaning, type, unit, source, owner, refresh frequency, availability time, and permitted use. Review:

- Row grain and duplicate entities.
- Missing, invalid, stale, and out-of-range values.
- Class balance or target distribution.
- Identifiers and high-cardinality fields.
- Sensitive or restricted attributes.
- Data collection and selection bias.
- Label quality, delay, and censoring.
- Features calculated after the outcome or decision.
- Train/serve schema and source parity.

Keep the raw source immutable and record data version, query, extraction date, and preprocessing version. Use only data that is authorized for the purpose.

## 3. Leakage-safe data splitting

Choose splits to represent deployment:

- **Random split:** independent, identically distributed rows where no group/time dependency exists.
- **Stratified split:** classification when preserving class ratios is appropriate.
- **Group split:** related records (same person, household, site, or device) must stay together.
- **Time split:** predictions concern future time periods.
- **Spatial or site split:** deployment occurs in unseen locations or organizations.

Separate a final test set before model selection. Use cross-validation or a validation set within the remaining training data for feature selection, hyperparameter search, early stopping, and threshold selection. All learned preprocessing must be fitted inside each training fold.

## 4. Baseline

Create a baseline before complex models:

- Classification: majority-class or domain-rule baseline.
- Regression: mean/median prediction or a simple linear model.
- Ranking: existing policy or popularity baseline.
- Forecasting: last value, seasonal naive, or current planning process.

Compare against both a simple statistical baseline and the existing business process. If the proposed model cannot improve a meaningful decision metric, investigate data quality, target definition, and whether ML is needed.

## 5. Preprocessing and feature engineering

Typical stages:

1. Validate schema, types, units, and valid ranges.
2. Impute missing values and optionally add missingness indicators.
3. Encode categories with an explicit unknown-category policy.
4. Scale numeric features when required by the estimator.
5. Process dates, text, images, or sequences using task-appropriate transformations.
6. Create only prediction-time-available features.
7. Select features or reduce dimensions inside cross-validation.

For scikit-learn, put fitted transforms in a `Pipeline` and use `ColumnTransformer` for mixed data. Avoid fitting imputers, scalers, encoders, feature selectors, resampling methods, or target encoders on all rows before splitting.

## 6. Training pipeline example

The following is a template for tabular classification. Set `target_column` to the actual target and select a split strategy that matches deployment.

```python
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_parquet("data/training.parquet")
target_column = "target"

X = df.drop(columns=[target_column])
y = df[target_column]

# Use a random split only when it represents the deployment setting.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

numeric_columns = X_train.select_dtypes(include="number").columns
categorical_columns = X_train.select_dtypes(exclude="number").columns

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ("scaler", StandardScaler()),
])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns),
])

workflow = Pipeline([
    ("preprocess", preprocessor),
    ("model", LogisticRegression(max_iter=1000)),
])

# Search is evaluated only within training data.
search = GridSearchCV(
    workflow,
    param_grid={"model__C": [0.01, 0.1, 1.0, 10.0]},
    scoring="roc_auc",
    cv=5,
    n_jobs=1,
    refit=True,
)
search.fit(X_train, y_train)
selected_workflow = search.best_estimator_

print("best parameters:", search.best_params_)
print("best cross-validation ROC AUC:", search.best_score_)

# Final test is reserved for the selected workflow.
test_predictions = selected_workflow.predict(X_test)
print(classification_report(y_test, test_predictions))
if hasattr(selected_workflow, "predict_proba") and y_test.nunique() == 2:
    test_probabilities = selected_workflow.predict_proba(X_test)[:, 1]
    print("test ROC AUC:", roc_auc_score(y_test, test_probabilities))
```

For regression, replace the estimator, search scoring, and metrics with regression-appropriate choices such as MAE, RMSE, or R². For grouped/time-dependent data, replace the random split and CV splitter with group-aware or temporal splitters. Do not copy this example unchanged if its assumptions do not match the problem.

## 7. Model selection and tuning

Use a documented search budget and compare a few appropriate model families rather than trying every algorithm blindly. Tune only on training folds or validation data. Check:

- Mean and variability of cross-validation metrics.
- Train/validation gaps and learning curves.
- Stability across seeds or temporal folds.
- Calibration or decision thresholds where relevant.
- Subgroup and slice performance.
- Sensitivity to missingness, outliers, and distribution shift.
- Latency, throughput, memory, and artifact size.

Use nested cross-validation when an unbiased performance estimate is needed while performing extensive model selection on limited data.

## 8. Evaluation plan

Choose metrics to match decision costs:

- Classification: confusion matrix, precision, recall, F1, ROC AUC, PR AUC, log loss, calibration.
- Regression: MAE, RMSE, R², quantile/prediction interval metrics, residual checks.
- Ranking/recommendation: ranking metrics and online/user outcomes.
- Forecasting: horizon-specific errors, backtesting, and seasonal baselines.
- Clustering: internal diagnostics, stability, domain review, and downstream usefulness.

Report confidence intervals or fold variability where useful, class/group support, threshold, evaluation population, and test period. A single aggregate metric can hide serious subgroup or temporal failures.

## 9. Robustness, fairness, and risk review

Before release, assess:

- Performance by relevant, lawful, and privacy-approved subgroups.
- Missing, malformed, extreme, and out-of-distribution inputs.
- Sensitivity to random seed, data sample, and time period.
- Label noise and delayed/biased outcomes.
- Privacy, retention, access control, and model artifact exposure.
- Human review, appeal, or fallback for consequential decisions.
- Failure modes and safe behavior when dependencies/data sources fail.

Fairness metrics and acceptable thresholds are context-specific and should be agreed with affected stakeholders and legal/compliance experts.

## 10. Final test and model freeze

After feature engineering and model selection:

1. Freeze feature definitions, preprocessing, parameters, and decision threshold.
2. Confirm no decision was based on the test set.
3. Evaluate the selected workflow once on the test set.
4. Record metrics, confidence/variability, data range, and limitations.
5. If changes are made after seeing test results, obtain a new independent test set or describe the test result as no longer an unbiased final estimate.

## 11. Save a reproducible artifact

Save the full fitted preprocessing/model pipeline and metadata, not just the estimator:

```python
import json
import pickle
from pathlib import Path
import sklearn

artifact_dir = Path("artifacts/model_v1")
artifact_dir.mkdir(parents=True, exist_ok=True)

model_path = artifact_dir / "pipeline.pkl"
with model_path.open("wb") as file:
    pickle.dump(selected_workflow, file)

metadata = {
    "model_version": "model_v1",
    "target": target_column,
    "feature_columns": list(X_train.columns),
    "best_parameters": search.best_params_,
    "cross_validation_score": float(search.best_score_),
    "sklearn_version": sklearn.__version__,
    "artifact_path": str(model_path),
}
(artifact_dir / "metadata.json").write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8",
)
```

Pickle/joblib can execute code when loading. Only load artifacts from trusted, controlled storage. Pin compatible library versions and include artifact checksums and data/code version identifiers in a real registry workflow.

## 12. Inference and input validation

At inference:

- Validate required columns, types, units, allowed ranges, and batch size.
- Apply exactly the fitted preprocessing used at training.
- Handle unknown categories and missing data according to documented policy.
- Preserve feature names and schema; reject or explicitly handle unexpected fields.
- Check outputs for finite values and valid ranges.
- Record model version and request metadata while avoiding unnecessary sensitive data.
- Return clear client errors for invalid input and operational alerts for service failures.

Never refit preprocessing on an individual production request.

## 13. Deployment stages

1. **Offline validation:** reproducible training, tests, artifact load, and schema checks.
2. **Shadow mode:** produce predictions without influencing decisions; compare with current process.
3. **Canary:** serve a small controlled share and monitor safety and service metrics.
4. **Gradual rollout:** expand only if defined quality and operational gates pass.
5. **Full release:** keep previous artifact and configuration available for rollback.

Use readiness/health checks, resource limits, authentication and authorization, rate limiting, secure secrets, observability, and deployment approvals appropriate to the environment.

## 14. Production monitoring

Monitor two groups of signals:

**Service health**
- Request volume, latency percentiles, timeouts, error rates.
- CPU, memory, GPU, queue depth, and dependency availability.
- Input schema errors, missingness, invalid values, and unseen categories.

**Model and data health**
- Feature and prediction distributions compared to a versioned reference.
- Delayed-label performance, calibration, and subgroup metrics.
- Drift metrics and their sample counts/uncertainty.
- Business outcomes and fallback/human-review rates.

Drift is a prompt to investigate, not proof of quality degradation. Define alert thresholds, ownership, and actions before launch.

## 15. Retraining and rollback

Retrain only when there is a documented trigger, sufficient representative labels, approval, and a reproducible evaluation plan. Compare a candidate with the current production version on the same relevant data. Do not automatically promote a model solely because a scheduled training job completed.

Rollback should restore the model, preprocessing, configuration, and dependencies together. Test rollback before relying on it; record why it occurred and how recovery was verified.

## 16. Testing the pipeline

- Unit tests for deterministic feature transformations and schema validation.
- Data tests for types, ranges, missingness, category drift, and row grain.
- Training tests for reproducibility, successful fit, and finite metrics.
- Serialization tests that load the artifact and reproduce predictions.
- Batch/online parity tests for identical inputs.
- API contract and load tests for expected request volume and latency.
- Regression tests for known edge cases and safe failure behavior.

## 17. Experiment tracking and governance

Track:

- Code commit/version, configuration, random seeds, and dependency versions.
- Dataset snapshot/query/version and split definition.
- Feature list, transformations, and selection decisions.
- Parameters, cross-validation results, final metrics, and limitations.
- Artifact URI/checksum, approvals, deployment versions, and rollback history.
- Data/model owners, access permissions, retention, and change history.

An experiment tracker or model registry can automate this, but the records and review process matter more than a specific vendor.

## 18. Common pipeline failures

- Randomly splitting time-dependent or grouped records.
- Using test data for feature selection, tuning, or threshold setting.
- Fitting preprocessing before the split.
- Optimizing a metric unrelated to the business decision.
- Ignoring class imbalance, data quality, and label delays.
- Deploying a different feature transformation from training.
- Treating drift or feature importance as causal evidence.
- Shipping without a rollback plan or monitoring owner.
- Trusting serialized artifacts from unknown sources.
- Retraining and promoting automatically without evaluation gates.

## 19. Release checklist

- [ ] Problem, target, prediction time, users, and success criteria are documented.
- [ ] Data provenance, permissions, quality, and leakage checks are complete.
- [ ] Split/CV strategy reflects production entities and time.
- [ ] Preprocessing and feature selection are fitted inside the training pipeline.
- [ ] Baselines and candidate models are compared using suitable metrics.
- [ ] Test set was held out until the model and thresholds were frozen.
- [ ] Robustness, subgroup risks, and operational constraints are reviewed.
- [ ] Full pipeline, metadata, dependencies, and schema are versioned.
- [ ] Inference validation, monitoring, alert ownership, and fallback are defined.
- [ ] Canary/release and rollback procedures have been tested.

