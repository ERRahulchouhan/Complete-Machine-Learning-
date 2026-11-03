# Charts for Machine Learning and Production

A practical catalog of charts used to explore data, evaluate models, explain predictions, and monitor production ML systems. No chart is universally required: choose one that answers a specific question, label it clearly, and show the data period, population, units, and sample size.

Examples use Matplotlib and scikit-learn, which are already used by this project. Seaborn and Plotly are optional alternatives for statistical and interactive dashboards.

## 1. Exploratory data analysis (EDA)

| Chart | Use it to answer | Notes |
|---|---|---|
| Histogram | What is the distribution of one numeric feature or target? | Check skew, multiple modes, clipping, and unusual tails. |
| KDE/density plot | How do smooth distributions compare? | Smoothing can hide small groups; do not treat it as observed counts. |
| Box plot | How do median, spread, and possible outliers compare across groups? | Box-plot outliers are flags for review, not automatic errors. |
| Violin plot | How do group distributions differ in shape? | Density width depends on smoothing and sample size. |
| Bar/count plot | How frequent are categories or classes? | Include rare and missing categories where relevant. |
| ECDF | What fraction of values is below a threshold? | Useful for percentiles and service-level thresholds. |
| Scatter plot | What is the relationship between two numeric features? | Overplotting can hide density; use transparency or hexbin for large data. |
| Hexbin/2D histogram | Where are observations concentrated in two dimensions? | Color represents count/density, not target value. |
| Pair plot/scatter matrix | What pairwise patterns exist across several features? | Use a representative sample for high row/column counts. |
| Correlation heatmap | Which numeric features move together linearly? | Correlation does not establish causality and may miss nonlinear relationships. |
| Missingness matrix/bar chart | Which fields are missing, and how much? | Missingness patterns can be informative; investigate the source. |
| Time-series line chart | How does a feature, target, or volume change over time? | Preserve time order; annotate events and data gaps. |

### Example: basic numeric EDA

```python
import matplotlib.pyplot as plt
import pandas as pd

numeric = df.select_dtypes(include="number")
numeric.hist(figsize=(12, 8), bins=30)
plt.tight_layout()
plt.show()

missing_rate = df.isna().mean().sort_values(ascending=False)
missing_rate[missing_rate > 0].plot(kind="bar", title="Missing-value rate by feature")
plt.ylabel("Fraction missing")
plt.tight_layout()
plt.show()

correlation = numeric.corr()
plt.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Pearson correlation")
plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=90)
plt.yticks(range(len(correlation.index)), correlation.index)
plt.title("Numeric feature correlation")
plt.tight_layout()
plt.show()
```

## 2. Classification evaluation

| Chart | Use it to answer | Notes |
|---|---|---|
| Confusion matrix | Which classes are confused? | Include class labels and counts; consider normalized rates for imbalanced classes. |
| ROC curve | How do true-positive and false-positive rates trade off across thresholds? | Can look optimistic with severe class imbalance; pair with PR analysis. |
| Precision-recall (PR) curve | How does precision trade off with recall? | Often more informative for rare positive classes. |
| Precision/recall/F1 by threshold | Which operating threshold fits business costs? | Select threshold on validation data, not the final test set. |
| Calibration/reliability curve | Do predicted probabilities match observed frequencies? | A well-ranked model may still have poorly calibrated probabilities. |
| Lift/gain/cumulative response | How much positive outcome is captured by reviewing the top-ranked fraction? | Define the ranking population and baseline clearly. |
| Class-support bar chart | How many examples support each reported class metric? | Essential context for small or rare classes. |

### Example: confusion matrix and ROC/PR curves

```python
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)

# Assumes a fitted classifier, X_test, and y_test.
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap="Blues")
plt.title("Test-set confusion matrix")
plt.tight_layout()
plt.show()

RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.title("ROC curve")
plt.tight_layout()
plt.show()

PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
plt.title("Precision-recall curve")
plt.tight_layout()
plt.show()
```

For multiclass tasks, state the averaging scheme (macro, weighted, or micro) and how the one-vs-rest curves are constructed. For threshold selection, use a validation set or cross-validation; keep the final test set for final evaluation.

## 3. Regression evaluation

| Chart | Use it to answer | Notes |
|---|---|---|
| Actual vs. predicted scatter | Are predictions aligned with observed values? | Add a y=x reference line; inspect systematic deviations. |
| Residuals vs. predicted | Are residuals biased or heteroscedastic? | Curvature or a funnel shape suggests model/data issues. |
| Residual histogram/KDE | Are errors centered and what is their shape? | Normal residuals are not required for every predictive use. |
| Residual Q-Q plot | How do residual quantiles compare with a normal reference? | Primarily useful for inference assumptions and tail diagnostics. |
| Error by segment/group | Does model quality differ across important slices? | Report sample counts and uncertainty for each group. |
| Prediction interval coverage/width | Are uncertainty intervals useful and empirically calibrated? | Evaluate on data not used to fit/calibrate intervals. |
| Metric over time | Is performance changing across periods? | Respect label delays and keep cohort definitions consistent. |

### Example: regression diagnostic plots

```python
import matplotlib.pyplot as plt
import numpy as np

# Assumes y_test and predictions are aligned and numeric.
residuals = np.asarray(y_test) - np.asarray(predictions)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].scatter(y_test, predictions, alpha=0.5)
limits = [
    min(np.min(y_test), np.min(predictions)),
    max(np.max(y_test), np.max(predictions)),
]
axes[0].plot(limits, limits, "r--")
axes[0].set(title="Actual vs. predicted", xlabel="Actual", ylabel="Predicted")

axes[1].scatter(predictions, residuals, alpha=0.5)
axes[1].axhline(0, color="red", linestyle="--")
axes[1].set(title="Residuals vs. predicted", xlabel="Predicted", ylabel="Residual")

axes[2].hist(residuals, bins=30)
axes[2].set(title="Residual distribution", xlabel="Residual", ylabel="Count")

plt.tight_layout()
plt.show()
```

## 4. Clustering and dimensionality reduction

| Chart | Use it to answer | Notes |
|---|---|---|
| Cluster scatter plot | How do assigned groups look in two dimensions? | Label axes with original or embedding dimensions; do not imply a plot proves true clusters. |
| Cluster-size bar chart | Are some clusters tiny, dominant, or empty? | Report the number of records and the noise/unassigned rate. |
| Silhouette plot | How are within-cluster cohesion and nearest-cluster separation distributed? | Requires at least two clusters; interpret with geometry and scale. |
| Elbow/inertia curve | How does K-means within-cluster inertia change with k? | The elbow can be ambiguous and is not a statistical guarantee. |
| Dendrogram | At what distances are hierarchical clusters merged? | Can be expensive and unreadable for very large datasets. |
| Reachability plot | What density valleys and separations does OPTICS reveal? | Extraction parameters still affect the final labels. |
| PCA explained-variance curve | How much variance do components explain cumulatively? | Variance retained does not guarantee downstream performance. |
| Embedding scatter plot | Are local neighborhoods visible in a low-dimensional representation? | t-SNE/UMAP-style global distances and apparent cluster sizes can mislead. |
| Cluster profile heatmap | How do standardized feature summaries differ by cluster? | Include cluster sizes; standardization hides raw units. |
| Stability chart | Does a clustering/embedding diagnostic remain similar across seeds or samples? | Use label-invariant metrics for partitions; align dimensions before comparing loadings. |

### Example: cluster profiles and sizes

```python
import matplotlib.pyplot as plt
import pandas as pd

# X is a DataFrame and labels contains one cluster label per row.
profile = X.assign(cluster=labels).groupby("cluster").mean(numeric_only=True)
sizes = pd.Series(labels).value_counts().sort_index()

sizes.plot(kind="bar", title="Rows by cluster")
plt.xlabel("Cluster")
plt.ylabel("Rows")
plt.tight_layout()
plt.show()

profile.plot(kind="bar", figsize=(12, 5))
plt.title("Mean feature values by cluster")
plt.ylabel("Feature mean (use standardized features for cross-feature comparison)")
plt.tight_layout()
plt.show()
```

If labels include `-1`, state whether it represents noise, an unknown group, or a real cluster according to the estimator.

## 5. Feature importance and model explanations

| Chart | Use it to answer | Notes |
|---|---|---|
| Permutation-importance bar chart | Which features reduce a chosen validation score when shuffled? | Correlated features can mask one another; calculate on held-out data. |
| Coefficient plot | What are the direction and magnitude of linear-model coefficients? | Compare standardized coefficients cautiously; coefficients are not causal effects. |
| Partial dependence plot | What is the average predicted response as a feature changes? | Correlated features can create unrealistic combinations. |
| ICE plot | How do individual predictions respond to a feature? | Many lines can clutter; sample carefully. |
| SHAP summary/beeswarm | How do feature contributions vary across observations? | Explanation quality depends on the background/distribution assumptions. |
| SHAP dependence plot | How does a feature contribution vary with its value or another feature? | Association is not causation. |
| Decision-tree plot | Which splits and rules define a small tree? | Deep trees become unreadable and can be unstable. |
| Error-by-feature plot | Where are errors larger across feature ranges? | Useful for detecting data gaps and subgroup failure. |

### Example: permutation importance

```python
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance

result = permutation_importance(
    model,
    X_validation,
    y_validation,
    n_repeats=10,
    random_state=42,
    scoring="neg_mean_squared_error",  # choose a metric suitable for the task
)
importance = pd.Series(result.importances_mean, index=X_validation.columns)
importance.sort_values().plot(kind="barh", title="Validation permutation importance")
plt.xlabel("Mean score decrease after shuffling")
plt.tight_layout()
plt.show()
```

## 6. Production monitoring charts

| Chart | Monitor | Questions it can reveal |
|---|---|---|
| Request volume over time | Service traffic | Is demand spiking, dropping, or failing by hour/day? |
| Latency percentile time series | p50/p90/p95/p99 response time | Is the service meeting its latency objectives? |
| Error/timeout rate | Failed requests and validation errors | Did a release, dependency, or upstream schema cause failures? |
| Feature distribution overlay | Training/reference vs. recent production | Are feature distributions shifting? |
| PSI/KS/drift metric trend | Quantified feature drift | Is change sustained and beyond an agreed threshold? |
| Prediction distribution over time | Score/label distribution | Did outputs shift after deployment or a data change? |
| Delayed-label performance trend | MAE/RMSE, precision/recall, etc. | Is model quality changing once ground truth arrives? |
| Calibration over time | Predicted probability vs observed rate | Are probabilities still trustworthy? |
| Missing/invalid feature rate | Data quality | Is an upstream source becoming incomplete or malformed? |
| Segment-level performance | Quality by product, region, cohort, or supported group | Are failures hidden by aggregate metrics? |
| Resource utilization | CPU, memory, GPU, queue depth | Is the service approaching capacity? |
| Rollout comparison | Canary vs. control outcomes | Is the new version safe to promote? |

Operational charts should have clear time windows, units, denominators, alert thresholds, and links to the model/data version. Avoid alerting on a noisy metric without minimum sample requirements or an action plan. Show delayed labels separately from immediate proxy metrics.

## 7. Data quality and drift charts

- **Numeric features:** overlay reference and recent histograms or ECDFs; show missingness and range violations.
- **Categorical features:** compare category-frequency bars, including unseen/other categories.
- **Target or label:** trend class balance or target quantiles; account for reporting delays.
- **Predictions:** monitor score quantiles, class proportions, and confidence distributions.
- **Subgroups:** facet key metrics by stable, approved cohorts; include counts and privacy safeguards.
- **Drift score:** plot a validated drift statistic over time with an operational threshold and a reference window.

Drift is a signal to investigate, not proof that model quality has degraded. Conversely, stable feature distributions do not guarantee stable performance.

## 8. Choosing a chart

1. Write down the question the chart should answer.
2. Choose a chart compatible with the variable types and sample size.
3. Use validation/test data for model evaluation; never tune repeatedly on final test charts.
4. Include baselines, sample counts, units, time windows, and uncertainty when relevant.
5. Use color accessibly and do not rely on color alone to encode meaning.
6. Avoid misleading axes, 3D effects, unnecessary dual axes, and unexplained smoothing.
7. For production dashboards, define the owner, alert threshold, and response action.
8. Use a table or metric when a chart adds no clarity.

## 9. Interactive charts and dashboard libraries

- **Matplotlib:** static plots and reports; included in this project.
- **Seaborn:** concise statistical plots; optional dependency.
- **Plotly:** interactive browser charts and dashboards; optional dependency.
- **Altair:** declarative charts; optional dependency.
- **Grafana/Prometheus or cloud monitoring tools:** service metrics, alerts, and operational dashboards.

Keep monitoring data access-controlled, avoid exposing sensitive records in dashboards, and aggregate or redact small groups where appropriate.
