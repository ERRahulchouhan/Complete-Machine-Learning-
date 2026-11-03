# DBSCAN — Interview Questions

Use these questions to practice concise answers, then explain the trade-offs and assumptions. Scenario prompts include a suggested reasoning path rather than a single universally correct answer.

## Conceptual questions

### Q1. Explain core, border, and noise points in DBSCAN.

**Suggested answer:** A core point has at least min_samples observations within eps; a border point is near a core but lacks that density itself; a noise point is not density-reachable from a core. Noise receives label -1.

### Q2. GPS coordinates are clustered, but eps was selected in raw degrees. What is wrong?

**Suggested answer:** Degrees are not uniform physical distances and longitude scale varies with latitude. Use an appropriate projected coordinate system or a geodesic-capable distance metric, scale/transform deliberately, and select eps in meaningful units.

### Q3. How do you prevent data leakage in an unsupervised workflow?

**Suggested answer:** Split observations before fitting learned preprocessing or representations when evaluating generalization. Fit the scaler/reducer on training data only, then transform held-out data; never use reference labels to fit the unsupervised estimator.

## Scenario-based questions

### Scenario 1. DBSCAN finds one large cluster and labels most rows noise. How do you debug it?

**Suggested reasoning:** Check feature scaling and distance metric, inspect k-distance plots, and test a defensible range of eps and min_samples. Consider whether density varies substantially; one global eps may not suit the data.

### Scenario 2. How do you serve DBSCAN for new observations?

**Suggested reasoning:** Standard DBSCAN has fit_predict but no native predict. Keep the fitted training data/core samples and implement a carefully validated neighborhood assignment, or use an algorithm with prediction support. Be explicit about uncertain/noise assignments.

### Scenario 3. How would you decide whether the result is useful to the business?

**Suggested reasoning:** Translate the goal into an observable downstream decision, review the result with domain experts, test stability, and measure downstream utility where possible. Internal scores alone do not establish value.

## Interview answer checklist

- State the algorithm's objective and assumptions.
- Explain preprocessing, parameter choices, and evaluation limits.
- Discuss stability, scale, and what happens to unseen data.
- Tie the result to a decision, monitoring plan, and operational constraints.
- Distinguish exploratory structure from causal or ground-truth claims.
