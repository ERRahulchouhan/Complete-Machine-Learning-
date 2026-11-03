# K-means — Interview Questions

Use these questions to practice concise answers, then explain the trade-offs and assumptions. Scenario prompts include a suggested reasoning path rather than a single universally correct answer.

## Conceptual questions

### Q1. What objective does K-means optimize, and what does that imply about its clusters?

**Suggested answer:** It minimizes the sum of squared distances from observations to their assigned centroid. This favors compact, roughly spherical groups and can be distorted by outliers or features on larger scales.

### Q2. A product team asks you to segment customers but has no known segment count. How would you choose k?

**Suggested answer:** Start with the business decision and actionable segment requirements. Compare a small range of k using inertia/elbow, silhouette, stability across seeds or samples, and whether segments are interpretable. No single score determines the answer.

### Q3. How do you prevent data leakage in an unsupervised workflow?

**Suggested answer:** Split observations before fitting learned preprocessing or representations when evaluating generalization. Fit the scaler/reducer on training data only, then transform held-out data; never use reference labels to fit the unsupervised estimator.

## Scenario-based questions

### Scenario 1. Your clusters change substantially when you rerun training. What would you investigate?

**Suggested reasoning:** Check scaling, outliers, overlap, initialization, and whether k is appropriate. Increase n_init, compare label-invariant ARI across seeds/resamples, and avoid using unstable segments operationally.

### Scenario 2. A new customer arrives after the model is trained. How can you assign a segment?

**Suggested reasoning:** Apply the exact stored preprocessing and assign the customer to the nearest fitted centroid using predict. Monitor distances to centroids; unusually large distances may indicate an out-of-distribution customer.

### Scenario 3. How would you decide whether the result is useful to the business?

**Suggested reasoning:** Translate the goal into an observable downstream decision, review the result with domain experts, test stability, and measure downstream utility where possible. Internal scores alone do not establish value.

## Interview answer checklist

- State the algorithm's objective and assumptions.
- Explain preprocessing, parameter choices, and evaluation limits.
- Discuss stability, scale, and what happens to unseen data.
- Tie the result to a decision, monitoring plan, and operational constraints.
- Distinguish exploratory structure from causal or ground-truth claims.
