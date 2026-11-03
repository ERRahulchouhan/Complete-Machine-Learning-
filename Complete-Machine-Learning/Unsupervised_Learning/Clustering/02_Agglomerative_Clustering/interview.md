# Agglomerative Clustering — Interview Questions

Use these questions to practice concise answers, then explain the trade-offs and assumptions. Scenario prompts include a suggested reasoning path rather than a single universally correct answer.

## Conceptual questions

### Q1. How do linkage choices change agglomerative clustering?

**Suggested answer:** Single linkage uses the nearest pair and may chain points; complete uses the farthest pair and favors compact groups; average uses mean pairwise distance; Ward merges to minimize within-cluster variance and is tied to Euclidean geometry.

### Q2. A stakeholder asks for segments at several levels of detail. How would you use this algorithm?

**Suggested answer:** Fit a hierarchy and inspect a dendrogram or cut the linkage tree at different cluster counts/distances. Choose a level using stability, interpretability, and the decision need rather than selecting a cut solely by appearance.

### Q3. How do you prevent data leakage in an unsupervised workflow?

**Suggested answer:** Split observations before fitting learned preprocessing or representations when evaluating generalization. Fit the scaler/reducer on training data only, then transform held-out data; never use reference labels to fit the unsupervised estimator.

## Scenario-based questions

### Scenario 1. The dataset has millions of rows. Is ordinary agglomerative clustering a good fit?

**Suggested reasoning:** Often not: pairwise distances and hierarchical updates can be memory- and compute-intensive. Consider sampling, connectivity constraints, BIRCH, or scalable alternatives; benchmark representative data before committing.

### Scenario 2. How would you assign a segment to a record arriving tomorrow?

**Suggested reasoning:** Many sklearn agglomerative estimators do not expose predict. You need a separately validated assignment rule, such as nearest training exemplar/cluster prototype, or a method with native prediction; do not refit the hierarchy per request.

### Scenario 3. How would you decide whether the result is useful to the business?

**Suggested reasoning:** Translate the goal into an observable downstream decision, review the result with domain experts, test stability, and measure downstream utility where possible. Internal scores alone do not establish value.

## Interview answer checklist

- State the algorithm's objective and assumptions.
- Explain preprocessing, parameter choices, and evaluation limits.
- Discuss stability, scale, and what happens to unseen data.
- Tie the result to a decision, monitoring plan, and operational constraints.
- Distinguish exploratory structure from causal or ground-truth claims.
