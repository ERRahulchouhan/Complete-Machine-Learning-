# OPTICS — Interview Questions

Use these questions to practice concise answers, then explain the trade-offs and assumptions. Scenario prompts include a suggested reasoning path rather than a single universally correct answer.

## Conceptual questions

### Q1. How does OPTICS differ from DBSCAN?

**Suggested answer:** OPTICS creates an ordering with reachability distances that exposes density structure across radii, rather than committing to one eps during fitting. Cluster extraction still depends on parameters and interpretation.

### Q2. You have groups with visibly different densities. Why might OPTICS help?

**Suggested answer:** Its reachability ordering can reveal structure at multiple density levels, making it easier to inspect than one DBSCAN radius. Validate extracted clusters and noise against domain expectations; varying density is not automatically solved.

### Q3. How do you prevent data leakage in an unsupervised workflow?

**Suggested answer:** Split observations before fitting learned preprocessing or representations when evaluating generalization. Fit the scaler/reducer on training data only, then transform held-out data; never use reference labels to fit the unsupervised estimator.

## Scenario-based questions

### Scenario 1. How would you explain a reachability plot to a product manager?

**Suggested reasoning:** Valleys suggest dense groups and peaks/separations may indicate boundaries, but the plot is a diagnostic rather than ground truth. Explain the extraction settings and show how conclusions change under reasonable parameter changes.

### Scenario 2. A production request needs an immediate cluster label. What limitation matters?

**Suggested reasoning:** OPTICS in sklearn does not generally provide a native inductive predict API. Define and validate a separate assignment method or use a prediction-capable model; do not refit OPTICS for every request.

### Scenario 3. How would you decide whether the result is useful to the business?

**Suggested reasoning:** Translate the goal into an observable downstream decision, review the result with domain experts, test stability, and measure downstream utility where possible. Internal scores alone do not establish value.

## Interview answer checklist

- State the algorithm's objective and assumptions.
- Explain preprocessing, parameter choices, and evaluation limits.
- Discuss stability, scale, and what happens to unseen data.
- Tie the result to a decision, monitoring plan, and operational constraints.
- Distinguish exploratory structure from causal or ground-truth claims.
