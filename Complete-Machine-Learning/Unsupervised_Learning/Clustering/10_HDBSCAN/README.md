# HDBSCAN — Interview Questions

Use these questions to practice concise answers, then explain the trade-offs and assumptions. Scenario prompts include a suggested reasoning path rather than a single universally correct answer.

## Conceptual questions

### Q1. What does HDBSCAN provide beyond a single DBSCAN eps?

**Suggested answer:** It builds a hierarchy over density scales and selects stable clusters, often handling varying densities better. It also labels uncertain observations as noise; results still depend on min_cluster_size and min_samples.

### Q2. A fraud team wants clusters but also expects many legitimate outliers. How would you interpret noise?

**Suggested answer:** Treat noise as unassigned/low-density observations, not automatically as fraud. Validate noise rate and characteristics with labels or analyst review; density outliers are not equivalent to harmful events.

### Q3. How do you prevent data leakage in an unsupervised workflow?

**Suggested answer:** Split observations before fitting learned preprocessing or representations when evaluating generalization. Fit the scaler/reducer on training data only, then transform held-out data; never use reference labels to fit the unsupervised estimator.

## Scenario-based questions

### Scenario 1. Almost every observation is noise. What do you investigate?

**Suggested reasoning:** Check scaling, metric, min_cluster_size, min_samples, sample size, and whether the data has density structure at all. Compare parameter sensitivity and do not force clusters when evidence is weak.

### Scenario 2. How would you assign new observations with sklearn HDBSCAN?

**Suggested reasoning:** The estimator does not offer a general native predict operation in the standard workflow. Use a validated approximate-prediction extension if appropriate, or batch-refit under controlled versioning; communicate assignment uncertainty.

### Scenario 3. How would you decide whether the result is useful to the business?

**Suggested reasoning:** Translate the goal into an observable downstream decision, review the result with domain experts, test stability, and measure downstream utility where possible. Internal scores alone do not establish value.

## Interview answer checklist

- State the algorithm's objective and assumptions.
- Explain preprocessing, parameter choices, and evaluation limits.
- Discuss stability, scale, and what happens to unseen data.
- Tie the result to a decision, monitoring plan, and operational constraints.
- Distinguish exploratory structure from causal or ground-truth claims.
