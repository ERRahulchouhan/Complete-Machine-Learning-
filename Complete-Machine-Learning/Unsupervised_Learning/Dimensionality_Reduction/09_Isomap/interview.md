# Isomap — Interview Questions

Use these questions to practice concise answers, then explain the trade-offs and assumptions. Scenario prompts include a suggested reasoning path rather than a single universally correct answer.

## Conceptual questions

### Q1. What is the central idea of Isomap?

**Suggested answer:** It approximates geodesic distances along a nearest-neighbor graph, then finds a low-dimensional embedding preserving those distances.

### Q2. A dataset lies on a curved manifold. Why might Isomap outperform linear PCA?

**Suggested answer:** If local neighborhoods form a connected graph that follows the manifold, graph shortest paths can approximate distances along the curve. Validate graph connectivity and neighborhood choices.

### Q3. How do you prevent data leakage in an unsupervised workflow?

**Suggested answer:** Split observations before fitting learned preprocessing or representations when evaluating generalization. Fit the scaler/reducer on training data only, then transform held-out data; never use reference labels to fit the unsupervised estimator.

## Scenario-based questions

### Scenario 1. The embedding contains disconnected or distorted regions. What do you check?

**Suggested reasoning:** Inspect n_neighbors, duplicate/noisy observations, scaling, and graph connectivity. Too few neighbors disconnects the graph; too many can create shortcuts across the manifold.

### Scenario 2. Can Isomap be deployed incrementally for each new row?

**Suggested reasoning:** The fitted sklearn estimator may support transform, but it relies on the learned neighborhood structure and can be sensitive to out-of-sample points. Test representative new data and monitor distances/embedding stability.

### Scenario 3. How would you decide whether the result is useful to the business?

**Suggested reasoning:** Translate the goal into an observable downstream decision, review the result with domain experts, test stability, and measure downstream utility where possible. Internal scores alone do not establish value.

## Interview answer checklist

- State the algorithm's objective and assumptions.
- Explain preprocessing, parameter choices, and evaluation limits.
- Discuss stability, scale, and what happens to unseen data.
- Tie the result to a decision, monitoring plan, and operational constraints.
- Distinguish exploratory structure from causal or ground-truth claims.
