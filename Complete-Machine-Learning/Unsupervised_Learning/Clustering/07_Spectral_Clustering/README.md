# Spectral Clustering — Interview Questions

Use these questions to practice concise answers, then explain the trade-offs and assumptions. Scenario prompts include a suggested reasoning path rather than a single universally correct answer.

## Conceptual questions

### Q1. Why can spectral clustering separate non-convex shapes that K-means struggles with?

**Suggested answer:** It builds a similarity graph and uses eigenvectors of a graph Laplacian to embed connected structure before clustering. Results depend on graph construction and the chosen number of clusters.

### Q2. You need to segment a small dataset shaped like two interlocking rings. What would you configure and validate?

**Suggested answer:** Try a nearest-neighbor affinity graph and an appropriate neighbor count, then fit with the expected cluster count only if justified. Visualize graph connectivity and compare stability; avoid relying on synthetic labels in a real task.

### Q3. How do you prevent data leakage in an unsupervised workflow?

**Suggested answer:** Split observations before fitting learned preprocessing or representations when evaluating generalization. Fit the scaler/reducer on training data only, then transform held-out data; never use reference labels to fit the unsupervised estimator.

## Scenario-based questions

### Scenario 1. The graph has disconnected components or results change with n_neighbors. What does that signal?

**Suggested reasoning:** Affinity construction may be too sparse/dense or poorly matched to data geometry. Inspect connectivity and tune neighbors/affinity using domain-informed sensitivity checks.

### Scenario 2. Can you directly use the fitted spectral model to label one new point?

**Suggested reasoning:** Common sklearn spectral clustering configurations do not provide a general inductive predict method. Build and validate an out-of-sample extension or choose a model with native prediction support.

### Scenario 3. How would you decide whether the result is useful to the business?

**Suggested reasoning:** Translate the goal into an observable downstream decision, review the result with domain experts, test stability, and measure downstream utility where possible. Internal scores alone do not establish value.

## Interview answer checklist

- State the algorithm's objective and assumptions.
- Explain preprocessing, parameter choices, and evaluation limits.
- Discuss stability, scale, and what happens to unseen data.
- Tie the result to a decision, monitoring plan, and operational constraints.
- Distinguish exploratory structure from causal or ground-truth claims.
