# Kernel PCA — Interview Questions

Use these questions to practice concise answers, then explain the trade-offs and assumptions. Scenario prompts include a suggested reasoning path rather than a single universally correct answer.

## Conceptual questions

### Q1. How does Kernel PCA capture nonlinear structure?

**Suggested answer:** It applies PCA in an implicit feature space defined by a kernel matrix. Kernel choice and parameters such as RBF gamma control the geometry and can strongly affect the embedding.

### Q2. PCA shows no useful separation, but the data appears curved. What experiment would you run?

**Suggested answer:** Compare an RBF or polynomial Kernel PCA embedding with PCA using training-only fitting and local-neighborhood/reconstruction diagnostics where available. Tune kernel parameters and validate downstream utility.

### Q3. How do you prevent data leakage in an unsupervised workflow?

**Suggested answer:** Split observations before fitting learned preprocessing or representations when evaluating generalization. Fit the scaler/reducer on training data only, then transform held-out data; never use reference labels to fit the unsupervised estimator.

## Scenario-based questions

### Scenario 1. Kernel PCA memory becomes a bottleneck. Why?

**Suggested reasoning:** It typically constructs an n-by-n kernel matrix, giving quadratic memory behavior. Reduce sample size, use approximations, or choose incremental/linear methods when scale is important.

### Scenario 2. The deployed Kernel PCA artifact cannot invert transformed data. What might be missing?

**Suggested reasoning:** Inverse transformation must be fitted/configured explicitly (for sklearn, fit_inverse_transform=True and a suitable setting). Also verify that a transformer is needed at all; ordinary transform may be sufficient for deployment.

### Scenario 3. How would you decide whether the result is useful to the business?

**Suggested reasoning:** Translate the goal into an observable downstream decision, review the result with domain experts, test stability, and measure downstream utility where possible. Internal scores alone do not establish value.

## Interview answer checklist

- State the algorithm's objective and assumptions.
- Explain preprocessing, parameter choices, and evaluation limits.
- Discuss stability, scale, and what happens to unseen data.
- Tie the result to a decision, monitoring plan, and operational constraints.
- Distinguish exploratory structure from causal or ground-truth claims.
