# Truncated SVD — Interview Questions

Use these questions to practice concise answers, then explain the trade-offs and assumptions. Scenario prompts include a suggested reasoning path rather than a single universally correct answer.

## Conceptual questions

### Q1. Why is Truncated SVD often preferred to PCA for TF-IDF?

**Suggested answer:** It does not center the matrix, preserving sparse representation. Centering text matrices would generally make them dense and greatly increase memory.

### Q2. You have millions of sparse document vectors. What should guide component count?

**Suggested answer:** Measure memory/runtime and downstream retrieval or classification performance on held-out data, while reviewing explained variance as a diagnostic. More components are not automatically better.

### Q3. How do you prevent data leakage in an unsupervised workflow?

**Suggested answer:** Split observations before fitting learned preprocessing or representations when evaluating generalization. Fit the scaler/reducer on training data only, then transform held-out data; never use reference labels to fit the unsupervised estimator.

## Scenario-based questions

### Scenario 1. A colleague interprets SVD components exactly like PCA. What caveat do you give?

**Suggested reasoning:** Truncated SVD does not center the data, so its components and variance interpretation differ from centered PCA. Component signs are arbitrary and semantic interpretation requires inspecting feature loadings.

### Scenario 2. Sparse input unexpectedly becomes dense in the pipeline. What do you inspect?

**Suggested reasoning:** Look for centering or dense-only preprocessing, verify matrix formats at each step, and use sparse-compatible transformations and estimators.

### Scenario 3. How would you decide whether the result is useful to the business?

**Suggested reasoning:** Translate the goal into an observable downstream decision, review the result with domain experts, test stability, and measure downstream utility where possible. Internal scores alone do not establish value.

## Interview answer checklist

- State the algorithm's objective and assumptions.
- Explain preprocessing, parameter choices, and evaluation limits.
- Discuss stability, scale, and what happens to unseen data.
- Tie the result to a decision, monitoring plan, and operational constraints.
- Distinguish exploratory structure from causal or ground-truth claims.
