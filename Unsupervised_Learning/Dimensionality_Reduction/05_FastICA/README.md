# FastICA — Interview Questions

Use these questions to practice concise answers, then explain the trade-offs and assumptions. Scenario prompts include a suggested reasoning path rather than a single universally correct answer.

## Conceptual questions

### Q1. How does ICA differ from PCA?

**Suggested answer:** PCA finds uncorrelated directions ordered by variance; ICA seeks statistically independent sources, often using non-Gaussianity. ICA component order and sign are arbitrary.

### Q2. A recording contains multiple mixed sources. What assumptions would you verify before trying ICA?

**Suggested answer:** Check that sources are sufficiently independent and non-Gaussian, the mixing is approximately linear and stationary, and observations/components are adequate. Evaluate recovered sources rather than assuming separation is correct.

### Q3. How do you prevent data leakage in an unsupervised workflow?

**Suggested answer:** Split observations before fitting learned preprocessing or representations when evaluating generalization. Fit the scaler/reducer on training data only, then transform held-out data; never use reference labels to fit the unsupervised estimator.

## Scenario-based questions

### Scenario 1. ICA fails to converge across runs. How would you debug?

**Suggested reasoning:** Standardize/whiten as appropriate, inspect rank and collinearity, reduce component count, raise max_iter, and compare initialization/algorithm settings. Treat convergence warnings as evidence to investigate.

### Scenario 2. The order and sign of sources change between runs. Is that a bug?

**Suggested reasoning:** Not necessarily. ICA components are identifiable only up to permutation and sign (and scale, depending on convention). Align components before comparing runs.

### Scenario 3. How would you decide whether the result is useful to the business?

**Suggested reasoning:** Translate the goal into an observable downstream decision, review the result with domain experts, test stability, and measure downstream utility where possible. Internal scores alone do not establish value.

## Interview answer checklist

- State the algorithm's objective and assumptions.
- Explain preprocessing, parameter choices, and evaluation limits.
- Discuss stability, scale, and what happens to unseen data.
- Tie the result to a decision, monitoring plan, and operational constraints.
- Distinguish exploratory structure from causal or ground-truth claims.
