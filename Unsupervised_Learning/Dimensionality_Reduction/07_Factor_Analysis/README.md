# Factor Analysis — Interview Questions

Use these questions to practice concise answers, then explain the trade-offs and assumptions. Scenario prompts include a suggested reasoning path rather than a single universally correct answer.

## Conceptual questions

### Q1. How does Factor Analysis differ from PCA?

**Suggested answer:** Factor Analysis models observed variables using latent factors plus feature-specific noise; PCA decomposes total variance without this explicit noise model.

### Q2. A survey team believes several questions reflect hidden traits. How would you use Factor Analysis?

**Suggested answer:** Check whether the measurement design and factor assumptions make sense, select factor count with domain theory and diagnostics, inspect loadings/residuals, and validate on a separate sample if possible.

### Q3. How do you prevent data leakage in an unsupervised workflow?

**Suggested answer:** Split observations before fitting learned preprocessing or representations when evaluating generalization. Fit the scaler/reducer on training data only, then transform held-out data; never use reference labels to fit the unsupervised estimator.

## Scenario-based questions

### Scenario 1. The estimated factors are difficult to interpret or unstable. What do you investigate?

**Suggested reasoning:** Review sample size, correlations, factor count, scaling, and model assumptions. Compare rotations or alternative factor counts and avoid assigning labels without subject-matter validation.

### Scenario 2. Can a factor be called the cause of the observed answers?

**Suggested reasoning:** No. A latent factor is a statistical representation under model assumptions, not proof of a causal construct. Causal claims need a suitable study design and external evidence.

### Scenario 3. How would you decide whether the result is useful to the business?

**Suggested reasoning:** Translate the goal into an observable downstream decision, review the result with domain experts, test stability, and measure downstream utility where possible. Internal scores alone do not establish value.

## Interview answer checklist

- State the algorithm's objective and assumptions.
- Explain preprocessing, parameter choices, and evaluation limits.
- Discuss stability, scale, and what happens to unseen data.
- Tie the result to a decision, monitoring plan, and operational constraints.
- Distinguish exploratory structure from causal or ground-truth claims.
