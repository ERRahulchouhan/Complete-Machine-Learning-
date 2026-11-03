# Principal Component Analysis (PCA) — Interview Questions

Use these questions to practice concise answers, then explain the trade-offs and assumptions. Scenario prompts include a suggested reasoning path rather than a single universally correct answer.

## Conceptual questions

### Q1. What does PCA optimize?

**Suggested answer:** It finds orthogonal directions that maximize projected variance, equivalently minimizing rank-k squared reconstruction error for centered data. It captures variance, not necessarily task relevance.

### Q2. A team wants fewer features for a model and asks to keep 95% variance. What should you clarify?

**Suggested answer:** Explain that the threshold is a compression criterion, not a guarantee of predictive performance. Fit scaling/PCA inside training folds, choose components on validation utility, and assess interpretability and downstream metrics.

### Q3. How do you prevent data leakage in an unsupervised workflow?

**Suggested answer:** Split observations before fitting learned preprocessing or representations when evaluating generalization. Fit the scaler/reducer on training data only, then transform held-out data; never use reference labels to fit the unsupervised estimator.

## Scenario-based questions

### Scenario 1. One feature is measured in dollars and another in millimeters. What is the preprocessing concern?

**Suggested reasoning:** High-scale features can dominate components. Standardize when unit differences should not determine importance; do not standardize blindly if absolute variance is meaningful by design.

### Scenario 2. How would you explain a principal component to a non-technical stakeholder?

**Suggested reasoning:** It is a weighted combination of original features capturing a direction of variation. Review loadings and sign ambiguity; components are not automatically causal or directly interpretable.

### Scenario 3. How would you decide whether the result is useful to the business?

**Suggested reasoning:** Translate the goal into an observable downstream decision, review the result with domain experts, test stability, and measure downstream utility where possible. Internal scores alone do not establish value.

## Interview answer checklist

- State the algorithm's objective and assumptions.
- Explain preprocessing, parameter choices, and evaluation limits.
- Discuss stability, scale, and what happens to unseen data.
- Tie the result to a decision, monitoring plan, and operational constraints.
- Distinguish exploratory structure from causal or ground-truth claims.
