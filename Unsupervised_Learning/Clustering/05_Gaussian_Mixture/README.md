# Gaussian Mixture Model — Interview Questions

Use these questions to practice concise answers, then explain the trade-offs and assumptions. Scenario prompts include a suggested reasoning path rather than a single universally correct answer.

## Conceptual questions

### Q1. What does a Gaussian mixture add compared with hard K-means labels?

**Suggested answer:** It models data as weighted Gaussian components and returns posterior membership probabilities. Assignments can express uncertainty, and covariance choices allow elliptical components; results rely on Gaussian-mixture assumptions.

### Q2. Customer A has probabilities 0.51 and 0.49 for two segments. How should the system use that result?

**Suggested answer:** Treat membership as uncertain. Consider retaining probabilities, escalating to a fallback, or avoiding a hard decision. Set thresholds based on decision costs and validate calibration/utility; do not present a near tie as certainty.

### Q3. How do you prevent data leakage in an unsupervised workflow?

**Suggested answer:** Split observations before fitting learned preprocessing or representations when evaluating generalization. Fit the scaler/reducer on training data only, then transform held-out data; never use reference labels to fit the unsupervised estimator.

## Scenario-based questions

### Scenario 1. How would you select the number of mixture components?

**Suggested reasoning:** Compare BIC/AIC, held-out log likelihood where appropriate, stability, component sizes, and domain usefulness. Information criteria are guides and can favor different counts; inspect degenerate or tiny components.

### Scenario 2. Training sometimes returns a singular covariance or unstable components. What do you try?

**Suggested reasoning:** Scale appropriately, inspect collinearity and sample size, use covariance regularization, try a simpler covariance type or initialization, and compare multiple seeds. Verify that improvements are stable rather than initialization artifacts.

### Scenario 3. How would you decide whether the result is useful to the business?

**Suggested reasoning:** Translate the goal into an observable downstream decision, review the result with domain experts, test stability, and measure downstream utility where possible. Internal scores alone do not establish value.

## Interview answer checklist

- State the algorithm's objective and assumptions.
- Explain preprocessing, parameter choices, and evaluation limits.
- Discuss stability, scale, and what happens to unseen data.
- Tie the result to a decision, monitoring plan, and operational constraints.
- Distinguish exploratory structure from causal or ground-truth claims.
