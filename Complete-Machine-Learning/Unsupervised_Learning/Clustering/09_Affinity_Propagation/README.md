# Affinity Propagation — Interview Questions

Use these questions to practice concise answers, then explain the trade-offs and assumptions. Scenario prompts include a suggested reasoning path rather than a single universally correct answer.

## Conceptual questions

### Q1. How does Affinity Propagation decide the number of clusters?

**Suggested answer:** It exchanges responsibility and availability messages between observations and selects exemplars. Preference strongly influences how many exemplars are chosen; damping can help convergence.

### Q2. A team wants representative products as exemplars without choosing k. When is this attractive?

**Suggested answer:** Affinity Propagation can select actual observations as exemplars and infer cluster count. It is most suitable for modest datasets with a meaningful similarity measure and enough memory.

### Q3. How do you prevent data leakage in an unsupervised workflow?

**Suggested answer:** Split observations before fitting learned preprocessing or representations when evaluating generalization. Fit the scaler/reducer on training data only, then transform held-out data; never use reference labels to fit the unsupervised estimator.

## Scenario-based questions

### Scenario 1. The run returns too many exemplars or does not converge. What do you adjust?

**Suggested reasoning:** Inspect preference, damping, similarity scaling, and convergence iterations. Run sensitivity checks and compare stability; changing preference can materially change cluster count.

### Scenario 2. Can it be applied unchanged to millions of records?

**Suggested reasoning:** Usually not: pairwise similarities require quadratic memory, which becomes prohibitive. Use sampling, a sparse/approximate method, or an algorithm designed for scale.

### Scenario 3. How would you decide whether the result is useful to the business?

**Suggested reasoning:** Translate the goal into an observable downstream decision, review the result with domain experts, test stability, and measure downstream utility where possible. Internal scores alone do not establish value.

## Interview answer checklist

- State the algorithm's objective and assumptions.
- Explain preprocessing, parameter choices, and evaluation limits.
- Discuss stability, scale, and what happens to unseen data.
- Tie the result to a decision, monitoring plan, and operational constraints.
- Distinguish exploratory structure from causal or ground-truth claims.
