# Incremental PCA — Interview Questions

Use these questions to practice concise answers, then explain the trade-offs and assumptions. Scenario prompts include a suggested reasoning path rather than a single universally correct answer.

## Conceptual questions

### Q1. Why use Incremental PCA instead of PCA?

**Suggested answer:** It updates components from batches to reduce peak memory and support out-of-core processing. The result can differ slightly from batch PCA and depends on batch composition/order.

### Q2. Your matrix cannot fit in memory. How would you process it safely?

**Suggested answer:** Choose n_components and a batch size at least that large, fit batches from the training partition only, and use partial_fit consistently. Keep preprocessing and feature order fixed.

### Q3. How do you prevent data leakage in an unsupervised workflow?

**Suggested answer:** Split observations before fitting learned preprocessing or representations when evaluating generalization. Fit the scaler/reducer on training data only, then transform held-out data; never use reference labels to fit the unsupervised estimator.

## Scenario-based questions

### Scenario 1. Components vary when you change batch order. How do you respond?

**Suggested reasoning:** Check whether data order is representative, batches are sufficiently large, and scale is stable. Compare reconstruction/variance and downstream utility across batch orders; document any approximation sensitivity.

### Scenario 2. A single batch has fewer rows than n_components. What happens?

**Suggested reasoning:** The batch cannot support the requested components. Adjust batch size/component count or combine batches; validate the first and final batch sizes in the ingestion pipeline.

### Scenario 3. How would you decide whether the result is useful to the business?

**Suggested reasoning:** Translate the goal into an observable downstream decision, review the result with domain experts, test stability, and measure downstream utility where possible. Internal scores alone do not establish value.

## Interview answer checklist

- State the algorithm's objective and assumptions.
- Explain preprocessing, parameter choices, and evaluation limits.
- Discuss stability, scale, and what happens to unseen data.
- Tie the result to a decision, monitoring plan, and operational constraints.
- Distinguish exploratory structure from causal or ground-truth claims.
