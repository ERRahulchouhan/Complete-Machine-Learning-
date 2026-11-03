# BIRCH — Interview Questions

Use these questions to practice concise answers, then explain the trade-offs and assumptions. Scenario prompts include a suggested reasoning path rather than a single universally correct answer.

## Conceptual questions

### Q1. What is a clustering-feature tree in BIRCH used for?

**Suggested answer:** It incrementally summarizes subclusters with count, linear sum, and squared sum, compressing data before optional global clustering. Its threshold controls summary granularity.

### Q2. A sensor stream is too large to keep fully in memory. Why consider BIRCH?

**Suggested answer:** BIRCH can incrementally build compact summaries and supports partial_fit workflows. Verify that threshold and ordering produce useful summaries and that the deployment pattern matches the data stream.

### Q3. How do you prevent data leakage in an unsupervised workflow?

**Suggested answer:** Split observations before fitting learned preprocessing or representations when evaluating generalization. Fit the scaler/reducer on training data only, then transform held-out data; never use reference labels to fit the unsupervised estimator.

## Scenario-based questions

### Scenario 1. The number of BIRCH subclusters is much larger than expected. What parameter do you investigate?

**Suggested reasoning:** Inspect the threshold and feature scale first; a smaller threshold yields finer subclusters. Consider a final global clustering stage and validate that compression has not erased important structure.

### Scenario 2. What is a limitation of assigning unseen records?

**Suggested reasoning:** BIRCH supports prediction when configured with a global clustering stage, but behavior depends on that stage and fitted preprocessing. Persist the full pipeline and test edge cases/out-of-distribution inputs.

### Scenario 3. How would you decide whether the result is useful to the business?

**Suggested reasoning:** Translate the goal into an observable downstream decision, review the result with domain experts, test stability, and measure downstream utility where possible. Internal scores alone do not establish value.

## Interview answer checklist

- State the algorithm's objective and assumptions.
- Explain preprocessing, parameter choices, and evaluation limits.
- Discuss stability, scale, and what happens to unseen data.
- Tie the result to a decision, monitoring plan, and operational constraints.
- Distinguish exploratory structure from causal or ground-truth claims.
