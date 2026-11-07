# Non-negative Matrix Factorization (NMF) — Interview Questions

Use these questions to practice concise answers, then explain the trade-offs and assumptions. Scenario prompts include a suggested reasoning path rather than a single universally correct answer.

## Conceptual questions

### Q1. What is the key input constraint for NMF?

**Suggested answer:** The input matrix must be non-negative. NMF approximates it as two non-negative lower-rank matrices, yielding additive component representations.

### Q2. You want to discover topics from TF-IDF data. How would you assess NMF components?

**Suggested answer:** Inspect high-weight terms per component, document-topic weights, topic coherence or human review, and stability across seeds. Tune component count; do not rely solely on reconstruction loss.

### Q3. How do you prevent data leakage in an unsupervised workflow?

**Suggested answer:** Split observations before fitting learned preprocessing or representations when evaluating generalization. Fit the scaler/reducer on training data only, then transform held-out data; never use reference labels to fit the unsupervised estimator.

## Scenario-based questions

### Scenario 1. Your standardized features contain negative values and NMF errors. What do you change?

**Suggested reasoning:** Use a transformation that preserves non-negativity, such as appropriate count/TF-IDF values or min-max scaling, only if semantically defensible. Do not shift data arbitrarily without considering what zeros and magnitudes mean.

### Scenario 2. Two runs produce different topics. How do you compare them?

**Suggested reasoning:** NMF can reach different local optima. Use deterministic initialization where possible, compare component alignment and reconstruction, and assess topic stability across seeds.

### Scenario 3. How would you decide whether the result is useful to the business?

**Suggested reasoning:** Translate the goal into an observable downstream decision, review the result with domain experts, test stability, and measure downstream utility where possible. Internal scores alone do not establish value.

## Interview answer checklist

- State the algorithm's objective and assumptions.
- Explain preprocessing, parameter choices, and evaluation limits.
- Discuss stability, scale, and what happens to unseen data.
- Tie the result to a decision, monitoring plan, and operational constraints.
- Distinguish exploratory structure from causal or ground-truth claims.
