# Locally Linear Embedding (LLE) — Interview Questions

Use these questions to practice concise answers, then explain the trade-offs and assumptions. Scenario prompts include a suggested reasoning path rather than a single universally correct answer.

## Conceptual questions

### Q1. What does LLE preserve?

**Suggested answer:** It preserves local reconstruction weights: each point is approximated from neighbors, and the embedding seeks coordinates with similar local relationships.

### Q2. You need to unfold a smooth nonlinear surface. What assumptions and settings would you check?

**Suggested answer:** The data should lie on a sufficiently sampled smooth manifold; choose n_neighbors to capture local geometry without connecting distant regions. Check conditioning and neighborhood graph connectivity.

### Q3. How do you prevent data leakage in an unsupervised workflow?

**Suggested answer:** Split observations before fitting learned preprocessing or representations when evaluating generalization. Fit the scaler/reducer on training data only, then transform held-out data; never use reference labels to fit the unsupervised estimator.

## Scenario-based questions

### Scenario 1. Small changes in n_neighbors produce very different embeddings. What do you conclude?

**Suggested reasoning:** The manifold structure may be weak, undersampled, noisy, or poorly represented by the chosen distance metric. Report sensitivity and avoid treating one embedding as definitive.

### Scenario 2. The eigensolver fails or produces unstable results. What would you try?

**Suggested reasoning:** Check duplicates, rank, graph connectivity, and neighbor count; try another supported method/eigensolver, reduce dimensionality, and validate the resulting local-neighborhood preservation.

### Scenario 3. How would you decide whether the result is useful to the business?

**Suggested reasoning:** Translate the goal into an observable downstream decision, review the result with domain experts, test stability, and measure downstream utility where possible. Internal scores alone do not establish value.

## Interview answer checklist

- State the algorithm's objective and assumptions.
- Explain preprocessing, parameter choices, and evaluation limits.
- Discuss stability, scale, and what happens to unseen data.
- Tie the result to a decision, monitoring plan, and operational constraints.
- Distinguish exploratory structure from causal or ground-truth claims.
