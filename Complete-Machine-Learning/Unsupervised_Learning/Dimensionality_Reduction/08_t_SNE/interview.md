# t-SNE — Interview Questions

Use these questions to practice concise answers, then explain the trade-offs and assumptions. Scenario prompts include a suggested reasoning path rather than a single universally correct answer.

## Conceptual questions

### Q1. Which structure does t-SNE try hardest to preserve?

**Suggested answer:** Local neighborhood relationships. Global distances, relative cluster sizes, and gaps between plotted groups are not reliable quantitative comparisons.

### Q2. A plot shows three separated islands. Can you report three natural clusters?

**Suggested answer:** Not from the plot alone. t-SNE can visually separate groups as part of its embedding objective. Validate clusters in original space with independent methods and domain evidence.

### Q3. How do you prevent data leakage in an unsupervised workflow?

**Suggested answer:** Split observations before fitting learned preprocessing or representations when evaluating generalization. Fit the scaler/reducer on training data only, then transform held-out data; never use reference labels to fit the unsupervised estimator.

## Scenario-based questions

### Scenario 1. Two t-SNE plots differ across perplexities or random seeds. Which one is correct?

**Suggested reasoning:** Neither is automatically authoritative. Run multiple defensible settings, report stability and local diagnostics, and avoid interpreting global layout as ground truth.

### Scenario 2. A production user asks for coordinates for a new record. What is the issue?

**Suggested reasoning:** Standard sklearn t-SNE has no general out-of-sample transform. Use a method with transform support or a validated parametric/approximate approach; do not refit and compare coordinates independently for each request.

### Scenario 3. How would you decide whether the result is useful to the business?

**Suggested reasoning:** Translate the goal into an observable downstream decision, review the result with domain experts, test stability, and measure downstream utility where possible. Internal scores alone do not establish value.

## Interview answer checklist

- State the algorithm's objective and assumptions.
- Explain preprocessing, parameter choices, and evaluation limits.
- Discuss stability, scale, and what happens to unseen data.
- Tie the result to a decision, monitoring plan, and operational constraints.
- Distinguish exploratory structure from causal or ground-truth claims.
