# Scenario-Based Interview Questions: Tree and Boosting Algorithms

This guide focuses only on Decision Tree, Random Forest, Gradient Boosting, XGBoost, and AdaBoost. Each prompt includes a sample reasoning path; in an interview, state assumptions and trade-offs rather than memorizing a single answer.

> **Course note:** The repository currently has regression notebooks for Decision Tree, Random Forest, Gradient Boosting, and XGBoost. AdaBoost is included here for interview preparation but does not yet have its own notebook.

## Decision Tree

### Scenario 1: A small dataset needs a model that a business stakeholder can understand

**Question:** Would you start with a Decision Tree? How would you keep it interpretable?

**Suggested answer:** A shallow tree can be a reasonable, easy-to-explain baseline because its splits can be visualized and translated into rules. I would cap `max_depth`, set a minimum number of samples per leaf, and validate on held-out data. A deep tree can memorize a small dataset, so interpretability does not mean the unconstrained tree is trustworthy.

### Scenario 2: Training performance is excellent, but test performance is poor

**Question:** What is likely happening, and what would you change?

**Suggested answer:** The tree is likely overfitting. I would compare train and validation metrics, prune or constrain depth, increase `min_samples_leaf`, consider `ccp_alpha`, and select settings with cross-validation. I would also check for leakage and a train/test distribution mismatch.

### Scenario 3: A feature has many possible split values

**Question:** Could that feature become misleadingly important?

**Suggested answer:** Yes. A tree can choose a split that looks useful by chance, especially with many candidate values or high-cardinality identifiers. I would remove identifiers that do not generalize, validate feature importance on held-out data, and consider permutation importance rather than relying only on impurity importance.

### Scenario 4: The dataset has a few extreme numeric outliers

**Question:** How might they affect a regression tree?

**Suggested answer:** A regression tree is less sensitive to feature scaling than distance-based models, but extreme target values can affect split selection and leaf predictions, which are typically averages. I would inspect data quality, use robust evaluation, and consider target transformations or robust alternatives only when justified.

### Scenario 5: The product requires a prediction for every request, including missing or unexpected fields

**Question:** Is a tree alone enough to make this production-ready?

**Suggested answer:** No. I would put imputation and feature handling in a fitted pipeline, validate request schemas, define behavior for missing and out-of-range values, monitor error rates, and version the full pipeline. A model should not silently accept a changed feature schema.

## Random Forest

### Scenario 1: A single tree is unstable across train/test splits

**Question:** Why might Random Forest help, and what does it trade off?

**Suggested answer:** Random Forest averages many trees trained on bootstrap samples with feature subsampling. This usually reduces variance and improves stability compared with one deep tree. The trade-offs are more memory and inference cost and less straightforward global interpretation.

### Scenario 2: You have thousands of input features and want to reduce correlation between trees

**Question:** What Random Forest setting or behavior is relevant?

**Suggested answer:** Feature subsampling at each split helps decorrelate trees. I would tune `max_features` along with tree count and leaf constraints using validation. More trees can stabilize estimates but do not automatically fix poor data, leakage, or a biased target.

### Scenario 3: The forest has many trees, but predictions are still poor on a future time period

**Question:** Would adding more trees solve this?

**Suggested answer:** Probably not if the future period has shifted. More trees mainly reduce estimator variance; they do not address temporal leakage, changing relationships, unrepresentative training data, or missing predictive features. I would use a time-aware validation split and investigate drift.

### Scenario 4: A stakeholder asks which features drive predictions

**Question:** How would you explain Random Forest feature importance responsibly?

**Suggested answer:** I would use held-out permutation importance or carefully selected local explanations, and note that correlated features can share or obscure importance. Impurity-based importance can favor features with many possible split points. Importance is not causality.

### Scenario 5: The service has a strict latency and artifact-size limit

**Question:** How would you decide whether Random Forest is appropriate?

**Suggested answer:** Benchmark prediction latency, throughput, memory, and artifact size on representative hardware. Tune tree count, depth, and leaf size, and compare with a simpler model or a boosting model. Choose based on measured service objectives, not only offline accuracy.

## Gradient Boosting

### Scenario 1: A simple model underfits nonlinear interactions

**Question:** Why might Gradient Boosting be a good next experiment?

**Suggested answer:** Gradient Boosting builds trees sequentially, with later trees correcting errors from the current ensemble. It can capture nonlinear patterns and interactions. I would compare it with a baseline using cross-validation and constrain depth, learning rate, and number of estimators to control overfitting.

### Scenario 2: Training score keeps improving while validation score worsens

**Question:** What does this suggest, and how would you respond?

**Suggested answer:** The ensemble is overfitting. I would use early stopping on a validation set, reduce tree depth or leaf complexity, lower the learning rate with an appropriate estimator count, and check regularization and data leakage.

### Scenario 3: You have a modest dataset with noisy labels

**Question:** What risks should you consider before using Gradient Boosting?

**Suggested answer:** Sequential error correction can focus on hard or mislabeled observations. I would inspect label quality, use conservative tree complexity and learning rate, validate robustly, and compare against Random Forest or a regularized linear model.

### Scenario 4: The model is accurate but takes too long to train

**Question:** What would you profile or tune?

**Suggested answer:** Measure training time by feature count, row count, and tree settings. Limit depth/leaves, reduce redundant features, use early stopping, and consider histogram-based or optimized implementations such as HistGradientBoosting or XGBoost. Confirm that speed changes preserve validation quality.

### Scenario 5: The dataset arrives in time order

**Question:** How should you evaluate Gradient Boosting?

**Suggested answer:** Use a chronological split or rolling-origin validation that matches deployment. Random shuffling could leak future patterns into training. All preprocessing and early-stopping decisions must use training/validation periods, leaving the final test period untouched.

## XGBoost

### Scenario 1: You need a strong model for structured tabular data

**Question:** Why might you choose XGBoost, and what would you compare it against?

**Suggested answer:** XGBoost is a regularized, optimized gradient-boosted tree implementation with useful missing-value handling and support for efficient training. I would compare it to a simple baseline, Random Forest, and a standard Gradient Boosting model using the same leakage-safe splits and business-relevant metrics.

### Scenario 2: The model memorizes training data

**Question:** Which XGBoost controls would you investigate?

**Suggested answer:** I would tune tree complexity (`max_depth` or `max_leaves`), minimum child weight, `gamma`, row and column subsampling, and L1/L2 regularization. I would use early stopping against a validation set and keep the final test set untouched.

### Scenario 3: A team sees a large offline score but poor live results

**Question:** How would you investigate before changing hyperparameters?

**Suggested answer:** Verify that offline splitting reflects production, check point-in-time correctness and leakage, compare live feature distributions and missingness, confirm serving-time preprocessing and feature order, and evaluate delayed labels. The issue may be data or deployment mismatch rather than model capacity.

### Scenario 4: Training is too slow on a large tabular dataset

**Question:** What options could you assess?

**Suggested answer:** Profile first. Depending on the dataset and installed version, assess histogram-based tree construction, thread settings, subsampling, feature pruning, and GPU support. Benchmark cost and validation quality together; GPU use is not automatically faster for small datasets.

### Scenario 5: You are asked to tune many parameters using the test set

**Question:** How do you respond?

**Suggested answer:** I would not use the test set for repeated model selection. Use cross-validation or a dedicated validation set for search and early stopping, then evaluate the selected model once on the final holdout. This keeps the test estimate meaningful.

## AdaBoost

### Scenario 1: A weak learner performs slightly better than chance, and you want an ensemble

**Question:** How does AdaBoost combine weak learners?

**Suggested answer:** It fits learners sequentially and increases emphasis on observations that previous learners handled poorly, then combines learner predictions with weights. It can turn weak learners into a stronger ensemble, but is sensitive to noisy labels and outliers.

### Scenario 2: A few mislabeled observations dominate training

**Question:** What could happen with AdaBoost, and what would you do?

**Suggested answer:** AdaBoost may keep increasing focus on examples that cannot be correctly fit, including mislabeled points or outliers. I would audit labels, compare robust alternatives, constrain the base learner, and use validation that reflects the intended deployment data.

### Scenario 3: You need a simple, fast baseline on a small structured dataset

**Question:** When might AdaBoost be worth trying?

**Suggested answer:** It can be a useful lightweight ensemble, often with shallow decision stumps as base learners. I would compare it to a single tree and Random Forest using the same validation protocol and check whether its sequential training time is acceptable.

### Scenario 4: AdaBoost performs worse after adding many noisy features

**Question:** How would you diagnose this?

**Suggested answer:** Inspect feature quality and label noise, use a validation-based feature-selection or preprocessing strategy, tune learning rate and estimator count, and compare with ensembles that are less sensitive to hard examples. Do not select features based on the test set.

### Scenario 5: The project is a regression task with large outliers

**Question:** Is standard AdaBoost automatically a good choice?

**Suggested answer:** No. The appropriate AdaBoost variant and loss depend on the target and library implementation. Some regression variants are particularly sensitive to large errors. Verify the estimator’s objective, compare robust metrics and alternatives, and test on representative outlier conditions.

## Choosing among these algorithms

### Scenario: You must choose one model for a first production baseline

**Question:** How do you decide among a Decision Tree, Random Forest, Gradient Boosting, XGBoost, and AdaBoost?

**Suggested answer:** Start from data size, feature types, latency, interpretability, missingness, noise, and the cost of errors. Use a simple tree when rule transparency is central; Random Forest for a robust nonlinear baseline; Gradient Boosting or XGBoost for a strong tabular model when tuning and validation are available; AdaBoost when a simpler sequential ensemble is suitable and label noise is manageable. Compare them on the same leakage-safe validation protocol and production constraints. There is no universally best algorithm.

