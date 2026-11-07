# EDA and Machine Learning Interview Questions

## Exploratory Data Analysis (EDA)

1. What is exploratory data analysis, and what are its main objectives?
   **Answer:** EDA is the systematic examination of data using summaries and visualizations to understand its structure, quality, patterns, relationships, and anomalies before formal modeling.
2. How does EDA differ from confirmatory data analysis?
   **Answer:** EDA is exploratory and generates hypotheses; confirmatory analysis tests prespecified hypotheses using statistical methods and assumptions.
3. What steps do you follow when beginning an analysis of an unfamiliar dataset?
   **Answer:** Clarify the question and data provenance, inspect dimensions and types, check quality and missingness, summarize variables, visualize distributions and relationships, and record findings.
4. How do you inspect a dataset's structure, dimensions, and data types?
   **Answer:** Check row and column counts, sample records, column names, inferred types, unique counts, summary statistics, and whether types match their meanings.
5. What is the difference between a population and a sample?
   **Answer:** A population is the complete group of interest; a sample is the subset observed and used to estimate population properties.
6. How do you distinguish numerical, categorical, ordinal, and datetime variables?
   **Answer:** Numerical values support arithmetic; categorical values denote groups; ordinal categories have meaningful order; datetime values represent points or intervals in time.
7. What are univariate, bivariate, and multivariate analyses?
   **Answer:** Univariate analysis studies one variable, bivariate analysis studies a pair, and multivariate analysis examines three or more variables together.
8. Which summary statistics do you examine first for a numerical variable, and why?
   **Answer:** Inspect count, missingness, mean, median, standard deviation, quartiles, minimum, and maximum to assess scale, spread, skew, and possible anomalies.
9. How do mean, median, and mode differ, and when is each most informative?
   **Answer:** The mean is the arithmetic average and is sensitive to extremes; the median is the midpoint and is robust; the mode is the most frequent value or category.
10. What do variance and standard deviation measure?
   **Answer:** Both quantify dispersion around the mean; variance is in squared units, while standard deviation is in the variable's original units.
11. What are quartiles, percentiles, and the interquartile range?
   **Answer:** Percentiles divide ordered observations into 100-position thresholds; quartiles are the 25th, 50th, and 75th percentiles; IQR is Q3 minus Q1.
12. How can skewness and kurtosis help describe a distribution?
   **Answer:** Skewness describes asymmetry; kurtosis describes tail heaviness relative to a reference distribution. Both are summaries, not substitutes for plots.
13. How do you interpret a histogram, and how can bin size affect its appearance?
   **Answer:** A histogram shows frequencies across numerical intervals. Too few bins hide structure; too many exaggerate noise, so compare reasonable bin widths.
14. When would you use a box plot, and what can it reveal?
   **Answer:** Use it to compare medians, quartiles, spread, and potential outliers across groups; flagged points require investigation, not automatic removal.
15. What is a density plot, and how does it differ from a histogram?
   **Answer:** A density plot is a smoothed estimate of a continuous distribution; unlike histogram counts, its area integrates to one and its shape depends on bandwidth.
16. How can a Q-Q plot help assess whether data follow a distribution?
   **Answer:** It compares sample quantiles with theoretical quantiles. Approximate linearity supports the chosen distribution; systematic curvature suggests departures.
17. How do you examine the frequency distribution of a categorical variable?
   **Answer:** Tabulate counts and proportions, include missing or unknown categories explicitly, and plot ordered bars while checking rare or inconsistent labels.
18. What is a contingency table, and how can you use it in EDA?
   **Answer:** It cross-tabulates counts for combinations of categorical variables; row or column proportions help compare distributions between groups.
19. How do you choose an appropriate visualization for a specific data question?
   **Answer:** Match the chart to variable types and the comparison: distributions, trends, relationships, or composition; prioritize legibility and avoid unnecessary decoration.
20. What are common ways to visualize relationships between two numerical variables?
   **Answer:** Use scatterplots, hexbin or binned plots for dense data, and optionally correlation summaries; inspect nonlinearity, clusters, and changing variance.
21. How would you explore the relationship between a numerical and a categorical variable?
   **Answer:** Compare grouped box or violin plots, distributions, and summary statistics; show sample sizes and consider uncertainty and confounding variables.
22. How would you explore the relationship between two categorical variables?
   **Answer:** Use cross-tabulated counts or normalized proportions, stacked or grouped bars, and an appropriate association test when inference is needed.
23. What does a correlation matrix show, and what does it not show?
   **Answer:** It summarizes pairwise association, usually linear correlation. It does not establish causality, capture all nonlinear relationships, or reveal confounding by itself.
24. How do Pearson, Spearman, and Kendall correlation differ?
   **Answer:** Pearson measures linear association; Spearman correlates ranks and captures monotonic association; Kendall measures rank concordance and is often robust for ordinal data.
25. Why does correlation not imply causation?
   **Answer:** Association may result from confounding, reverse causality, selection, or coincidence. Causal claims require a suitable design and explicit assumptions.
26. What is Simpson's paradox, and how can stratified analysis help reveal it?
   **Answer:** An aggregate association can reverse or disappear within subgroups because subgroup composition differs. Compare relevant strata and explain which estimate answers the question.
27. How do you identify duplicate rows, and how do you decide whether to remove them?
   **Answer:** Check exact and key-based duplicates, then inspect source and business rules. Remove only records that are genuinely duplicated, not valid repeated events.
28. How do you detect missing values, and how do you measure their extent?
   **Answer:** Count nulls by column and row, calculate proportions, visualize patterns, and check sentinel values such as blank strings or coded placeholders.
29. What are MCAR, MAR, and MNAR missing-data mechanisms?
   **Answer:** MCAR is unrelated to observed or unobserved values; MAR depends on observed data; MNAR depends on the missing value or unobserved information.
30. How do you decide whether to drop rows or columns with missing values?
   **Answer:** Consider the missingness mechanism, amount, feature importance, sample size, and bias risk; compare dropping with justified imputation and report the choice.
31. What are the advantages and risks of imputing missing values with the mean or median?
   **Answer:** These methods are simple and preserve row count, but can reduce variance and distort relationships; median is more robust to skew and outliers.
32. When might you use forward-fill, backward-fill, or interpolation for missing data?
   **Answer:** They can suit ordered time series when values evolve smoothly or persist, but must respect time order and avoid using future information improperly.
33. How can a missingness indicator feature be useful?
   **Answer:** It lets a model learn that absence itself may carry information, especially when missingness is systematic; validate it without leaking target information.
34. How do you identify invalid, inconsistent, or out-of-range values?
   **Answer:** Apply documented domain constraints, type and format checks, allowed-value lists, cross-field rules, and targeted review of violations with the data owner.
35. What are common methods for detecting outliers?
   **Answer:** Use plots, IQR fences, robust z-scores, domain limits, or multivariate methods; choose based on distribution and treat detections as candidates.
36. How does the IQR rule flag potential outliers?
   **Answer:** It commonly flags values below Q1 − 1.5×IQR or above Q3 + 1.5×IQR. This is a screening convention, not proof of error.
37. How can z-scores be used to identify unusual observations, and what assumptions do they make?
   **Answer:** A z-score measures standard deviations from the mean; large absolute scores may be unusual. Classical cutoffs work best for roughly normal data and are outlier-sensitive.
38. Why should an outlier not be removed without investigating its cause?
   **Answer:** It may be a valid rare event central to the problem. Removal can bias estimates, hide risk, or discard an important population segment.
39. How do you distinguish a data-entry error from a valid extreme observation?
   **Answer:** Check source records, units, validation rules, related fields, and domain context; correct only with evidence and preserve an audit trail.
40. What is the difference between univariate and multivariate outlier detection?
   **Answer:** Univariate methods assess unusual values in one feature; multivariate methods identify unusual combinations that may look normal feature by feature. Examples include robust covariance distances and density-based approaches; their usefulness depends on sample size, scaling, and dimensionality.
41. How do you detect inconsistent category labels such as variations in capitalization or spelling?
   **Answer:** Inspect distinct values and frequencies, normalize safe formatting differences, map known aliases using explicit rules, and review ambiguous mappings before changing them.
42. How do you handle inconsistent units or measurement scales in a dataset?
   **Answer:** Identify units from metadata or domain checks, convert values to a common documented unit, and validate ranges before analysis.
43. How do you identify potential data leakage during EDA?
   **Answer:** Trace feature timing and provenance, look for post-outcome or target-derived fields, and verify that every feature would be available at prediction time.
44. Why is it important to inspect the target variable before modeling?
   **Answer:** Its distribution, quality, prevalence, scale, and definition determine the problem type, split strategy, useful metrics, and potential imbalance or label errors.
45. How do you examine class imbalance in a classification dataset?
   **Answer:** Report class counts and proportions overall and by relevant splits or groups, then compare against the task's error costs and baseline performance.
46. What visualizations help assess a target variable's relationship with input features?
   **Answer:** Use grouped distributions, box or bar plots, scatterplots, and correlation summaries as appropriate; inspect patterns on training data without inferring causality.
47. How can grouping and aggregation uncover patterns in data?
   **Answer:** Summaries by meaningful segments reveal differences obscured by individual records, but small group sizes and confounding can make aggregates misleading.
48. What is a pivot table, and how can it support exploratory analysis?
   **Answer:** A pivot table summarizes a measure across row and column categories, making segment comparisons and patterns easier to inspect.
49. How do you analyze data collected over time?
   **Answer:** Sort by time, validate timestamps and intervals, plot the series, inspect trend, seasonality, shifts, and missing periods, and preserve chronological order.
50. What is seasonality, and how can you distinguish it from a trend?
   **Answer:** Seasonality is a repeating pattern at a known period; a trend is a longer-term direction. Decomposition or seasonal plots can separate them.
51. How can you detect gaps, duplicate timestamps, or irregular intervals in time-series data?
   **Answer:** Compare timestamps with the expected frequency, count duplicates, and inspect time deltas; handle each according to whether it reflects collection or business rules.
52. What is a cohort analysis, and what kinds of questions can it answer?
   **Answer:** It groups entities by a shared starting event or period and tracks outcomes over time, such as retention, repeat purchase, or engagement.
53. What is a segmented analysis, and why might an overall average be misleading?
   **Answer:** It compares defined subgroups. An aggregate can mask variation or reverse subgroup patterns when group sizes or compositions differ.
54. How can you use pair plots, and when do they become impractical?
   **Answer:** Pair plots show pairwise feature distributions and relationships; they are useful for a few variables but cluttered and expensive for many.
55. How do heatmaps help explore relationships or missingness?
   **Answer:** Color encodes values in a matrix, such as correlations or missingness patterns. Include a clear scale and avoid implying precision beyond the data.
56. What is a scatterplot matrix, and what limitations does it have?
   **Answer:** It displays pairwise scatterplots among variables, often with diagonal distributions. It becomes cluttered for many features and can obscure overplotting.
57. How do you choose transformations for heavily skewed numerical variables?
   **Answer:** Inspect the distribution and model needs; consider log, square-root, or power transforms, handle zero or negative values appropriately, and validate on training data.
58. What is a log transformation, and when can it be inappropriate?
   **Answer:** It maps positive values to logarithms, compressing large values and sometimes reducing right skew. It is unsuitable without adjustment for nonpositive values and complicates interpretation.
59. What is the difference between normalization and standardization?
   **Answer:** Normalization often rescales values to a bounded range; standardization subtracts the mean and divides by standard deviation. Terminology can vary, so state the formula.
60. How can you assess whether a numerical feature needs scaling?
   **Answer:** Consider the algorithm: distance-, gradient-, and regularization-based methods often benefit, while tree splits are generally scale-insensitive. Fit scaling on training data only.
61. What is multicollinearity, and how might you detect it during EDA?
   **Answer:** It is strong dependence among predictors, which can destabilize coefficient estimates. Inspect correlations, variance inflation factors, and coefficient stability.
62. How can high-cardinality categorical features complicate analysis?
   **Answer:** They create sparse encodings, many rare levels, and overfitting or privacy risks. Group rare categories or use carefully validated encodings.
63. How do you investigate whether a dataset is representative of the population of interest?
   **Answer:** Compare sampling and coverage with the target population across important characteristics, identify exclusions and response bias, and consult collection documentation.
64. What are sampling bias and survivorship bias, and how might they appear in a dataset?
   **Answer:** Sampling bias arises when inclusion is systematically unrepresentative; survivorship bias excludes failures or dropouts, making observed outcomes look overly favorable.
65. How can data collection processes introduce bias into an analysis?
   **Answer:** Selection criteria, measurement devices, survey wording, missing responses, and operational changes can produce systematic differences unrelated to the phenomenon of interest.
66. How do you document assumptions, data-quality issues, and decisions made during EDA?
   **Answer:** Keep a reproducible notebook or report with data sources, definitions, checks, exclusions, transformations, assumptions, findings, and unresolved limitations.
67. What makes an EDA visualization clear, accurate, and accessible?
   **Answer:** Use an appropriate chart, labeled axes and units, readable text, color-independent cues, honest scales, and concise context such as sample size.
68. How do you avoid misleading visualizations through axis choices, scales, or binning?
   **Answer:** Use scales suited to the comparison, disclose transformations, choose defensible bins, show baselines when relevant, and avoid truncation that exaggerates differences.
69. How do you decide when EDA is sufficient to move on to modeling?
   **Answer:** Proceed when the target and data quality are understood, major leakage and sampling risks are addressed, and a defensible validation plan is in place.
70. How would you summarize the most important findings from an EDA for a non-technical audience?
   **Answer:** Lead with the decision-relevant patterns, explain their size and uncertainty in plain language, show a few clear visuals, and state caveats and next steps.

## Machine Learning Foundations and Data Preparation

71. What is machine learning, and how does it differ from traditional rule-based programming?
   **Answer:** Machine learning fits patterns from examples to make predictions or decisions; rule-based programming encodes behavior explicitly. Both require validation and appropriate inputs.
72. What are the main differences between supervised, unsupervised, and reinforcement learning?
   **Answer:** Supervised learning uses labeled examples; unsupervised learning finds structure without labels; reinforcement learning learns actions through rewards and interaction.
73. What is the difference between classification and regression?
   **Answer:** Classification predicts discrete classes or class probabilities; regression predicts continuous numerical quantities.
74. What are features, labels, parameters, and hyperparameters?
   **Answer:** Features are inputs, labels are target outcomes, parameters are learned from training data, and hyperparameters configure the learning process.
75. What is the difference between training, validation, and test datasets?
   **Answer:** Training data fits parameters; validation data guides model or hyperparameter choices; test data estimates final generalization after choices are complete.
76. Why should the test set be kept separate until final evaluation?
   **Answer:** Repeated test-set use turns it into an implicit tuning set, making its score optimistic and no longer an independent final estimate.
77. What is a baseline model, and why should you establish one?
   **Answer:** A baseline is a simple reference, such as majority class or mean prediction. It checks the pipeline and establishes a minimum useful performance level.
78. What is the bias-variance tradeoff?
   **Answer:** High bias reflects overly simple assumptions; high variance reflects sensitivity to training data. The goal is to balance both for good generalization.
79. What is underfitting, and how can you recognize it?
   **Answer:** Underfitting occurs when a model cannot capture relevant patterns; it typically performs poorly on both training and validation data.
80. What is overfitting, and how can you recognize it?
   **Answer:** Overfitting occurs when a model learns noise or peculiarities of training data; training performance is strong but validation or test performance is much worse.
81. How do model complexity and dataset size affect overfitting?
   **Answer:** More flexible models can fit more patterns and noise; additional representative data often reduces variance, though it does not fix leakage or poor labels.
82. What is regularization, and how does it help generalization?
   **Answer:** Regularization penalizes model complexity or large parameters during fitting, discouraging overly complex solutions and often reducing variance.
83. What is the difference between L1 and L2 regularization?
   **Answer:** L1 penalizes absolute coefficient values and can set some to zero; L2 penalizes squared values and typically shrinks coefficients smoothly.
84. What is cross-validation, and why is it useful?
   **Answer:** Cross-validation repeatedly trains and evaluates on different partitions to estimate performance more reliably and support model selection when data are limited.
85. How does k-fold cross-validation work?
   **Answer:** Divide data into k folds, train on k−1 and validate on the remaining fold, repeat for each fold, then summarize the scores.
86. When should you use stratified cross-validation?
   **Answer:** Use it for classification when preserving approximate class proportions in each fold is important, especially with imbalanced labels.
87. What is a time-series split, and why can ordinary random splitting be invalid for temporal data?
   **Answer:** A time-series split trains on earlier observations and validates on later ones. Random splitting can expose future information and misrepresent deployment.
88. What is data leakage, and what are common ways it occurs?
   **Answer:** Leakage is use of information unavailable at prediction time or from validation/test data; examples include target-derived features and preprocessing fitted on all data.
89. How can preprocessing before splitting cause data leakage?
   **Answer:** Statistics such as imputation values, scales, or selected features then incorporate held-out data. Split first and fit transformations on each training fold only.
90. How do you build a preprocessing and modeling pipeline to prevent leakage?
   **Answer:** Put imputation, encoding, scaling, feature selection, and the estimator in one pipeline; cross-validate the entire pipeline so each transform fits only its training fold.
91. How do you handle categorical features for machine learning?
   **Answer:** Choose an encoding suited to category order, cardinality, and algorithm; fit it on training data and handle unseen categories explicitly.
92. What is one-hot encoding, and what are its limitations?
   **Answer:** It creates a binary column per category. It is interpretable but can be wide and sparse for high-cardinality features and may need unknown-level handling.
93. What is ordinal encoding, and when is it appropriate?
   **Answer:** It maps categories to integer codes; use it when categories have a meaningful order, not for nominal categories where numbers imply false ordering.
94. How can target encoding cause leakage, and how can you reduce that risk?
   **Answer:** Encoding categories using their target averages can reveal labels, especially for rare categories. Use out-of-fold or smoothed encodings fitted within training folds.
95. How do you encode high-cardinality categorical variables?
   **Answer:** Options include frequency, hashing, or regularized target encoding; group rare levels where appropriate and validate encoding within folds to prevent leakage.
96. Which feature-scaling methods do you know, and when would you use each?
   **Answer:** Standard scaling centers and scales by standard deviation; min-max bounds values; robust scaling uses median and IQR. Select based on algorithm and outliers.
97. Which algorithms are sensitive to feature scaling, and which are generally less sensitive?
   **Answer:** k-NN, SVMs, PCA, and many gradient-based models are scale-sensitive; decision trees and tree ensembles generally depend on feature ordering, not units.
98. How do you handle missing values in a machine-learning pipeline?
   **Answer:** Inspect why values are missing, then impute or use an estimator that supports missingness. Fit imputation only on training folds and consider indicators.
99. How do you handle imbalanced classes during model training?
   **Answer:** Start with suitable metrics and stratified splits; consider class weights or training-only resampling, then tune thresholds to reflect error costs.
100. What is the difference between oversampling, undersampling, and class weighting?
   **Answer:** Oversampling increases minority examples, undersampling reduces majority examples, and class weighting changes loss penalties without changing sample counts.
101. What is SMOTE, and what precautions are needed when using it?
   **Answer:** SMOTE synthesizes minority samples by interpolating among neighbors. Apply it only within training folds and check whether interpolation makes sense for the features.
102. Why must resampling be performed only on training data or inside cross-validation folds?
   **Answer:** Resampling before splitting can leak information or create overly similar train and test samples, producing an optimistic evaluation.
103. What is feature engineering, and how can it improve model performance?
   **Answer:** It creates or transforms inputs using domain knowledge, such as ratios or date parts, to expose useful structure; validate that features are available at prediction time.
104. What is feature selection, and how does it differ from dimensionality reduction?
   **Answer:** Feature selection keeps a subset of original variables; dimensionality reduction maps them into fewer transformed variables, often reducing interpretability.
105. What are filter, wrapper, and embedded feature-selection methods?
   **Answer:** Filters rank features using data statistics; wrappers evaluate subsets with a model; embedded methods select during training, such as L1-regularized regression.
106. What is principal component analysis (PCA), and what problem does it solve?
   **Answer:** PCA projects correlated numerical features onto orthogonal directions of decreasing variance, providing dimensionality reduction and sometimes compact representations.
107. What information can be lost when transforming features with PCA?
   **Answer:** Discarded components lose variation, including potentially predictive low-variance signals; transformed components also lose direct feature-level interpretability.
108. Why should features generally be scaled before PCA?
   **Answer:** PCA is variance-based, so variables with larger units can dominate. Scaling makes contributions more comparable when that matches the analysis goal.
109. What is the curse of dimensionality?
   **Answer:** As dimensions grow, data become sparse and distance or density estimates less reliable; more data, feature selection, or regularization may help.
110. How can you detect and reduce redundant features?
   **Answer:** Check correlations, domain overlap, stability, and model diagnostics; remove or combine duplicates when justified and compare validation performance. For linear models, collinearity can destabilize coefficients even if predictions remain useful, so retain features when they add meaningful signal or domain value.
111. How would you create useful features from dates and timestamps?
   **Answer:** Derive calendar parts, elapsed time, durations, or cyclic encodings as appropriate; account for time zones and ensure calculations use only information available then.
112. How would you represent text data for a traditional machine-learning model?
   **Answer:** Clean and tokenize text as appropriate, then use representations such as word or character n-grams, TF-IDF, or pretrained embeddings with a suitable classifier.
113. What is the difference between a bag-of-words representation and TF-IDF?
   **Answer:** Bag-of-words records token counts; TF-IDF downweights terms common across documents and emphasizes terms distinctive to a document.
114. How do you prevent target leakage when constructing aggregated or historical features?
   **Answer:** Use only records preceding each prediction time, exclude the current target, and compute aggregates with time-aware or out-of-fold logic.
115. What is concept drift, and how does it differ from data drift?
   **Answer:** Data drift is a change in input distribution; concept drift is a change in the relationship between inputs and outcome, potentially reducing predictive validity.
116. What is the difference between batch learning and online learning?
   **Answer:** Batch learning trains periodically on accumulated data; online learning updates incrementally as data arrive, requiring controls for order, drift, and update quality.
117. What assumptions should be considered when building a machine-learning model?
   **Answer:** Check assumptions about sampling, label quality, feature availability, independence, stationarity, metric suitability, and whether the training distribution matches deployment.
118. How can reproducibility be improved in a machine-learning workflow?
   **Answer:** Version data and code, record seeds and configurations, track environments and artifacts, and make preprocessing, splitting, and evaluation executable and documented.
119. Why should preprocessing choices be fitted on training data only?
   **Answer:** This simulates deployment, where future data are unseen, and prevents validation or test information from influencing learned preprocessing parameters.
120. How do you decide whether a problem is suitable for a machine-learning approach?
   **Answer:** Confirm a measurable objective, adequate representative data, feasible feedback and evaluation, acceptable risks, and value beyond a simpler rules-based solution.

## Machine-Learning Algorithms

121. How does linear regression make predictions?
   **Answer:** It computes a weighted sum of input features plus an intercept, typically fitting coefficients to minimize squared residuals.
122. What assumptions are commonly associated with ordinary least squares regression?
   **Answer:** Linearity in parameters, independent errors, constant error variance, and no perfect multicollinearity; normal errors are mainly needed for classical inference.
123. How do you interpret coefficients in a linear regression model?
   **Answer:** A coefficient is the expected change in prediction for a one-unit feature increase, holding other included features fixed, under the model specification.
124. What are residuals, and how can residual plots help diagnose a regression model?
   **Answer:** Residuals are observed minus predicted values. Plots can reveal nonlinearity, unequal variance, dependence, outliers, or systematic model error.
125. How do outliers and influential points affect linear regression?
   **Answer:** Extreme responses can inflate errors; high-leverage points can strongly change fitted coefficients. Use diagnostics and domain review rather than automatic deletion.
126. What is logistic regression, and why is it used for classification?
   **Answer:** It models class log-odds as a function of features and converts them to probabilities, offering an interpretable classification baseline.
127. How do logistic regression coefficients relate to odds and log-odds?
   **Answer:** A coefficient is a change in log-odds per feature unit; exponentiating it gives the multiplicative change in odds, holding other features fixed.
128. What is the sigmoid function, and what role does it play in binary classification?
   **Answer:** The sigmoid maps any real-valued score to a value between zero and one, which can represent a positive-class probability.
129. How does a decision tree choose splits?
   **Answer:** It searches candidate feature thresholds or categories for splits that reduce impurity or loss, subject to stopping constraints.
130. What are entropy, information gain, and Gini impurity?
   **Answer:** Entropy and Gini measure class impurity; information gain is the reduction in entropy after a split. Lower child impurity indicates better separation.
131. Why can decision trees overfit, and how can pruning help?
   **Answer:** Deep trees can create leaves tailored to noise. Limiting depth, leaf size, or pruning reduces complexity and can improve validation performance.
132. What is the difference between pre-pruning and post-pruning?
   **Answer:** Pre-pruning stops growth using constraints; post-pruning first grows a tree and then removes branches that do not improve a validation or complexity criterion.
133. How does a random forest improve on a single decision tree?
   **Answer:** It averages many decorrelated trees trained on bootstrap samples, generally reducing variance while retaining nonlinear interactions.
134. What are bagging and bootstrap sampling?
   **Answer:** Bootstrap sampling draws training sets with replacement; bagging fits models on these samples and aggregates predictions to reduce variance.
135. How does feature randomness help a random forest?
   **Answer:** Restricting candidate features at each split makes trees less correlated, so averaging them can reduce ensemble variance more effectively.
136. How can you estimate feature importance in a tree-based model, and what are the limitations?
   **Answer:** Impurity importance sums split improvements; permutation importance measures score loss after shuffling. Both can mislead with correlated features, bias, or distribution shifts.
137. What is gradient boosting, and how does it differ from bagging?
   **Answer:** Boosting adds models sequentially to correct prior errors; bagging trains models independently on resampled data and averages them.
138. How does gradient boosting build an ensemble sequentially?
   **Answer:** Each new learner fits the current residuals or loss gradients, and its scaled contribution updates the ensemble prediction.
139. What are learning rate and number of estimators in boosting?
   **Answer:** The learning rate scales each learner's contribution; the number of estimators sets how many are added. They are tuned together to balance fit and overfitting.
140. How do XGBoost, LightGBM, and CatBoost differ at a high level?
   **Answer:** All are gradient-boosted tree libraries. They differ in optimization and growth strategies, efficiency features, and categorical-feature handling; best choice depends on data and constraints.
141. What is a support vector machine (SVM), and how does it find a decision boundary?
   **Answer:** An SVM selects a boundary with a large margin between classes, with support vectors determining the boundary and regularization controlling violations.
142. What is the kernel trick in an SVM?
   **Answer:** A kernel computes inner products corresponding to a higher-dimensional feature space without explicitly constructing that space, enabling nonlinear boundaries.
143. How do the C and gamma parameters affect an RBF-kernel SVM?
   **Answer:** C controls the penalty for training errors; gamma controls the RBF influence radius. Larger values of either can create more complex boundaries.
144. What is the k-nearest neighbors (k-NN) algorithm, and how does it make predictions?
   **Answer:** It finds the k closest training examples and predicts from their labels or values, often by voting or averaging.
145. How does the choice of distance metric affect k-NN?
   **Answer:** It defines which observations count as neighbors; Euclidean, Manhattan, or other metrics suit different feature scales, geometry, and data types.
146. Why is feature scaling important for k-NN?
   **Answer:** Distance calculations can be dominated by large-scale features, so scaling makes distances more representative of intended feature contributions.
147. What is the Naive Bayes classifier, and what does its conditional-independence assumption mean?
   **Answer:** It applies Bayes' rule with a simplifying assumption that features are conditionally independent given the class; this assumption is often approximate.
148. In what kinds of problems can Naive Bayes be an effective baseline?
   **Answer:** It is fast and often useful for high-dimensional sparse text classification, especially when a simple probabilistic baseline is valuable.
149. How does k-means clustering work?
   **Answer:** It assigns points to the nearest of k centroids, recomputes centroids as cluster means, and repeats until assignments or objective stabilize.
150. How do you choose the number of clusters in k-means?
   **Answer:** Combine domain usefulness with methods such as silhouette scores or elbow plots; no single statistic guarantees meaningful clusters.
151. What are the limitations of k-means clustering?
   **Answer:** It requires k, is sensitive to scaling and initialization, favors roughly spherical clusters, and is affected by outliers and unequal densities.
152. How do hierarchical clustering and k-means differ?
   **Answer:** Hierarchical clustering builds a nested merge or split structure and can be cut at different levels; k-means directly optimizes a fixed-k partition.
153. What is DBSCAN, and how does it identify clusters and noise?
   **Answer:** DBSCAN groups dense regions using neighborhood radius and minimum points; observations outside dense reachable regions are labeled noise.
154. How do you evaluate clustering when ground-truth labels are unavailable?
   **Answer:** Inspect internal measures such as silhouette alongside stability, visualization, domain coherence, and usefulness; internal scores alone do not prove meaningful clusters.
155. What is the difference between a parametric and a non-parametric model?
   **Answer:** Parametric models assume a fixed-form structure with a finite parameter set; non-parametric models can adapt complexity with data, often requiring more data or computation.
156. What is an ensemble model, and why can combining models help?
   **Answer:** An ensemble combines multiple learners; diversity in their errors can improve robustness or accuracy compared with one model.
157. What are stacking, blending, and voting in ensemble learning?
   **Answer:** Voting aggregates predictions; stacking trains a meta-model on out-of-fold base predictions; blending uses a holdout set for that meta-model.
158. What is a neural network, and what are its basic components?
   **Answer:** It is a layered function composed of units with learned weights, biases, and nonlinear activations, trained to map inputs to outputs.
159. What are activation functions, and why are they needed?
   **Answer:** Activations introduce nonlinearity; without them, stacked linear layers collapse to a single linear transformation. Examples include ReLU, sigmoid, and tanh.
160. What is backpropagation, and how does it train a neural network?
   **Answer:** Backpropagation applies the chain rule to compute loss gradients for network parameters; an optimizer uses those gradients to update weights.

## Evaluation, Tuning, and Interpretation

161. Which metrics are appropriate for evaluating a regression model?
   **Answer:** Common choices include MAE, RMSE, and R-squared; select based on error costs, scale, and interpretability, and compare with a baseline.
162. How do MAE, MSE, and RMSE differ in interpretation and sensitivity to outliers?
   **Answer:** MAE averages absolute errors; MSE squares errors and penalizes extremes more; RMSE is the square root of MSE and uses target units.
163. What is the coefficient of determination (R-squared), and what are its limitations?
   **Answer:** R-squared measures explained variance relative to a mean baseline on a dataset. It can be negative out of sample and does not establish causation or model validity.
164. What is adjusted R-squared, and how does it account for the number of predictors?
   **Answer:** Adjusted R-squared penalizes adding predictors relative to sample size; it can decrease when a feature adds little explanatory value.
165. What is a confusion matrix?
   **Answer:** It tabulates actual versus predicted classes, showing counts of true and false positives and negatives, or their multiclass equivalents.
166. How do accuracy, precision, recall, and specificity differ?
   **Answer:** Accuracy is correct predictions overall; precision is positive predictions that are correct; recall is actual positives found; specificity is actual negatives correctly rejected.
167. Why can accuracy be misleading on an imbalanced dataset?
   **Answer:** A model can achieve high accuracy by predicting the majority class while missing most minority cases; inspect class-specific metrics and costs.
168. What is the F1 score, and when is it useful?
   **Answer:** F1 is the harmonic mean of precision and recall. It is useful when both matter, but ignores true negatives and depends on threshold.
169. How do macro, micro, and weighted averages differ for multiclass metrics?
   **Answer:** Macro averages class metrics equally; weighted averages weight by class support; micro aggregates individual decisions before calculating the metric.
170. What is the ROC curve, and how is the area under it interpreted?
   **Answer:** ROC plots true-positive rate against false-positive rate across thresholds. AUC summarizes ranking discrimination, but may be less informative with severe imbalance.
171. What is a precision-recall curve, and when is it more informative than an ROC curve?
   **Answer:** It plots precision against recall over thresholds and focuses on positive-class retrieval; it is often more revealing when positives are rare.
172. What is log loss, and what does it measure about predicted probabilities?
   **Answer:** Log loss penalizes the negative log-probability assigned to the true class; confident incorrect predictions receive especially large penalties.
173. What is a decision threshold, and how do you choose one for a business objective?
   **Answer:** It maps predicted probabilities to classes. Choose it on validation data using error costs, capacity, and operational constraints, not the test set.
174. What does it mean for a classifier's predicted probabilities to be calibrated?
   **Answer:** Among cases assigned probability p, approximately a proportion p should be positive; assess with calibration plots or proper scoring rules.
175. What is the difference between ranking performance and probability calibration?
   **Answer:** Ranking asks whether positives receive higher scores than negatives; calibration asks whether probabilities match observed frequencies. A model can excel at one but not the other.
176. How do you evaluate a multiclass classification model?
   **Answer:** Use a confusion matrix, per-class precision and recall, macro or weighted summaries, and probability metrics where relevant; account for class costs.
177. How do you evaluate a model when false positives and false negatives have different costs?
   **Answer:** Define the costs with stakeholders, evaluate a cost-sensitive metric or expected utility, and select threshold and operating point on validation data.
178. What is hyperparameter tuning, and how does it differ from fitting model parameters?
   **Answer:** Parameters are learned during model fitting; hyperparameters configure that process and are selected using validation or cross-validation performance.
179. How do grid search and random search differ?
   **Answer:** Grid search evaluates every specified combination; random search samples combinations and can explore more values efficiently when only some hyperparameters matter.
180. What is Bayesian optimization, and when might it be preferable for tuning?
   **Answer:** It uses results from prior trials to choose promising next configurations; it can help when model evaluations are costly and the search space is manageable.
181. How can nested cross-validation provide a less biased estimate during model selection?
   **Answer:** Inner folds tune hyperparameters; outer folds evaluate the entire selection process on held-out data, reducing optimism from reusing validation scores.
182. What is permutation feature importance, and what can make it misleading?
   **Answer:** It measures score degradation after a feature is shuffled. Correlated features can substitute for each other, and results depend on the chosen metric and data.
183. How do SHAP values help explain individual predictions?
   **Answer:** SHAP assigns each feature a contribution relative to a baseline prediction under a defined attribution framework; explanations are not causal effects.
184. What is partial dependence, and how does it differ from an individual conditional expectation plot?
   **Answer:** Partial dependence averages predicted response while varying a feature; ICE shows individual curves. Both can mislead when features are strongly correlated.
185. How can you assess whether a model's performance is stable across data splits or subgroups?
   **Answer:** Compare repeated or cross-validation scores and subgroup metrics, report uncertainty and sample sizes, and investigate meaningful variation or distribution shifts.

## Deployment, Monitoring, and Responsible ML

186. What considerations guide the choice between model performance, interpretability, latency, and cost?
   **Answer:** Choose based on the decision's risk and constraints: required performance, explanation needs, response time, maintenance, fairness, and total operating cost.
187. What steps are involved in deploying a trained machine-learning model?
   **Answer:** Package the model and preprocessing, validate inputs, integrate inference, test performance and safety, release with monitoring, and plan rollback and updates.
188. Why should the preprocessing pipeline be versioned and deployed with the model?
   **Answer:** Predictions depend on identical transformations at training and inference; versioning prevents mismatches and supports reproducibility and rollback.
189. What is the difference between batch inference and real-time inference?
   **Answer:** Batch inference processes many records on a schedule; real-time inference returns a prediction per request under latency and availability constraints.
190. Which model and data artifacts should be tracked to make a deployment reproducible?
   **Answer:** Track model weights, code, preprocessing, data and label versions, feature definitions, environment, configuration, evaluation results, and approval metadata.
191. What metrics and signals should you monitor after deploying a model?
   **Answer:** Monitor input quality and drift, prediction distributions, latency, errors, availability, resource use, and delayed outcome metrics and subgroup performance.
192. How can you detect data drift and concept drift in production?
   **Answer:** Compare incoming feature distributions to a reference for data drift; once labels arrive, monitor predictive performance and outcome relationships for concept drift.
193. What actions might you take when a production model's performance degrades?
   **Answer:** Verify data and pipeline changes, assess impact and affected groups, mitigate or roll back if needed, investigate causes, retrain only with validated data, and re-evaluate.
194. Why is it important to monitor model performance across relevant user groups?
   **Answer:** Aggregate metrics can conceal unequal errors or harms. Group-level monitoring helps identify disparities, provided privacy and sample-size safeguards are respected.
195. What is algorithmic fairness, and how can you investigate potential bias in a model?
   **Answer:** Fairness concerns equitable treatment and outcomes under context-specific definitions. Examine data, subgroup errors, relevant fairness measures, and stakeholder impacts; no single metric fits all cases.
196. What privacy risks should be considered when collecting data and training models?
   **Answer:** Minimize and protect personal data, control access and retention, assess re-identification and memorization risks, and comply with applicable consent and legal requirements.
197. How can you make machine-learning results understandable to stakeholders?
   **Answer:** Connect metrics and explanations to the decision, use plain language and relevant examples, disclose uncertainty and limitations, and avoid overstating causal conclusions.
198. What is the difference between a model card and a dataset datasheet?
   **Answer:** A model card documents a model's intended use, evaluation, and limitations; a dataset datasheet documents data origin, composition, collection, and recommended uses.
199. How would you design an A/B test to compare a new model with the current system?
   **Answer:** Define a primary outcome and guardrails, randomize eligible units, size the experiment for meaningful effects, prevent contamination, and analyze according to the plan.
200. How would you explain the end-to-end machine-learning lifecycle, from problem definition through monitoring, for a project you have worked on?
   **Answer:** Describe the decision and success metric, data and EDA, leakage-safe validation, baseline and model selection, deployment, monitoring, impact, limitations, and lessons learned using a concrete example.