# Elastic Net Regression

Combines L1 and L2 penalties to encourage sparsity while stabilizing groups of correlated features.

**Mathematical objective:** Minimize (1/(2n))||y − Xw||²₂ + α·l1_ratio·||w||₁ + α(1−l1_ratio)||w||²₂/2.

**Assumptions and limitations:** Scale inputs and tune both α and l1_ratio with cross-validation. It is often preferable to Lasso for correlated predictors.

**Use case:** High-dimensional prediction where groups of variables carry related information.

## Run
From the repository root, install `python -m pip install -e .`, then run:
`python 01_Supervised_Learning/Regression/05_ElasticNet_Regression/train.py`

The `config.yaml` controls estimator settings, seed, split, and scaling. The deterministic dataset is synthetic and generated offline by `ml_course.data`; no real-world records are bundled. The preprocessing pipeline is fitted only on training rows. `test_model.py` checks model training, artifact round-trip prediction, and invalid input handling.

Optional estimator dependencies: none beyond requirements.txt.
