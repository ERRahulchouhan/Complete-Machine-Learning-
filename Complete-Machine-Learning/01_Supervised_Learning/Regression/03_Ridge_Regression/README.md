# Ridge Regression

Linear regression with an L2 penalty. Shrinking coefficients reduces variance when features are correlated or numerous.

**Mathematical objective:** Minimize ||y − Xw||²₂ + α||w||²₂. α ≥ 0 controls the shrinkage strength; intercept is not penalized.

**Assumptions and limitations:** Standardize features, choose α using cross-validation, and retain all features (Ridge usually does not set coefficients exactly to zero).

**Use case:** Stable pricing or demand models with correlated predictors.

## Run
From the repository root, install `python -m pip install -e .`, then run:
`python 01_Supervised_Learning/Regression/03_Ridge_Regression/train.py`

The `config.yaml` controls estimator settings, seed, split, and scaling. The deterministic dataset is synthetic and generated offline by `ml_course.data`; no real-world records are bundled. The preprocessing pipeline is fitted only on training rows. `test_model.py` checks model training, artifact round-trip prediction, and invalid input handling.

Optional estimator dependencies: none beyond requirements.txt.
