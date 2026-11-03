# Gradient Boosting Regression

Builds an additive sequence of weak trees, each correcting residual errors from the current ensemble.

**Mathematical objective:** Fₘ(x)=Fₘ₋₁(x)+ηhₘ(x), where hₘ follows the negative loss gradient and η is the learning rate.

**Assumptions and limitations:** Tune learning rate with tree count and depth; use validation/early stopping where supported and watch for overfitting.

**Use case:** Structured tabular regression when nonlinear interactions improve on linear baselines.

## Run
From the repository root, install `python -m pip install -e .`, then run:
`python 01_Supervised_Learning/Regression/09_Gradient_Boosting_Regression/train.py`

The `config.yaml` controls estimator settings, seed, split, and scaling. The deterministic dataset is synthetic and generated offline by `ml_course.data`; no real-world records are bundled. The preprocessing pipeline is fitted only on training rows. `test_model.py` checks model training, artifact round-trip prediction, and invalid input handling.

Optional estimator dependencies: none beyond requirements.txt.
