# XGBoost Regression

Regularized gradient-boosted decision trees with efficient split finding, shrinkage, and missing-value handling.

**Mathematical objective:** Second-order objective approximation uses gradient and Hessian statistics; tree complexity is penalized.

**Assumptions and limitations:** Install the optional xgboost extra. Tune depth, learning rate, row/column sampling, and regularization on validation data.

**Use case:** High-performance tabular prediction with nonlinearities and interactions.

## Run
From the repository root, install `python -m pip install -e .`, then run:
`python 01_Supervised_Learning/Regression/10_XGBoost_Regression/train.py`

The `config.yaml` controls estimator settings, seed, split, and scaling. The deterministic dataset is synthetic and generated offline by `ml_course.data`; no real-world records are bundled. The preprocessing pipeline is fitted only on training rows. `test_model.py` checks model training, artifact round-trip prediction, and invalid input handling.

Optional estimator dependencies: xgboost.
