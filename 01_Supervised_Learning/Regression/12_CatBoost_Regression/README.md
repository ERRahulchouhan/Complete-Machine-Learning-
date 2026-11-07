# CatBoost Regression

Gradient boosting with ordered target statistics and ordered boosting techniques, particularly useful with categorical inputs.

**Mathematical objective:** Ordered boosting reduces target leakage from category statistics; symmetric trees are used by default.

**Assumptions and limitations:** Install the optional catboost extra. Declare categorical columns explicitly and validate category handling on unseen values.

**Use case:** Business tabular data with high-cardinality categorical features.

## Run
From the repository root, install `python -m pip install -e .`, then run:
`python 01_Supervised_Learning/Regression/12_CatBoost_Regression/train.py`

The `config.yaml` controls estimator settings, seed, split, and scaling. The deterministic dataset is synthetic and generated offline by `ml_course.data`; no real-world records are bundled. The preprocessing pipeline is fitted only on training rows. `test_model.py` checks model training, artifact round-trip prediction, and invalid input handling.

Optional estimator dependencies: catboost.
