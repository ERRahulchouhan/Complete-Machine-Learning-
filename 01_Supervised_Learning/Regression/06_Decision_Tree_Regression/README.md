# Decision Tree Regression

Partitions feature space into regions and predicts a constant (typically the mean) in each region.

**Mathematical objective:** Greedily choose splits minimizing within-leaf squared error; prediction is the mean target in the reached leaf.

**Assumptions and limitations:** Trees need little scaling but can overfit. Limit depth/leaf size and assess stability across folds.

**Use case:** Interpretable nonlinear rules, threshold effects, and mixed-scale tabular features.

## Run
From the repository root, install `python -m pip install -e .`, then run:
`python 01_Supervised_Learning/Regression/06_Decision_Tree_Regression/train.py`

The `config.yaml` controls estimator settings, seed, split, and scaling. The deterministic dataset is synthetic and generated offline by `ml_course.data`; no real-world records are bundled. The preprocessing pipeline is fitted only on training rows. `test_model.py` checks model training, artifact round-trip prediction, and invalid input handling.

Optional estimator dependencies: none beyond requirements.txt.
