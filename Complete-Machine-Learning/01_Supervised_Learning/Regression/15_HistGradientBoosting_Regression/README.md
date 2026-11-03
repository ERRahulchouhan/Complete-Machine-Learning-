# Histogram Gradient Boosting Regression

Gradient-boosted trees that bin continuous features into histograms for efficient split search.

**Mathematical objective:** Boosting optimizes a differentiable loss while selecting splits over binned feature values.

**Assumptions and limitations:** Tune leaf count, learning rate, and minimum samples per leaf. It can natively handle missing numeric values.

**Use case:** Scalable nonlinear regression on medium-to-large tabular datasets.

## Run
From the repository root, install `python -m pip install -e .`, then run:
`python 01_Supervised_Learning/Regression/15_HistGradientBoosting_Regression/train.py`

The `config.yaml` controls estimator settings, seed, split, and scaling. The deterministic dataset is synthetic and generated offline by `ml_course.data`; no real-world records are bundled. The preprocessing pipeline is fitted only on training rows. `test_model.py` checks model training, artifact round-trip prediction, and invalid input handling.

Optional estimator dependencies: none beyond requirements.txt.
