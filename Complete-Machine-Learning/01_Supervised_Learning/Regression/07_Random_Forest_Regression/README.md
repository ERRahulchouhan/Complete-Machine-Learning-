# Random Forest Regression

Averages decision trees trained on bootstrap samples with randomized feature subsets, reducing tree variance.

**Mathematical objective:** For B trees, prediction is (1/B)Σᵦ fᵦ(x). Random feature sampling decorrelates trees.

**Assumptions and limitations:** Tune tree count, depth, and leaf size; monitor inference cost and avoid extrapolation expectations outside training ranges.

**Use case:** Robust nonlinear tabular prediction and feature interaction discovery.

## Run
From the repository root, install `python -m pip install -e .`, then run:
`python 01_Supervised_Learning/Regression/07_Random_Forest_Regression/train.py`

The `config.yaml` controls estimator settings, seed, split, and scaling. The deterministic dataset is synthetic and generated offline by `ml_course.data`; no real-world records are bundled. The preprocessing pipeline is fitted only on training rows. `test_model.py` checks model training, artifact round-trip prediction, and invalid input handling.

Optional estimator dependencies: none beyond requirements.txt.
