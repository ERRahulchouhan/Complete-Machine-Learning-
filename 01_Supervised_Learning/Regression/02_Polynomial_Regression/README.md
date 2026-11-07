# Polynomial Regression

Adds powers and interactions of input features, then fits a linear model in the expanded feature space. The resulting curve can model nonlinear relationships.

**Mathematical objective:** φ(x) = [1, x, x², …]; predict with φ(x)ᵀw. It is still linear in its fitted coefficients.

**Assumptions and limitations:** Polynomial degree controls flexibility. High degrees can oscillate and extrapolate badly; scaling before polynomial expansion can improve numerical conditioning.

**Use case:** Growth curves, calibration curves, and smooth nonlinear relationships with few input dimensions.

## Run
From the repository root, install `python -m pip install -e .`, then run:
`python 01_Supervised_Learning/Regression/02_Polynomial_Regression/train.py`

The `config.yaml` controls estimator settings, seed, split, and scaling. The deterministic dataset is synthetic and generated offline by `ml_course.data`; no real-world records are bundled. The preprocessing pipeline is fitted only on training rows. `test_model.py` checks model training, artifact round-trip prediction, and invalid input handling.

Optional estimator dependencies: none beyond requirements.txt.
