# Linear Regression

Predicts a continuous target as a weighted sum of input features. It is interpretable and a strong first baseline when effects are approximately linear.

**Mathematical objective:** y = Xw + b; minimize mean squared error. The closed-form solution is w = (XᵀX)⁻¹Xᵀy when the inverse exists; numerical implementations use a stable least-squares solver.

**Assumptions and limitations:** Ordinary least squares has no regularization. Check linearity, independence, constant error variance, and influential observations; normal residuals matter for classical inference, not for prediction itself.

**Use case:** House prices, demand, and forecasting numeric outcomes when a transparent additive baseline is useful.

## Run
From the repository root, install `python -m pip install -e .`, then run:
`python 01_Supervised_Learning/Regression/01_Linear_Regression/train.py`

The `config.yaml` controls estimator settings, seed, split, and scaling. The deterministic dataset is synthetic and generated offline by `ml_course.data`; no real-world records are bundled. The preprocessing pipeline is fitted only on training rows. `test_model.py` checks model training, artifact round-trip prediction, and invalid input handling.

Optional estimator dependencies: none beyond requirements.txt.
