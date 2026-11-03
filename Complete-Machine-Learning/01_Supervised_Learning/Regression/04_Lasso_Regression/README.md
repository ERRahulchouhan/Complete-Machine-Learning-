# Lasso Regression

Linear regression with an L1 penalty. It can shrink some coefficients exactly to zero, producing sparse models.

**Mathematical objective:** Minimize (1/(2n))||y − Xw||²₂ + α||w||₁. The nonsmooth L1 term is optimized iteratively.

**Assumptions and limitations:** Scale features before fitting; correlated features may be selected unpredictably. Tune α and inspect convergence warnings.

**Use case:** Compact models and feature selection when sparse effects are plausible.

## Run
From the repository root, install `python -m pip install -e .`, then run:
`python 01_Supervised_Learning/Regression/04_Lasso_Regression/train.py`

The `config.yaml` controls estimator settings, seed, split, and scaling. The deterministic dataset is synthetic and generated offline by `ml_course.data`; no real-world records are bundled. The preprocessing pipeline is fitted only on training rows. `test_model.py` checks model training, artifact round-trip prediction, and invalid input handling.

Optional estimator dependencies: none beyond requirements.txt.
