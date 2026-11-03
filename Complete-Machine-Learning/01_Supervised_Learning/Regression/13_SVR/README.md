# Support Vector Regression

Fits a function with a margin of tolerance ε and penalizes points outside the margin. Kernels enable nonlinear fits.

**Mathematical objective:** Minimize ½||w||² + CΣ(ξᵢ+ξᵢ*) subject to ε-insensitive residual constraints.

**Assumptions and limitations:** Scale features and target when appropriate; tune C, epsilon, and kernel parameters. Kernel SVR can be expensive at large n.

**Use case:** Smaller datasets where a smooth nonlinear function and margin-based robustness are useful.

## Run
From the repository root, install `python -m pip install -e .`, then run:
`python 01_Supervised_Learning/Regression/13_SVR/train.py`

The `config.yaml` controls estimator settings, seed, split, and scaling. The deterministic dataset is synthetic and generated offline by `ml_course.data`; no real-world records are bundled. The preprocessing pipeline is fitted only on training rows. `test_model.py` checks model training, artifact round-trip prediction, and invalid input handling.

Optional estimator dependencies: none beyond requirements.txt.
