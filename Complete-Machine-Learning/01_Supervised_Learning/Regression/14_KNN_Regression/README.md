# K-Nearest Neighbors Regression

Predicts from nearby training examples, averaging targets using a chosen neighborhood and distance weighting.

**Mathematical objective:** ŷ(x)=Σᵢwᵢyᵢ/Σᵢwᵢ over the k nearest points.

**Assumptions and limitations:** Scale features, tune k and distance weighting, and account for inference/memory cost and the curse of dimensionality.

**Use case:** Local interpolation on small datasets with meaningful distance geometry.

## Run
From the repository root, install `python -m pip install -e .`, then run:
`python 01_Supervised_Learning/Regression/14_KNN_Regression/train.py`

The `config.yaml` controls estimator settings, seed, split, and scaling. The deterministic dataset is synthetic and generated offline by `ml_course.data`; no real-world records are bundled. The preprocessing pipeline is fitted only on training rows. `test_model.py` checks model training, artifact round-trip prediction, and invalid input handling.

Optional estimator dependencies: none beyond requirements.txt.
