# LightGBM Regression

A gradient-boosted tree implementation designed for efficiency, including histogram-based splitting and leaf-wise growth.

**Mathematical objective:** Quantized feature histograms accelerate split search; leaf-wise expansion can reduce loss quickly but needs complexity limits.

**Assumptions and limitations:** Install the optional lightgbm extra. Set num_leaves/max_depth and minimum leaf data to control overfitting.

**Use case:** Large tabular datasets where training throughput and memory use matter.

## Run
From the repository root, install `python -m pip install -e .`, then run:
`python 01_Supervised_Learning/Regression/11_LightGBM_Regression/train.py`

The `config.yaml` controls estimator settings, seed, split, and scaling. The deterministic dataset is synthetic and generated offline by `ml_course.data`; no real-world records are bundled. The preprocessing pipeline is fitted only on training rows. `test_model.py` checks model training, artifact round-trip prediction, and invalid input handling.

Optional estimator dependencies: lightgbm.
