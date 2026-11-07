# Extra Trees Regression

An ensemble of randomized trees. Candidate split thresholds are randomized as well as feature selection, often reducing variance further.

**Mathematical objective:** Average predictions of trees whose split features and thresholds are randomized; split quality still guides selection among candidates.

**Assumptions and limitations:** Compare with Random Forest using identical splits and validation; randomness may increase bias while reducing variance.

**Use case:** Fast nonlinear tabular baselines with many potentially useful features.

## Run
From the repository root, install `python -m pip install -e .`, then run:
`python 01_Supervised_Learning/Regression/08_Extra_Trees_Regression/train.py`

The `config.yaml` controls estimator settings, seed, split, and scaling. The deterministic dataset is synthetic and generated offline by `ml_course.data`; no real-world records are bundled. The preprocessing pipeline is fitted only on training rows. `test_model.py` checks model training, artifact round-trip prediction, and invalid input handling.

Optional estimator dependencies: none beyond requirements.txt.
