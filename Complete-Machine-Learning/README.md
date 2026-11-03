# Complete Machine Learning

A phased, executable machine-learning curriculum. The fundamentals, 15 regression algorithms, and core unsupervised clustering and dimensionality-reduction algorithms are implemented. Other phases are explicitly listed in `LEARNING_ROADMAP.md`.

## Setup
```powershell
cd Complete-Machine-Learning
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```
Optional native boosters: `python -m pip install -r requirements-optional.txt` (also installs deployment/explainability tools).

## Run an algorithm
```powershell
python 01_Supervised_Learning\Regression\01_Linear_Regression\train.py
python 01_Supervised_Learning\Regression\01_Linear_Regression\predict.py --input '[{"size":100,"rooms":3,"age":10,"distance":2}]'
python -m pytest
```
Training uses a deterministic synthetic offline housing-like dataset. It is for learning only, not a real estate dataset or production recommendation. Estimators, preprocessing, split seed, metrics, and artifact path are configuration-driven. Train/test split precedes fitted preprocessing; holdout data does not fit preprocessing.

Open each `*_Regression.ipynb` in Jupyter. Notebooks contain 21 cell-wise sections and separate executable cells. For advanced boosters install their corresponding optional package before training. Python 3.10+ is supported; use package versions compatible with your Python platform.

## Unsupervised learning
Open the notebooks under `Unsupervised_Learning\Clustering` and `Unsupervised_Learning\Dimensionality_Reduction`. They use deterministic synthetic data or scikit-learn's built-in digits dataset, so no external dataset download is required. Each notebook explains one algorithm, prepares data, fits and evaluates an example, and discusses limitations. Clustering covers K-means, agglomerative clustering, DBSCAN, OPTICS, Gaussian mixtures, Mean Shift, spectral clustering, BIRCH, Affinity Propagation, and HDBSCAN. Dimensionality reduction covers PCA, Incremental PCA, Kernel PCA, Truncated SVD, FastICA, NMF, Factor Analysis, t-SNE, Isomap, and Locally Linear Embedding.

For practical pandas workflows in ML data preparation, see [`pandas.md`](./pandas.md).
For model-evaluation, explainability, and production-monitoring visualizations, see [`chart.md`](./chart.md).
For leakage-safe feature processing, selection methods, and production feature governance, see [`feature_selection_and_feature_processing.md`](./feature_selection_and_feature_processing.md).
For the complete model lifecycle from problem definition through deployment and monitoring, see [`model_develoment_pipeline.md`](./model_develoment_pipeline.md).

To run the shared API example, train the linear model first, then build from the repository root with `docker build -t ml-regression -f Dockerfile .` and run `docker run --rm -p 8000:8000 ml-regression`. The Kubernetes manifest is a starting point for this image; provide production secrets, ingress, resource limits, and a trusted artifact workflow before use.

## Production caveat
These are robust learning templates, not a claim that a model is production-ready. Add organization-specific authentication, authorization, privacy, monitoring/alerts, artifact registry, data governance, CI/CD approvals, infrastructure, and rollback testing before deployment. Model files must only be loaded from trusted sources.
