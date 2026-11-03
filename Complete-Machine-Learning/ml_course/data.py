from __future__ import annotations
import numpy as np
import pandas as pd

FEATURES = ["size", "rooms", "age", "distance"]
TARGET = "value"

def make_regression_frame(n_samples: int = 600, random_state: int = 42) -> pd.DataFrame:
    """Create a deterministic, offline teaching dataset."""
    if n_samples < 20:
        raise ValueError("n_samples must be at least 20")
    rng = np.random.default_rng(random_state)
    frame = pd.DataFrame({
        "size": rng.uniform(30, 250, n_samples),
        "rooms": rng.integers(1, 8, n_samples),
        "age": rng.uniform(0, 80, n_samples),
        "distance": rng.uniform(0.2, 30, n_samples),
    })
    noise = rng.normal(0, 18000, n_samples)
    frame[TARGET] = 1200 * frame["size"] + 9000 * frame["rooms"] - 350 * frame["age"] - 1800 * frame["distance"] + noise
    return frame
