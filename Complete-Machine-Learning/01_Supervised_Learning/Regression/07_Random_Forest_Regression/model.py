from __future__ import annotations
from importlib import import_module
from typing import Any
import yaml
from pathlib import Path

def build_model(config_path: str | Path = Path(__file__).with_name("config.yaml")) -> Any:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    specification = config["estimator"]
    module = import_module(specification["module"])
    estimator_class = getattr(module, specification["class"])
    parameters = dict(specification.get("parameters", {}))
    if specification["class"] == "make_pipeline":
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        return estimator_class(PolynomialFeatures(**parameters), LinearRegression())
    return estimator_class(**parameters)
