from __future__ import annotations
import logging
from pathlib import Path
from model import build_model
from ml_course.pipeline import train

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    location = Path(__file__).resolve().parent
    print(train(location / "config.yaml", build_model()))
