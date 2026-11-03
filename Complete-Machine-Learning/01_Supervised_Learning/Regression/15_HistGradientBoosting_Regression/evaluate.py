from pathlib import Path
from ml_course.pipeline import evaluate

if __name__ == "__main__":
    print(evaluate(Path(__file__).resolve().parent / "models" / "model.joblib"))
