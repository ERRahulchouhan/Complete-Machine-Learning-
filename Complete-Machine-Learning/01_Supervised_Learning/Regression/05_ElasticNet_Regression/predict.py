from __future__ import annotations
import argparse
import json
from pathlib import Path
from ml_course.pipeline import predict

def main() -> None:
    parser = argparse.ArgumentParser(description="Predict with the trained regression pipeline")
    parser.add_argument("--input", required=True, help="JSON array of feature objects")
    parser.add_argument("--model", default=str(Path(__file__).resolve().parent / "models" / "model.joblib"))
    args = parser.parse_args()
    rows = json.loads(args.input)
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        parser.error("--input must be a JSON array of objects")
    print(json.dumps({"predictions": predict(args.model, rows)}))

if __name__ == "__main__":
    main()
