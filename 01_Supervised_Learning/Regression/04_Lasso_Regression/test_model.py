from pathlib import Path
import importlib.util
import pytest
from ml_course.pipeline import predict, train

model_path = Path(__file__).with_name("model.py")
model_spec = importlib.util.spec_from_file_location(f"algorithm_model_{model_path.parent.name}", model_path)
if model_spec is None or model_spec.loader is None:
    raise ImportError(f"Unable to load model factory from {model_path}")
model_module = importlib.util.module_from_spec(model_spec)
model_spec.loader.exec_module(model_module)
build_model = model_module.build_model

try:
    build_model()
except ModuleNotFoundError as error:
    pytest.skip(f"Optional estimator dependency is unavailable: {error.name}", allow_module_level=True)

def test_train_and_predict_round_trip(tmp_path: Path) -> None:
    config = Path(__file__).with_name("config.yaml")
    result = train(config, build_model(), output_dir=tmp_path)
    assert result["n_train"] == 480
    assert result["n_test"] == 120
    prediction = predict(tmp_path / "models" / "model.joblib", [{"size": 100, "rooms": 3, "age": 10, "distance": 2}])
    assert len(prediction) == 1
    assert prediction[0] == pytest.approx(prediction[0])

def test_prediction_rejects_schema_mismatch(tmp_path: Path) -> None:
    train(Path(__file__).with_name("config.yaml"), build_model(), output_dir=tmp_path)
    with pytest.raises(ValueError, match="schema mismatch"):
        predict(tmp_path / "models" / "model.joblib", [{"size": 10}])
