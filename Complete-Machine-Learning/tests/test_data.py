from ml_course.data import make_regression_frame

def test_synthetic_dataset_is_reproducible() -> None:
    first = make_regression_frame(50, 7)
    second = make_regression_frame(50, 7)
    assert first.equals(second)
    assert first.shape == (50, 5)

def test_dataset_rejects_too_few_rows() -> None:
    import pytest
    with pytest.raises(ValueError):
        make_regression_frame(5)
