import numpy as np
import sys
from pathlib import Path
project_root = Path().resolve()
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
from model import SVMPredictor, RandomForestPredictor
import pytest


def test_predict_before_training():
    """Test that predicting before training raises error."""
    predictor = SVMPredictor()
    X = np.random.randn(10, 5)

    with pytest.raises(AttributeError):
        predictor.predict(X)
