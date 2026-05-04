import sys
from pathlib import Path
project_root = Path().resolve()
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
import pytest
from utils import summarize_dataset, validate_status_value, summarize_vars, validate_required_columns 

class VoiceDataset:
    def __init__(self):
        self.data = None

dataset = VoiceDataset()

def test_none_dataset_exception():
    """Test requier data in dataset before calculating summary"""

    with pytest.raises(ValueError):
        summarize_dataset(dataset, "/output")
