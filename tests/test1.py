
import pandas as pd
import sys
from pathlib import Path
project_root = Path().resolve()
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
from utils import summarize_dataset, validate_status_value, summarize_vars, validate_required_columns 


def test_validate_required_columns_passes():
    """Test required column validation when column exists."""
    df = pd.DataFrame({"status": [0, 1], "feature": [1.2, 3.4]})
    assert validate_required_columns(df, ["status"]) is True
