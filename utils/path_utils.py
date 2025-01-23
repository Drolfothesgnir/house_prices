import os
from pathlib import Path

def get_project_root():
    """Get absolute path to project root directory."""
    # If this file is in utils/path_utils.py, go up two levels
    return str(Path(__file__).resolve().parents[1])

def get_data_path(filename):
    """
    Get absolute path to a file in the data directory.
    Handles both 'train.csv' and 'data/train.csv' formats.
    """
    # Remove 'data/' prefix if it exists
    if filename.startswith('data/'):
        filename = filename[5:]
    
    data_dir = os.path.join(get_project_root(), "data")
    return os.path.join(data_dir, filename)