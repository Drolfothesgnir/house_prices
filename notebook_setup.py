# notebook_setup.py
import os
import sys
from pathlib import Path

def setup_notebook_environment():
    root_dir = Path(os.getcwd()).absolute()
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    
    # Setup any other notebook-specific configurations here
    return root_dir