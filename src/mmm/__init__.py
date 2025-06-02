"""
mmm/__init__.py
"""
import os
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
PROJECT_ROOT = HERE.parent.parent
WORKING_DIR = Path(os.environ.get(
    "PBS_O_WORKDIR",
    os.getcwd(),
))
FALLBACK_TOKENIZER_PATH = PROJECT_ROOT

OUTPUTS_DIR = WORKING_DIR.joinpath("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)
