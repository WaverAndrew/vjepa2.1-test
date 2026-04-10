"""
Hardcoded path setup — imported at the top of every eval/ script.
Ensures vjepa21_lib and the vjepa2 repo are importable without env vars.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
VJEPA2_DIR = "/scratch/3206024/vjepa2_official"

for p in [str(PROJECT_ROOT), VJEPA2_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)
