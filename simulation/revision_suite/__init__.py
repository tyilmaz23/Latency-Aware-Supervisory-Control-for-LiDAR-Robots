"""Versioned paired revision experiments; no training or API calls at import."""
import os
from pathlib import Path
VERSION="1.0.0"
os.environ.setdefault("MPLCONFIGDIR",str(Path(__file__).resolve().parents[1]/".runtime"/"matplotlib"))
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
