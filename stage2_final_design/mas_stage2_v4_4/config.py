from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GCR_ROOT = REPO_ROOT / "Stage2-GCR+"
gcr_text = str(GCR_ROOT)
if GCR_ROOT.exists() and gcr_text not in sys.path:
    sys.path.insert(0, gcr_text)

from stage2_gcr_plus.config import Stage2V44Config

__all__ = ["Stage2V44Config"]
