from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
for candidate in (PACKAGE_ROOT, REPO_ROOT):
    text = str(candidate)
    if text not in sys.path:
        sys.path.insert(0, text)

from stage2_gcr_plus.orchestration.runner import main


if __name__ == "__main__":
    main()
