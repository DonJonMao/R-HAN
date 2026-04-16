from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_paths() -> tuple[Path, Path, Path]:
    package_root = Path(__file__).resolve().parent
    ucc_root = package_root.parent
    vendor_root = ucc_root / "vendor"
    repo_root = ucc_root.parent
    for candidate in (ucc_root, vendor_root, repo_root):
        text = str(candidate)
        if text not in sys.path:
            sys.path.insert(0, text)
    return ucc_root, vendor_root, repo_root
