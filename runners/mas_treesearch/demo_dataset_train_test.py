from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    runpy.run_path(str(root / "demo_dataset_train_test.py"), run_name="__main__")


if __name__ == "__main__":
    main()
