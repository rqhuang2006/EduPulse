from __future__ import annotations

import sys
from pathlib import Path


def _add_repo_vendor() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    vendor_dir = repo_root / ".deps3"
    if vendor_dir.exists():
        vendor = str(vendor_dir)
        if vendor not in sys.path:
            sys.path.insert(0, vendor)


_add_repo_vendor()
