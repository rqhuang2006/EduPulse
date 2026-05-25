from __future__ import annotations

import shutil
from pathlib import Path


def export_artifact(domain: str, name: str, src: str | Path, target: str | Path) -> Path:
    src_path = Path(src)
    target_path = Path(target)
    if not src_path.exists():
        raise FileNotFoundError(f"[{domain}] missing source artifact '{name}': {src_path}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, target_path)
    if not target_path.exists():
        raise FileNotFoundError(f"[{domain}] artifact export failed '{name}': {target_path}")
    return target_path
