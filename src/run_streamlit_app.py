from __future__ import annotations

import site
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "streamlit_runtime"
site.addsitedir(site.getusersitepackages())
site.addsitedir(str(RUNTIME))

if not (RUNTIME / "streamlit" / "__init__.py").exists():
    bootstrap = ROOT / "src" / "bootstrap_streamlit_runtime.py"
    completed = subprocess.run([sys.executable, str(bootstrap)], cwd=str(ROOT), check=False)
    if completed.returncode != 0:
        raise SystemExit("Failed to bootstrap local Streamlit runtime.")
    site.addsitedir(str(RUNTIME))

from streamlit.web import cli as stcli  # noqa: E402


def main() -> None:
    sys.argv = [
        "streamlit",
        "run",
        str(ROOT / "app.py"),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    raise SystemExit(stcli.main())


if __name__ == "__main__":
    main()
