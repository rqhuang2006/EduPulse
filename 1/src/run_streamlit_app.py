from __future__ import annotations

import os
import site
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "streamlit_runtime"
LOCAL_PACKAGES = ROOT / ".python_packages"

site.addsitedir(site.getusersitepackages())
site.addsitedir(str(LOCAL_PACKAGES))
site.addsitedir(str(RUNTIME))

if not (RUNTIME / "streamlit" / "__init__.py").exists():
    bootstrap = ROOT / "src" / "bootstrap_streamlit_runtime.py"
    completed = subprocess.run([sys.executable, str(bootstrap)], cwd=str(ROOT), check=False)
    if completed.returncode != 0:
        raise SystemExit("Failed to bootstrap local Streamlit runtime.")
    site.addsitedir(str(RUNTIME))

from streamlit.web import cli as stcli  # noqa: E402


def main() -> None:
    server_port = os.getenv("STREAMLIT_SERVER_PORT", "8501")
    server_address = os.getenv("STREAMLIT_SERVER_ADDRESS", "127.0.0.1")
    sys.argv = [
        "streamlit",
        "run",
        str(ROOT / "app.py"),
        "--server.address",
        server_address,
        "--server.port",
        server_port,
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    raise SystemExit(stcli.main())


if __name__ == "__main__":
    main()
