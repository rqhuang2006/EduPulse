from __future__ import annotations

import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WHEELHOUSE = ROOT / "wheelhouse"
RUNTIME = ROOT / "streamlit_runtime"


def select_wheels() -> list[Path]:
    preferred = [
        "streamlit-",
        "altair-",
        "attrs-",
        "blinker-",
        "cachetools-",
        "certifi-",
        "charset_normalizer-",
        "click-",
        "colorama-",
        "gitdb-",
        "gitpython-",
        "idna-",
        "jinja2-",
        "jsonschema-",
        "jsonschema_specifications-",
        "markupsafe-",
        "narwhals-",
        "numpy-",
        "packaging-",
        "pandas-",
        "pillow-",
        "protobuf-",
        "pyarrow-",
        "pydeck-",
        "python_dateutil-",
        "referencing-",
        "requests-",
        "rpds_py-",
        "six-",
        "smmap-",
        "tenacity-",
        "toml-",
        "tornado-",
        "typing_extensions-",
        "tzdata-",
        "urllib3-",
        "watchdog-",
        "openpyxl-",
        "et_xmlfile-",
    ]
    wheel_paths: list[Path] = []
    for prefix in preferred:
        matches = sorted(WHEELHOUSE.glob(f"{prefix}*.whl"))
        if matches:
            wheel_paths.append(matches[-1])
    return wheel_paths


def extract_wheel(wheel_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(wheel_path) as zf:
        zf.extractall(destination)


def main() -> None:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    marker = RUNTIME / "streamlit" / "__init__.py"
    if marker.exists():
        print(f"Streamlit runtime already ready: {RUNTIME}")
        return
    wheels = select_wheels()
    if not wheels:
        raise RuntimeError(f"No wheels found in {WHEELHOUSE}")
    for wheel in wheels:
        print(f"Extracting {wheel.name}")
        extract_wheel(wheel, RUNTIME)
    if not marker.exists():
        raise RuntimeError("Streamlit runtime bootstrap failed: streamlit package not found after extraction.")
    print(f"Streamlit runtime ready: {RUNTIME}")


if __name__ == "__main__":
    main()
