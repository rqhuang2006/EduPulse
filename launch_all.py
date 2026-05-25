from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import Iterable

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


ROOT = Path(__file__).resolve().parent
RUNTIME_PYTHON = ROOT / "runtime" / "python.exe"
FRONTEND_ROOT = ROOT / "1"
BACKEND_ROOT = ROOT / "服创赛"
TMP_ROOT = ROOT / ".tmp"

FRONTEND_ENTRY = FRONTEND_ROOT / "src" / "run_streamlit_app.py"
FRONTEND_BOOTSTRAP = FRONTEND_ROOT / "src" / "bootstrap_streamlit_runtime.py"
FRONTEND_RUNTIME_MARKER = FRONTEND_ROOT / "streamlit_runtime" / "streamlit" / "__init__.py"
FRONTEND_REQUIREMENTS = FRONTEND_ROOT / "requirements.txt"
FRONTEND_LOCAL_PACKAGES = FRONTEND_ROOT / ".python_packages"

BACKEND_ENTRY = BACKEND_ROOT / "run_harness_server.py"
BACKEND_LOCAL_PACKAGES = BACKEND_ROOT / ".deps3"

BACKEND_INSTALL_PACKAGES = [
    "fastapi",
    "uvicorn",
    "pyyaml",
    "pandas",
    "numpy",
    "joblib",
    "scikit-learn",
    "openpyxl",
    "pyarrow",
    "xgboost",
    "openai",
    "httpx",
]


def ensure_bundled_python() -> None:
    if not RUNTIME_PYTHON.exists():
        raise FileNotFoundError(
            f"Bundled Python 3.14 runtime is missing: {RUNTIME_PYTHON}. "
            "Please keep the runtime directory with this project."
        )
    current = Path(sys.executable).resolve()
    bundled = RUNTIME_PYTHON.resolve()
    if current != bundled:
        os.execv(str(bundled), [str(bundled), str(Path(__file__).resolve()), *sys.argv[1:]])
    if sys.version_info[:2] != (3, 14):
        raise RuntimeError(f"Bundled runtime must be Python 3.14, got {sys.version.split()[0]}.")


def info(message: str) -> None:
    print(f"[launcher] {message}", flush=True)


def build_frontend_package_list() -> list[str]:
    if not FRONTEND_REQUIREMENTS.exists():
        return ["pandas", "numpy", "scikit-learn", "joblib", "openpyxl", "matplotlib", "seaborn", "shap", "lightgbm", "xgboost"]
    packages: list[str] = []
    for line in FRONTEND_REQUIREMENTS.read_text(encoding="utf-8-sig").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        package_name = text.split("#", 1)[0].strip()
        if not package_name:
            continue
        if package_name.lower().startswith("streamlit"):
            continue
        packages.append(package_name)
    return packages


def unique_existing_paths(paths: Iterable[Path | str]) -> list[str]:
    result: list[str] = []
    for path in paths:
        text = str(path)
        if text and Path(text).exists() and text not in result:
            result.append(text)
    return result


def merge_pythonpath(extra_paths: Iterable[Path | str], base_env: dict[str, str] | None = None) -> str:
    env = base_env or os.environ
    existing = [item for item in env.get("PYTHONPATH", "").split(os.pathsep) if item]
    merged = unique_existing_paths(extra_paths) + [item for item in existing if item]
    return os.pathsep.join(merged)


def run_python_module_check(python_exe: str, extra_paths: list[str], modules: list[str]) -> list[str]:
    code = (
        "import importlib.util, json, sys\n"
        "extra = json.loads(sys.argv[1])\n"
        "modules = json.loads(sys.argv[2])\n"
        "for path in reversed(extra):\n"
        "    if path and path not in sys.path:\n"
        "        sys.path.insert(0, path)\n"
        "missing = [name for name in modules if importlib.util.find_spec(name) is None]\n"
        "print(json.dumps(missing, ensure_ascii=False))\n"
    )
    completed = subprocess.run(
        [python_exe, "-c", code, json.dumps(extra_paths, ensure_ascii=False), json.dumps(modules, ensure_ascii=False)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "模块检查失败。")
    try:
        payload = json.loads(completed.stdout.strip() or "[]")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"模块检查返回异常：{completed.stdout}") from exc
    return [str(item) for item in payload]


def run_python_smoke_check(python_exe: str, cwd: Path, extra_paths: list[str], code: str) -> None:
    bootstrap = (
        "import json, sys\n"
        "extra = json.loads(sys.argv[1])\n"
        "for path in reversed(extra):\n"
        "    if path and path not in sys.path:\n"
        "        sys.path.insert(0, path)\n"
    )
    completed = subprocess.run(
        [python_exe, "-c", bootstrap + code, json.dumps(extra_paths, ensure_ascii=False)],
        cwd=str(cwd),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "烟雾测试失败。"
        raise RuntimeError(detail)


def pip_install(python_exe: str, target: Path, packages: list[str]) -> None:
    if not packages:
        return
    target.mkdir(parents=True, exist_ok=True)
    info(f"安装依赖到 {target} ...")
    completed = subprocess.run(
        [python_exe, "-m", "pip", "install", "--upgrade", "--target", str(target), *packages],
        cwd=str(ROOT),
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"依赖安装失败，目标目录：{target}")


def ensure_streamlit_runtime(python_exe: str) -> None:
    if FRONTEND_RUNTIME_MARKER.exists():
        return
    info("检测到前端本地 Streamlit runtime 未展开，正在准备 ...")
    completed = subprocess.run([python_exe, str(FRONTEND_BOOTSTRAP)], cwd=str(FRONTEND_ROOT), check=False)
    if completed.returncode != 0 or not FRONTEND_RUNTIME_MARKER.exists():
        raise RuntimeError("前端 Streamlit runtime 准备失败。")


def ensure_frontend_dependencies(python_exe: str, auto_install: bool) -> None:
    required_modules = ["pandas", "numpy", "openpyxl"]
    optional_modules = ["joblib", "sklearn", "matplotlib", "seaborn", "shap", "lightgbm", "xgboost"]
    extra_paths = unique_existing_paths([FRONTEND_LOCAL_PACKAGES, BACKEND_LOCAL_PACKAGES])
    missing_required = run_python_module_check(python_exe, extra_paths, required_modules)
    missing_optional = run_python_module_check(python_exe, extra_paths, optional_modules)
    if not missing_required and not (auto_install and missing_optional):
        if missing_optional:
            info("前端可选依赖缺失：" + ", ".join(missing_optional) + "。页面可启动，但部分扩展能力可能受限。")
        return
    if not auto_install and missing_required:
        raise RuntimeError(
            "前端缺少启动依赖："
            + ", ".join(missing_required)
            + "。请运行“一键启动并补依赖.cmd”，或手动安装到 1\\.python_packages。"
        )
    pip_install(python_exe, FRONTEND_LOCAL_PACKAGES, build_frontend_package_list())
    all_missing_after = run_python_module_check(
        python_exe,
        unique_existing_paths([FRONTEND_LOCAL_PACKAGES, BACKEND_LOCAL_PACKAGES]),
        required_modules,
    )
    if all_missing_after:
        raise RuntimeError("前端依赖补装后仍缺失：" + ", ".join(all_missing_after))


def ensure_backend_dependencies(python_exe: str, auto_install: bool) -> None:
    extra_paths = unique_existing_paths([BACKEND_LOCAL_PACKAGES, BACKEND_ROOT])
    smoke_code = (
        "import run_harness_server\n"
        "import harness.harness_api\n"
        "import harness.frontend_fusion\n"
    )
    try:
        run_python_smoke_check(python_exe, BACKEND_ROOT, extra_paths, smoke_code)
        return
    except RuntimeError as exc:
        if not auto_install:
            raise RuntimeError(
                "后端启动环境检查未通过："
                + str(exc)
                + "。请运行“一键启动并补依赖.cmd”，或手动安装到 服创赛\\.deps3。"
            ) from exc
    pip_install(python_exe, BACKEND_LOCAL_PACKAGES, BACKEND_INSTALL_PACKAGES)
    run_python_smoke_check(python_exe, BACKEND_ROOT, unique_existing_paths([BACKEND_LOCAL_PACKAGES, BACKEND_ROOT]), smoke_code)


def wait_for_http(url: str, timeout_sec: float = 30.0) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.5) as response:
                if 200 <= response.status < 500:
                    return True
        except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError):
            time.sleep(1.0)
    return False


def read_log_tail(path: Path, max_lines: int = 20) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(lines[-max_lines:])


def terminate_process(proc: subprocess.Popen[bytes] | subprocess.Popen[str] | None, label: str) -> None:
    if proc is None or proc.poll() is not None:
        return
    info(f"正在关闭{label} ...")
    proc.terminate()
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def maybe_open_browser(url: str, enabled: bool) -> None:
    if not enabled:
        return
    if wait_for_http(url, timeout_sec=40):
        try:
            webbrowser.open(url)
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统一启动知行镜前后端。")
    parser.add_argument("--auto-install", action="store_true", help="缺依赖时自动安装到项目目录。")
    parser.add_argument("--check-only", action="store_true", help="仅检查启动条件，不真正启动服务。")
    parser.add_argument("--no-browser", action="store_true", help="启动后不自动打开浏览器。")
    parser.add_argument("--frontend-port", type=int, default=8501, help="前端端口，默认 8501。")
    parser.add_argument("--backend-port", type=int, default=8000, help="后端端口，默认 8000。")
    parser.add_argument("--host", default="127.0.0.1", help="前后端监听地址，默认 127.0.0.1。")
    return parser.parse_args()


def main() -> int:
    ensure_bundled_python()
    args = parse_args()
    python_exe = sys.executable

    if not FRONTEND_ENTRY.exists():
        raise FileNotFoundError(str(FRONTEND_ENTRY))
    if not BACKEND_ENTRY.exists():
        raise FileNotFoundError(str(BACKEND_ENTRY))

    TMP_ROOT.mkdir(parents=True, exist_ok=True)

    info(f"使用 Python: {python_exe}")
    ensure_streamlit_runtime(python_exe)
    ensure_backend_dependencies(python_exe, auto_install=args.auto_install)
    ensure_frontend_dependencies(python_exe, auto_install=args.auto_install)

    frontend_env = os.environ.copy()
    frontend_env["PYTHONPATH"] = merge_pythonpath([FRONTEND_LOCAL_PACKAGES, BACKEND_LOCAL_PACKAGES], frontend_env)
    frontend_env["HARNESS_BASE_URL"] = f"http://{args.host}:{args.backend_port}"
    frontend_env["STREAMLIT_SERVER_PORT"] = str(args.frontend_port)
    frontend_env["STREAMLIT_SERVER_ADDRESS"] = args.host

    backend_env = os.environ.copy()
    backend_env["PYTHONPATH"] = merge_pythonpath([BACKEND_LOCAL_PACKAGES], backend_env)
    backend_env["FRONTEND_ROOT"] = str(FRONTEND_ROOT)

    frontend_url = f"http://{args.host}:{args.frontend_port}"
    backend_health_url = f"http://{args.host}:{args.backend_port}/health"

    if args.check_only:
        info("检查完成。")
        info(f"前端入口: {FRONTEND_ENTRY}")
        info(f"后端入口: {BACKEND_ENTRY}")
        info(f"前端地址: {frontend_url}")
        info(f"后端健康检查: {backend_health_url}")
        return 0

    backend_stdout_path = TMP_ROOT / "launcher_backend_stdout.log"
    backend_stderr_path = TMP_ROOT / "launcher_backend_stderr.log"
    backend_stdout = backend_stdout_path.open("w", encoding="utf-8")
    backend_stderr = backend_stderr_path.open("w", encoding="utf-8")

    backend_proc: subprocess.Popen[str] | None = None
    frontend_proc: subprocess.Popen[str] | None = None
    try:
        info("正在启动后端 harness 服务 ...")
        backend_proc = subprocess.Popen(
            [python_exe, str(BACKEND_ENTRY), "--host", args.host, "--port", str(args.backend_port)],
            cwd=str(BACKEND_ROOT),
            env=backend_env,
            stdout=backend_stdout,
            stderr=backend_stderr,
            text=True,
        )
        if not wait_for_http(backend_health_url, timeout_sec=30):
            raise RuntimeError(
                "后端 harness 服务启动超时。\n"
                + ("stdout:\n" + read_log_tail(backend_stdout_path) + "\n" if backend_stdout_path.exists() else "")
                + ("stderr:\n" + read_log_tail(backend_stderr_path) if backend_stderr_path.exists() else "")
            )
        info(f"后端已启动：{backend_health_url}")

        info("正在启动前端 Streamlit ...")
        frontend_proc = subprocess.Popen(
            [python_exe, str(FRONTEND_ENTRY)],
            cwd=str(FRONTEND_ROOT),
            env=frontend_env,
            text=True,
        )
        info(f"前端地址：{frontend_url}")
        info("关闭当前终端时，会一并结束前后端。")
        maybe_open_browser(frontend_url, enabled=not args.no_browser)
        return frontend_proc.wait()
    except KeyboardInterrupt:
        info("收到中断信号，正在关闭服务。")
        return 130
    finally:
        terminate_process(frontend_proc, "前端")
        terminate_process(backend_proc, "后端")
        backend_stdout.close()
        backend_stderr.close()


if __name__ == "__main__":
    raise SystemExit(main())
