# EduPulse Demo Key Upload

This repository contains a lightweight key upload of the EduPulse A14 demo.

The upload keeps the project structure, source code, startup scripts, core configuration files, and selected CSV/JSON/Markdown deliverables needed to review the demo logic.

## Included

- Streamlit demo entry under `1/`
- study / life / sport / harness source files under `服创赛/`
- startup scripts: `launch_all.py`, `启动系统.ps1`, `启动说明.md`
- selected generated outputs used for demo review
- upload manifest and Git ignore rules

## Excluded

To keep the GitHub repository small and auditable, this key upload excludes:

- raw Excel data
- local Python runtime bundles
- model binaries and native libraries
- large generated frontend bundles
- historical harness run archives
- cache files and bytecode

The full local demo remains unchanged. Large raw data and runtime artifacts should be shared separately through Git LFS or GitHub Releases if needed.
