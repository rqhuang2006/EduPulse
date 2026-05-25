param(
    [switch]$AutoInstall,
    [switch]$AutoInstallPython,
    [switch]$CheckOnly,
    [switch]$NoBrowser,
    [int]$FrontendPort = 8501,
    [int]$BackendPort = 8000
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$RuntimePython = Join-Path $Root "runtime\python.exe"
$Launcher = Join-Path $Root "launch_all.py"

if ($AutoInstallPython) {
    Write-Host "[launcher] Bundled Python is used; system Python installation is skipped." -ForegroundColor Yellow
}

if (-not (Test-Path -LiteralPath $RuntimePython)) {
    throw "Bundled Python 3.14 runtime is missing: $RuntimePython"
}

$Version = & $RuntimePython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
if (-not $Version.StartsWith("3.14.")) {
    throw "Bundled runtime must be Python 3.14, got $Version"
}

$ArgsList = @($Launcher, "--frontend-port", "$FrontendPort", "--backend-port", "$BackendPort")
if ($AutoInstall) {
    $ArgsList += "--auto-install"
}
if ($CheckOnly) {
    $ArgsList += "--check-only"
}
if ($NoBrowser) {
    $ArgsList += "--no-browser"
}

Write-Host "[launcher] Using bundled Python: $RuntimePython" -ForegroundColor Cyan
& $RuntimePython @ArgsList
exit $LASTEXITCODE
