$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$Port = 8000
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$OnnxPath = Join-Path $ProjectRoot "models\weights\voiceprint_model.onnx"
$BestModelPath = Join-Path $ProjectRoot "models\weights\best_model.pth"
$RequirementsPath = Join-Path $ProjectRoot "requirements.txt"
$FfmpegPath = Join-Path $ProjectRoot "bin\ffmpeg.exe"
$DepsStampPath = Join-Path $ProjectRoot ".venv\.deps_installed"

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host $Message -ForegroundColor Cyan
}

function Require-Command([string]$Name) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command not found: $Name"
    }
}

Write-Step "[1/6] Project root: $ProjectRoot"

if (-not (Test-Path $VenvPython)) {
    Write-Step "[2/6] Creating virtual environment..."
    Require-Command "python"
    & python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create virtual environment."
    }
} else {
    Write-Step "[2/6] Virtual environment already exists."
}

Write-Step "[3/6] Checking Python dependencies..."
$NeedInstallDeps = $true

if ((Test-Path $DepsStampPath) -and (Test-Path $RequirementsPath)) {
    $DepsStampTime = (Get-Item $DepsStampPath).LastWriteTimeUtc
    $RequirementsTime = (Get-Item $RequirementsPath).LastWriteTimeUtc
    if ($DepsStampTime -ge $RequirementsTime) {
        $NeedInstallDeps = $false
    }
}

if ($NeedInstallDeps) {
    Write-Step "[3/6] Installing dependencies from requirements.txt..."
    & $VenvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to upgrade pip."
    }
    & $VenvPython -m pip install -r $RequirementsPath
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install requirements."
    }
    Set-Content -Path $DepsStampPath -Value (Get-Date).ToString("o") -Encoding utf8
} else {
    Write-Step "[3/6] Dependencies are already installed."
}

Write-Step "[4/6] Checking model files..."
if (-not (Test-Path $OnnxPath)) {
    if (Test-Path $BestModelPath) {
        Write-Host "ONNX model not found. Exporting from best_model.pth..." -ForegroundColor Yellow
        & $VenvPython export_only.py
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to export ONNX model."
        }
    } else {
        throw "Neither voiceprint_model.onnx nor best_model.pth was found under models\weights."
    }
} else {
    Write-Step "[4/6] ONNX model is ready."
}

if (-not (Test-Path $FfmpegPath)) {
    Write-Warning "ffmpeg.exe was not found under bin\. Voice registration/login may fail for some webm recordings."
}

Write-Step "[5/6] Checking port $Port..."
try {
    $PortInUse = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
} catch {
    $PortInUse = $null
}

if ($PortInUse) {
    throw "Port $Port is already in use. Stop the existing service or change the port in start_project.ps1."
}

Write-Step "[6/6] Starting service..."
Write-Host "URL: http://127.0.0.1:$Port" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the server." -ForegroundColor Green

Start-Process powershell -WindowStyle Hidden -ArgumentList @(
    "-NoProfile",
    "-Command",
    "Start-Sleep -Seconds 3; Start-Process 'http://127.0.0.1:$Port'"
) | Out-Null

& $VenvPython -m uvicorn backend.main:app --host 127.0.0.1 --port $Port
