<# : LRA-MAKR one-click installer for Windows (batch + PowerShell in one file)
@echo off
setlocal
title LRA-MAKR trainer setup
powershell -NoProfile -ExecutionPolicy Bypass -Command "$f='%~f0'; & ([ScriptBlock]::Create([IO.File]::ReadAllText($f)))"
echo.
pause
exit /b
#>

# Installs the LRA-MAKR local trainer (ostris/ai-toolkit + a small helper
# server) into %LOCALAPPDATA%\LRA-MAKR, adds "LRA-MAKR Trainer" shortcuts and
# starts it. Safe to run again: it updates the trainer and keeps downloads.
#
# Optional environment variables:
#   LRA_BRANCH   which LRA-MAKR branch to install (default: main)
#   TORCH_INDEX  PyTorch wheel index (default: CUDA 13.0 builds)

# Windows PowerShell 5.1 turns stderr from tools like "py" or "git" into
# terminating errors under "Stop", so failures are checked explicitly instead
$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"  # makes Invoke-WebRequest much faster

$Repo = "yrw2go-ui/LRA-MAKR"
$Branch = if ($env:LRA_BRANCH) { $env:LRA_BRANCH } else { "main" }
$AppUrl = "https://lora-makr.vercel.app"
$Root = Join-Path $env:LOCALAPPDATA "LRA-MAKR"
$TrainerDir = Join-Path $Root "local-trainer"

function Step($text) { Write-Host "`n==> $text" -ForegroundColor Cyan }
function Info($text) { Write-Host "    $text" }
function Warn($text) { Write-Host "    $text" -ForegroundColor Yellow }
function Has($cmd) { [bool](Get-Command $cmd -ErrorAction SilentlyContinue) }

function Refresh-Path {
    $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $user = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$machine;$user"
}

function Install-WithWinget($id, $label) {
    if (-not (Has "winget")) {
        throw "$label is missing and winget isn't available to install it. Install $label yourself, then run this installer again."
    }
    Info "Installing $label (Windows may ask for permission)..."
    winget install --id $id --exact --silent --accept-source-agreements --accept-package-agreements | Out-Host
    Refresh-Path
}

# Returns the path of a Python 3.10-3.12 interpreter, or $null. Uses the
# "py" launcher, since plain "python" can be the Microsoft Store stub.
function Find-Python {
    foreach ($version in "3.12", "3.11", "3.10") {
        if (Has "py") {
            $exe = & py "-$version" -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $exe) { return "$exe".Trim() }
        }
        $guess = Join-Path $env:LOCALAPPDATA ("Programs\Python\Python" + $version.Replace(".", "") + "\python.exe")
        if (Test-Path $guess) { return $guess }
    }
    return $null
}

try {
    Write-Host "LRA-MAKR local trainer installer" -ForegroundColor Green
    Write-Host "Installs to $Root"

    Step "Checking your graphics card"
    if (Has "nvidia-smi") {
        $gpu = (& nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>$null | Select-Object -First 1)
        Info "Found: $gpu"
    } else {
        Warn "No NVIDIA graphics card found (nvidia-smi is missing)."
        Warn "Training needs an NVIDIA GPU with recent drivers."
        $answer = Read-Host "    Continue anyway? [y/N]"
        if ($answer -notmatch "^[yY]") { throw "Cancelled." }
    }

    Step "Checking Python"
    $python = Find-Python
    if (-not $python) {
        Install-WithWinget "Python.Python.3.12" "Python 3.12"
        $python = Find-Python
        if (-not $python) { throw "Python was installed but can't be found yet. Close this window and run the installer again." }
    }
    Info "Using $python"

    Step "Checking Git"
    if (-not (Has "git")) {
        Install-WithWinget "Git.Git" "Git"
        if (-not (Has "git")) { throw "Git was installed but can't be found yet. Close this window and run the installer again." }
    }
    Info ("Using " + (& git --version))

    Step "Downloading the LRA-MAKR trainer ($Branch)"
    $tmp = Join-Path ([IO.Path]::GetTempPath()) ("lra-makr-" + [Guid]::NewGuid())
    New-Item -ItemType Directory -Path $tmp | Out-Null
    $zip = Join-Path $tmp "lra-makr.zip"
    Invoke-WebRequest -UseBasicParsing "https://github.com/$Repo/archive/refs/heads/$Branch.zip" -OutFile $zip -ErrorAction Stop
    Expand-Archive -Path $zip -DestinationPath $tmp -Force -ErrorAction Stop
    $source = Get-ChildItem $tmp -Directory | Where-Object { Test-Path (Join-Path $_.FullName "local-trainer\server.py") } | Select-Object -First 1
    if (-not $source) { throw "The download didn't contain the trainer. Check the branch name ($Branch)." }
    New-Item -ItemType Directory -Path $TrainerDir -Force -ErrorAction Stop | Out-Null
    # update the scripts but keep ai-toolkit and its downloads
    Copy-Item -Path (Join-Path $source.FullName "local-trainer\*") -Destination $TrainerDir -Recurse -Force -ErrorAction Stop
    Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
    Info "Saved to $TrainerDir"

    Step "Installing ai-toolkit and PyTorch (10-30 minutes, several GB)"
    $env:PYTHON = $python
    $env:LRA_NO_PAUSE = "1"
    $toolkit = Join-Path $TrainerDir "ai-toolkit"
    if (Test-Path (Join-Path $toolkit ".git")) {
        Info "Updating the existing ai-toolkit"
        & git -C $toolkit pull --ff-only | Out-Host
    }
    & cmd /c "`"$(Join-Path $TrainerDir 'setup.bat')`""
    if ($LASTEXITCODE -ne 0) { throw "Setup failed; see the messages above." }

    Step "Hugging Face sign-in (optional)"
    Info "Needed only for gated models: FLUX.1 dev, FLUX.2, FLUX.2 klein 9B, Krea 2, LTX-2.5."
    $answer = Read-Host "    Sign in now? You'll need an access token from huggingface.co/settings/tokens [y/N]"
    if ($answer -match "^[yY]") {
        $scripts = Join-Path $toolkit "venv\Scripts"
        $hf = @("hf.exe", "huggingface-cli.exe") | ForEach-Object { Join-Path $scripts $_ } | Where-Object { Test-Path $_ } | Select-Object -First 1
        if ($hf -and $hf.EndsWith("hf.exe")) { & $hf auth login } elseif ($hf) { & $hf login } else { Warn "Couldn't find the Hugging Face login tool; skipping." }
    }

    Step "Adding LRA-MAKR Trainer shortcuts"
    $start = Join-Path $TrainerDir "start.bat"
    try {
        $shell = New-Object -ComObject WScript.Shell
        $places = @(
            [Environment]::GetFolderPath("Desktop"),
            (Join-Path ([Environment]::GetFolderPath("StartMenu")) "Programs")
        )
        foreach ($place in $places) {
            $link = $shell.CreateShortcut((Join-Path $place "LRA-MAKR Trainer.lnk"))
            $link.TargetPath = $start
            $link.WorkingDirectory = $TrainerDir
            $link.Description = "Start the LRA-MAKR local trainer"
            $link.Save()
        }
        Info "Added to your Desktop and Start menu"
    } catch {
        Warn "Couldn't create shortcuts ($($_.Exception.Message)). Start it with: $start"
    }

    Step "Starting the trainer"
    Start-Process -FilePath "cmd.exe" -ArgumentList "/k", "`"$start`"" -WorkingDirectory $TrainerDir
    Write-Host ""
    Write-Host "All set! Keep the trainer window open while you train." -ForegroundColor Green
    Write-Host "Open $AppUrl, choose 'Train on: This PC', and click Retry."
    Write-Host "Next time, start it from the 'LRA-MAKR Trainer' shortcut."
} catch {
    Write-Host ""
    Write-Host "Setup stopped: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "You can run this installer again; it picks up where it left off."
}
