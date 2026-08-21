# Windows Parallel Execution Script
#
# ── Run ────────────────────────────────────────────────────────────────────────
# powershell -ExecutionPolicy Bypass -File Algorithms\Run\run_windows.ps1
#
# ── Monitor single task (real-time, last 20 lines) ─────────────────────────────
# Get-Content Algorithms\Run\logs\func_0.log -Tail 20
#
# ── Monitor all tasks (latest line of each log) ────────────────────────────────
# 0..8 | ForEach-Object { $last = Get-Content "Algorithms\Run\logs\func_$_.log" -Tail 1 -ErrorAction SilentlyContinue; Write-Host "func_$_ : $last" }
#
# ── Follow a log live (Ctrl+C to stop) ─────────────────────────────────────────
# Get-Content Algorithms\Run\logs\func_0.log -Wait -Tail 20
#
# ── Task range reference ────────────────────────────────────────────────────────
# seq 0-8:  CEC17-MTSO   | seq 9-18:  WCCI20-MTSO  | seq 19-27: C2TOP
# seq 28-33: CEC19-MaTSO | seq 34-43: WCCI20-MaTSO
# seq 44: PEPVM | seq 45: PKACP | seq 46-59: MRNP
# ───────────────────────────────────────────────────────────────────────────────


# ==================== Configuration ====================
$funcRange   = 0..8     # 0-8=CEC17, 9-18=WCCI20, 19-27=C2TOP
$maxParallel = $null     # $null=unlimited; set e.g. 4 to limit

# ==================== Path Setup ====================
$scriptDir    = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot  = (Resolve-Path "$scriptDir\..\.." ).Path
$outputScript = "$scriptDir\output.py"
$logDir       = "$scriptDir\logs"

# Find real Python interpreter (not venv launcher)
# Priority: venv site-packages python > system python
$venvCfg = Get-Content "$projectRoot\.venv\pyvenv.cfg" -ErrorAction SilentlyContinue
$basePython = $null
if ($venvCfg) {
    $homeLine = ($venvCfg | Where-Object { $_ -match "^base-executable\s*=" }) -replace ".*=\s*", ""
    if ($homeLine -and (Test-Path $homeLine.Trim())) {
        $basePython = $homeLine.Trim()
    }
}
# Fallback: find python in PATH that is NOT the venv launcher
if (-not $basePython) {
    $allPython = where.exe python 2>$null
    foreach ($p in $allPython) {
        if ($p -notmatch "\.venv" -and $p -notmatch "WindowsApps" -and (Test-Path $p)) {
            $basePython = $p
            break
        }
    }
}
if (-not $basePython) { $basePython = "python" }

# Build command that activates venv site-packages then runs script
# We use the base python but add venv site-packages to PYTHONPATH
$venvSitePackages = "$projectRoot\.venv\Lib\site-packages"
$pythonCmd = $basePython

New-Item -ItemType Directory -Path $logDir -Force | Out-Null

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Project Root    : $projectRoot"
Write-Host "Python          : $pythonCmd"
Write-Host "Venv Packages   : $venvSitePackages"
Write-Host "Log Dir         : $logDir"
Write-Host "Task Range      : $($funcRange[0]) ~ $($funcRange[-1])"
if ($maxParallel) { Write-Host "Max Parallel    : $maxParallel" } else { Write-Host "Max Parallel    : Unlimited" }
Write-Host "==========================================" -ForegroundColor Cyan

# ==================== Start Tasks ====================
$processes = @()
$batchCount = 0

foreach ($funcNum in $funcRange) {
    $logFile = "$logDir\func_$funcNum.log"
    $ts      = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$ts] Starting func=$funcNum  ->  $logFile"

    # Set PYTHONPATH to include venv site-packages so imports work
    $env:PYTHONPATH = $venvSitePackages

    # -u: unbuffered output so logs appear immediately
    $proc = Start-Process -FilePath $pythonCmd `
        -ArgumentList "-u `"$outputScript`" --func $funcNum" `
        -WorkingDirectory $projectRoot `
        -RedirectStandardOutput $logFile `
        -RedirectStandardError "$logDir\func_${funcNum}_err.log" `
        -PassThru `
        -WindowStyle Hidden

    $processes += $proc
    $batchCount++

    if ($maxParallel -and ($batchCount % $maxParallel -eq 0)) {
        Write-Host "Waiting for batch to finish..." -ForegroundColor Yellow
        $processes | ForEach-Object { $_.WaitForExit() }
        $processes = @()
        Write-Host "Batch done, starting next..." -ForegroundColor Green
    }
}

# Wait for remaining processes
if ($processes.Count -gt 0) {
    Write-Host "Waiting for all tasks to complete (this may take a long time)..." -ForegroundColor Yellow
    $processes | ForEach-Object { $_.WaitForExit() }
}

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] All tasks completed!"
Write-Host "View log: Get-Content $logDir"
Write-Host "==========================================" -ForegroundColor Cyan
