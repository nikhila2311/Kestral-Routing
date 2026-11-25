# Clean-machine integration test (Windows PowerShell 5.1 or later). macOS/Linux: scripts/integration_test.sh
#
# Usage, from anywhere:
#   powershell -ExecutionPolicy Bypass -File scripts\integration_test.ps1
#   options: -RawDir <folder with the data pack>  -Port <free port>  -Keep (leave the temp folder; it holds customer data)
#
# Tests the COMMITTED state (HEAD) in a fresh clone with a fresh venv. Nothing in this working copy is changed.
param(
    [string]$RawDir = "",
    [int]$Port = 8765,
    [switch]$Keep
)

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $RawDir) { $RawDir = Join-Path $Repo "data\raw" }
$Work = Join-Path ([IO.Path]::GetTempPath()) ("kestrel_it_" + [guid]::NewGuid().ToString("N").Substring(0, 8))
$Clone = Join-Path $Work "repo"
$Logs = Join-Path $Work "logs"
$VenvPython = Join-Path $Clone ".venv\Scripts\python.exe"
$Url = "http://127.0.0.1:$Port"
$Results = New-Object System.Collections.ArrayList
$script:Failed = $false
$script:Server = $null

function Invoke-Logged {
    # Runs a program, sends its output to a log file, throws on a non-zero exit code.
    param([string]$Name, [string]$Exe, [string[]]$Arguments, [string]$Dir = $Clone)
    $out = Join-Path $Logs "$Name.out.log"
    $err = Join-Path $Logs "$Name.err.log"
    $quoted = $Arguments | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }
    $process = Start-Process -FilePath $Exe -ArgumentList $quoted -WorkingDirectory $Dir -NoNewWindow -PassThru `
        -RedirectStandardOutput $out -RedirectStandardError $err
    $null = $process.Handle  # keeps the exit code readable after the process ends
    $process.WaitForExit()
    if ($process.ExitCode -ne 0) {
        Get-Content $out, $err -ErrorAction SilentlyContinue | Select-Object -Last 30 | ForEach-Object { Write-Host "    $_" }
        throw "$Name exited with code $($process.ExitCode)"
    }
}

function Invoke-Step {
    # After the first failure the remaining steps are reported as SKIP.
    param([string]$Name, [scriptblock]$Body)
    if ($script:Failed) {
        [void]$Results.Add([pscustomobject]@{ Status = "SKIP"; Step = $Name; Seconds = 0 })
        Write-Host "SKIP  $Name"
        return
    }
    $watch = [Diagnostics.Stopwatch]::StartNew()
    $status = "PASS"
    $note = ""
    try { $global:LASTEXITCODE = 0; & $Body | Out-Null }
    catch { $status = "FAIL"; $note = " - $($_.Exception.Message)"; $script:Failed = $true }
    $seconds = [int]$watch.Elapsed.TotalSeconds
    [void]$Results.Add([pscustomobject]@{ Status = $status; Step = $Name; Seconds = $seconds })
    Write-Host "$status  $Name (${seconds}s)$note"
}

function Test-PortOpen {
    $client = New-Object Net.Sockets.TcpClient
    try { $client.Connect("127.0.0.1", $Port); return $true } catch { return $false } finally { $client.Close() }
}

function Stop-Server {
    if ($script:Server -and -not $script:Server.HasExited) {
        # The venv python.exe starts a child process; /T stops both.
        $null = Start-Process taskkill.exe -ArgumentList "/PID", $script:Server.Id, "/T", "/F" -WindowStyle Hidden -Wait
        $null = $script:Server.WaitForExit(10000)
    }
}

function Find-Python {
    foreach ($candidate in @(@("py", "-3.13"), @("py", "-3.12"), @("python"), @("python3"))) {
        if (-not (Get-Command $candidate[0] -ErrorAction SilentlyContinue)) { continue }
        $rest = @($candidate | Select-Object -Skip 1)
        $ErrorActionPreference = "Continue"
        $exe = & $candidate[0] @rest -c "import sys; print(sys.executable if sys.version_info[:2] >= (3, 12) else '')" 2>$null
        $ErrorActionPreference = "Stop"
        if ($exe) { return "$exe".Trim() }
    }
    throw "Python 3.12 or newer not found"
}

function Get-GitLines {
    param([string[]]$Arguments)
    $lines = & git -C $Clone @Arguments
    if ($LASTEXITCODE -ne 0) { throw "git $($Arguments -join ' ') failed" }
    return @($lines)
}

$env:PYTHONHASHSEED = "42"
New-Item -ItemType Directory -Path $Logs -Force | Out-Null
Write-Host "Repo:  $Repo"
Write-Host "Temp:  $Work"
if (& git -C $Repo status --porcelain) { Write-Host "NOTE   uncommitted changes exist; only the last commit is tested." }

try {
    Invoke-Step "clone" { Invoke-Logged "clone" "git" @("clone", "--quiet", "--no-hardlinks", $Repo, $Clone) $Work }

    Invoke-Step "git tracking (data/ and models/ untracked, predictions.csv tracked)" {
        if (Get-GitLines @("ls-files", "data", "models")) { throw "a file under data/ or models/ is tracked" }
        if (-not (Get-GitLines @("ls-files", "predictions.csv"))) { throw "predictions.csv is not tracked" }
    }

    Invoke-Step "copy data/raw" {
        if (-not (Test-Path (Join-Path $RawDir "train.csv"))) { throw "data pack not found in $RawDir" }
        $target = Join-Path $Clone "data\raw"
        New-Item -ItemType Directory -Path $target -Force | Out-Null
        Copy-Item -Path (Join-Path $RawDir "*") -Destination $target -Recurse
    }

    Invoke-Step "fresh venv" { Invoke-Logged "venv" (Find-Python) @("-m", "venv", ".venv") }

    Invoke-Step "pip install -r requirements.txt" {
        Invoke-Logged "pip" $VenvPython @("-m", "pip", "install", "--quiet", "--disable-pip-version-check", "-r", "requirements.txt")
    }

    # predictions.csv is deleted first, so the last step cannot pass on the committed copy.
    Invoke-Step "remove committed predictions.csv from the clone" { Remove-Item (Join-Path $Clone "predictions.csv") }

    $pipeline = @(
        @("clean", "src.clean"), @("audit", "src.audit"), @("baselines", "src.baselines"),
        @("experiments", "src.experiments"), @("router", "src.router"), @("evaluate", "src.evaluation"),
        @("train", "src.predictions", "train"), @("predict", "src.predictions", "predict")
    )
    foreach ($entry in $pipeline) {
        Invoke-Step "pipeline: $($entry[0])" {
            Invoke-Logged $entry[0] $VenvPython (@("-m") + @($entry | Select-Object -Skip 1))
        }
    }

    Invoke-Step "predictions.csv byte-identical to the committed one" {
        # --no-filters hashes the raw bytes: a CRLF change counts as a difference.
        $new = Get-GitLines @("hash-object", "--no-filters", "predictions.csv")
        $committed = Get-GitLines @("rev-parse", "HEAD:predictions.csv")
        if ("$new" -ne "$committed") { throw "differs from HEAD" }
    }

    Invoke-Step "quick tests (pytest -m 'not slow')" {
        Invoke-Logged "pytest" $VenvPython @("-m", "pytest", "tests", "-q", "-m", "not slow")
    }

    Invoke-Step "start API" {
        if (Test-PortOpen) { throw "port $Port is already in use; pass -Port" }
        $script:Server = Start-Process -FilePath $VenvPython -WorkingDirectory $Clone -WindowStyle Hidden -PassThru `
            -ArgumentList "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", $Port `
            -RedirectStandardOutput (Join-Path $Logs "api.out.log") -RedirectStandardError (Join-Path $Logs "api.err.log")
        $null = $script:Server.Handle
        for ($i = 0; -not (Test-PortOpen); $i++) {
            if ($script:Server.HasExited) { throw "the API process exited" }
            if ($i -ge 60) { throw "the API did not start in 60 s" }
            Start-Sleep -Seconds 1
        }
    }

    Invoke-Step "GET /health" {
        $health = Invoke-RestMethod "$Url/health" -TimeoutSec 15
        if ($health.status -ne "ok" -or $health.model_loaded -ne $true) { throw "unexpected answer" }
    }

    Invoke-Step "POST /predict" {
        $body = '{"request_text": "My mixer stopped working"}'
        $answer = Invoke-RestMethod "$Url/predict" -Method Post -ContentType "application/json" -Body $body -TimeoutSec 15
        $teams = & $VenvPython -c "import sys; sys.path.insert(0, sys.argv[1]); from src.config import TEAMS; print('|'.join(TEAMS))" $Clone
        if (-not $answer.team -or $teams.Split("|") -notcontains $answer.team) { throw "no canonical team in the answer" }
    }

    Invoke-Step "stop API" {
        Stop-Server
        Start-Sleep -Seconds 1
        if (Test-PortOpen) { throw "port $Port still answers" }
    }
}
finally {
    Stop-Server
    if ($Keep) {
        Write-Host "Kept $Work (contains customer data; delete it when done)"
    }
    else {
        for ($i = 0; $i -lt 5 -and (Test-Path $Work); $i++) {
            try { Remove-Item -Recurse -Force $Work -ErrorAction Stop } catch { Start-Sleep -Seconds 2 }
        }
        if (Test-Path $Work) { Write-Host "WARNING could not delete $Work (contains customer data; delete it by hand)" }
    }
}

Write-Host ""
$Results | ForEach-Object { Write-Host ("{0}  {1}" -f $_.Status, $_.Step) }
if ($script:Failed) { Write-Host "RESULT: FAIL"; exit 1 }
Write-Host "RESULT: PASS"
exit 0
