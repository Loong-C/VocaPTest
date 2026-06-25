param(
    [string]$HostName = "187.77.136.20",
    [string]$User = "root",
    [string]$Branch = "master",
    [string]$RepoUrl = "https://github.com/Loong-C/VocaPTest.git",
    [string]$AppRoot = "/srv/vocaptest",
    [switch]$SkipModelSync,
    [switch]$SkipSystemPackages,
    [switch]$SkipPythonDeps,
    [switch]$SkipServiceInstall,
    [switch]$SkipNginxInstall,
    [switch]$RunUpdateInForeground,
    [int]$RemoteUpdateTimeoutMinutes = 20,
    [int]$RemotePollSeconds = 5
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$remote = "$User@$HostName"
$remoteTmp = "/tmp/vocaptest-update-server.sh"
$remoteAppDir = "$AppRoot/app"
$localUpdateScript = Join-Path $repoRoot "deploy/update_server.sh"
$localModelDir = Join-Path $repoRoot "data/processed/models"
$SshOptions = @(
    "-o", "ServerAliveInterval=15",
    "-o", "ServerAliveCountMax=20"
)

function Invoke-Remote {
    param([string]$Command)
    ssh @SshOptions $remote $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Remote command failed with exit code $LASTEXITCODE"
    }
}

function Invoke-Scp {
    param(
        [string]$Source,
        [string]$Destination
    )
    scp @SshOptions $Source $Destination
    if ($LASTEXITCODE -ne 0) {
        throw "SCP failed with exit code $LASTEXITCODE"
    }
}

function Invoke-RemoteUpdate {
    param([string]$Command)

    if ($RunUpdateInForeground) {
        Invoke-Remote $Command
        return
    }

    $stamp = Get-Date -Format "yyyyMMddHHmmss"
    $remoteLog = "/tmp/vocaptest-deploy-$stamp.log"
    $remoteStatus = "$remoteLog.status"
    $startCommand = "bash -lc 'rm -f $remoteLog $remoteStatus; ( $Command > $remoteLog 2>&1; echo `$? > $remoteStatus ) </dev/null >/dev/null 2>&1 & echo `$!'"

    $remotePid = Invoke-Remote $startCommand | Select-Object -Last 1
    Write-Host "[vocaptest-deploy] Remote update PID: $remotePid"
    Write-Host "[vocaptest-deploy] Remote update log: $remoteLog"

    $deadline = (Get-Date).AddMinutes($RemoteUpdateTimeoutMinutes)
    $lastTail = ""
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds $RemotePollSeconds

        $status = Invoke-Remote "test -f $remoteStatus && cat $remoteStatus || echo RUNNING" | Select-Object -Last 1
        $tail = Invoke-Remote "tail -40 $remoteLog 2>/dev/null || true"
        $tailText = $tail -join "`n"
        if ($tailText -and $tailText -ne $lastTail) {
            Write-Host $tailText
            $lastTail = $tailText
        }

        if ($status -ne "RUNNING") {
            if ($status -ne "0") {
                throw "Remote update failed with exit code $status. See $remoteLog"
            }
            return
        }
    }

    throw "Remote update timed out after $RemoteUpdateTimeoutMinutes minutes. See $remoteLog"
}

Write-Host "[vocaptest-deploy] Uploading update script to $remote"
Invoke-Scp $localUpdateScript "${remote}:$remoteTmp"

Write-Host "[vocaptest-deploy] Running server update on $remote"
$remoteEnv = @(
    "APP_ROOT=$AppRoot",
    "REPO_URL=$RepoUrl",
    "BRANCH=$Branch"
)
if ($SkipSystemPackages) { $remoteEnv += "SKIP_SYSTEM_PACKAGES=1" }
if ($SkipPythonDeps) { $remoteEnv += "SKIP_PYTHON_DEPS=1" }
if ($SkipServiceInstall) { $remoteEnv += "SKIP_SERVICE_INSTALL=1" }
if ($SkipNginxInstall) { $remoteEnv += "SKIP_NGINX_INSTALL=1" }
$remoteCommand = "chmod +x $remoteTmp && $($remoteEnv -join ' ') $remoteTmp"
Invoke-RemoteUpdate $remoteCommand

if (-not $SkipModelSync) {
    if (-not (Test-Path $localModelDir)) {
        throw "Model directory not found: $localModelDir"
    }

    $models = Get-ChildItem -LiteralPath $localModelDir -Filter "*.pkl" -File
    if ($models.Count -eq 0) {
        throw "No .pkl model artifacts found in $localModelDir"
    }

    Write-Host "[vocaptest-deploy] Syncing model artifacts"
    Invoke-Remote "mkdir -p '$remoteAppDir/data/processed/models'"
    foreach ($model in $models) {
        Invoke-Scp $model.FullName "${remote}:$remoteAppDir/data/processed/models/"
    }

    Write-Host "[vocaptest-deploy] Restarting service after model sync"
    Invoke-Remote "systemctl restart vocaptest"
}

Write-Host "[vocaptest-deploy] Checking service health"
Invoke-Remote "systemctl --no-pager --full status vocaptest | sed -n '1,18p'"
Invoke-Remote "curl -fsS http://127.0.0.1:8000/health"

Write-Host "[vocaptest-deploy] Done: https://linkukai.com/VocaPTest/"
