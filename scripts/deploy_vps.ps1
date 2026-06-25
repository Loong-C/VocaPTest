param(
    [string]$HostName = "187.77.136.20",
    [string]$User = "root",
    [string]$Branch = "master",
    [string]$RepoUrl = "https://github.com/Loong-C/VocaPTest.git",
    [string]$AppRoot = "/srv/vocaptest",
    [switch]$SkipModelSync
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$remote = "$User@$HostName"
$remoteTmp = "/tmp/vocaptest-update-server.sh"
$remoteAppDir = "$AppRoot/app"
$localUpdateScript = Join-Path $repoRoot "deploy/update_server.sh"
$localModelDir = Join-Path $repoRoot "data/processed/models"

function Invoke-Remote {
    param([string]$Command)
    ssh $remote $Command
}

Write-Host "[vocaptest-deploy] Uploading update script to $remote"
scp $localUpdateScript "${remote}:$remoteTmp"

Write-Host "[vocaptest-deploy] Running server update on $remote"
$remoteCommand = "chmod +x $remoteTmp && APP_ROOT='$AppRoot' REPO_URL='$RepoUrl' BRANCH='$Branch' $remoteTmp"
Invoke-Remote $remoteCommand

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
        scp $model.FullName "${remote}:$remoteAppDir/data/processed/models/"
    }

    Write-Host "[vocaptest-deploy] Restarting service after model sync"
    Invoke-Remote "systemctl restart vocaptest"
}

Write-Host "[vocaptest-deploy] Checking service health"
Invoke-Remote "systemctl --no-pager --full status vocaptest | sed -n '1,18p'"
Invoke-Remote "curl -fsS http://127.0.0.1:8000/health"

Write-Host "[vocaptest-deploy] Done: https://linkukai.com/VocaPTest/"
