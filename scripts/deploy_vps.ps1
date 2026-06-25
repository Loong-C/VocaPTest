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
    scp -O @SshOptions $Source $Destination
    if ($LASTEXITCODE -ne 0) {
        throw "SCP failed with exit code $LASTEXITCODE"
    }
}

function Invoke-SshTransfer {
    param(
        [string]$Source,
        [string]$Destination,
        [switch]$StripCR
    )

    $remoteCmd = "cat > '$Destination'"
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = "ssh"
    $psi.Arguments = "-o ServerAliveInterval=15 -o ServerAliveCountMax=20 $remote $remoteCmd"
    $psi.RedirectStandardInput = $true
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true

    $proc = [System.Diagnostics.Process]::Start($psi)
    $fileStream = [System.IO.File]::OpenRead($Source)
    $buf = New-Object byte[] (1MB)
    if ($StripCR) {
        $outBuf = New-Object byte[] (1MB)
        while (($read = $fileStream.Read($buf, 0, $buf.Length)) -gt 0) {
            $outLen = 0
            for ($i = 0; $i -lt $read; $i++) {
                if ($buf[$i] -ne 13) {
                    $outBuf[$outLen] = $buf[$i]
                    $outLen++
                }
            }
            if ($outLen -gt 0) {
                $proc.StandardInput.BaseStream.Write($outBuf, 0, $outLen)
            }
        }
    }
    else {
        while (($read = $fileStream.Read($buf, 0, $buf.Length)) -gt 0) {
            $proc.StandardInput.BaseStream.Write($buf, 0, $read)
        }
    }
    $fileStream.Close()
    $proc.StandardInput.Close()
    $proc.WaitForExit()

    if ($proc.ExitCode -ne 0) {
        throw "SSH file transfer failed with exit code $($proc.ExitCode)"
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
Invoke-SshTransfer -StripCR $localUpdateScript $remoteTmp

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
        Write-Host "[vocaptest-deploy]   Transferring $($model.Name) ($([math]::Round($model.Length/1MB, 1)) MB)..."
        $remoteModelPath = "$remoteAppDir/data/processed/models/$($model.Name)"
        Invoke-SshTransfer $model.FullName $remoteModelPath
    }

    Write-Host "[vocaptest-deploy] Restarting service after model sync"
    Invoke-Remote "systemctl restart vocaptest"
}

Write-Host "[vocaptest-deploy] Checking service health"
Invoke-Remote "systemctl --no-pager --full status vocaptest | sed -n '1,18p'"
Invoke-Remote "curl -fsS http://127.0.0.1:8000/health"

Write-Host "[vocaptest-deploy] Done: https://linkukai.com/VocaPTest/"
