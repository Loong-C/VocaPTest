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
    [int]$RemotePollSeconds = 15
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
    "-o", "ServerAliveCountMax=20",
    "-o", "ConnectTimeout=10"
)

function Invoke-Remote {
    param([string]$Command)
    ssh @SshOptions $remote $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Remote command failed with exit code $LASTEXITCODE"
    }
}

function Invoke-ScpTransfer {
    param(
        [string]$Source,
        [string]$Destination,
        [int]$MaxRetries = 3,
        [int]$RetryDelaySeconds = 5
    )

    $attempt = 0
    while ($true) {
        $attempt++
        try {
            scp -O @SshOptions $Source $Destination
            if ($LASTEXITCODE -eq 0) { return }
        }
        catch { }

        if ($attempt -ge $MaxRetries) {
            throw "SCP failed after $MaxRetries attempts (exit code $LASTEXITCODE)"
        }
        Write-Host "[vocaptest-deploy]   SCP retry $attempt/$MaxRetries in ${RetryDelaySeconds}s..."
        Start-Sleep -Seconds $RetryDelaySeconds
    }
}

function ConvertTo-ShellSingleQuoted {
    param([string]$Value)
    return "'" + $Value.Replace("'", "'""'""'") + "'"
}

function Invoke-SshTransfer {
    param(
        [string]$Source,
        [string]$Destination,
        [int]$MaxRetries = 3,
        [int]$RetryDelaySeconds = 5
    )

    $remoteDir = $Destination.Substring(0, $Destination.LastIndexOf("/"))
    $attempt = 0
    while ($true) {
        $attempt++
        $remoteTemp = "$Destination.tmp.$PID.$attempt"
        $remoteCmd = "mkdir -p $(ConvertTo-ShellSingleQuoted $remoteDir) && cat > $(ConvertTo-ShellSingleQuoted $remoteTemp) && mv $(ConvertTo-ShellSingleQuoted $remoteTemp) $(ConvertTo-ShellSingleQuoted $Destination)"

        $psi = [System.Diagnostics.ProcessStartInfo]::new()
        $psi.FileName = "ssh"
        $psi.Arguments = "$($SshOptions -join ' ') $remote $remoteCmd"
        $psi.RedirectStandardInput = $true
        $psi.UseShellExecute = $false
        $psi.CreateNoWindow = $true

        $proc = [System.Diagnostics.Process]::Start($psi)
        $fileStream = [System.IO.File]::OpenRead($Source)
        try {
            $buf = New-Object byte[] (1MB)
            while (($read = $fileStream.Read($buf, 0, $buf.Length)) -gt 0) {
                $proc.StandardInput.BaseStream.Write($buf, 0, $read)
            }
        }
        finally {
            $fileStream.Close()
            $proc.StandardInput.Close()
        }
        $proc.WaitForExit()

        if ($proc.ExitCode -eq 0) { return }

        ssh @SshOptions $remote "rm -f $(ConvertTo-ShellSingleQuoted $remoteTemp)" | Out-Null
        if ($attempt -ge $MaxRetries) {
            throw "SSH file transfer failed after $MaxRetries attempts (exit code $($proc.ExitCode))"
        }
        Write-Host "[vocaptest-deploy]   SSH transfer retry $attempt/$MaxRetries in ${RetryDelaySeconds}s..."
        Start-Sleep -Seconds $RetryDelaySeconds
    }
}

function New-LfTempFile {
    param([string]$Source)

    $name = "vocaptest-" + [System.IO.Path]::GetRandomFileName() + ".sh"
    $target = Join-Path ([System.IO.Path]::GetTempPath()) $name
    $content = (Get-Content -LiteralPath $Source -Raw).Replace("`r`n", "`n").Replace("`r", "`n")
    $utf8NoBom = [System.Text.UTF8Encoding]::new($false)
    [System.IO.File]::WriteAllText($target, $content, $utf8NoBom)
    return $target
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

        $combined = ssh @SshOptions $remote "echo STATUS_BEGIN; test -f $remoteStatus && cat $remoteStatus || echo RUNNING; echo STATUS_END; echo LOG_BEGIN; tail -40 $remoteLog 2>/dev/null || true; echo LOG_END"
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[vocaptest-deploy] SSH poll failed (exit $LASTEXITCODE), will retry..."
            continue
        }

        $combinedText = $combined -join "`n"
        if ($combinedText -match 'STATUS_BEGIN\s*\n?(.*?)\s*STATUS_END') {
            $status = $Matches[1].Trim()
        } else {
            $status = "RUNNING"
        }

        if ($combinedText -match 'LOG_BEGIN\s*\n?(.*?)\s*LOG_END') {
            $tailText = $Matches[1].Trim()
        } else {
            $tailText = ""
        }

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
$normalizedUpdateScript = New-LfTempFile $localUpdateScript
try {
    Invoke-ScpTransfer $normalizedUpdateScript "${remote}:$remoteTmp"
}
finally {
    Remove-Item -LiteralPath $normalizedUpdateScript -ErrorAction SilentlyContinue
}

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
