# malvalid launcher logic for Windows (called by malvalid.bat; ASCII only, runs on Windows PowerShell 5.1+).
#
# First run: sets up malvalid in a private environment under %LOCALAPPDATA%\malvalid (needs internet, a few
# minutes); later runs start in seconds. Then it starts the malvalid web UI on 127.0.0.1 and opens the
# browser on it, already signed in. Close the window (or press Ctrl+C) to stop.
#
# Settings (environment variables, all optional):
#   MALVALID_HOME      app directory (default %LOCALAPPDATA%\malvalid). Delete it to uninstall.
#   MALVALID_RUNS_DIR  where runs are stored (default %MALVALID_HOME%\runs)
#   MALVALID_PORT      first port to try (default 8765; the next free one is used if it is busy)
#   MALVALID_REINSTALL=1  force a fresh install of the environment
# Extra arguments are passed to `malvalid serve` (e.g. --no-browser).
#
# No param() block on purpose: with one, PowerShell would treat this as an advanced script and swallow
# common parameters such as --verbose / -v / --debug instead of passing them on to `malvalid serve`.
$ServeArgs = @($args)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # Invoke-WebRequest is very slow with the progress bar on

$UvVersion = '0.12.19'    # pinned uv release, downloaded only when neither uv nor Python 3.11 is installed
# SHA-256 of each uv archive this launcher may download, copied from the official release's per-asset
# .sha256 files (cross-checked against its dist-manifest.json and sha256.sum). They are pinned here, not
# fetched next to the archive, so a tampered release asset cannot pass the check. Update with $UvVersion.
$UvSha256 = @{
    'x86_64-pc-windows-msvc'  = '6dbb02d79e419522f1c500f0adb1cddcff0cda7d59b0d66ea7f5e3b4a1b2f5f0'
    'aarch64-pc-windows-msvc' = '115b54cb823bc48260670f5782001add6067ac8d98d18c8263a833704e287de9'
}
$PyVersion = '3.11'
$LauncherRev = '3'        # bump to force a reinstall after a launcher change

function Say([string]$Message) { Write-Host "malvalid: $Message" }
function Warn([string]$Message) { Write-Host "malvalid: WARNING: $Message" -ForegroundColor Yellow }
function Fail([string]$Message) {
    Write-Host "malvalid: ERROR: $Message" -ForegroundColor Red
    exit 1
}

# ---- where things are --------------------------------------------------------------------------------

$Repo = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if (-not (Test-Path -LiteralPath (Join-Path $Repo 'pyproject.toml')) -or
    -not (Test-Path -LiteralPath (Join-Path $Repo 'src\malvalid'))) {
    Fail "this launcher must stay in the launchers folder of the MalValid download ($Repo does not look like one)"
}
$Constraints = Join-Path $Repo 'constraints\lock-py311.txt'

# Relative paths in the settings are taken relative to the folder the launcher was started from.
function Resolve-UserPath([string]$Path) { return $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Path) }

if ($env:MALVALID_HOME) { $App = Resolve-UserPath $env:MALVALID_HOME }
elseif ($env:LOCALAPPDATA) { $App = Join-Path $env:LOCALAPPDATA 'malvalid' }
else { $App = Join-Path $env:USERPROFILE 'AppData\Local\malvalid' }
$App = [System.IO.Path]::GetFullPath($App)
$Venv = Join-Path $App 'venv'
$Py = Join-Path $Venv 'Scripts\python.exe'
if ($env:MALVALID_RUNS_DIR) { $Runs = [System.IO.Path]::GetFullPath((Resolve-UserPath $env:MALVALID_RUNS_DIR)) } else { $Runs = Join-Path $App 'runs' }
# A trailing backslash before the closing quote ("D:\runs\") breaks native argument parsing in PowerShell 5.1.
if ($Runs.EndsWith('\')) { $Runs = $Runs + '.' }
$Stamp = Join-Path $App 'install.stamp'
$UvLocal = Join-Path $App 'uv\uv.exe'
# Keep uv's Python downloads and cache inside the app directory, so deleting it uninstalls everything.
if (-not $env:UV_PYTHON_INSTALL_DIR) { $env:UV_PYTHON_INSTALL_DIR = Join-Path $App 'python' }
if (-not $env:UV_CACHE_DIR) { $env:UV_CACHE_DIR = Join-Path $App 'cache' }
$env:PYTHONUTF8 = '1'     # UTF-8 file I/O in malvalid and its run subprocesses, whatever the Windows code page

# Windows on Arm: LightGBM and XGBoost publish no win_arm64 wheels, so MalValid uses an x64 Python, which
# Windows 11 runs through its built-in x64 emulation. uv provides it (a native Arm Python would not work).
$CpuArch = $env:PROCESSOR_ARCHITEW6432
if (-not $CpuArch) { $CpuArch = $env:PROCESSOR_ARCHITECTURE }
$IsArm64 = ($CpuArch -eq 'ARM64')
if ($IsArm64) { $UvPythonRequest = 'cpython-3.11-windows-x86_64-none' } else { $UvPythonRequest = $PyVersion }

# ---- helpers -----------------------------------------------------------------------------------------

function Get-Fingerprint {
    $parts = New-Object System.Collections.Generic.List[byte]
    $parts.AddRange([System.Text.Encoding]::UTF8.GetBytes("rev=$LauncherRev|repo=$Repo|py=$PyVersion|"))
    $parts.AddRange([System.IO.File]::ReadAllBytes((Join-Path $Repo 'pyproject.toml')))
    if (Test-Path -LiteralPath $Constraints) { $parts.AddRange([System.IO.File]::ReadAllBytes($Constraints)) }
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try { $hash = $sha.ComputeHash($parts.ToArray()) } finally { $sha.Dispose() }
    return ([System.BitConverter]::ToString($hash) -replace '-', '').ToLowerInvariant()
}

# Windows PowerShell 5.1 turns redirected native stderr (2>$null) into errors, which 'Stop' makes fatal;
# functions that probe native commands therefore use a local 'Continue'.
function Test-Imports {
    $ErrorActionPreference = 'Continue'
    if (-not (Test-Path -LiteralPath $Py)) { return $false }
    & $Py -c 'import malvalid.web, uvicorn, starlette' 2>$null | Out-Null
    return ($LASTEXITCODE -eq 0)
}

function Test-NeedsInstall {
    if ($env:MALVALID_REINSTALL) { return $true }
    if (-not (Test-Path -LiteralPath $Py) -or -not (Test-Path -LiteralPath $Stamp)) { return $true }
    $old = "$(Get-Content -LiteralPath $Stamp -Raw)".Trim()
    if ($old -ne (Get-Fingerprint)) { return $true }
    return -not (Test-Imports)
}

function Find-Uv {
    $cmd = Get-Command uv.exe -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($cmd) { return $cmd.Source }
    if (Test-Path -LiteralPath $UvLocal) { return $UvLocal }
    return $null
}

# A 64-bit Python 3.11 that can create virtual environments (the pinned model-library wheels are 64-bit
# only). Skips the Microsoft Store "python" alias. Returns $null if none is found, so the caller falls back
# to uv, which installs a 64-bit Python 3.11.
function Find-Python311 {
    $ErrorActionPreference = 'Continue'
    $candidates = @()
    $pyLauncher = Get-Command py.exe -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($pyLauncher) {
        foreach ($sel in @("-$PyVersion-64", "-$PyVersion")) {
            try {
                $exe = & $pyLauncher.Source $sel -c 'import sys, venv, ensurepip; print(sys.executable)' 2>$null
                if ($LASTEXITCODE -eq 0 -and $exe) { $candidates += ($exe | Select-Object -First 1).Trim() }
            } catch { }
        }
    }
    foreach ($name in @('python3.11.exe', 'python.exe', 'python3.exe')) {
        foreach ($c in @(Get-Command $name -CommandType Application -ErrorAction SilentlyContinue)) {
            if ($c.Source -like '*\WindowsApps\*') { continue }
            $candidates += $c.Source
        }
    }
    foreach ($c in ($candidates | Select-Object -Unique)) {
        try {
            # No double quotes inside: Windows PowerShell 5.1 passes them to native programs unescaped.
            # Prints e.g. "3.11 64"; chr(80) is 'P', so struct.calcsize('P') is the pointer size in bytes.
            $v = & $c -c 'import sys, struct, venv, ensurepip; print(str(sys.version_info[0]) + chr(46) + str(sys.version_info[1]) + chr(32) + str(struct.calcsize(chr(80)) * 8))' 2>$null
            if ($LASTEXITCODE -eq 0 -and $v -and ($v | Select-Object -First 1).Trim() -eq "$PyVersion 64") { return $c }
        } catch { }
    }
    return $null
}

function Install-Uv {
    $arch = $CpuArch
    switch ($arch) {
        'AMD64' { $target = 'x86_64-pc-windows-msvc' }
        'ARM64' { $target = 'aarch64-pc-windows-msvc' }
        default { Fail "there is no prebuilt uv for this CPU ($arch); install Python $PyVersion from python.org and run this again" }
    }
    $asset = "uv-$target.zip"
    $url = "https://github.com/astral-sh/uv/releases/download/$UvVersion/$asset"
    $want = $UvSha256[$target]
    if (-not $want) {
        Fail "this launcher has no pinned checksum for uv $UvVersion on $target, so it will not download it; install uv (https://docs.astral.sh/uv/) or Python $PyVersion (64-bit) yourself and run this again"
    }
    Say "Neither uv nor Python $PyVersion was found, so the launcher downloads uv once"
    Say "(uv is the standalone Python installer from astral.sh; it then installs Python $PyVersion for MalValid)."
    Say "  downloading: uv $UvVersion for $target"
    Say "  from:        $url"
    Say "  sha256:      $want (pinned in this launcher)"
    Say "  to:          $UvLocal"
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
    $tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("malvalid-uv-" + [System.Guid]::NewGuid().ToString('N'))
    [void][System.IO.Directory]::CreateDirectory($tmp)
    try {
        $zip = Join-Path $tmp $asset
        # Invoke-WebRequest uses the Windows proxy settings; HTTPS_PROXY (which uv and pip read) is honoured too.
        $web = @{ UseBasicParsing = $true }
        if ($env:HTTPS_PROXY) { $web['Proxy'] = $env:HTTPS_PROXY; $web['ProxyUseDefaultCredentials'] = $true }
        try {
            Invoke-WebRequest @web -Uri $url -OutFile $zip
        } catch {
            Fail "could not download $url ($($_.Exception.Message)). Are you online? Behind a proxy, check the Windows proxy settings or set HTTPS_PROXY."
        }
        $got = (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($want -ne $got) { Fail "checksum mismatch for $asset (expected $want, got $got); not using it" }
        $out = Join-Path $tmp 'x'
        Expand-Archive -LiteralPath $zip -DestinationPath $out -Force
        $exe = Get-ChildItem -LiteralPath $out -Recurse -Filter 'uv.exe' | Select-Object -First 1
        if (-not $exe) { Fail "the uv download did not contain uv.exe" }
        [void][System.IO.Directory]::CreateDirectory((Split-Path -Parent $UvLocal))
        [System.IO.File]::Copy($exe.FullName, $UvLocal, $true)
    } finally {
        Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
    }
    Say "uv $UvVersion installed (checksum verified)."
}

function Invoke-Checked([string]$What, [scriptblock]$Block) {
    & $Block
    if ($LASTEXITCODE -ne 0) { Fail "$What failed (exit code $LASTEXITCODE; see the messages above)" }
}

# Install the repository (editable) with these extras, pinned versions first. Returns $true on success.
# The installers' output goes to the console (Out-Host), never into the return value: a function returns
# everything its commands write, and pip's progress lines would make a failure look like success.
# Runs from the repository with relative paths: uv splits an absolute constraints path at spaces
# ("C:\Users\Jane Doe\..."), which would make the pinned install fail for no real reason.
function Install-MalValidPackage([string]$Uv, [string]$Extras) {
    $pkg = '.[' + $Extras + ']'
    $c = 'constraints\lock-py311.txt'
    Push-Location -LiteralPath $Repo
    try {
        if ($Uv) { & $Uv pip install --python $Py -c $c -e $pkg | Out-Host } else { & $Py -m pip install -c $c -e $pkg | Out-Host }
        if ($LASTEXITCODE -eq 0) { return $true }
        Warn 'installing the pinned versions (constraints\lock-py311.txt) failed (see above);'
        Warn 'retrying with the newest compatible versions'
        if ($Uv) { & $Uv pip install --python $Py -e $pkg | Out-Host } else { & $Py -m pip install -e $pkg | Out-Host }
        return ($LASTEXITCODE -eq 0)
    } finally {
        Pop-Location
    }
}

# LightGBM and XGBoost load native DLLs that need the Microsoft Visual C++ Redistributable (VCOMP140.DLL,
# MSVCP140.DLL); the wheels do not bundle it. Warns and returns $false if they cannot be loaded.
function Test-ModelLibraries {
    $ErrorActionPreference = 'Continue'   # see Test-Imports
    $out = & $Py -c 'import lightgbm, xgboost' 2>&1
    if ($LASTEXITCODE -eq 0) { return $true }
    Warn 'LightGBM / XGBoost cannot be loaded, so LightGBM and XGBoost models (and the synthetic demo) will fail:'
    foreach ($line in @($out | Select-Object -Last 3)) { Write-Host "    $line" }
    Warn 'They need the Microsoft Visual C++ Redistributable (x64). Install it from'
    Warn '    https://aka.ms/vs/17/release/vc_redist.x64.exe'
    Warn 'and start MalValid again (no reinstall needed).'
    return $false
}

function Install-MalValid {
    $uv = Find-Uv
    $sysPy = $null
    if ($uv) { Say "Using uv: $uv" }
    else {
        # On Arm, a Python found on this computer is likely a native Arm one; uv provides an x64 Python instead.
        if (-not $IsArm64) { $sysPy = Find-Python311 }
        if ($sysPy) { Say "Using Python ${PyVersion}: $sysPy" }
        else { Install-Uv; $uv = $UvLocal }
    }
    if ($IsArm64) { Say 'Windows on Arm: using an x64 Python (LightGBM and XGBoost have no Arm builds for Windows).' }
    Say "Setting up MalValid in $App"
    Say "(first run only: downloads about 300 MB of Python packages and takes a few minutes)"
    try {
        if (Test-Path -LiteralPath $Stamp) { Remove-Item -LiteralPath $Stamp -Force }
        if (Test-Path -LiteralPath $Venv) { Remove-Item -LiteralPath $Venv -Recurse -Force }
    } catch {
        Fail "could not remove the old environment $Venv ($($_.Exception.Message)). Close every other MalValid window and run this again."
    }
    if ($uv) {
        Say "[1/3] Creating a private Python $PyVersion environment (uv downloads Python itself if needed)"
        Invoke-Checked 'creating the Python environment' { & $uv venv --quiet --python $UvPythonRequest $Venv }
    } else {
        Say "[1/3] Creating a private Python $PyVersion environment"
        Invoke-Checked 'creating the Python environment' { & $sysPy -m venv $Venv }
        $ErrorActionPreference = 'Continue'   # see Test-Imports
        & $Py -m pip install --quiet --upgrade pip 2>$null | Out-Null
        $ErrorActionPreference = 'Stop'
    }
    Say "[2/3] Installing MalValid and its dependencies (pinned versions from constraints\lock-py311.txt)"
    # The web UI plus ONNX models and raw-PE featurization. If the optional extras have no build for this
    # platform, fall back to the web UI alone (LightGBM and XGBoost models still work).
    if (-not (Install-MalValidPackage $uv 'web,onnx,featurize')) {
        Warn 'the ONNX / raw-PE extras could not be installed here; installing the web UI without them'
        Warn '(LightGBM and XGBoost model files work; ONNX model files will not)'
        if (-not (Install-MalValidPackage $uv 'web')) { Fail 'installing malvalid failed (see the messages above)' }
    }
    Say '[3/3] Checking the installation'
    if (-not (Test-Imports)) { Fail 'malvalid was installed but does not import (see the messages above)' }
    Set-Content -LiteralPath $Stamp -Value (Get-Fingerprint) -Encoding ASCII
    Say 'Setup complete.'
}

# ---- main ----------------------------------------------------------------------------------------------

try { [void][System.IO.Directory]::CreateDirectory($App) } catch { Fail "cannot create $App ($($_.Exception.Message))" }
if (Test-NeedsInstall) {
    # One installer at a time (a double double-click). An exclusively opened file, not a directory: Windows
    # releases it even when the window is closed mid-setup, so no stale lock blocks the next double-click.
    $lockPath = Join-Path $App '.install-lock'
    try { $lock = [System.IO.File]::Open($lockPath, 'OpenOrCreate', 'ReadWrite', 'None') }
    catch { Fail 'another MalValid window is setting up MalValid right now; wait for it to finish, then run this again' }
    try { Install-MalValid } finally { $lock.Dispose() }
}
[void](Test-ModelLibraries)

# Windows has no OS sandbox that malvalid can use (bubblewrap is Linux-only), so models run with reduced
# isolation, and every report and the web UI say so.
Warn 'Windows has no OS sandbox for MalValid, so models run with REDUCED ISOLATION: a separate worker'
Warn 'process with pickle refusal and resource limits, but no network or file-system isolation. Reports are'
Warn "marked 'isolation: process_only'. Only evaluate models whose origin you trust (for untrusted models,"
Warn 'run MalValid on Linux or inside WSL2, which have the bubblewrap sandbox).'

try { [void][System.IO.Directory]::CreateDirectory($Runs) } catch { Fail "cannot create the runs directory $Runs ($($_.Exception.Message))" }
Set-Location -LiteralPath $App
$port = 8765
if ($env:MALVALID_PORT) {
    $parsed = 0
    if (-not [int]::TryParse($env:MALVALID_PORT, [ref]$parsed) -or $parsed -lt 1 -or $parsed -gt 65535) {
        Fail "MALVALID_PORT must be a port number from 1 to 65535 (it is '$($env:MALVALID_PORT)')"
    }
    $port = $parsed
}
Write-Host ''
Write-Host '=============================================================================='
Write-Host '  MalValid is running - close this window to stop it (or press Ctrl+C).'
Write-Host '  Your browser opens on it automatically, already signed in.'
Write-Host "  Runs are saved in: $Runs"
Write-Host '=============================================================================='
Write-Host ''
$pyArgs = @('-m', 'malvalid', 'serve', '--host', '127.0.0.1', '--port', "$port", '--find-free-port',
            '--runs-dir', $Runs, '--allow-reduced-isolation') + $ServeArgs
& $Py @pyArgs
exit $LASTEXITCODE
