# install_services.ps1
# Run as Administrator to install Ulisse Brain services via NSSM

$NssmPath = "C:\nssm-2.24\win64\nssm.exe"
$WorkDir = "D:\Work\UlisseBrain"
$PythonPath = "C:\Users\canna\AppData\Local\Programs\Python\Python311\python.exe"

# Verify NSSM exists
if (-not (Test-Path $NssmPath)) {
    Write-Error "NSSM not found at $NssmPath. Please update the path in the script."
    exit
}

# Create logs directory if not exists
if (-not (Test-Path "$WorkDir\logs")) {
    New-Item -ItemType Directory -Path "$WorkDir\logs" | Out-Null
}

# 1. Install Backend Service
$ServiceName = "UlisseBackend"
$AppPath = "webapp\backend\app.py"

Write-Host "Installing $ServiceName..."
& $NssmPath install $ServiceName $PythonPath "$WorkDir\$AppPath"
& $NssmPath set $ServiceName AppDirectory $WorkDir
& $NssmPath set $ServiceName AppStdout "$WorkDir\logs\backend_service.log"
& $NssmPath set $ServiceName AppStderr "$WorkDir\logs\backend_service.log"
& $NssmPath set $ServiceName DisplayName "Ulisse Brain - Backend"
& $NssmPath set $ServiceName Description "Ulisse Brain Flask/FastAPI Backend Service"
& $NssmPath set $ServiceName Start SERVICE_AUTO_START

# 2. Install Sync Watcher Service
$SyncServiceName = "UlisseSync"
$SyncAppPath = "scripts\sync_watcher.py"

Write-Host "Installing $SyncServiceName..."
& $NssmPath install $SyncServiceName $PythonPath "$WorkDir\$SyncAppPath"
& $NssmPath set $SyncServiceName AppDirectory $WorkDir
& $NssmPath set $SyncServiceName AppStdout "$WorkDir\logs\sync_service.log"
& $NssmPath set $SyncServiceName AppStderr "$WorkDir\logs\sync_service.log"
& $NssmPath set $SyncServiceName DisplayName "Ulisse Brain - Sync Watcher"
& $NssmPath set $SyncServiceName Description "Ulisse Brain Filesystem Sync Watcher"
& $NssmPath set $SyncServiceName Start SERVICE_AUTO_START

Write-Host ""
Write-Host "Services installed successfully!"
Write-Host "You can start them with:"
Write-Host "  Start-Service $ServiceName"
Write-Host "  Start-Service $SyncServiceName"
Write-Host ""
Write-Host "Or use services.msc to manage them."
