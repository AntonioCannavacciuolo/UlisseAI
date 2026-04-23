# setup_autostart.ps1
# Run once as Administrator to register UlisseBrain as a Windows Scheduled Task.

$TaskName    = "UlisseBrain"
$BatFile     = "D:\Work\UlisseBrain\start_ulisse.bat"
$WorkDir     = "D:\Work\UlisseBrain"

# Remove existing task if present
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Write-Host "Removing existing task '$TaskName'..."
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

# Action: run the batch file via cmd.exe
$Action = New-ScheduledTaskAction `
    -Execute "cmd.exe" `
    -Argument "/c `"$BatFile`"" `
    -WorkingDirectory $WorkDir

# Trigger: at system startup, delayed by 30 seconds
$Trigger = New-ScheduledTaskTrigger -AtStartup
$Trigger.Delay = "PT30S"

# Settings
$Settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Hours 0) `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -StartWhenAvailable

# Principal: current user, run whether logged on or not
$CurrentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$Principal = New-ScheduledTaskPrincipal `
    -UserId $CurrentUser `
    -LogonType S4U `
    -RunLevel Highest

# Register
Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Principal $Principal `
    -Description "Starts Ulisse Brain backend and sync watcher on system startup." `
    -Force | Out-Null

Write-Host "UlisseBrain autostart configured successfully"
Write-Host ""
Write-Host "  Task name : $TaskName"
Write-Host "  Trigger   : At startup + 30s delay"
Write-Host "  User      : $CurrentUser"
Write-Host "  Script    : $BatFile"
Write-Host ""
Write-Host "To run immediately: Start-ScheduledTask -TaskName '$TaskName'"
