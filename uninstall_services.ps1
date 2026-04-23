# uninstall_services.ps1
# Run as Administrator to remove Ulisse Brain services

$NssmPath = "C:\nssm-2.24\win64\nssm.exe"

$Services = @("UlisseBackend", "UlisseSync")

foreach ($Service in $Services) {
    Write-Host "Removing $Service..."
    Stop-Service $Service -ErrorAction SilentlyContinue
    & $NssmPath remove $Service confirm
}

Write-Host "Services removed successfully."
