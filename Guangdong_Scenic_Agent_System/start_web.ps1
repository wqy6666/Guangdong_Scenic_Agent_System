param(
    [string]$Python = "",
    [ValidateRange(1, 65535)]
    [int]$Port = 8000,
    [string]$BindAddress = "0.0.0.0",
    [switch]$PublicMode,
    [string[]]$AllowedOrigins = @(),
    [string[]]$AllowedModelHosts = @()
)

$ErrorActionPreference = "Stop"
if (-not $Python) {
    $candidates = @(
        (Join-Path $PSScriptRoot '.venv/Scripts/python.exe'),
        (Join-Path (Split-Path $PSScriptRoot -Parent) '.venv/Scripts/python.exe')
    )
    $Python = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (-not $Python) { $Python = 'python' }
}

$env:PYTHONIOENCODING = 'utf-8'
$env:SCENIC_PORT = [string]$Port
$env:SCENIC_HOST = $BindAddress
$env:SCENIC_PUBLIC_MODE = if ($PublicMode) { '1' } else { '0' }
if ($AllowedOrigins.Count) { $env:SCENIC_ALLOWED_ORIGINS = $AllowedOrigins -join ',' }
if ($AllowedModelHosts.Count) { $env:SCENIC_ALLOWED_MODEL_HOSTS = $AllowedModelHosts -join ',' }
Push-Location $PSScriptRoot
try {
    Write-Host 'Loading models. Wait for Application startup complete before opening the page.'
    if ($BindAddress -eq '0.0.0.0') {
        Write-Host "Computer: http://127.0.0.1:$Port"
        $lanAddresses = @(Get-NetIPConfiguration -ErrorAction SilentlyContinue |
            Where-Object { $_.IPv4DefaultGateway } |
            ForEach-Object { $_.IPv4Address.IPAddress } |
            Sort-Object -Unique)
        foreach ($lanAddress in $lanAddresses) {
            Write-Host "Phone (same Wi-Fi): http://${lanAddress}:$Port"
        }
        if (-not $lanAddresses.Count) { Write-Host 'No LAN address found. Connect to Wi-Fi and check ipconfig.' }
    } else {
        Write-Host "Listening at http://${BindAddress}:$Port"
    }
    & $Python -B (Join-Path $PSScriptRoot 'src/web_api.py')
} finally {
    Pop-Location
}
