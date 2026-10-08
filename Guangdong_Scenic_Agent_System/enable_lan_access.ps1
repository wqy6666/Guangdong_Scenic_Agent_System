#Requires -RunAsAdministrator
param(
    [ValidateRange(1, 65535)]
    [int]$Port = 8000
)

$ErrorActionPreference = 'Stop'
$listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
    Where-Object { $_.LocalAddress -notin @('127.0.0.1', '::1') } |
    Select-Object -First 1
if (-not $listener) {
    throw 'Start this project with .\start_web.ps1 before enabling LAN access.'
}

$webProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $($listener.OwningProcess)"
$webEntry = Join-Path $PSScriptRoot 'src\web_api.py'
if (-not $webProcess.ExecutablePath -or
    -not $webProcess.CommandLine.Contains($webEntry)) {
    throw "Port $Port is not owned by this project's start_web.ps1 process. No firewall rules were changed."
}
$webProgram = $webProcess.ExecutablePath

# Scope the allow rule to this Python process, Wi-Fi, this port and the local subnet.
$ruleName = "ScenicWeb-LAN-$Port"
$existingRule = Get-NetFirewallRule -Name $ruleName -ErrorAction SilentlyContinue
$ruleParameters = @{
    Name = $ruleName
    DisplayName = "Scenic Web LAN TCP $Port"
    Direction = 'Inbound'
    Action = 'Allow'
    Enabled = 'True'
    Protocol = 'TCP'
    LocalPort = $Port
    Program = $webProgram
    RemoteAddress = 'LocalSubnet'
    InterfaceType = 'Wireless'
    Profile = 'Any'
}
if ($existingRule) {
    $ruleParameters.Remove('DisplayName')
    Set-NetFirewallRule @ruleParameters -NewDisplayName "Scenic Web LAN TCP $Port"
} else {
    New-NetFirewallRule @ruleParameters | Out-Null
}

# Dismissing the Windows Python prompt creates a TCP block that overrides allows.
# Only disable those automatically created TCP rules for this exact executable.
$promptBlocks = @(Get-NetFirewallRule -Name 'TCP Query User*' -ErrorAction SilentlyContinue |
    Where-Object { $_.Enabled -eq 'True' -and $_.Direction -eq 'Inbound' -and $_.Action -eq 'Block' } |
    Where-Object {
        $programFilter = $_ | Get-NetFirewallApplicationFilter
        $programFilter.Program -eq $webProgram
    })
foreach ($block in $promptBlocks) {
    Disable-NetFirewallRule -Name $block.Name
    Write-Host "Disabled conflicting Python TCP block: $($block.Name)"
}

Write-Host "LAN access enabled for TCP $Port on Wi-Fi. The Windows firewall remains enabled."
Write-Host "To disable this allow rule later: Disable-NetFirewallRule -Name '$ruleName'"
foreach ($block in $promptBlocks) {
    Write-Host "To restore the previous Python TCP block: Enable-NetFirewallRule -Name '$($block.Name)'"
}
