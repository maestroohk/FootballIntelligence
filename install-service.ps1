param([switch]$Remove, [switch]$MainsOnly, [int]$Port = 5281, [int]$Threads = 2)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$task = "FootballIntelligence"
$service = Join-Path $root ".service"

$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) { throw "Run this script from an administrator PowerShell window." }

if (Get-ScheduledTask -TaskName $task -ErrorAction SilentlyContinue) {
    Stop-ScheduledTask -TaskName $task -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName $task -Confirm:$false
    Get-Process FootballIntelligence.Web -ErrorAction SilentlyContinue | Where-Object { $_.Path -like "$service*" } | Stop-Process -Force
}
if ($Remove) { Write-Host "Removed the $task task."; return }

dotnet publish (Join-Path $root "src\FootballIntelligence.Web") -c Release -o (Join-Path $service "app") --nologo -v q
if ($LASTEXITCODE -ne 0) { throw "dotnet publish failed" }

$start = Join-Path $service "start.cmd"
@"
@echo off
set ASPNETCORE_ENVIRONMENT=Production
set ASPNETCORE_URLS=http://localhost:$Port
set Engine__DatabasePath=$root\data\football.db
set Engine__PythonPath=$root\.venv\Scripts\python.exe
set Engine__EngineDirectory=$root\engine
set FIE_THREADS=$Threads
set FIE_RESEARCH_PROCESSES=1
cd /d "$service\app"
"$service\app\FootballIntelligence.Web.exe" >> "$service\service.log" 2>&1
"@ | Set-Content -Path $start -Encoding ascii

$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$start`""
$atBoot = New-ScheduledTaskTrigger -AtStartup
$retry = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 15)
$settings = New-ScheduledTaskSettingsSet -RunOnlyIfNetworkAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 5) -StartWhenAvailable
$settings.DisallowStartIfOnBatteries = [bool]$MainsOnly
$settings.StopIfGoingOnBatteries = [bool]$MainsOnly
$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest
Register-ScheduledTask -TaskName $task -Action $action -Trigger @($atBoot, $retry) -Settings $settings -Principal $principal `
    -Description "Football Intelligence web app and prediction engine (http://localhost:$Port). Starts with a network connection; no sign-in needed." | Out-Null
Start-ScheduledTask -TaskName $task
Write-Host "Installed. Football Intelligence will run at http://localhost:$Port whenever the laptop is on$(if ($MainsOnly) { ' mains power' }) with a network connection."
Write-Host "Log: $service\service.log. Remove with: .\install-service.ps1 -Remove"
