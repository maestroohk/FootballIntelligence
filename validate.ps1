param(
    [switch]$Research,
    [switch]$NoMigrate,
    [switch]$Start
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
$failed = @()

function Step($text) { Write-Host "`n== $text" -ForegroundColor Cyan }

Step "Tools"
if (-not (Test-Path $python)) { throw "No virtual environment at .venv; run setup.ps1 -SkipData first" }
if (-not (Get-Command dotnet -ErrorAction SilentlyContinue)) { throw "dotnet SDK not found" }
Write-Host "profile: $(if ($env:FIE_PROFILE) { $env:FIE_PROFILE } else { 'full' })"
& $python -m pip install --quiet -r (Join-Path $root "engine\requirements.txt")

Step ".NET build"
dotnet build (Join-Path $root "src\FootballIntelligence.Web") -nologo -v q
if ($LASTEXITCODE -ne 0) { $failed += "dotnet build" }

Step "Engine tests"
Push-Location $root
& $python -m pytest -q
if ($LASTEXITCODE -ne 0) { $failed += "engine tests" }
Pop-Location

Step "Data, models and predictions"
Push-Location (Join-Path $root "engine")
$doctorArgs = @("-m", "fie", "doctor")
if ($Research) { $doctorArgs += "--research" }
if ($NoMigrate) { $doctorArgs += "--no-migrate" }
& $python @doctorArgs
if ($LASTEXITCODE -ne 0) { $failed += "doctor checks (see report)" }
Pop-Location

Step "Summary"
if ($failed.Count -eq 0) { Write-Host "All automated checks passed." -ForegroundColor Green }
else { Write-Host ("Failed: " + ($failed -join ", ")) -ForegroundColor Yellow }
Write-Host "Reports are in data\reports. Browser checklist: README, 'Validating on the research machine'."

if ($Start) {
    Step "Starting the app on http://localhost:5281"
    dotnet run --project (Join-Path $root "src\FootballIntelligence.Web") --urls http://localhost:5281
}
