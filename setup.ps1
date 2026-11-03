param(
    [switch]$Quick,
    [switch]$SkipData
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"

function Step($text) { Write-Host "`n== $text" -ForegroundColor Cyan }

function Run-Job($kind, $jobArgs = "{}") {
    Step "fie run $kind $jobArgs"
    Push-Location (Join-Path $root "engine")
    try {
        $passed = if ($PSNativeCommandArgumentPassing -and $PSNativeCommandArgumentPassing -ne "Legacy") { $jobArgs } else { $jobArgs -replace '"', '\"' }
        $output = & $python -m fie run $kind --args $passed
        $output | Select-Object -Last 2 | ForEach-Object { Write-Host $_ }
        if (($output | Select-Object -First 1) -notmatch "^succeeded") { throw "job $kind did not succeed" }
    }
    finally { Pop-Location }
}

Step "Python environment"
if (-not (Test-Path $python)) { if (Get-Command py -ErrorAction SilentlyContinue) { py -3.12 -m venv (Join-Path $root ".venv") } else { python -m venv (Join-Path $root ".venv") } }
& $python -m pip install --quiet --upgrade pip
& $python -m pip install --quiet -r (Join-Path $root "engine\requirements.txt")

Step ".NET build"
dotnet build (Join-Path $root "src\FootballIntelligence.Web") -nologo -v q
if ($LASTEXITCODE -ne 0) { throw "dotnet build failed" }

Step "Database schema"
Push-Location (Join-Path $root "engine"); & $python -m fie init; Pop-Location

if ($SkipData) { Write-Host "Skipping data download and model training."; exit 0 }

if ($Quick) {
    Run-Job "ingest_history" '{"codes": ["E0"], "seasons": [2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013]}'
    Run-Job "research_full"
    Run-Job "publish_predictions"
    Write-Host "`nQuick pipeline finished (Premier League 2005/06-2013/14 only)."
    exit 0
}

Run-Job "ingest_history"
Run-Job "ingest_statsbomb"
Run-Job "ingest_fixtures"
Run-Job "research_full"
Run-Job "expand_coverage"
Run-Job "publish_predictions"
Run-Job "train_inplay"
Run-Job "experiments" '{"names": ["ht_recalibration"]}'
Run-Job "final_test"
Run-Job "predict_upcoming"
Run-Job "build_report"
Write-Host "`nFull pipeline finished. Start the app with: dotnet run --project src/FootballIntelligence.Web --urls http://localhost:5281"
