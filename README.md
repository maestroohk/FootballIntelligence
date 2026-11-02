# Football Intelligence

A football forecasting and live match-intelligence system: walk-forward evaluated pre-match models, an in-play goal-hazard model with historical event replay, and an ASP.NET Core Blazor dashboard.

- `engine/` – Python research engine (`fie` package): ingestion, ratings and ML models, backtests, in-play model, live/replay processing, durable job queue and worker.
- `src/FootballIntelligence.Web/` – ASP.NET Core Blazor Server dashboard. Reads the shared SQLite database, queues jobs, supervises the Python worker and pushes updates to the browser over the Blazor SignalR circuit.
- `data/` – SQLite database (`football.db`), raw CSV cache and model caches (git-ignored).
- `docs/` – `RESEARCH.md` (methods and sources) and `REPORT.md` (measured results, experiments, limitations).

## Requirements

- .NET SDK 10
- Python 3.12

## First-time setup

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python -m pip install numpy scipy pandas scikit-learn lightgbm pytest requests
```

Build the database and models (each command runs one job in the foreground; the same jobs can be queued from the dashboard):

```bash
cd engine && ../.venv/Scripts/python -m fie run ingest_history
```

```bash
cd engine && ../.venv/Scripts/python -m fie run ingest_statsbomb
```

```bash
cd engine && ../.venv/Scripts/python -m fie run research_full
```

```bash
cd engine && ../.venv/Scripts/python -m fie run publish_predictions
```

```bash
cd engine && ../.venv/Scripts/python -m fie run train_inplay
```

```bash
cd engine && ../.venv/Scripts/python -m fie run experiments
```

```bash
cd engine && ../.venv/Scripts/python -m fie run final_test
```

```bash
cd engine && ../.venv/Scripts/python -m fie run expand_coverage
```

```bash
cd engine && ../.venv/Scripts/python -m fie run build_report
```

`research_full` and the shared models train on the original 11 divisions; `expand_coverage` adds ratings and predictions for the seven further divisions without retraining the champion. `final_test` refuses to run twice unless forced.

`ingest_history` downloads about 400 CSV files for 18 divisions (~10 minutes). `ingest_statsbomb` streams 1,551 event files and keeps only compact tables (~15 minutes). `research_full` runs every walk-forward model with three worker processes (~25 minutes on a 4-thread budget).

## Run the application

```bash
dotnet run --project src/FootballIntelligence.Web --urls http://localhost:5281 --environment Development
```

Open http://localhost:5281. The web app starts the Python worker automatically (`python -m fie worker`) and restarts it if it dies; a worker that is already running is adopted instead of duplicated. The worker runs an update cycle every 6 hours (results, fixtures, forward predictions, forward evaluation, drift check, live-provider poll) and a full research refresh weekly.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `FOOTBALL_DATA_API_KEY` | unset | Enables live score polling from football-data.org. Without it the Live tracker states that no live feed is connected and offers replay only. |
| `FIE_THREADS` | 4 | Thread cap for numerical libraries and LightGBM. |
| `FIE_MAX_JOBS` | 2 | Concurrent background jobs. |
| `FIE_CYCLE_HOURS` | 6 | Update-cycle interval (0 disables). |
| `FIE_RESEARCH_DAYS` | 7 | Challenger research interval (0 disables). |
| `FIE_DB` | `data/football.db` | Database path. |

`appsettings.json` → `Engine` holds the database, Python and engine paths and `StartWorker`.

## Tests

```bash
cd engine && ../.venv/Scripts/python -m pytest -q tests
```

The tests cover probability invariants and market consistency, Dixon–Coles mass preservation, parameter recovery, walk-forward leakage, in-play monotonicity and score/red-card responses, live-feed duplicates, out-of-order delivery and corrections, and immutability of stored predictions.

## Data attribution

Event data: StatsBomb Open Data (https://github.com/statsbomb/open-data). Results, statistics and odds: football-data.co.uk. Fixtures: OpenFootball (public domain).
