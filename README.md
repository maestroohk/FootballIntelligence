# Football Intelligence

A football forecasting and live match-intelligence system: walk-forward evaluated pre-match models, an in-play goal-hazard model with historical event replay, and an ASP.NET Core Blazor dashboard.

- `engine/` – Python research engine (`fie` package): ingestion, ratings and ML models, backtests, in-play model, live/replay processing, durable job queue and worker.
- `src/FootballIntelligence.Web/` – ASP.NET Core Blazor Server dashboard. Reads the shared SQLite database, queues jobs, supervises the Python worker and pushes updates to the browser over the Blazor SignalR circuit.
- `data/` – SQLite database (`football.db`), raw CSV cache and model caches (git-ignored).
- `docs/` – `RESEARCH.md` (methods and sources) and `REPORT.md` (measured results, experiments, limitations).

## Architecture

```
football-data.co.uk ─┐                                   ┌─ Blazor Server dashboard (SignalR circuit pushes
StatsBomb Open Data ─┼─> ingestion (identity, provenance) │   updates when snapshots, jobs or notifications change)
OpenFootball ────────┤        │                           │
football-data.org ───┘        v                           │
                       SQLite (WAL) <───────────────────────┤  queues jobs, reads results
                              ^                           │
   Python worker ── job queue ┘                           └─ EngineSupervisor starts/adopts the worker
     ├─ walk-forward ratings (Elo, Poisson, Dixon–Coles, online/pi experiments)
     ├─ features + LightGBM, logistic, learned ensemble, bookmaker benchmark
     ├─ backtests, paired bootstrap comparisons, final test, drift monitor
     ├─ forward predictions (immutable, ≥3 h before kick-off) with explanations
     ├─ in-play hazard model, replay and live-provider processing
     └─ scheduler: 6-hourly update cycle, weekly challenger research
```

- Every pre-match market comes from one Dixon–Coles joint score matrix built from the champion's expected goals, so 1X2, totals, BTTS and half markets are mutually consistent.
- In-play forecasts start from the stored pre-match expected goals and step a Markov chain over the remaining minutes with learned time, score-state and red-card multipliers.
- Predictions are immutable (database trigger); new model versions publish new rows and old versions are kept.

## Requirements

- .NET SDK 10
- Python 3.12

## First-time setup

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python -m pip install -r engine/requirements.txt
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
.venv/Scripts/python -m pytest -q
```

The tests cover probability invariants and market consistency, Dixon–Coles mass preservation, parameter recovery, walk-forward leakage, in-play monotonicity and score/red-card responses, live-feed duplicates, out-of-order delivery and corrections, and immutability of stored predictions.

## Data attribution

Event data: StatsBomb Open Data (https://github.com/statsbomb/open-data). Results, statistics and odds: football-data.co.uk. Fixtures: OpenFootball (public domain).
