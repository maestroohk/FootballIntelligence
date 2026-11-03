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

On Windows, one script restores dependencies, builds, creates the database and regenerates all data and models from the public sources:

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
```

`setup.ps1 -SkipData` only installs and builds; `setup.ps1 -Quick` runs a 5-minute pipeline smoke test on the Premier League 2005/06–2013/14.

The full run executes these jobs in order (each can also be run alone with `cd engine` then `../.venv/Scripts/python -m fie run <job>`, or queued from the dashboard): `ingest_history`, `ingest_statsbomb`, `ingest_fixtures`, `research_full`, `expand_coverage`, `publish_predictions`, `train_inplay`, `experiments` (half-time recalibration), `final_test`, `predict_upcoming`, `build_report`.

Measured on this project's development laptop (i7-13650HX, 4-thread budget): about 15 minutes of downloads, 50 minutes for `research_full`, and 1.5–2 hours in total; about 2 GB of disk under `data/`. No GPU is needed. `research_full` and the shared models train on the original 11 divisions; `expand_coverage` adds the seven further divisions without retraining the champion. `final_test` refuses to run twice unless forced. The pipeline is deterministic: an isolated from-scratch rerun reproduced every model's validation results within 2e-8.

## Run the application

```bash
dotnet run --project src/FootballIntelligence.Web --urls http://localhost:5281 --environment Development
```

Open http://localhost:5281. The web app starts the Python worker automatically (`python -m fie worker`) and restarts it if it dies; a worker that is already running is adopted instead of duplicated. The worker runs an update cycle every 6 hours (results, fixtures, forward predictions, forward evaluation, drift check, live-provider poll) and a full research refresh weekly.

## Capabilities

- Pre-match forecasts for 18 divisions in 11 countries: 1X2, expected goals, full scoreline matrix, totals 0.5–4.5, BTTS, clean sheets, goals per half — all from one joint score distribution.
- Walk-forward evaluation (validation 2012/13–2022/23, untouched final test 2023/24 onwards), paired bootstrap comparisons, calibration and reliability analysis, bookmaker benchmark.
- In-play goal-hazard model with historical event replay (clearly labelled REPLAY) and a score-based live adapter for football-data.org.
- Player and team-tactics profiles from StatsBomb Open Data (2015/16).

## Current limitations

- No live event feed is connected; live tracking needs `FOOTBALL_DATA_API_KEY` and is untested against the real API.
- Player, lineup and event data cover StatsBomb 2015/16 only; no injury, weather or travel data.
- The model does not beat bookmaker closing odds. See `docs/REPORT.md` for measured results and open issues.

## Data

No data is committed. Every dataset is downloaded from its public source by the ingestion jobs above:

| Source | Used for | Terms |
|---|---|---|
| football-data.co.uk | results, match statistics, odds | free to use; attribution given |
| StatsBomb Open Data | events, xG, lineups (2015/16) | research use with attribution to StatsBomb |
| OpenFootball football.json | future fixtures | public domain |

Downloads are cached in `data/raw/` and the database is `data/football.db`; both are git-ignored and can be rebuilt with the commands above.

## Team crests

Teams are shown with a generated shield in the club's real colours (a curated list in `Services/TeamIdentity.cs`, with a hashed colour for other teams) and leagues with their country flag. Each team has a profile page (`/team/{id}`) with record, home/away split, form, fixtures and the league table computed from stored results. To show official crests, add image files you are licensed to use as `src/FootballIntelligence.Web/wwwroot/crests/{slug}.svg` (slug of the team name, e.g. `manchester-united.svg`; `{team_id}` also works, and `.png` / `.webp` are accepted); the app picks them up automatically and falls back to the shield otherwise. No logos are hotlinked or bundled. Team names shown in the app come from the canonical team record plus provider aliases (for example football-data.co.uk's `Man United` is displayed as `Manchester United`), so aliases never create duplicate teams.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `FOOTBALL_DATA_API_KEY` | unset | Enables live score polling from football-data.org. Without it the Live tracker states that no live feed is connected and offers replay only. |
| `FIE_THREADS` | 4 | Thread cap for numerical libraries and LightGBM. |
| `FIE_MAX_JOBS` | 2 | Concurrent background jobs. |
| `FIE_CYCLE_HOURS` | 6 | Update-cycle interval (0 disables). |
| `FIE_RESEARCH_DAYS` | 7 | Challenger research interval (0 disables). |
| `FIE_DB` | `data/football.db` | Database path. |
| `FIE_RESEARCH_PROCESSES` | CPU threads ÷ 4, between 1 and 3 | Parallel processes for walk-forward research. |
| `FIE_CACHE` | `data/cache` | Model cache directory (lets you run isolated experiments). |
| `FIE_LIVE_POLL_SECONDS` | 60 | Live-provider polling interval when an API key is set. |

`appsettings.json` → `Engine` holds the database, Python and engine paths and `StartWorker`.

## Continuing on another Windows laptop

Downloads use the Windows certificate store (`truststore`), so they work behind HTTPS-inspecting corporate proxies. Some corporate web filters block football-data.co.uk (rated as gambling); on such a network the history and odds cannot be downloaded, so move the data with a checksummed bundle instead (run from `engine/`, app stopped):

```powershell
..\.venv\Scripts\python -m fie bundle-export D:\fie-bundle   # on the machine that has the data
..\.venv\Scripts\python -m fie bundle-verify D:\fie-bundle   # on this machine, after copying the folder
..\.venv\Scripts\python -m fie bundle-import D:\fie-bundle   # add --replace to overwrite existing history (backed up first)
..\.venv\Scripts\python -m fie assets                          # report what is present and what is still missing
```

The bundle holds a consistent database snapshot, model caches and raw downloads with a SHA-256 manifest. Import verifies every file, copies via `.part` files and skips files already restored, so an interrupted import resumes. With raw downloads present, `ingest_history` rebuilds tables without downloading. The Data health page shows the same asset check.

1. Install Git, the .NET 10 SDK and Python 3.12 (the `py` launcher is used if present).
2. Clone and set up:
   ```powershell
   git clone https://github.com/maestroohk/FootballIntelligence.git
   cd FootballIntelligence
   powershell -ExecutionPolicy Bypass -File setup.ps1
   ```
3. Optional secrets go in environment variables, never in the repository, for example `setx FOOTBALL_DATA_API_KEY "<your-key>"`.
4. Size the compute budget to the machine with `FIE_THREADS`, `FIE_MAX_JOBS` and `FIE_RESEARCH_PROCESSES` (defaults are conservative because the original laptop overheated under sustained load).
5. Start the app with `dotnet run --project src/FootballIntelligence.Web --urls http://localhost:5281` and run the tests with `.venv/Scripts/python -m pytest -q`.

Nothing in `data/` is in Git: the database, raw downloads and model caches are rebuilt by `setup.ps1`. Model versions, evaluation results and every experiment are recorded in `docs/RESULTS.md` (generated), with the narrative in `docs/REPORT.md`, methods and sources in `docs/RESEARCH.md` and the build timeline in `docs/SESSION_LOG.md`. The current champion is `ensemble@v3`: a learned blend of Dixon–Coles, a shot-xG Dixon–Coles variant, Elo and LightGBM goal rates.

Current development priorities:
- Connect a live provider key and validate live tracking against real matches.
- Obtain confirmed lineups and re-test the starting-XI adjustment on more seasons.
- Add licensed crest files under `wwwroot/crests/`.
- Champions League support, which needs a cross-country strength scale.
- Forward-performance monitoring once stored forward predictions have results.

## Tests

```bash
.venv/Scripts/python -m pytest -q
```

The tests cover probability invariants and market consistency, Dixon–Coles mass preservation, parameter recovery, walk-forward leakage, in-play monotonicity and score/red-card responses, live-feed duplicates, out-of-order delivery and corrections, and immutability of stored predictions.

## Data attribution

Event data: StatsBomb Open Data (https://github.com/statsbomb/open-data). Results, statistics and odds: football-data.co.uk. Fixtures: OpenFootball (public domain).
