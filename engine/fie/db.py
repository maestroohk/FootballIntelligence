import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    source TEXT PRIMARY KEY,
    url TEXT,
    licence TEXT,
    coverage TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS competitions (
    code TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    country TEXT NOT NULL,
    tier INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS teams (
    team_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    country TEXT NOT NULL,
    UNIQUE(name, country)
);

CREATE TABLE IF NOT EXISTS team_aliases (
    alias TEXT NOT NULL,
    source TEXT NOT NULL,
    country TEXT NOT NULL,
    team_id INTEGER NOT NULL REFERENCES teams(team_id),
    method TEXT NOT NULL,
    PRIMARY KEY(alias, source, country)
);
CREATE INDEX IF NOT EXISTS ix_team_aliases_team ON team_aliases(team_id, source);

CREATE TABLE IF NOT EXISTS matches (
    match_id INTEGER PRIMARY KEY,
    competition TEXT NOT NULL REFERENCES competitions(code),
    season INTEGER NOT NULL,
    match_date TEXT NOT NULL,
    kickoff TEXT,
    home_team_id INTEGER NOT NULL REFERENCES teams(team_id),
    away_team_id INTEGER NOT NULL REFERENCES teams(team_id),
    status TEXT NOT NULL,
    fthg INTEGER, ftag INTEGER, hthg INTEGER, htag INTEGER,
    referee TEXT, attendance INTEGER,
    hs INTEGER, aws INTEGER, hst INTEGER, ast INTEGER,
    hc INTEGER, ac INTEGER, hf INTEGER, af INTEGER,
    hy INTEGER, ay INTEGER, hr INTEGER, ar INTEGER,
    source TEXT NOT NULL,
    source_row_hash TEXT,
    first_seen TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_matches_date ON matches(match_date);
CREATE INDEX IF NOT EXISTS ix_matches_pair ON matches(competition, season, home_team_id, away_team_id);
CREATE INDEX IF NOT EXISTS ix_matches_comp ON matches(competition, season);

CREATE TABLE IF NOT EXISTS odds (
    match_id INTEGER NOT NULL REFERENCES matches(match_id),
    bookmaker TEXT NOT NULL,
    timing TEXT NOT NULL,
    home REAL, draw REAL, away REAL,
    over25 REAL, under25 REAL,
    PRIMARY KEY(match_id, bookmaker, timing)
);

CREATE TABLE IF NOT EXISTS field_catalog (
    field TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    source TEXT NOT NULL,
    description TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sb_matches (
    sb_match_id INTEGER PRIMARY KEY,
    match_id INTEGER REFERENCES matches(match_id),
    competition TEXT NOT NULL,
    season INTEGER NOT NULL,
    match_date TEXT NOT NULL,
    kickoff TEXT,
    home_name TEXT NOT NULL,
    away_name TEXT NOT NULL,
    home_team_id INTEGER,
    away_team_id INTEGER,
    home_score INTEGER, away_score INTEGER,
    home_formation TEXT, away_formation TEXT,
    first_half_end REAL, second_half_end REAL,
    processed INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS sb_events (
    sb_match_id INTEGER NOT NULL,
    seq INTEGER NOT NULL,
    period INTEGER NOT NULL,
    minute INTEGER NOT NULL,
    second INTEGER NOT NULL,
    clock REAL NOT NULL,
    kind TEXT NOT NULL,
    side TEXT NOT NULL,
    player_id INTEGER,
    player_name TEXT,
    xg REAL,
    x REAL, y REAL,
    outcome TEXT,
    detail TEXT,
    pattern TEXT,
    PRIMARY KEY(sb_match_id, seq)
);

CREATE TABLE IF NOT EXISTS sb_team_minutes (
    sb_match_id INTEGER NOT NULL,
    side TEXT NOT NULL,
    period INTEGER NOT NULL,
    minute INTEGER NOT NULL,
    passes INTEGER, passes_own60 INTEGER, final_third_passes INTEGER, pressures INTEGER, recoveries INTEGER,
    def_actions_high INTEGER, box_touches INTEGER, possession_seconds REAL, shots INTEGER, xg REAL,
    PRIMARY KEY(sb_match_id, side, period, minute)
);

CREATE TABLE IF NOT EXISTS players (
    player_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    nickname TEXT,
    country TEXT,
    source TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS appearances (
    sb_match_id INTEGER NOT NULL,
    player_id INTEGER NOT NULL,
    side TEXT NOT NULL,
    team_name TEXT NOT NULL,
    position TEXT,
    started INTEGER NOT NULL,
    minute_on REAL, minute_off REAL, minutes REAL,
    goals INTEGER, own_goals INTEGER, assists INTEGER, shots INTEGER, xg REAL, xa REAL,
    key_passes INTEGER, passes INTEGER, passes_completed INTEGER, progressive_passes INTEGER,
    progressive_carries INTEGER, crosses INTEGER, pressures INTEGER, tackles INTEGER,
    interceptions INTEGER, recoveries INTEGER, clearances INTEGER, dribbles INTEGER,
    fouls INTEGER, yellow INTEGER, red INTEGER,
    gk_shots_faced INTEGER, gk_goals_conceded INTEGER, gk_xg_faced REAL, gk_saves INTEGER,
    PRIMARY KEY(sb_match_id, player_id)
);
CREATE INDEX IF NOT EXISTS ix_appearances_player ON appearances(player_id, position);

CREATE TABLE IF NOT EXISTS model_versions (
    version TEXT PRIMARY KEY,
    family TEXT NOT NULL,
    params TEXT NOT NULL,
    trained_through TEXT,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id INTEGER PRIMARY KEY,
    match_id INTEGER NOT NULL REFERENCES matches(match_id),
    model_version TEXT NOT NULL,
    purpose TEXT NOT NULL,
    created_at TEXT NOT NULL,
    data_cutoff TEXT NOT NULL,
    p_home REAL NOT NULL, p_draw REAL NOT NULL, p_away REAL NOT NULL,
    lambda_home REAL, lambda_away REAL,
    markets TEXT,
    UNIQUE(match_id, model_version, purpose)
);
CREATE INDEX IF NOT EXISTS ix_predictions_model ON predictions(model_version, purpose);
CREATE INDEX IF NOT EXISTS ix_predictions_purpose_match ON predictions(purpose, match_id, prediction_id);

CREATE TRIGGER IF NOT EXISTS predictions_immutable
BEFORE UPDATE ON predictions
BEGIN
    SELECT RAISE(ABORT, 'predictions are immutable');
END;

CREATE TRIGGER IF NOT EXISTS predictions_no_post_result_forward
BEFORE INSERT ON predictions
WHEN NEW.purpose = 'forward' AND (SELECT status FROM matches WHERE match_id = NEW.match_id) <> 'scheduled'
BEGIN
    SELECT RAISE(ABORT, 'forward prediction for a match that is no longer scheduled');
END;

CREATE TABLE IF NOT EXISTS prediction_flags (
    prediction_id INTEGER NOT NULL REFERENCES predictions(prediction_id),
    flag TEXT NOT NULL,
    reason TEXT NOT NULL,
    flagged_at TEXT NOT NULL,
    PRIMARY KEY(prediction_id, flag)
);

CREATE TABLE IF NOT EXISTS backtest_runs (
    run_id INTEGER PRIMARY KEY,
    model_version TEXT NOT NULL,
    split TEXT NOT NULL,
    created_at TEXT NOT NULL,
    train_window TEXT,
    test_period TEXT,
    n_matches INTEGER,
    metrics TEXT NOT NULL,
    by_league TEXT,
    by_season TEXT,
    reliability TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    name TEXT NOT NULL,
    hypothesis TEXT NOT NULL,
    result TEXT,
    decision TEXT,
    details TEXT
);

CREATE TABLE IF NOT EXISTS jobs (
    job_id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,
    args TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL,
    stage TEXT,
    done_units INTEGER NOT NULL DEFAULT 0,
    total_units INTEGER,
    created_at TEXT NOT NULL,
    started_at TEXT,
    heartbeat_at TEXT,
    finished_at TEXT,
    error TEXT,
    retries INTEGER NOT NULL DEFAULT 0,
    max_retries INTEGER NOT NULL DEFAULT 2,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    worker_pid INTEGER,
    result TEXT
);

CREATE TABLE IF NOT EXISTS job_logs (
    log_id INTEGER PRIMARY KEY,
    job_id INTEGER NOT NULL,
    ts TEXT NOT NULL,
    level TEXT NOT NULL,
    message TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_job_logs ON job_logs(job_id);

CREATE TABLE IF NOT EXISTS notifications (
    notification_id INTEGER PRIMARY KEY,
    ts TEXT NOT NULL,
    level TEXT NOT NULL,
    title TEXT NOT NULL,
    body TEXT,
    seen INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS provider_health (
    provider TEXT PRIMARY KEY,
    last_attempt TEXT,
    last_success TEXT,
    last_error TEXT,
    requests_today INTEGER DEFAULT 0,
    rate_limit TEXT,
    status TEXT
);

CREATE TABLE IF NOT EXISTS live_sessions (
    live_id INTEGER PRIMARY KEY,
    mode TEXT NOT NULL,
    provider TEXT NOT NULL,
    sb_match_id INTEGER,
    match_id INTEGER,
    home_name TEXT, away_name TEXT,
    status TEXT NOT NULL,
    speed REAL,
    created_at TEXT NOT NULL,
    last_event_at TEXT,
    last_update_at TEXT,
    minute REAL, home_goals INTEGER, away_goals INTEGER,
    prematch TEXT,
    job_id INTEGER
);

CREATE TABLE IF NOT EXISTS live_events (
    live_id INTEGER NOT NULL,
    provider_event_id TEXT NOT NULL,
    provider_seq INTEGER,
    received_at TEXT NOT NULL,
    period INTEGER, clock REAL,
    kind TEXT NOT NULL,
    side TEXT,
    player_name TEXT,
    xg REAL,
    payload TEXT,
    superseded INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(live_id, provider_event_id)
);

CREATE TABLE IF NOT EXISTS live_snapshots (
    snapshot_id INTEGER PRIMARY KEY,
    live_id INTEGER NOT NULL,
    ts TEXT NOT NULL,
    clock REAL NOT NULL,
    trigger TEXT NOT NULL,
    home_goals INTEGER, away_goals INTEGER,
    p_home REAL, p_draw REAL, p_away REAL,
    p_another_goal REAL, exp_remaining REAL,
    state TEXT,
    explanation TEXT
);
CREATE INDEX IF NOT EXISTS ix_live_snapshots ON live_snapshots(live_id, snapshot_id);

CREATE TABLE IF NOT EXISTS live_model (
    version TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    params TEXT NOT NULL,
    metrics TEXT,
    status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def clean(obj):
    if isinstance(obj, float):
        return None if obj != obj or obj in (float("inf"), float("-inf")) else obj
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            return clean(obj.item())
        except Exception:
            return str(obj)
    return obj


def dumps(obj) -> str:
    return json.dumps(clean(obj), default=str, allow_nan=False)


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def connect(path=None) -> sqlite3.Connection:
    target = Path(path or DB_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(target), timeout=60, isolation_level=None)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=60000")
    return con


def migrate(con):
    sql = con.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='matches'").fetchone()
    if sql and "UNIQUE(competition, season, home_team_id, away_team_id)" in sql[0]:
        con.execute("PRAGMA foreign_keys=OFF")
        con.execute("PRAGMA legacy_alter_table=ON")
        con.executescript("""
            BEGIN;
            ALTER TABLE matches RENAME TO matches_old;
            CREATE TABLE matches (
                match_id INTEGER PRIMARY KEY,
                competition TEXT NOT NULL REFERENCES competitions(code),
                season INTEGER NOT NULL,
                match_date TEXT NOT NULL,
                kickoff TEXT,
                home_team_id INTEGER NOT NULL REFERENCES teams(team_id),
                away_team_id INTEGER NOT NULL REFERENCES teams(team_id),
                status TEXT NOT NULL,
                fthg INTEGER, ftag INTEGER, hthg INTEGER, htag INTEGER,
                referee TEXT, attendance INTEGER,
                hs INTEGER, aws INTEGER, hst INTEGER, ast INTEGER,
                hc INTEGER, ac INTEGER, hf INTEGER, af INTEGER,
                hy INTEGER, ay INTEGER, hr INTEGER, ar INTEGER,
                source TEXT NOT NULL,
                source_row_hash TEXT,
                first_seen TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            INSERT INTO matches SELECT * FROM matches_old;
            DROP TABLE matches_old;
            COMMIT;
        """)
        con.execute("PRAGMA foreign_keys=ON")


def add_columns(con):
    cols = {r[1] for r in con.execute("PRAGMA table_info(sb_events)")}
    if cols and "pattern" not in cols:
        con.execute("ALTER TABLE sb_events ADD COLUMN pattern TEXT")


def init(path=None) -> sqlite3.Connection:
    con = connect(path)
    con.executescript("CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL);")
    if con.execute("SELECT 1 FROM sqlite_master WHERE name='matches'").fetchone():
        migrate(con)
    con.executescript(SCHEMA)
    add_columns(con)
    return con


@contextmanager
def transaction(con):
    con.execute("BEGIN IMMEDIATE")
    try:
        yield con
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise


def set_kv(con, key, value):
    con.execute(
        "INSERT INTO kv(key, value, updated_at) VALUES(?,?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
        (key, dumps(value), now()),
    )


def get_kv(con, key, default=None):
    row = con.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
    return json.loads(row["value"]) if row else default


def notify(con, level, title, body=None):
    con.execute("INSERT INTO notifications(ts, level, title, body) VALUES(?,?,?,?)", (now(), level, title, body))


def provider_event(con, provider, ok, error=None, rate_limit=None):
    ts = now()
    con.execute(
        """INSERT INTO provider_health(provider, last_attempt, last_success, last_error, requests_today, rate_limit, status)
           VALUES(?,?,?,?,1,?,?)
           ON CONFLICT(provider) DO UPDATE SET
             last_attempt=excluded.last_attempt,
             last_success=COALESCE(excluded.last_success, provider_health.last_success),
             last_error=CASE WHEN excluded.last_error IS NULL THEN provider_health.last_error ELSE excluded.last_error END,
             requests_today=provider_health.requests_today+1,
             rate_limit=COALESCE(excluded.rate_limit, provider_health.rate_limit),
             status=excluded.status""",
        (provider, ts, ts if ok else None, error, rate_limit, "ok" if ok else "error"),
    )
