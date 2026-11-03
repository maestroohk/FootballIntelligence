import json
import sqlite3
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from . import bundle, db
from .config import DATA, DB_PATH, PROFILE, ROOT

RESEARCH_STEPS = ["research_full", "final_test", "train_inplay", "predict_upcoming", "evaluate_forward", "build_report"]


class Report:
    def __init__(self):
        self.rows = []

    def add(self, area, status, detail):
        self.rows.append((area, status, detail))
        print(f"[{status.upper():7}] {area}: {detail}", flush=True)

    def markdown(self, started):
        counts = {s: sum(1 for r in self.rows if r[1] == s) for s in ("pass", "warn", "fail", "skipped")}
        lines = [f"# Validation report", "", f"- Started: {started}", f"- Profile: {PROFILE}", f"- Database: {DB_PATH}",
                 f"- Result: {counts['pass']} passed, {counts['warn']} warnings, {counts['fail']} failed, {counts['skipped']} not validated", "",
                 "| Area | Status | Detail |", "|---|---|---|"]
        lines += [f"| {a} | {s} | {d.replace('|', '/')} |" for a, s, d in self.rows]
        return "\n".join(lines) + "\n"


def _requirements(rep):
    missing = []
    for line in (ROOT / "engine" / "requirements.txt").read_text().splitlines():
        if "==" not in line:
            continue
        name, want = line.strip().split("==")
        try:
            have = metadata.version(name)
        except metadata.PackageNotFoundError:
            have = None
        if have != want:
            missing.append(f"{name} {have or 'missing'} (want {want})")
    rep.add("Python packages", "pass" if not missing else "fail", f"Python {sys.version.split()[0]}; " + ("all pinned versions installed" if not missing else "; ".join(missing)))


def _database(rep, migrate):
    if not DB_PATH.exists():
        rep.add("Database", "fail", f"{DB_PATH} does not exist; import a bundle or run setup.ps1")
        return False
    con = sqlite3.connect(DB_PATH)
    try:
        check = con.execute("PRAGMA quick_check").fetchone()[0]
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        con.close()
    if check != "ok":
        rep.add("Database integrity", "fail", f"quick_check: {check}")
        return False
    rep.add("Database integrity", "pass", f"quick_check ok, {len(tables)} tables, {DB_PATH.stat().st_size / 1e6:.0f} MB")
    if not migrate:
        rep.add("Schema migration", "skipped", "run without --no-migrate to bring the schema up to date")
        return True
    if not _schema_changes():
        rep.add("Schema migration", "pass", "schema already current; nothing to migrate")
        return True
    backup = bundle._backup_database()
    db.init().close()
    rep.add("Schema migration", "pass", f"schema brought up to date after backup {backup.name}")
    return True


def _schema(con):
    order = {"table": 0, "index": 1, "view": 2, "trigger": 3}
    return sorted(((order.get(r[0], 4), r[1], r[2] or "") for r in con.execute("SELECT type, name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'")))


def _schema_changes():
    import tempfile
    src = sqlite3.connect(DB_PATH)
    before = _schema(src)
    src.close()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        probe = Path(tmp) / "schema.db"
        con = sqlite3.connect(probe)
        for _, _, sql in before:
            if sql:
                con.execute(sql)
        con.commit()
        con.close()
        migrated = db.init(probe)
        after = _schema(migrated)
        migrated.close()
    return before != after


def _assets(rep, con):
    a = bundle.assets(con)
    inv = a["database"]
    rep.add("Datasets", "pass" if inv["matches_history"] else "fail",
            f"{inv['matches_history'] or 0:,} historical matches, latest result {inv['latest_result']}, {inv['odds'] or 0:,} odds rows, {inv['statsbomb_matches'] or 0:,} StatsBomb matches, {inv['players'] or 0:,} players")
    rep.add("Champion model", "pass" if inv["champion"] else "fail", f"champion {inv['champion']}" if inv["champion"] else "no research_summary: run research_full")
    import pandas as pd
    from .research import CACHE
    bad = []
    for name in bundle.REQUIRED_CACHES:
        try:
            pd.read_pickle(CACHE / f"{name}.pkl")
        except Exception as exc:
            bad.append(f"{name} ({type(exc).__name__})")
    rep.add("Model artifacts", "pass" if not bad else "fail", "required caches load in this environment" if not bad else "cannot load: " + ", ".join(bad))
    return bool(inv["champion"]) and not bad


def _predictions(rep, con):
    from .forward import kickoff_utc_upper_bound
    from .scoreline import dixon_coles_matrix, outcome_probs
    rows = con.execute("""SELECT p.prediction_id, p.purpose, p.p_home, p.p_draw, p.p_away, p.lambda_home, p.lambda_away, p.markets, p.created_at, p.data_cutoff,
                                 m.match_date, m.kickoff, m.competition,
                                 (SELECT 1 FROM prediction_flags f WHERE f.prediction_id=p.prediction_id) flagged
                          FROM predictions p JOIN matches m ON m.match_id=p.match_id
                          WHERE p.model_version NOT LIKE 'bookmaker%'
                          ORDER BY p.prediction_id DESC LIMIT 20000""").fetchall()
    if not rows:
        rep.add("Prediction consistency", "skipped", "no stored predictions")
        return
    bad_sum = bad_matrix = bad_markets = late = leak = 0
    for r in rows:
        if abs(r["p_home"] + r["p_draw"] + r["p_away"] - 1) > 1e-6:
            bad_sum += 1
        mk = json.loads(r["markets"] or "{}")
        if r["lambda_home"] is not None and mk.get("rho") is not None:
            h, d, a = outcome_probs(dixon_coles_matrix(r["lambda_home"], r["lambda_away"], mk["rho"]))
            if max(abs(h - r["p_home"]), abs(d - r["p_draw"]), abs(a - r["p_away"])) > 1e-6:
                bad_matrix += 1
        if "p_home" in mk and abs(mk["p_home"] - r["p_home"]) > 1e-9:
            bad_markets += 1
        if r["data_cutoff"] and r["data_cutoff"] > r["match_date"]:
            leak += 1
        if r["purpose"] == "forward" and not r["flagged"]:
            created = datetime.fromisoformat(r["created_at"].replace("Z", "+00:00"))
            if created >= kickoff_utc_upper_bound(r["match_date"], r["kickoff"], r["competition"]):
                late += 1
    n = len(rows)
    rep.add("Prediction consistency", "pass" if not (bad_sum or bad_matrix or bad_markets) else "fail",
            f"{n:,} latest predictions: {bad_sum} with 1X2 not summing to 1, {bad_matrix} not matching their score matrix, {bad_markets} with stored markets disagreeing")
    rep.add("Leakage guard", "pass" if not (leak or late) else "fail", f"{leak} predictions with data cutoff after the match date, {late} unflagged forward predictions stored too close to kick-off")


def _jobs(rep, con):
    stuck = con.execute("SELECT COUNT(*) FROM jobs WHERE status='running'").fetchone()[0]
    failed = con.execute("SELECT kind, error FROM jobs WHERE status='failed' ORDER BY job_id DESC LIMIT 3").fetchall()
    rep.add("Background jobs", "warn" if stuck or failed else "pass",
            f"{stuck} marked running (re-queued by the worker if their process is gone); recent failures: " + ("; ".join(f"{k}: {(e or '')[:80]}" for k, e in failed) or "none"))


def _run(rep, kind, args=None):
    from . import jobs
    con = db.connect()
    job_id = jobs.enqueue(con, kind, {**(args or {}), "allow_heavy": True}, dedupe=False)
    con.execute("UPDATE jobs SET status='running', started_at=? WHERE job_id=?", (db.now(), job_id))
    jobs.run_job(job_id)
    row = con.execute("SELECT status, error FROM jobs WHERE job_id=?", (job_id,)).fetchone()
    con.close()
    rep.add(f"Job {kind}", "pass" if row["status"] == "succeeded" else "fail", f"job #{job_id} {row['status']}" + (f": {row['error'][:200]}" if row["error"] else ""))
    return row["status"] == "succeeded"


def run(research=False, migrate=True):
    started = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    rep = Report()
    _requirements(rep)
    if _database(rep, migrate):
        con = db.connect()
        ready = _assets(rep, con)
        _predictions(rep, con)
        _jobs(rep, con)
        con.close()
        if research and ready:
            for kind in RESEARCH_STEPS:
                if kind == "final_test":
                    con = db.connect()
                    done = db.get_kv(con, "final_test_run")
                    con.close()
                    if done:
                        rep.add("Job final_test", "skipped", f"already evaluated once at {done['at']}; it is not repeated so the test period stays untouched")
                        continue
                if not _run(rep, kind):
                    break
            con = db.connect()
            _predictions(rep, con)
            con.close()
        elif research:
            rep.add("Research pipeline", "skipped", "datasets or champion artifacts missing; fix the failures above first")
        else:
            rep.add("Research pipeline", "skipped", "not requested (pass --research)")
    for area in ("Browser workflows", "Fan/Analyst parity in the UI"):
        rep.add(area, "skipped", "manual: follow the browser checklist in README")
    out = DATA / "reports"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"validation-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
    path.write_text(rep.markdown(started), encoding="utf-8")
    print(f"\nreport: {path}")
    return 1 if any(r[1] == "fail" for r in rep.rows) else 0
