import hashlib
import json
import os
import platform
import shutil
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from . import db
from .config import DATA, DB_PATH, RAW, ROOT
from .research import CACHE

FORMAT = 1
REQUIRED_CACHES = ["frame", "features", "oos_ensemble"]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip() or None
    except OSError:
        return None


def inventory(con):
    def one(sql):
        try:
            return con.execute(sql).fetchone()[0]
        except sqlite3.Error:
            return None
    summary = db.get_kv(con, "research_summary") if one("SELECT COUNT(*) FROM kv") is not None else None
    return {
        "matches_history": one("SELECT COUNT(*) FROM matches WHERE source LIKE 'football-data%'"),
        "matches_total": one("SELECT COUNT(*) FROM matches"),
        "odds": one("SELECT COUNT(*) FROM odds"),
        "predictions": one("SELECT COUNT(*) FROM predictions"),
        "forward_predictions": one("SELECT COUNT(*) FROM predictions WHERE purpose='forward'"),
        "model_versions": one("SELECT COUNT(*) FROM model_versions"),
        "statsbomb_matches": one("SELECT COUNT(*) FROM sb_matches WHERE processed=1"),
        "players": one("SELECT COUNT(*) FROM players"),
        "champion": (summary or {}).get("selection", {}).get("champion"),
        "latest_result": one("SELECT MAX(match_date) FROM matches WHERE status='finished'"),
    }


def assets(con):
    inv = inventory(con)
    caches = sorted(p.stem for p in CACHE.glob("*.pkl")) if CACHE.exists() else []
    raw = sum(1 for p in RAW.rglob("*") if p.is_file()) if RAW.exists() else 0
    missing = []
    if not inv["matches_history"]:
        missing.append({"asset": "football-data.co.uk history", "fix": "import a bundle (fie bundle-import) or run ingest_history on a network that can reach football-data.co.uk"})
    if not inv["champion"]:
        missing.append({"asset": "trained champion model", "fix": "import a bundle, or run research_full after the history is available"})
    for name in REQUIRED_CACHES:
        if name not in caches:
            missing.append({"asset": f"model cache {name}", "fix": "import a bundle, or rebuild with research_full"})
    if not inv["statsbomb_matches"]:
        missing.append({"asset": "StatsBomb events", "fix": "run ingest_statsbomb"})
    out = {"checked_at": db.now(), "database": inv, "model_caches": caches, "raw_files": raw, "missing": missing, "operational": not missing}
    db.set_kv(con, "assets_status", out)
    return out


def export(target):
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    size = DB_PATH.stat().st_size + sum(p.stat().st_size for base in (CACHE, RAW) if base.exists() for p in base.rglob("*") if p.is_file())
    free = shutil.disk_usage(target).free
    if size * 1.05 > free:
        raise RuntimeError(f"not enough disk space at {target}: need {size / 1e9:.1f} GB, {free / 1e9:.1f} GB free")
    (target / "cache").mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(DB_PATH)
    dst = sqlite3.connect(target / "football.db")
    src.backup(dst)
    dst.close()
    src.close()
    files = ["football.db"]
    for folder, base in (("cache", CACHE), ("raw", RAW)):
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file():
                rel = f"{folder}/{p.relative_to(base).as_posix()}"
                out = target / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                if not out.exists() or out.stat().st_size != p.stat().st_size or _sha256(out) != _sha256(p):
                    shutil.copy2(p, out)
                files.append(rel)
    con = sqlite3.connect(target / "football.db")
    con.row_factory = sqlite3.Row
    manifest = {"format": FORMAT, "created_at": db.now(), "machine": platform.node(), "commit": _commit(), "inventory": inventory(con),
                "files": [{"path": f, "size": (target / f).stat().st_size, "sha256": _sha256(target / f)} for f in files]}
    con.close()
    (target / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def verify(source):
    source = Path(source)
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest.get("format") != FORMAT:
        raise RuntimeError(f"unsupported bundle format {manifest.get('format')}")
    problems = []
    for f in manifest["files"]:
        p = source / f["path"]
        if not p.exists():
            problems.append(f"missing {f['path']}")
        elif p.stat().st_size != f["size"] or _sha256(p) != f["sha256"]:
            problems.append(f"checksum mismatch {f['path']}")
    return manifest, problems


def _destination(rel):
    folder, _, rest = rel.partition("/")
    return DB_PATH if rel == "football.db" else (CACHE if folder == "cache" else RAW) / rest


def _needs_copy(dest, f):
    return not (dest.exists() and dest.stat().st_size == f["size"] and _sha256(dest) == f["sha256"])


def _worker_active():
    if not DB_PATH.exists():
        return False
    con = db.connect()
    try:
        beat = db.get_kv(con, "worker_heartbeat")
    finally:
        con.close()
    if not beat:
        return False
    age = datetime.now(timezone.utc) - datetime.fromisoformat(beat.replace("Z", "+00:00"))
    return age.total_seconds() < 60


def _validate_staged(staged):
    problems = []
    db_part = staged.get("football.db")
    if db_part is not None:
        con = sqlite3.connect(db_part)
        try:
            if con.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                problems.append("database integrity check failed")
            tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            missing = {"matches", "teams", "predictions", "kv"} - tables
            if missing:
                problems.append(f"database is missing tables: {', '.join(sorted(missing))}")
        except sqlite3.DatabaseError as exc:
            problems.append(f"database unreadable: {exc}")
        finally:
            con.close()
    import pandas as pd
    for rel, part in staged.items():
        if rel.startswith("cache/") and Path(rel).stem in REQUIRED_CACHES:
            try:
                pd.read_pickle(part)
            except Exception as exc:
                problems.append(f"model cache {rel} cannot be loaded by this environment ({type(exc).__name__}: {str(exc)[:120]})")
    return problems


def _backup_database():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    target = DB_PATH.with_name(f"football.db.bak-{stamp}")
    src = sqlite3.connect(DB_PATH)
    dst = sqlite3.connect(target)
    src.backup(dst)
    dst.close()
    src.close()
    return target


def import_bundle(source, replace=False, force=False, progress=print):
    source = Path(source)
    progress("verifying bundle checksums")
    manifest, problems = verify(source)
    if problems:
        raise RuntimeError("bundle failed verification: " + "; ".join(problems[:10]))
    if DB_PATH.exists() and not replace:
        con = db.connect()
        current = inventory(con)
        con.close()
        if current["matches_history"]:
            raise RuntimeError("this machine already has match history; pass --replace to overwrite it (the current database is backed up first)")
    if _worker_active() and not force:
        raise RuntimeError("the engine worker is running and writing to the database; stop the web app and worker first (or pass --force)")
    todo = [f for f in manifest["files"] if _needs_copy(_destination(f["path"]), f)]
    needed = sum(f["size"] for f in todo) + (DB_PATH.stat().st_size if DB_PATH.exists() and any(f["path"] == "football.db" for f in todo) else 0)
    DATA.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(DATA).free
    if needed * 1.05 > free:
        raise RuntimeError(f"not enough disk space: need {needed / 1e9:.1f} GB, {free / 1e9:.1f} GB free")
    staged = {}
    for i, f in enumerate(todo, 1):
        dest = _destination(f["path"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        if part.exists() and part.stat().st_size == f["size"] and _sha256(part) == f["sha256"]:
            progress(f"[{i}/{len(todo)}] already staged {f['path']}")
        else:
            shutil.copy2(source / f["path"], part)
            if _sha256(part) != f["sha256"]:
                part.unlink()
                raise RuntimeError(f"copy of {f['path']} did not match its checksum; rerun to resume")
            progress(f"[{i}/{len(todo)}] staged {f['path']}")
        staged[f["path"]] = part
    problems = _validate_staged(staged)
    if problems:
        raise RuntimeError("staged files failed validation, nothing was replaced: " + "; ".join(problems))
    backup = None
    if "football.db" in staged and DB_PATH.exists():
        backup = _backup_database()
        progress(f"backed up current database to {backup.name}")
    for rel, part in sorted(staged.items(), key=lambda kv: kv[0] == "football.db"):
        dest = _destination(rel)
        if rel == "football.db":
            for suffix in ("-wal", "-shm"):
                side = dest.with_name(dest.name + suffix)
                if side.exists():
                    side.unlink()
        os.replace(part, dest)
    progress(f"replaced {len(staged)} files, {len(manifest['files']) - len(staged)} already up to date")
    con = db.init()
    db.set_kv(con, "restored_from_bundle", {"at": db.now(), "bundle_created_at": manifest["created_at"], "machine": manifest["machine"],
                                           "commit": manifest["commit"], "files": len(manifest["files"]), "replaced": len(staged),
                                           "backup": backup.name if backup else None})
    status = assets(con)
    db.notify(con, "success" if status["operational"] else "warning", "Data bundle imported",
              f"{len(staged)} files restored from {manifest['machine']} ({manifest['created_at'][:10]}); " + ("all required assets present" if status["operational"] else f"{len(status['missing'])} assets still missing"))
    return status
