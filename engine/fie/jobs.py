import json
import os
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timedelta, timezone

from . import db
from .config import MAX_CONCURRENT_JOBS, ROOT

STALE_SECONDS = 90
CANCEL_GRACE_SECONDS = 30
HANDLERS = {}


def handler(kind):
    def wrap(fn):
        HANDLERS[kind] = fn
        return fn
    return wrap


def enqueue(con, kind, args=None, max_retries=2, dedupe=True):
    payload = db.dumps(args or {})
    existing = con.execute("SELECT job_id FROM jobs WHERE kind=? AND args=? AND status IN ('queued','running') AND cancel_requested=0 ORDER BY job_id LIMIT 1",
                           (kind, payload)).fetchone()
    if existing and dedupe:
        return existing[0]
    con.execute("INSERT INTO jobs(kind, args, status, created_at, max_retries) VALUES(?,?,?,?,?)",
                (kind, payload, "queued", db.now(), max_retries))
    return con.execute("SELECT last_insert_rowid()").fetchone()[0]


class JobContext:
    def __init__(self, con, job_id):
        self.con = con
        self.job_id = job_id
        self._stop = threading.Event()
        self._hb = threading.Thread(target=self._heartbeat, daemon=True)
        self._hb.start()

    def _heartbeat(self):
        hb = db.connect()
        while not self._stop.wait(5):
            hb.execute("UPDATE jobs SET heartbeat_at=? WHERE job_id=?", (db.now(), self.job_id))
        hb.close()

    def stop(self):
        self._stop.set()

    def progress(self, done, total=None, stage=None):
        self.con.execute("UPDATE jobs SET done_units=?, total_units=COALESCE(?, total_units), stage=COALESCE(?, stage), heartbeat_at=? WHERE job_id=?",
                         (int(done), total, stage, db.now(), self.job_id))

    def stage(self, stage):
        self.con.execute("UPDATE jobs SET stage=?, heartbeat_at=? WHERE job_id=?", (stage, db.now(), self.job_id))
        self.log(stage)

    def log(self, message, level="info"):
        self.con.execute("INSERT INTO job_logs(job_id, ts, level, message) VALUES(?,?,?,?)", (self.job_id, db.now(), level, str(message)[:4000]))

    def cancelled(self):
        row = self.con.execute("SELECT cancel_requested FROM jobs WHERE job_id=?", (self.job_id,)).fetchone()
        return bool(row and row["cancel_requested"])


class Cancelled(Exception):
    pass


def run_job(job_id):
    from . import tasks
    con = db.connect()
    job = con.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
    ctx = JobContext(con, job_id)
    try:
        fn = HANDLERS[job["kind"]]
        args = json.loads(job["args"] or "{}")
        from .config import HEAVY_JOBS, LIGHT
        if LIGHT and job["kind"] in HEAVY_JOBS and not args.pop("allow_heavy", False):
            raise RuntimeError(f"{job['kind']} is a full-research job and this machine runs the light profile (FIE_PROFILE=light); run it on the research machine or queue it with allow_heavy")
        args.pop("allow_heavy", None)
        result = fn(ctx, **args)
        status = "cancelled" if ctx.cancelled() else "succeeded"
        con.execute("UPDATE jobs SET status=?, finished_at=?, result=?, heartbeat_at=? WHERE job_id=?",
                    (status, db.now(), db.dumps(result)[:20000] if result is not None else None, db.now(), job_id))
        ctx.log(f"job {status}")
        if job["kind"] in tasks.NOTIFY_KINDS:
            db.notify(con, "success" if status == "succeeded" else "info", f"{tasks.NOTIFY_KINDS[job['kind']]} {status}", f"Job #{job_id}")
    except Cancelled:
        con.execute("UPDATE jobs SET status='cancelled', finished_at=? WHERE job_id=?", (db.now(), job_id))
        ctx.log("job cancelled")
    except Exception as exc:
        tb = traceback.format_exc()
        ctx.log(tb, "error")
        con.execute("UPDATE jobs SET status='failed', finished_at=?, error=? WHERE job_id=?", (db.now(), f"{type(exc).__name__}: {exc}"[:2000], job_id))
        db.notify(con, "error", f"Job #{job_id} ({job['kind']}) failed", str(exc)[:300])
    finally:
        ctx.stop()


def _pid_alive(pid):
    if not pid:
        return False
    if os.name == "nt":
        import ctypes
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return code.value == 259
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def recover(con):
    cutoff = (datetime.now(timezone.utc) - timedelta(seconds=STALE_SECONDS)).strftime("%Y-%m-%dT%H:%M:%S")
    for job in con.execute("SELECT * FROM jobs WHERE status='running'").fetchall():
        stale = (job["heartbeat_at"] or job["started_at"] or "") < cutoff
        if stale and not _pid_alive(job["worker_pid"]):
            if job["retries"] < job["max_retries"] and not job["cancel_requested"]:
                con.execute("UPDATE jobs SET status='queued', retries=retries+1, worker_pid=NULL, stage='recovered after interruption' WHERE job_id=?", (job["job_id"],))
                con.execute("INSERT INTO job_logs(job_id, ts, level, message) VALUES(?,?,?,?)", (job["job_id"], db.now(), "warning", "recovered after worker interruption; requeued"))
            else:
                con.execute("UPDATE jobs SET status='failed', finished_at=?, error='interrupted' WHERE job_id=?", (db.now(), job["job_id"]))


def worker_loop(poll=2.0, scheduler=None):
    con = db.init()
    other = db.get_kv(con, "worker")
    if other and other.get("pid") != os.getpid() and _pid_alive(other.get("pid")):
        print(f"another worker (pid {other['pid']}) is alive; exiting", flush=True)
        return
    db.set_kv(con, "worker", {"pid": os.getpid(), "started_at": db.now()})
    procs = {}
    cancel_seen = {}
    while True:
        db.set_kv(con, "worker_heartbeat", db.now())
        recover(con)
        for job_id, proc in list(procs.items()):
            if proc.poll() is not None:
                procs.pop(job_id)
                row = con.execute("SELECT status FROM jobs WHERE job_id=?", (job_id,)).fetchone()
                if row and row["status"] == "running":
                    con.execute("UPDATE jobs SET status='failed', finished_at=?, error=? WHERE job_id=?", (db.now(), f"process exited with code {proc.returncode}", job_id))
                continue
            row = con.execute("SELECT cancel_requested FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row and row["cancel_requested"]:
                first = cancel_seen.setdefault(job_id, time.time())
                if time.time() - first > CANCEL_GRACE_SECONDS:
                    proc.kill()
                    con.execute("UPDATE jobs SET status='cancelled', finished_at=? WHERE job_id=?", (db.now(), job_id))
        for job in con.execute("SELECT job_id FROM jobs WHERE status='queued' AND cancel_requested=1").fetchall():
            con.execute("UPDATE jobs SET status='cancelled', finished_at=? WHERE job_id=?", (db.now(), job["job_id"]))
        while len(procs) < MAX_CONCURRENT_JOBS:
            job = con.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY job_id LIMIT 1").fetchone()
            if not job:
                break
            exclusive = {"research_full", "ingest_history"}
            if job["kind"] in exclusive and any(con.execute("SELECT kind FROM jobs WHERE job_id=?", (j,)).fetchone()["kind"] == job["kind"] for j in procs):
                break
            claimed = con.execute("UPDATE jobs SET status='running', started_at=COALESCE(started_at, ?), heartbeat_at=? WHERE job_id=? AND status='queued'",
                                  (db.now(), db.now(), job["job_id"])).rowcount
            if not claimed:
                continue
            proc = subprocess.Popen([sys.executable, "-m", "fie", "run-job", str(job["job_id"])], cwd=str(ROOT / "engine"),
                                    creationflags=0x00004000 if os.name == "nt" else 0)
            con.execute("UPDATE jobs SET worker_pid=? WHERE job_id=?", (proc.pid, job["job_id"]))
            procs[job["job_id"]] = proc
        if scheduler:
            scheduler(con)
        time.sleep(poll)
