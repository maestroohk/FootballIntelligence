import os
from datetime import datetime, timedelta, timezone

from . import db, jobs

CYCLE_HOURS = float(os.environ.get("FIE_CYCLE_HOURS", "6"))


def tick(con):
    if CYCLE_HOURS <= 0:
        return
    last = db.get_kv(con, "scheduler_last_cycle")
    now = datetime.now(timezone.utc)
    if last and now - datetime.fromisoformat(last) < timedelta(hours=CYCLE_HOURS):
        return
    active = con.execute("SELECT 1 FROM jobs WHERE kind='daily_cycle' AND status IN ('queued','running')").fetchone()
    if not active:
        jobs.enqueue(con, "daily_cycle", {"trigger": "scheduler"})
    db.set_kv(con, "scheduler_last_cycle", now.isoformat())
