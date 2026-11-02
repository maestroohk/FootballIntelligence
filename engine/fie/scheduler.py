import os
from datetime import datetime, timedelta, timezone

from . import db, jobs

CYCLE_HOURS = float(os.environ.get("FIE_CYCLE_HOURS", "6"))
RESEARCH_DAYS = float(os.environ.get("FIE_RESEARCH_DAYS", "7"))


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
    last_research = db.get_kv(con, "scheduler_last_research")
    if RESEARCH_DAYS > 0 and (last_research is None or now - datetime.fromisoformat(last_research) >= timedelta(days=RESEARCH_DAYS)):
        if last_research is not None:
            jobs.enqueue(con, "research_full", {"refresh": True})
        db.set_kv(con, "scheduler_last_research", now.isoformat())
