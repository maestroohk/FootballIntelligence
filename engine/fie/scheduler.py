import os
from datetime import datetime, timedelta, timezone

from . import db, jobs
from .config import LIGHT

CYCLE_HOURS = float(os.environ.get("FIE_CYCLE_HOURS", "24" if LIGHT else "6"))
RESEARCH_DAYS = float(os.environ.get("FIE_RESEARCH_DAYS", "0" if LIGHT else "7"))
RESEARCH_MIN_NEW = int(os.environ.get("FIE_RESEARCH_MIN_NEW", "300"))


LIVE_POLL_SECONDS = float(os.environ.get("FIE_LIVE_POLL_SECONDS", "60"))
_last_live_poll = [0.0]


def poll_live(con):
    import time
    if not os.environ.get("FOOTBALL_DATA_API_KEY") or time.time() - _last_live_poll[0] < LIVE_POLL_SECONDS:
        return
    _last_live_poll[0] = time.time()
    from .live import poll_live_provider
    try:
        poll_live_provider(con)
    except Exception as exc:
        db.provider_event(con, "football-data.org live", False, str(exc)[:300])


def tick(con):
    poll_live(con)
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
            new = con.execute("SELECT COUNT(*) FROM matches WHERE status='finished' AND updated_at > ?", (last_research,)).fetchone()[0]
            if new < RESEARCH_MIN_NEW:
                db.set_kv(con, "research_postponed", {"at": now.isoformat(), "new_results": new, "needed": RESEARCH_MIN_NEW})
                db.set_kv(con, "scheduler_last_research", now.isoformat())
                return
            jobs.enqueue(con, "research_full", {"refresh": True})
        db.set_kv(con, "scheduler_last_research", now.isoformat())
