import argparse
import json
import sys

from . import db, jobs
from .config import limit_threads


def main():
    limit_threads()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="fie")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    sub.add_parser("worker")
    rj = sub.add_parser("run-job")
    rj.add_argument("job_id", type=int)
    enq = sub.add_parser("enqueue")
    enq.add_argument("kind")
    enq.add_argument("--args", default="{}")
    run = sub.add_parser("run")
    run.add_argument("kind")
    run.add_argument("--args", default="{}")
    args = parser.parse_args()
    from . import tasks, scheduler
    if args.cmd == "init":
        db.init()
        print("initialised")
    elif args.cmd == "worker":
        db.init()
        jobs.worker_loop(scheduler=scheduler.tick)
    elif args.cmd == "run-job":
        jobs.run_job(args.job_id)
    elif args.cmd == "enqueue":
        con = db.init()
        print(jobs.enqueue(con, args.kind, json.loads(args.args)))
    elif args.cmd == "run":
        con = db.init()
        job_id = jobs.enqueue(con, args.kind, json.loads(args.args))
        con.execute("UPDATE jobs SET status='running', started_at=? WHERE job_id=?", (db.now(), job_id))
        jobs.run_job(job_id)
        row = con.execute("SELECT status, result, error FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        print(row["status"], row["error"] or "")
        print(row["result"])


if __name__ == "__main__":
    main()
