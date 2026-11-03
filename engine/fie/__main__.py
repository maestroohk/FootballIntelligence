import argparse
import json
import sys

import truststore

from . import db, jobs
from .config import limit_threads


def main():
    truststore.inject_into_ssl()
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
    sub.add_parser("assets")
    ex = sub.add_parser("bundle-export")
    ex.add_argument("target")
    ver = sub.add_parser("bundle-verify")
    ver.add_argument("source")
    imp = sub.add_parser("bundle-import")
    imp.add_argument("source")
    imp.add_argument("--replace", action="store_true")
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
    elif args.cmd in ("assets", "bundle-export", "bundle-verify", "bundle-import"):
        from . import bundle
        if args.cmd == "assets":
            print(json.dumps(bundle.assets(db.init()), indent=1))
        elif args.cmd == "bundle-export":
            m = bundle.export(args.target)
            print(f"exported {len(m['files'])} files to {args.target}")
            print(json.dumps(m["inventory"], indent=1))
        elif args.cmd == "bundle-verify":
            m, problems = bundle.verify(args.source)
            print(f"{len(m['files'])} files from {m['machine']} created {m['created_at']}: " + ("ok" if not problems else f"{len(problems)} problems"))
            for p in problems:
                print(" ", p)
            sys.exit(1 if problems else 0)
        else:
            status = bundle.import_bundle(args.source, replace=args.replace)
            print("operational" if status["operational"] else "still missing: " + ", ".join(m["asset"] for m in status["missing"]))
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
