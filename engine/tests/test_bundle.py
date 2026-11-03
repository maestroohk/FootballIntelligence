import json

import pandas as pd
import pytest

from fie import bundle, db


@pytest.fixture()
def machine(tmp_path, monkeypatch):
    def use(name):
        root = tmp_path / name
        for module in (bundle, db):
            monkeypatch.setattr(module, "DB_PATH", root / "football.db")
        monkeypatch.setattr(bundle, "DATA", root)
        monkeypatch.setattr(bundle, "CACHE", root / "cache")
        monkeypatch.setattr(bundle, "RAW", root / "raw")
        return root
    return use


def _source_with_history(root):
    con = db.init(root / "football.db")
    con.execute("INSERT INTO competitions VALUES('E0','PL','ENG',1)")
    con.execute("INSERT INTO teams(team_id, name, country) VALUES(1,'A','ENG'),(2,'B','ENG')")
    con.execute("INSERT INTO matches(match_id, competition, season, match_date, home_team_id, away_team_id, status, fthg, ftag, source, first_seen, updated_at) VALUES(1,'E0',2024,'2024-08-17',1,2,'finished',2,1,'football-data.co.uk',?,?)", (db.now(), db.now()))
    con.close()
    (root / "cache").mkdir(parents=True)
    for name in bundle.REQUIRED_CACHES:
        pd.DataFrame({"x": [1, 2]}).to_pickle(root / "cache" / f"{name}.pkl")
    (root / "raw").mkdir()
    (root / "raw" / "E0_2024.csv").write_text("Div,HomeTeam\nE0,A\n")


def test_round_trip_resume_and_corruption(machine, tmp_path):
    _source_with_history(machine("laptop_a"))
    bundle.export(tmp_path / "bundle")
    manifest, problems = bundle.verify(tmp_path / "bundle")
    assert not problems and len(manifest["files"]) == 5

    machine("laptop_b")
    status = bundle.import_bundle(tmp_path / "bundle", progress=lambda _: None)
    assert status["database"]["matches_history"] == 1
    assert not [m for m in status["missing"] if m["asset"].startswith("model cache")]

    with pytest.raises(RuntimeError, match="already has match history"):
        bundle.import_bundle(tmp_path / "bundle", progress=lambda _: None)

    log = []
    bundle.import_bundle(tmp_path / "bundle", replace=True, progress=log.append)
    assert any("already up to date" in line for line in log)

    with open(tmp_path / "bundle" / "raw" / "E0_2024.csv", "a") as f:
        f.write("tampered\n")
    _, problems = bundle.verify(tmp_path / "bundle")
    assert problems == ["checksum mismatch raw/E0_2024.csv"]
    with pytest.raises(RuntimeError, match="failed verification"):
        bundle.import_bundle(tmp_path / "bundle", replace=True, progress=lambda _: None)


def test_invalid_database_is_rejected_before_anything_is_replaced(machine, tmp_path):
    _source_with_history(machine("laptop_a"))
    bundle.export(tmp_path / "bundle")
    broken = tmp_path / "bundle" / "football.db"
    broken.write_bytes(b"not a database" * 100)
    manifest = json.loads((tmp_path / "bundle" / "manifest.json").read_text())
    for f in manifest["files"]:
        if f["path"] == "football.db":
            f["size"], f["sha256"] = broken.stat().st_size, bundle._sha256(broken)
    (tmp_path / "bundle" / "manifest.json").write_text(json.dumps(manifest))

    target = machine("laptop_b")
    db.init(target / "football.db").close()
    with pytest.raises(RuntimeError, match="nothing was replaced"):
        bundle.import_bundle(tmp_path / "bundle", progress=lambda _: None)
    assert not (target / "cache" / "frame.pkl").exists()
    con = db.connect(target / "football.db")
    assert con.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 0


def test_doctor_detects_pending_migrations_without_touching_data(machine, monkeypatch):
    from fie import doctor
    root = machine("home")
    monkeypatch_target = root / "football.db"
    _source_with_history(root)
    monkeypatch.setattr(doctor, "DB_PATH", monkeypatch_target)
    assert not doctor._schema_changes()
    con = db.connect(monkeypatch_target)
    con.execute("DROP TABLE notifications")
    con.close()
    assert doctor._schema_changes()
    con = db.connect(monkeypatch_target)
    assert con.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 1


def test_light_profile_refuses_heavy_jobs_unless_allowed(machine, monkeypatch):
    from fie import config, jobs
    root = machine("laptop")
    con = db.init(root / "football.db")
    monkeypatch.setattr(config, "LIGHT", True)
    job_id = jobs.enqueue(con, "research_full", {})
    jobs.run_job(job_id)
    row = con.execute("SELECT status, error FROM jobs WHERE job_id=?", (job_id,)).fetchone()
    assert row["status"] == "failed" and "light profile" in row["error"]
