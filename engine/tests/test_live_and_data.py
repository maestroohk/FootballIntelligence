import sqlite3

import numpy as np
import pandas as pd
import pytest

from fie import backtest, db
from fie.inplay import InPlayModel
from fie.live import LiveMatch


@pytest.fixture()
def con(tmp_path):
    c = db.init(tmp_path / "test.db")
    c.execute("INSERT INTO live_sessions(live_id, mode, provider, status, created_at) VALUES(1,'replay','test','running',?)", (db.now(),))
    return c


def _model():
    m = InPlayModel()
    m.columns = m.feature_names()
    m.beta = np.zeros(len(m.columns))
    m.se = np.ones(len(m.columns))
    return m


def _live(con):
    meta = {"minutes_per_match": 94.0, "half_end": (47.0, 94.0)}
    return LiveMatch(con, 1, _model(), meta, 1.4, 1.1)


def _goal(eid, period, clock, side):
    return {"id": eid, "seq": int(clock * 100), "period": period, "clock": clock, "kind": "goal", "side": side, "payload": {}}


def test_duplicate_events_are_ignored(con):
    lm = _live(con)
    first = lm.ingest([_goal("g1", 1, 10.0, "home")])
    again = lm.ingest([_goal("g1", 1, 10.0, "home")])
    assert len(first) == 1 and len(again) == 0
    assert lm.state(2, 60.0)["home_goals"] == 1


def test_out_of_order_events_reconstruct_same_state(con):
    lm = _live(con)
    lm.ingest([_goal("g2", 2, 70.0, "away")])
    lm.ingest([_goal("g1", 1, 20.0, "home")])
    early = lm.state(1, 30.0)
    late = lm.state(2, 80.0)
    assert (early["home_goals"], early["away_goals"]) == (1, 0)
    assert (late["home_goals"], late["away_goals"]) == (1, 1)


def test_provider_correction_removes_goal(con):
    lm = _live(con)
    lm.ingest([_goal("g1", 2, 55.0, "home")])
    before = lm.snapshot(2, 56.0, "goal", ["goal"])
    lm.ingest([{"id": "c1", "period": 2, "clock": 57.0, "kind": "correction", "side": "home", "payload": {"supersedes": "g1"}}])
    after = lm.snapshot(2, 57.0, "correction", ["goal disallowed"])
    assert lm.state(2, 58.0)["home_goals"] == 0
    assert after["p_home"] < before["p_home"]


def test_snapshots_are_probability_consistent(con):
    lm = _live(con)
    fc = lm.snapshot(1, 0.0, "kickoff", [])
    assert abs(fc["p_home"] + fc["p_draw"] + fc["p_away"] - 1) < 1e-9
    lm.finished = True
    final = lm.snapshot(2, 94.0, "full_time", [])
    assert final["p_another_goal"] == 0.0
    assert final["p_draw"] == pytest.approx(1.0)


def test_predictions_are_immutable_and_forward_requires_scheduled(con):
    con.execute("INSERT INTO competitions VALUES('E0','PL','ENG',1)")
    con.execute("INSERT INTO teams(team_id, name, country) VALUES(1,'A','ENG'),(2,'B','ENG')")
    con.execute("INSERT INTO matches(match_id, competition, season, match_date, home_team_id, away_team_id, status, fthg, ftag, source, first_seen, updated_at) VALUES(1,'E0',2026,'2026-11-01',1,2,'finished',1,0,'t',?,?)", (db.now(), db.now()))
    con.execute("INSERT INTO matches(match_id, competition, season, match_date, home_team_id, away_team_id, status, source, first_seen, updated_at) VALUES(2,'E0',2026,'2026-11-09',2,1,'scheduled','t',?,?)", (db.now(), db.now()))
    with pytest.raises(sqlite3.IntegrityError):
        con.execute("INSERT INTO predictions(match_id, model_version, purpose, created_at, data_cutoff, p_home, p_draw, p_away) VALUES(1,'m','forward',?, '2026-10-30', .5,.3,.2)", (db.now(),))
    con.execute("INSERT INTO predictions(match_id, model_version, purpose, created_at, data_cutoff, p_home, p_draw, p_away) VALUES(2,'m','forward',?, '2026-11-02', .5,.3,.2)", (db.now(),))
    with pytest.raises(sqlite3.IntegrityError):
        con.execute("UPDATE predictions SET p_home=.9 WHERE match_id=2")


def test_walk_forward_training_data_precedes_predictions():
    rng = np.random.default_rng(3)
    rows = []
    teams = list(range(8))
    date = pd.Timestamp("2009-08-01")
    for season in (2009, 2010):
        for h in teams:
            for a in teams:
                if h != a:
                    rows.append({"match_id": len(rows) + 1, "date": date, "season": season, "home": h, "away": a, "hg": rng.poisson(1.5), "ag": rng.poisson(1.1),
                                 "tier": 1, "country": "ENG", "competition": "E0", "status": "finished", "home_name": str(h), "away_name": str(a)})
                    date += pd.Timedelta(days=2)
    frame = pd.DataFrame(rows)
    out = backtest.walk_forward_strength(frame, "dc", start=pd.Timestamp("2010-01-01"), xi=0.002, ridge=1.0, dixon_coles=True, window_days=1000)
    merged = out.merge(frame[["match_id", "date"]], on="match_id")
    assert (merged["fit_cutoff"] <= merged["date"]).all()
    assert len(merged) > 0


class _Resp:
    def __init__(self, payload, headers=None):
        self._payload = payload
        self.headers = headers or {}

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_live_provider_reports_missing_key(con, monkeypatch):
    from fie import live
    monkeypatch.delenv("FOOTBALL_DATA_API_KEY", raising=False)
    assert live.poll_live_provider(con)["status"] == "not_configured"
    row = con.execute("SELECT status, last_error FROM provider_health WHERE provider='football-data.org live'").fetchone()
    assert row["status"] == "not_configured" and "not set" in row["last_error"]


def test_live_provider_tracks_and_updates_matches(con, monkeypatch):
    from fie import live
    monkeypatch.setenv("FOOTBALL_DATA_API_KEY", "test-key")
    payload = {"matches": [{"id": 1, "competition": {"code": "PL"}, "homeTeam": {"name": "Arsenal FC"}, "awayTeam": {"name": "Chelsea FC"},
                            "minute": 30, "score": {"fullTime": {"home": 1, "away": 0}}},
                           {"id": 2, "competition": {"code": "XYZ"}, "homeTeam": {"name": "A"}, "awayTeam": {"name": "B"}, "score": {"fullTime": {}}}]}
    monkeypatch.setattr(live.requests, "get", lambda *a, **k: _Resp(payload, {"X-Requests-Available-Minute": "9"}))
    assert live.poll_live_provider(con)["tracked"] == ["fdo-1"]
    payload["matches"][0]["score"]["fullTime"] = {"home": 1, "away": 1}
    payload["matches"][0]["minute"] = 70
    live.poll_live_provider(con)
    rows = con.execute("SELECT * FROM live_sessions WHERE provider='football-data.org'").fetchall()
    assert len(rows) == 1 and (rows[0]["home_goals"], rows[0]["away_goals"], rows[0]["minute"]) == (1, 1, 70)
    assert con.execute("SELECT rate_limit FROM provider_health WHERE provider='football-data.org live'").fetchone()[0].startswith("9")



def test_live_provider_produces_forecasts_from_score_changes(con, monkeypatch):
    from fie import live
    monkeypatch.setenv("FOOTBALL_DATA_API_KEY", "test-key")
    con.execute("INSERT INTO competitions VALUES('E0','PL','ENG',1)")
    con.execute("INSERT INTO teams(team_id, name, country) VALUES(1,'Arsenal','ENG'),(2,'Chelsea','ENG')")
    con.execute("INSERT INTO matches(match_id, competition, season, match_date, home_team_id, away_team_id, status, source, first_seen, updated_at) VALUES(9,'E0',2026,'2026-11-07',1,2,'scheduled','t',?,?)", (db.now(), db.now()))
    con.execute("INSERT INTO predictions(match_id, model_version, purpose, created_at, data_cutoff, p_home, p_draw, p_away, lambda_home, lambda_away) VALUES(9,'m','forward',?,'2026-11-01',.5,.25,.25,1.6,1.0)", (db.now(),))
    model = _model()
    params = {"minutes_per_match": 94.0, "half_end": (47.0, 94.0)}
    payload = {"matches": [{"id": 7, "utcDate": "2026-11-07T15:00:00Z", "status": "IN_PLAY", "competition": {"code": "PL"}, "homeTeam": {"name": "Arsenal FC"},
                            "awayTeam": {"name": "Chelsea FC"}, "minute": 20, "score": {"fullTime": {"home": 0, "away": 0}}}]}
    monkeypatch.setattr(live.requests, "get", lambda *a, **k: _Resp(payload))
    live.poll_live_provider(con, model, params)
    s0 = con.execute("SELECT * FROM live_snapshots ORDER BY snapshot_id DESC LIMIT 1").fetchone()
    assert s0 is not None
    payload["matches"][0].update({"minute": 55, "score": {"fullTime": {"home": 1, "away": 0}}})
    live.poll_live_provider(con, model, params)
    s1 = con.execute("SELECT * FROM live_snapshots ORDER BY snapshot_id DESC LIMIT 1").fetchone()
    assert s1["trigger"] == "goal" and s1["home_goals"] == 1 and s1["p_home"] > s0["p_home"]
    payload["matches"][0].update({"minute": 57, "score": {"fullTime": {"home": 0, "away": 0}}})
    live.poll_live_provider(con, model, params)
    s2 = con.execute("SELECT * FROM live_snapshots ORDER BY snapshot_id DESC LIMIT 1").fetchone()
    assert s2["home_goals"] == 0 and s2["trigger"] == "correction"
    payload["matches"] = []
    live.poll_live_provider(con, model, params)
    assert con.execute("SELECT status FROM live_sessions WHERE provider='football-data.org'").fetchone()[0] == "ended_or_unavailable"



def test_repeated_meetings_and_rescheduled_fixtures(con):
    from fie import ingest_fdco
    ingest_fdco.ensure_reference(con)
    rows = pd.DataFrame([
        {"Date": "01/09/2020", "HomeTeam": "Celtic", "AwayTeam": "Rangers", "FTHG": 1, "FTAG": 0},
        {"Date": "04/09/2020", "HomeTeam": "Celtic", "AwayTeam": "Rangers", "FTHG": 2, "FTAG": 2},
    ])
    counts = ingest_fdco.upsert_rows(con, rows, "SC0", 2020)
    assert counts["inserted"] == 2 and counts["corrected"] == 0
    fixture = pd.DataFrame([{"Date": "10/01/2021", "HomeTeam": "Celtic", "AwayTeam": "Rangers"}])
    ingest_fdco.upsert_rows(con, fixture, "SC0", 2020, scheduled=True)
    played_late = pd.DataFrame([{"Date": "17/01/2021", "HomeTeam": "Celtic", "AwayTeam": "Rangers", "FTHG": 0, "FTAG": 1}])
    counts = ingest_fdco.upsert_rows(con, played_late, "SC0", 2020)
    assert counts["updated"] == 1 and counts["inserted"] == 0
    stored = con.execute("SELECT match_date, status, fthg, ftag FROM matches WHERE competition='SC0' ORDER BY match_date").fetchall()
    assert [tuple(r) for r in stored] == [("2020-09-01", "finished", 1, 0), ("2020-09-04", "finished", 2, 2), ("2021-01-17", "finished", 0, 1)]



def test_forward_predictions_require_future_kickoff(con):
    from datetime import datetime, timedelta, timezone
    from fie import forward
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    soon = now + timedelta(hours=1)
    later = now + timedelta(days=2)
    assert not forward.safely_before_kickoff(soon.strftime("%Y-%m-%d"), soon.strftime("%H:%M"))
    assert forward.safely_before_kickoff(later.strftime("%Y-%m-%d"), "15:00")
    con.execute("INSERT INTO competitions VALUES('E0','PL','ENG',1)")
    con.execute("INSERT INTO teams(team_id, name, country) VALUES(1,'A','ENG'),(2,'B','ENG')")
    con.execute("INSERT INTO matches(match_id, competition, season, match_date, kickoff, home_team_id, away_team_id, status, source, first_seen, updated_at) VALUES(5,'E0',2026,?,?,1,2,'scheduled','t',?,?)",
                (soon.strftime("%Y-%m-%d"), soon.strftime("%H:%M"), db.now(), db.now()))
    con.execute("INSERT INTO predictions(match_id, model_version, purpose, created_at, data_cutoff, p_home, p_draw, p_away) VALUES(5,'m','forward',?,'2026-01-01',.4,.3,.3)", (db.now(),))
    assert forward.flag_late_forward_predictions(con) == 1


def _scored_matches(con, stated, true, n=1200, seed=5):
    rng = np.random.default_rng(seed)
    con.execute("INSERT INTO competitions VALUES('E0','PL','ENG',1)")
    con.executemany("INSERT INTO teams(team_id, name, country) VALUES(?,?,'ENG')", [(i, f"T{i}") for i in range(1, 21)])
    for i in range(n):
        h, a = rng.choice(np.arange(1, 21), 2, replace=False)
        y = rng.choice(3, p=true)
        hg, ag = (2, 0) if y == 0 else (1, 1) if y == 1 else (0, 2)
        date = (pd.Timestamp("2024-08-01") + pd.Timedelta(days=i // 10)).strftime("%Y-%m-%d")
        con.execute("INSERT INTO matches(match_id, competition, season, match_date, home_team_id, away_team_id, status, fthg, ftag, source, first_seen, updated_at) VALUES(?,?,?,?,?,?,'finished',?,?,'t',?,?)",
                    (i + 1, "E0", 2024, date, int(h), int(a), hg, ag, db.now(), db.now()))
        con.execute("INSERT INTO predictions(match_id, model_version, purpose, created_at, data_cutoff, p_home, p_draw, p_away, lambda_home, lambda_away) VALUES(?,'m','backtest',?,?,?,?,?,1.4,1.1)",
                    (i + 1, db.now(), date, *stated))


def test_weakness_report_flags_overconfidence_but_not_calibrated_models(con, tmp_path):
    from fie import forward
    _scored_matches(con, (0.45, 0.27, 0.28), (0.45, 0.27, 0.28))
    calibrated = forward.weakness_report(con)
    assert calibrated["n"] == 1200
    assert not [f["key"] for f in calibrated["findings"] if f["flagged"] and f["key"].startswith(("rate_", "fav_"))]
    other = db.init(tmp_path / "over.db")
    _scored_matches(other, (0.75, 0.15, 0.10), (0.50, 0.25, 0.25))
    flagged = {f["key"] for f in forward.weakness_report(other)["findings"] if f["flagged"]}
    assert {"rate_home", "fav_home_70"} <= flagged


def test_kickoff_guard_uses_the_leagues_utc_offset():
    from datetime import datetime, timedelta, timezone
    from fie import forward
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    istanbul_local = now + timedelta(hours=4)
    assert not forward.safely_before_kickoff(istanbul_local.strftime("%Y-%m-%d"), istanbul_local.strftime("%H:%M"), "T1")
    london_local = now + timedelta(hours=5)
    assert forward.safely_before_kickoff(london_local.strftime("%Y-%m-%d"), london_local.strftime("%H:%M"), "E0")


def test_weakness_report_scores_each_match_once_preferring_the_pre_match_forecast(con):
    from fie import forward
    con.execute("INSERT INTO competitions VALUES('E0','PL','ENG',1)")
    con.execute("INSERT INTO teams(team_id, name, country) VALUES(1,'A','ENG'),(2,'B','ENG')")
    con.execute("INSERT INTO matches(match_id, competition, season, match_date, home_team_id, away_team_id, status, source, first_seen, updated_at) VALUES(1,'E0',2026,'2026-11-01',1,2,'scheduled','t',?,?)", (db.now(), db.now()))
    con.execute("INSERT INTO predictions(match_id, model_version, purpose, created_at, data_cutoff, p_home, p_draw, p_away) VALUES(1,'m','forward',?,'2026-10-30',.5,.3,.2)", (db.now(),))
    con.execute("UPDATE matches SET status='finished', fthg=1, ftag=0 WHERE match_id=1")
    con.execute("INSERT INTO predictions(match_id, model_version, purpose, created_at, data_cutoff, p_home, p_draw, p_away) VALUES(1,'m','backtest',?,'2026-10-30',.4,.3,.3)", (db.now(),))
    report = forward.weakness_report(con)
    assert report["n"] == 1 and report["sources"] == {"forward": 1}


def test_feed_normalisation_handles_duplicates_order_corrections_and_bad_events(con):
    from fie import live
    session = live.simulated_session(con, "A", "B")
    first = live.ingest_feed(con, session, [
        {"id": "e2", "kind": "goal", "side": "away", "period": 2, "clock": 70, "seq": 2},
        {"id": "e1", "kind": "goal", "side": "home", "period": 1, "clock": 12, "seq": 1},
        {"id": "e1", "kind": "goal", "side": "home", "period": 1, "clock": 12, "seq": 1},
        {"id": "e3", "kind": "own_goal", "side": "home", "period": 2, "clock": 80},
        {"id": "bad1", "kind": "teleport", "side": "home"},
        {"id": "bad2", "kind": "goal", "side": "middle", "clock": 10},
    ])
    assert first["accepted"] == 3 and first["duplicates"] == 1 and len(first["rejected"]) == 2
    assert first["score"] == [2, 1]
    second = live.ingest_feed(con, session, [{"id": "c1", "kind": "correction", "period": 2, "clock": 81, "payload": {"supersedes": "e3"}}])
    assert second["score"] == [1, 1]
    row = con.execute("SELECT mode, home_goals, away_goals, minute FROM live_sessions WHERE live_id=?", (session,)).fetchone()
    assert tuple(row) == ("simulated", 1, 1, 81.0)
