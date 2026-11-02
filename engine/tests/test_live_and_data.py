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
