import sqlite3

import pandas as pd
import pytest

from fie import db, odds, paper


@pytest.fixture
def con(tmp_path):
    c = db.init(tmp_path / "test.db")
    c.execute("INSERT INTO competitions VALUES('E0','PL','ENG',1)")
    c.execute("INSERT INTO teams(team_id, name, country) VALUES(1,'A','ENG'),(2,'B','ENG'),(3,'C','ENG'),(4,'D','ENG')")
    for mid, home, away in ((1, 1, 2), (2, 3, 4)):
        c.execute("INSERT INTO matches(match_id, competition, season, match_date, kickoff, home_team_id, away_team_id, status, source, first_seen, updated_at) "
                  "VALUES(?,'E0',2099,'2099-01-10','15:00',?,?,'scheduled','t',?,?)", (mid, home, away, db.now(), db.now()))
        c.execute("INSERT INTO predictions(match_id, model_version, purpose, created_at, data_cutoff, p_home, p_draw, p_away, lambda_home, lambda_away, markets) "
                  "VALUES(?,'m','forward',?,'2099-01-01',.6,.22,.18,2.0,0.8,'{\"rho\": -0.05}')", (mid, db.now()))
    return c


def _prices(con, match_id, home, draw, away, observed="2099-01-09T12:00:00.000000Z", book="bet365"):
    for sel, price in (("home", home), ("draw", draw), ("away", away)):
        con.execute("INSERT INTO market_odds VALUES(?,?,?,?,?,?,?,?,?)", (match_id, "1x2", 0.0, sel, book, "pre_close", price, observed, "test"))


def test_value_price_is_selected_and_fair_price_is_not(con):
    _prices(con, 1, 2.6, 4.5, 7.0)
    _prices(con, 2, 1.55, 4.0, 3.9)
    assert paper.record(con) == {"bet": 1, "no_bet": 1, "skipped": 0}
    rows = {r["match_id"]: r for r in con.execute("SELECT * FROM paper_decisions")}
    assert rows[1]["decision"] == "bet" and rows[1]["selection"] == "home" and rows[1]["odds"] == 2.6 and rows[1]["stake"] == 1.0
    assert rows[1]["odds_observed_at"] == "2099-01-09T12:00:00.000000Z"
    assert rows[2]["decision"] == "no_bet" and rows[2]["stake"] == 0.0 and "below" in rows[2]["reason"]
    assert paper.record(con) == {"bet": 0, "no_bet": 0, "skipped": 0}


def test_decisions_are_immutable(con):
    _prices(con, 1, 2.6, 4.5, 7.0)
    paper.record(con)
    with pytest.raises(sqlite3.DatabaseError):
        con.execute("UPDATE paper_decisions SET odds = 3.0")
    with pytest.raises(sqlite3.DatabaseError):
        con.execute("DELETE FROM paper_decisions")


def test_daily_exposure_limit(con):
    _prices(con, 1, 2.6, 4.5, 7.0)
    _prices(con, 2, 2.6, 4.5, 7.0)
    made = paper.record(con, {**paper.RULE, "max_daily_units": 1.0})
    assert made == {"bet": 1, "no_bet": 1, "skipped": 0}
    reasons = [r[0] for r in con.execute("SELECT reason FROM paper_decisions WHERE decision = 'no_bet'")]
    assert reasons == ["daily exposure limit reached"]


def test_historical_and_closing_prices_are_never_used(con):
    _prices(con, 1, 2.6, 4.5, 7.0, observed=odds.HISTORICAL_OBSERVED)
    for sel, price in (("home", 2.6), ("draw", 4.5), ("away", 7.0)):
        con.execute("INSERT INTO market_odds VALUES(1,'1x2',0.0,?,'bet365','close',?,'2099-01-10T15:00:00Z','test')", (sel, price))
    paper.record(con)
    row = con.execute("SELECT decision, reason FROM paper_decisions WHERE match_id = 1").fetchone()
    assert row["decision"] == "no_bet" and "no observed pre-match price" in row["reason"]


def test_settlement_and_summary(con):
    _prices(con, 1, 2.6, 4.5, 7.0)
    paper.record(con)
    assert paper.settle_open(con) == 0
    con.execute("UPDATE matches SET status = 'finished', fthg = 2, ftag = 1 WHERE match_id = 1")
    assert paper.settle_open(con) == 1
    assert paper.settle_open(con) == 0
    s = paper.summary(con)
    assert s["settled"] == 1 and s["open"] == 0
    assert s["profit"] == pytest.approx(1.6) and s["roi"] == pytest.approx(1.6)


def test_row_prices_parses_books_timings_and_lines():
    row = pd.Series({"PSH": 2.1, "PSD": 3.4, "PSA": 3.6, "PSCH": 2.0, "PSCD": 3.5, "PSCA": 3.9, "B365H": "x", "B365D": 3.3, "B365A": 3.5,
                     "P>2.5": 1.9, "P<2.5": 1.95, "AHh": -0.25, "PAHH": 1.92, "PAHA": 1.96, "AHCh": 0.3, "PCAHH": 1.9, "PCAHA": 1.9})
    out = odds.row_prices(row)
    keys = {(m, line, sel, book, timing) for m, line, sel, book, timing, _ in out}
    assert ("1x2", 0.0, "home", "pinnacle", "pre_close") in keys and ("1x2", 0.0, "away", "pinnacle", "close") in keys
    assert not any(book == "bet365" for _, _, _, book, _, _ in out)
    assert ("ou", 2.5, "under", "pinnacle", "pre_close") in keys
    assert ("ah", -0.25, "home", "pinnacle", "pre_close") in keys
    assert not any(m == "ah" and timing == "close" for m, _, _, _, timing, _ in out)


def test_decision_records_the_scheduled_date(con):
    _prices(con, 1, 2.6, 4.5, 7.0)
    paper.record(con)
    assert tuple(con.execute("SELECT match_date, kickoff FROM paper_decisions WHERE match_id = 1").fetchone()) == ("2099-01-10", "15:00")


def test_rescheduled_match_is_void(con):
    _prices(con, 1, 2.6, 4.5, 7.0)
    paper.record(con)
    con.execute("UPDATE matches SET match_date = '2099-01-25', status = 'finished', fthg = 2, ftag = 0 WHERE match_id = 1")
    assert paper.settle_open(con) == 1
    row = con.execute("SELECT outcome, profit FROM paper_settlements").fetchone()
    assert tuple(row) == ("void", 0.0)
    s = paper.summary(con)
    assert s["void"] == 1 and s["settled"] == 0 and s["open"] == 0 and s["profit"] == 0.0


def test_match_without_result_is_void_only_after_grace_period(con):
    _prices(con, 1, 2.6, 4.5, 7.0)
    paper.record(con)
    assert paper.settle_open(con, today="2099-01-20") == 0
    assert paper.settle_open(con, today="2099-01-30") == 1
    assert con.execute("SELECT outcome FROM paper_settlements").fetchone()[0] == "void"


def test_settlements_are_immutable_and_report_clv_and_drawdown(con):
    _prices(con, 1, 2.6, 4.5, 7.0)
    _prices(con, 2, 2.6, 4.5, 7.0)
    paper.record(con)
    con.execute("INSERT INTO market_odds VALUES(1,'1x2',0.0,'home','bet365','close',2.4,'documented: closing prices collected at kick-off','test')")
    con.execute("UPDATE matches SET status = 'finished', fthg = 2, ftag = 1 WHERE match_id = 1")
    con.execute("UPDATE matches SET status = 'finished', fthg = 0, ftag = 1 WHERE match_id = 2")
    assert paper.settle_open(con) == 2
    with pytest.raises(sqlite3.DatabaseError):
        con.execute("UPDATE paper_settlements SET profit = 5")
    s = paper.summary(con)
    assert s["profit"] == pytest.approx(0.6) and s["max_drawdown"] == pytest.approx(1.0)
    assert s["clv_n"] == 1 and s["clv_mean"] == pytest.approx(2.6 / 2.4 - 1)


def test_legacy_tables_are_migrated(tmp_path):
    c = sqlite3.connect(tmp_path / "old.db")
    c.executescript("""CREATE TABLE matches(match_id INTEGER PRIMARY KEY, match_date TEXT, kickoff TEXT, updated_at TEXT);
        INSERT INTO matches VALUES(1, '2099-01-10', '15:00', '2099-01-01T00:00:00Z');
        CREATE TABLE paper_decisions (decision_id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, match_id INTEGER NOT NULL, prediction_id INTEGER NOT NULL,
            rule TEXT NOT NULL, decision TEXT NOT NULL, reason TEXT NOT NULL, market TEXT, line REAL, selection TEXT, bookmaker TEXT,
            odds REAL, odds_observed_at TEXT, model_prob REAL, market_prob REAL, ev REAL, stake REAL, UNIQUE(match_id, rule));
        CREATE TABLE paper_settlements (decision_id INTEGER PRIMARY KEY, settled_at TEXT NOT NULL, home_goals INTEGER NOT NULL,
            away_goals INTEGER NOT NULL, profit REAL NOT NULL);
        INSERT INTO paper_decisions(decision_id, created_at, match_id, prediction_id, rule, decision, reason, stake)
            VALUES(1, '2099-01-02T00:00:00Z', 1, 1, 'r', 'bet', 'x', 1.0);
        INSERT INTO paper_settlements VALUES(1, '2099-01-11', 1, 0, 0.9);""")
    c.row_factory = sqlite3.Row
    c.isolation_level = None
    paper.ensure(c)
    assert tuple(c.execute("SELECT match_date, kickoff FROM paper_decisions").fetchone()) == ("2099-01-10", "15:00")
    assert tuple(c.execute("SELECT outcome, profit FROM paper_settlements").fetchone()) == ("settled", 0.9)
    with pytest.raises(sqlite3.DatabaseError):
        c.execute("UPDATE paper_decisions SET odds = 2")
