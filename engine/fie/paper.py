import os

import numpy as np
import pandas as pd

from . import db
from .forward import safely_before_kickoff
from .jobs import handler
from .value import SELECTIONS, devig, evaluate, expected_return, outcome_distribution, settle
from .scoreline import dixon_coles_matrix

RULE = {"name": "research-ev5", "min_ev": 0.05, "odds_min": 1.5, "odds_max": 4.0, "stake": 1.0,
        "max_daily_units": float(os.environ.get("FIE_PAPER_MAX_DAILY_UNITS", "10")), "bookmakers": ["bet365", "market_avg"],
        "validated": False}
BANKROLL = float(os.environ.get("FIE_PAPER_BANKROLL", "100"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_decisions (
    decision_id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    match_id INTEGER NOT NULL REFERENCES matches(match_id),
    prediction_id INTEGER NOT NULL REFERENCES predictions(prediction_id),
    rule TEXT NOT NULL,
    decision TEXT NOT NULL,
    reason TEXT NOT NULL,
    market TEXT, line REAL, selection TEXT, bookmaker TEXT,
    odds REAL, odds_observed_at TEXT, model_prob REAL, market_prob REAL, ev REAL, stake REAL,
    match_date TEXT, kickoff TEXT,
    UNIQUE(match_id, rule)
);
CREATE TABLE IF NOT EXISTS paper_settlements (
    decision_id INTEGER PRIMARY KEY REFERENCES paper_decisions(decision_id),
    settled_at TEXT NOT NULL,
    home_goals INTEGER,
    away_goals INTEGER,
    profit REAL NOT NULL,
    outcome TEXT NOT NULL DEFAULT 'settled'
);
CREATE TRIGGER IF NOT EXISTS paper_settlements_immutable BEFORE UPDATE ON paper_settlements
BEGIN SELECT RAISE(ABORT, 'paper settlements are immutable'); END;
CREATE TRIGGER IF NOT EXISTS paper_decisions_immutable BEFORE UPDATE ON paper_decisions
BEGIN SELECT RAISE(ABORT, 'paper decisions are immutable'); END;
CREATE TRIGGER IF NOT EXISTS paper_decisions_no_delete BEFORE DELETE ON paper_decisions
BEGIN SELECT RAISE(ABORT, 'paper decisions are immutable'); END;
"""


def _schema(con):
    for statement in SCHEMA.split(";\n"):
        if statement.strip():
            con.execute(statement)


def ensure(con):
    cols = {r[1] for r in con.execute("PRAGMA table_info(paper_decisions)")}
    scols = {r[1] for r in con.execute("PRAGMA table_info(paper_settlements)")}
    with db.transaction(con):
        if cols and "match_date" not in cols:
            con.execute("DROP TRIGGER IF EXISTS paper_decisions_immutable")
            con.execute("ALTER TABLE paper_decisions ADD COLUMN match_date TEXT")
            con.execute("ALTER TABLE paper_decisions ADD COLUMN kickoff TEXT")
            con.execute("""UPDATE paper_decisions SET match_date = m.match_date, kickoff = m.kickoff FROM matches m
                           WHERE m.match_id = paper_decisions.match_id AND m.updated_at <= paper_decisions.created_at""")
        if scols and "outcome" not in scols:
            con.execute("DROP TRIGGER IF EXISTS paper_settlements_immutable")
            con.execute("ALTER TABLE paper_settlements RENAME TO paper_settlements_old")
        _schema(con)
        if scols and "outcome" not in scols:
            con.execute("""INSERT INTO paper_settlements(decision_id, settled_at, home_goals, away_goals, profit)
                           SELECT decision_id, settled_at, home_goals, away_goals, profit FROM paper_settlements_old""")
            con.execute("DROP TABLE paper_settlements_old")


def record(con, rule=RULE):
    ensure(con)
    rows = con.execute("""
        SELECT p.prediction_id, p.match_id, p.lambda_home, p.lambda_away, p.markets, m.match_date, m.kickoff, m.competition
        FROM predictions p JOIN matches m ON m.match_id = p.match_id
        WHERE p.purpose = 'forward' AND m.status = 'scheduled' AND p.lambda_home IS NOT NULL
          AND p.prediction_id NOT IN (SELECT prediction_id FROM prediction_flags)
          AND p.prediction_id = (SELECT MAX(p2.prediction_id) FROM predictions p2 WHERE p2.match_id = p.match_id AND p2.purpose = 'forward'
                                 AND p2.prediction_id NOT IN (SELECT prediction_id FROM prediction_flags))
          AND p.match_id NOT IN (SELECT match_id FROM paper_decisions WHERE rule = ?)""", (rule["name"],)).fetchall()
    import json
    made = {"bet": 0, "no_bet": 0, "skipped": 0}
    used = {}
    for r in sorted(rows, key=lambda x: (x["match_date"], x["kickoff"] or "")):
        if not safely_before_kickoff(r["match_date"], r["kickoff"], r["competition"]):
            made["skipped"] += 1
            continue
        prices = con.execute("""SELECT market, line, selection, bookmaker, odds, observed_at FROM market_odds
                                WHERE match_id = ? AND timing = 'pre_close' AND observed_at LIKE '20%'
                                  AND observed_at = (SELECT MAX(o2.observed_at) FROM market_odds o2 WHERE o2.match_id = market_odds.match_id
                                                     AND o2.bookmaker = market_odds.bookmaker AND o2.market = market_odds.market AND o2.timing = 'pre_close')""",
                             (r["match_id"],)).fetchall()
        rho = json.loads(r["markets"] or "{}").get("rho", 0.0) or 0.0
        m = dixon_coles_matrix(r["lambda_home"], r["lambda_away"], rho)
        m = m / m.sum()
        best = None
        quotes = {}
        for q in prices:
            if q["bookmaker"] in rule["bookmakers"]:
                quotes.setdefault((q["bookmaker"], q["market"], q["line"]), {})[q["selection"]] = (q["odds"], q["observed_at"])
        for (book, market, line), sel in quotes.items():
            names = SELECTIONS[market]
            if any(n not in sel for n in names):
                continue
            fair = devig([sel[n][0] for n in names])
            for n, f in zip(names, fair):
                odds, observed = sel[n]
                if not rule["odds_min"] <= odds <= rule["odds_max"]:
                    continue
                ev = expected_return(m, market, line, n, odds)
                if best is None or ev > best["ev"]:
                    best = {"market": market, "line": line, "selection": n, "bookmaker": book, "odds": odds, "observed": observed,
                            "model_prob": outcome_distribution(m, market, line, n)["win"], "market_prob": float(f), "ev": ev}
        day = r["match_date"]
        if best is None:
            decision, reason = "no_bet", "no observed pre-match price in the rule's bookmakers and odds range"
        elif best["ev"] < rule["min_ev"]:
            decision, reason = "no_bet", f"best model expected value {best['ev']:+.3f} is below the {rule['min_ev']:+.2f} threshold"
        elif used.get(day, 0) + rule["stake"] > rule["max_daily_units"]:
            decision, reason = "no_bet", "daily exposure limit reached"
        else:
            decision, reason = "bet", f"model expected value {best['ev']:+.3f} at {best['odds']:.2f}"
            used[day] = used.get(day, 0) + rule["stake"]
        b = best or {}
        con.execute("""INSERT INTO paper_decisions(created_at, match_id, prediction_id, rule, decision, reason, market, line, selection, bookmaker,
                       odds, odds_observed_at, model_prob, market_prob, ev, stake, match_date, kickoff) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (db.now(), r["match_id"], r["prediction_id"], rule["name"], decision, reason, b.get("market"), b.get("line"), b.get("selection"),
                     b.get("bookmaker"), b.get("odds"), b.get("observed"), b.get("model_prob"), b.get("market_prob"), b.get("ev"),
                     rule["stake"] if decision == "bet" else 0.0, r["match_date"], r["kickoff"]))
        made[decision] += 1
    return made


MOVED_DAYS = 2
MISSING_DAYS = 14

CLOSING = "COALESCE(" + ", ".join(
    f"""(SELECT o.odds FROM market_odds o WHERE o.match_id = d.match_id AND o.market = d.market AND o.line = d.line
         AND o.selection = d.selection AND o.timing = 'close' AND o.bookmaker = {book})""" for book in ("d.bookmaker", "'pinnacle'", "'market_avg'")) + ")"


def settle_open(con, today=None):
    ensure(con)
    today = pd.Timestamp(today) if today else pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    rows = con.execute("""SELECT d.*, m.fthg, m.ftag, m.status, m.match_date AS played_date FROM paper_decisions d JOIN matches m ON m.match_id = d.match_id
                          WHERE d.decision = 'bet' AND d.decision_id NOT IN (SELECT decision_id FROM paper_settlements)""").fetchall()
    n = 0
    for r in rows:
        planned = pd.Timestamp(r["match_date"] or r["played_date"])
        moved = abs((pd.Timestamp(r["played_date"]) - planned).days) > MOVED_DAYS
        missing = r["status"] != "finished" and (today - planned).days > MISSING_DAYS
        if moved or missing:
            con.execute("INSERT INTO paper_settlements(decision_id, settled_at, profit, outcome) VALUES(?,?,0,'void')", (r["decision_id"], db.now()))
        elif r["status"] == "finished" and r["fthg"] is not None:
            profit = settle(r["market"], r["line"], r["selection"], r["odds"], r["fthg"], r["ftag"]) * r["stake"]
            con.execute("INSERT INTO paper_settlements(decision_id, settled_at, home_goals, away_goals, profit) VALUES(?,?,?,?,?)",
                        (r["decision_id"], db.now(), r["fthg"], r["ftag"], profit))
        else:
            continue
        n += 1
    return n


def summary(con):
    ensure(con)
    d = pd.read_sql_query(f"""SELECT d.*, s.profit, s.settled_at, s.outcome, {CLOSING} AS closing_odds
                              FROM paper_decisions d LEFT JOIN paper_settlements s USING(decision_id)""", con)
    bets = d[d["decision"] == "bet"]
    done = bets.dropna(subset=["profit"])
    settled = done[done["outcome"] == "settled"].sort_values(["match_date", "decision_id"])
    cum = settled["profit"].cumsum().to_numpy()
    drawdown = float((np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:] - cum).max()) if len(cum) else 0.0
    clv = (settled["odds"] / settled["closing_odds"] - 1).dropna()
    return {"rule": RULE, "bankroll": BANKROLL, "decisions": int(len(d)), "bets": int(len(bets)), "no_bets": int((d["decision"] == "no_bet").sum()),
            "settled": int(len(settled)), "void": int((done["outcome"] == "void").sum()), "open": int(len(bets) - len(done)),
            "profit": float(settled["profit"].sum()), "roi": float(settled["profit"].sum() / settled["stake"].sum()) if len(settled) else None,
            "max_drawdown": drawdown, "clv_mean": float(clv.mean()) if len(clv) else None, "clv_n": int(len(clv))}


def board(con):
    import json
    rows = con.execute(f"""
        SELECT p.prediction_id, p.match_id, p.model_version, p.created_at, p.lambda_home, p.lambda_away, p.markets,
               m.match_date, m.kickoff, m.competition, {_name('m.home_team_id')} home, {_name('m.away_team_id')} away
        FROM predictions p JOIN matches m ON m.match_id = p.match_id
        WHERE p.purpose = 'forward' AND m.status = 'scheduled' AND p.lambda_home IS NOT NULL
          AND p.prediction_id NOT IN (SELECT prediction_id FROM prediction_flags)
          AND p.prediction_id = (SELECT MAX(p2.prediction_id) FROM predictions p2 WHERE p2.match_id = p.match_id AND p2.purpose = 'forward'
                                 AND p2.prediction_id NOT IN (SELECT prediction_id FROM prediction_flags))
          AND EXISTS (SELECT 1 FROM market_odds o WHERE o.match_id = p.match_id AND o.timing = 'pre_close' AND o.observed_at LIKE '20%')
        ORDER BY m.match_date, m.kickoff""").fetchall()
    matches = []
    for r in rows:
        rho = json.loads(r["markets"] or "{}").get("rho", 0.0) or 0.0
        quotes = {}
        for q in con.execute("""SELECT market, line, selection, bookmaker, odds, observed_at FROM market_odds o
                                WHERE match_id = ? AND timing = 'pre_close' AND observed_at = (SELECT MAX(o2.observed_at) FROM market_odds o2
                                  WHERE o2.match_id = o.match_id AND o2.bookmaker = o.bookmaker AND o2.market = o.market AND o2.timing = 'pre_close')""",
                             (r["match_id"],)):
            quotes.setdefault((q["market"], q["line"], q["bookmaker"]), {"observed_at": q["observed_at"], "prices": {}})["prices"][q["selection"]] = q["odds"]
        markets = []
        for (market, line, book), q in sorted(quotes.items()):
            if any(s not in q["prices"] for s in SELECTIONS[market]):
                continue
            ev = evaluate(r["lambda_home"], r["lambda_away"], rho, market, line, q["prices"])
            markets.append({"market": market, "line": line, "bookmaker": book, "observed_at": q["observed_at"], "margin": ev["margin"],
                            "selections": ev["selections"]})
        matches.append({"match_id": r["match_id"], "date": r["match_date"], "kickoff": r["kickoff"], "competition": r["competition"],
                        "home": _short(r["home"]), "away": _short(r["away"]), "model_version": r["model_version"], "predicted_at": r["created_at"],
                        "safely_before_kickoff": bool(safely_before_kickoff(r["match_date"], r["kickoff"], r["competition"])), "markets": markets})
    coverage = [dict(r) for r in con.execute("""SELECT bookmaker, timing, CASE WHEN observed_at LIKE '20%' THEN 'recorded snapshot' ELSE 'historical file' END kind,
                                                   COUNT(*) prices, COUNT(DISTINCT match_id) matches, MAX(observed_at) latest
                                                   FROM market_odds GROUP BY 1, 2, 3 ORDER BY 3 DESC, 4 DESC""")]
    out = {"generated_at": db.now(), "matches": matches, "coverage": coverage}
    db.set_kv(con, "value_board", out)
    return out


def _short(name):
    for pre in ("AFC ", "FC "):
        if name.startswith(pre):
            name = name[len(pre):]
    return name[:-3] if name.endswith(" FC") else name


def _name(col):
    return (f"COALESCE((SELECT a.alias FROM team_aliases a WHERE a.team_id = {col} AND a.source IN ('statsbomb', 'openfootball') "
            f"ORDER BY a.source = 'statsbomb' DESC LIMIT 1), (SELECT name FROM teams WHERE team_id = {col}))")


@handler("paper_cycle")
def paper_cycle(ctx):
    ctx.stage("settling finished paper selections")
    settled = settle_open(ctx.con)
    ctx.stage("recording paper decisions for upcoming fixtures")
    made = record(ctx.con)
    ctx.stage("evaluating upcoming prices against the model")
    b = board(ctx.con)
    return {"settled": settled, **made, "board_matches": len(b["matches"]), "summary": summary(ctx.con)}
