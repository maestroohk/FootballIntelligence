import numpy as np
import pandas as pd

from . import db, metrics, research
from .jobs import handler
from .scoreline import dixon_coles_matrix
from .value import devig, settle

DEVELOPMENT = list(range(2012, 2020))
HOLDOUT = [2020, 2021, 2022]
FINAL = [2023, 2024, 2025, 2026]
BOOKS = ["pinnacle", "bet365", "market_avg"]
MIN_EV = [0.0, 0.02, 0.05, 0.10]
ODDS_BANDS = [(1.01, 2.5), (2.5, 6.0), (1.01, 6.0)]
MIN_BETS = 300


def _period(season):
    return "development" if season in DEVELOPMENT else "holdout" if season in HOLDOUT else "final" if season in FINAL else None


def build_candidates(con, model="ensemble"):
    oos = research.load_cached(f"oos_{model}")
    oos = oos[(oos["status"] == "finished") & (oos["season"] >= 2012)].dropna(subset=["lh"])
    prices = pd.read_sql_query("SELECT match_id, market, line, selection, bookmaker, timing, odds FROM market_odds WHERE bookmaker IN ('pinnacle','bet365','market_avg','market_max')", con)
    pre = prices[prices["timing"] == "pre_close"]
    close = prices[prices["timing"] == "close"]
    ref_close = close[close["bookmaker"] == "pinnacle"].set_index(["match_id", "market", "line", "selection"])["odds"]
    avg_close = close[close["bookmaker"] == "market_avg"].set_index(["match_id", "market", "line", "selection"])["odds"]
    info = oos.set_index("match_id")
    rows = []
    for match_id, g in pre[pre["match_id"].isin(info.index)].groupby("match_id"):
        r = info.loc[match_id]
        m = dixon_coles_matrix(r["lh"], r["la"], r["rho"])
        m = m / m.sum()
        hg, ag = int(r["hg"]), int(r["ag"])
        for (book, market, line), sel_rows in g.groupby(["bookmaker", "market", "line"]):
            quotes = dict(zip(sel_rows["selection"], sel_rows["odds"]))
            sels = {"1x2": ("home", "draw", "away"), "ou": ("over", "under"), "ah": ("home", "away")}[market]
            if any(s not in quotes for s in sels):
                continue
            fair = devig([quotes[s] for s in sels])
            for s, mp in zip(sels, fair):
                odds = quotes[s]
                ev = sum(m[h, a] * settle(market, line, s, odds, h, a) for h in range(m.shape[0]) for a in range(m.shape[1]))
                key = (match_id, market, line, s)
                closing = ref_close.get(key, avg_close.get(key, np.nan))
                rows.append((match_id, r["season"], r["competition"], r["date"], book, market, line, s, odds, float(mp), float(ev),
                             settle(market, line, s, odds, hg, ag), closing, 1.0 / sum(1.0 / quotes[x] for x in sels) - 1.0))
    cand = pd.DataFrame(rows, columns=["match_id", "season", "competition", "date", "bookmaker", "market", "line", "selection", "odds",
                                       "market_prob", "ev", "profit", "closing_odds", "payout_minus_one"])
    cand["period"] = cand["season"].map(_period)
    research.save_cached("strategy_candidates", cand)
    return cand


def pick(cand, market, book, min_ev, lo, hi):
    c = cand[(cand["market"] == market) & (cand["bookmaker"] == book) & (cand["odds"] >= lo) & (cand["odds"] <= hi) & (cand["ev"] >= min_ev)]
    return c.sort_values("ev", ascending=False).drop_duplicates(["match_id", "market"])


def summarise(sel, n_boot=1000, seed=3):
    if sel.empty:
        return {"n": 0}
    sel = sel.sort_values(["date", "match_id"])
    profit = sel["profit"].to_numpy()
    cum = np.cumsum(profit)
    drawdown = float((np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:] - cum).max())
    streak = best = 0
    for p in profit:
        streak = streak + 1 if p < 0 else 0
        best = max(best, streak)
    week = sel["date"].dt.to_period("W").astype(str).to_numpy()
    weeks = np.unique(week)
    by = {w: profit[week == w] for w in weeks}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pickw = rng.choice(weeks, len(weeks))
        s = np.concatenate([by[w] for w in pickw])
        boots.append(s.sum() / len(s))
    clv = (sel["odds"] / sel["closing_odds"] - 1).dropna()
    won = (sel["profit"] > 0).mean()
    return {"n": int(len(sel)), "staked": float(len(sel)), "profit": float(profit.sum()), "roi": float(profit.mean()),
            "roi_ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], "max_drawdown": drawdown,
            "longest_losing_streak": int(best), "avg_odds": float(sel["odds"].mean()), "mean_model_ev": float(sel["ev"].mean()),
            "win_rate": float(won), "clv_mean": float(clv.mean()) if len(clv) else None, "clv_beat_share": float((clv > 0).mean()) if len(clv) else None,
            "clv_n": int(len(clv)), "seasons": sorted(int(s) for s in sel["season"].unique())}


def equity_curve(sel, points=120):
    sel = sel.sort_values(["date", "match_id"])
    cum = np.cumsum(sel["profit"].to_numpy())
    if len(cum) == 0:
        return []
    idx = np.unique(np.linspace(0, len(cum) - 1, min(points, len(cum))).astype(int))
    return [{"date": str(sel["date"].iloc[i].date()), "units": float(cum[i])} for i in idx]


def run(con):
    cand = research.load_cached("strategy_candidates")
    if cand is None:
        cand = build_candidates(con)
    results = {"periods": {"development": DEVELOPMENT, "holdout": HOLDOUT, "final": FINAL}, "rules": [], "baselines": {}}
    for market in ("1x2", "ou", "ah"):
        for book in BOOKS:
            sub = cand[(cand["market"] == market) & (cand["bookmaker"] == book)]
            if sub.empty:
                continue
            dev = sub[sub["period"] == "development"]
            grid = []
            for min_ev in MIN_EV:
                for lo, hi in ODDS_BANDS:
                    s = summarise(pick(dev, market, book, min_ev, lo, hi), n_boot=200)
                    if s["n"] >= MIN_BETS:
                        grid.append(((min_ev, lo, hi), s))
            if not grid:
                results["rules"].append({"market": market, "bookmaker": book, "selected": None, "reason": "fewer than 300 development bets for every rule (prices exist only from 2019)"})
                continue
            (min_ev, lo, hi), dev_s = max(grid, key=lambda g: g[1]["roi"])
            hold = pick(sub[sub["period"] == "holdout"], market, book, min_ev, lo, hi)
            final = pick(sub[sub["period"] == "final"], market, book, min_ev, lo, hi)
            results["rules"].append({"market": market, "bookmaker": book, "selected": {"min_ev": min_ev, "odds_min": lo, "odds_max": hi},
                                     "development": dev_s, "holdout": summarise(hold), "final": summarise(final),
                                     "grid": [{"min_ev": g[0][0], "odds_min": g[0][1], "odds_max": g[0][2], "n": g[1]["n"], "roi": g[1]["roi"]} for g in grid],
                                     "equity_holdout_final": equity_curve(pd.concat([hold, final]))})
        for book in BOOKS:
            sub = cand[(cand["market"] == market) & (cand["bookmaker"] == book) & (cand["period"].isin(["holdout", "final"]))]
            if sub.empty:
                continue
            fav = sub.sort_values("market_prob", ascending=False).drop_duplicates(["match_id", "market"])
            results["baselines"][f"{market}/{book}/every selection"] = summarise(sub, n_boot=200)
            results["baselines"][f"{market}/{book}/market favourite"] = summarise(fav, n_boot=200)
    m = cand[(cand["market"] == "1x2") & (cand["bookmaker"] == "pinnacle")]
    results["probability_quality"] = probability_quality(con, m)
    db.set_kv(con, "strategy_research", results)
    return results


def probability_quality(con, pinnacle_1x2):
    oos = research.load_cached("oos_ensemble").set_index("match_id")
    out = {}
    for period in ("development", "holdout", "final"):
        ids = pinnacle_1x2.loc[pinnacle_1x2["period"] == period, "match_id"].unique()
        if len(ids) == 0:
            continue
        mk = pinnacle_1x2[pinnacle_1x2["match_id"].isin(ids)].pivot_table(index="match_id", columns="selection", values="market_prob")[["home", "draw", "away"]]
        o = oos.loc[mk.index]
        y = metrics.outcome_index(o["hg"], o["ag"])
        pm = o[["p_home", "p_draw", "p_away"]].to_numpy()
        pk = mk.to_numpy()
        out[period] = {"n": int(len(mk)), "model_log_loss": float(metrics.log_loss_rows(pm, y).mean()), "pinnacle_open_log_loss": float(metrics.log_loss_rows(pk, y).mean()),
                       "model_brier": float(metrics.brier_rows(pm, y).mean()), "pinnacle_open_brier": float(metrics.brier_rows(pk, y).mean())}
    return out


ODDS_BREAKS = [1.01, 1.5, 2.0, 2.5, 3.5, 5.0, 8.0, 1000.0]


def _band_rows(c):
    c = c.assign(band=pd.cut(c["odds"], ODDS_BREAKS, right=False).astype(str), won=(c["profit"] > 0).astype(float))
    return [{"band": b, "n": int(len(g)), "market_prob": float(g["market_prob"].mean()), "win_rate": float(g["won"].mean()),
             "roi": float(g["profit"].mean())} for b, g in c.groupby("band", sort=False) if len(g) >= 200]


def market_efficiency(con):
    cand = research.load_cached("strategy_candidates")
    if cand is None:
        cand = build_candidates(con)
    late = cand[cand["period"].isin(["holdout", "final"])]
    x12 = late[late["market"] == "1x2"].sort_values("odds")
    out = {"scope": "seasons 2020/21 onwards, every 1X2 selection backed at 1 unit (descriptive, not a strategy)", "bookmakers": {}, "longshot": {}, "draws": {}, "leagues": [], "band_rules": []}
    for book, g in x12.groupby("bookmaker"):
        per = g.drop_duplicates("match_id")
        out["bookmakers"][book] = {"matches": int(len(per)), "mean_margin": float((-per["payout_minus_one"] / (1 + per["payout_minus_one"])).mean()),
                                   "roi_all": float(g["profit"].mean())}
        out["longshot"][book] = _band_rows(g)
        d = g[g["selection"] == "draw"].assign(bucket=lambda t: pd.cut(t["market_prob"], [0, .2, .24, .27, .3, .33, 1]).astype(str), won=lambda t: (t["profit"] > 0).astype(float))
        out["draws"][book] = [{"band": b, "n": int(len(h)), "market_prob": float(h["market_prob"].mean()), "rate": float(h["won"].mean())}
                              for b, h in d.groupby("bucket", sort=False) if len(h) >= 200]
    pin = x12[x12["bookmaker"] == "pinnacle"]
    for comp, g in pin.groupby("competition"):
        mx = x12[(x12["bookmaker"] == "market_max") & (x12["competition"] == comp)]
        won = g[g["profit"] > 0]
        out["leagues"].append({"competition": comp, "matches": int(g["match_id"].nunique()),
                               "margin": float((-g["payout_minus_one"] / (1 + g["payout_minus_one"])).mean()),
                               "log_loss": float(-np.log(won["market_prob"]).mean()), "roi_pinnacle": float(g["profit"].mean()),
                               "roi_best_price": float(mx["profit"].mean()) if len(mx) else None})
    for book in ("pinnacle", "market_max"):
        sub = cand[(cand["market"] == "1x2") & (cand["bookmaker"] == book)]
        dev = sub[sub["period"] == "development"]
        bands = [(lo, hi) for lo, hi in zip(ODDS_BREAKS[:-1], ODDS_BREAKS[1:])]
        grid = [((lo, hi), summarise(dev[(dev["odds"] >= lo) & (dev["odds"] < hi)], n_boot=200)) for lo, hi in bands]
        grid = [g for g in grid if g[1]["n"] >= MIN_BETS]
        if not grid:
            continue
        (lo, hi), dev_s = max(grid, key=lambda g: g[1]["roi"])
        pick_ = lambda per: sub[(sub["period"] == per) & (sub["odds"] >= lo) & (sub["odds"] < hi)]
        out["band_rules"].append({"bookmaker": book, "odds_min": lo, "odds_max": hi, "development": dev_s,
                                  "holdout": summarise(pick_("holdout")), "final": summarise(pick_("final"))})
    out["opening_vs_closing"] = opening_vs_closing(con)
    db.set_kv(con, "market_efficiency", out)
    return out


def opening_vs_closing(con):
    q = pd.read_sql_query("""SELECT o.match_id, o.timing, o.selection, o.odds, m.fthg, m.ftag, m.season FROM market_odds o JOIN matches m ON m.match_id = o.match_id
                             WHERE o.market = '1x2' AND o.bookmaker = 'pinnacle' AND m.status = 'finished' AND m.season >= 2012 AND o.observed_at NOT LIKE '20%'""", con)
    w = q.pivot_table(index=["match_id", "season", "fthg", "ftag"], columns=["timing", "selection"], values="odds").dropna().reset_index()
    y = metrics.outcome_index(w["fthg"], w["ftag"])
    out = {"n": int(len(w))}
    probs = {}
    for t in ("pre_close", "close"):
        p = metrics.devig(w[(t, "home")], w[(t, "draw")], w[(t, "away")])
        probs[t] = p
        out[f"{t}_log_loss"] = float(metrics.log_loss_rows(p, y).mean())
    moved = probs["close"][np.arange(len(y)), y] - probs["pre_close"][np.arange(len(y)), y]
    out["share_moved_towards_result"] = float((moved > 0.005).mean())
    out["share_moved_against_result"] = float((moved < -0.005).mean())
    seasons = w["season"].to_numpy()
    out["by_season"] = [{"season": int(s), "n": int((seasons == s).sum()),
                         "pre_close": float(metrics.log_loss_rows(probs["pre_close"][seasons == s], y[seasons == s]).mean()),
                         "close": float(metrics.log_loss_rows(probs["close"][seasons == s], y[seasons == s]).mean())} for s in sorted(set(seasons))]
    return out


@handler("market_efficiency")
def market_efficiency_job(ctx):
    ctx.stage("summarising market efficiency by bookmaker, odds band and league")
    out = market_efficiency(ctx.con)
    return {"bookmakers": out["bookmakers"], "band_rules": [{k: r[k] for k in ("bookmaker", "odds_min", "odds_max")} | {p: r[p].get("roi") for p in ("development", "holdout", "final")} for r in out["band_rules"]]}


@handler("strategy_research")
def strategy_research(ctx, rebuild=False):
    if rebuild:
        ctx.stage("building candidate selections from walk-forward forecasts and pre-closing prices")
        build_candidates(ctx.con)
    ctx.stage("evaluating pre-registered selection rules")
    res = run(ctx.con)
    return {"rules": [{k: (v if k in ("market", "bookmaker", "selected") else {kk: vv for kk, vv in v.items() if kk in ("n", "roi")} if isinstance(v, dict) else v) for k, v in r.items() if k in ("market", "bookmaker", "selected", "holdout", "final")} for r in res["rules"]]}
