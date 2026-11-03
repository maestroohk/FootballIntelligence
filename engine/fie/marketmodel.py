import numpy as np
import pandas as pd

from . import db, metrics, research
from .config import PRIMARY_COMPETITIONS, VALIDATION_SEASONS
from .jobs import handler

FOOTBALL = ["elo_diff", "h_rest", "a_rest", "h_m28", "a_m28", "h_promoted", "a_promoted", "h_xf", "h_xa", "a_xf", "a_xa", "h_n", "a_n"]


def market_probs(con, book="pinnacle"):
    q = pd.read_sql_query("""SELECT match_id, selection, odds FROM market_odds
                             WHERE market = '1x2' AND timing = 'pre_close' AND bookmaker = ? AND observed_at NOT LIKE '20%'""", con, params=(book,))
    w = q.pivot_table(index="match_id", columns="selection", values="odds").dropna()[["home", "draw", "away"]]
    p = metrics.devig(w["home"], w["draw"], w["away"])
    return pd.DataFrame({"match_id": w.index, "m_home": p[:, 0], "m_draw": p[:, 1], "m_away": p[:, 2]})


def dataset(con):
    oos = research.load_cached("oos_ensemble")
    feats = research.load_cached("features")[["match_id"] + FOOTBALL]
    d = oos[oos["status"] == "finished"].merge(market_probs(con), on="match_id").merge(feats, on="match_id", how="left")
    d = d.sort_values(["date", "match_id"]).reset_index(drop=True)
    eps = 1e-6
    d["mk_hd"] = np.log((d["m_home"] + eps) / (d["m_away"] + eps))
    d["mk_dd"] = np.log((d["m_draw"] + eps) / (d["m_away"] + eps))
    d["ch_hd"] = np.log((d["p_home"] + eps) / (d["p_away"] + eps))
    d["ch_dd"] = np.log((d["p_draw"] + eps) / (d["p_away"] + eps))
    d["dis_hd"] = d["ch_hd"] - d["mk_hd"]
    d["dis_dd"] = d["ch_dd"] - d["mk_dd"]
    d["rest_diff"] = (d["h_rest"].clip(upper=21) - d["a_rest"].clip(upper=21))
    d["m28_diff"] = d["h_m28"] - d["a_m28"]
    d["xg_form"] = (d["h_xf"] - d["h_xa"]) - (d["a_xf"] - d["a_xa"])
    return d


SETS = {
    "market_recalibrated": ["mk_hd", "mk_dd"],
    "market_plus_champion": ["mk_hd", "mk_dd", "dis_hd", "dis_dd"],
    "market_plus_football": ["mk_hd", "mk_dd", "dis_hd", "dis_dd", "elo_diff", "rest_diff", "m28_diff", "h_promoted", "a_promoted", "xg_form"],
}


def walk_forward(d, cols, first_season=2013, lookback=4, C=1.0):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    out = np.full((len(d), 3), np.nan)
    for season in sorted(d.loc[d["season"] >= first_season, "season"].unique()):
        tr = (d["season"] < season) & (d["season"] >= season - lookback)
        te = (d["season"] == season).to_numpy()
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=C, max_iter=1000))
        model.fit(d.loc[tr, cols], metrics.outcome_index(d.loc[tr, "hg"], d.loc[tr, "ag"]))
        out[te] = model.predict_proba(d.loc[te, cols])
    return out


def _cmp(la, lb, block):
    return metrics.paired_bootstrap(la, lb, block=block)


def study(con):
    d = dataset(con)
    preds = {"market_raw": d[["m_home", "m_draw", "m_away"]].to_numpy(), "champion": d[["p_home", "p_draw", "p_away"]].to_numpy()}
    for name, cols in SETS.items():
        preds[name] = walk_forward(d, cols)
    y = metrics.outcome_index(d["hg"], d["ag"])
    block = (d["competition"] + d["season"].astype(str)).to_numpy()
    out = {"book": "pinnacle pre-closing (collection time not published)", "periods": {}}
    for period, seasons in (("validation_2013_2022", [s for s in VALIDATION_SEASONS if s >= 2013]), ("confirmation_2023_on", [2023, 2024, 2025, 2026])):
        res = {}
        for scope, comps in (("primary", PRIMARY_COMPETITIONS), ("all", None)):
            k = d["season"].isin(seasons).to_numpy() & (np.ones(len(d), bool) if comps is None else d["competition"].isin(comps).to_numpy())
            k &= ~np.isnan(preds["market_recalibrated"]).any(axis=1)
            losses = {n: metrics.log_loss_rows(p[k], y[k]) for n, p in preds.items()}
            res[scope] = {"n": int(k.sum()), "log_loss": {n: float(v.mean()) for n, v in losses.items()},
                          "recalibrated_vs_raw": _cmp(losses["market_raw"], losses["market_recalibrated"], block[k]),
                          "champion_info_vs_recalibrated": _cmp(losses["market_recalibrated"], losses["market_plus_champion"], block[k]),
                          "football_info_vs_recalibrated": _cmp(losses["market_recalibrated"], losses["market_plus_football"], block[k]),
                          "football_info_vs_champion_info": _cmp(losses["market_plus_champion"], losses["market_plus_football"], block[k])}
        out["periods"][period] = res
    v = out["periods"]["validation_2013_2022"]["primary"]
    out["passed_validation"] = {k: v[k]["ci_low"] > 0 for k in ("champion_info_vs_recalibrated", "football_info_vs_recalibrated")}
    db.set_kv(con, "market_model_study", out)
    return out


STAKES = ["h_dead", "a_dead", "h_danger", "a_danger", "h_chasing", "a_chasing"]


def stakes_study(con):
    from .features import standings_features
    d = dataset(con).merge(standings_features(research.load_cached("frame")), on="match_id", how="left")
    d = d[d["h_dead"].notna()].reset_index(drop=True)
    base = walk_forward(d, ["ch_hd", "ch_dd"])
    cand = walk_forward(d, ["ch_hd", "ch_dd"] + STAKES)
    y = metrics.outcome_index(d["hg"], d["ag"])
    block = (d["competition"] + d["season"].astype(str)).to_numpy()
    raw = d[["p_home", "p_draw", "p_away"]].to_numpy()
    mkt = d[["m_home", "m_draw", "m_away"]].to_numpy()
    out = {"leagues": "single round-robin leagues (split and play-off leagues excluded)", "periods": {}}
    for period, seasons in (("validation_2013_2022", [s for s in VALIDATION_SEASONS if s >= 2013]), ("confirmation_2023_on", [2023, 2024, 2025, 2026])):
        res = {}
        for scope in ("all", "late_season"):
            k = d["season"].isin(seasons).to_numpy() & ~np.isnan(cand).any(axis=1)
            if scope == "late_season":
                k &= (d["season_left"] <= 0.2).to_numpy()
            lb, lc = metrics.log_loss_rows(base[k], y[k]), metrics.log_loss_rows(cand[k], y[k])
            lr, lm = metrics.log_loss_rows(raw[k], y[k]), metrics.log_loss_rows(mkt[k], y[k])
            res[scope] = {"n": int(k.sum()), "champion": float(lr.mean()), "champion_recalibrated": float(lb.mean()), "with_stakes": float(lc.mean()),
                          "market": float(lm.mean()), "stakes_vs_recalibrated": _cmp(lb, lc, block[k]),
                          "gap_to_market_before": float((lb - lm).mean()), "gap_to_market_after": float((lc - lm).mean())}
        out["periods"][period] = res
    out["passed_validation"] = out["periods"]["validation_2013_2022"]["all"]["stakes_vs_recalibrated"]["ci_low"] > 0
    db.set_kv(con, "stakes_study", out)
    return out


@handler("stakes_study")
def stakes_study_job(ctx):
    ctx.stage("testing end-of-season stakes features against the recalibrated champion")
    res = stakes_study(ctx.con)
    return {p: {s: {k: v for k, v in r.items() if k != "stakes_vs_recalibrated"} | {"gain": r["stakes_vs_recalibrated"]} for s, r in v.items()} for p, v in res["periods"].items()}


@handler("market_model_study")
def market_model_study(ctx):
    ctx.stage("fitting market-informed challengers walk-forward")
    res = study(ctx.con)
    return {p: {s: r["log_loss"] for s, r in v.items()} for p, v in res["periods"].items()}
