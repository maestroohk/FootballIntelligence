import numpy as np
import pandas as pd

from . import db, metrics, research
from .config import PRIMARY_COMPETITIONS, VALIDATION_SEASONS
from .jobs import handler

FOOTBALL = ["elo_diff", "h_rest", "a_rest", "h_m28", "a_m28", "h_promoted", "a_promoted", "h_xf", "h_xa", "a_xf", "a_xa", "h_n", "a_n"]


def market_probs(con, books=("pinnacle",)):
    out = []
    for book in books:
        q = pd.read_sql_query("""SELECT match_id, selection, odds FROM market_odds
                                 WHERE market = '1x2' AND timing = 'pre_close' AND bookmaker = ? AND observed_at NOT LIKE '20%'""", con, params=(book,))
        w = q.pivot_table(index="match_id", columns="selection", values="odds").dropna()[["home", "draw", "away"]]
        p = metrics.devig(w["home"], w["draw"], w["away"])
        out.append(pd.DataFrame({"match_id": w.index, "m_home": p[:, 0], "m_draw": p[:, 1], "m_away": p[:, 2], "label_book": book}))
    return pd.concat(out).drop_duplicates("match_id")


def dataset(con, books=("pinnacle",)):
    oos = research.load_cached("oos_ensemble")
    feats = research.load_cached("features")[["match_id"] + FOOTBALL]
    d = oos[oos["status"] == "finished"].merge(market_probs(con, books), on="match_id").merge(feats, on="match_id", how="left")
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
    out["data_version"] = research.data_version(con)
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
    out["data_version"] = research.data_version(con)
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


def _gbm(train_x, labels, weights, test_x):
    import lightgbm as lgb
    from .config import THREADS
    params = {"objective": "multiclass", "num_class": 3, "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 300,
              "feature_fraction": 0.8, "verbose": -1, "num_threads": THREADS, "seed": 11}
    booster = lgb.train(params, lgb.Dataset(train_x, labels, weight=weights), num_boost_round=300)
    return booster.predict(test_x)


def distillation_study(con, first_season=2014, lookback=4, books=("pinnacle",)):
    from .features import FEATURE_COLUMNS
    d = dataset(con, books)
    feats = research.load_cached("features")
    d = d.merge(feats[["match_id"] + [c for c in FEATURE_COLUMNS + ["dc_log_lh", "dc_log_la"] if c not in d.columns]], on="match_id", how="left")
    cols = list(dict.fromkeys(FEATURE_COLUMNS + ["dc_log_lh", "dc_log_la", "ch_hd", "ch_dd"]))
    y = metrics.outcome_index(d["hg"], d["ag"])
    soft = d[["m_home", "m_draw", "m_away"]].to_numpy()
    hard_p = np.full((len(d), 3), np.nan)
    soft_p = np.full((len(d), 3), np.nan)
    for season in sorted(d.loc[d["season"] >= first_season, "season"].unique()):
        tr = ((d["season"] < season) & (d["season"] >= season - lookback)).to_numpy()
        te = (d["season"] == season).to_numpy()
        x_tr, x_te = d.loc[tr, cols].to_numpy(float), d.loc[te, cols].to_numpy(float)
        hard_p[te] = _gbm(x_tr, y[tr], None, x_te)
        n = tr.sum()
        soft_p[te] = _gbm(np.vstack([x_tr] * 3), np.repeat([0, 1, 2], n), soft[tr].T.reshape(-1), x_te)
    keep = ~np.isnan(soft_p).any(axis=1)
    research.save_cached("oos_distilled" if books == ("pinnacle",) else "oos_distilled_" + "_".join(books), pd.DataFrame({"match_id": d.loc[keep, "match_id"].to_numpy(), "p_home": soft_p[keep, 0],
                                                        "p_draw": soft_p[keep, 1], "p_away": soft_p[keep, 2]}))
    block = (d["competition"] + d["season"].astype(str)).to_numpy()
    champ = d[["p_home", "p_draw", "p_away"]].to_numpy()
    out = {"features": cols, "periods": {}}
    for period, seasons in (("validation_2014_2022", [s for s in VALIDATION_SEASONS if s >= first_season]), ("confirmation_2023_on", [2023, 2024, 2025, 2026])):
        res = {}
        for scope, comps in (("primary", PRIMARY_COMPETITIONS), ("all", None)):
            k = d["season"].isin(seasons).to_numpy() & (np.ones(len(d), bool) if comps is None else d["competition"].isin(comps).to_numpy())
            l_c, l_h, l_s = (metrics.log_loss_rows(p[k], y[k]) for p in (champ, hard_p, soft_p))
            l_m = metrics.log_loss_rows(soft[k], y[k])
            res[scope] = {"n": int(k.sum()), "log_loss": {"champion": float(l_c.mean()), "gbm_outcome_labels": float(l_h.mean()),
                                                         "gbm_market_labels": float(l_s.mean()), "market": float(l_m.mean())},
                          "market_labels_vs_champion": _cmp(l_c, l_s, block[k]), "market_labels_vs_outcome_labels": _cmp(l_h, l_s, block[k])}
        out["periods"][period] = res
    out["passed_validation"] = out["periods"]["validation_2014_2022"]["primary"]["market_labels_vs_champion"]["ci_low"] > 0
    out["data_version"] = research.data_version(con, ("oos_ensemble", "features"))
    out["label_books"] = list(books)
    db.set_kv(con, "distillation_study" if books == ("pinnacle",) else "distillation_study_" + "_".join(books), out)
    return out


@handler("distillation_study")
def distillation_study_job(ctx, books=None):
    ctx.stage("training football-only models on outcome labels and on market labels")
    res = distillation_study(ctx.con, books=tuple(books) if books else ("pinnacle",))
    return {p: {s: r["log_loss"] | {"vs_champion": r["market_labels_vs_champion"]} for s, r in v.items()} for p, v in res["periods"].items()}


CHALLENGER = "market_hybrid@v2"
TEACHERS = ("pinnacle", "market_avg")


def challenger_forecasts(con, lookback=4):
    import json
    from datetime import date, timedelta
    from .features import FEATURE_COLUMNS
    from .forward import FORWARD_HORIZON_DAYS, safely_before_kickoff
    from .shapes import dixon_coles, solve_rates
    from .scoreline import markets
    d = dataset(con, TEACHERS)
    feats = research.load_cached("features")
    extra = [c for c in FEATURE_COLUMNS + ["dc_log_lh", "dc_log_la"] if c not in d.columns]
    d = d.merge(feats[["match_id"] + extra], on="match_id", how="left")
    cols = list(dict.fromkeys(FEATURE_COLUMNS + ["dc_log_lh", "dc_log_la", "ch_hd", "ch_dd"]))
    latest = int(d["season"].max())
    tr = d[d["season"] > latest - lookback]
    until = (date.today() + timedelta(days=FORWARD_HORIZON_DAYS)).isoformat()
    rows = con.execute("""SELECT p.prediction_id, p.match_id, p.data_cutoff, p.p_home, p.p_draw, p.p_away, p.lambda_home, p.lambda_away, p.markets,
                                 m.match_date, m.kickoff, m.competition
                          FROM predictions p JOIN matches m ON m.match_id = p.match_id
                          WHERE p.purpose = 'forward' AND m.status = 'scheduled' AND m.match_date BETWEEN date('now') AND ? AND p.lambda_home IS NOT NULL
                            AND p.prediction_id NOT IN (SELECT prediction_id FROM prediction_flags)
                            AND p.prediction_id = (SELECT MAX(p2.prediction_id) FROM predictions p2 WHERE p2.match_id = p.match_id AND p2.purpose = 'forward')
                            AND NOT EXISTS (SELECT 1 FROM predictions c WHERE c.match_id = p.match_id AND c.purpose = 'challenger' AND c.model_version = ?)""",
                       (until, CHALLENGER)).fetchall()
    rows = [r for r in rows if safely_before_kickoff(r["match_date"], r["kickoff"], r["competition"])]
    if not rows or tr.empty:
        return {"stored": 0, "training_rows": int(len(tr))}
    up = pd.DataFrame([dict(r) for r in rows]).merge(feats[["match_id"] + [c for c in cols if c in feats.columns]], on="match_id", how="left")
    eps = 1e-6
    up["ch_hd"] = np.log((up["p_home"] + eps) / (up["p_away"] + eps))
    up["ch_dd"] = np.log((up["p_draw"] + eps) / (up["p_away"] + eps))
    soft = tr[["m_home", "m_draw", "m_away"]].to_numpy()
    x_tr = tr[cols].to_numpy(float)
    n = len(tr)
    probs = _gbm(np.vstack([x_tr] * 3), np.repeat([0, 1, 2], n), soft.T.reshape(-1), up[cols].to_numpy(float))
    rho = np.array([json.loads(r["markets"] or "{}").get("rho", 0.0) or 0.0 for r in rows])
    lh0, la0 = up["lambda_home"].to_numpy(), up["lambda_away"].to_numpy()
    lh, la = solve_rates(probs, rho, lh0, la0, total=lh0 + la0)
    mats = dixon_coles(lh, la, rho)
    stored = 0
    with db.transaction(con):
        for i, r in enumerate(rows):
            mk = markets(mats[i])
            mk.update({"rho": float(rho[i]), "base_prediction_id": r["prediction_id"], "market_trained_1x2": [float(v) for v in probs[i]]})
            con.execute("""INSERT OR IGNORE INTO predictions(match_id, model_version, purpose, created_at, data_cutoff, p_home, p_draw, p_away, lambda_home, lambda_away, markets)
                           VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                        (r["match_id"], CHALLENGER, "challenger", db.now(), r["data_cutoff"], mk["p_home"], mk["p_draw"], mk["p_away"], float(lh[i]), float(la[i]), db.dumps(mk)))
            stored += 1
    return {"stored": stored, "training_rows": int(n), "training_seasons": sorted(int(s) for s in tr["season"].unique())}


@handler("challenger_forecasts")
def challenger_forecasts_job(ctx):
    ctx.stage("forecasting upcoming fixtures with the market-trained challenger")
    return challenger_forecasts(ctx.con)
