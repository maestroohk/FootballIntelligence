import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import backtest, db, metrics
from .config import CORE_COMPETITIONS, DATA, THREADS
from .features import FEATURE_COLUMNS, apply_xg_proxy, fit_xg_proxy, load_matches, team_form_features
from .scoreline import dixon_coles_matrix

CACHE = Path(os.environ.get("FIE_CACHE", DATA / "cache"))

STRENGTH_DEFAULTS = {"xi": 0.0019, "ridge": 1.0, "window_days": 1100}


def cache_path(name):
    CACHE.mkdir(parents=True, exist_ok=True)
    return CACHE / f"{name}.pkl"


def load_cached(name):
    p = cache_path(name)
    return pd.read_pickle(p) if p.exists() else None


def save_cached(name, frame):
    frame.to_pickle(cache_path(name))


def prepared_frame(con, include_scheduled=True, refit_proxy=False):
    frame = load_matches(con, include_scheduled=include_scheduled)
    coef = None if refit_proxy else db.get_kv(con, "xg_proxy")
    if coef is None:
        coef = fit_xg_proxy(frame[(frame["season"] < 2010) & (frame["status"] == "finished") & (frame["competition"].isin(CORE_COMPETITIONS))])
    frame = apply_xg_proxy(frame, coef)
    return frame, coef


def run_strength(frame, name, progress=None, **params):
    out = backtest.walk_forward_strength(frame, name, progress=progress, **{**STRENGTH_DEFAULTS, **params})
    return backtest.attach(out, frame)


def run_elo(frame, **params):
    out, ratings = backtest.walk_forward_elo(frame, **params)
    return backtest.attach(out, frame), ratings


def run_frequency(frame):
    return backtest.attach(backtest.walk_forward_frequency(frame), frame)


def feature_table(frame, elo_ratings, dc_oos, home_adv=65.0):
    f = team_form_features(frame)
    f = f.merge(elo_ratings, on="match_id", how="left")
    f["elo_diff"] = f["elo_h"] - f["elo_a"] + home_adv
    f = f.merge(dc_oos[["match_id", "lh", "la"]].rename(columns={"lh": "dc_lh", "la": "dc_la"}), on="match_id", how="left")
    f["dc_log_lh"] = np.log(f["dc_lh"])
    f["dc_log_la"] = np.log(f["dc_la"])
    return f


def run_gbm(features, dc_oos, name="gbm", use_dc=True, first_season=2010, params=None, train_competitions=CORE_COMPETITIONS):
    import lightgbm as lgb
    cols = FEATURE_COLUMNS + (["dc_log_lh", "dc_log_la"] if use_dc else [])
    params = {"objective": "poisson", "learning_rate": 0.03, "num_leaves": 15, "min_data_in_leaf": 200, "feature_fraction": 0.8,
              "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0, "verbose": -1, "num_threads": THREADS, **(params or {})}
    rho = dc_oos.set_index("match_id")["rho"]
    outs = []
    seasons = sorted(features.loc[features["season"] >= first_season, "season"].unique())
    for season in seasons:
        train = features[(features["season"] < season) & (features["season"] >= 2007) & (features["status"] == "finished") & (features["competition"].isin(train_competitions))].dropna(subset=["dc_lh"])
        target = features[features["season"] == season].dropna(subset=["dc_lh"])
        preds = {}
        contrib = {}
        sched = (target["status"] == "scheduled").to_numpy()
        for side, ycol in (("lh", "hg"), ("la", "ag")):
            ds = lgb.Dataset(train[cols], train[ycol])
            booster = lgb.train(params, ds, num_boost_round=400)
            preds[side] = booster.predict(target[cols])
            contrib[side] = [None] * len(target)
            if sched.any():
                pc = booster.predict(target.loc[sched, cols], pred_contrib=True)
                idx = np.where(sched)[0]
                for k, row in zip(idx, pc):
                    top = sorted(zip(cols, row[:-1]), key=lambda t: -abs(t[1]))[:6]
                    contrib[side][k] = db.dumps([{"feature": f, "value": float(v)} for f, v in top])
        o = pd.DataFrame({"match_id": target["match_id"].to_numpy(), "lh": preds["lh"], "la": preds["la"],
                          "rho": rho.reindex(target["match_id"]).to_numpy(), "both_known": True, "fit_cutoff": pd.Timestamp(f"{season}-07-01"),
                          "contrib_h": contrib["lh"], "contrib_a": contrib["la"]})
        outs.append(backtest.finalise(o, features, name))
    return backtest.attach(pd.concat(outs, ignore_index=True), features)


def run_gbm_importance(features, use_dc=True, season=2022):
    import lightgbm as lgb
    cols = FEATURE_COLUMNS + (["dc_log_lh", "dc_log_la"] if use_dc else [])
    train = features[(features["season"] < season) & (features["season"] >= 2007) & (features["status"] == "finished")].dropna(subset=["dc_lh"])
    params = {"objective": "poisson", "learning_rate": 0.03, "num_leaves": 15, "min_data_in_leaf": 200, "verbose": -1, "num_threads": THREADS}
    booster = lgb.train(params, lgb.Dataset(train[cols], train["hg"]), num_boost_round=400)
    gain = booster.feature_importance("gain")
    return sorted(({"feature": c, "gain_share": float(g / gain.sum())} for c, g in zip(cols, gain)), key=lambda r: -r["gain_share"])


def run_logistic(features, name="logistic", first_season=2010):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    cols = FEATURE_COLUMNS
    outs = []
    for season in sorted(features.loc[features["season"] >= first_season, "season"].unique()):
        train = features[(features["season"] < season) & (features["season"] >= 2007) & (features["status"] == "finished") & (features["competition"].isin(CORE_COMPETITIONS))]
        target = features[features["season"] == season]
        y = metrics.outcome_index(train["hg"], train["ag"])
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.1, max_iter=500))
        model.fit(train[cols], y)
        p = model.predict_proba(target[cols])
        o = pd.DataFrame({"match_id": target["match_id"].to_numpy(), "p_home": p[:, 0], "p_draw": p[:, 1], "p_away": p[:, 2],
                          "lh": np.nan, "la": np.nan, "rho": np.nan, "score_p": np.nan, "over25": np.nan, "model": name})
        outs.append(o)
    return backtest.attach(pd.concat(outs, ignore_index=True), features)


def _score_nll(lh, la, rho, hg, ag):
    total = 0.0
    for a, b, h, g in zip(lh, la, hg, ag):
        m = dixon_coles_matrix(a, b, rho, 10)
        total -= np.log(max(m[min(int(h), 10), min(int(g), 10)], 1e-12))
    return total


def fit_ensemble_weights(components, frame_rows, spread=False):
    logs_h = np.column_stack([np.log(c["lh"].to_numpy()) for c in components])
    logs_a = np.column_stack([np.log(c["la"].to_numpy()) for c in components])
    hg = frame_rows["hg"].to_numpy()
    ag = frame_rows["ag"].to_numpy()
    k = logs_h.shape[1]
    from scipy.stats import poisson

    def rates(theta):
        w = np.exp(theta[:k]) / np.exp(theta[:k]).sum()
        scale = theta[k]
        rho = np.tanh(theta[k + 1]) * 0.3
        s = np.exp(theta[k + 2]) if spread else 1.0
        eh, ea = logs_h @ w, logs_a @ w
        mid, half = (eh + ea) / 2, (eh - ea) / 2
        return np.exp(mid + s * half + scale), np.exp(mid - s * half + scale), rho, w, scale, s

    def objective(theta):
        lh, la, rho, *_ = rates(theta)
        ll = poisson.logpmf(hg, lh) + poisson.logpmf(ag, la)
        tau = np.ones_like(lh)
        tau = np.where((hg == 0) & (ag == 0), 1 - lh * la * rho, tau)
        tau = np.where((hg == 0) & (ag == 1), 1 + lh * rho, tau)
        tau = np.where((hg == 1) & (ag == 0), 1 + la * rho, tau)
        tau = np.where((hg == 1) & (ag == 1), 1 - rho, tau)
        return -(ll + np.log(np.clip(tau, 1e-9, None))).sum()

    dim = k + 2 + (1 if spread else 0)
    rng = np.random.default_rng(20261102)
    starts = [np.zeros(dim)] + [np.concatenate([rng.normal(0, 1.5, k), np.zeros(dim - k)]) for _ in range(5)]
    res = min((minimize(objective, x0, method="L-BFGS-B") for x0 in starts), key=lambda r: r.fun)
    _, _, rho, w, scale, s = rates(res.x)
    return {"weights": w.tolist(), "scale": float(scale), "rho": float(rho), "spread": float(s)}


def combine_rates(keyed, ids, params):
    eh = sum(w * np.log(c.loc[ids, "lh"].to_numpy()) for w, c in zip(params["weights"], keyed))
    ea = sum(w * np.log(c.loc[ids, "la"].to_numpy()) for w, c in zip(params["weights"], keyed))
    s = params.get("spread", 1.0)
    mid, half = (eh + ea) / 2, (eh - ea) / 2
    return np.exp(mid + s * half + params["scale"]), np.exp(mid - s * half + params["scale"])


def run_ensemble(component_oos, names, frame, name="ensemble", first_season=2012, lookback=3, spread=False):
    keyed = [c.set_index("match_id") for c in component_oos]
    common = keyed[0].index
    for c in keyed[1:]:
        common = common.intersection(c.index)
    common = common.sort_values()
    base = frame.set_index("match_id").loc[common].reset_index().sort_values(["date", "match_id"]).reset_index(drop=True)
    outs = []
    fitted = {}
    for season in sorted(base.loc[base["season"] >= first_season, "season"].unique()):
        train = base[(base["season"] < season) & (base["season"] >= season - lookback) & (base["status"] == "finished") & (base["competition"].isin(CORE_COMPETITIONS))]
        target = base[base["season"] == season]
        comps_train = [c.loc[train["match_id"]] for c in keyed]
        params = fit_ensemble_weights(comps_train, train, spread=spread)
        fitted[int(season)] = {"weights": dict(zip(names, params["weights"])), "scale": params["scale"], "rho": params["rho"], "spread": params["spread"]}
        lh, la = combine_rates(keyed, target["match_id"], params)
        o = pd.DataFrame({"match_id": target["match_id"].to_numpy(), "lh": lh, "la": la, "rho": params["rho"], "both_known": True,
                          "fit_cutoff": keyed[0].loc[target["match_id"], "fit_cutoff"].to_numpy()})
        outs.append(backtest.finalise(o, frame, name))
    return backtest.attach(pd.concat(outs, ignore_index=True), frame), fitted


def bookmaker_oos(con, frame):
    odds = pd.read_sql_query("SELECT * FROM odds WHERE timing='close' AND bookmaker IN ('pinnacle','market_avg')", con)
    odds["rank"] = odds["bookmaker"].map({"pinnacle": 0, "market_avg": 1})
    odds = odds.sort_values("rank").drop_duplicates("match_id")
    p = metrics.devig(odds["home"], odds["draw"], odds["away"], "power")
    o = pd.DataFrame({"match_id": odds["match_id"].to_numpy(), "p_home": p[:, 0], "p_draw": p[:, 1], "p_away": p[:, 2],
                      "lh": np.nan, "la": np.nan, "rho": np.nan, "score_p": np.nan, "over25": np.nan, "model": "bookmaker_close",
                      "bookmaker": odds["bookmaker"].to_numpy()})
    with np.errstate(divide="ignore", invalid="ignore"):
        ou = odds[["over25", "under25"]].to_numpy(dtype=float)
        inv = 1 / ou
        o["over25"] = inv[:, 0] / inv.sum(axis=1)
    return backtest.attach(o, frame)


def record_experiment(con, name, hypothesis, result, decision, details=None):
    con.execute("INSERT INTO experiments(created_at, name, hypothesis, result, decision, details) VALUES(?,?,?,?,?,?)",
                (db.now(), name, hypothesis, result, decision, db.dumps(details or {})))


def store_backtest(con, model_version, oos, split, competitions, train_window, notes=None):
    rep = backtest.full_report(oos, split, competitions)
    if rep is None:
        return None
    con.execute(
        "INSERT INTO backtest_runs(model_version, split, created_at, train_window, test_period, n_matches, metrics, by_league, by_season, reliability, notes) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        (model_version, split, db.now(), train_window, f"{rep['dates'][0]} to {rep['dates'][1]}", rep["overall"]["n"], db.dumps(rep["overall"]),
         db.dumps(rep["by_league"]), db.dumps(rep["by_season"]),
         db.dumps({"outcomes": rep["reliability"], "over25": rep["reliability_over25"], "by_confidence": rep["by_confidence"], "by_favourite": rep["by_favourite"]}),
         notes),
    )
    return rep
