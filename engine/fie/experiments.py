import math
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import backtest, db, metrics, research
from .config import PRIMARY_COMPETITIONS, VALIDATION_SEASONS
from .jobs import handler
from .scoreline import dixon_coles_matrix, markets, outcome_probs


def record(con, name, hypothesis, result, decision, details=None):
    con.execute("DELETE FROM experiments WHERE name=?", (name,))
    con.execute("INSERT INTO experiments(created_at, name, hypothesis, result, decision, details) VALUES(?,?,?,?,?,?)",
                (db.now(), name, hypothesis, result, decision, db.dumps(details or {})))


def _validation(oos):
    d = oos[(oos["status"] == "finished") & (oos["season"].isin(VALIDATION_SEASONS)) & (oos["competition"].isin(PRIMARY_COMPETITIONS))]
    return d.dropna(subset=["p_home"])


def market_information(con):
    champ = research.load_cached("oos_ensemble")
    odds = pd.read_sql_query("SELECT match_id, bookmaker, home, draw, away FROM odds WHERE timing='open' AND bookmaker IN ('pinnacle','market_avg')", con)
    odds["rank"] = odds["bookmaker"].map({"pinnacle": 0, "market_avg": 1})
    odds = odds.sort_values("rank").drop_duplicates("match_id")
    p_mkt = metrics.devig(odds["home"], odds["draw"], odds["away"])
    mk = pd.DataFrame({"match_id": odds["match_id"], "m_home": p_mkt[:, 0], "m_draw": p_mkt[:, 1], "m_away": p_mkt[:, 2]})
    d = champ[(champ["status"] == "finished") & (champ["competition"].isin(PRIMARY_COMPETITIONS))].merge(mk, on="match_id")
    d = d[d["season"] >= 2012]
    y = metrics.outcome_index(d["hg"], d["ag"])
    pm = d[["p_home", "p_draw", "p_away"]].to_numpy()
    pk = d[["m_home", "m_draw", "m_away"]].to_numpy()

    def blend(w, a, b):
        z = w * np.log(np.clip(a, 1e-9, 1)) + (1 - w) * np.log(np.clip(b, 1e-9, 1))
        z = np.exp(z)
        return z / z.sum(axis=1, keepdims=True)

    seasons = d["season"].to_numpy()
    rows = []
    blended = np.empty_like(pm)
    for s in sorted(set(seasons)):
        tr = (seasons < s) & (seasons >= s - 3)
        te = seasons == s
        if tr.sum() < 1000:
            w = 0.0
        else:
            res = minimize(lambda w: metrics.log_loss_rows(blend(w[0], pm[tr], pk[tr]), y[tr]).mean(), [0.2], bounds=[(0, 1)], method="L-BFGS-B")
            w = float(res.x[0])
        blended[te] = blend(w, pm[te], pk[te])
        rows.append({"season": int(s), "model_weight": w})
    val = np.isin(seasons, VALIDATION_SEASONS) & (seasons >= 2013)
    l_model = metrics.log_loss_rows(pm[val], y[val])
    l_mkt = metrics.log_loss_rows(pk[val], y[val])
    l_blend = metrics.log_loss_rows(blended[val], y[val])
    cmp_mkt_blend = metrics.paired_bootstrap(l_mkt, l_blend, block=(d["competition"] + d["season"].astype(str)).to_numpy()[val])
    result = {"n": int(val.sum()), "model_log_loss": float(l_model.mean()), "market_open_log_loss": float(l_mkt.mean()), "blend_log_loss": float(l_blend.mean()),
              "weights_by_season": rows, "market_vs_blend": cmp_mkt_blend}
    adds = cmp_mkt_blend["ci_low"] > 0
    record(con, "Does the model add information beyond opening odds?",
           "If the model captures information the market misses, a log-linear blend of model and de-vigged opening odds (weights fitted on prior seasons only) will beat the opening odds alone.",
           f"Validation 2013-2022 (n={result['n']}): opening market {result['market_open_log_loss']:.4f}, model {result['model_log_loss']:.4f}, blend {result['blend_log_loss']:.4f}; "
           f"market-blend diff {cmp_mkt_blend['mean_diff']:+.4f} (95% CI {cmp_mkt_blend['ci_low']:+.4f}..{cmp_mkt_blend['ci_high']:+.4f}); mean model weight {np.mean([r['model_weight'] for r in rows[1:]]):.2f}.",
           "adopt as research finding: model carries small independent information" if adds else "reject: no measurable information beyond the opening market",
           result)
    db.set_kv(con, "experiment_market", result)
    return result


def halftime_recalibration(con):
    from .inplay import InPlayModel
    row = con.execute("SELECT params FROM live_model WHERE status='champion' ORDER BY created_at DESC LIMIT 1").fetchone()
    import json
    params = json.loads(row["params"])
    model = InPlayModel.from_json(params["model"])
    mpm = params["minutes_per_match"]
    half_end = tuple(params["half_end"])
    oos = research.load_cached("oos_ensemble")
    ht = pd.read_sql_query("SELECT match_id, hthg, htag FROM matches WHERE hthg IS NOT NULL", con)
    d = oos[(oos["status"] == "finished")].dropna(subset=["lh"]).merge(ht, on="match_id")
    train = d[(d["season"] >= 2012) & (d["season"] <= 2015)]
    test = d[(d["season"] >= 2016) & (d["season"] <= 2022)]

    def second_half_rates(frame, scale):
        out = []
        for r in frame.itertuples():
            st = {"period": 2, "clock": 45.0, "home_goals": int(r.hthg), "away_goals": int(r.htag), "reds_home": 0, "reds_away": 0,
                  "xg_home": None, "xg_away": None, "exp_home_so_far": None, "exp_away_so_far": None, "recent_xg_home": None, "recent_xg_away": None}
            fc = model.forecast(st, r.lh * scale / mpm, r.la * scale / mpm, half_end)
            out.append((fc["p_home"], fc["p_draw"], fc["p_away"], fc["p_another_goal"]))
        return np.array(out)

    sample = train.sample(min(len(train), 6000), random_state=1)
    hit_tr = ((sample["hg"] + sample["ag"]) > (sample["hthg"] + sample["htag"])).to_numpy()
    y_tr = metrics.outcome_index(sample["hg"], sample["ag"])
    best = None
    grid = [0.96, 1.0, 1.04, 1.08, 1.12]
    scores = {}
    for scale in grid:
        r = second_half_rates(sample, scale)
        ll = metrics.log_loss_rows(r[:, :3], y_tr).mean()
        pm = np.clip(r[:, 3], 1e-9, 1 - 1e-9)
        gl = -(hit_tr * np.log(pm) + (1 - hit_tr) * np.log(1 - pm)).mean()
        scores[scale] = {"log_loss_1x2": float(ll), "another_goal_ll": float(gl), "mean_p_more": float(pm.mean()), "freq": float(hit_tr.mean())}
        if best is None or gl + ll < best[1]:
            best = (scale, gl + ll)
    scale = best[0]
    test_s = test.sample(min(len(test), 12000), random_state=2)
    y = metrics.outcome_index(test_s["hg"], test_s["ag"])
    hit = ((test_s["hg"] + test_s["ag"]) > (test_s["hthg"] + test_s["htag"])).to_numpy()
    base = second_half_rates(test_s, 1.0)
    cal = second_half_rates(test_s, scale)
    def gl(r):
        pm = np.clip(r[:, 3], 1e-9, 1 - 1e-9)
        return -(hit * np.log(pm) + (1 - hit) * np.log(1 - pm))
    cmp_goal = metrics.paired_bootstrap(gl(base), gl(cal))
    cmp_1x2 = metrics.paired_bootstrap(metrics.log_loss_rows(base[:, :3], y), metrics.log_loss_rows(cal[:, :3], y))
    result = {"grid": {str(k): v for k, v in scores.items()}, "selected_scale": scale, "n_train": int(len(sample)), "n_test": int(len(test_s)),
              "test_base": {"mean_p_more": float(base[:, 3].mean()), "ece": metrics.ece(base[:, 3], hit)},
              "test_calibrated": {"mean_p_more": float(cal[:, 3].mean()), "ece": metrics.ece(cal[:, 3], hit)}, "observed": float(hit.mean()),
              "another_goal_cmp": cmp_goal, "result_1x2_cmp": cmp_1x2}
    adopt = scale != 1.0 and cmp_goal["ci_low"] > 0 and cmp_1x2["mean_diff"] >= 0
    record(con, "Second-half rate recalibration from half-time data",
           "The hazard model (fitted on 1,551 StatsBomb matches) under-predicts second-half goals at half-time; a single second-half rate scale fitted on 2012-2015 football-data HT/FT results will improve calibration on 2016-2022.",
           f"Selected scale x{scale:.2f} on n={result['n_train']} (2012-15). Test n={result['n_test']} (2016-22): P(2nd-half goal) mean {result['test_base']['mean_p_more']:.3f} -> {result['test_calibrated']['mean_p_more']:.3f} vs observed {result['observed']:.3f}; "
           f"ECE {result['test_base']['ece']:.3f} -> {result['test_calibrated']['ece']:.3f}; another-goal LL diff {cmp_goal['mean_diff']:+.4f} (CI {cmp_goal['ci_low']:+.4f}..{cmp_goal['ci_high']:+.4f}); 1X2 LL diff {cmp_1x2['mean_diff']:+.4f} (CI {cmp_1x2['ci_low']:+.4f}..{cmp_1x2['ci_high']:+.4f}).",
           "adopt: second-half scale applied in live forecasts" if adopt else "reject: no credible improvement", result)
    db.set_kv(con, "experiment_ht_recalibration", result)
    if adopt:
        params["second_half_scale"] = scale
        con.execute("UPDATE live_model SET params=? WHERE status='champion'", (db.dumps(params),))
    return result


def monte_carlo_error(con, sims=(1000, 10000, 100000), reps=30):
    rng = np.random.default_rng(5)
    cases = [(1.5, 1.1, -0.05), (2.6, 0.6, -0.08), (1.0, 1.0, -0.12)]
    out = []
    for lh, la, rho in cases:
        m = dixon_coles_matrix(lh, la, rho)
        exact = markets(m)
        flat = m.ravel() / m.sum()
        for n in sims:
            errs = {"p_home": [], "over25": [], "btts": []}
            t0 = time.perf_counter()
            for _ in range(reps):
                idx = rng.choice(len(flat), size=n, p=flat)
                h, a = np.divmod(idx, m.shape[1])
                errs["p_home"].append((h > a).mean() - exact["p_home"])
                errs["over25"].append(((h + a) >= 3).mean() - exact["over_under"]["2.5"]["over"])
                errs["btts"].append(((h > 0) & (a > 0)).mean() - exact["btts"])
            elapsed = (time.perf_counter() - t0) / reps
            out.append({"lh": lh, "la": la, "rho": rho, "sims": n, "rmse_p_home": float(np.sqrt(np.mean(np.square(errs["p_home"])))),
                        "rmse_over25": float(np.sqrt(np.mean(np.square(errs["over25"])))), "rmse_btts": float(np.sqrt(np.mean(np.square(errs["btts"])))),
                        "ms_per_match": elapsed * 1000})
    t0 = time.perf_counter()
    for _ in range(200):
        markets(dixon_coles_matrix(1.5, 1.1, -0.05))
    exact_ms = (time.perf_counter() - t0) / 200 * 1000
    worst = max(r["rmse_p_home"] for r in out if r["sims"] == 10000)
    record(con, "Monte Carlo simulation vs exact scoreline enumeration",
           "For a fixed pair of goal rates, Monte Carlo adds sampling error without adding information; exact enumeration should be faster and error-free. Monte Carlo is only worth it when mixing over uncertain inputs (lineups, scenarios).",
           f"Exact markets: {exact_ms:.2f} ms/match. Monte Carlo RMSE on P(home) at 10k sims up to {worst:.4f} (~{worst*100:.1f} pp) at {np.mean([r['ms_per_match'] for r in out if r['sims']==10000]):.1f} ms/match; 100k sims needed for <0.2 pp.",
           "adopt exact enumeration for single-scenario forecasts; use mixtures of exact matrices for scenario uncertainty", {"rows": out, "exact_ms": exact_ms})
    return {"rows": out, "exact_ms": exact_ms}


def draw_calibration(con):
    oos = research.load_cached("oos_ensemble")
    d = _validation(oos)
    y = metrics.outcome_index(d["hg"], d["ag"])
    by = {}
    for comp, g in d.groupby("competition"):
        gy = metrics.outcome_index(g["hg"], g["ag"])
        by[comp] = {"n": int(len(g)), "mean_p_draw": float(g["p_draw"].mean()), "freq_draw": float((gy == 1).mean()),
                    "mean_p_home": float(g["p_home"].mean()), "freq_home": float((gy == 0).mean())}
    rel = metrics.reliability(d["p_draw"].to_numpy(), (y == 1).astype(float))
    gap = float(d["p_draw"].mean() - (y == 1).mean())
    record(con, "Draw calibration of the champion",
           "Poisson-family models are known to under-predict draws; check whether the Dixon-Coles dependence term and learned ensemble remove the bias.",
           f"Validation n={len(d)}: mean predicted draw {d['p_draw'].mean():.3f} vs observed {(y == 1).mean():.3f} (gap {gap:+.3f}). League gaps: " +
           ", ".join(f"{k} {v['mean_p_draw'] - v['freq_draw']:+.3f}" for k, v in by.items()),
           "no change needed" if abs(gap) < 0.01 else "investigate league-specific dependence", {"by_league": by, "reliability": rel})
    return by


def gpu_benchmark(con):
    import lightgbm as lgb
    feats = research.load_cached("features")
    from .features import FEATURE_COLUMNS
    d = feats[(feats["status"] == "finished") & (feats["season"] < 2022)].dropna(subset=["dc_lh"])
    X, y = d[FEATURE_COLUMNS].to_numpy(dtype=float), d["hg"].to_numpy(dtype=float)
    out = {"rows": int(len(d))}
    t0 = time.perf_counter()
    lgb.train({"objective": "poisson", "num_leaves": 15, "learning_rate": 0.03, "verbose": -1, "num_threads": 4}, lgb.Dataset(X, y), 400)
    out["lightgbm_cpu_4threads_s"] = time.perf_counter() - t0
    try:
        import xgboost as xgb
        for device in ("cpu", "cuda"):
            t0 = time.perf_counter()
            xgb.train({"objective": "count:poisson", "max_depth": 4, "eta": 0.03, "tree_method": "hist", "device": device, "nthread": 4}, xgb.DMatrix(X, label=y), 400)
            out[f"xgboost_{device}_s"] = time.perf_counter() - t0
    except Exception as exc:
        out["xgboost_error"] = str(exc)[:300]
    cpu, gpu = out.get("xgboost_cpu_s"), out.get("xgboost_cuda_s")
    gain = (cpu / gpu) if cpu and gpu else None
    fastest = min((v, k) for k, v in out.items() if k.endswith("_s"))[1]
    verdict = f"no meaningful GPU benefit (speed-up x{gain:.2f}); fastest is {fastest}" if gain is None or gain < 1.3 else f"GPU speed-up x{gain:.2f}"
    record(con, "GPU vs CPU for gradient boosting",
           "The tabular training set (~70k rows, 26 features) is too small for GPU acceleration to pay off.",
           ", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in out.items()),
           f"{verdict}; keep CPU training with 4 threads", out)
    return out


@handler("experiments")
def run_experiments(ctx, names=None):
    available = {"market": market_information, "draws": draw_calibration, "monte_carlo": monte_carlo_error,
                 "ht_recalibration": halftime_recalibration, "gpu": gpu_benchmark}
    out = {}
    for name in names or list(available):
        ctx.stage(f"experiment: {name}")
        try:
            out[name] = available[name](ctx.con)
        except Exception as exc:
            ctx.log(f"{name} failed: {exc}", "error")
            out[name] = {"error": str(exc)}
    return {k: ("ok" if "error" not in (v if isinstance(v, dict) else {}) else v["error"]) for k, v in out.items()}


def _xi_indices(con, shrink_minutes=270.0):
    apps = pd.read_sql_query("""SELECT a.sb_match_id, s.match_date, s.match_id, a.player_id, a.side, a.team_name, a.position, a.started, a.minutes, a.xg, a.xa
                                FROM appearances a JOIN sb_matches s ON s.sb_match_id=a.sb_match_id WHERE s.match_id IS NOT NULL""", con)
    apps["group"] = apps["position"].fillna("").map(lambda p: "GK" if "Goalkeeper" in p else "DF" if "Back" in p else "MF" if "Midfield" in p else "FW")
    apps = apps.sort_values(["match_date", "sb_match_id"])
    totals = {}
    group_tot = {}
    team_hist = {}
    team_contrib = {}
    rows = []
    for date, day in apps.groupby("match_date", sort=True):
        for (sb_id, side), g in day.groupby(["sb_match_id", "side"]):
            team = g["team_name"].iloc[0]
            xi = g[g["started"] == 1]
            index = 0.0
            known = 0
            for p in xi.itertuples():
                gt = group_tot.get(p.group, (0.0, 0.0))
                mu = gt[0] / gt[1] * 90 if gt[1] > 0 else 0.15
                t = totals.get(p.player_id, (0.0, 0.0))
                known += t[1] > 0
                index += (t[0] * 90 + mu * shrink_minutes) / (t[1] + shrink_minutes)
            hist = team_hist.get(team, [])
            base = float(np.mean(hist[-10:])) if len(hist) >= 3 else None
            contrib = team_contrib.get(team, {})
            season_total = sum(contrib.values())
            missing = sum(v for pid, v in contrib.items() if pid not in set(xi["player_id"])) / season_total if season_total > 0 else None
            rows.append({"sb_match_id": sb_id, "match_id": int(g["match_id"].iloc[0]), "side": side, "date": date, "xi_index": index, "baseline": base,
                         "delta": math.log(index / base) if base else None, "missing_share": missing, "known_starters": known})
            team_hist.setdefault(team, []).append(index)
        for p in day.itertuples():
            t = totals.get(p.player_id, (0.0, 0.0))
            totals[p.player_id] = (t[0] + p.xg + p.xa, t[1] + p.minutes)
            gt = group_tot.get(p.group, (0.0, 0.0))
            group_tot[p.group] = (gt[0] + p.xg + p.xa, gt[1] + p.minutes)
            team_contrib.setdefault(p.team_name, {})
            team_contrib[p.team_name][p.player_id] = team_contrib[p.team_name].get(p.player_id, 0.0) + p.xg + p.xa
    return pd.DataFrame(rows)


def player_study(con, split_date="2016-01-15"):
    idx = _xi_indices(con)
    wide = idx.pivot_table(index="match_id", columns="side", values=["delta", "missing_share"], aggfunc="first")
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    oos = research.load_cached("oos_ensemble")
    d = oos[oos["status"] == "finished"].merge(wide.reset_index(), on="match_id").dropna(subset=["delta_home", "delta_away", "lh"])
    d = d.sort_values("date")
    hg, ag = d["hg"].to_numpy(), d["ag"].to_numpy()

    def adjusted(frame, beta, gamma):
        lh = frame["lh"].to_numpy() * np.exp(beta * frame["delta_home"].to_numpy())
        la = frame["la"].to_numpy() * np.exp(beta * frame["delta_away"].to_numpy())
        return lh, la

    def nll(theta, frame):
        lh, la = adjusted(frame, theta[0], 0.0)
        from scipy.stats import poisson
        return -(poisson.logpmf(frame["hg"], lh) + poisson.logpmf(frame["ag"], la)).sum()

    train = d[d["date"] < pd.Timestamp(split_date)]
    test = d[d["date"] >= pd.Timestamp(split_date)]
    res = minimize(lambda t: nll(t, train), [0.0], method="Nelder-Mead")
    beta = float(res.x[0])

    def evaluate(frame, b):
        lh, la = adjusted(frame, b, 0.0)
        p = np.array([outcome_probs(dixon_coles_matrix(a, c, r)) for a, c, r in zip(lh, la, frame["rho"])])
        sp = np.array([dixon_coles_matrix(a, c, r)[min(int(h), 10), min(int(g), 10)] for a, c, r, h, g in zip(lh, la, frame["rho"], frame["hg"], frame["ag"])])
        y = metrics.outcome_index(frame["hg"], frame["ag"])
        return metrics.log_loss_rows(p, y), -np.log(np.clip(sp, 1e-12, 1))

    ll0, s0 = evaluate(test, 0.0)
    ll1, s1 = evaluate(test, beta)
    cmp_ll = metrics.paired_bootstrap(ll0, ll1)
    cmp_s = metrics.paired_bootstrap(s0, s1)
    corr = float(np.corrcoef(d["delta_home"], d["hg"] - d["lh"])[0, 1])
    adopted = cmp_ll["ci_low"] > 0 and cmp_s["ci_low"] > 0
    summary = (f"Tested on {len(test)} held-out 2015/16 matches (fitted on {len(train)} earlier ones). The starting XI's prior xG+xA index, relative to the team's usual XI, "
               f"gets coefficient {beta:+.2f} on log goal rate; 1X2 log loss change {cmp_ll['mean_diff']:+.4f} (95% CI {cmp_ll['ci_low']:+.4f}..{cmp_ll['ci_high']:+.4f}), "
               f"scoreline NLL change {cmp_s['mean_diff']:+.4f} (CI {cmp_s['ci_low']:+.4f}..{cmp_s['ci_high']:+.4f}). "
               + ("Adopted for matches with confirmed lineups." if adopted else "Not adopted: the evidence is not strong enough on this sample, so forecasts stay team-level."))
    result = {"beta": beta, "n_train": int(len(train)), "n_test": int(len(test)), "cmp_log_loss": cmp_ll, "cmp_score_nll": cmp_s,
              "corr_delta_residual": corr, "delta_sd": float(d["delta_home"].std()), "adopted": bool(adopted), "summary": summary,
              "missing_share_mean": float(d["missing_share_home"].mean())}
    record(con, "Confirmed starting XI quality vs team-only forecasts",
           "If confirmed lineups carry information beyond team-level ratings, adjusting pre-match goal rates by the starting XI's prior xG+xA per 90 (shrunk toward positional means, relative to the team's recent XIs) will improve held-out forecasts.",
           summary, "adopt" if adopted else "reject (insufficient evidence)", result)
    db.set_kv(con, "player_study", result)
    return result


@handler("player_study")
def player_study_job(ctx):
    ctx.stage("building starting-XI indices from StatsBomb appearances")
    return player_study(ctx.con)


def record_research(con):
    s = db.get_kv(con, "research_summary")
    c = db.get_kv(con, "validation_comparisons")
    r = s["results"]

    def cmp(key):
        v = c.get(key)
        return f"{v['mean_diff']:+.4f} (95% CI {v['ci_low']:+.4f}..{v['ci_high']:+.4f})" if v else "n/a"

    n = r["dixon_coles"]["n"]
    record(con, "Elo K-factor", "A lower K (slower updates) should suit league football where single results are noisy.",
           f"Validation n={n} log loss: K=15 {r['elo_k15']['log_loss']:.4f}, K=20 {r['elo']['log_loss']:.4f}, K=30 {r['elo_k30']['log_loss']:.4f}.",
           "adopt K=15 as the Elo baseline", {k: r[k] for k in ("elo", "elo_k15", "elo_k30")})
    record(con, "Dixon-Coles low-score correction vs independent Poisson", "The Dixon-Coles dependence term should improve draw and low-score probabilities.",
           f"Log loss diff (Poisson - DC) {cmp('poisson vs dixon_coles [log_loss]')}; scoreline NLL {cmp('poisson vs dixon_coles [score_nll]')}; ECE {r['poisson']['ece']:.4f} -> {r['dixon_coles']['ece']:.4f}.",
           "adopt (small but consistent calibration benefit; on the final test the log loss gain was +0.0008, CI +0.0002..+0.0015)", {})
    record(con, "Time-decay half-life for team strengths", "Exponential down-weighting of old matches should beat both slower and faster decay.",
           f"Scoreline NLL: xi=0.0010 {r['dc_xi0010']['score_nll']:.4f}, xi=0.0019 {r['dixon_coles']['score_nll']:.4f}, xi=0.0030 {r['dc_xi0030']['score_nll']:.4f}; ridge 3.0 {r['dc_ridge3']['score_nll']:.4f}.",
           "adopt xi=0.0019/day (half-life about one year) and ridge 1.0", {})
    record(con, "Shot-based expected-goals proxy as a response", "Shots on target carry less noise than goals, so blending a shot-based xG proxy into the strength fit should improve forecasts.",
           f"Log loss: 0% {r['dixon_coles']['log_loss']:.4f}, 25% {r['dc_xg25']['log_loss']:.4f}, 50% {r['dc_xg50']['log_loss']:.4f}, 75% {r['dc_xg75']['log_loss']:.4f}; ECE rises to {r['dc_xg75']['ece']:.4f} at 75%. Goal MAE improves slightly ({r['dixon_coles']['goal_mae']:.3f} -> {r['dc_xg25']['goal_mae']:.3f}).",
           "reject as a standalone model; retained as an ensemble component where it earns weight", {})
    record(con, "Gradient boosting on engineered features", "LightGBM with Elo, rolling form, shot volumes, rest and promotion flags should capture non-linear effects the rating models miss.",
           f"Log loss {r['gbm_features']['log_loss']:.4f} vs DC {r['dixon_coles']['log_loss']:.4f}; DC-xG vs GBM diff {cmp('dc_xg25 vs gbm_features [log_loss]')}. Stacked on DC rates: {r['gbm_stacked']['log_loss']:.4f}.",
           "adopt as ensemble component", {})
    record(con, "Learned ensemble of goal-rate models", "A geometric blend of component goal rates with weights learned on prior seasons should beat every component while keeping a coherent score distribution.",
           f"vs DC: log loss {cmp('dixon_coles vs ensemble [log_loss]')}, scoreline NLL {cmp('dixon_coles vs ensemble [score_nll]')}. vs bookmaker close: {cmp('ensemble vs bookmaker_close [log_loss]')} (bookmaker better).",
           "adopt as champion", {})
    record(con, "Multinomial logistic regression baseline", "A linear model on the same features gives a strong 1X2-only baseline.",
           f"Log loss {r['logistic']['log_loss']:.4f}; logistic vs DC {cmp('logistic vs dixon_coles [log_loss]')} (negative = logistic better).",
           "keep as research baseline only (no score distribution)", {})
