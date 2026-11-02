import math
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import db, metrics, research
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
    import json
    from .inplay import InPlayModel
    row = con.execute("SELECT params FROM live_model WHERE status='champion' ORDER BY created_at DESC LIMIT 1").fetchone()
    params = json.loads(row["params"])
    model = InPlayModel.from_json(params["model"])
    mpm = params["minutes_per_match"]
    half_end = tuple(params["half_end"])
    oos = research.load_cached("oos_ensemble")
    ht = pd.read_sql_query("SELECT match_id, hthg, htag FROM matches WHERE hthg IS NOT NULL", con)
    d = oos[(oos["status"] == "finished")].dropna(subset=["lh"]).merge(ht, on="match_id")

    def forecasts(frame, scale):
        model.second_half_scale = scale
        out = []
        for r in frame.itertuples():
            st = {"period": 2, "clock": 45.0, "home_goals": int(r.hthg), "away_goals": int(r.htag), "reds_home": 0, "reds_away": 0,
                  "xg_home": None, "xg_away": None, "exp_home_so_far": None, "exp_away_so_far": None, "recent_xg_home": None, "recent_xg_away": None}
            fc = model.forecast(st, r.lh / mpm, r.la / mpm, half_end)
            out.append((fc["p_home"], fc["p_draw"], fc["p_away"], fc["p_another_goal"]))
        return np.array(out)

    def losses(frame, r):
        y = metrics.outcome_index(frame["hg"], frame["ag"])
        hit = ((frame["hg"] + frame["ag"]) > (frame["hthg"] + frame["htag"])).to_numpy()
        pm = np.clip(r[:, 3], 1e-9, 1 - 1e-9)
        return metrics.log_loss_rows(r[:, :3], y), -(hit * np.log(pm) + (1 - hit) * np.log(1 - pm)), hit

    groups = {"1": d["tier"] == 1, "2+": d["tier"] >= 2}
    result = {}
    scales = {}
    lines = []
    for label, mask in groups.items():
        train = d[mask & (d["season"] >= 2012) & (d["season"] <= 2015)].sample(frac=1, random_state=1).head(5000)
        test = d[mask & (d["season"] >= 2016) & (d["season"] <= 2022)].sample(frac=1, random_state=2).head(8000)
        grid = {}
        for scale in (0.96, 1.0, 1.04, 1.08, 1.12, 1.16):
            l1, lg, _ = losses(train, forecasts(train, scale))
            grid[scale] = float(l1.mean() + lg.mean())
        best = min(grid, key=grid.get)
        base = forecasts(test, 1.0)
        cal = forecasts(test, best)
        b1, bg, hit = losses(test, base)
        c1, cg, _ = losses(test, cal)
        cmp_goal = metrics.paired_bootstrap(bg, cg)
        cmp_1x2 = metrics.paired_bootstrap(b1, c1)
        adopt = best != 1.0 and cmp_goal["ci_low"] > 0 and cmp_1x2["mean_diff"] >= 0
        scales[label] = best if adopt else 1.0
        result[label] = {"grid": {str(k): v for k, v in grid.items()}, "selected": best, "adopted": adopt, "n_train": int(len(train)), "n_test": int(len(test)),
                         "observed": float(hit.mean()), "base_mean_p": float(base[:, 3].mean()), "cal_mean_p": float(cal[:, 3].mean()),
                         "base_ece": metrics.ece(base[:, 3], hit), "cal_ece": metrics.ece(cal[:, 3], hit), "another_goal_cmp": cmp_goal, "result_1x2_cmp": cmp_1x2}
        lines.append(f"tier {label}: scale x{best:.2f} (train n={len(train)}), test n={len(test)}: P(2nd-half goal) {base[:, 3].mean():.3f} -> {cal[:, 3].mean():.3f} vs observed {hit.mean():.3f}, "
                     f"ECE {result[label]['base_ece']:.3f} -> {result[label]['cal_ece']:.3f}, another-goal LL {cmp_goal['mean_diff']:+.4f} (CI {cmp_goal['ci_low']:+.4f}..{cmp_goal['ci_high']:+.4f}), "
                     f"1X2 LL {cmp_1x2['mean_diff']:+.4f} (CI {cmp_1x2['ci_low']:+.4f}..{cmp_1x2['ci_high']:+.4f}){' adopted' if adopt else ' not adopted'}")
    record(con, "Second-half rate recalibration from half-time data",
           "The hazard model (fitted on StatsBomb 2015/16 top-flight matches) mis-states second-half scoring at half-time; a second-half rate scale per tier group, fitted on 2012-2015 football-data HT/FT results, will improve calibration on 2016-2022.",
           "; ".join(lines), "adopt per-tier scales where credible: " + ", ".join(f"tier {k} x{v:.2f}" for k, v in scales.items()), result)
    db.set_kv(con, "experiment_ht_recalibration", result)
    params["second_half_scale_by_tier"] = scales
    params.pop("second_half_scale", None)
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
                 "ht_recalibration": halftime_recalibration, "gpu": gpu_benchmark, "inplay_windows": inplay_windows, "stoppage_era": stoppage_era_check,
                 "provider_xg": provider_xg_study}
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
    rotation = (test["delta_home"].abs() + test["delta_away"].abs()).to_numpy()
    cut = np.quantile(rotation, [1 / 3, 2 / 3])
    bands = {}
    for label, mask in (("usual XIs (lowest third of deviation)", rotation <= cut[0]), ("moderate rotation", (rotation > cut[0]) & (rotation <= cut[1])),
                        ("heavy rotation (top third)", rotation > cut[1])):
        bands[label] = {"n": int(mask.sum()), "team_only_log_loss": float(ll0[mask].mean()), "xi_adjusted_log_loss": float(ll1[mask].mean()),
                        "team_only_score_nll": float(s0[mask].mean()), "xi_adjusted_score_nll": float(s1[mask].mean())}
    adopted = cmp_ll["ci_low"] > 0 and cmp_s["ci_low"] > 0
    summary = (f"Tested on {len(test)} held-out 2015/16 matches (fitted on {len(train)} earlier ones). The starting XI's prior xG+xA index, relative to the team's usual XI, "
               f"gets coefficient {beta:+.2f} on log goal rate; 1X2 log loss change {cmp_ll['mean_diff']:+.4f} (95% CI {cmp_ll['ci_low']:+.4f}..{cmp_ll['ci_high']:+.4f}), "
               f"scoreline NLL change {cmp_s['mean_diff']:+.4f} (CI {cmp_s['ci_low']:+.4f}..{cmp_s['ci_high']:+.4f}). "
               + ("Adopted for matches with confirmed lineups." if adopted else "Not adopted: the evidence is not strong enough on this sample, so forecasts stay team-level."))
    result = {"beta": beta, "n_train": int(len(train)), "n_test": int(len(test)), "cmp_log_loss": cmp_ll, "cmp_score_nll": cmp_s,
              "corr_delta_residual": corr, "delta_sd": float(d["delta_home"].std()), "adopted": bool(adopted), "summary": summary,
              "missing_share_mean": float(d["missing_share_home"].mean()), "by_rotation": bands}
    record(con, "Forecast quality under lineup uncertainty",
           "Team-only forecasts implicitly assume a typical lineup, so they should be worse when the confirmed XI deviates strongly from the team's usual XI; an XI adjustment should help most there.",
           "; ".join(f"{k}: n={v['n']}, team-only LL {v['team_only_log_loss']:.4f}, XI-adjusted {v['xi_adjusted_log_loss']:.4f}" for k, v in bands.items()),
           "analysis only (small samples; see the XI experiment for the adoption decision)", bands)
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


def inplay_windows(con, split_date="2016-02-01"):
    from . import inplay_train
    from .live import load_live_model
    model, params = load_live_model(con)
    mpm, half_end = params["minutes_per_match"], tuple(params["half_end"])
    oos = research.load_cached("oos_ensemble")
    matches, grouped = inplay_train.load_sb(con, oos[["match_id", "lh", "la"]])
    test = matches[matches["match_date"] >= split_date]
    rows = []
    for m in test.itertuples():
        if m.sb_match_id not in grouped.groups:
            continue
        goals, reds, shots = inplay_train.match_timeline(grouped.get_group(m.sb_match_id))
        g = goals[["period", "clock", "side"]].to_numpy()
        for period, start, stop in ((1, 5, 45), (2, 50, 90)):
            for clock in range(start, stop, 5):
                st = inplay_train.state_at(goals, reds, shots, period, float(clock), m.lh, m.la, mpm)
                fc = model.forecast(st, m.lh / mpm, m.la / mpm, half_end)
                after = g[(g[:, 0] > period) | ((g[:, 0] == period) & (g[:, 1] >= clock))] if len(g) else g
                in10 = any((r[0] == period and r[1] < clock + 10) for r in after)
                nxt = after[0][2] if len(after) else "none"
                fh_final = int((g[:, 2] == "home").sum()) if len(g) else 0
                fa_final = int((g[:, 2] == "away").sum()) if len(g) else 0
                lead = st["home_goals"] - st["away_goals"]
                holds = None
                if lead != 0:
                    holds = int((fh_final > fa_final) if lead > 0 else (fa_final > fh_final))
                rows.append({"clock": clock, "period": period, "p10": fc["p_goal_10"], "hit10": int(in10),
                             "p_next": (fc["p_next_home"], fc["p_next_away"], fc["p_no_more"]), "next": {"home": 0, "away": 1, "none": 2}[nxt],
                             "p_hold": fc["p_leader_holds"], "holds": holds})
    df = pd.DataFrame(rows)
    p10 = df["p10"].to_numpy()
    hit = df["hit10"].to_numpy()
    pn = np.array(df["p_next"].tolist())
    yn = df["next"].to_numpy()
    base_rate = hit.mean()
    ll10 = -(hit * np.log(np.clip(p10, 1e-9, 1)) + (1 - hit) * np.log(np.clip(1 - p10, 1e-9, 1))).mean()
    ll10_base = -(hit * np.log(base_rate) + (1 - hit) * np.log(1 - base_rate)).mean()
    lead_rows = df.dropna(subset=["holds"])
    ph = lead_rows["p_hold"].to_numpy(dtype=float)
    hh = lead_rows["holds"].to_numpy(dtype=float)
    result_hold = {"n": int(len(lead_rows)), "mean_p": float(ph.mean()), "freq": float(hh.mean()), "ece": metrics.ece(ph, hh),
                   "brier": float(((ph - hh) ** 2).mean()), "reliability": metrics.reliability(ph, hh)}
    result = {"leader_holds": result_hold, "n_checkpoints": int(len(df)), "n_matches": int(df.shape[0] // 16),
              "goal_10min": {"mean_p": float(p10.mean()), "freq": float(base_rate), "ece": metrics.ece(p10, hit), "log_loss": float(ll10),
                             "log_loss_constant_base_rate": float(ll10_base), "reliability": metrics.reliability(p10, hit)},
              "next_goal": {"log_loss": float(metrics.log_loss_rows(pn, yn).mean()), "accuracy": float((pn.argmax(axis=1) == yn).mean()),
                            "mean_p": pn.mean(axis=0).tolist(), "freq": np.bincount(yn, minlength=3).tolist()}}
    record(con, "In-play short-horizon and next-goal calibration (held-out replay)",
           "The hazard model's 10-minute goal probabilities and next-scorer probabilities should be calibrated on held-out StatsBomb matches, beating a constant base rate.",
           f"{result['n_checkpoints']} checkpoints every 5 minutes: P(goal in next 10') mean {p10.mean():.3f} vs observed {base_rate:.3f}, ECE {result['goal_10min']['ece']:.3f}, "
           f"log loss {ll10:.4f} vs constant base rate {ll10_base:.4f}; next-goal (home/away/none) log loss {result['next_goal']['log_loss']:.4f}, accuracy {result['next_goal']['accuracy']:.3f}; "
           f"leader goes on to win: predicted {result_hold['mean_p']:.3f} vs observed {result_hold['freq']:.3f} over {result_hold['n']} checkpoints with a lead, ECE {result_hold['ece']:.3f}, Brier {result_hold['brier']:.4f}.",
           "keep (calibrated)" if result["goal_10min"]["ece"] < 0.03 else "investigate calibration", result)
    db.set_kv(con, "experiment_inplay_windows", result)
    return result


def stoppage_era_check(con):
    from .live import load_live_model
    model, params = load_live_model(con)
    mpm, half_end = params["minutes_per_match"], tuple(params["half_end"])
    oos = research.load_cached("oos_ensemble")
    ht = pd.read_sql_query("SELECT match_id, hthg, htag FROM matches WHERE hthg IS NOT NULL", con)
    d = oos[oos["status"] == "finished"].dropna(subset=["lh"]).merge(ht, on="match_id")
    d = d[d["competition"].isin(PRIMARY_COMPETITIONS)]
    out = {}
    for season in range(2016, 2027):
        g = d[d["season"] == season]
        if len(g) > 2500:
            g = g.sample(2500, random_state=season)
        preds = []
        for r in g.itertuples():
            st = {"period": 2, "clock": 45.0, "home_goals": int(r.hthg), "away_goals": int(r.htag), "reds_home": 0, "reds_away": 0,
                  "xg_home": None, "xg_away": None, "exp_home_so_far": None, "exp_away_so_far": None, "recent_xg_home": None, "recent_xg_away": None}
            preds.append(model.forecast(st, r.lh / mpm, r.la / mpm, half_end)["exp_remaining"])
        actual = (g["hg"] + g["ag"] - g["hthg"] - g["htag"]).to_numpy()
        first = (g["hthg"] + g["htag"]).to_numpy()
        out[int(season)] = {"n": int(len(g)), "pred_2h_goals": float(np.mean(preds)), "actual_2h_goals": float(actual.mean()),
                            "actual_1h_goals": float(first.mean()), "pre_match_total": float((g["lh"] + g["la"]).mean())}
    early = [v["actual_2h_goals"] - v["pred_2h_goals"] for s, v in out.items() if s <= 2022]
    late = [v["actual_2h_goals"] - v["pred_2h_goals"] for s, v in out.items() if s >= 2023]
    record(con, "Second-half goals before and after the 2023 stoppage-time change (analysis only)",
           "Longer added time since 2023/24 should raise second-half goals relative to a model fitted on 2015/16 half lengths. This uses final-test seasons for diagnosis only, not for tuning.",
           f"Mean (actual - predicted) second-half goals: 2016-2022 {np.mean(early):+.3f}, 2023-2026 {np.mean(late):+.3f}. Per season: " +
           ", ".join(f"{s}: {v['actual_2h_goals']:.2f} vs {v['pred_2h_goals']:.2f}" for s, v in out.items()),
           "flag for future recalibration on post-2023 data once more seasons are available" if np.mean(late) - np.mean(early) > 0.03 else "no era effect detected", out)
    db.set_kv(con, "experiment_stoppage_era", out)
    return out



def _team_match_metrics(con):
    q = """
        WITH sides AS (
            SELECT sb_match_id, match_id, match_date, 'home' side, home_team_id team FROM sb_matches WHERE match_id IS NOT NULL
            UNION ALL SELECT sb_match_id, match_id, match_date, 'away', away_team_id FROM sb_matches WHERE match_id IS NOT NULL)
        SELECT s.sb_match_id, s.match_id, s.match_date, s.side, s.team,
               (SELECT COALESCE(SUM(xg), 0) FROM sb_events e WHERE e.sb_match_id=s.sb_match_id AND e.kind='shot' AND e.side=s.side AND COALESCE(e.detail,'')<>'Penalty') xg_for,
               (SELECT COALESCE(SUM(xg), 0) FROM sb_events e WHERE e.sb_match_id=s.sb_match_id AND e.kind='shot' AND e.side<>s.side AND COALESCE(e.detail,'')<>'Penalty') xg_against,
               (SELECT COUNT(*) FROM sb_events e WHERE e.sb_match_id=s.sb_match_id AND e.kind='goal' AND e.side=s.side) goals_for,
               (SELECT COUNT(*) FROM sb_events e WHERE e.sb_match_id=s.sb_match_id AND e.kind='goal' AND e.side<>s.side) goals_against,
               (SELECT CAST(SUM(CASE WHEN m.side=s.side THEN m.final_third_passes END) AS REAL) / NULLIF(SUM(m.final_third_passes), 0) FROM sb_team_minutes m WHERE m.sb_match_id=s.sb_match_id) tilt
        FROM sides s"""
    return pd.read_sql_query(q, con)


def provider_xg_study(con, split_date="2016-01-15", window=8, min_prior=5):
    tm = _team_match_metrics(con).sort_values(["match_date", "sb_match_id"])
    rows = []
    hist = {}
    for date, day in tm.groupby("match_date", sort=True):
        for r in day.itertuples():
            h = hist.get(r.team, [])[-window:]
            if len(h) >= min_prior:
                arr = np.array(h)
                rows.append({"match_id": r.match_id, "side": r.side, "xgf": arr[:, 0].mean(), "xga": arr[:, 1].mean(), "gf": arr[:, 2].mean(),
                             "ga": arr[:, 3].mean(), "tilt": np.nanmean(arr[:, 4]), "n": len(h)})
        for r in day.itertuples():
            hist.setdefault(r.team, []).append((r.xg_for, r.xg_against, r.goals_for, r.goals_against, r.tilt if r.tilt is not None else np.nan))
    f = pd.DataFrame(rows)
    wide = f.pivot_table(index="match_id", columns="side", values=["xgf", "xga", "gf", "ga", "tilt"], aggfunc="first")
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    oos = research.load_cached("oos_ensemble")
    d = oos[oos["status"] == "finished"].merge(wide.reset_index(), on="match_id").dropna(subset=["xgf_home", "xgf_away", "lh"]).sort_values("date")
    eps = 0.2
    d["x_home"] = np.log((d["xgf_home"] + d["xga_away"] + eps) / (d["gf_home"] + d["ga_away"] + eps))
    d["x_away"] = np.log((d["xgf_away"] + d["xga_home"] + eps) / (d["gf_away"] + d["ga_home"] + eps))
    d["t_home"] = d["tilt_home"] - d["tilt_away"]
    from scipy.stats import poisson as _poisson

    def rates(frame, theta, use_tilt):
        bh = theta[0] * frame["x_home"].to_numpy() + (theta[1] * frame["t_home"].to_numpy() if use_tilt else 0)
        ba = theta[0] * frame["x_away"].to_numpy() - (theta[1] * frame["t_home"].to_numpy() if use_tilt else 0)
        return frame["lh"].to_numpy() * np.exp(bh), frame["la"].to_numpy() * np.exp(ba)

    def nll(theta, frame, use_tilt):
        lh, la = rates(frame, theta, use_tilt)
        return -(_poisson.logpmf(frame["hg"], lh) + _poisson.logpmf(frame["ag"], la)).sum()

    train = d[d["date"] < pd.Timestamp(split_date)]
    test = d[d["date"] >= pd.Timestamp(split_date)]
    out = {"n_train": int(len(train)), "n_test": int(len(test))}
    for label, use_tilt in (("xg_vs_goals", False), ("xg_vs_goals_plus_tilt", True)):
        theta = minimize(lambda t: nll(t, train, use_tilt), [0.0, 0.0], method="Nelder-Mead").x
        lh0, la0 = rates(test, [0.0, 0.0], False)
        lh1, la1 = rates(test, theta, use_tilt)
        rho = test["rho"].to_numpy()
        y = metrics.outcome_index(test["hg"], test["ag"])

        def losses(lh, la):
            p = np.array([outcome_probs(dixon_coles_matrix(a, b, r)) for a, b, r in zip(lh, la, rho)])
            sp = np.array([dixon_coles_matrix(a, b, r)[min(int(h), 10), min(int(g), 10)] for a, b, r, h, g in zip(lh, la, rho, test["hg"], test["ag"])])
            return metrics.log_loss_rows(p, y), -np.log(np.clip(sp, 1e-12, 1))

        b1, bs = losses(lh0, la0)
        c1, cs = losses(lh1, la1)
        out[label] = {"theta": [float(t) for t in theta], "log_loss_cmp": metrics.paired_bootstrap(b1, c1), "score_nll_cmp": metrics.paired_bootstrap(bs, cs),
                      "base_log_loss": float(b1.mean()), "new_log_loss": float(c1.mean())}
    a, b = out["xg_vs_goals"], out["xg_vs_goals_plus_tilt"]
    adopted = a["log_loss_cmp"]["ci_low"] > 0 and a["score_nll_cmp"]["ci_low"] > 0
    record(con, "Provider xG and field tilt beyond the champion (StatsBomb 2015/16)",
           "Teams whose recent StatsBomb xG exceeds their goals (for and against) have been unlucky and should regress upwards; adjusting the champion's goal rates by the log ratio of recent xG to recent goals (and by field-tilt difference) should improve held-out forecasts.",
           f"Fitted on {out['n_train']} matches before {split_date}, tested on {out['n_test']}. xG/goals adjustment: coefficient {a['theta'][0]:+.2f}, log loss {a['base_log_loss']:.4f} -> {a['new_log_loss']:.4f} "
           f"(diff {a['log_loss_cmp']['mean_diff']:+.4f}, CI {a['log_loss_cmp']['ci_low']:+.4f}..{a['log_loss_cmp']['ci_high']:+.4f}), score NLL diff {a['score_nll_cmp']['mean_diff']:+.4f} "
           f"(CI {a['score_nll_cmp']['ci_low']:+.4f}..{a['score_nll_cmp']['ci_high']:+.4f}). Adding field tilt (coefficient {b['theta'][1]:+.2f}): log loss diff {b['log_loss_cmp']['mean_diff']:+.4f} "
           f"(CI {b['log_loss_cmp']['ci_low']:+.4f}..{b['log_loss_cmp']['ci_high']:+.4f}).",
           "adopt where provider xG exists" if adopted else "not adopted: no credible gain on one season of provider xG (the champion already uses shots on target)", out)
    db.set_kv(con, "experiment_provider_xg", out)
    return out
