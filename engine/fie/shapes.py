import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from . import db, metrics, research
from .config import CORE_COMPETITIONS, PRIMARY_COMPETITIONS, VALIDATION_SEASONS
from .jobs import handler

G = 11
GOALS = np.arange(G)
LOGFACT = gammaln(GOALS + 1)


def _pois(lam):
    lam = np.clip(np.asarray(lam, float), 1e-9, None)[:, None]
    return np.exp(GOALS * np.log(lam) - lam - LOGFACT)


def _tau(m, lh, la, rho):
    rho = np.broadcast_to(rho, lh.shape)
    m = m.copy()
    m[:, 0, 0] *= 1 - lh * la * rho
    m[:, 0, 1] *= 1 + lh * rho
    m[:, 1, 0] *= 1 + la * rho
    m[:, 1, 1] *= 1 - rho
    m = np.clip(m, 1e-15, None)
    return m / m.sum(axis=(1, 2), keepdims=True)


def dixon_coles(lh, la, rho):
    return _tau(_pois(lh)[:, :, None] * _pois(la)[:, None, :], lh, la, rho)


def bivariate(lh, la, rho, c):
    l3 = c * np.minimum(lh, la)
    p1, p2, p3 = _pois(lh - l3), _pois(la - l3), _pois(l3)
    m = np.zeros((len(lh), G, G))
    for k in range(G):
        m[:, k:, k:] += p3[:, k, None, None] * p1[:, :G - k, None] * p2[:, None, :G - k]
    return _tau(m, lh, la, rho)


_GRID = np.linspace(-6.0, 3.5, 400)


def _com_table(nu):
    logp = GOALS[None, :] * _GRID[:, None] - nu * LOGFACT[None, :]
    p = np.exp(logp - logp.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    return p, p @ GOALS


def _com(mean, nu):
    table, means = _com_table(nu)
    x = np.interp(np.clip(mean, means[0], means[-1]), means, np.arange(len(_GRID)))
    i = np.clip(np.floor(x).astype(int), 0, len(_GRID) - 2)
    f = (x - i)[:, None]
    p = table[i] * (1 - f) + table[i + 1] * f
    return p / p.sum(axis=1, keepdims=True)


def com_poisson(lh, la, rho, nu_h, nu_a):
    return _tau(_com(lh, nu_h)[:, :, None] * _com(la, nu_a)[:, None, :], lh, la, rho)


def gap_rho(lh, la, r0, r1):
    return dixon_coles(lh, la, np.clip(r0 + r1 * np.abs(lh - la), -0.4, 0.4))


def diagonal_gap(lh, la, rho, a, b):
    m = dixon_coles(lh, la, rho)
    i = np.arange(G)
    m[:, i, i] *= np.exp(a + b * np.abs(lh - la))[:, None]
    return m / m.sum(axis=(1, 2), keepdims=True)


FAMILIES = {
    "dixon_coles": (lambda lh, la, t: dixon_coles(lh, la, t[0]), [0.0], [(-0.3, 0.3)]),
    "bivariate_poisson": (lambda lh, la, t: bivariate(lh, la, t[0], t[1]), [0.0, 0.05], [(-0.3, 0.3), (0.0, 0.6)]),
    "com_poisson": (lambda lh, la, t: com_poisson(lh, la, t[0], t[1], t[2]), [0.0, 1.0, 1.0], [(-0.3, 0.3), (0.6, 1.8), (0.6, 1.8)]),
    "gap_rho": (lambda lh, la, t: gap_rho(lh, la, t[0], t[1]), [0.0, 0.0], [(-0.3, 0.3), (-0.3, 0.3)]),
    "diagonal_gap": (lambda lh, la, t: diagonal_gap(lh, la, t[0], t[1], t[2]), [0.0, 0.0, 0.0], [(-0.3, 0.3), (-0.5, 0.5), (-1.0, 1.0)]),
}


def score_nll(m, hg, ag):
    h = np.minimum(hg, G - 1).astype(int)
    a = np.minimum(ag, G - 1).astype(int)
    return -np.log(m[np.arange(len(m)), h, a])


def outcome(m):
    home = np.tril(m, -1).sum(axis=(1, 2))
    draw = np.trace(m, axis1=1, axis2=2)
    return np.column_stack([home, draw, 1 - home - draw])


def fit(family, lh, la, hg, ag):
    fn, x0, bounds = FAMILIES[family]
    res = minimize(lambda t: score_nll(fn(lh, la, t), hg, ag).mean(), x0, bounds=bounds, method="L-BFGS-B")
    return [float(v) for v in res.x], float(res.fun)


def walk_forward(oos, family, first_season=2013, lookback=3):
    d = oos[oos["lh"].notna()].sort_values(["date", "match_id"])
    outs, params = [], {}
    for season in sorted(d.loc[d["season"] >= first_season, "season"].unique()):
        tr = d[(d["season"] < season) & (d["season"] >= season - lookback) & (d["status"] == "finished") & d["competition"].isin(CORE_COMPETITIONS)]
        te = d[d["season"] == season]
        theta, _ = fit(family, tr["lh"].to_numpy(), tr["la"].to_numpy(), tr["hg"].to_numpy(), tr["ag"].to_numpy())
        params[int(season)] = theta
        m = FAMILIES[family][0](te["lh"].to_numpy(), te["la"].to_numpy(), theta)
        p = outcome(m)
        o = te[["match_id", "competition", "season", "date", "hg", "ag", "status", "lh", "la"]].copy()
        o[["p_home", "p_draw", "p_away"]] = p
        fin = (te["status"] == "finished").to_numpy()
        o["score_nll"] = np.nan
        o.loc[fin, "score_nll"] = score_nll(m[fin], te["hg"].to_numpy()[fin], te["ag"].to_numpy()[fin])
        outs.append(o)
    return pd.concat(outs, ignore_index=True), params


def _losses(o):
    f = o[o["status"] == "finished"]
    y = metrics.outcome_index(f["hg"], f["ag"])
    p = f[["p_home", "p_draw", "p_away"]].to_numpy()
    return f, metrics.log_loss_rows(p, y), metrics.rps_rows(p, y)


def gap_table(o):
    f = o[o["status"] == "finished"].copy()
    f["band"] = pd.cut(f["lh"] - f["la"], [-9, -1, -0.5, -0.2, 0.2, 0.5, 1, 1.5, 9]).astype(str)
    s = np.sign(f["hg"] - f["ag"])
    f["act_home"], f["act_draw"] = (s > 0).astype(float), (s == 0).astype(float)
    g = f.groupby("band", sort=False)
    return [{"band": b, "n": int(len(x)), "p_home": float(x["p_home"].mean()), "act_home": float(x["act_home"].mean()),
             "p_draw": float(x["p_draw"].mean()), "act_draw": float(x["act_draw"].mean())} for b, x in g]


def compare(base, cand, seasons, competitions):
    ids = base[base["season"].isin(seasons) & base["competition"].isin(competitions) & (base["status"] == "finished")]["match_id"]
    b = base.set_index("match_id").loc[ids].reset_index()
    c = cand.set_index("match_id").loc[ids].reset_index()
    fb, lb, rb = _losses(b)
    fc, lc, rc = _losses(c)
    block = (fb["competition"] + fb["season"].astype(str)).to_numpy()
    out = {"n": int(len(ids)), "log_loss": float(lc.mean()), "base_log_loss": float(lb.mean()),
           "log_loss_vs_base": metrics.paired_bootstrap(lb, lc, block=block),
           "rps": float(rc.mean()), "base_rps": float(rb.mean()),
           "score_nll": float(fc["score_nll"].mean()), "base_score_nll": float(fb["score_nll"].mean()),
           "score_vs_base": metrics.paired_bootstrap(fb["score_nll"].to_numpy(), fc["score_nll"].to_numpy(), block=block),
           "by_league": {}}
    for comp in competitions:
        k = (fb["competition"] == comp).to_numpy()
        if k.sum() > 300:
            out["by_league"][comp] = metrics.paired_bootstrap(lb[k], lc[k], block=block[k])
    return out


def study(con, families=("bivariate_poisson", "com_poisson", "gap_rho", "diagonal_gap"), confirm=True):
    oos = research.load_cached("oos_ensemble")
    base, base_params = walk_forward(oos, "dixon_coles")
    val = [s for s in VALIDATION_SEASONS if s >= 2013]
    results = db.get_kv(con, "shape_study") or {}
    results.setdefault("candidates", {})
    results["baseline"] = {"family": "dixon_coles", "params": base_params, "gap_table_validation": gap_table(base[base["season"].isin(val)])}
    for fam in families:
        cand, params = walk_forward(oos, fam)
        v = compare(base, cand, val, PRIMARY_COMPETITIONS)
        v_core = compare(base, cand, val, CORE_COMPETITIONS)
        ll, sc = v["log_loss_vs_base"], v["score_vs_base"]
        worse = [c for c, r in v["by_league"].items() if r["ci_high"] < 0]
        passed = ll["ci_low"] > 0 and sc["mean_diff"] > 0 and not worse
        entry = {"params": params, "validation_primary": v, "validation_core": v_core, "passed_validation": passed, "worse_leagues": worse,
                 "gap_table_validation": gap_table(cand[cand["season"].isin(val)])}
        if passed and confirm:
            entry["confirmation_2023_on"] = compare(base, cand, [2023, 2024, 2025], PRIMARY_COMPETITIONS)
        research.save_cached(f"oos_shape_{fam}", cand)
        results["candidates"][fam] = entry
    results["data_version"] = research.data_version(con)
    db.set_kv(con, "shape_study", results)
    return results


@handler("shape_study")
def shape_study(ctx, families=None):
    ctx.stage("fitting score-distribution families walk-forward on the champion's expected goals")
    res = study(ctx.con, **({"families": families} if families else {}))
    return {f: {"passed": e["passed_validation"], "log_loss_diff": e["validation_primary"]["log_loss_vs_base"]["mean_diff"],
                "score_diff": e["validation_primary"]["score_vs_base"]["mean_diff"]} for f, e in res["candidates"].items()}
