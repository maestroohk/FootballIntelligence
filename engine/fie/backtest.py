import numpy as np
import pandas as pd

from . import metrics
from .config import FINAL_TEST_SEASONS, FORWARD_SEASON, VALIDATION_SEASONS
from .ratings import Elo, StrengthModel, matrix_probs, ordered_logit_fit, ordered_logit_predict, poisson_link_fit
from .scoreline import dixon_coles_matrix, outcome_probs, poisson_matrix

PREDICT_FROM = pd.Timestamp("2007-07-01")


def split_of(season):
    if season in FINAL_TEST_SEASONS or season == FORWARD_SEASON:
        return "final_test"
    if season in VALIDATION_SEASONS:
        return "validation"
    return "warmup"


def _score_cols(out, frame, rho_col="rho"):
    p = np.empty((len(out), 3))
    score_p = np.full(len(out), np.nan)
    over25 = np.empty(len(out))
    hg = frame["hg"].to_numpy()
    ag = frame["ag"].to_numpy()
    for k, (lh, la, rho) in enumerate(zip(out["lh"], out["la"], out[rho_col])):
        m = dixon_coles_matrix(lh, la, rho)
        p[k] = outcome_probs(m)
        tot = np.add.outer(np.arange(m.shape[0]), np.arange(m.shape[1]))
        over25[k] = m[tot >= 3].sum()
        if not np.isnan(hg[k]):
            h, a = int(hg[k]), int(ag[k])
            score_p[k] = m[h, a] if h < m.shape[0] and a < m.shape[1] else 1e-6
    out["p_home"], out["p_draw"], out["p_away"] = p[:, 0], p[:, 1], p[:, 2]
    out["score_p"] = score_p
    out["over25"] = over25
    return out


def blocks(dates, start, days=7):
    first = max(start, dates.min())
    first = first - pd.Timedelta(days=first.weekday())
    last = dates.max() + pd.Timedelta(days=1)
    return pd.date_range(first, last + pd.Timedelta(days=days), freq=f"{days}D")


def walk_forward_strength(frame, name, progress=None, start=PREDICT_FROM, block_days=7, **params):
    outs = []
    countries = sorted(frame["country"].unique())
    total = sum(len(blocks(frame.loc[frame["country"] == c, "date"], start, block_days)) for c in countries)
    done = 0
    for country in countries:
        f = frame[frame["country"] == country]
        finished = f[f["status"] == "finished"]
        model = StrengthModel(**params)
        edges = blocks(f["date"], start, block_days)
        for lo, hi in zip(edges[:-1], edges[1:]):
            done += 1
            target = f[(f["date"] >= lo) & (f["date"] < hi)]
            if target.empty:
                continue
            train = finished[finished["date"] < lo]
            assert train["date"].max() < target["date"].min()
            model.fit(train, lo)
            priors = model.tier_priors(train[train["date"] >= lo - pd.Timedelta(days=365)])
            lh, la, known, rho = model.rates(target["home"], target["away"], target["tier"], priors)
            outs.append(pd.DataFrame({"match_id": target["match_id"].to_numpy(), "lh": lh, "la": la, "rho": rho,
                                      "both_known": known, "fit_cutoff": lo}))
            if progress and done % 25 == 0:
                progress(done, total, f"{name} {country} {lo.date()}")
    out = pd.concat(outs, ignore_index=True)
    return finalise(out, frame, name)


def finalise(out, frame, name):
    base = frame.set_index("match_id").loc[out["match_id"]].reset_index()
    out = out.reset_index(drop=True)
    out = _score_cols(out, base)
    out["model"] = name
    return out


def walk_forward_elo(frame, name="elo", k=20.0, home=65.0, tier_gap=120.0, season_regress=0.1, progress=None):
    pre = {}
    for country in sorted(frame["country"].unique()):
        f = frame[frame["country"] == country]
        elo = Elo(k=k, home=home, tier_gap=tier_gap, season_regress=season_regress)
        dates = f["date"].to_numpy()
        rows = f.to_dict("records")
        i = 0
        while i < len(rows):
            j = i
            while j < len(rows) and dates[j] == dates[i]:
                j += 1
            for r in rows[i:j]:
                elo.season_start(r["home"], r["season"], r["tier"])
                elo.season_start(r["away"], r["season"], r["tier"])
                pre[r["match_id"]] = (elo.rating(r["home"], r["tier"]), elo.rating(r["away"], r["tier"]))
            for r in rows[i:j]:
                if r["status"] == "finished":
                    elo.update(r["home"], r["away"], r["hg"], r["ag"], r["tier"], r["season"])
            i = j
    ratings = pd.DataFrame([(m, h, a) for m, (h, a) in pre.items()], columns=["match_id", "elo_h", "elo_a"])
    g = frame.merge(ratings, on="match_id")
    g["elo_diff"] = g["elo_h"] - g["elo_a"] + home
    outs = []
    for season in sorted(g.loc[g["date"] >= PREDICT_FROM, "season"].unique()):
        train = g[(g["season"] < season) & (g["season"] >= season - 4) & (g["status"] == "finished")]
        target = g[g["season"] == season]
        y = metrics.outcome_index(train["hg"], train["ag"])
        theta = ordered_logit_fit(train["elo_diff"], y)
        ch = poisson_link_fit(train["elo_diff"], train["hg"])
        ca = poisson_link_fit(-train["elo_diff"], train["ag"])
        x = target["elo_diff"].to_numpy() / 400
        lh = np.exp(ch[0] + ch[1] * x)
        la = np.exp(ca[0] - ca[1] * x)
        o = pd.DataFrame({"match_id": target["match_id"].to_numpy(), "lh": lh, "la": la, "rho": 0.0, "both_known": True, "fit_cutoff": pd.Timestamp(f"{season}-07-01")})
        o = finalise(o, frame, name)
        p_direct = ordered_logit_predict(theta, target["elo_diff"].to_numpy())
        o["p_home"], o["p_draw"], o["p_away"] = p_direct[:, 0], p_direct[:, 1], p_direct[:, 2]
        outs.append(o)
    return pd.concat(outs, ignore_index=True), ratings


def walk_forward_frequency(frame, name="frequency"):
    outs = []
    for season in sorted(frame.loc[frame["date"] >= PREDICT_FROM, "season"].unique()):
        target = frame[frame["season"] == season]
        for comp, t in target.groupby("competition"):
            train = frame[(frame["competition"] == comp) & (frame["season"] < season) & (frame["season"] >= season - 5) & (frame["status"] == "finished")]
            if train.empty:
                continue
            y = metrics.outcome_index(train["hg"], train["ag"])
            freq = np.bincount(y, minlength=3) / len(y)
            o = pd.DataFrame({"match_id": t["match_id"].to_numpy(), "lh": train["hg"].mean(), "la": train["ag"].mean(), "rho": 0.0, "both_known": True, "fit_cutoff": pd.Timestamp(f"{season}-07-01")})
            o = finalise(o, frame, name)
            o["p_home"], o["p_draw"], o["p_away"] = freq
            outs.append(o)
    return pd.concat(outs, ignore_index=True)


def attach(oos, frame):
    cols = ["match_id", "competition", "season", "date", "hg", "ag", "status", "tier", "home", "away"]
    return oos.merge(frame[cols], on="match_id", how="left")


def evaluate(oos, split=None, competitions=None, seasons=None, extra_mask=None):
    d = oos[oos["status"] == "finished"].copy()
    d["split"] = d["season"].map(split_of)
    if split:
        d = d[d["split"] == split]
    if competitions:
        d = d[d["competition"].isin(competitions)]
    if seasons:
        d = d[d["season"].isin(seasons)]
    if extra_mask is not None:
        d = d[extra_mask(d)]
    if d.empty:
        return None
    p = d[["p_home", "p_draw", "p_away"]].to_numpy()
    y = metrics.outcome_index(d["hg"], d["ag"])
    btts = None
    if d["lh"].notna().all():
        btts = np.array([dixon_coles_matrix(a, b, r)[1:, 1:].sum() for a, b, r in zip(d["lh"], d["la"], d["rho"].fillna(0.0))])
    return metrics.summarise(p, y, d["lh"], d["la"], d["hg"].to_numpy(), d["ag"].to_numpy(), d["score_p"].to_numpy(), d["over25"].to_numpy(), btts)


def row_losses(oos):
    d = oos[oos["status"] == "finished"]
    p = d[["p_home", "p_draw", "p_away"]].to_numpy()
    y = metrics.outcome_index(d["hg"], d["ag"])
    return pd.DataFrame({"match_id": d["match_id"].to_numpy(), "season": d["season"].to_numpy(), "competition": d["competition"].to_numpy(),
                         "log_loss": metrics.log_loss_rows(p, y), "rps": metrics.rps_rows(p, y),
                         "score_nll": -np.log(np.clip(d["score_p"].to_numpy(), 1e-12, 1))})


def compare(oos_a, oos_b, split="validation", competitions=None, metric="log_loss"):
    la = row_losses(oos_a)
    lb = row_losses(oos_b)
    j = la.merge(lb, on=["match_id", "season", "competition"], suffixes=("_a", "_b"))
    j = j[j["season"].map(split_of) == split]
    if competitions:
        j = j[j["competition"].isin(competitions)]
    return metrics.paired_bootstrap(j[f"{metric}_a"], j[f"{metric}_b"], block=(j["competition"] + j["season"].astype(str)).to_numpy())


def full_report(oos, split, competitions):
    d = oos[(oos["status"] == "finished") & (oos["season"].map(split_of) == split) & (oos["competition"].isin(competitions))]
    if d.empty:
        return None
    p = d[["p_home", "p_draw", "p_away"]].to_numpy()
    y = metrics.outcome_index(d["hg"], d["ag"])
    overall = evaluate(d, competitions=competitions)
    by_league = {c: evaluate(g) for c, g in d.groupby("competition")}
    by_season = {int(s): evaluate(g) for s, g in d.groupby("season")}
    conf = d[["p_home", "p_draw", "p_away"]].max(axis=1)
    bands = pd.cut(conf, [0, 0.4, 0.5, 0.6, 0.7, 1.0])
    by_conf = {}
    for band, g in d.groupby(bands, observed=True):
        gp = g[["p_home", "p_draw", "p_away"]].to_numpy()
        gy = metrics.outcome_index(g["hg"], g["ag"])
        by_conf[str(band)] = {"n": int(len(g)), "mean_top_p": float(gp.max(axis=1).mean()), "top_hit_rate": float((gp.argmax(axis=1) == gy).mean()),
                              "log_loss": float(metrics.log_loss_rows(gp, gy).mean())}
    fav_home = d["p_home"] >= d["p_away"]
    by_fav = {"home_favourite": evaluate(d[fav_home]), "away_favourite": evaluate(d[~fav_home])}
    over = d["over25"].to_numpy()
    hit = ((d["hg"] + d["ag"]) > 2.5).to_numpy()
    return {
        "overall": overall, "by_league": by_league, "by_season": by_season, "by_confidence": by_conf, "by_favourite": by_fav,
        "reliability": metrics.reliability_all(p, y), "reliability_over25": metrics.reliability(over, hit),
        "seasons": [int(d["season"].min()), int(d["season"].max())],
        "dates": [str(d["date"].min().date()), str(d["date"].max().date())],
    }
