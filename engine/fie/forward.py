import json
from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import backtest, db, metrics, research
from .config import CURRENT_SEASON, PRIMARY_COMPETITIONS
from .scoreline import dixon_coles_matrix, markets


def _merge(old, new):
    if old is None:
        return new
    keep = old[~old["match_id"].isin(new["match_id"])]
    return pd.concat([keep, new], ignore_index=True)


def refresh_current(ctx, lookback_days=21):
    from .tasks import ELO_CONFIGS, STRENGTH_CONFIGS
    con = ctx.con
    summary = db.get_kv(con, "research_summary")
    weights = db.get_kv(con, "ensemble_weights")
    if not summary or not weights:
        raise RuntimeError("run research_full first")
    ctx.stage("loading matches")
    frame, coef = research.prepared_frame(con)
    research.save_cached("frame", frame)
    start = pd.Timestamp(date.today() - timedelta(days=lookback_days))
    comps = weights["components"]
    oos = {}
    for name in [n for n in comps if n in STRENGTH_CONFIGS] + [summary["best"]["dc"]]:
        ctx.stage(f"refreshing {name} from {start.date()}")
        new = backtest.walk_forward_strength(frame, name, start=start, **{**research.STRENGTH_DEFAULTS, **STRENGTH_CONFIGS[name]})
        new = backtest.attach(new, frame)
        oos[name] = _merge(research.load_cached(name), new)
        research.save_cached(name, oos[name])
    best_elo = summary["best"]["elo"]
    ctx.stage(f"refreshing {best_elo}")
    elo_out, ratings = research.run_elo(frame, name=best_elo, **ELO_CONFIGS[best_elo])
    research.save_cached(best_elo, elo_out)
    research.save_cached(best_elo + "_ratings", ratings)
    oos[best_elo] = elo_out
    ctx.stage("features and LightGBM for current season")
    features = research.feature_table(frame, ratings, oos[summary["best"]["dc"]])
    research.save_cached("features", features)
    gbm_new = research.run_gbm(features, oos[summary["best"]["dc"]], name="gbm_features", use_dc=False, first_season=CURRENT_SEASON)
    oos["gbm_features"] = _merge(research.load_cached("oos_gbm_features"), gbm_new)
    research.save_cached("oos_gbm_features", oos["gbm_features"])
    ctx.stage("ensemble for current season")
    w = weights["by_season"].get(str(CURRENT_SEASON))
    if w is None:
        ens, fitted = research.run_ensemble([oos[n] for n in comps], comps, frame, first_season=CURRENT_SEASON)
        weights["by_season"][str(CURRENT_SEASON)] = fitted[CURRENT_SEASON]
        db.set_kv(con, "ensemble_weights", weights)
    else:
        ens = apply_ensemble(oos, comps, w, frame, frame[frame["season"] == CURRENT_SEASON]["match_id"])
    oos["ensemble"] = _merge(research.load_cached("oos_ensemble"), ens)
    research.save_cached("oos_ensemble", oos["ensemble"])
    return oos


def apply_ensemble(oos, comps, w, frame, match_ids):
    keyed = [oos[n].set_index("match_id") for n in comps]
    ids = pd.Index(match_ids)
    for k in keyed:
        ids = ids.intersection(k.index)
    lh, la = research.combine_rates(keyed, ids, {"weights": [w["weights"][n] for n in comps], "scale": w["scale"], "spread": w.get("spread", 1.0)})
    cut = keyed[0].loc[ids, "fit_cutoff"].to_numpy()
    o = pd.DataFrame({"match_id": ids.to_numpy(), "lh": lh, "la": la, "rho": w["rho"], "both_known": True, "fit_cutoff": cut})
    return backtest.attach(backtest.finalise(o, frame, "ensemble"), frame)


def champion_oos(con):
    summary = db.get_kv(con, "research_summary")
    champ = summary["selection"]["champion"]
    return champ, research.load_cached(f"oos_{champ}") if research.load_cached(f"oos_{champ}") is not None else research.load_cached(champ)


def _explain_payload(match_id, components):
    if not components:
        return {}
    comps, w, oos = components
    out = {"components": []}
    for n in comps:
        frame = oos.get(n)
        if frame is None:
            continue
        row = frame[frame["match_id"] == match_id]
        if row.empty:
            continue
        r = row.iloc[0]
        out["components"].append({"model": n, "lh": float(r["lh"]), "la": float(r["la"]), "weight": float(w["weights"][n])})
        if n == "gbm_features" and "contrib_h" in row.columns and isinstance(r.get("contrib_h"), str):
            out["gbm_contrib"] = {"home": json.loads(r["contrib_h"]), "away": json.loads(r["contrib_a"])}
    return out


FORWARD_HORIZON_DAYS = 10
KICKOFF_SAFETY_HOURS = 3.0


def kickoff_utc_upper_bound(match_date, kickoff):
    from datetime import datetime, timezone
    hhmm = (kickoff or "00:00").strip()[:5]
    try:
        local = datetime.fromisoformat(f"{match_date}T{hhmm}")
    except ValueError:
        local = datetime.fromisoformat(f"{match_date}T00:00")
    return (local - timedelta(hours=KICKOFF_SAFETY_HOURS)).replace(tzinfo=timezone.utc)


def safely_before_kickoff(match_date, kickoff):
    from datetime import datetime, timezone
    return datetime.now(timezone.utc) < kickoff_utc_upper_bound(match_date, kickoff if kickoff else "00:00")


def flag_late_forward_predictions(con):
    from datetime import datetime
    rows = con.execute("""SELECT p.prediction_id, p.created_at, m.match_date, m.kickoff FROM predictions p JOIN matches m ON m.match_id=p.match_id
                          WHERE p.purpose='forward' AND p.prediction_id NOT IN (SELECT prediction_id FROM prediction_flags WHERE flag='after_kickoff_possible')""").fetchall()
    flagged = 0
    for r in rows:
        created = datetime.fromisoformat(r["created_at"].replace("Z", "+00:00"))
        if created >= kickoff_utc_upper_bound(r["match_date"], r["kickoff"]):
            con.execute("INSERT OR IGNORE INTO prediction_flags VALUES(?,?,?,?)",
                        (r["prediction_id"], "after_kickoff_possible", "stored within 3 hours of or after the listed local kick-off; excluded from forward evaluation", db.now()))
            flagged += 1
    return flagged


def publish_forward(con, oos_champion, version, components=None, horizon_days=FORWARD_HORIZON_DAYS):
    today = pd.Timestamp(date.today())
    sched = oos_champion[(oos_champion["status"] == "scheduled") & (oos_champion["date"] >= today) & (oos_champion["date"] <= today + pd.Timedelta(days=horizon_days))]
    status = dict(con.execute("SELECT match_id, status FROM matches WHERE match_id IN (%s)" % ",".join(str(int(x)) for x in sched["match_id"]) or "0").fetchall()) if len(sched) else {}
    inserted = 0
    kickoffs = dict(con.execute("SELECT match_id, kickoff FROM matches WHERE match_id IN (%s)" % ",".join(str(int(x)) for x in sched["match_id"]) or "0").fetchall()) if len(sched) else {}
    for r in sched.itertuples():
        if status.get(r.match_id) != "scheduled":
            continue
        if not safely_before_kickoff(str(pd.Timestamp(r.date).date()), kickoffs.get(r.match_id)):
            continue
        exists = con.execute("SELECT 1 FROM predictions WHERE match_id=? AND model_version=? AND purpose='forward'", (int(r.match_id), version)).fetchone()
        if exists:
            continue
        m = dixon_coles_matrix(r.lh, r.la, r.rho)
        mk = markets(m)
        mk["rho"] = float(r.rho)
        mk.update(_explain_payload(r.match_id, components))
        con.execute("INSERT INTO predictions(match_id, model_version, purpose, created_at, data_cutoff, p_home, p_draw, p_away, lambda_home, lambda_away, markets) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (int(r.match_id), version, "forward", db.now(), str(pd.Timestamp(r.fit_cutoff).date()), mk["p_home"], mk["p_draw"], mk["p_away"], float(r.lh), float(r.la), db.dumps(mk)))
        inserted += 1
    return inserted


def publish_backtest(con, oos, version, since_season=2012):
    d = oos[(oos["status"] == "finished") & (oos["season"] >= since_season)]
    existing = {r[0] for r in con.execute("SELECT match_id FROM predictions WHERE model_version=? AND purpose='backtest'", (version,))}
    rows = []
    for r in d.itertuples():
        if r.match_id in existing:
            continue
        mk = {"rho": None if pd.isna(r.rho) else float(r.rho), "over25": None if pd.isna(r.over25) else float(r.over25)}
        rows.append((int(r.match_id), version, "backtest", db.now(), str(pd.Timestamp(r.fit_cutoff).date()) if hasattr(r, "fit_cutoff") and not pd.isna(r.fit_cutoff) else f"{r.season}-07-01",
                     float(r.p_home), float(r.p_draw), float(r.p_away), None if pd.isna(r.lh) else float(r.lh), None if pd.isna(r.la) else float(r.la), db.dumps(mk)))
    with db.transaction(con):
        con.executemany("INSERT OR IGNORE INTO predictions(match_id, model_version, purpose, created_at, data_cutoff, p_home, p_draw, p_away, lambda_home, lambda_away, markets) VALUES(?,?,?,?,?,?,?,?,?,?,?)", rows)
    return len(rows)


def team_strengths(con, frame, summary):
    from .ratings import StrengthModel
    from .tasks import STRENGTH_CONFIGS
    name = summary["best"]["dc"]
    out = {}
    as_of = pd.Timestamp(date.today())
    finished = frame[frame["status"] == "finished"]
    for country in sorted(frame["country"].unique()):
        f = finished[finished["country"] == country]
        model = StrengthModel(**{**research.STRENGTH_DEFAULTS, **STRENGTH_CONFIGS[name]}).fit(f, as_of)
        names = dict(zip(frame["home"], frame["home_name"]))
        current = set(frame.loc[(frame["country"] == country) & (frame["season"] == CURRENT_SEASON), "home"])
        a, d, mu, h, rho = model.params
        rows = []
        for team, i in model.team_index.items():
            if team in current:
                rows.append({"team_id": int(team), "name": names.get(team, str(team)), "attack": float(a[i]), "defence": float(d[i])})
        out[country] = {"teams": sorted(rows, key=lambda r: -(r["attack"] + r["defence"])), "home_adv": [float(x) for x in h], "mu": [float(x) for x in mu],
                        "rho": float(rho), "fitted_at": db.now(), "n_matches": model.fit_info["n_matches"]}
    db.set_kv(con, "team_strengths", out)
    return {k: len(v["teams"]) for k, v in out.items()}


def evaluate_forward(con):
    flag_late_forward_predictions(con)
    rows = pd.read_sql_query("""SELECT p.match_id, p.model_version, p.p_home, p.p_draw, p.p_away, p.lambda_home, p.lambda_away, m.fthg, m.ftag, m.match_date, m.competition
                                FROM predictions p JOIN matches m ON m.match_id=p.match_id
                                WHERE p.purpose='forward' AND m.status='finished'
                                  AND p.prediction_id NOT IN (SELECT prediction_id FROM prediction_flags WHERE flag='after_kickoff_possible')""", con)
    out = {"n": int(len(rows))}
    if rows.empty:
        db.set_kv(con, "forward_performance", out)
        return out
    for version, g in rows.groupby("model_version"):
        p = g[["p_home", "p_draw", "p_away"]].to_numpy()
        y = metrics.outcome_index(g["fthg"], g["ftag"])
        out[version] = metrics.summarise(p, y, g["lambda_home"], g["lambda_away"], g["fthg"].to_numpy(), g["ftag"].to_numpy())
    db.set_kv(con, "forward_performance", out)
    return out


def drift_check(con, window=300, threshold=0.03):
    summary = db.get_kv(con, "research_summary")
    champ = summary["selection"]["champion"]
    oos = research.load_cached(f"oos_{champ}")
    d = oos[(oos["status"] == "finished") & (oos["competition"].isin(PRIMARY_COMPETITIONS))].sort_values("date")
    p = d[["p_home", "p_draw", "p_away"]].to_numpy()
    y = metrics.outcome_index(d["hg"], d["ag"])
    loss = metrics.log_loss_rows(p, y)
    val = summary["results"][champ]["log_loss"]
    recent = loss[-window:]
    rolling = pd.Series(loss).rolling(window).mean()
    mean_recent = float(recent.mean())
    se = float(recent.std(ddof=1) / np.sqrt(len(recent)))
    alarm = mean_recent - val > max(threshold, 2 * se)
    result = {"champion": champ, "window": window, "recent_log_loss": mean_recent, "recent_se": se, "validation_log_loss": val,
              "alarm": bool(alarm), "last_date": str(d["date"].max().date()),
              "rolling": [{"date": str(dt.date()), "log_loss": float(v)} for dt, v in zip(d["date"].iloc[window::max(1, window // 3)], rolling.iloc[window::max(1, window // 3)])]}
    db.set_kv(con, "drift", result)
    if alarm:
        db.notify(con, "warning", "Model drift alarm", f"{champ} recent log loss {mean_recent:.4f} vs validation {val:.4f}; challenger review recommended")
    return result
