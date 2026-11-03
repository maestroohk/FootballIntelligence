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


MAX_UTC_OFFSET_HOURS = {"ENG": 1, "SCO": 1, "POR": 1, "ESP": 2, "GER": 2, "ITA": 2, "FRA": 2, "NED": 2, "BEL": 2, "GRE": 3, "TUR": 3}


def kickoff_utc_upper_bound(match_date, kickoff, competition=None):
    from datetime import datetime, timezone
    from .config import COMPETITIONS
    hhmm = (kickoff or "00:00").strip()[:5]
    try:
        local = datetime.fromisoformat(f"{match_date}T{hhmm}")
    except ValueError:
        local = datetime.fromisoformat(f"{match_date}T00:00")
    country = COMPETITIONS[competition][1] if competition in COMPETITIONS else None
    offset = MAX_UTC_OFFSET_HOURS.get(country, 3)
    return (local - timedelta(hours=KICKOFF_SAFETY_HOURS + offset)).replace(tzinfo=timezone.utc)


def safely_before_kickoff(match_date, kickoff, competition=None):
    from datetime import datetime, timezone
    return datetime.now(timezone.utc) < kickoff_utc_upper_bound(match_date, kickoff if kickoff else "00:00", competition)


def flag_late_forward_predictions(con):
    from datetime import datetime
    rows = con.execute("""SELECT p.prediction_id, p.created_at, m.match_date, m.kickoff, m.competition FROM predictions p JOIN matches m ON m.match_id=p.match_id
                          WHERE p.purpose='forward' AND p.prediction_id NOT IN (SELECT prediction_id FROM prediction_flags WHERE flag='after_kickoff_possible')""").fetchall()
    flagged = 0
    for r in rows:
        created = datetime.fromisoformat(r["created_at"].replace("Z", "+00:00"))
        if created >= kickoff_utc_upper_bound(r["match_date"], r["kickoff"], r["competition"]):
            con.execute("INSERT OR IGNORE INTO prediction_flags VALUES(?,?,?,?)",
                        (r["prediction_id"], "after_kickoff_possible", "stored within 3 hours of or after kick-off (listed local time, latest UTC offset for the country); excluded from forward evaluation", db.now()))
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
        if not safely_before_kickoff(str(pd.Timestamp(r.date).date()), kickoffs.get(r.match_id), r.competition):
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
        rows.append((int(r.match_id), version, "backtest", db.now(), str(pd.Timestamp(r.fit_cutoff).date()) if hasattr(r, "fit_cutoff") and not pd.isna(r.fit_cutoff) else f"{int(r.season)}-07-01",
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
        config = {**research.STRENGTH_DEFAULTS, **STRENGTH_CONFIGS[name]}
        f = finished[(finished["country"] == country) & (finished["date"] > as_of - pd.Timedelta(days=config["window_days"]))]
        if f.empty:
            continue
        model = StrengthModel(**config).fit(f, as_of)
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


def _finding(key, question, n, expected, observed, se, unit="share"):
    z = (observed - expected) / se if se > 0 else 0.0
    flagged = n >= 200 and abs(z) >= 2.5
    return {"key": key, "question": question, "n": int(n), "expected": float(expected), "observed": float(observed), "z": float(z), "flagged": bool(flagged), "unit": unit}


def weakness_report(con, min_n=200):
    d = pd.read_sql_query("""
        SELECT p.match_id, p.purpose, p.model_version, p.p_home, p.p_draw, p.p_away, p.lambda_home lh, p.lambda_away la,
               m.competition, m.season, m.match_date, m.home_team_id home, m.away_team_id away, m.fthg hg, m.ftag ag
        FROM predictions p JOIN matches m ON m.match_id=p.match_id
        WHERE m.status='finished' AND m.fthg IS NOT NULL
          AND p.prediction_id = (SELECT p2.prediction_id FROM predictions p2
                                 WHERE p2.match_id=p.match_id AND p2.model_version NOT LIKE 'bookmaker%'
                                   AND p2.prediction_id NOT IN (SELECT prediction_id FROM prediction_flags)
                                 ORDER BY p2.purpose='forward' DESC, p2.prediction_id DESC LIMIT 1)""", con)
    out = {"at": db.now(), "n": int(len(d)), "min_n": min_n, "findings": [], "by_league": [], "by_season": [], "sources": {}}
    if d.empty:
        db.set_kv(con, "weaknesses", out)
        return out
    out["sources"] = {k: int(v) for k, v in d["purpose"].value_counts().items()}
    d["date"] = pd.to_datetime(d["match_date"])
    y = metrics.outcome_index(d["hg"], d["ag"])
    p = d[["p_home", "p_draw", "p_away"]].to_numpy()
    d["ll"] = metrics.log_loss_rows(p, y)
    for i, (key, label) in enumerate((("home", "home wins"), ("draw", "draws"), ("away", "away wins"))):
        pred = p[:, i]
        obs = (y == i).astype(float)
        out["findings"].append(_finding(f"rate_{key}", f"Are {label} predicted at the right rate?", len(d), pred.mean(), obs.mean(), np.sqrt((pred * (1 - pred)).sum()) / len(d)))
    fav_home = d["p_home"] >= d["p_away"]
    fav_p = np.where(fav_home, d["p_home"], d["p_away"])
    fav_won = np.where(fav_home, y == 0, y == 2).astype(float)
    for lo, hi in ((0.5, 0.6), (0.6, 0.7), (0.7, 1.01)):
        for side, mask in (("home", fav_home), ("away", ~fav_home)):
            m = mask & (fav_p >= lo) & (fav_p < hi)
            if m.sum() == 0:
                continue
            fp = fav_p[m]
            out["findings"].append(_finding(f"fav_{side}_{int(lo * 100)}", f"Are {side} favourites at {int(lo * 100)}–{min(100, int(hi * 100))}% too confident?", m.sum(), fp.mean(), fav_won[m].mean(), np.sqrt((fp * (1 - fp)).sum()) / m.sum()))
    g = d.dropna(subset=["lh", "la"])
    if len(g):
        actual = (g["hg"] + g["ag"]).to_numpy(float)
        expected = (g["lh"] + g["la"]).to_numpy(float)
        out["findings"].append(_finding("goals_total", "Are total goals forecast too high or too low?", len(g), expected.mean(), actual.mean(), np.sqrt(expected.sum()) / len(g), "goals"))
    seasons = pd.read_sql_query("SELECT DISTINCT competition, season, home_team_id team FROM matches UNION SELECT DISTINCT competition, season, away_team_id FROM matches", con)
    present = set(zip(seasons["competition"], seasons["season"], seasons["team"]))
    known = set(zip(seasons["competition"], seasons["season"]))
    for side, team_col, own, opp in (("home", "home", "lh", "la"), ("away", "away", "la", "lh")):
        prev_known = [(c, s - 1) in known for c, s in zip(d["competition"], d["season"])]
        promoted = np.array([pk and (c, s - 1, t) not in present for pk, c, s, t in zip(prev_known, d["competition"], d["season"], d[team_col])])
        m = promoted & d[own].notna().to_numpy()
        if m.sum():
            goals = np.where(side == "home", d["hg"], d["ag"])[m].astype(float)
            exp = d[own].to_numpy(float)[m]
            out["findings"].append(_finding(f"promoted_{side}", f"Are newly promoted teams' goals forecast correctly when playing {side}?", m.sum(), exp.mean(), goals.mean(), np.sqrt(exp.sum()) / m.sum(), "goals"))
    long = pd.concat([d[["match_id", "date", "home"]].rename(columns={"home": "team"}), d[["match_id", "date", "away"]].rename(columns={"away": "team"})])
    allm = pd.read_sql_query("SELECT match_date, home_team_id team FROM matches WHERE status='finished' UNION ALL SELECT match_date, away_team_id FROM matches WHERE status='finished'", con)
    allm["date"] = pd.to_datetime(allm["match_date"])
    allm = allm.sort_values("date")
    allm["prev"] = allm.groupby("team")["date"].shift()
    gaps = long.merge(allm[["team", "date", "prev"]], on=["team", "date"], how="left")
    gaps["gap"] = (gaps["date"] - gaps["prev"]).dt.days
    rested = gaps[(gaps["gap"] >= 13) & (gaps["gap"] <= 21)].groupby("match_id").size()
    m = d["match_id"].isin(rested.index).to_numpy()
    if m.sum() and (~m).sum():
        a, b = d["ll"][m], d["ll"][~m]
        out["findings"].append(_finding("after_break", "Is the model worse after a 13–21 day break (e.g. international breaks)?", m.sum(), b.mean(), a.mean(), np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)), "log loss"))
    for c, grp in d.groupby("competition"):
        out["by_league"].append({"competition": c, "n": int(len(grp)), "log_loss": float(grp["ll"].mean())})
    for s, grp in d.groupby("season"):
        out["by_season"].append({"season": int(s), "n": int(len(grp)), "log_loss": float(grp["ll"].mean()), "top_hit": float((grp[["p_home", "p_draw", "p_away"]].to_numpy().argmax(axis=1) == metrics.outcome_index(grp["hg"], grp["ag"])).mean())})
    flagged = [f for f in out["findings"] if f["flagged"]]
    prev = db.get_kv(con, "weaknesses")
    prev_flags = {f["key"] for f in (prev or {}).get("findings", []) if f.get("flagged")}
    new = [f for f in flagged if f["key"] not in prev_flags]
    db.set_kv(con, "weaknesses", out)
    if new:
        db.notify(con, "warning", "Systematic weakness detected", "; ".join(f["question"] for f in new[:3]))
    return out


def coverage_report(con, oos_champion, version, horizon_days=FORWARD_HORIZON_DAYS):
    today = date.today().isoformat()
    until = (date.today() + timedelta(days=horizon_days)).isoformat()
    produced = set(oos_champion.loc[oos_champion["status"] == "scheduled", "match_id"].astype(int))
    rows = con.execute("""SELECT m.match_id, m.competition, m.match_date, m.kickoff, th.name home, ta.name away,
                                 (SELECT COUNT(*) FROM matches h WHERE h.status = 'finished' AND m.home_team_id IN (h.home_team_id, h.away_team_id)) home_n,
                                 (SELECT COUNT(*) FROM matches a WHERE a.status = 'finished' AND m.away_team_id IN (a.home_team_id, a.away_team_id)) away_n,
                                 EXISTS (SELECT 1 FROM predictions p WHERE p.match_id = m.match_id AND p.purpose = 'forward'
                                         AND p.prediction_id NOT IN (SELECT prediction_id FROM prediction_flags)) predicted
                          FROM matches m JOIN teams th ON th.team_id = m.home_team_id JOIN teams ta ON ta.team_id = m.away_team_id
                          WHERE m.status = 'scheduled' AND m.match_date BETWEEN ? AND ?""", (today, until)).fetchall()
    missing, leagues = [], {}
    for r in rows:
        lg = leagues.setdefault(r["competition"], {"fixtures": 0, "predicted": 0})
        lg["fixtures"] += 1
        if r["predicted"]:
            lg["predicted"] += 1
            continue
        if r["home_n"] == 0 or r["away_n"] == 0:
            reason = f"no match history for {r['home'] if r['home_n'] == 0 else r['away']}"
        elif not safely_before_kickoff(r["match_date"], r["kickoff"], r["competition"]):
            reason = "too close to kick-off for a pre-match forecast"
        elif r["match_id"] not in produced:
            reason = "not produced by the last model run; it will be included in the next update cycle"
        else:
            reason = "waiting for the next prediction run"
        missing.append({"match_id": r["match_id"], "competition": r["competition"], "date": r["match_date"], "kickoff": r["kickoff"],
                        "home": r["home"], "away": r["away"], "reason": reason})
    later = con.execute("SELECT COUNT(*) FROM matches WHERE status = 'scheduled' AND match_date > ?", (until,)).fetchone()[0]
    out = {"generated_at": db.now(), "model_version": version, "horizon_days": horizon_days, "fixtures_in_window": len(rows),
           "predicted": sum(1 for r in rows if r["predicted"]), "missing": missing, "leagues": leagues, "later_fixtures": later}
    db.set_kv(con, "prediction_coverage", out)
    return out
