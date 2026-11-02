import math

import numpy as np
import pandas as pd

from . import db, metrics
from .inplay import InPlayModel, extrapolation_forecast, state_features, time_bin

CHECKPOINTS = [(1, 15.0), (1, 30.0), (2, 45.0), (2, 60.0), (2, 75.0), (2, 85.0)]
RED_CARDS = ("Red Card", "Second Yellow")


def load_sb(con, prematch):
    matches = pd.read_sql_query("SELECT * FROM sb_matches WHERE processed=1 AND match_id IS NOT NULL", con)
    matches = matches.merge(prematch, on="match_id", how="inner")
    events = pd.read_sql_query("SELECT sb_match_id, period, clock, kind, side, xg, detail FROM sb_events WHERE kind IN ('goal','card','shot')", con)
    return matches, events.groupby("sb_match_id")


def match_timeline(ev):
    ev = ev[ev["side"].isin(["home", "away"])]
    goals = ev[ev["kind"] == "goal"].sort_values(["period", "clock"])
    reds = ev[(ev["kind"] == "card") & (ev["detail"].isin(RED_CARDS))]
    shots = ev[ev["kind"] == "shot"]
    return goals, reds, shots


def state_at(goals, reds, shots, period, clock, lam_h, lam_a, minutes_per_match):
    def before(df):
        return df[(df["period"] < period) | ((df["period"] == period) & (df["clock"] < clock))]

    g = before(goals)
    r = before(reds)
    s = before(shots)
    elapsed_play = min(clock, 47) if period == 1 else (47 + clock - 45)
    recent = s[(s["period"] == period) & (s["clock"] >= clock - 10)]
    return {
        "period": period, "clock": clock,
        "home_goals": int((g["side"] == "home").sum()), "away_goals": int((g["side"] == "away").sum()),
        "reds_home": int((r["side"] == "home").sum()), "reds_away": int((r["side"] == "away").sum()),
        "xg_home": float(s.loc[s["side"] == "home", "xg"].sum()), "xg_away": float(s.loc[s["side"] == "away", "xg"].sum()),
        "exp_home_so_far": lam_h * elapsed_play / minutes_per_match, "exp_away_so_far": lam_a * elapsed_play / minutes_per_match,
        "recent_xg_home": float(recent.loc[recent["side"] == "home", "xg"].sum()), "recent_xg_away": float(recent.loc[recent["side"] == "away", "xg"].sum()),
    }


def build_panel(matches, grouped, minutes_per_match):
    rows = []
    for m in matches.itertuples():
        if m.sb_match_id not in grouped.groups:
            continue
        ev = grouped.get_group(m.sb_match_id)
        goals, reds, shots = match_timeline(ev)
        ends = {1: m.first_half_end or 47.0, 2: m.second_half_end or 94.5}
        shot_arr = {side: shots[shots["side"] == side][["period", "clock", "xg"]].to_numpy() for side in ("home", "away")}
        goal_arr = goals[["period", "clock", "side"]].to_numpy()
        red_arr = reds[["period", "clock", "side"]].to_numpy()
        for period in (1, 2):
            t = 0.0 if period == 1 else 45.0
            while t < ends[period] - 1e-9:
                dt = min(1.0, ends[period] - t)
                prior = lambda arr: arr[(arr[:, 0] < period) | ((arr[:, 0] == period) & (arr[:, 1] < t))] if len(arr) else arr
                g_prev = prior(goal_arr)
                r_prev = prior(red_arr)
                hg = int((g_prev[:, 2] == "home").sum()) if len(g_prev) else 0
                ag = int((g_prev[:, 2] == "away").sum()) if len(g_prev) else 0
                rh = int((r_prev[:, 2] == "home").sum()) if len(r_prev) else 0
                ra = int((r_prev[:, 2] == "away").sum()) if len(r_prev) else 0
                elapsed_play = min(t, 47) if period == 1 else 47 + t - 45
                in_slot = goal_arr[(goal_arr[:, 0] == period) & (goal_arr[:, 1] >= t) & (goal_arr[:, 1] < t + dt)] if len(goal_arr) else goal_arr
                for side, lam, diff, own, opp in (("home", m.lh, hg - ag, rh, ra), ("away", m.la, ag - hg, ra, rh)):
                    sa = shot_arr[side]
                    sp = prior(sa)
                    xg_so_far = float(sp[:, 2].sum()) if len(sp) else 0.0
                    rec = sp[(sp[:, 0] == period) & (sp[:, 1] >= t - 10)] if len(sp) else sp
                    f = state_features(diff, t >= 70, own, opp, xg_so_far, lam * elapsed_play / minutes_per_match,
                                       float(rec[:, 2].sum()) if len(rec) else 0.0, use_evidence=True)
                    goals_in = int((in_slot[:, 2] == side).sum()) if len(in_slot) else 0
                    rows.append({"sb_match_id": m.sb_match_id, "date": m.match_date, "bin": time_bin(period, t), "f": f, "goals": goals_in,
                                 "lambda_pm": lam / minutes_per_match * dt})
                t += dt
    return {"rows": rows}


def checkpoint_eval(models, matches, grouped, minutes_per_match, half_end):
    recs = []
    for m in matches.itertuples():
        if m.sb_match_id not in grouped.groups:
            continue
        ev = grouped.get_group(m.sb_match_id)
        goals, reds, shots = match_timeline(ev)
        fh = int((goals["side"] == "home").sum())
        fa = int((goals["side"] == "away").sum())
        y = 0 if fh > fa else 1 if fh == fa else 2
        for period, clock in CHECKPOINTS:
            st = state_at(goals, reds, shots, period, clock, m.lh, m.la, minutes_per_match)
            more = int(fh + fa > st["home_goals"] + st["away_goals"])
            for name, model in models.items():
                if name == "extrapolation":
                    fc = extrapolation_forecast(st, m.lh, m.la, half_end)
                else:
                    fc = model.forecast(st, m.lh / minutes_per_match, m.la / minutes_per_match, half_end)
                recs.append({"sb_match_id": m.sb_match_id, "model": name, "checkpoint": f"{'1H' if period == 1 else '2H'} {int(clock)}'",
                             "p_home": fc["p_home"], "p_draw": fc["p_draw"], "p_away": fc["p_away"], "y": y,
                             "p_more": fc["p_another_goal"], "more": more, "final_total": fh + fa})
    return pd.DataFrame(recs)


def summarise_checkpoints(df):
    out = {}
    for (model, cp), g in df.groupby(["model", "checkpoint"]):
        p = g[["p_home", "p_draw", "p_away"]].to_numpy()
        y = g["y"].to_numpy()
        pm = np.clip(g["p_more"].to_numpy(), 1e-9, 1 - 1e-9)
        hit = g["more"].to_numpy()
        out.setdefault(model, {})[cp] = {
            "n": int(len(g)), "log_loss_1x2": float(metrics.log_loss_rows(p, y).mean()), "brier_1x2": float(metrics.brier_rows(p, y).mean()),
            "another_goal_log_loss": float(-(hit * np.log(pm) + (1 - hit) * np.log(1 - pm)).mean()),
            "another_goal_brier": float(((pm - hit) ** 2).mean()), "another_goal_ece": metrics.ece(pm, hit),
            "mean_p_more": float(pm.mean()), "freq_more": float(hit.mean()),
        }
    return out


def panel_loglik(model, panel):
    x = model.design(panel["rows"])
    y = np.array([r["goals"] for r in panel["rows"]], float)
    off = np.array([math.log(r["lambda_pm"]) for r in panel["rows"]])
    eta = x @ model.beta + off
    return float((y * eta - np.exp(eta)).sum() / max(len(set(r["sb_match_id"] for r in panel["rows"])), 1))


def train_and_evaluate(con, prematch, split_date="2016-02-01", progress=None):
    matches, grouped = load_sb(con, prematch)
    halves = matches[["first_half_end", "second_half_end"]].dropna()
    half_end = (float(halves["first_half_end"].mean()), float(halves["second_half_end"].mean()))
    minutes_per_match = half_end[0] + (half_end[1] - 45)
    train_m = matches[matches["match_date"] < split_date]
    test_m = matches[matches["match_date"] >= split_date]
    if progress:
        progress(1, 6, "building training panel")
    train_panel = build_panel(train_m, grouped, minutes_per_match)
    test_panel = build_panel(test_m, grouped, minutes_per_match)
    if progress:
        progress(2, 6, "fitting hazard models")
    hazard = InPlayModel(use_evidence=False).fit(train_panel, half_end)
    evidence = InPlayModel(use_evidence=True).fit(train_panel, half_end)
    constant = InPlayModel(use_evidence=False, mode="constant")
    constant.columns, constant.beta, constant.se, constant.half_end = hazard.columns, np.zeros(len(hazard.columns)), np.zeros(len(hazard.columns)), half_end
    panel_ll = {"hazard": panel_loglik(hazard, test_panel), "hazard_evidence": panel_loglik(evidence, test_panel), "constant": panel_loglik(constant, test_panel)}
    if progress:
        progress(3, 6, "checkpoint evaluation on held-out matches")
    models = {"constant_rate": constant, "hazard": hazard, "hazard_evidence": evidence, "extrapolation": None}
    cp = checkpoint_eval(models, test_m, grouped, minutes_per_match, half_end)
    summary = summarise_checkpoints(cp)
    pm = cp[cp["model"].isin(["hazard", "hazard_evidence"])]
    rel = {name: metrics.reliability(g["p_more"].to_numpy(), g["more"].to_numpy()) for name, g in pm.groupby("model")}
    paired = {}
    for a, b in (("hazard", "hazard_evidence"), ("constant_rate", "hazard"), ("extrapolation", "hazard")):
        la = cp[cp["model"] == a].reset_index(drop=True)
        lb = cp[cp["model"] == b].reset_index(drop=True)
        loss_a = metrics.log_loss_rows(la[["p_home", "p_draw", "p_away"]].to_numpy(), la["y"].to_numpy())
        loss_b = metrics.log_loss_rows(lb[["p_home", "p_draw", "p_away"]].to_numpy(), lb["y"].to_numpy())
        paired[f"{a}_vs_{b}"] = metrics.paired_bootstrap(loss_a, loss_b, block=la["sb_match_id"].to_numpy())
    if progress:
        progress(4, 6, "refitting on all matches")
    full_panel = build_panel(matches, grouped, minutes_per_match)
    better_evidence = paired["hazard_vs_hazard_evidence"]["ci_low"] > 0
    final = InPlayModel(use_evidence=better_evidence).fit(full_panel, half_end)
    return {
        "half_end": half_end, "minutes_per_match": minutes_per_match,
        "n_train_matches": int(len(train_m)), "n_test_matches": int(len(test_m)), "split_date": split_date,
        "panel_loglik_per_match": panel_ll, "checkpoints": summary, "reliability_another_goal": rel, "paired": paired,
        "coefficients": {"hazard": hazard.coefficients(), "hazard_evidence": evidence.coefficients()},
        "selected": "hazard_evidence" if better_evidence else "hazard",
        "final_model": final.to_json(), "final_coefficients": final.coefficients(),
    }


def halftime_eval(oos, inplay_model, minutes_per_match, half_end, seasons):
    d = oos[(oos["status"] == "finished") & oos["season"].isin(seasons)].dropna(subset=["lh"])
    con = db.connect()
    ht = pd.read_sql_query("SELECT match_id, hthg, htag, hr, ar FROM matches WHERE hthg IS NOT NULL", con)
    d = d.merge(ht, on="match_id")
    recs = []
    const = InPlayModel(mode="constant")
    const.columns, const.beta, const.se, const.half_end = inplay_model.columns, np.zeros(len(inplay_model.columns)), np.zeros(len(inplay_model.columns)), half_end
    for r in d.itertuples():
        st = {"period": 2, "clock": 45.0, "home_goals": int(r.hthg), "away_goals": int(r.htag), "reds_home": 0, "reds_away": 0,
              "xg_home": None, "xg_away": None, "exp_home_so_far": None, "exp_away_so_far": None, "recent_xg_home": None, "recent_xg_away": None}
        y = 0 if r.hg > r.ag else 1 if r.hg == r.ag else 2
        more = int(r.hg + r.ag > r.hthg + r.htag)
        for name, model in (("hazard", inplay_model), ("constant_rate", const)):
            fc = model.forecast(st, r.lh / minutes_per_match, r.la / minutes_per_match, half_end)
            recs.append({"match_id": r.match_id, "model": name, "p_home": fc["p_home"], "p_draw": fc["p_draw"], "p_away": fc["p_away"], "y": y,
                         "p_more": fc["p_another_goal"], "more": more, "competition": r.competition})
        fc = extrapolation_forecast(st, r.lh, r.la, half_end)
        recs.append({"match_id": r.match_id, "model": "extrapolation", "p_home": fc["p_home"], "p_draw": fc["p_draw"], "p_away": fc["p_away"], "y": y,
                     "p_more": fc["p_another_goal"], "more": more, "competition": r.competition})
    df = pd.DataFrame(recs)
    df["checkpoint"] = "HT"
    summary = summarise_checkpoints(df)
    paired = {}
    for a, b in (("constant_rate", "hazard"), ("extrapolation", "hazard")):
        la = df[df["model"] == a].reset_index(drop=True)
        lb = df[df["model"] == b].reset_index(drop=True)
        paired[f"{a}_vs_{b}"] = metrics.paired_bootstrap(metrics.log_loss_rows(la[["p_home", "p_draw", "p_away"]].to_numpy(), la["y"].to_numpy()),
                                                         metrics.log_loss_rows(lb[["p_home", "p_draw", "p_away"]].to_numpy(), lb["y"].to_numpy()))
    return {"summary": summary, "paired": paired, "n": int(len(d)), "seasons": seasons}
