from collections import defaultdict, deque

import numpy as np
import pandas as pd

from .config import COMPETITIONS


def load_matches(con, include_scheduled=False):
    status = "('finished','scheduled')" if include_scheduled else "('finished')"
    frame = pd.read_sql_query(
        f"""SELECT m.match_id, m.competition, m.season, m.match_date, m.kickoff, m.status,
                  m.home_team_id home, m.away_team_id away, th.name home_name, ta.name away_name,
                  m.fthg hg, m.ftag ag, m.hthg, m.htag, m.hs, m.aws, m.hst, m.ast, m.hc, m.ac,
                  m.hf, m.af, m.hy, m.ay, m.hr, m.ar
           FROM matches m JOIN teams th ON th.team_id=m.home_team_id JOIN teams ta ON ta.team_id=m.away_team_id
           WHERE m.status IN {status}""",
        con,
    )
    frame["date"] = pd.to_datetime(frame["match_date"])
    frame["country"] = frame["competition"].map(lambda c: COMPETITIONS[c][1])
    frame["tier"] = frame["competition"].map(lambda c: COMPETITIONS[c][2])
    frame = frame.sort_values(["date", "match_id"]).reset_index(drop=True)
    return frame


def fit_xg_proxy(frame):
    f = frame.dropna(subset=["hst", "ast", "hs", "aws"])
    sot = np.concatenate([f["hst"], f["ast"]]).astype(float)
    off = np.concatenate([f["hs"] - f["hst"], f["aws"] - f["ast"]]).astype(float)
    goals = np.concatenate([f["hg"], f["ag"]]).astype(float)
    off = np.clip(off, 0, None)
    x = np.column_stack([sot, off])
    coef, *_ = np.linalg.lstsq(x, goals, rcond=None)
    return {"sot": float(coef[0]), "off_target": float(coef[1]), "n": int(len(f))}


def apply_xg_proxy(frame, coef):
    frame = frame.copy()
    off_h = np.clip(frame["hs"] - frame["hst"], 0, None)
    off_a = np.clip(frame["aws"] - frame["ast"], 0, None)
    frame["xgp_h"] = coef["sot"] * frame["hst"] + coef["off_target"] * off_h
    frame["xgp_a"] = coef["sot"] * frame["ast"] + coef["off_target"] * off_a
    return frame


def team_form_features(frame, half_life=8.0):
    alpha = 1 - 0.5 ** (1 / half_life)
    state = defaultdict(lambda: {"n": 0})
    last_date = {}
    recent = defaultdict(deque)
    season_tier = {}
    keys = ["gf", "ga", "sf", "sa", "tf", "ta", "xf", "xa"]
    cols = defaultdict(list)
    dates = frame["date"].to_numpy()
    rows = frame.to_dict("records")
    i = 0
    n = len(rows)
    while i < n:
        j = i
        while j < n and dates[j] == dates[i]:
            j += 1
        block = rows[i:j]
        for r in block:
            for side, team, opp in (("h", r["home"], r["away"]), ("a", r["away"], r["home"])):
                s = state[team]
                for k in keys:
                    cols[f"{side}_{k}"].append(s.get(k, np.nan))
                cols[f"{side}_n"].append(s["n"])
                ld = last_date.get(team)
                cols[f"{side}_rest"].append((r["date"] - ld).days if ld is not None else np.nan)
                q = recent[team]
                while q and (r["date"] - q[0]).days > 28:
                    q.popleft()
                cols[f"{side}_m28"].append(len(q))
                prev = season_tier.get((team, r["season"] - 1))
                cols[f"{side}_promoted"].append(1.0 if prev is not None and prev > r["tier"] else 0.0 if prev is not None else np.nan)
        for r in block:
            if r.get("status") != "finished" or pd.isna(r["hg"]):
                continue
            for team, gf, ga, sf, sa, tf, ta, xf, xa in (
                (r["home"], r["hg"], r["ag"], r["hs"], r["aws"], r["hst"], r["ast"], r.get("xgp_h"), r.get("xgp_a")),
                (r["away"], r["ag"], r["hg"], r["aws"], r["hs"], r["ast"], r["hst"], r.get("xgp_a"), r.get("xgp_h")),
            ):
                s = state[team]
                for k, v in zip(keys, (gf, ga, sf, sa, tf, ta, xf, xa)):
                    if v is None or pd.isna(v):
                        continue
                    s[k] = v if k not in s or np.isnan(s[k]) else (1 - alpha) * s[k] + alpha * v
                s["n"] += 1
                last_date[team] = r["date"]
                recent[team].append(r["date"])
                season_tier[(team, r["season"])] = r["tier"]
        i = j
    out = frame.copy()
    for k, v in cols.items():
        out[k] = v
    return out


FEATURE_COLUMNS = [
    "tier", "h_gf", "h_ga", "h_sf", "h_sa", "h_tf", "h_ta", "h_xf", "h_xa", "h_n", "h_rest", "h_m28", "h_promoted",
    "a_gf", "a_ga", "a_sf", "a_sa", "a_tf", "a_ta", "a_xf", "a_xa", "a_n", "a_rest", "a_m28", "a_promoted",
    "elo_h", "elo_a", "elo_diff",
]


SPLIT_LEAGUES = {"SC0", "SC1", "B1", "G1"}


def standings_features(frame):
    teams = frame.groupby(["competition", "season"])["home"].agg(lambda s: set(s)).to_dict()
    for k, v in frame.groupby(["competition", "season"])["away"].agg(lambda s: set(s)).items():
        teams[k] = teams.get(k, set()) | v
    pts = defaultdict(lambda: defaultdict(int))
    played = defaultdict(lambda: defaultdict(int))
    cols = defaultdict(list)
    rows = frame.to_dict("records")
    dates = frame["date"].to_numpy()
    i, n = 0, len(rows)
    while i < n:
        j = i
        while j < n and dates[j] == dates[i]:
            j += 1
        block = rows[i:j]
        for r in block:
            key = (r["competition"], r["season"])
            members = teams[key]
            size = len(members)
            if r["competition"] in SPLIT_LEAGUES or size < 10:
                for side in ("h", "a"):
                    for c in ("dead", "danger", "chasing"):
                        cols[f"{side}_{c}"].append(np.nan)
                cols["season_left"].append(np.nan)
                continue
            games = 2 * (size - 1)
            table = sorted(((pts[key][t], games - played[key][t]) for t in members), reverse=True)
            leader, fourth = table[0][0], table[min(3, size - 1)][0]
            drop_pts, drop_left = table[size - 3]
            safe_line = table[size - 4][0]
            for side, team in (("h", r["home"]), ("a", r["away"])):
                p, left = pts[key][team], games - played[key][team]
                safe = p > drop_pts + 3 * drop_left
                down = p + 3 * left < safe_line
                can_climb = p + 3 * left >= fourth
                cols[f"{side}_dead"].append(float((safe and not can_climb) or down))
                cols[f"{side}_danger"].append(float(not safe and not down and left <= 10 and p - drop_pts <= 3))
                cols[f"{side}_chasing"].append(float(can_climb and left <= 10 and (fourth - p <= 3 or leader - p <= 3)))
            cols["season_left"].append((games - (played[key][r["home"]] + played[key][r["away"]]) / 2) / games)
        for r in block:
            if r.get("status") != "finished" or pd.isna(r["hg"]):
                continue
            key = (r["competition"], r["season"])
            hp, ap = (3, 0) if r["hg"] > r["ag"] else (1, 1) if r["hg"] == r["ag"] else (0, 3)
            pts[key][r["home"]] += hp
            pts[key][r["away"]] += ap
            played[key][r["home"]] += 1
            played[key][r["away"]] += 1
        i = j
    out = frame[["match_id"]].copy()
    for k, v in cols.items():
        out[k] = v
    return out
