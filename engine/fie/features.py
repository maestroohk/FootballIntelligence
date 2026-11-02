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
