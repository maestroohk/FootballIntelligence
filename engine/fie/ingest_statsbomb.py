import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import requests

from . import db
from .config import STATSBOMB_BASE
from .identity import resolve_team

SOURCE = "StatsBomb Open Data"

LEAGUE_SEASONS = {
    (2, 27): ("E0", "ENG", 2015),
    (11, 27): ("SP1", "ESP", 2015),
    (9, 27): ("D1", "GER", 2015),
    (12, 27): ("I1", "ITA", 2015),
    (7, 27): ("F1", "FRA", 2015),
}

GOAL = (120.0, 40.0)


def _get(url, session, retries=3):
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=120)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def _dist(loc):
    return ((GOAL[0] - loc[0]) ** 2 + (GOAL[1] - loc[1]) ** 2) ** 0.5


def _clock(ev):
    return ev["minute"] + ev["second"] / 60.0


def parse_match(meta, events):
    home_name = meta["home_team"]["home_team_name"]
    away_name = meta["away_team"]["away_team_name"]
    side_of = {home_name: "home", away_name: "away"}
    compact = []
    minutes = defaultdict(lambda: defaultdict(float))
    players = {}
    stats = defaultdict(lambda: defaultdict(float))
    on_pitch = {}
    starters = set()
    positions = {}
    formations = {}
    period_end = {}
    pass_to_xg = {}
    for ev in events:
        if ev["type"]["name"] == "Shot" and "key_pass_id" in ev.get("shot", {}):
            pass_to_xg[ev["shot"]["key_pass_id"]] = ev["shot"].get("statsbomb_xg", 0.0)
    for ev in events:
        etype = ev["type"]["name"]
        team = ev.get("team", {}).get("name")
        side = side_of.get(team, "unknown")
        period = ev["period"]
        if period > 2:
            continue
        clock = _clock(ev)
        key = (side, period, ev["minute"])
        pid = ev.get("player", {}).get("id")
        pname = ev.get("player", {}).get("name")
        loc = ev.get("location")
        ps = stats[pid] if pid else None
        poss_team = ev.get("possession_team", {}).get("name")
        if poss_team in side_of and ev.get("duration"):
            minutes[(side_of[poss_team], period, ev["minute"])]["possession_seconds"] += float(ev["duration"])
        if etype == "Starting XI":
            formations[side] = str(ev["tactics"].get("formation"))
            for p in ev["tactics"]["lineup"]:
                on_pitch[p["player"]["id"]] = (side, 0.0)
                starters.add(p["player"]["id"])
                positions[p["player"]["id"]] = p["position"]["name"]
                players[p["player"]["id"]] = (p["player"]["name"], side, team)
            continue
        if etype == "Half End":
            period_end[period] = max(period_end.get(period, 0), clock)
            continue
        if pid and pid not in players:
            players[pid] = (pname, side, team)
        if etype == "Pass":
            p = ev["pass"]
            complete = "outcome" not in p
            minutes[key]["passes"] += 1
            if loc and loc[0] < 72:
                minutes[key]["passes_own60"] += 1
            end = p.get("end_location")
            if complete and end and end[0] >= 80:
                minutes[key]["final_third_passes"] += 1
            if ps is not None:
                ps["passes"] += 1
                ps["passes_completed"] += complete
                if p.get("shot_assist") or p.get("goal_assist"):
                    ps["key_passes"] += 1
                    ps["xa"] += pass_to_xg.get(ev["id"], 0.0)
                if p.get("goal_assist"):
                    ps["assists"] += 1
                if p.get("cross"):
                    ps["crosses"] += 1
                if complete and loc and end and loc[0] >= 48 and _dist(end) <= 0.75 * _dist(loc):
                    ps["progressive_passes"] += 1
        elif etype == "Carry":
            end = ev.get("carry", {}).get("end_location")
            if ps is not None and loc and end and loc[0] >= 48 and _dist(loc) - _dist(end) >= 10:
                ps["progressive_carries"] += 1
        elif etype == "Pressure":
            minutes[key]["pressures"] += 1
            if ps is not None:
                ps["pressures"] += 1
        elif etype == "Ball Recovery":
            if not ev.get("ball_recovery", {}).get("recovery_failure"):
                minutes[key]["recoveries"] += 1
                if ps is not None:
                    ps["recoveries"] += 1
        elif etype == "Interception":
            if loc and loc[0] > 48:
                minutes[key]["def_actions_high"] += 1
            if ps is not None:
                ps["interceptions"] += 1
        elif etype == "Duel":
            if ev.get("duel", {}).get("type", {}).get("name") == "Tackle":
                if loc and loc[0] > 48:
                    minutes[key]["def_actions_high"] += 1
                if ps is not None:
                    ps["tackles"] += 1
        elif etype == "Clearance":
            if ps is not None:
                ps["clearances"] += 1
        elif etype == "Dribble":
            if ps is not None and ev.get("dribble", {}).get("outcome", {}).get("name") == "Complete":
                ps["dribbles"] += 1
        elif etype == "Foul Committed":
            if loc and loc[0] > 48:
                minutes[key]["def_actions_high"] += 1
            if ps is not None:
                ps["fouls"] += 1
            card = ev.get("foul_committed", {}).get("card", {}).get("name")
            if card:
                compact.append(_compact(ev, side, "card", detail=card))
                _card(ps, card, clock)
        elif etype == "Bad Behaviour":
            card = ev.get("bad_behaviour", {}).get("card", {}).get("name")
            if card:
                compact.append(_compact(ev, side, "card", detail=card))
                _card(ps, card, clock)
        elif etype == "Shot":
            s = ev["shot"]
            xg = float(s.get("statsbomb_xg", 0.0))
            outcome = s.get("outcome", {}).get("name")
            minutes[key]["shots"] += 1
            minutes[key]["xg"] += xg
            compact.append(_compact(ev, side, "shot", xg=xg, outcome=outcome, detail=s.get("type", {}).get("name")))
            if ps is not None:
                ps["shots"] += 1
                ps["xg"] += xg
                ps["goals"] += outcome == "Goal"
            if outcome == "Goal":
                compact.append(_compact(ev, side, "goal", xg=xg, outcome="Goal", detail=s.get("type", {}).get("name")))
        elif etype == "Own Goal For":
            compact.append(_compact(ev, side, "goal", outcome="Own Goal", detail="Own Goal"))
        elif etype == "Own Goal Against":
            if ps is not None:
                ps["own_goals"] += 1
        elif etype == "Substitution":
            rep = ev["substitution"]["replacement"]
            compact.append(_compact(ev, side, "substitution", detail=json.dumps({"off": pname, "on": rep["name"], "off_id": pid, "on_id": rep["id"]})))
            on_pitch[rep["id"]] = (side, clock)
            players[rep["id"]] = (rep["name"], side, team)
            positions.setdefault(rep["id"], ev.get("position", {}).get("name"))
            stats[pid]["minute_off"] = clock
        elif etype == "Tactical Shift":
            formations[side + "_shift"] = str(ev.get("tactics", {}).get("formation"))
            compact.append(_compact(ev, side, "formation", detail=str(ev.get("tactics", {}).get("formation"))))
        elif etype == "Injury Stoppage":
            compact.append(_compact(ev, side, "injury_stoppage"))
        if etype in ("Pass", "Carry", "Shot", "Dribble", "Ball Receipt*") and loc and loc[0] > 102 and 18 < loc[1] < 62:
            minutes[key]["box_touches"] += 1
    full_time = period_end.get(2, max((_clock(e) for e in events if e["period"] == 2), default=90.0))
    first_half_end = period_end.get(1, 45.0)
    team_shots = defaultdict(lambda: {"on_target": 0, "goals": 0, "xg_on_target": 0.0})
    for c in compact:
        if c["kind"] == "shot" and c["outcome"] in ("Goal", "Saved", "Saved To Post"):
            t = team_shots[c["side"]]
            t["on_target"] += 1
            t["goals"] += c["outcome"] == "Goal"
            t["xg_on_target"] += c["xg"] or 0.0
    appearances = []
    for pid, (name, side, team) in players.items():
        if pid not in on_pitch:
            continue
        s = stats[pid]
        start = on_pitch[pid][1]
        off = s.get("minute_off", full_time)
        if s.get("red"):
            off = min(off, s.get("red_minute", off))
        row = {k: s.get(k, 0) for k in ("goals", "own_goals", "assists", "shots", "xg", "xa", "key_passes", "passes", "passes_completed",
                                        "progressive_passes", "progressive_carries", "crosses", "pressures", "tackles", "interceptions",
                                        "recoveries", "clearances", "dribbles", "fouls", "yellow", "red")}
        pos = positions.get(pid)
        gk = {}
        if pos == "Goalkeeper" and pid in starters:
            opp = "away" if side == "home" else "home"
            t = team_shots[opp]
            gk = {"gk_shots_faced": t["on_target"], "gk_goals_conceded": t["goals"], "gk_xg_faced": t["xg_on_target"], "gk_saves": t["on_target"] - t["goals"]}
        appearances.append({"player_id": pid, "name": name, "side": side, "team_name": team, "position": pos, "started": int(pid in starters),
                            "minute_on": start, "minute_off": off, "minutes": max(0.0, off - start), **row,
                            **{k: gk.get(k) for k in ("gk_shots_faced", "gk_goals_conceded", "gk_xg_faced", "gk_saves")}})
    return {"events": compact, "minutes": minutes, "appearances": appearances, "formations": formations,
            "first_half_end": first_half_end, "full_time": full_time}


def _card(ps, card, clock):
    if ps is None:
        return
    if card == "Yellow Card":
        ps["yellow"] += 1
    else:
        ps["red"] += 1
        ps["red_minute"] = clock


def _compact(ev, side, kind, xg=None, outcome=None, detail=None):
    loc = ev.get("location") or [None, None]
    return {"seq": ev["index"] * 10 + (1 if kind == "goal" else 0), "period": ev["period"], "minute": ev["minute"], "second": ev["second"], "clock": _clock(ev),
            "kind": kind, "side": side, "player_id": ev.get("player", {}).get("id"), "player_name": ev.get("player", {}).get("name"),
            "xg": xg, "x": loc[0], "y": loc[1], "outcome": outcome, "detail": detail}


def store_match(con, sb_id, parsed):
    with db.transaction(con):
        con.execute("DELETE FROM sb_events WHERE sb_match_id=?", (sb_id,))
        con.execute("DELETE FROM sb_team_minutes WHERE sb_match_id=?", (sb_id,))
        con.execute("DELETE FROM appearances WHERE sb_match_id=?", (sb_id,))
        con.executemany(
            "INSERT INTO sb_events VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [(sb_id, e["seq"], e["period"], e["minute"], e["second"], e["clock"], e["kind"], e["side"], e["player_id"], e["player_name"],
              e["xg"], e["x"], e["y"], e["outcome"], e["detail"]) for e in parsed["events"]],
        )
        con.executemany(
            "INSERT INTO sb_team_minutes VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [(sb_id, side, period, minute, int(v["passes"]), int(v["passes_own60"]), int(v["final_third_passes"]), int(v["pressures"]),
              int(v["recoveries"]), int(v["def_actions_high"]), int(v["box_touches"]), float(v["possession_seconds"]), int(v["shots"]), float(v["xg"]))
             for (side, period, minute), v in parsed["minutes"].items() if side in ("home", "away")],
        )
        for a in parsed["appearances"]:
            con.execute("INSERT OR IGNORE INTO players(player_id, name, source) VALUES(?,?,?)", (a["player_id"], a["name"], SOURCE))
            con.execute(
                f"INSERT INTO appearances VALUES({','.join('?' * 34)})",
                (sb_id, a["player_id"], a["side"], a["team_name"], a["position"], a["started"], a["minute_on"], a["minute_off"], a["minutes"],
                 a["goals"], a["own_goals"], a["assists"], a["shots"], a["xg"], a["xa"], a["key_passes"], a["passes"], a["passes_completed"],
                 a["progressive_passes"], a["progressive_carries"], a["crosses"], a["pressures"], a["tackles"], a["interceptions"],
                 a["recoveries"], a["clearances"], a["dribbles"], a["fouls"], a["yellow"], a["red"],
                 a["gk_shots_faced"], a["gk_goals_conceded"], a["gk_xg_faced"], a["gk_saves"]),
            )
        f = parsed["formations"]
        con.execute("UPDATE sb_matches SET home_formation=?, away_formation=?, first_half_end=?, second_half_end=?, processed=1 WHERE sb_match_id=?",
                    (f.get("home"), f.get("away"), parsed["first_half_end"], parsed["full_time"], sb_id))


def link_matches(con):
    linked = unresolved = 0
    for r in con.execute("SELECT * FROM sb_matches WHERE match_id IS NULL").fetchall():
        code, country, season = next(v for v in LEAGUE_SEASONS.values() if v[0] == r["competition"])
        pool = [(x["team_id"], x["name"]) for x in con.execute(
            "SELECT DISTINCT t.team_id, t.name FROM matches m JOIN teams t ON t.team_id IN (m.home_team_id, m.away_team_id) WHERE m.competition=? AND m.season=?",
            (code, season))]
        h, _ = resolve_team(con, r["home_name"], "statsbomb", country, pool)
        a, _ = resolve_team(con, r["away_name"], "statsbomb", country, pool)
        if h is None or a is None:
            unresolved += 1
            continue
        d = date.fromisoformat(r["match_date"])
        m = con.execute(
            "SELECT match_id FROM matches WHERE home_team_id=? AND away_team_id=? AND match_date BETWEEN ? AND ?",
            (h, a, (d - timedelta(days=2)).isoformat(), (d + timedelta(days=2)).isoformat()),
        ).fetchone()
        con.execute("UPDATE sb_matches SET home_team_id=?, away_team_id=?, match_id=? WHERE sb_match_id=?", (h, a, m["match_id"] if m else None, r["sb_match_id"]))
        linked += m is not None
    return {"linked": linked, "unresolved_teams": unresolved}


def ingest(con, progress=None, workers=3, cancelled=None):
    session = requests.Session()
    session.headers["User-Agent"] = "FootballIntelligence research (local)"
    for (comp_id, season_id), (code, country, season) in LEAGUE_SEASONS.items():
        matches = _get(f"{STATSBOMB_BASE}/matches/{comp_id}/{season_id}.json", session)
        db.provider_event(con, SOURCE, True)
        with db.transaction(con):
            for m in matches:
                con.execute(
                    "INSERT OR IGNORE INTO sb_matches(sb_match_id, competition, season, match_date, kickoff, home_name, away_name, home_score, away_score) VALUES(?,?,?,?,?,?,?,?,?)",
                    (m["match_id"], code, season, m["match_date"], m.get("kick_off"), m["home_team"]["home_team_name"], m["away_team"]["away_team_name"], m["home_score"], m["away_score"]),
                )
    todo = con.execute("SELECT sb_match_id, competition FROM sb_matches WHERE processed=0 ORDER BY match_date").fetchall()
    meta_cache = {}
    for (comp_id, season_id), (code, _, _) in LEAGUE_SEASONS.items():
        for m in _get(f"{STATSBOMB_BASE}/matches/{comp_id}/{season_id}.json", session):
            meta_cache[m["match_id"]] = m

    def fetch(sb_id):
        events = _get(f"{STATSBOMB_BASE}/events/{sb_id}.json", session)
        return sb_id, parse_match(meta_cache[sb_id], events)

    done = errors = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        ids = [r["sb_match_id"] for r in todo]
        for start in range(0, len(ids), workers * 4):
            if cancelled and cancelled():
                break
            chunk = ids[start:start + workers * 4]
            for fut in [pool.submit(fetch, i) for i in chunk]:
                try:
                    sb_id, parsed = fut.result()
                    store_match(con, sb_id, parsed)
                    done += 1
                except Exception as exc:
                    errors += 1
                    db.provider_event(con, SOURCE, False, str(exc)[:300])
            if progress:
                progress(start + len(chunk), len(ids), f"statsbomb {done} parsed, {errors} errors")
    with db.transaction(con):
        link = link_matches(con)
    return {"parsed": done, "errors": errors, **link}
