import json
import math
import os
import random
import time

import numpy as np
import requests

from . import db
from .inplay import InPlayModel

RED_CARDS = ("Red Card", "Second Yellow")
STATE_TRIGGERS = ("goal", "red_card", "substitution", "correction")


def load_live_model(con, tier=1):
    row = con.execute("SELECT params FROM live_model WHERE status='champion' ORDER BY created_at DESC LIMIT 1").fetchone()
    if not row:
        raise RuntimeError("no trained in-play model; run the train_inplay job first")
    params = json.loads(row["params"])
    model = InPlayModel.from_json(params["model"])
    by_tier = params.get("second_half_scale_by_tier") or {}
    model.second_half_scale = float(by_tier.get("1" if tier == 1 else "2+", params.get("second_half_scale", 1.0)))
    return model, params


EVENT_KINDS = {"goal", "own_goal", "penalty_goal", "red_card", "yellow_card", "substitution", "shot", "status", "correction"}
SIDELESS_KINDS = {"status", "correction"}


def normalise_event(raw):
    if not isinstance(raw, dict):
        raise ValueError("event must be an object")
    eid = str(raw.get("id") or "").strip()
    if not eid:
        raise ValueError("event has no id")
    kind = str(raw.get("kind") or raw.get("type") or "").strip().lower().replace(" ", "_")
    if kind not in EVENT_KINDS:
        raise ValueError(f"{eid}: unsupported kind {kind!r}")
    try:
        period = int(raw.get("period", 1))
        clock = float(raw.get("clock", raw.get("minute", 0)))
    except (TypeError, ValueError):
        raise ValueError(f"{eid}: period and clock must be numbers")
    if not 1 <= period <= 5 or not 0 <= clock <= 130:
        raise ValueError(f"{eid}: period {period} or clock {clock} out of range")
    side = raw.get("side")
    if kind not in SIDELESS_KINDS and side not in ("home", "away"):
        raise ValueError(f"{eid}: side must be home or away")
    xg = raw.get("xg")
    if xg is not None and not 0 <= float(xg) <= 1:
        raise ValueError(f"{eid}: xg must be between 0 and 1")
    payload = dict(raw.get("payload") or {})
    if kind == "correction" and not payload.get("supersedes"):
        raise ValueError(f"{eid}: a correction needs payload.supersedes")
    if kind == "own_goal":
        kind, side, payload["own_goal"] = "goal", side, True
    if kind == "penalty_goal":
        kind, payload["penalty"] = "goal", True
    return {"id": eid, "seq": raw.get("seq"), "period": period, "clock": clock, "kind": kind, "side": side,
            "player": raw.get("player"), "xg": None if xg is None else float(xg), "payload": payload}


def store_events(con, live_id, events):
    accepted = []
    for e in events:
        cur = con.execute("INSERT OR IGNORE INTO live_events(live_id, provider_event_id, provider_seq, received_at, period, clock, kind, side, player_name, xg, payload, superseded) VALUES(?,?,?,?,?,?,?,?,?,?,?,0)",
                          (live_id, e["id"], e.get("seq"), db.now(), e["period"], e["clock"], e["kind"], e.get("side"), e.get("player"), e.get("xg"), db.dumps(e.get("payload") or {})))
        if cur.rowcount:
            accepted.append(e)
            if e["kind"] == "correction":
                con.execute("UPDATE live_events SET superseded=1 WHERE live_id=? AND provider_event_id=?", (live_id, e["payload"]["supersedes"]))
    return accepted


def ingest_feed(con, live_id, raw_events):
    events, rejected = [], []
    for raw in raw_events:
        try:
            events.append(normalise_event(raw))
        except (ValueError, TypeError) as exc:
            rejected.append(str(exc))
    accepted = store_events(con, live_id, events)
    goals = con.execute("SELECT side, COUNT(*) FROM live_events WHERE live_id=? AND kind='goal' AND superseded=0 GROUP BY side", (live_id,)).fetchall()
    score = {r[0]: r[1] for r in goals}
    last = con.execute("SELECT MAX(clock) FROM live_events WHERE live_id=? AND superseded=0", (live_id,)).fetchone()[0]
    con.execute("UPDATE live_sessions SET home_goals=?, away_goals=?, minute=?, last_event_at=?, last_update_at=? WHERE live_id=?",
                (score.get("home", 0), score.get("away", 0), last, db.now(), db.now(), live_id))
    return {"accepted": len(accepted), "duplicates": len(events) - len(accepted), "rejected": rejected,
            "score": [score.get("home", 0), score.get("away", 0)]}


def simulated_session(con, home_name, away_name):
    con.execute("INSERT INTO live_sessions(mode, provider, match_id, home_name, away_name, status, created_at, minute, home_goals, away_goals, prematch) VALUES('simulated','development',NULL,?,?,'running',?,0,0,0,'{}')",
                (home_name, away_name, db.now()))
    return con.execute("SELECT last_insert_rowid()").fetchone()[0]


class LiveMatch:
    def __init__(self, con, live_id, model, meta, lambda_home, lambda_away, interval_minutes=1.0):
        self.con = con
        self.live_id = live_id
        self.model = model
        self.meta = meta
        self.lh = lambda_home
        self.la = lambda_away
        self.mpm = meta["minutes_per_match"]
        self.interval = interval_minutes
        self.last_snapshot_clock = None
        self.last_forecast = None
        self.pending_reasons = []
        self.first_half_end = meta["half_end"][0]
        self.finished = False

    def ingest(self, events):
        return store_events(self.con, self.live_id, events)

    def state(self, period, clock):
        rows = self.con.execute("SELECT * FROM live_events WHERE live_id=? AND superseded=0 AND kind<>'correction' ORDER BY period, clock, provider_seq", (self.live_id,)).fetchall()
        st = {"period": period, "clock": clock, "home_goals": 0, "away_goals": 0, "reds_home": 0, "reds_away": 0,
              "xg_home": 0.0, "xg_away": 0.0, "shots_home": 0, "shots_away": 0, "sot_home": 0, "sot_away": 0,
              "recent_xg_home": 0.0, "recent_xg_away": 0.0, "subs_home": 0, "subs_away": 0}
        momentum = {"home": {"box": 0, "f3": 0, "passes": 0, "press": 0, "poss": 0.0, "def_high": 0, "own60": 0},
                    "away": {"box": 0, "f3": 0, "passes": 0, "press": 0, "poss": 0.0, "def_high": 0, "own60": 0}}
        totals = {"home": {"passes": 0, "press": 0, "poss": 0.0, "def_high": 0, "own60": 0, "recoveries": 0},
                  "away": {"passes": 0, "press": 0, "poss": 0.0, "def_high": 0, "own60": 0, "recoveries": 0}}
        for r in rows:
            if (r["period"], r["clock"]) > (period, clock):
                continue
            side = r["side"]
            recent = r["period"] == period and r["clock"] >= clock - 10
            if r["kind"] == "goal":
                st[f"{side}_goals"] += 1
            elif r["kind"] == "red_card":
                st[f"reds_{side}"] += 1
            elif r["kind"] == "substitution":
                st[f"subs_{side}"] += 1
            elif r["kind"] == "shot":
                st[f"xg_{side}"] += r["xg"] or 0.0
                st[f"shots_{side}"] += 1
                if json.loads(r["payload"]).get("on_target"):
                    st[f"sot_{side}"] += 1
                if recent:
                    st[f"recent_xg_{side}"] += r["xg"] or 0.0
            elif r["kind"] == "minute_stats":
                p = json.loads(r["payload"])
                t = totals[side]
                t["passes"] += p["passes"]
                t["press"] += p["pressures"]
                t["poss"] += p["possession_seconds"]
                t["def_high"] += p["def_actions_high"]
                t["own60"] += p["passes_own60"]
                t["recoveries"] += p["recoveries"]
                if recent:
                    m = momentum[side]
                    m["box"] += p["box_touches"]
                    m["f3"] += p["final_third_passes"]
                    m["passes"] += p["passes"]
                    m["press"] += p["pressures"]
                    m["poss"] += p["possession_seconds"]
        elapsed_play = min(clock, self.meta["half_end"][0]) if period == 1 else self.meta["half_end"][0] + clock - 45
        st["exp_home_so_far"] = self.lh * elapsed_play / self.mpm
        st["exp_away_so_far"] = self.la * elapsed_play / self.mpm
        poss_total = totals["home"]["poss"] + totals["away"]["poss"]
        recent_poss = momentum["home"]["poss"] + momentum["away"]["poss"]
        st["possession_home"] = totals["home"]["poss"] / poss_total if poss_total else None
        st["recent_possession_home"] = momentum["home"]["poss"] / recent_poss if recent_poss else None
        st["ppda_home"] = totals["away"]["own60"] / totals["home"]["def_high"] if totals["home"]["def_high"] else None
        st["ppda_away"] = totals["home"]["own60"] / totals["away"]["def_high"] if totals["away"]["def_high"] else None
        st["pressures_home"], st["pressures_away"] = totals["home"]["press"], totals["away"]["press"]
        st["recoveries_home"], st["recoveries_away"] = totals["home"]["recoveries"], totals["away"]["recoveries"]
        st["momentum"] = momentum
        return st

    def snapshot(self, period, clock, trigger, reasons):
        st = self.state(period, clock)
        st["finished"] = self.finished
        fc = self.model.forecast(st, self.lh / self.mpm, self.la / self.mpm, self.meta["half_end"])
        explanation = self.explain(st, fc, reasons)
        payload = {k: v for k, v in fc.items() if k != "matrix_full"}
        payload["matrix"] = np.round(fc["matrix_full"][:7, :7], 5).tolist()
        payload["state"] = {k: v for k, v in st.items()}
        payload["prematch"] = {"lambda_home": self.lh, "lambda_away": self.la}
        payload["model"] = {"use_evidence": self.model.use_evidence}
        self.con.execute(
            "INSERT INTO live_snapshots(live_id, ts, clock, trigger, home_goals, away_goals, p_home, p_draw, p_away, p_another_goal, exp_remaining, state, explanation) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (self.live_id, db.now(), clock if period == 1 else clock + self.first_half_end - 45, trigger, st["home_goals"], st["away_goals"], fc["p_home"], fc["p_draw"], fc["p_away"],
             fc["p_another_goal"], fc["exp_remaining"], db.dumps(payload), db.dumps(explanation)),
        )
        self.con.execute("UPDATE live_sessions SET last_update_at=?, minute=?, home_goals=?, away_goals=? WHERE live_id=?",
                         (db.now(), clock, st["home_goals"], st["away_goals"], self.live_id))
        self.last_forecast = fc
        self.last_snapshot_clock = (period, clock)
        return fc

    def explain(self, st, fc, reasons):
        out = []
        prev = self.last_forecast
        if prev is not None:
            dh = fc["p_home"] - prev["p_home"]
            da = fc["p_away"] - prev["p_away"]
            dd = fc["p_draw"] - prev["p_draw"]
            for r in reasons:
                out.append({"type": "event", "text": r})
            if reasons:
                out.append({"type": "delta", "text": f"Home {prev['p_home']:.0%} -> {fc['p_home']:.0%}, draw {prev['p_draw']:.0%} -> {fc['p_draw']:.0%}, away {prev['p_away']:.0%} -> {fc['p_away']:.0%}"})
            elif max(abs(dh), abs(da), abs(dd)) >= 0.005:
                lead = st["home_goals"] - st["away_goals"]
                who = "leading team's" if lead != 0 else "draw"
                out.append({"type": "time", "text": f"Time decay: fewer minutes remain; {who} probability {'rises' if (lead > 0 and dh > 0) or (lead < 0 and da > 0) or (lead == 0 and dd > 0) else 'shifts'} ({max(abs(dh), abs(da), abs(dd)):.1%})."})
        if self.model.use_evidence:
            for side in ("home", "away"):
                xg, exp = st[f"xg_{side}"], st[f"exp_{side}_so_far"]
                ratio = math.log((xg + 0.3) / (exp + 0.3))
                beta = self.model.beta[self.model.columns.index("xg_ratio")]
                mult = math.exp(beta * ratio)
                if abs(mult - 1) > 0.03:
                    out.append({"type": "evidence", "text": f"{side.title()} xG so far {xg:.2f} vs {exp:.2f} expected pre-match: learned evidence term scales {side} scoring rate x{mult:.2f}."})
        for side, opp in (("home", "away"), ("away", "home")):
            if st[f"xg_{side}"] > st[f"xg_{opp}"] + 0.5 and st["possession_home" if side == "away" else "possession_home"] is not None:
                poss = st["possession_home"] if side == "home" else 1 - st["possession_home"]
                if poss < 0.45:
                    out.append({"type": "observation", "text": f"{side.title()} has generated more xG ({st[f'xg_{side}']:.2f} vs {st[f'xg_{opp}']:.2f}) despite {poss:.0%} possession."})
        m = st["momentum"]
        box_h, box_a = m["home"]["box"], m["away"]["box"]
        if box_h + box_a >= 6 and max(box_h, box_a) >= 2 * max(1, min(box_h, box_a)):
            side = "home" if box_h > box_a else "away"
            sig = self.model.use_evidence and abs(self.model.beta[self.model.columns.index("recent_xg")] / max(self.model.se[self.model.columns.index("recent_xg")], 1e-9)) >= 2
            out.append({"type": "observation", "text": f"{side.title()} pressure in the last 10 minutes ({max(box_h, box_a)} box touches vs {min(box_h, box_a)})" +
                        ("; recent xG carries measurable predictive weight." if sig else "; historical evidence that short-term pressure predicts an imminent goal is weak, so the forecast barely moves.")})
        return out


def _sb_stream(con, sb_match_id):
    m = con.execute("SELECT * FROM sb_matches WHERE sb_match_id=?", (sb_match_id,)).fetchone()
    events = []
    for e in con.execute("SELECT * FROM sb_events WHERE sb_match_id=? ORDER BY period, clock, seq", (sb_match_id,)):
        kind = e["kind"]
        payload = {}
        if kind == "card":
            if e["detail"] not in RED_CARDS:
                kind = "yellow_card"
            else:
                kind = "red_card"
        if kind == "shot":
            payload = {"outcome": e["outcome"], "on_target": e["outcome"] in ("Goal", "Saved", "Saved To Post"), "type": e["detail"], "x": e["x"], "y": e["y"]}
        if kind == "substitution":
            payload = json.loads(e["detail"])
        if kind == "goal":
            payload = {"type": e["detail"]}
        if kind == "formation":
            payload = {"formation": e["detail"]}
        events.append({"id": f"sb-{sb_match_id}-{e['seq']}", "seq": e["seq"], "period": e["period"], "clock": e["clock"], "kind": kind,
                       "side": e["side"], "player": e["player_name"], "xg": e["xg"], "payload": payload})
    for t in con.execute("SELECT * FROM sb_team_minutes WHERE sb_match_id=?", (sb_match_id,)):
        events.append({"id": f"sbm-{sb_match_id}-{t['side']}-{t['period']}-{t['minute']}", "seq": 10 ** 9 + t["minute"], "period": t["period"],
                       "clock": t["minute"] + 0.999, "kind": "minute_stats", "side": t["side"], "player": None, "xg": None,
                       "payload": {k: t[k] for k in ("passes", "passes_own60", "final_third_passes", "pressures", "recoveries", "def_actions_high", "box_touches", "possession_seconds")}})
    events.sort(key=lambda e: (e["period"], e["clock"], e["seq"]))
    return m, events


def _describe(e, names):
    side = names.get(e["side"], e["side"])
    who = f" ({e['player']})" if e.get("player") else ""
    minute = int(e["clock"]) + 1
    if e["kind"] == "goal":
        return f"Goal for {side}{who} at {minute}'" + (" (own goal)" if e["payload"].get("type") == "Own Goal" else "")
    if e["kind"] == "red_card":
        return f"Red card for {side}{who} at {minute}': scoring rates switch to the learned man-advantage multipliers"
    if e["kind"] == "substitution":
        return f"Substitution for {side} at {minute}': {e['payload'].get('off')} off, {e['payload'].get('on')} on (no validated player-level substitution effect; rates unchanged)"
    if e["kind"] == "correction":
        return f"Provider correction received at {minute}'"
    return e["kind"]


def run_replay(ctx, sb_match_id, speed=60.0, interval_minutes=1.0, faults=True, seed=11):
    try:
        return _run_replay(ctx, sb_match_id, speed, interval_minutes, faults, seed)
    except Exception:
        ctx.con.execute("UPDATE live_sessions SET status='failed' WHERE job_id=? AND status='running'", (ctx.job_id,))
        raise


def _run_replay(ctx, sb_match_id, speed, interval_minutes, faults, seed):
    con = ctx.con
    model, params = load_live_model(con)
    m, events = _sb_stream(con, sb_match_id)
    if m is None:
        raise ValueError("unknown StatsBomb match")
    pre = con.execute("SELECT lambda_home, lambda_away, model_version FROM predictions WHERE match_id=? AND purpose='backtest' AND lambda_home IS NOT NULL ORDER BY prediction_id DESC LIMIT 1", (m["match_id"],)).fetchone()
    if pre is None:
        raise ValueError("no stored pre-match prediction for this match")
    con.execute("INSERT INTO live_sessions(mode, provider, sb_match_id, match_id, home_name, away_name, status, speed, created_at, minute, home_goals, away_goals, prematch, job_id) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                ("replay", "StatsBomb Open Data replay (simulation)", sb_match_id, m["match_id"], m["home_name"], m["away_name"], "running", speed, db.now(), 0, 0, 0,
                 db.dumps({"lambda_home": pre["lambda_home"], "lambda_away": pre["lambda_away"], "model_version": pre["model_version"]}), ctx.job_id))
    live_id = con.execute("SELECT last_insert_rowid()").fetchone()[0]
    ctx.log(f"live session {live_id} replaying StatsBomb match {sb_match_id} at x{speed}")
    lm = LiveMatch(con, live_id, model, params, pre["lambda_home"], pre["lambda_away"], interval_minutes)
    names = {"home": m["home_name"], "away": m["away_name"]}
    rng = random.Random(seed)
    queue = []
    for e in events:
        deliver_at = (e["period"], e["clock"])
        if faults and e["kind"] in ("shot", "minute_stats") and rng.random() < 0.05:
            deliver_at = (e["period"], e["clock"] + 2.0)
        queue.append((deliver_at, e))
        if faults and rng.random() < 0.03:
            queue.append((deliver_at, dict(e)))
    queue.sort(key=lambda q: q[0])
    ends = {1: m["first_half_end"] or 47.0, 2: m["second_half_end"] or 94.5}
    lm.snapshot(1, 0.0, "kickoff", [f"Kick-off. Pre-match expected goals {pre['lambda_home']:.2f} - {pre['lambda_away']:.2f} ({pre['model_version']})"])
    stats = {"delivered": 0, "duplicates_ignored": 0, "late_events": 0}
    qi = 0
    for period in (1, 2):
        clock = 0.0 if period == 1 else 45.0
        next_interval = clock + interval_minutes
        while clock < ends[period]:
            if ctx.cancelled():
                con.execute("UPDATE live_sessions SET status='cancelled' WHERE live_id=?", (live_id,))
                return {"live_id": live_id, "cancelled": True}
            step = 0.25
            clock = min(clock + step, ends[period])
            batch = []
            while qi < len(queue) and queue[qi][0] <= (period, clock):
                batch.append(queue[qi][1])
                qi += 1
            if batch:
                accepted = lm.ingest(batch)
                stats["delivered"] += len(batch)
                stats["duplicates_ignored"] += len(batch) - len(accepted)
                stats["late_events"] += sum(1 for e in accepted if (e["period"], e["clock"]) < (period, clock - 1.0))
                con.execute("UPDATE live_sessions SET last_event_at=? WHERE live_id=?", (db.now(), live_id))
                important = [e for e in accepted if e["kind"] in ("goal", "red_card", "substitution", "correction")]
                if important:
                    lm.snapshot(period, clock, important[0]["kind"], [_describe(e, names) for e in important])
                    next_interval = clock + interval_minutes
            if clock >= next_interval:
                lm.snapshot(period, clock, "interval", [])
                next_interval += interval_minutes
            ctx.progress(int(clock if period == 1 else clock + (ends[1] - 45)), int(ends[1] + ends[2] - 45), f"{'1st' if period == 1 else '2nd'} half {int(clock)}'")
            time.sleep(step * 60 / speed)
        if period == 1:
            lm.first_half_end = ends[1]
            lm.snapshot(1, ends[1], "half_time", ["Half-time"])
    lm.finished = True
    lm.snapshot(2, ends[2], "full_time", ["Full-time"])
    con.execute("UPDATE live_sessions SET status='finished' WHERE live_id=?", (live_id,))
    return {"live_id": live_id, **stats}


FOOTBALL_DATA_ORG = "https://api.football-data.org/v4"
FDO_COMPETITIONS = {"PL": "E0", "ELC": "E1", "PD": "SP1", "BL1": "D1", "SA": "I1", "FL1": "F1", "CL": "CL"}


def _match_for_provider(con, code, home_name, away_name, date_str):
    from .config import COMPETITIONS
    from .identity import resolve_team
    country = COMPETITIONS[code][1]
    pool = [(r["team_id"], r["name"]) for r in con.execute(
        "SELECT DISTINCT t.team_id, t.name FROM matches m JOIN teams t ON t.team_id IN (m.home_team_id, m.away_team_id) WHERE m.competition=?", (code,))]
    h, _ = resolve_team(con, home_name, "football-data.org", country, pool, min_score=0.8)
    a, _ = resolve_team(con, away_name, "football-data.org", country, pool, min_score=0.8)
    if h is None or a is None:
        return None
    return con.execute("SELECT m.match_id, m.competition FROM matches m WHERE m.home_team_id=? AND m.away_team_id=? AND m.match_date BETWEEN date(?, '-1 day') AND date(?, '+1 day')",
                       (h, a, date_str, date_str)).fetchone()


def poll_live_provider(con, model=None, params=None):
    key = os.environ.get("FOOTBALL_DATA_API_KEY")
    if not key:
        con.execute("INSERT INTO provider_health(provider, last_attempt, status, last_error) VALUES(?,?,?,?) ON CONFLICT(provider) DO UPDATE SET last_attempt=excluded.last_attempt, status=excluded.status, last_error=excluded.last_error",
                    ("football-data.org live", db.now(), "not_configured", "FOOTBALL_DATA_API_KEY not set; live tracking unavailable"))
        return {"status": "not_configured"}
    try:
        resp = requests.get(f"{FOOTBALL_DATA_ORG}/matches", params={"status": "IN_PLAY,PAUSED"}, headers={"X-Auth-Token": key}, timeout=30)
        remaining = resp.headers.get("X-Requests-Available-Minute")
        resp.raise_for_status()
        db.provider_event(con, "football-data.org live", True, rate_limit=f"{remaining} requests left this minute" if remaining else None)
    except Exception as exc:
        db.provider_event(con, "football-data.org live", False, str(exc)[:300])
        con.execute("UPDATE live_sessions SET status='stale' WHERE mode='live' AND status='running'")
        return {"status": "error", "error": str(exc)}
    tracked = []
    seen = set()
    for mt in resp.json().get("matches", []):
        code = FDO_COMPETITIONS.get(mt.get("competition", {}).get("code"))
        if not code:
            continue
        ext = f"fdo-{mt['id']}"
        seen.add(ext)
        score = mt.get("score", {}).get("fullTime", {}) or {}
        hg, ag = score.get("home") or 0, score.get("away") or 0
        minute = mt.get("minute")
        row = con.execute("SELECT * FROM live_sessions WHERE provider='football-data.org' AND prematch LIKE ?", (f'%"{ext}"%',)).fetchone()
        if row is None:
            ours = None
            if code in ("E0", "E1", "SP1", "D1", "I1", "F1"):
                ours = _match_for_provider(con, code, mt["homeTeam"]["name"], mt["awayTeam"]["name"], (mt.get("utcDate") or db.now())[:10])
            pre = None
            if ours is not None:
                pre = con.execute("SELECT lambda_home, lambda_away, model_version FROM predictions WHERE match_id=? AND lambda_home IS NOT NULL ORDER BY CASE purpose WHEN 'forward' THEN 0 ELSE 1 END, prediction_id DESC LIMIT 1",
                                  (ours["match_id"],)).fetchone()
            prematch = {"external_id": ext}
            if pre is not None:
                prematch.update({"lambda_home": pre["lambda_home"], "lambda_away": pre["lambda_away"], "model_version": pre["model_version"]})
            con.execute("INSERT INTO live_sessions(mode, provider, match_id, home_name, away_name, status, created_at, minute, home_goals, away_goals, prematch) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                        ("live", "football-data.org", ours["match_id"] if ours else None, mt["homeTeam"]["name"], mt["awayTeam"]["name"], "running", db.now(), None, 0, 0, db.dumps(prematch)))
            row = con.execute("SELECT * FROM live_sessions WHERE live_id=last_insert_rowid()").fetchone()
        tracked.append(ext)
        prematch = json.loads(row["prematch"])
        con.execute("UPDATE live_sessions SET last_event_at=?, status='running' WHERE live_id=?", (db.now(), row["live_id"]))
        if "lambda_home" not in prematch:
            con.execute("UPDATE live_sessions SET minute=?, home_goals=?, away_goals=?, last_update_at=? WHERE live_id=?", (minute, hg, ag, db.now(), row["live_id"]))
            continue
        if model is None:
            model, params = load_live_model(con, tier=1 if code in ("E0", "SP1", "D1", "I1", "F1") else 2)
        lm = LiveMatch(con, row["live_id"], model, params, prematch["lambda_home"], prematch["lambda_away"])
        clock = float(min(minute or 0, 130))
        period = 2 if clock > 45 else 1
        if mt.get("status") == "PAUSED":
            period, clock = 2, 45.0
        events, reasons = [], []
        current = lm.state(period, clock)
        for side, new in (("home", hg), ("away", ag)):
            have = current[f"{side}_goals"]
            for n in range(have, new):
                events.append({"id": f"{ext}-goal-{side}-{n + 1}", "seq": int(clock * 100) + n, "period": period, "clock": max(clock - 0.5, 0.0),
                               "kind": "goal", "side": side, "payload": {"approximate_time": True}})
                reasons.append(f"Goal for {side} (score {hg}-{ag} at about {int(clock)} min; provider gives no exact goal time)")
            if new < have:
                last = con.execute("SELECT provider_event_id FROM live_events WHERE live_id=? AND kind='goal' AND side=? AND superseded=0 ORDER BY period DESC, clock DESC LIMIT 1",
                                   (row["live_id"], side)).fetchone()
                if last:
                    events.append({"id": f"{ext}-corr-{last[0]}", "period": period, "clock": clock, "kind": "correction", "side": side, "payload": {"supersedes": last[0]}})
                    reasons.append(f"Provider score correction: {side} goal removed")
        accepted = lm.ingest(events) if events else []
        last_clock = row["minute"] if row["minute"] is not None else -1
        if accepted or clock - last_clock >= 1:
            trigger = "goal" if any(e["kind"] == "goal" for e in accepted) else "correction" if accepted else "interval"
            lm.snapshot(period, clock, trigger, reasons)
    for row in con.execute("SELECT live_id, prematch FROM live_sessions WHERE provider='football-data.org' AND status='running'").fetchall():
        if json.loads(row["prematch"]).get("external_id") not in seen:
            con.execute("UPDATE live_sessions SET status='ended_or_unavailable' WHERE live_id=?", (row["live_id"],))
    return {"status": "ok", "tracked": tracked}
