import hashlib
import io
import time
from datetime import datetime

import pandas as pd
import requests

from . import db
from .config import COMPETITIONS, CURRENT_SEASON, FDCO_BASE, FDCO_FIXTURES, FIRST_SEASON, RAW, season_code
from .identity import get_or_create_team

SOURCE = "football-data.co.uk"

STAT_COLUMNS = {
    "HS": "hs", "AS": "aws", "HST": "hst", "AST": "ast", "HC": "hc", "AC": "ac",
    "HF": "hf", "AF": "af", "HY": "hy", "AY": "ay", "HR": "hr", "AR": "ar",
}

ODDS_SETS = [
    ("pinnacle", "close", "PSCH", "PSCD", "PSCA", "PC>2.5", "PC<2.5"),
    ("pinnacle", "open", "PSH", "PSD", "PSA", "P>2.5", "P<2.5"),
    ("market_avg", "close", "AvgCH", "AvgCD", "AvgCA", "AvgC>2.5", "AvgC<2.5"),
    ("market_avg", "open", "AvgH", "AvgD", "AvgA", "Avg>2.5", "Avg<2.5"),
    ("market_avg", "open", "BbAvH", "BbAvD", "BbAvA", "BbAv>2.5", "BbAv<2.5"),
    ("bet365", "close", "B365CH", "B365CD", "B365CA", "B365C>2.5", "B365C<2.5"),
    ("bet365", "open", "B365H", "B365D", "B365A", "B365>2.5", "B365<2.5"),
]

FIELD_CATALOG = [
    ("fthg/ftag", "observed", SOURCE, "Full-time goals"),
    ("hthg/htag", "observed", SOURCE, "Half-time goals; NULL where the source omits them"),
    ("hs/aws, hst/ast", "observed", SOURCE, "Shots and shots on target; NULL when the source does not provide them (most second tiers and the smaller top flights before 2017/18; English and big-five top flights are complete)"),
    ("hc/ac, hf/af, hy/ay, hr/ar", "observed", SOURCE, "Corners, fouls, cards"),
    ("odds", "observed", SOURCE, "Bookmaker decimal odds; closing odds from 2012/13 onwards"),
    ("rest_days", "derived", "engine", "Days since previous league match in the database (cup matches absent, so a lower bound on fatigue)"),
    ("xg_proxy", "estimated", "engine", "Shot-based expected goals proxy fitted on prior seasons; not provider xG"),
    ("sb_xg", "observed", "StatsBomb Open Data", "Provider shot xG, 2015/16 big-5 leagues and selected tournaments"),
    ("possession, passes, pressures", "observed", "StatsBomb Open Data", "Event-derived team metrics, StatsBomb matches only"),
    ("ppda", "derived", "engine", "Opponent passes in own 60% divided by defensive actions there, StatsBomb matches only"),
    ("lineups, substitutions", "observed", "StatsBomb Open Data", "Only for StatsBomb-covered matches"),
    ("injuries", "unavailable", "none", "No legitimate open injury feed integrated"),
    ("weather, pitch, travel", "unavailable", "none", "No legitimate source integrated"),
]


def _download(url, path, force=False):
    if path.exists() and not force:
        return path.read_bytes()
    resp = requests.get(url, timeout=60, headers={"User-Agent": "FootballIntelligence research (local)"})
    resp.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(resp.content)
    time.sleep(0.5)
    return resp.content


def _read_csv(content: bytes) -> pd.DataFrame:
    for enc in ("utf-8-sig", "latin-1"):
        try:
            frame = pd.read_csv(io.BytesIO(content), encoding=enc, on_bad_lines="skip", engine="python")
            break
        except UnicodeDecodeError:
            continue
    frame = frame.dropna(how="all")
    frame = frame.loc[:, [c for c in frame.columns if not str(c).startswith("Unnamed")]]
    return frame.dropna(subset=[c for c in ("HomeTeam", "AwayTeam") if c in frame.columns])


def _parse_date(value):
    value = str(value).strip()
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _int(row, col):
    value = row.get(col)
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _float(row, col):
    value = row.get(col)
    if value is None or pd.isna(value):
        return None
    try:
        value = float(value)
    except ValueError:
        return None
    return value if value > 1.0 else None


def ensure_reference(con):
    for code, (name, country, tier) in COMPETITIONS.items():
        con.execute("INSERT OR IGNORE INTO competitions VALUES(?,?,?,?)", (code, name, country, tier))
    for field, status, source, desc in FIELD_CATALOG:
        con.execute("INSERT OR REPLACE INTO field_catalog VALUES(?,?,?,?)", (field, status, source, desc))
    con.execute(
        "INSERT OR REPLACE INTO sources VALUES(?,?,?,?,?)",
        (SOURCE, "https://www.football-data.co.uk", "Free to use per site ('My data is free'); no redistribution terms published; attribution given",
         "Results, HT scores, shots, corners, fouls, cards, bookmaker odds; 1993+ (stats ~2000+, closing odds 2012/13+)",
         "No lineups, events, xG or injuries; kick-off times in UK local time from 2019/20"),
    )
    con.execute(
        "INSERT OR REPLACE INTO sources VALUES(?,?,?,?,?)",
        ("StatsBomb Open Data", "https://github.com/statsbomb/open-data", "Free for research with attribution to StatsBomb (see repo LICENSE)",
         "Full event data incl. shot xG, lineups: 2015/16 big-5 leagues; selected tournaments and club-specific seasons",
         "Historical only; no live feed; selection-biased club-specific seasons"),
    )


def find_existing(con, code, season, home, away, date, status):
    rows = con.execute(
        "SELECT match_id, status, fthg, ftag, source_row_hash, match_date FROM matches WHERE competition=? AND season=? AND home_team_id=? AND away_team_id=?",
        (code, season, home, away),
    ).fetchall()
    if not rows:
        return None
    target = datetime.fromisoformat(date)
    same_day = [r for r in rows if r["match_date"] == date]
    if same_day:
        return same_day[0]
    pending = [r for r in rows if r["status"] in ("scheduled", "result_missing")]
    if pending:
        return min(pending, key=lambda r: abs((datetime.fromisoformat(r["match_date"]) - target).days))
    return None


def upsert_rows(con, frame: pd.DataFrame, code: str, season: int, scheduled=False):
    country = COMPETITIONS[code][1]
    stamp = db.now()
    counts = {"inserted": 0, "updated": 0, "unchanged": 0, "corrected": 0, "skipped": 0}
    for _, row in frame.iterrows():
        date = _parse_date(row.get("Date"))
        if not date:
            counts["skipped"] += 1
            continue
        home = get_or_create_team(con, str(row["HomeTeam"]).strip(), country, SOURCE)
        away = get_or_create_team(con, str(row["AwayTeam"]).strip(), country, SOURCE)
        fthg, ftag = _int(row, "FTHG"), _int(row, "FTAG")
        if fthg is None:
            fthg, ftag = _int(row, "HG"), _int(row, "AG")
        status = "finished" if fthg is not None and ftag is not None and not scheduled else "scheduled"
        kickoff = str(row.get("Time")).strip() if "Time" in row and not pd.isna(row.get("Time")) else None
        values = {
            "match_date": date, "kickoff": kickoff, "status": status, "fthg": fthg, "ftag": ftag,
            "hthg": _int(row, "HTHG"), "htag": _int(row, "HTAG"),
            "referee": None if pd.isna(row.get("Referee", float("nan"))) else str(row.get("Referee")).strip(),
            "attendance": _int(row, "Attendance"),
        }
        for src, dst in STAT_COLUMNS.items():
            values[dst] = _int(row, src)
        digest = hashlib.sha1(repr(sorted(values.items())).encode()).hexdigest()
        existing = find_existing(con, code, season, home, away, date, status)
        if existing is None:
            cols = ", ".join(values)
            con.execute(
                f"INSERT INTO matches(competition, season, home_team_id, away_team_id, source, source_row_hash, first_seen, updated_at, {cols}) VALUES(?,?,?,?,?,?,?,?,{','.join('?' * len(values))})",
                (code, season, home, away, SOURCE, digest, stamp, stamp, *values.values()),
            )
            match_id = con.execute("SELECT last_insert_rowid()").fetchone()[0]
            counts["inserted"] += 1
        else:
            match_id = existing["match_id"]
            if existing["source_row_hash"] == digest and existing["status"] != "result_missing":
                counts["unchanged"] += 1
            elif existing["status"] == "finished" and status == "scheduled":
                counts["unchanged"] += 1
            else:
                if existing["status"] == "finished" and (existing["fthg"], existing["ftag"]) != (fthg, ftag):
                    counts["corrected"] += 1
                    db.notify(con, "warning", "Result corrected by source", f"{code} {season} match {match_id}: {existing['fthg']}-{existing['ftag']} -> {fthg}-{ftag}")
                sets = ", ".join(f"{k}=?" for k in values)
                con.execute(f"UPDATE matches SET {sets}, source_row_hash=?, updated_at=? WHERE match_id=?", (*values.values(), digest, stamp, match_id))
                counts["updated"] += 1
        for bookmaker, timing, h, d, a, over, under in ODDS_SETS:
            ph, pd_, pa = _float(row, h), _float(row, d), _float(row, a)
            if ph and pd_ and pa:
                con.execute(
                    "INSERT OR REPLACE INTO odds VALUES(?,?,?,?,?,?,?,?)",
                    (match_id, bookmaker, timing, ph, pd_, pa, _float(row, over), _float(row, under)),
                )
    return counts


def mark_stale_fixtures(con, grace_days=3):
    return con.execute(
        "UPDATE matches SET status='result_missing', updated_at=? WHERE status='scheduled' AND match_date < date('now', ?)",
        (db.now(), f"-{grace_days} day"),
    ).rowcount


def ingest_history(con, seasons=None, codes=None, progress=None, refresh_current=True):
    ensure_reference(con)
    seasons = seasons or list(range(FIRST_SEASON, CURRENT_SEASON + 1))
    codes = codes or list(COMPETITIONS)
    units = [(s, c) for s in seasons for c in codes]
    totals = {"inserted": 0, "updated": 0, "unchanged": 0, "corrected": 0, "skipped": 0, "files": 0, "failed": []}
    for i, (season, code) in enumerate(units):
        url = f"{FDCO_BASE}/{season_code(season)}/{code}.csv"
        path = RAW / "fdco" / season_code(season) / f"{code}.csv"
        force = refresh_current and season == CURRENT_SEASON
        try:
            content = _download(url, path, force=force)
            db.provider_event(con, SOURCE, True)
        except Exception as exc:
            db.provider_event(con, SOURCE, False, str(exc)[:300])
            totals["failed"].append(f"{code} {season}: {exc}")
            if progress:
                progress(i + 1, len(units), f"failed {code} {season}")
            continue
        frame = _read_csv(content)
        with db.transaction(con):
            counts = upsert_rows(con, frame, code, season)
        for k, v in counts.items():
            totals[k] += v
        totals["files"] += 1
        if progress:
            progress(i + 1, len(units), f"{code} {season}: +{counts['inserted']} ~{counts['updated']}")
    totals["stale_fixtures_marked"] = mark_stale_fixtures(con)
    return totals


def ingest_fixtures(con):
    ensure_reference(con)
    try:
        resp = requests.get(FDCO_FIXTURES, timeout=60, headers={"User-Agent": "FootballIntelligence research (local)"})
        resp.raise_for_status()
        db.provider_event(con, SOURCE + " fixtures", True)
    except Exception as exc:
        db.provider_event(con, SOURCE + " fixtures", False, str(exc)[:300])
        raise
    path = RAW / "fdco" / "fixtures" / f"fixtures_{datetime.now():%Y%m%d}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(resp.content)
    frame = _read_csv(resp.content)
    frame = frame[frame["Div"].isin(list(COMPETITIONS))]
    totals = {}
    with db.transaction(con):
        for code, part in frame.groupby("Div"):
            counts = upsert_rows(con, part, code, CURRENT_SEASON, scheduled=True)
            totals[code] = counts
    return totals


def data_quality(con):
    rows = con.execute(
        """SELECT competition, season, COUNT(*) n,
                  SUM(status='finished') finished,
                  SUM(hthg IS NULL AND status='finished') missing_ht,
                  SUM(hs IS NULL AND status='finished') missing_shots,
                  SUM(hst IS NULL AND status='finished') missing_sot,
                  SUM(hc IS NULL AND status='finished') missing_corners,
                  (SELECT COUNT(*) FROM odds o JOIN matches m2 ON m2.match_id=o.match_id
                     WHERE m2.competition=m.competition AND m2.season=m.season AND o.bookmaker='pinnacle' AND o.timing='close') pinnacle_close
           FROM matches m GROUP BY competition, season ORDER BY competition, season"""
    ).fetchall()
    return [dict(r) for r in rows]
