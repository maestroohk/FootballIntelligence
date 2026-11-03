from pathlib import Path

import numpy as np
import pandas as pd

from . import db
from .config import COMPETITIONS, RAW
from .ingest_fdco import _parse_date, _read_csv, find_existing
from .identity import get_or_create_team

PRE = "pre_close"
CLOSE = "close"
HISTORICAL_OBSERVED = "documented: pre-closing prices collected before kick-off"
HISTORICAL_CLOSE = "documented: closing prices collected at kick-off"

BOOKS_1X2 = [("pinnacle", "PS", "PSC"), ("bet365", "B365", "B365C"), ("market_avg", "Avg", "AvgC"), ("market_max", "Max", "MaxC"),
             ("market_avg", "BbAv", None), ("market_max", "BbMx", None)]
BOOKS_OU = [("pinnacle", "P", "PC"), ("bet365", "B365", "B365C"), ("market_avg", "Avg", "AvgC"), ("market_max", "Max", "MaxC"),
            ("market_avg", "BbAv", None), ("market_max", "BbMx", None)]
BOOKS_AH = [("pinnacle", "P", "PC"), ("bet365", "B365", "B365C"), ("market_avg", "Avg", "AvgC"), ("market_max", "Max", "MaxC"),
            ("market_avg", "BbAv", None), ("market_max", "BbMx", None)]

SCHEMA = """
CREATE TABLE IF NOT EXISTS market_odds (
    match_id INTEGER NOT NULL REFERENCES matches(match_id),
    market TEXT NOT NULL,
    line REAL NOT NULL,
    selection TEXT NOT NULL,
    bookmaker TEXT NOT NULL,
    timing TEXT NOT NULL,
    odds REAL NOT NULL,
    observed_at TEXT NOT NULL,
    source TEXT NOT NULL,
    PRIMARY KEY(match_id, market, line, selection, bookmaker, timing, observed_at)
);
CREATE INDEX IF NOT EXISTS ix_market_odds_match ON market_odds(match_id, market, timing);
"""


def _price(row, col):
    if col is None or col not in row.index:
        return None
    v = row[col]
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) and v > 1.0 else None


def _line(row, *cols):
    for c in cols:
        if c in row.index:
            try:
                v = float(row[c])
            except (TypeError, ValueError):
                continue
            if np.isfinite(v) and abs(v) <= 6 and abs(v * 4 - round(v * 4)) < 1e-9:
                return v
    return None


def row_prices(row):
    out = []
    seen = set()
    for book, pre, close in BOOKS_1X2:
        for timing, prefix in ((PRE, pre), (CLOSE, close)):
            if prefix is None or (book, timing, "1x2") in seen:
                continue
            prices = [_price(row, f"{prefix}{s}") for s in ("H", "D", "A")]
            if all(prices):
                seen.add((book, timing, "1x2"))
                out += [("1x2", 0.0, sel, book, timing, p) for sel, p in zip(("home", "draw", "away"), prices)]
    for book, pre, close in BOOKS_OU:
        for timing, prefix in ((PRE, pre), (CLOSE, close)):
            if prefix is None or (book, timing, "ou") in seen:
                continue
            prices = [_price(row, f"{prefix}>2.5"), _price(row, f"{prefix}<2.5")]
            if all(prices):
                seen.add((book, timing, "ou"))
                out += [("ou", 2.5, sel, book, timing, p) for sel, p in zip(("over", "under"), prices)]
    for timing, line_cols in ((PRE, ("AHh", "BbAHh")), (CLOSE, ("AHCh",))):
        line = _line(row, *line_cols)
        if line is None:
            continue
        for book, pre, close in BOOKS_AH:
            prefix = pre if timing == PRE else close
            if prefix is None or (book, timing, "ah") in seen:
                continue
            infix = "AH" if timing == PRE else "AH"
            prices = [_price(row, f"{prefix}{infix}H"), _price(row, f"{prefix}{infix}A")]
            if all(prices):
                seen.add((book, timing, "ah"))
                out += [("ah", line, sel, book, timing, p) for sel, p in zip(("home", "away"), prices)]
    return out


def ingest_file(con, path: Path, code: str, season: int, observed_at=HISTORICAL_OBSERVED, scheduled=False):
    frame = _read_csv(path.read_bytes())
    if "Div" in frame.columns:
        frame = frame[frame["Div"] == code]
    country = COMPETITIONS[code][1]
    rows = []
    unmatched = 0
    for _, row in frame.iterrows():
        date = _parse_date(row.get("Date"))
        if not date:
            continue
        home = get_or_create_team(con, str(row["HomeTeam"]).strip(), country, "football-data.co.uk", season)
        away = get_or_create_team(con, str(row["AwayTeam"]).strip(), country, "football-data.co.uk", season)
        match = find_existing(con, code, season, home, away, date, "scheduled" if scheduled else "finished")
        if match is None:
            unmatched += 1
            continue
        for market, line, sel, book, timing, price in row_prices(row):
            if scheduled and timing == CLOSE:
                continue
            seen = HISTORICAL_CLOSE if timing == CLOSE and observed_at == HISTORICAL_OBSERVED else observed_at
            rows.append((match["match_id"], market, line, sel, book, timing, price, seen, path.name))
    with db.transaction(con):
        con.executemany("INSERT OR IGNORE INTO market_odds VALUES(?,?,?,?,?,?,?,?,?)", rows)
    return len(rows), unmatched


def ingest_history(con, progress=None):
    con.executescript(SCHEMA)
    files = sorted((RAW / "fdco").glob("*/*.csv"))
    files = [f for f in files if f.parent.name != "fixtures" and f.stem in COMPETITIONS]
    total = unmatched = 0
    for i, f in enumerate(files):
        season = 2000 + int(f.parent.name[:2])
        n, u = ingest_file(con, f, f.stem, season)
        total += n
        unmatched += u
        if progress:
            progress(i + 1, len(files), f"{f.stem} {season}: {n} prices")
    return {"files": len(files), "prices": total, "unmatched_rows": unmatched}


def ingest_fixture_snapshot(con, path: Path, season: int):
    con.executescript(SCHEMA)
    from datetime import datetime, timezone
    frame = _read_csv(path.read_bytes())
    observed = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    total = 0
    for code in sorted(set(frame["Div"]) & set(COMPETITIONS)):
        n, _ = ingest_file(con, path, code, season, observed_at=observed, scheduled=True)
        total += n
    return {"prices": total, "observed_at": observed}
