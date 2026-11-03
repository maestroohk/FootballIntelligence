import re
import unicodedata
from difflib import SequenceMatcher

MANUAL_ALIASES = {
    ("statsbomb", "ENG"): {
        "Manchester United": "Man United",
        "Manchester City": "Man City",
        "Tottenham Hotspur": "Tottenham",
        "West Ham United": "West Ham",
        "Newcastle United": "Newcastle",
        "Leicester City": "Leicester",
        "Swansea City": "Swansea",
        "Stoke City": "Stoke",
        "Norwich City": "Norwich",
        "West Bromwich Albion": "West Brom",
        "AFC Bournemouth": "Bournemouth",
        "Watford": "Watford",
    },
    ("statsbomb", "ESP"): {
        "Athletic Club": "Ath Bilbao",
        "Atlético Madrid": "Ath Madrid",
        "Real Betis": "Betis",
        "Celta Vigo": "Celta",
        "RC Deportivo La Coruña": "La Coruna",
        "Deportivo La Coruña": "La Coruna",
        "Espanyol": "Espanol",
        "RCD Espanyol": "Espanol",
        "Rayo Vallecano": "Vallecano",
        "Real Sociedad": "Sociedad",
        "Sporting Gijón": "Sp Gijon",
        "Las Palmas": "Las Palmas",
        "UD Las Palmas": "Las Palmas",
        "Real Madrid": "Real Madrid",
        "Barcelona": "Barcelona",
        "Villarreal": "Villarreal",
        "Levante UD": "Levante",
        "Getafe": "Getafe",
        "Granada": "Granada",
        "Eibar": "Eibar",
        "Málaga": "Malaga",
        "Sevilla": "Sevilla",
        "Valencia": "Valencia",
    },
    ("statsbomb", "GER"): {
        "Bayern Munich": "Bayern Munich",
        "Borussia Dortmund": "Dortmund",
        "Borussia Mönchengladbach": "M'gladbach",
        "Bayer Leverkusen": "Leverkusen",
        "Eintracht Frankfurt": "Ein Frankfurt",
        "FC Köln": "FC Koln",
        "1. FC Köln": "FC Koln",
        "Hertha Berlin": "Hertha",
        "Hertha BSC": "Hertha",
        "Hamburger SV": "Hamburg",
        "Hannover 96": "Hannover",
        "TSG 1899 Hoffenheim": "Hoffenheim",
        "Hoffenheim": "Hoffenheim",
        "FSV Mainz 05": "Mainz",
        "Mainz 05": "Mainz",
        "FC Augsburg": "Augsburg",
        "Augsburg": "Augsburg",
        "VfB Stuttgart": "Stuttgart",
        "VfL Wolfsburg": "Wolfsburg",
        "Werder Bremen": "Werder Bremen",
        "SV Darmstadt 98": "Darmstadt",
        "Darmstadt 98": "Darmstadt",
        "FC Ingolstadt": "Ingolstadt",
        "FC Ingolstadt 04": "Ingolstadt",
        "Schalke 04": "Schalke 04",
    },
    ("statsbomb", "ITA"): {
        "AC Milan": "Milan",
        "Internazionale": "Inter",
        "Inter Milan": "Inter",
        "AS Roma": "Roma",
        "Hellas Verona": "Verona",
        "Chievo": "Chievo",
        "Chievo Verona": "Chievo",
        "Frosinone": "Frosinone",
        "Carpi": "Carpi",
        "Sassuolo": "Sassuolo",
    },
    ("statsbomb", "FRA"): {
        "Paris Saint-Germain": "Paris SG",
        "Olympique Lyonnais": "Lyon",
        "Olympique Marseille": "Marseille",
        "Saint-Étienne": "St Etienne",
        "AS Saint-Étienne": "St Etienne",
        "AS Monaco": "Monaco",
        "OGC Nice": "Nice",
        "Stade Rennais": "Rennes",
        "Rennes": "Rennes",
        "Girondins Bordeaux": "Bordeaux",
        "Bordeaux": "Bordeaux",
        "Lille": "Lille",
        "LOSC Lille": "Lille",
        "Montpellier": "Montpellier",
        "Montpellier HSC": "Montpellier",
        "Gazélec Ajaccio": "Ajaccio GFCO",
        "GFC Ajaccio": "Ajaccio GFCO",
        "Troyes": "Troyes",
        "ESTAC Troyes": "Troyes",
        "Stade de Reims": "Reims",
        "Reims": "Reims",
        "SM Caen": "Caen",
        "Caen": "Caen",
        "Angers": "Angers",
        "Angers SCO": "Angers",
        "Guingamp": "Guingamp",
        "Lorient": "Lorient",
        "Nantes": "Nantes",
        "Bastia": "Bastia",
        "Toulouse": "Toulouse",
    },
}

SYNONYMS = {"utd": "united", "weds": "wednesday", "st": "saint"}

STOP_TOKENS = {"fc", "cf", "afc", "kv", "sc", "ac", "as", "ss", "us", "ud", "cd", "rc", "rcd", "sv", "vfl", "vfb", "tsg", "fsv", "club", "de", "the", "calcio", "1", "04", "05", "96", "98", "1899"}


def normalise(name: str) -> str:
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    tokens = [SYNONYMS.get(t, t) for t in text.split() if t not in STOP_TOKENS]
    return " ".join(tokens)


def similarity(a: str, b: str) -> float:
    na, nb = normalise(a), normalise(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    ratio = SequenceMatcher(None, na, nb).ratio()
    ta, tb = set(na.split()), set(nb.split())
    overlap = len(ta & tb) / max(1, min(len(ta), len(tb)))
    return max(ratio, 0.9 * overlap)


def resolve_team(con, alias: str, source: str, country: str, candidates=None, min_score=0.72):
    row = con.execute("SELECT team_id FROM team_aliases WHERE alias=? AND source=? AND country=?", (alias, source, country)).fetchone()
    if row:
        return row["team_id"], "cached"
    manual = MANUAL_ALIASES.get((source, country), {}).get(alias)
    if manual:
        team = con.execute("SELECT team_id FROM teams WHERE name=? AND country=?", (manual, country)).fetchone()
        if team:
            con.execute("INSERT OR REPLACE INTO team_aliases VALUES(?,?,?,?,?)", (alias, source, country, team["team_id"], "manual"))
            return team["team_id"], "manual"
    pool = candidates if candidates is not None else [(r["team_id"], r["name"]) for r in con.execute("SELECT team_id, name FROM teams WHERE country=?", (country,))]
    scored = sorted(((similarity(alias, name), tid, name) for tid, name in pool), reverse=True)
    if scored and scored[0][0] >= min_score and (len(scored) == 1 or scored[0][0] - scored[1][0] >= 0.05):
        con.execute("INSERT OR REPLACE INTO team_aliases VALUES(?,?,?,?,?)", (alias, source, country, scored[0][1], f"fuzzy:{scored[0][0]:.2f}"))
        return scored[0][1], "fuzzy"
    return None, "unresolved"


def get_or_create_team(con, name: str, country: str, source: str, season=None) -> int:
    row = con.execute("SELECT team_id FROM team_aliases WHERE alias=? AND source=? AND country=?", (name, source, country)).fetchone()
    if row:
        return row["team_id"]
    if season is not None:
        active = [(r["team_id"], r["name"]) for r in con.execute("""SELECT DISTINCT t.team_id, t.name FROM teams t JOIN matches m ON t.team_id IN (m.home_team_id, m.away_team_id)
                                                                    WHERE t.country = ? AND m.season >= ? AND m.status = 'finished'""", (country, season - 1))]
        tid, _ = resolve_team(con, name, source, country, active, min_score=0.85)
        if tid is not None:
            return tid
        from . import db
        db.notify(con, "warning", f"New team created: {name}", f"{country}: no existing team matched this {source} name, so it starts without history")
    con.execute("INSERT OR IGNORE INTO teams(name, country) VALUES(?,?)", (name, country))
    team_id = con.execute("SELECT team_id FROM teams WHERE name=? AND country=?", (name, country)).fetchone()["team_id"]
    con.execute("INSERT OR REPLACE INTO team_aliases VALUES(?,?,?,?,?)", (name, source, country, team_id, "canonical"))
    return team_id


def repair_new_teams(con):
    from . import db
    stamp = db.now()
    merged = []
    candidates = con.execute("""SELECT t.team_id, t.name, t.country FROM teams t
                                WHERE NOT EXISTS (SELECT 1 FROM matches m WHERE t.team_id IN (m.home_team_id, m.away_team_id) AND m.status = 'finished')
                                  AND EXISTS (SELECT 1 FROM matches m WHERE t.team_id IN (m.home_team_id, m.away_team_id) AND m.status = 'scheduled')""").fetchall()
    for t in candidates:
        season = con.execute("SELECT MAX(season) FROM matches WHERE ? IN (home_team_id, away_team_id)", (t["team_id"],)).fetchone()[0]
        active = [(r["team_id"], r["name"]) for r in con.execute("""SELECT DISTINCT tt.team_id, tt.name FROM teams tt JOIN matches m ON tt.team_id IN (m.home_team_id, m.away_team_id)
                                                                     WHERE tt.country = ? AND m.season >= ? AND m.status = 'finished'""", (t["country"], season - 1))]
        scored = sorted(((similarity(t["name"], name), tid) for tid, name in active), reverse=True)
        if not scored or scored[0][0] < 0.85 or (len(scored) > 1 and scored[0][0] - scored[1][0] < 0.05):
            continue
        real = scored[0][1]
        con.execute("UPDATE team_aliases SET team_id = ? WHERE team_id = ?", (real, t["team_id"]))
        for m in con.execute("SELECT * FROM matches WHERE status = 'scheduled' AND ? IN (home_team_id, away_team_id)", (t["team_id"],)).fetchall():
            home = real if m["home_team_id"] == t["team_id"] else m["home_team_id"]
            away = real if m["away_team_id"] == t["team_id"] else m["away_team_id"]
            con.execute("UPDATE matches SET status = 'duplicate', updated_at = ? WHERE match_id = ?", (stamp, m["match_id"]))
            con.execute("""INSERT OR IGNORE INTO prediction_flags(prediction_id, flag, reason, flagged_at)
                           SELECT prediction_id, 'team_identity', 'a team was stored under a new name without history', ? FROM predictions WHERE match_id = ?""", (stamp, m["match_id"]))
            twin = con.execute("SELECT 1 FROM matches WHERE competition = ? AND season = ? AND home_team_id = ? AND away_team_id = ? AND status IN ('scheduled', 'finished')",
                               (m["competition"], m["season"], home, away)).fetchone()
            if not twin:
                con.execute("""INSERT INTO matches(competition, season, match_date, kickoff, home_team_id, away_team_id, status, source, first_seen, updated_at)
                               VALUES(?,?,?,?,?,?,'scheduled',?,?,?)""", (m["competition"], m["season"], m["match_date"], m["kickoff"], home, away, m["source"], stamp, stamp))
        merged.append({"from": t["name"], "to": con.execute("SELECT name FROM teams WHERE team_id = ?", (real,)).fetchone()[0]})
    return merged
