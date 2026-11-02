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

STOP_TOKENS = {"fc", "cf", "afc", "sc", "ac", "as", "ss", "us", "ud", "cd", "rc", "rcd", "sv", "vfl", "vfb", "tsg", "fsv", "club", "de", "the", "calcio", "1", "04", "05", "96", "98", "1899"}


def normalise(name: str) -> str:
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    tokens = [t for t in text.split() if t not in STOP_TOKENS]
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


def get_or_create_team(con, name: str, country: str, source: str) -> int:
    row = con.execute("SELECT team_id FROM team_aliases WHERE alias=? AND source=? AND country=?", (name, source, country)).fetchone()
    if row:
        return row["team_id"]
    con.execute("INSERT OR IGNORE INTO teams(name, country) VALUES(?,?)", (name, country))
    team_id = con.execute("SELECT team_id FROM teams WHERE name=? AND country=?", (name, country)).fetchone()["team_id"]
    con.execute("INSERT OR REPLACE INTO team_aliases VALUES(?,?,?,?,?)", (name, source, country, team_id, "canonical"))
    return team_id
