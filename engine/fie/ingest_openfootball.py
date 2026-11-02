import requests

from . import db
from .config import COMPETITIONS, CURRENT_SEASON
from .identity import normalise, resolve_team

SOURCE = "OpenFootball"
BASE = "https://raw.githubusercontent.com/openfootball/football.json/master"
FILES = {"E0": "en.1", "E1": "en.2", "SP1": "es.1", "D1": "de.1", "I1": "it.1", "F1": "fr.1", "N1": "nl.1", "P1": "pt.1"}

ALIASES = {
    "Manchester United FC": "Man United", "Manchester City FC": "Man City", "Nottingham Forest FC": "Nott'm Forest",
    "Wolverhampton Wanderers FC": "Wolves", "Sheffield Wednesday FC": "Sheffield Weds", "Brighton & Hove Albion FC": "Brighton",
    "Queens Park Rangers FC": "QPR", "West Bromwich Albion FC": "West Brom", "AFC Bournemouth": "Bournemouth",
    "Athletic Club": "Ath Bilbao", "Club Atlético de Madrid": "Ath Madrid", "Atlético Madrid": "Ath Madrid", "RC Celta de Vigo": "Celta",
    "Real Betis Balompié": "Betis", "RCD Espanyol de Barcelona": "Espanol", "Rayo Vallecano de Madrid": "Vallecano", "Real Sociedad de Fútbol": "Sociedad",
    "Deportivo Alavés": "Alaves", "RCD Mallorca": "Mallorca", "CA Osasuna": "Osasuna", "Real Oviedo": "Oviedo",
    "FC Bayern München": "Bayern Munich", "Borussia Dortmund": "Dortmund", "Borussia Mönchengladbach": "M'gladbach", "Bayer 04 Leverkusen": "Leverkusen",
    "Eintracht Frankfurt": "Ein Frankfurt", "1. FC Köln": "FC Koln", "Hamburger SV": "Hamburg", "TSG 1899 Hoffenheim": "Hoffenheim", "1. FSV Mainz 05": "Mainz",
    "FC St. Pauli": "St Pauli", "1. FC Union Berlin": "Union Berlin", "1. FC Heidenheim 1846": "Heidenheim", "SV Werder Bremen": "Werder Bremen",
    "AC Milan": "Milan", "FC Internazionale Milano": "Inter", "AS Roma": "Roma", "Hellas Verona FC": "Verona", "SSC Napoli": "Napoli",
    "Paris Saint-Germain FC": "Paris SG", "Olympique Lyonnais": "Lyon", "Olympique de Marseille": "Marseille", "AS Saint-Étienne": "St Etienne",
    "Stade Rennais FC 1901": "Rennes", "OGC Nice": "Nice", "AS Monaco FC": "Monaco", "Lille OSC": "Lille", "RC Strasbourg Alsace": "Strasbourg",
    "Stade Brestois 29": "Brest", "RC Lens": "Lens", "Paris FC": "Paris FC",
    "PSV": "PSV Eindhoven", "PSV Eindhoven": "PSV Eindhoven", "AFC Ajax": "Ajax", "FC Twente": "Twente", "FC Utrecht": "Utrecht", "NEC": "Nijmegen",
    "NEC Nijmegen": "Nijmegen", "SC Heerenveen": "Heerenveen", "sc Heerenveen": "Heerenveen", "Fortuna Sittard": "For Sittard", "PEC Zwolle": "Zwolle",
    "Heracles Almelo": "Heracles", "FC Groningen": "Groningen", "AZ": "AZ Alkmaar", "AZ Alkmaar": "AZ Alkmaar", "Feyenoord": "Feyenoord",
    "SL Benfica": "Benfica", "Sporting CP": "Sp Lisbon", "Sporting Clube de Portugal": "Sp Lisbon", "Sporting Clube de Braga": "Sp Braga", "FC Porto": "Porto", "SC Braga": "Sp Braga", "Vitória SC": "Guimaraes", "Vitória Guimarães": "Guimaraes",
    "Gil Vicente FC": "Gil Vicente", "Rio Ave FC": "Rio Ave", "Moreirense FC": "Moreirense", "FC Famalicão": "Famalicao", "Casa Pia AC": "Casa Pia",
    "CD Santa Clara": "Santa Clara", "GD Estoril Praia": "Estoril", "FC Arouca": "Arouca", "CD Nacional": "Nacional", "AVS Futebol SAD": "AVS",
}


def ingest(con, season=CURRENT_SEASON):
    label = f"{season}-{(season + 1) % 100:02d}"
    con.execute("INSERT OR REPLACE INTO sources VALUES(?,?,?,?,?)",
                (SOURCE, "https://github.com/openfootball/football.json", "Public domain (per repository README)",
                 "Season fixture lists with dates, kick-off times and (lagging) results for top leagues",
                 "Results lag behind other sources; used here for fixtures only; team names resolved to football-data.co.uk identities"))
    stamp = db.now()
    summary = {}
    for code, fname in FILES.items():
        country = COMPETITIONS[code][1]
        try:
            resp = requests.get(f"{BASE}/{label}/{fname}.json", timeout=60)
            resp.raise_for_status()
            db.provider_event(con, SOURCE, True)
        except Exception as exc:
            db.provider_event(con, SOURCE, False, str(exc)[:300])
            summary[code] = {"error": str(exc)}
            continue
        pool = [(r["team_id"], r["name"]) for r in con.execute(
            "SELECT DISTINCT t.team_id, t.name FROM matches m JOIN teams t ON t.team_id IN (m.home_team_id, m.away_team_id) WHERE m.competition=? AND m.season=?",
            (code, season))]
        by_name = {name: tid for tid, name in pool}
        mapping = {}
        unresolved = set()
        for m in resp.json().get("matches", []):
            for name in (m["team1"], m["team2"]):
                if name in mapping or name in unresolved:
                    continue
                if name in ALIASES and ALIASES[name] in by_name:
                    mapping[name] = by_name[ALIASES[name]]
                    con.execute("INSERT OR REPLACE INTO team_aliases VALUES(?,?,?,?,?)", (name, "openfootball", country, mapping[name], "manual"))
                    continue
                tid, _ = resolve_team(con, name, "openfootball", country, pool, min_score=0.8)
                if tid is None:
                    unresolved.add(name)
                else:
                    mapping[name] = tid
        checked = agreed = inserted = 0
        with db.transaction(con):
            for m in resp.json().get("matches", []):
                h, a = mapping.get(m["team1"]), mapping.get(m["team2"])
                if h is None or a is None:
                    continue
                from .ingest_fdco import find_existing
                row = find_existing(con, code, season, h, a, m["date"], "scheduled")
                sc = m.get("score")
                ft = sc.get("ft") if isinstance(sc, dict) else (sc[:2] if isinstance(sc, list) and len(sc) >= 2 else None)
                if row is not None:
                    if ft and row["status"] == "finished":
                        checked += 1
                        agreed += (row["fthg"], row["ftag"]) == tuple(ft)
                    continue
                if ft:
                    continue
                con.execute("INSERT INTO matches(competition, season, match_date, kickoff, home_team_id, away_team_id, status, source, first_seen, updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                            (code, season, m["date"], m.get("time"), h, a, "scheduled", SOURCE, stamp, stamp))
                inserted += 1
        summary[code] = {"teams_mapped": len(mapping), "unresolved": sorted(unresolved), "results_cross_checked": checked, "results_agree": agreed, "fixtures_added": inserted}
        if checked and agreed < checked:
            db.notify(con, "warning", f"OpenFootball {code}: {checked - agreed} result disagreements", "Check team identity mapping")
    return summary
