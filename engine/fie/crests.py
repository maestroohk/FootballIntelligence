import json
import re
import time
from pathlib import Path

import requests

from . import db
from .config import ROOT
from .identity import normalise, similarity
from .jobs import handler

USER_AGENT = "FootballIntelligence/1.0 (personal football analytics; https://github.com/maestroohk/FootballIntelligence)"
OUT = ROOT / "src" / "FootballIntelligence.Web" / "wwwroot" / "crests"
COUNTRIES = {"ENG": ["Q145", "Q21"], "SCO": ["Q145", "Q22"], "ESP": ["Q29"], "GER": ["Q183"], "ITA": ["Q38"], "FRA": ["Q142"],
             "NED": ["Q55"], "POR": ["Q45"], "BEL": ["Q31"], "GRE": ["Q41"], "TUR": ["Q43"]}
LANGS = "en,de,es,it,fr,nl,pt,el,tr"
ACCEPTED = re.compile(r"^(pd|cc0|cc-by(-sa)?-[0-9.]+.*|cc-by(-sa)?)$")
LOGO_WORDS = re.compile(r"logo|crest|wappen|escudo|stemma|emblem|badge|blason|ecusson|embleem|logotipo|simbolo|arma", re.I)
OUTDATED = re.compile(r"former|avant|ancien|historic|\bold\b|flag|text logo|\((logo,? )?(19\d\d|200\d)\)|logo (19\d\d|200\d)\b", re.I)
EXCLUDED_CLUBS = {"Q982889"}
MIN_SCORE = 0.92
MIN_SITELINKS = 8


def _get(url, params, timeout=90, attempts=4):
    for attempt in range(attempts):
        try:
            r = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=timeout)
        except requests.ConnectionError:
            if attempt == attempts - 1:
                raise
            time.sleep(15 * (attempt + 1))
            continue
        if r.status_code in (429, 503) and attempt < attempts - 1:
            time.sleep(15 * (attempt + 1))
            continue
        r.raise_for_status()
        return r


def wikidata_clubs(country):
    values = " ".join(f"wd:{q}" for q in COUNTRIES[country])
    query = f"""SELECT ?club ?logo ?links (GROUP_CONCAT(DISTINCT ?name; separator="|") AS ?names) WHERE {{
        VALUES ?c {{ {values} }} ?club wdt:P31/wdt:P279* wd:Q476028; wdt:P17 ?c; wdt:P154 ?logo; wikibase:sitelinks ?links.
        FILTER(?links >= {MIN_SITELINKS})
        {{ ?club rdfs:label ?name }} UNION {{ ?club skos:altLabel ?name }}
        FILTER(LANG(?name) IN ("{'","'.join(LANGS.split(','))}")) }} GROUP BY ?club ?logo ?links"""
    rows = _get("https://query.wikidata.org/sparql", {"query": query, "format": "json"}).json()["results"]["bindings"]
    clubs = {}
    for b in rows:
        qid = b["club"]["value"].rsplit("/", 1)[-1]
        if qid in EXCLUDED_CLUBS:
            continue
        c = clubs.setdefault(qid, {"qid": qid, "logos": set(), "names": set()})
        c["logos"].add(requests.utils.unquote(b["logo"]["value"].rsplit("/", 1)[-1]))
        c["names"].update(n for n in b["names"]["value"].split("|") if n)
    return list(clubs.values())


def match_teams(teams, clubs):
    out = {}
    for t in teams:
        scored = []
        for c in clubs:
            best = max((max(similarity(a, n) for n in c["names"]) for a in t["aliases"]), default=0.0)
            scored.append((best, c["qid"]))
        scored.sort(reverse=True)
        if scored and scored[0][0] >= MIN_SCORE and (len(scored) == 1 or scored[0][0] - scored[1][0] >= 0.05):
            out[t["team_id"]] = (scored[0][1], scored[0][0])
    return out


def commons_info(files):
    info = {}
    files = list(files)
    for i in range(0, len(files), 40):
        batch = files[i:i + 40]
        r = _get("https://commons.wikimedia.org/w/api.php", {
            "action": "query", "format": "json", "prop": "imageinfo", "iiprop": "url|extmetadata|mime", "iiurlwidth": 128,
            "titles": "|".join(f"File:{f}" for f in batch)}).json()
        norm = {n["to"]: n["from"] for n in r["query"].get("normalized", [])}
        for page in r["query"]["pages"].values():
            if "imageinfo" not in page:
                continue
            ii = page["imageinfo"][0]
            meta = {k: v.get("value") for k, v in ii.get("extmetadata", {}).items()}
            title = norm.get(page["title"], page["title"]).removeprefix("File:")
            info[title] = {"thumb": ii.get("thumburl"), "page": ii.get("descriptionurl"), "mime": ii.get("mime"),
                           "license": (meta.get("License") or "").lower(), "license_name": meta.get("LicenseShortName"),
                           "artist": re.sub(r"<[^>]+>", "", meta.get("Artist") or "").strip(), "description": re.sub(r"<[^>]+>", "", meta.get("ImageDescription") or ""),
                           "restrictions": meta.get("Restrictions") or ""}
        time.sleep(1)
    return info


def acceptable(file, meta, names=()):
    if not meta or not meta.get("thumb") or meta.get("mime") not in ("image/svg+xml", "image/png"):
        return False, "not an SVG or PNG image"
    if not ACCEPTED.match(meta["license"]):
        return False, f"licence not accepted: {meta.get('license_name') or meta['license'] or 'unknown'}"
    tokens = set(normalise(file.rsplit(".", 1)[0]).split())
    named = any(set(normalise(n).split()) & tokens for n in names)
    if not (LOGO_WORDS.search(file) or (meta["mime"] == "image/svg+xml" and named)):
        return False, "file is not named as a crest or logo"
    if OUTDATED.search(file):
        return False, "file is a former crest, flag or wordmark"
    return True, ""


def display_keys(alias):
    a = alias.strip()
    keys = {a.lower()}
    for pre in ("AFC ", "FC "):
        if a.startswith(pre):
            keys.add(a[len(pre):].lower())
    if a.endswith(" FC"):
        keys.add(a[:-3].lower())
    return keys


def slug(name):
    return re.sub(r"[^a-z0-9]+", "-", normalise(name)).strip("-")


def active_teams(con, since):
    rows = con.execute("""SELECT t.team_id, t.name, t.country, GROUP_CONCAT(a.alias, '|') aliases FROM teams t
                          LEFT JOIN team_aliases a ON a.team_id = t.team_id
                          WHERE EXISTS (SELECT 1 FROM matches m WHERE t.team_id IN (m.home_team_id, m.away_team_id) AND m.season >= ?)
                          GROUP BY t.team_id""", (since,)).fetchall()
    return [{"team_id": r["team_id"], "name": r["name"], "country": r["country"],
             "aliases": sorted({r["name"], *(r["aliases"] or "").split("|")} - {""})} for r in rows]


def fetch(con, since=2024, progress=None):
    OUT.mkdir(parents=True, exist_ok=True)
    teams = active_teams(con, since)
    manifest = {"source": "Wikidata logo image (P154) files on Wikimedia Commons, licence checked per file", "teams": {}, "aliases": {}, "rejected": {}, "errors": {}}
    countries = sorted({t["country"] for t in teams if t["country"] in COUNTRIES})
    for i, country in enumerate(countries):
        group = [t for t in teams if t["country"] == country]
        try:
            fetch_country(country, group, manifest)
        except requests.RequestException as exc:
            manifest["errors"][country] = str(exc)[:200]
        if progress:
            progress(i + 1, len(countries), f"{country}: {sum(1 for t in group if str(t['team_id']) in manifest['teams'])}/{len(group)} crests")
    keep = {e["file"] for e in manifest["teams"].values()}
    if not manifest["errors"]:
        for f in OUT.glob("*.png"):
            if f.name not in keep:
                f.unlink()
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    summary = {"teams": len(teams), "with_crest": len(manifest["teams"]), "rejected": len(manifest["rejected"]), "errors": manifest["errors"]}
    db.set_kv(con, "crests", summary)
    return summary


def fetch_country(country, group, manifest):
    clubs = wikidata_clubs(country)
    matched = match_teams(group, clubs)
    files = {qid: sorted(next(c["logos"] for c in clubs if c["qid"] == qid)) for qid, _ in matched.values()}
    info = commons_info({f for fs in files.values() for f in fs})
    for t in group:
        if t["team_id"] not in matched:
            manifest["rejected"][t["name"]] = "no unambiguous Wikidata club with a Commons logo"
            continue
        qid, score = matched[t["team_id"]]
        reasons = []
        chosen = None
        names = next(c["names"] for c in clubs if c["qid"] == qid)
        for f in files[qid]:
            ok, why = acceptable(f, info.get(f), names)
            if ok:
                chosen = f
                break
            reasons.append(why)
        if chosen is None:
            manifest["rejected"][t["name"]] = "; ".join(sorted(set(reasons))) or "no logo file"
            continue
        meta = info[chosen]
        name = f"{country.lower()}-{slug(t['name'])}.png"
        path = OUT / name
        if not path.exists():
            img = _get(meta["thumb"], None, timeout=60)
            path.write_bytes(img.content)
            time.sleep(0.5)
        matched_name = max(((similarity(a, n), n) for a in t["aliases"] for n in names))[1]
        entry = {"file": name, "team": t["name"], "matched_name": matched_name, "wikidata": qid, "match_score": round(score, 3), "commons_file": chosen, "commons_page": meta["page"],
                 "licence": meta["license_name"], "author": meta["artist"] or None, "restrictions": meta["restrictions"] or None}
        manifest["teams"][str(t["team_id"])] = entry
        for a in t["aliases"]:
            for key in display_keys(a):
                manifest["aliases"].setdefault(key, name)


@handler("fetch_crests")
def fetch_crests(ctx, since=2024):
    ctx.stage("matching clubs to Wikidata and checking Commons licences")
    return fetch(ctx.con, since, progress=lambda d, t, s: ctx.progress(d, t, s))
