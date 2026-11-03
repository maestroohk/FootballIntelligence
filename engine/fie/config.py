import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
DB_PATH = Path(os.environ.get("FIE_DB", DATA / "football.db"))

PROFILE = os.environ.get("FIE_PROFILE", "full").strip().lower()
LIGHT = PROFILE == "light"
HEAVY_JOBS = {"ingest_history", "research_full", "expand_coverage", "final_test", "train_inplay", "experiments", "publish_predictions", "player_study"}

THREADS = int(os.environ.get("FIE_THREADS", "2" if LIGHT else "4"))
MAX_CONCURRENT_JOBS = int(os.environ.get("FIE_MAX_JOBS", "1" if LIGHT else "2"))
RESEARCH_PROCESSES = int(os.environ.get("FIE_RESEARCH_PROCESSES", str(max(1, min(3, (os.cpu_count() or 4) // 4)))))

FDCO_BASE = "https://www.football-data.co.uk/mmz4281"
FDCO_FIXTURES = "https://www.football-data.co.uk/fixtures.csv"
STATSBOMB_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

COUNTRIES = {
    "ENG": {"name": "England", "divisions": ["E0", "E1", "E2"]},
    "ESP": {"name": "Spain", "divisions": ["SP1", "SP2"]},
    "GER": {"name": "Germany", "divisions": ["D1", "D2"]},
    "ITA": {"name": "Italy", "divisions": ["I1", "I2"]},
    "FRA": {"name": "France", "divisions": ["F1", "F2"]},
    "NED": {"name": "Netherlands", "divisions": ["N1"]},
    "POR": {"name": "Portugal", "divisions": ["P1"]},
    "BEL": {"name": "Belgium", "divisions": ["B1"]},
    "SCO": {"name": "Scotland", "divisions": ["SC0", "SC1"]},
    "TUR": {"name": "Turkey", "divisions": ["T1"]},
    "GRE": {"name": "Greece", "divisions": ["G1"]},
}

COMPETITIONS = {
    "E0": ("Premier League", "ENG", 1),
    "E1": ("Championship", "ENG", 2),
    "E2": ("League One", "ENG", 3),
    "SP1": ("La Liga", "ESP", 1),
    "SP2": ("Segunda Division", "ESP", 2),
    "D1": ("Bundesliga", "GER", 1),
    "D2": ("2. Bundesliga", "GER", 2),
    "I1": ("Serie A", "ITA", 1),
    "I2": ("Serie B", "ITA", 2),
    "F1": ("Ligue 1", "FRA", 1),
    "F2": ("Ligue 2", "FRA", 2),
    "N1": ("Eredivisie", "NED", 1),
    "P1": ("Primeira Liga", "POR", 1),
    "B1": ("Pro League", "BEL", 1),
    "SC0": ("Scottish Premiership", "SCO", 1),
    "SC1": ("Scottish Championship", "SCO", 2),
    "T1": ("Super Lig", "TUR", 1),
    "G1": ("Super League Greece", "GRE", 1),
}

CORE_COMPETITIONS = ["E0", "E1", "E2", "SP1", "SP2", "D1", "D2", "I1", "I2", "F1", "F2"]
EXPANSION_COMPETITIONS = ["N1", "P1", "B1", "SC0", "SC1", "T1", "G1"]

PRIMARY_COMPETITIONS = ["E0", "E1", "SP1", "D1", "I1", "F1"]

FIRST_SEASON = 2005
CURRENT_SEASON = 2026

VALIDATION_SEASONS = list(range(2012, 2023))
FINAL_TEST_SEASONS = [2023, 2024, 2025]
FORWARD_SEASON = 2026


def season_code(start_year: int) -> str:
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_label(start_year: int) -> str:
    return f"{start_year}/{(start_year + 1) % 100:02d}"


def limit_threads():
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, str(THREADS))
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        if os.name == "nt":
            import ctypes
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.kernel32.SetPriorityClass(handle, 0x00004000)
