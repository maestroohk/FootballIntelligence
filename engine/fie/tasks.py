import json
import os
import time

import numpy as np
import pandas as pd

from . import backtest, db, ingest_fdco, ingest_statsbomb, research
from .config import FINAL_TEST_SEASONS, FORWARD_SEASON, PRIMARY_COMPETITIONS, VALIDATION_SEASONS
from .jobs import Cancelled, handler

NOTIFY_KINDS = {
    "ingest_history": "Historical ingestion", "ingest_statsbomb": "StatsBomb ingestion", "research_full": "Model research run",
    "train_inplay": "In-play model training", "final_test": "Final test evaluation", "daily_cycle": "Daily update cycle",
    "predict_upcoming": "Upcoming predictions",
}

PRIMARY = PRIMARY_COMPETITIONS


def _check(ctx):
    if ctx.cancelled():
        raise Cancelled()


@handler("ingest_history")
def ingest_history(ctx, seasons=None, codes=None):
    ctx.stage("downloading football-data.co.uk CSVs")
    return ingest_fdco.ingest_history(ctx.con, seasons=seasons, codes=codes, progress=lambda d, t, s: (ctx.progress(d, t, s), _check(ctx)))


@handler("ingest_fixtures")
def ingest_fixtures(ctx):
    ctx.stage("downloading upcoming fixtures")
    return ingest_fdco.ingest_fixtures(ctx.con)


@handler("ingest_statsbomb")
def ingest_sb(ctx):
    ctx.stage("downloading StatsBomb events")
    return ingest_statsbomb.ingest(ctx.con, progress=lambda d, t, s: ctx.progress(d, t, s), cancelled=ctx.cancelled)


STRENGTH_CONFIGS = {
    "poisson": {"dixon_coles": False},
    "dixon_coles": {"dixon_coles": True},
    "dc_xi0010": {"dixon_coles": True, "xi": 0.0010},
    "dc_xi0030": {"dixon_coles": True, "xi": 0.0030},
    "dc_ridge3": {"dixon_coles": True, "ridge": 3.0},
    "dc_xg25": {"dixon_coles": True, "xg_weight": 0.25},
    "dc_xg50": {"dixon_coles": True, "xg_weight": 0.5},
    "dc_xg75": {"dixon_coles": True, "xg_weight": 0.75},
}

ELO_CONFIGS = {"elo": {"k": 20.0}, "elo_k15": {"k": 15.0}, "elo_k30": {"k": 30.0}}


def strength_oos(ctx, frame, name, refresh=False):
    cached = None if refresh else research.load_cached(name)
    if cached is not None:
        return cached
    ctx.stage(f"walk-forward {name}")
    out = research.run_strength(frame, name, progress=lambda d, t, s: (ctx.progress(d, t, s), _check(ctx)), **STRENGTH_CONFIGS[name])
    research.save_cached(name, out)
    return out


def _strength_worker(name):
    from .config import limit_threads
    limit_threads()
    frame = research.load_cached("frame")
    out = research.run_strength(frame, name, **STRENGTH_CONFIGS[name])
    research.save_cached(name, out)
    return name


def elo_oos(ctx, frame, name, refresh=False):
    cached = None if refresh else research.load_cached(name)
    cached_r = None if refresh else research.load_cached(name + "_ratings")
    if cached is not None and cached_r is not None:
        return cached, cached_r
    ctx.stage(f"walk-forward {name}")
    out, ratings = research.run_elo(frame, name=name, **ELO_CONFIGS[name])
    research.save_cached(name, out)
    research.save_cached(name + "_ratings", ratings)
    return out, ratings


def _val(oos):
    return backtest.evaluate(oos, split="validation", competitions=PRIMARY)


@handler("research_full")
def research_full(ctx, refresh=False, configs=None):
    con = ctx.con
    ctx.stage("loading matches")
    frame, coef = research.prepared_frame(con)
    research.save_cached("frame", frame)
    db.set_kv(con, "xg_proxy", coef)
    results = {}
    oos = {}
    ctx.stage("frequency baseline")
    oos["frequency"] = research.run_frequency(frame)
    for name in ELO_CONFIGS:
        oos[name], ratings = elo_oos(ctx, frame, name, refresh)
        _check(ctx)
    names = list(configs or STRENGTH_CONFIGS)
    missing = [n for n in names if refresh or research.load_cached(n) is None]
    if missing:
        ctx.stage(f"walk-forward strength models in parallel: {', '.join(missing)}")
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(_strength_worker, n): n for n in missing}
            for k, fut in enumerate(as_completed(futures)):
                fut.result()
                ctx.progress(k + 1, len(missing), f"finished {futures[fut]}")
                _check(ctx)
    for name in names:
        oos[name] = research.load_cached(name)
    for name, o in oos.items():
        results[name] = _val(o)
    best_elo = min(ELO_CONFIGS, key=lambda n: results[n]["log_loss"])
    dc_variants = [n for n in STRENGTH_CONFIGS if n.startswith("dc_xi") or n.startswith("dc_ridge") or n == "dixon_coles"]
    best_dc = min(dc_variants, key=lambda n: results[n]["score_nll"])
    xg_variants = [n for n in STRENGTH_CONFIGS if n.startswith("dc_xg")]
    best_xg = min(xg_variants, key=lambda n: results[n]["score_nll"])
    ctx.log(f"best elo={best_elo} best dc={best_dc} best xg={best_xg}")
    _, elo_ratings = elo_oos(ctx, frame, best_elo)
    ctx.stage("feature table")
    features = research.feature_table(frame, elo_ratings, oos[best_dc], home_adv=65.0)
    research.save_cached("features", features)
    def cached(name, fn):
        hit = None if refresh else research.load_cached(f"oos_{name}")
        if hit is not None:
            return hit
        ctx.stage(name)
        return fn()

    oos["gbm_stacked"] = cached("gbm_stacked", lambda: research.run_gbm(features, oos[best_dc], name="gbm_stacked", use_dc=True))
    _check(ctx)
    oos["gbm_features"] = cached("gbm_features", lambda: research.run_gbm(features, oos[best_dc], name="gbm_features", use_dc=False))
    _check(ctx)
    oos["logistic"] = cached("logistic", lambda: research.run_logistic(features))
    ctx.stage("ensemble weight learning")
    comp_names = [best_dc, best_xg, best_elo, "gbm_features"]
    ens, weights = research.run_ensemble([oos[n] for n in comp_names], comp_names, frame)
    oos["ensemble"] = ens
    db.set_kv(con, "ensemble_weights", {"components": comp_names, "by_season": weights})
    ctx.stage("bookmaker benchmark")
    oos["bookmaker_close"] = research.bookmaker_oos(con, frame)
    for name in ("gbm_stacked", "gbm_features", "logistic", "ensemble", "bookmaker_close"):
        results[name] = _val(oos[name])
    for name, o in oos.items():
        research.save_cached(f"oos_{name}", o)
    ctx.stage("paired comparisons")
    comparisons = {}
    ladder = [("frequency", best_elo), (best_elo, "poisson"), ("poisson", "dixon_coles"), ("dixon_coles", best_dc),
              (best_dc, best_xg), (best_xg, "gbm_features"), (best_dc, "gbm_stacked"), (best_dc, "ensemble"), (best_xg, "ensemble"),
              ("logistic", best_dc), ("ensemble", "bookmaker_close")]
    for a, b in ladder:
        if a == b:
            continue
        for metric in ("log_loss", "rps"):
            comparisons[f"{a} vs {b} [{metric}]"] = backtest.compare(oos[a], oos[b], "validation", PRIMARY, metric)
        if a != "logistic" and b not in ("bookmaker_close", "logistic"):
            comparisons[f"{a} vs {b} [score_nll]"] = backtest.compare(oos[a], oos[b], "validation", PRIMARY, "score_nll")
    db.set_kv(con, "validation_comparisons", comparisons)
    ctx.stage("storing backtest runs")
    con.execute("DELETE FROM backtest_runs WHERE split='validation'")
    for name, o in oos.items():
        research.store_backtest(con, name, o, "validation", PRIMARY, "rolling, data before each weekly block" if name not in ("frequency", "bookmaker_close") else "prior seasons")
    importance = research.run_gbm_importance(features, use_dc=False)
    db.set_kv(con, "gbm_importance", importance)
    selection = select_champion(results, comparisons, best_dc, best_xg)
    db.set_kv(con, "research_summary", {"results": results, "best": {"elo": best_elo, "dc": best_dc, "xg": best_xg}, "selection": selection,
                                         "xg_proxy": coef, "finished_at": db.now()})
    register_versions(con, results, selection, best_dc, best_xg, best_elo)
    return {"selection": selection, "validation_log_loss": {k: round(v["log_loss"], 4) for k, v in results.items() if v}}


def select_champion(results, comparisons, best_dc, best_xg):
    candidates = ["ensemble", best_xg, "gbm_stacked"]
    champion = best_dc
    reasons = [f"{best_dc} is the baseline champion"]
    for c in candidates:
        key = f"{champion} vs {c} [score_nll]"
        alt = f"{c} vs {champion} [score_nll]"
        cmp = comparisons.get(key)
        if cmp is None and alt in comparisons:
            raw = comparisons[alt]
            cmp = {"mean_diff": -raw["mean_diff"], "ci_low": -raw["ci_high"], "ci_high": -raw["ci_low"]}
        if cmp is None:
            continue
        ll = comparisons.get(f"{champion} vs {c} [log_loss]")
        if cmp["ci_low"] > 0 and (ll is None or ll["mean_diff"] > 0):
            reasons.append(f"{c} beats {champion} on score NLL (diff {cmp['mean_diff']:.4f}, 95% CI {cmp['ci_low']:.4f}..{cmp['ci_high']:.4f})")
            champion = c
        else:
            reasons.append(f"{c} not promoted over {champion}: score NLL diff {cmp['mean_diff']:.4f}, CI {cmp['ci_low']:.4f}..{cmp['ci_high']:.4f}")
    return {"champion": champion, "reasons": reasons}


def register_versions(con, results, selection, best_dc, best_xg, best_elo):
    for name, res in results.items():
        if res is None:
            continue
        family = "benchmark" if name == "bookmaker_close" else name.split("_")[0]
        status = "champion" if name == selection["champion"] else "benchmark" if name == "bookmaker_close" else "challenger"
        version = f"{name}@v1"
        params = STRENGTH_CONFIGS.get(name) or ELO_CONFIGS.get(name) or {}
        con.execute("INSERT INTO model_versions(version, family, params, trained_through, created_at, status, notes) VALUES(?,?,?,?,?,?,?) "
                    "ON CONFLICT(version) DO UPDATE SET status=excluded.status, params=excluded.params",
                    (version, family, db.dumps(params), "walk-forward", db.now(), status, None))


def _champion_version(con):
    summary = db.get_kv(con, "research_summary")
    return summary["selection"]["champion"], f"{summary['selection']['champion']}@v1"


@handler("publish_predictions")
def publish_predictions(ctx):
    from . import forward
    con = ctx.con
    champ, version = _champion_version(con)
    ctx.stage(f"publishing backtest predictions for {version}")
    oos = research.load_cached(f"oos_{champ}")
    n_back = forward.publish_backtest(con, oos, version)
    bench = research.load_cached("oos_bookmaker_close")
    n_bench = forward.publish_backtest(con, bench, "bookmaker_close@v1") if bench is not None else 0
    ctx.stage("publishing forward predictions")
    n_fwd = forward.publish_forward(con, oos, version)
    ctx.stage("team strengths")
    frame = research.load_cached("frame")
    strengths = forward.team_strengths(con, frame, db.get_kv(con, "research_summary"))
    return {"backtest": n_back, "benchmark": n_bench, "forward": n_fwd, "strengths": strengths}


@handler("predict_upcoming")
def predict_upcoming(ctx, lookback_days=21):
    from . import forward
    con = ctx.con
    champ, version = _champion_version(con)
    oos = forward.refresh_current(ctx, lookback_days)
    ctx.stage("storing forward predictions")
    weights = db.get_kv(con, "ensemble_weights")
    comps = (weights["components"], weights["by_season"][str(CURRENT_SEASON_PUBLISH)], oos) if champ == "ensemble" else None
    n_fwd = forward.publish_forward(con, oos[champ] if champ in oos else research.load_cached(f"oos_{champ}"), version, comps)
    n_back = forward.publish_backtest(con, research.load_cached(f"oos_{champ}"), version, since_season=CURRENT_SEASON_PUBLISH)
    forward.team_strengths(con, research.load_cached("frame"), db.get_kv(con, "research_summary"))
    if n_fwd:
        db.notify(con, "success", f"{n_fwd} new forward predictions stored", version)
    return {"forward": n_fwd, "backtest_new": n_back}


CURRENT_SEASON_PUBLISH = 2026


@handler("evaluate_forward")
def evaluate_forward(ctx):
    from . import forward
    ctx.stage("comparing stored forward predictions with results")
    perf = forward.evaluate_forward(ctx.con)
    ctx.stage("drift detection")
    drift = forward.drift_check(ctx.con)
    return {"forward": perf, "drift_alarm": drift["alarm"], "recent_log_loss": drift["recent_log_loss"]}


@handler("daily_cycle")
def daily_cycle(ctx, trigger="manual"):
    from datetime import datetime
    ctx.stage("ingesting current-season results")
    res = ingest_fdco.ingest_history(ctx.con, seasons=[CURRENT_SEASON_PUBLISH], progress=lambda d, t, s: ctx.progress(d, t, s))
    ctx.stage("ingesting fixtures")
    try:
        fx = ingest_fdco.ingest_fixtures(ctx.con)
    except Exception as exc:
        ctx.log(f"fixtures failed: {exc}", "warning")
        fx = None
    out = {"results": {k: v for k, v in res.items() if k != "failed"}, "fixtures": fx}
    out["evaluation"] = evaluate_forward(ctx)
    out["prediction"] = predict_upcoming(ctx)
    from .live import poll_live_provider
    out["live"] = poll_live_provider(ctx.con)
    return out


@handler("poll_live")
def poll_live(ctx):
    from .live import poll_live_provider
    return poll_live_provider(ctx.con)


@handler("live_replay")
def live_replay(ctx, sb_match_id, speed=60.0, interval_minutes=1.0, faults=True):
    from .live import run_replay
    return run_replay(ctx, int(sb_match_id), float(speed), float(interval_minutes), bool(faults))


@handler("train_inplay")
def train_inplay(ctx):
    from . import inplay_train
    con = ctx.con
    champ, version = _champion_version(con)
    oos = research.load_cached(f"oos_{champ}")
    prematch = oos[["match_id", "lh", "la"]]
    res = inplay_train.train_and_evaluate(con, prematch, progress=lambda d, t, s: ctx.progress(d, t, s))
    ctx.stage("half-time evaluation on football-data.co.uk HT scores")
    from .inplay import InPlayModel
    hazard = InPlayModel.from_json(res["final_model"]) if not res["final_model"]["use_evidence"] else None
    if hazard is None:
        base = inplay_train.InPlayModel(use_evidence=False)
        matches, grouped = inplay_train.load_sb(con, prematch)
        hazard = base.fit(inplay_train.build_panel(matches, grouped, res["minutes_per_match"]), tuple(res["half_end"]))
    ht = inplay_train.halftime_eval(oos, hazard, res["minutes_per_match"], tuple(res["half_end"]), seasons=[2016, 2017, 2018, 2019, 2020, 2021, 2022])
    res["halftime_eval"] = ht
    version = f"inplay-{db.now()[:19]}"
    con.execute("UPDATE live_model SET status='retired' WHERE status='champion'")
    con.execute("INSERT INTO live_model(version, created_at, params, metrics, status) VALUES(?,?,?,?,?)",
                (version, db.now(), db.dumps({"model": res["final_model"], "half_end": res["half_end"], "minutes_per_match": res["minutes_per_match"],
                                                 "prematch_model": version_name(champ)}), db.dumps({k: v for k, v in res.items() if k != "final_model"}), "champion"))
    return {"version": version, "selected": res["selected"], "paired": res["paired"], "halftime_paired": ht["paired"]}


def version_name(champ):
    return f"{champ}@v1"
