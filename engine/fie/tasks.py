import pandas as pd

from . import backtest, db, ingest_fdco, ingest_statsbomb, metrics, research
from .config import PRIMARY_COMPETITIONS, RESEARCH_PROCESSES
from .jobs import Cancelled, handler

MODEL_VERSIONS = {"ensemble": "v3"}


def model_version(name):
    return f"{name}@{MODEL_VERSIONS.get(name, 'v1')}"


NOTIFY_KINDS = {
    "ingest_history": "Historical ingestion", "ingest_statsbomb": "StatsBomb ingestion", "research_full": "Model research run",
    "train_inplay": "In-play model training", "final_test": "Final test evaluation", "daily_cycle": "Daily update cycle",
    "predict_upcoming": "Upcoming predictions", "evaluate_forward": "Prediction evaluation", "ingest_fixtures": "Fixture refresh",
    "check_assets": "Asset check", "rollback": "Champion rollback", "ingest_market_odds": "Market price ingestion",
    "strategy_research": "Strategy research", "paper_cycle": "Paper portfolio update",
}
ROUTINE_KINDS = {"daily_cycle", "predict_upcoming", "evaluate_forward", "ingest_fixtures", "check_assets", "paper_cycle"}

PRIMARY = PRIMARY_COMPETITIONS


def _check(ctx):
    if ctx.cancelled():
        raise Cancelled()


@handler("ingest_history")
def ingest_history(ctx, seasons=None, codes=None):
    ctx.stage("downloading football-data.co.uk CSVs")
    return ingest_fdco.ingest_history(ctx.con, seasons=seasons, codes=codes, progress=lambda d, t, s: (ctx.progress(d, t, s), _check(ctx)))


@handler("ingest_market_odds")
def ingest_market_odds(ctx):
    from . import odds
    ctx.stage("reading cached football-data.co.uk files")
    return odds.ingest_history(ctx.con, progress=lambda d, t, s: ctx.progress(d, t, s))


@handler("ingest_fixtures")
def ingest_fixtures(ctx):
    from . import ingest_openfootball
    ctx.stage("downloading upcoming fixtures")
    try:
        out = {"football-data.co.uk": ingest_fdco.ingest_fixtures(ctx.con)}
    except Exception as e:
        out = {"football-data.co.uk": {"error": str(e)[:200]}}
    ctx.stage("downloading OpenFootball season fixtures")
    out["openfootball"] = ingest_openfootball.ingest(ctx.con)
    return out


@handler("check_assets")
def check_assets(ctx):
    from . import bundle
    return bundle.assets(ctx.con)


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
        with ProcessPoolExecutor(max_workers=RESEARCH_PROCESSES) as pool:
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
    history = db.get_kv(con, "champion_history", [])
    selection = league_guard(con, selection, history, oos)
    if not history or history[-1]["champion"] != selection["champion"]:
        history.append({"at": db.now(), "champion": selection["champion"], "reason": selection["reasons"][-1]})
        db.set_kv(con, "champion_history", history)
    db.set_kv(con, "xg_proxy", coef)
    db.set_kv(con, "ensemble_weights", {"components": comp_names, "by_season": weights})
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


def league_guard(con, selection, history, oos):
    previous = history[-1]["champion"] if history else None
    candidate = selection["champion"]
    log = db.get_kv(con, "promotion_log", [])
    entry = {"at": db.now(), "candidate": candidate, "previous": previous, "leagues": {}}
    if previous is None or previous == candidate or previous not in oos:
        entry["decision"] = "initial" if previous is None else "kept" if previous == candidate else "promoted"
        if previous is not None and previous != candidate:
            entry["reason"] = f"{previous} was not evaluated in this run, so no league-by-league comparison was possible"
        log.append(entry)
        db.set_kv(con, "promotion_log", log[-50:])
        return selection
    worse = []
    for league in PRIMARY:
        cmp = backtest.compare(oos[candidate], oos[previous], "validation", [league], "log_loss")
        entry["leagues"][league] = cmp
        if cmp["ci_low"] > 0:
            worse.append(league)
    if worse:
        entry["decision"] = "rejected"
        entry["reason"] = f"{candidate} is significantly worse than {previous} in {', '.join(worse)} (league log loss, 95% CI above zero)"
        selection = {"champion": previous, "reasons": selection["reasons"] + [entry["reason"] + "; champion kept"]}
        db.notify(con, "warning", "Challenger rejected", entry["reason"])
    else:
        entry["decision"] = "promoted"
        entry["reason"] = f"{candidate} beats {previous} overall and is not significantly worse in any primary league"
        db.notify(con, "success", "New champion promoted", entry["reason"])
    log.append(entry)
    db.set_kv(con, "promotion_log", log[-50:])
    return selection


def register_versions(con, results, selection, best_dc, best_xg, best_elo):
    for name, res in results.items():
        if res is None:
            continue
        family = "benchmark" if name == "bookmaker_close" else name.split("_")[0]
        status = "champion" if name == selection["champion"] else "benchmark" if name == "bookmaker_close" else "challenger"
        version = model_version(name)
        params = STRENGTH_CONFIGS.get(name) or ELO_CONFIGS.get(name) or {}
        con.execute("INSERT INTO model_versions(version, family, params, trained_through, created_at, status, notes) VALUES(?,?,?,?,?,?,?) "
                    "ON CONFLICT(version) DO UPDATE SET status=excluded.status, params=excluded.params",
                    (version, family, db.dumps(params), "walk-forward", db.now(), status, None))


def _champion_version(con):
    summary = db.get_kv(con, "research_summary")
    if not summary:
        raise RuntimeError("no champion model on this machine: import a data bundle or run research_full first")
    return summary["selection"]["champion"], model_version(summary["selection"]["champion"])


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
    cov = forward.coverage_report(con, oos[champ] if champ in oos else research.load_cached(f"oos_{champ}"), version)
    if n_fwd:
        db.notify(con, "success", f"{n_fwd} new forward predictions stored", version, seen=True)
    return {"forward": n_fwd, "backtest_new": n_back, "fixtures_in_window": cov["fixtures_in_window"], "predicted": cov["predicted"], "missing": len(cov["missing"])}


CURRENT_SEASON_PUBLISH = 2026


@handler("evaluate_forward")
def evaluate_forward(ctx):
    from . import forward
    ctx.stage("comparing stored forward predictions with results")
    perf = forward.evaluate_forward(ctx.con)
    ctx.stage("looking for systematic weaknesses")
    weak = forward.weakness_report(ctx.con)
    flagged = [f["key"] for f in weak["findings"] if f["flagged"]]
    if not db.get_kv(ctx.con, "research_summary"):
        ctx.log("no champion model on this machine; drift check skipped", "warning")
        return {"forward": perf, "weaknesses": flagged, "drift": "skipped: no champion"}
    ctx.stage("drift detection")
    drift = forward.drift_check(ctx.con)
    return {"forward": perf, "weaknesses": flagged, "drift_alarm": drift["alarm"], "recent_log_loss": drift["recent_log_loss"]}


@handler("daily_cycle")
def daily_cycle(ctx, trigger="manual"):
    from .config import LIGHT
    if LIGHT:
        ctx.log("light profile: football-data.co.uk results download skipped", "info")
        res = {}
    else:
        ctx.stage("ingesting current-season results")
        res = ingest_fdco.ingest_history(ctx.con, seasons=[CURRENT_SEASON_PUBLISH], progress=lambda d, t, s: ctx.progress(d, t, s))
    ctx.stage("ingesting fixtures")
    try:
        fx = ingest_fdco.ingest_fixtures(ctx.con)
    except Exception as exc:
        ctx.log(f"fixtures failed: {exc}", "warning")
        fx = None
    ctx.stage("ingesting OpenFootball season fixtures")
    try:
        from . import ingest_openfootball
        of = ingest_openfootball.ingest(ctx.con)
    except Exception as exc:
        ctx.log(f"openfootball failed: {exc}", "warning")
        of = None
    with db.transaction(ctx.con):
        from .identity import repair_new_teams
        res["teams_merged"] = repair_new_teams(ctx.con)
        res["duplicates_reconciled"] = ingest_fdco.reconcile_duplicates(ctx.con)
    ctx.stage("recording market prices")
    try:
        from . import odds
        from .config import RAW
        latest = max((RAW / "fdco" / "fixtures").glob("fixtures_*.csv"), default=None)
        snap = odds.ingest_fixture_snapshot(ctx.con, latest, CURRENT_SEASON_PUBLISH) if latest is not None and fx is not None else None
        if not LIGHT:
            for f in sorted((RAW / "fdco").glob(f"{str(CURRENT_SEASON_PUBLISH)[2:]}{str(CURRENT_SEASON_PUBLISH + 1)[2:]}/*.csv")):
                if f.stem in odds.COMPETITIONS:
                    odds.ingest_file(ctx.con, f, f.stem, CURRENT_SEASON_PUBLISH)
    except Exception as exc:
        ctx.log(f"market prices failed: {exc}", "warning")
        snap = None
    out = {"results": {k: v for k, v in res.items() if k != "failed"}, "fixtures": fx, "openfootball": of, "prices": snap}
    out["evaluation"] = evaluate_forward(ctx)
    if db.get_kv(ctx.con, "research_summary"):
        out["prediction"] = predict_upcoming(ctx)
    else:
        ctx.log("no champion model on this machine; forward predictions skipped", "warning")
        out["prediction"] = "skipped: no champion model"
    if not LIGHT and db.get_kv(ctx.con, "research_summary"):
        try:
            from . import marketmodel
            ctx.stage("forecasting with the market-trained challenger")
            out["challenger"] = marketmodel.challenger_forecasts(ctx.con)
        except Exception as exc:
            ctx.log(f"challenger forecasts failed: {exc}", "warning")
    try:
        from . import paper
        out["paper"] = paper.paper_cycle(ctx)
    except Exception as exc:
        ctx.log(f"paper portfolio failed: {exc}", "warning")
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
    ctx.stage("re-validating the second-half recalibration for the new model")
    recal = experiments.halftime_recalibration(con)
    return {"version": version, "selected": res["selected"], "paired": res["paired"], "halftime_paired": ht["paired"],
            "second_half_scale_by_tier": {k: v["selected"] if v["adopted"] else 1.0 for k, v in recal.items()}}


def version_name(champ):
    return model_version(champ)


from . import experiments


@handler("final_test")
def final_test(ctx, force=False):
    con = ctx.con
    previous = db.get_kv(con, "final_test_run")
    if previous and not force:
        raise RuntimeError(f"final test already evaluated at {previous['at']}; re-running it invites tuning on the test period (pass force only for audits)")
    summary = db.get_kv(con, "research_summary")
    champ = summary["selection"]["champion"]
    names = ["frequency", summary["best"]["elo"], "poisson", "dixon_coles", summary["best"]["xg"], "gbm_features", "gbm_stacked", "logistic", "ensemble", "bookmaker_close"]
    oos = {n: research.load_cached(f"oos_{n}") for n in names}
    ctx.stage("evaluating frozen models on the final test period")
    con.execute("DELETE FROM backtest_runs WHERE split='final_test'")
    results = {}
    common = None
    for n, o in oos.items():
        ids = set(o.loc[(o["status"] == "finished") & (o["season"].map(backtest.split_of) == "final_test") & (o["competition"].isin(PRIMARY)), "match_id"])
        common = ids if common is None else common & ids
    for n, o in oos.items():
        sub = o[o["match_id"].isin(common)]
        rep = research.store_backtest(con, n, sub, "final_test", PRIMARY, "walk-forward; all choices frozen on validation", "untouched until this run")
        results[n] = rep["overall"] if rep else None
    ctx.stage("paired comparisons")
    comparisons = {}
    for a, b in ((summary["best"]["elo"], champ), ("dixon_coles", champ), ("poisson", "dixon_coles"), (summary["best"]["xg"], champ), ("gbm_features", champ), (champ, "bookmaker_close")):
        for metric in ("log_loss", "rps") + (("score_nll",) if b != "bookmaker_close" else ()):
            comparisons[f"{a} vs {b} [{metric}]"] = backtest.compare(oos[a][oos[a]["match_id"].isin(common)], oos[b][oos[b]["match_id"].isin(common)], "final_test", PRIMARY, metric)
    db.set_kv(con, "final_test_comparisons", comparisons)
    db.set_kv(con, "final_test_run", {"at": db.now(), "champion": champ, "n": len(common), "results": results})
    return {"n": len(common), "log_loss": {k: round(v["log_loss"], 4) for k, v in results.items() if v}}


@handler("rollback")
def rollback(ctx):
    con = ctx.con
    history = db.get_kv(con, "champion_history", [])
    if len(history) < 2:
        raise RuntimeError("no previous champion to roll back to")
    current = history.pop()
    previous = history[-1]
    summary = db.get_kv(con, "research_summary")
    summary["selection"] = {"champion": previous["champion"], "reasons": summary["selection"]["reasons"] + [f"rolled back from {current['champion']} at {db.now()}"]}
    db.set_kv(con, "research_summary", summary)
    db.set_kv(con, "champion_history", history)
    con.execute("UPDATE model_versions SET status='challenger' WHERE status='champion'")
    con.execute("UPDATE model_versions SET status='champion' WHERE version=?", (model_version(previous["champion"]),))
    db.notify(con, "warning", "Champion rolled back", f"{current['champion']} -> {previous['champion']}")
    return {"from": current["champion"], "to": previous["champion"]}
from . import report



def _load_component(name, best_elo):
    return research.load_cached(name) if name in STRENGTH_CONFIGS or name == best_elo else research.load_cached(f"oos_{name}")


@handler("expand_coverage")
def expand_coverage(ctx):
    from . import forward
    from .config import EXPANSION_COMPETITIONS
    con = ctx.con
    summary = db.get_kv(con, "research_summary")
    weights = db.get_kv(con, "ensemble_weights")
    best_elo = summary["best"]["elo"]
    ctx.stage("loading matches")
    frame, coef = research.prepared_frame(con)
    research.save_cached("frame", frame)
    new_countries = sorted(frame.loc[frame["competition"].isin(EXPANSION_COMPETITIONS), "country"].unique())
    sub = frame[frame["country"].isin(new_countries)]
    for name in sorted(set([n for n in weights["components"] if n in STRENGTH_CONFIGS] + [summary["best"]["dc"], "poisson"])):
        ctx.stage(f"walk-forward {name} for {len(new_countries)} new countries")
        new = research.run_strength(sub, name, **STRENGTH_CONFIGS[name])
        old = research.load_cached(name)
        research.save_cached(name, pd.concat([old[~old["match_id"].isin(new["match_id"])], new], ignore_index=True))
    ctx.stage(f"{best_elo} for all countries")
    elo_out, ratings = research.run_elo(frame, name=best_elo, **ELO_CONFIGS[best_elo])
    old_elo = research.load_cached(best_elo)
    research.save_cached(best_elo, pd.concat([old_elo[~old_elo["match_id"].isin(sub["match_id"])], elo_out[elo_out["match_id"].isin(sub["match_id"])]], ignore_index=True))
    research.save_cached(best_elo + "_ratings", ratings)
    ctx.stage("features and LightGBM trained on the original divisions only")
    dc = research.load_cached(summary["best"]["dc"])
    features = research.feature_table(frame, ratings, dc)
    research.save_cached("features", features)
    gbm = research.run_gbm(features, dc, name="gbm_features", use_dc=False)
    old_gbm = research.load_cached("oos_gbm_features")
    core = old_gbm[["match_id", "lh"]].merge(gbm[["match_id", "lh"]], on="match_id", suffixes=("", "_new"))
    drift = float((core["lh"] - core["lh_new"]).abs().max())
    ctx.log(f"max change in core LightGBM rates after retraining: {drift:.2e}")
    research.save_cached("oos_gbm_features", pd.concat([old_gbm[~old_gbm["match_id"].isin(sub["match_id"])], gbm[gbm["match_id"].isin(sub["match_id"])]], ignore_index=True))
    ctx.stage("applying frozen ensemble weights to the new leagues")
    comps = weights["components"]
    oos = {n: _load_component(n, best_elo) for n in comps}
    parts = []
    for season, w in weights["by_season"].items():
        ids = sub.loc[sub["season"] == int(season), "match_id"]
        if len(ids):
            parts.append(forward.apply_ensemble(oos, comps, w, frame, ids))
    ens_new = pd.concat(parts, ignore_index=True)
    old_ens = research.load_cached("oos_ensemble")
    research.save_cached("oos_ensemble", pd.concat([old_ens[~old_ens["match_id"].isin(ens_new["match_id"])], ens_new], ignore_index=True))
    ctx.stage("bookmaker benchmark")
    research.save_cached("oos_bookmaker_close", research.bookmaker_oos(con, frame))
    ctx.stage("evaluating the new leagues")
    names = [best_elo, "poisson", summary["best"]["dc"], "gbm_features", "ensemble", "bookmaker_close"]
    evals = {"frequency": research.run_frequency(frame)}
    for n in names:
        evals[n] = _load_component(n, best_elo)
    common = None
    for n, o in evals.items():
        ids = set(o.loc[(o["status"] == "finished") & (o["competition"].isin(EXPANSION_COMPETITIONS)) & (o["season"] >= 2012), "match_id"])
        common = ids if common is None else common & ids
    con.execute("DELETE FROM backtest_runs WHERE split='expansion'")
    results = {}
    for n, o in evals.items():
        d = o[o["match_id"].isin(common)]
        overall = backtest.evaluate(d)
        by_league = {c: backtest.evaluate(g) for c, g in d.groupby("competition")}
        con.execute("INSERT INTO backtest_runs(model_version, split, created_at, train_window, test_period, n_matches, metrics, by_league, notes) VALUES(?,?,?,?,?,?,?,?,?)",
                    (n, "expansion", db.now(), "own-country ratings walk-forward; shared models trained on original divisions only",
                     f"{d['date'].min().date()} to {d['date'].max().date()}", overall["n"], db.dumps(overall), db.dumps(by_league), "leagues added after the model freeze"))
        results[n] = overall
    comparisons = {}
    for a, b in ((summary["best"]["dc"], "ensemble"), (best_elo, "ensemble"), ("gbm_features", "ensemble"), ("ensemble", "bookmaker_close")):
        for metric in ("log_loss", "rps"):
            la = backtest.row_losses(evals[a][evals[a]["match_id"].isin(common)])
            lb = backtest.row_losses(evals[b][evals[b]["match_id"].isin(common)])
            j = la.merge(lb, on=["match_id", "season", "competition"], suffixes=("_a", "_b"))
            comparisons[f"{a} vs {b} [{metric}]"] = metrics.paired_bootstrap(j[f"{metric}_a"], j[f"{metric}_b"], block=(j["competition"] + j["season"].astype(str)).to_numpy())
    db.set_kv(con, "expansion_comparisons", comparisons)
    db.set_kv(con, "expansion_results", {"results": results, "n": len(common), "core_gbm_max_change": drift})
    ctx.stage("publishing predictions")
    champ, version = _champion_version(con)
    n_back = forward.publish_backtest(con, research.load_cached(f"oos_{champ}"), version)
    n_bench = forward.publish_backtest(con, research.load_cached("oos_bookmaker_close"), "bookmaker_close@v1")
    return {"n": len(common), "log_loss": {k: round(v["log_loss"], 4) for k, v in results.items()}, "core_gbm_max_change": drift, "published": n_back, "benchmark": n_bench}


from . import crests, marketmodel, paper, shapes, strategy
