import json

from . import db
from .config import ROOT
from .jobs import handler

LABELS = {
    "frequency": "Frequency baseline", "elo": "Elo K=20", "elo_k15": "Elo K=15", "elo_k30": "Elo K=30", "poisson": "Poisson (Maher)",
    "dixon_coles": "Dixon-Coles", "dc_xi0010": "Dixon-Coles slow decay", "dc_xi0030": "Dixon-Coles fast decay", "dc_ridge3": "Dixon-Coles ridge 3",
    "dc_xg25": "DC + shot-xG 25%", "dc_xg50": "DC + shot-xG 50%", "dc_xg75": "DC + shot-xG 75%", "gbm_features": "LightGBM features",
    "gbm_stacked": "LightGBM stacked on DC", "logistic": "Multinomial logistic", "ensemble": "Learned ensemble (champion)", "bookmaker_close": "Bookmaker closing odds (benchmark)",
}


WINDOWS = {
    "frequency": "previous 5 seasons of the same league",
    "elo": "updated after every match; outcome/goal mapping refit each season on the 4 prior seasons",
    "poisson": "time-weighted matches from the previous ~3 years, refit weekly",
    "dc": "time-weighted matches from the previous ~3 years, refit weekly",
    "gbm": "all matches from 2007/08 to the previous season, refit each season",
    "logistic": "all matches from 2007/08 to the previous season, refit each season",
    "ensemble": "component models refit as listed; blend weights fitted on the previous 3 seasons",
    "bookmaker_close": "market closing odds (benchmark, not a model)",
}


def _f(v, d=4):
    return "—" if v is None else f"{v:.{d}f}"


def _table(con, split):
    rows = con.execute("SELECT * FROM backtest_runs WHERE split=? AND run_id IN (SELECT MAX(run_id) FROM backtest_runs WHERE split=? GROUP BY model_version)", (split, split)).fetchall()
    rows = sorted(rows, key=lambda r: json.loads(r["metrics"])["log_loss"])
    out = ["| Model | Training window | Test period | Matches | Log loss | Brier | RPS | ECE | Accuracy | Goal MAE | Total-goals MAE | Score NLL | O2.5 log loss | BTTS log loss |",
           "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        m = json.loads(r["metrics"])
        key = r["model_version"]
        family = "elo" if key.startswith("elo") else "gbm" if key.startswith("gbm") else "dc" if key.startswith(("dc_", "dixon")) else key
        window = WINDOWS.get(family, "")
        out.append(f"| {LABELS.get(r['model_version'], r['model_version'])} | {window} | {r['test_period']} | {r['n_matches']:,} | {_f(m.get('log_loss'))} | {_f(m.get('brier'))} | {_f(m.get('rps'))} | "
                   f"{_f(m.get('ece'))} | {_f(m.get('accuracy'), 3)} | {_f(m.get('goal_mae'), 3)} | {_f(m.get('total_goals_mae'), 3)} | {_f(m.get('score_nll'))} | {_f(m.get('over25_log_loss'))} | {_f(m.get('btts_log_loss'))} |")
    return "\n".join(out)


def _comparisons(values):
    out = ["| Comparison (A vs B) | Metric | n | Mean A − B | 95% CI | Verdict |", "|---|---|---:|---:|---|---|"]
    for name, v in values.items():
        pair, metric = name.split(" [")
        a, b = pair.split(" vs ")
        verdict = "B better" if v["ci_low"] > 0 else "A better" if v["ci_high"] < 0 else "inconclusive"
        out.append(f"| {LABELS.get(a, a)} vs {LABELS.get(b, b)} | {metric.rstrip(']')} | {v['n']:,} | {v['mean_diff']:+.4f} | {v['ci_low']:+.4f} .. {v['ci_high']:+.4f} | {verdict} |")
    return "\n".join(out)


def _by_league(con, model, split):
    r = con.execute("SELECT by_league FROM backtest_runs WHERE model_version=? AND split=? ORDER BY run_id DESC LIMIT 1", (model, split)).fetchone()
    if not r:
        return ""
    out = ["| League | Matches | Log loss | RPS | ECE | Accuracy | Goal MAE |", "|---|---:|---:|---:|---:|---:|---:|"]
    for code, m in json.loads(r["by_league"]).items():
        out.append(f"| {code} | {m['n']:,} | {_f(m['log_loss'])} | {_f(m['rps'])} | {_f(m['ece'])} | {_f(m['accuracy'], 3)} | {_f(m.get('goal_mae'), 3)} |")
    return "\n".join(out)


def _by_confidence(con, model, split):
    r = con.execute("SELECT reliability FROM backtest_runs WHERE model_version=? AND split=? ORDER BY run_id DESC LIMIT 1", (model, split)).fetchone()
    if not r:
        return ""
    rel = json.loads(r["reliability"])
    out = ["| Top-outcome probability | Matches | Mean predicted | Hit rate | Log loss |", "|---|---:|---:|---:|---:|"]
    for band, m in rel["by_confidence"].items():
        out.append(f"| {band} | {m['n']:,} | {m['mean_top_p']:.3f} | {m['top_hit_rate']:.3f} | {m['log_loss']:.4f} |")
    out.append("")
    out.append("| Outcome | Bin | Matches | Mean predicted | Observed |")
    out.append("|---|---|---:|---:|---:|")
    for outcome, bins in rel["outcomes"].items():
        for b in bins:
            if b["n"] >= 100:
                out.append(f"| {outcome} | {b['lo']:.1f}–{b['hi']:.1f} | {b['n']:,} | {b['mean_p']:.3f} | {b['freq']:.3f} |")
    return "\n".join(out)


def _inplay(con):
    row = con.execute("SELECT version, params, metrics FROM live_model WHERE status='champion' ORDER BY created_at DESC LIMIT 1").fetchone()
    if not row:
        return "No in-play model trained."
    m = json.loads(row["metrics"])
    p = json.loads(row["params"])
    lines = [f"Model `{row['version']}`; training/test split at {m['split_date']} ({m['n_train_matches']} / {m['n_test_matches']} StatsBomb matches); "
             f"average half ends {m['half_end'][0]:.1f}' and {m['half_end'][1]:.1f}'; second-half scale by tier {p.get('second_half_scale_by_tier')}.", "",
             "Checkpoint evaluation on held-out matches (final-result log loss / 'another goal' log loss / 'another goal' ECE):", "",
             "| Model | " + " | ".join(sorted(next(iter(m["checkpoints"].values())).keys())) + " |",
             "|---|" + "---:|" * len(next(iter(m["checkpoints"].values())))]
    for model, cps in m["checkpoints"].items():
        lines.append(f"| {model} | " + " | ".join(f"{cps[k]['log_loss_1x2']:.3f} / {cps[k]['another_goal_log_loss']:.3f} / {cps[k]['another_goal_ece']:.3f}" for k in sorted(cps)) + " |")
    lines += ["", "Half-time forecasts on football-data.co.uk HT scores (2016/17–2022/23, all 11 divisions):", "",
              "| Model | Matches | 1X2 log loss | P(2nd-half goal) predicted | observed | ECE |", "|---|---:|---:|---:|---:|---:|"]
    for model, v in m["halftime_eval"]["summary"].items():
        h = v["HT"]
        lines.append(f"| {model} | {h['n']:,} | {h['log_loss_1x2']:.4f} | {h['mean_p_more']:.3f} | {h['freq_more']:.3f} | {h['another_goal_ece']:.3f} |")
    lines += ["", _comparisons({f"{k.replace('_vs_', ' vs ')} [1X2 log loss, held-out SB]": v for k, v in m["paired"].items()}),
              _comparisons({f"{k.replace('_vs_', ' vs ')} [1X2 log loss, half-time]": v for k, v in m["halftime_eval"]["paired"].items()}).split("\n", 2)[2], "",
              "Learned multipliers (hazard model):", "", "| Term | Multiplier | s.e. of log |", "|---|---:|---:|"]
    for c in m["coefficients"]["hazard"]:
        lines.append(f"| {c['name']} | ×{c['multiplier']:.3f} | {c['se']:.3f} |")
    return "\n".join(lines)


def build(con):
    summary = db.get_kv(con, "research_summary")
    final = db.get_kv(con, "final_test_run")
    parts = ["# Measured results", "", "Generated from the database by the `build_report` job; every number below is computed from stored out-of-sample predictions.", "",
             f"Champion: **{summary['selection']['champion']}**. Selection log: " + "; ".join(summary["selection"]["reasons"]) + ".", "",
             "## Validation (2012/13–2022/23, six primary leagues, walk-forward)", "", _table(con, "validation"), "",
             "## Final test (2023/24–2026/27 to date, untouched until all choices were frozen)", "",
             f"First evaluated at {final['at'] if final else 'n/a'}." if final else "Not yet evaluated.", "", _table(con, "final_test"), "",
             "## Paired comparisons, validation", "", _comparisons(db.get_kv(con, "validation_comparisons", {})), "",
             "## Paired comparisons, final test", "", _comparisons(db.get_kv(con, "final_test_comparisons", {})), "",
             "## Champion by league, final test", "", _by_league(con, summary["selection"]["champion"], "final_test"), "",
             "## Champion calibration, final test", "", _by_confidence(con, summary["selection"]["champion"], "final_test"), "",
             "## Ensemble weights by season", "", "| Season | " + " | ".join(db.get_kv(con, "ensemble_weights")["components"]) + " | scale | ρ |",
             "|---|" + "---:|" * (len(db.get_kv(con, "ensemble_weights")["components"]) + 2)]
    w = db.get_kv(con, "ensemble_weights")
    for season, f in sorted(w["by_season"].items()):
        parts.append(f"| {season} | " + " | ".join(f"{f['weights'][c]:.2f}" for c in w["components"]) + f" | {f['scale']:+.3f} | {f['rho']:+.3f} |")
    parts += ["", "## In-play model", "", _inplay(con), "", "## Experiment log", ""]
    for e in con.execute("SELECT * FROM experiments ORDER BY experiment_id"):
        parts += [f"### {e['name']}", "", f"- Hypothesis: {e['hypothesis']}", f"- Result: {e['result']}", f"- Decision: {e['decision']}", ""]
    text = "\n".join(parts)
    (ROOT / "docs" / "RESULTS.md").write_text(text, encoding="utf-8")
    return {"chars": len(text)}


@handler("build_report")
def build_report(ctx):
    ctx.stage("writing docs/RESULTS.md")
    return build(ctx.con)
