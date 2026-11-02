# Session log

Session start: 2026-11-02 21:14 (UTC+3). Target completion: 2026-11-03 05:14. Times below are local and come from the job table where available.

| Time | Area | What happened |
|---|---|---|
| 21:14 | Setup | Hardware: i7-13650HX (14 cores / 20 threads), 40 GB RAM, RTX 4060 Laptop 8 GB, 48 GB free disk. Compute kept to 4 threads, below-normal priority, at most 2 concurrent jobs (laptop overheats under sustained load). |
| 21:15–21:25 | Research | Verified football-data.co.uk columns and terms, StatsBomb Open Data structure and attribution terms, fixture feeds; Understat unreachable, ClubElo returned 502. |
| 21:18–21:45 | Data | Ingested 242 football-data.co.uk files (95,047 matches, 11 divisions, 2005/06–2026/27) and 1,550 StatsBomb event files (compact tables only). Identity resolution linked all StatsBomb matches with zero score mismatches. |
| 21:29–21:41 | Models | Walk-forward Elo, Poisson, Dixon–Coles variants, shot-xG proxy variants, LightGBM, logistic, learned ensemble and bookmaker benchmark. Fixed a bootstrap bug and aligned the validation window (2012/13–2022/23). Champion: ensemble. |
| 21:42–21:46 | In-play | Goal-hazard model trained on StatsBomb 2015/16, evaluated on held-out matches and on 30,950 half-time states. |
| 21:47–22:00 | App | Blazor dashboard brought up and verified in the browser; fixed NaN JSON, static asset serving, prediction selection, replay prior; first replays verified (goals, red cards, duplicates, late events). |
| 22:00–22:10 | Experiments | Market information, draw calibration, Monte Carlo vs exact, GPU benchmark, half-time recalibration. |
| 22:10 | Evaluation | Final test (2023/24–2026/27) evaluated for the first time, after freezing all choices. |
| 22:13 | Forward | 72 forward predictions with component and SHAP explanations; OpenFootball fixtures added (345/345 overlapping results agree). |
| 22:19–22:31 | Data / in-play | Found 18 StatsBomb goals attributed to an unknown side; fixed (team ids), re-processed 20 matches, retrained; per-tier second-half recalibration; short-horizon and next-goal calibration; stoppage-era check. |
| 22:32–22:36 | Evaluation | BTTS and over/under metrics added; validation and final-test tables re-computed with the new metrics (no model change). Job cancellation tested from the UI. |
| 22:39–22:43 | Players / docs | Lineup-uncertainty analysis; generated results report; completion report draft. |
| 22:43–22:50 | Experiments | Online Poisson and pi-ratings; LightGBM with dynamic-rating features; ensembles built on them (no gain over the champion). |
| 22:51–23:06 | Coverage | Seven divisions added. Found and fixed the match-key bug (repeated meetings merged), repaired a foreign-key rename side effect, re-ingested; generalisation test on 26,176 unseen-league matches. |
| 23:06–23:10 | Integrity | Kick-off guard for forward predictions; 14 late rows flagged and excluded; mobile layout and Players index fixes. |
| 23:09–23:13 | In-play | Leader-holds calibration; cumulative xG chart; full UI replay re-verified. |
| 23:13–23:25 | Versioning | Found that the published ensemble predictions came from an earlier fit; traced the cause to Nelder–Mead landing on different local optima after row-order changes; switched to deterministic multi-start L-BFGS (`ensemble@v3`), re-published, re-ran the final-test audit (identical figures). |
| 23:26–23:45 | Tactics / experiments | Re-parsed StatsBomb events to keep shot play patterns; team tactical profiles page; provider-xG experiment (no gain); goal-count calibration on the performance page; unused-code cleanup; pinned requirements. |
| 23:45–00:00 | Reproducibility | Isolated from-scratch rerun of all models; champion reproduced exactly; Elo K=20/30 drift traced to order-sensitive Nelder–Mead in the outcome mapping and fixed; all 17 models now agree within 2e-8. |

The final test was computed four times in total: once after the model freeze (22:10), once to add BTTS and over/under metrics (22:35), and as audits after the reproducibility fixes (23:22 and 00:00). No model choice was made from final-test numbers; the champion and its components were fixed on validation.
