# Completion report

Session: 2 November 2026, started 21:14 (UTC+3), target end 05:14. All figures below were measured in this session; the full generated tables are in `RESULTS.md` (rebuild with the `build_report` job) and in the app's Model lab.

## 1. What was built

| Deliverable | Status | Where |
|---|---|---|
| Local application (ASP.NET Core Blazor Server + Python engine) | Working | `src/FootballIntelligence.Web`, `engine/fie`; start instructions in `README.md` |
| Pre-match engine (1X2, expected goals, full scoreline matrix, BTTS, clean sheets, O/U 0.5–4.5, goal in each half, expected goal difference, uncertainty indicators) | Working; all markets derived from one joint distribution, checked by tests | Match analysis page, `scoreline.py` |
| Live engine | Working in **historical replay** (clearly labelled simulation) and as a football-data.org score-based adapter that is **untested against the real API** because no key was available (covered by mocked tests) | Live tracker, `live.py`, `inplay.py` |
| Historical datasets and ingestion | 133,769 finished matches in 18 divisions across 11 countries (2005/06–2026/27), 1,551 StatsBomb event matches, about 2,100 future fixtures (football-data.co.uk and OpenFootball); incremental and resumable | `ingest_fdco.py`, `ingest_statsbomb.py`, `ingest_openfootball.py` |
| Model evaluation dashboard | Working | Model lab, Performance |
| Backtesting, calibration and reliability analysis | Walk-forward validation 2012/13–2022/23 (26,057 matches) and an untouched final test 2023/24–2026/27 (7,515 matches) | `backtest.py`, `RESULTS.md` |
| Background jobs | Durable SQLite queue, bounded concurrency (2 jobs, 4 threads, below-normal priority), heartbeats, progress, ETA, cancel/retry, logs, recovery after restart, in-app notifications | Jobs page, `jobs.py` |
| Continuous learning | 6-hourly cycle (results, fixtures, forward predictions, forward evaluation, drift check), weekly challenger research, champion history and rollback | `scheduler.py`, `tasks.py` |

## 2. Model comparison (six primary leagues, identical match sets)

| Model | Training window | Validation log loss (n = 26,057) | Final-test log loss (n = 7,515) | Final-test Brier | Final-test ECE | Final-test goal MAE | Final-test scoreline NLL |
|---|---|---:|---:|---:|---:|---:|---:|
| Frequency baseline | previous 5 seasons, same league | 1.0708 | 1.0743 | 0.6501 | 0.0030 | 0.964 | 3.0137 |
| Elo (K = 15) | updated every match; mapping refit per season | 0.9990 | 0.9990 | 0.5968 | 0.0088 | 0.905 | 2.9238 |
| Poisson (Maher) | ~3 years, time-weighted, weekly refit | 0.9988 | 0.9998 | 0.5972 | 0.0159 | 0.901 | 2.9101 |
| Dixon–Coles | ~3 years, time-weighted, weekly refit | 0.9984 | 0.9990 | 0.5969 | 0.0123 | 0.901 | 2.9100 |
| DC + shot-based xG proxy (25%) | as Dixon–Coles | 0.9989 | 0.9993 | 0.5970 | 0.0182 | 0.892 | 2.9146 |
| LightGBM (features) | 2007/08 to previous season, refit per season | 0.9953 | 0.9951 | 0.5943 | 0.0134 | 0.898 | 2.9026 |
| LightGBM stacked on DC | as above | 0.9949 | 0.9940 | 0.5935 | 0.0139 | 0.898 | 2.8998 |
| Multinomial logistic (1X2 only) | as above | 0.9954 | 0.9942 | 0.5938 | 0.0084 | — | — |
| **Learned ensemble (champion, `ensemble@v2`)** | components as above; weights on previous 3 seasons | **0.9947** | **0.9949** | **0.5940** | **0.0143** | **0.897** | **2.8996** |
| Bookmaker closing odds (benchmark) | market | 0.9811 | 0.9773 | 0.5825 | 0.0101 | — | — |

Validation period 2012-08-10 to 2023-06-04; final test 2023-08-04 to 2026-11-01. The final test was first run once after every modelling choice was frozen; it was re-run once more only to add BTTS and over/under metrics after a metric-definition fix (the predictions did not change).

Key paired comparisons (block bootstrap over league-seasons, 95% CI, positive = second model better):

- Ensemble vs Dixon–Coles: validation +0.0036 log loss [+0.0028, +0.0045]; final test +0.0041 [+0.0022, +0.0060]. The improvement replicated out of sample.
- Ensemble vs Elo K=15: final test +0.0041 [+0.0029, +0.0052].
- Dixon–Coles vs Poisson: validation +0.0004 [−0.0000, +0.0009]; final test +0.0008 [+0.0002, +0.0015]. Small.
- Ensemble vs bookmaker closing odds: the bookmaker is better by 0.0136 (validation) and 0.0176 (final test). The model does not beat the market.
- Over 2.5 goals log loss: ensemble 0.6774 vs bookmaker 0.6696 on the final test.

Generalisation: seven divisions added after the model freeze (Eredivisie, Primeira Liga, Pro League, Scottish Premiership and Championship, Süper Lig, Super League Greece; 26,176 matches from 2012/13) were never used for selection, ensemble weighting or LightGBM training. The ranking held: ensemble 0.9751, LightGBM 0.9772, Elo 0.9795, Dixon–Coles 0.9813, frequency 1.0698, bookmaker 0.9583. Ensemble vs Dixon–Coles +0.0062 [+0.0047, +0.0078]; vs LightGBM +0.0022 [+0.0012, +0.0031].

Calibration: the champion's draw rate is unbiased on validation (25.5% predicted vs 25.5% observed). It is slightly under-confident (temperature exponent 1.07 in both periods); final-test class-wise ECE rose to 0.014 from 0.008. A spread correction reduced validation log loss by only 0.0002 and was not promoted.

## 3. Live / in-play model

- Structure: each team's per-minute scoring rate = pre-match expected goals ÷ expected playing minutes × learned multipliers for eight match-time bins (including stoppage time), score state (leading/trailing by 1 or 2+, and after 70′) and red cards. The final score distribution is computed exactly by stepping a Markov chain over the remaining minutes; P(goal in next 5/10/15′), next scorer, leader-holds and comeback probabilities come from the same recursion.
- Learned effects (StatsBomb 2015/16): extra man ×1.76, man down ×0.59; trailing by two ×1.12; first 15 minutes ×0.76; second-half stoppage ×1.11.
- Held-out replay (636 matches, 3,816 checkpoints): hazard beats a constant-rate model by 0.0061 log loss [+0.0002, +0.0123] and linear extrapolation by 0.327 [+0.272, +0.382].
- Half-time evaluation on 30,950 football-data.co.uk matches: hazard beats constant-rate by 0.0024 [+0.0020, +0.0029]; linear extrapolation is worse by 0.373.
- P(goal in the next 10 minutes): 24.7% predicted vs 25.0% observed, ECE 0.004 over 10,176 checkpoints. Next-goal team log loss 0.936.
- Leader goes on to win: 81.6% predicted vs 80.9% observed over 5,303 checkpoints with a lead (ECE 0.016).
- In-match xG evidence term: no measurable gain (−0.0007 [−0.0029, +0.0015]) → not used, so accumulated xG never double-counts pre-match strength.
- Second-half recalibration: ×1.04 for top divisions adopted (ECE 0.028 → 0.015 on 8,000 held-out matches); ×1.08 for lower divisions not adopted (CI spanned zero).
- Replay robustness: duplicates, 2-minute-late deliveries and corrections are handled (one replay: 242 deliveries, 10 duplicates ignored, 7 late events re-ordered).

## 4. Features and experiments: what helped and what did not

Helped (adopted): LightGBM on engineered features (Elo gap carries 62% of gain; recent shots and goals, promotion flags, rest days the rest); learned rate-level ensemble; Dixon–Coles correction (small); one-year half-life time decay; Elo K=15 over K=20/30; in-play score-state and red-card multipliers; top-flight second-half recalibration; exact enumeration over Monte Carlo.

Dynamic ratings: an online (match-by-match) Poisson rating with a small learning rate beat weekly Dixon–Coles on its own (validation log loss 0.9971 vs 0.9984, best of seven settings, so slightly optimistic); pi-ratings did not beat Elo (0.9997 vs 0.9990). Feeding both to LightGBM improved LightGBM (+0.0006 log loss [+0.0001, +0.0012], +0.0021 scoreline NLL [+0.0013, +0.0029]), but an ensemble built on it did not beat the champion (+0.0001 [−0.0002, +0.0004]), so the champion was kept.

Did not help (rejected): shot-based xG proxy as a training response (25% neutral, 50–75% worse and miscalibrated); stronger ridge; slower or faster decay; stacked-GBM in the ensemble (no gain); spread calibration (negligible); in-match xG evidence; starting-XI quality adjustment (+0.0009 log loss, CI −0.0001..+0.0019, n = 730 — promising but not proven); model information beyond the opening market (blend weight on the model ≈ 0); GPU training (no speed-up at this data size: LightGBM CPU 0.73 s vs XGBoost CUDA 5.4 s).

Every experiment with its hypothesis, numbers and decision is in the experiment log (Model lab → Experiment log, and `RESULTS.md`).

## 5. Errors found and corrected during the session

1. Paired bootstrap crashed on array truthiness (`if block:`) — fixed.
2. Ensemble initially evaluated on 2012+ while other models used 2010+, making the comparison table unfair — validation window aligned to 2012/13–2022/23 for all models.
3. Python wrote `NaN` into JSON, which .NET rejects — all JSON now goes through a NaN-safe serializer; stored rows repaired.
4. StatsBomb ingestion: duplicate event keys for shot+goal and a column-count mismatch — fixed before bulk ingestion.
5. StatsBomb events attributed to side "unknown" when event team names differed from match metadata (18 goals in 20 matches) — now matched by team id; all 1,551 matches reconcile goal-by-goal with final scores; in-play model retrained.
6. Shrinkage bug in the player study (prior weight 90× too small) — fixed before conclusions were drawn.
7. Red-card minutes not recorded for player minutes — fixed.
8. Past-dated unplayed fixtures counted as upcoming and one received a forward prediction — fixtures are marked `result_missing` after 3 days and forward predictions are limited to a 10-day horizon.
9. Match pages and replays picked the bookmaker benchmark row (no expected goals) instead of the champion — selection fixed.
10. Static assets (CSS and the Blazor script) were served empty/500 because static web assets were not enabled — fixed; stylesheet cache-busting added.
11. Second-half replay timeline used the average first-half length instead of the actual one; full-time snapshot still showed a residual goal probability — fixed.
12. Reliability parsing failed on missing over/under odds — made null-safe; bookmaker O/U metrics now computed on matches with odds.
13. The match key (competition, season, home, away) silently merged repeated meetings in leagues with splits or play-offs (Scotland, Belgium, Greece): 4,377 matches lost and 4,079 false "result corrected" alerts when the new leagues were first ingested. Fixed with date-aware matching (finished matches match on the exact date; date tolerance only for unplayed fixtures), a non-unique index, re-ingestion and a regression test. The original 11 divisions were unaffected (95,046 finished matches before and after).
14. The table migration renamed `matches` and SQLite rewrote foreign keys in `odds`, `predictions`, `sb_matches` and a trigger to point at the temporary name. Repaired in place (integrity and foreign-key checks clean) and the migration now uses legacy rename semantics.
15. A ±4-day matching window merged two genuine Scottish Championship meetings a few days apart — tightened to exact dates for finished matches.
16. Forward predictions were stored for same-day fixtures whose kick-off had already passed (their results were simply not yet ingested). Forward predictions now require a listed kick-off at least 3 hours in the future (local times vary by source); 14 existing rows were flagged `after_kickoff_possible` (rows stay immutable) and are excluded from forward evaluation and labelled in the UI.
17. The Players page took 6.8 s because of an unindexed per-player subquery; an index brought it to 0.19 s.
18. Version drift: the first published backtest predictions (`ensemble@v1`) were produced before the ensemble optimiser's iteration cap was raised from 4,000 to 6,000, which changed fitted weights in some seasons, while every evaluation figure since then refers to the newer fit. The current champion is now registered as `ensemble@v2` with its own predictions; v1 is retired and kept as history. Re-computing the final-test metrics from v2 reproduces the reported figures to within 1e-6.

## 6. Data-provider limitations

- No licensed live event feed. Live forecasting from a real feed is limited to score and minute via football-data.org (needs a key; the free tier gives no goal times, cards or lineups, so goal times are approximate and red cards are not seen).
- Event data, lineups and player data exist only for StatsBomb 2015/16 (four full leagues plus 34 Bundesliga matches); player profiles are historical.
- No injury, availability, weather, pitch or travel data from a legitimate free source.
- No provider xG or xGOT for 2005–2026; xG for history is a shot-count proxy.
- UEFA Champions League is not covered (no free source with results plus statistics across seasons, and cross-league team strengths were out of scope).
- Kick-off times are local times from each source.

## 7. Incomplete or partial features

- Live tracking of real matches is untested end to end (no API key); replay mode is fully working.
- Player availability is shown as unknown for current fixtures; lineup-based adjustments are not used (evidence insufficient).
- Tactical analysis is descriptive (shots, corners, fouls; possession, PPDA, pressures for StatsBomb matches) — no formation-matchup model.
- No Bayesian hierarchical, bivariate Poisson, pi-rating or true state-space model (see future work).
- About 90 valid forward predictions exist so far (14 more are flagged as stored too close to kick-off); forward performance tracking only becomes meaningful after results arrive.
- Model selection used the six primary leagues; seven further divisions were evaluated as an out-of-sample generalisation test; second and third tiers of the original countries are used for strength linkage and receive predictions but are not separately validated.

## 8. Prioritised plan

1. Connect a licensed live event feed (goals with times, cards, substitutions, lineups) and re-validate the in-play model on post-2023 data with longer stoppage times.
2. Obtain confirmed lineups for current fixtures and re-test the starting-XI adjustment on a larger sample (multiple seasons of event data).
3. Add pi-ratings and a state-space (Kalman) strength model as ensemble components; test a bivariate Poisson.
4. Calibrate the ensemble's slight under-confidence with a method evaluated on future seasons only.
5. Add the Champions League and other cross-league competitions, which need a common cross-country strength scale (e.g. linking through European results or club Elo).
6. Build a market-informed product variant (blend with opening odds) separately from the pure model, since the market currently dominates.
