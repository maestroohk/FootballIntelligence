# Research notes

Concise notes on the methods and data sources considered, what was implemented, and the limits of each. Measured results live in `REPORT.md` and in the app's Model lab.

## Forecasting methods

| Method | Key references | Status here | Notes and limitations |
|---|---|---|---|
| Independent Poisson attack/defence ratings | Maher (1982) | Implemented (`ratings.StrengthModel`, `dixon_coles=False`) | Log-linear rates with ridge penalty instead of sum-to-zero constraints; teams across tiers of a country share one scale so promotions are handled. Ignores score dependence. |
| Dixon–Coles low-score correction + time decay | Dixon & Coles (1997) | Implemented, baseline champion | τ correction on 0-0/1-0/0-1/1-1, exponential time weights (ξ = 0.0019/day ≈ 1-year half-life chosen on validation). Gain over plain Poisson is small. |
| Bivariate Poisson | Karlis & Ntzoufras (2003) | Not implemented | Shared-component covariance is usually near zero for league football; DC already covers the low-score dependence that matters. Candidate for future work. |
| Elo ratings | Hvattum & Arntzen (2010); World Football Elo | Implemented (K = 15/20/30, goal-margin multiplier, season regression) | Ordered logit maps rating gap to 1X2; Poisson link maps it to goal rates so Elo also yields a scoreline matrix. |
| Glicko / pi-ratings | Glickman (1999); Constantinou & Fenton (2013) | Not implemented | Pi-ratings performed well in the 2017 Soccer Prediction Challenge (Hubáček et al. 2019); future ensemble component. |
| Bayesian hierarchical models | Baio & Blangiardo (2010) | Not implemented | Equivalent point forecasts to penalised MLE here; full posterior would give better parameter uncertainty for promoted teams. |
| Dynamic / state-space strengths | Rue & Salvesen (2000); Koopman & Lit (2015) | Approximated by weekly refits with time decay | A true state-space model would update strengths match by match; weekly refit is the practical compromise. |
| Expected goals (xG), xGOT | StatsBomb, Opta | xG used where licensed (StatsBomb 2015/16); a shot-count xG proxy for history | No open provider xG covers 2005–2026; the proxy is `0.208·SOT + 0.018·off-target` fitted on 2005–09 only. xGOT is not in the open data. |
| xT / EPV / VAEP | Singh (2018); Fernández et al. (2019, 2021); Decroos et al. (2019) | Not implemented | Need full event data for every match; open data covers one season. Progressive-pass/carry counts are used as simple possession-value proxies on player pages. |
| Gradient boosting | Hubáček, Šourek & Železný (2019) | Implemented (LightGBM Poisson for home/away goals) | Features: Elo, exponentially weighted goals/shots/SOT/xG-proxy for and against, rest days, matches in 28 days, promotion flag, tier. |
| Multinomial logistic regression | — | Implemented (1X2 baseline) | No score distribution; research comparator only. |
| Probabilistic ensemble | — | Implemented, champion | Weighted geometric mean of component goal rates (weights, scale and DC ρ fitted per season on the previous three seasons' out-of-sample predictions). Combining at the rate level keeps every market coherent with one joint score distribution. |
| Calibration | Platt; temperature scaling | Tested (spread parameter on rate difference) | Ensemble is slightly under-confident (temperature exponent 1.07); correction gave a negligible gain and was not promoted. |
| Markov / hazard in-play model | Dixon & Robinson (1998); Volf (2009); Robberechts et al. (2021) | Implemented | Piecewise-constant goal hazards with pre-match rate as offset, score-state and red-card multipliers, exact forward recursion over minutes. |
| Survival / goal hazard | Volf (2009) | Implemented as above | Time-of-match profile in eight bins (incl. stoppage time). |
| Monte Carlo simulation | — | Benchmarked | Exact enumeration of the 11×11 score grid is 0.29 ms and error-free; 10k simulations still carry ~0.6 pp error. Simulation only pays off when mixing uncertain scenarios. |
| Temporal validation / drift | — | Implemented | Walk-forward weekly refits, untouched final period, rolling drift monitor, champion history with rollback. |
| Player contribution / lineups | — | Tested on StatsBomb 2015/16 | Starting-XI xG+xA index improved held-out log loss by 0.0009 (CI −0.0002..+0.0020): not adopted. |
| Injury and availability modelling | — | Not implemented | No legitimate open injury feed; the app shows availability as unknown rather than guessing. |

## Data sources

| Source | Licence / terms | Coverage used | Reliability | Live? | Decision |
|---|---|---|---|---|---|
| football-data.co.uk | "My data is free"; no redistribution terms published | 18 divisions in 11 countries, 2005/06–2026/27: results, HT scores, shots, SOT, corners, fouls, cards, referee, opening and closing odds | High; 0 score mismatches against StatsBomb (1,551 matches) and OpenFootball (470 matches) | No | Primary historical source |
| StatsBomb Open Data | Free for research, attribution to StatsBomb required | 2015/16 Premier League, La Liga, Serie A, Ligue 1 (full), Bundesliga (34 matches); event data with shot xG, lineups, substitutions, cards | High | No | In-play model training, replay, player profiles |
| OpenFootball (football.json) | Public domain | 2026/27 fixture lists for the six primary leagues plus the Eredivisie and Primeira Liga | Fixtures reliable; results lag | No | Future fixtures |
| football-data.org | Free tier with API key (10 req/min) | Live scores, fixtures | Good | Scores only on free tier | Adapter implemented; inactive without `FOOTBALL_DATA_API_KEY` |
| ClubElo | Free API | Club Elo history | — | No | Returned HTTP 502 during this session; own Elo implemented instead |
| Understat | No API; terms do not grant scraping rights | xG 2014+ | — | No | Unreachable from this machine; not used |
| FBref / Opta | Scraping restricted; Opta data removed in 2025 | — | — | No | Not used |
| Transfermarkt | Terms forbid scraping | Injuries, market values | — | No | Not used |
| Commercial event feeds (Opta, StatsPerform, Sportradar) | Paid | Live events, lineups, injuries | High | Yes | Architecture ready (`live_events` with provider ids, de-duplication, corrections) but not integrated |

## Modelling decisions worth knowing

- Every rate model is fitted per country on all tiers together, so promoted and relegated teams keep a continuous rating.
- Training data for each weekly block ends strictly before the block starts; this is asserted in code and covered by a test.
- Model selection uses the validation seasons 2012/13–2022/23 only. Seasons 2023/24 onward were evaluated once, after all choices were frozen.
- Forward predictions are written before kick-off, are immutable (database trigger) and are only created for matches still scheduled.
- In-play forecasts treat accumulated xG as evidence only through a learned coefficient; the evidence variant was not selected because it did not improve held-out accuracy, which also removes any risk of double counting pre-match strength and in-match xG.
