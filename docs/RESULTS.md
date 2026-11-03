# Measured results

Generated from the database by the `build_report` job; every number below is computed from stored out-of-sample predictions.

Champion: **ensemble**. Selection log: dixon_coles is the baseline champion; ensemble beats dixon_coles on score NLL (diff 0.0098, 95% CI 0.0081..0.0115); dc_xg25 not promoted over ensemble: score NLL diff -0.0109, CI -0.0127..-0.0090.

## Validation (2012/13–2022/23, six primary leagues, walk-forward)

| Model | Training window | Test period | Matches | Log loss | Brier | RPS | ECE | Accuracy | Goal MAE | Total-goals MAE | Score NLL | O2.5 log loss | BTTS log loss |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Bookmaker closing odds (benchmark) | market closing odds (benchmark, not a model) | 2012-08-10 to 2023-06-04 | 26,053 | 0.9811 | 0.5844 | 0.1989 | 0.0054 | 0.522 | — | — | — | — | — |
| Learned ensemble (champion) | component models refit as listed; blend weights fitted on the previous 3 seasons | 2012-08-10 to 2023-06-04 | 26,057 | 0.9947 | 0.5937 | 0.2031 | 0.0085 | 0.514 | 0.896 | 1.302 | 2.8832 | 0.6808 | 0.6883 |
| LightGBM stacked on DC | all matches from 2007/08 to the previous season, refit each season | 2012-08-10 to 2023-06-04 | 26,057 | 0.9949 | 0.5938 | 0.2031 | 0.0062 | 0.513 | 0.896 | 1.303 | 2.8853 | 0.6822 | 0.6890 |
| LightGBM features | all matches from 2007/08 to the previous season, refit each season | 2012-08-10 to 2023-06-04 | 26,057 | 0.9953 | 0.5941 | 0.2032 | 0.0067 | 0.512 | 0.896 | 1.304 | 2.8875 | 0.6826 | 0.6895 |
| Multinomial logistic | all matches from 2007/08 to the previous season, refit each season | 2012-08-10 to 2023-06-04 | 26,057 | 0.9954 | 0.5938 | 0.2031 | 0.0071 | 0.515 | — | — | — | — | — |
| Dixon-Coles | time-weighted matches from the previous ~3 years, refit weekly | 2012-08-10 to 2023-06-04 | 26,057 | 0.9984 | 0.5963 | 0.2042 | 0.0077 | 0.510 | 0.902 | 1.306 | 2.8929 | 0.6854 | 0.6913 |
| Elo K=15 | updated after every match; outcome/goal mapping refit each season on the 4 prior seasons | 2012-08-10 to 2023-06-04 | 26,057 | 0.9987 | 0.5963 | 0.2043 | 0.0089 | 0.512 | 0.903 | 1.324 | 2.8995 | 0.6888 | 0.6927 |
| Poisson (Maher) | time-weighted matches from the previous ~3 years, refit weekly | 2012-08-10 to 2023-06-04 | 26,057 | 0.9988 | 0.5964 | 0.2042 | 0.0117 | 0.510 | 0.902 | 1.306 | 2.8934 | 0.6854 | 0.6918 |
| DC + shot-xG 25% | time-weighted matches from the previous ~3 years, refit weekly | 2012-08-10 to 2023-06-04 | 26,057 | 0.9989 | 0.5967 | 0.2043 | 0.0173 | 0.511 | 0.894 | 1.298 | 2.8940 | 0.6872 | 0.6916 |
| Dixon-Coles slow decay | time-weighted matches from the previous ~3 years, refit weekly | 2012-08-10 to 2023-06-04 | 26,057 | 0.9991 | 0.5968 | 0.2045 | 0.0076 | 0.508 | 0.903 | 1.306 | 2.8939 | 0.6855 | 0.6910 |
| Dixon-Coles fast decay | time-weighted matches from the previous ~3 years, refit weekly | 2012-08-10 to 2023-06-04 | 26,057 | 0.9993 | 0.5969 | 0.2045 | 0.0071 | 0.511 | 0.903 | 1.307 | 2.8957 | 0.6863 | 0.6921 |
| Elo K=20 | updated after every match; outcome/goal mapping refit each season on the 4 prior seasons | 2012-08-10 to 2023-06-04 | 26,057 | 0.9998 | 0.5971 | 0.2047 | 0.0096 | 0.511 | 0.904 | 1.325 | 2.9008 | 0.6891 | 0.6927 |
| Dixon-Coles ridge 3 | time-weighted matches from the previous ~3 years, refit weekly | 2012-08-10 to 2023-06-04 | 26,057 | 1.0012 | 0.5982 | 0.2051 | 0.0101 | 0.508 | 0.903 | 1.305 | 2.8950 | 0.6846 | 0.6907 |
| DC + shot-xG 50% | time-weighted matches from the previous ~3 years, refit weekly | 2012-08-10 to 2023-06-04 | 26,057 | 1.0022 | 0.5990 | 0.2051 | 0.0296 | 0.511 | 0.891 | 1.302 | 2.9069 | 0.6950 | 0.6944 |
| Elo K=30 | updated after every match; outcome/goal mapping refit each season on the 4 prior seasons | 2012-08-10 to 2023-06-04 | 26,057 | 1.0026 | 0.5990 | 0.2055 | 0.0097 | 0.508 | 0.905 | 1.325 | 2.9043 | 0.6895 | 0.6925 |
| DC + shot-xG 75% | time-weighted matches from the previous ~3 years, refit weekly | 2012-08-10 to 2023-06-04 | 26,057 | 1.0089 | 0.6037 | 0.2069 | 0.0406 | 0.510 | 0.893 | 1.315 | 2.9339 | 0.7100 | 0.7002 |
| Frequency baseline | previous 5 seasons of the same league | 2012-08-10 to 2023-06-04 | 26,057 | 1.0708 | 0.6476 | 0.2289 | 0.0078 | 0.443 | 0.961 | 1.327 | 2.9987 | 0.6911 | 0.6922 |

## Final test (2023/24–2026/27 to date, untouched until all choices were frozen)

First evaluated at 2026-11-02T21:00:25.845885Z.

| Model | Training window | Test period | Matches | Log loss | Brier | RPS | ECE | Accuracy | Goal MAE | Total-goals MAE | Score NLL | O2.5 log loss | BTTS log loss |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Bookmaker closing odds (benchmark) | market closing odds (benchmark, not a model) | 2023-08-04 to 2026-11-01 | 7,515 | 0.9773 | 0.5825 | 0.1974 | 0.0101 | 0.531 | — | — | — | 0.6696 | — |
| LightGBM stacked on DC | all matches from 2007/08 to the previous season, refit each season | 2023-08-04 to 2026-11-01 | 7,515 | 0.9940 | 0.5935 | 0.2026 | 0.0139 | 0.511 | 0.898 | 1.299 | 2.8998 | 0.6775 | 0.6849 |
| Multinomial logistic | all matches from 2007/08 to the previous season, refit each season | 2023-08-04 to 2026-11-01 | 7,515 | 0.9942 | 0.5938 | 0.2026 | 0.0084 | 0.513 | — | — | — | — | — |
| Learned ensemble (champion) | component models refit as listed; blend weights fitted on the previous 3 seasons | 2023-08-04 to 2026-11-01 | 7,515 | 0.9949 | 0.5940 | 0.2028 | 0.0142 | 0.512 | 0.897 | 1.297 | 2.8996 | 0.6774 | 0.6848 |
| LightGBM features | all matches from 2007/08 to the previous season, refit each season | 2023-08-04 to 2026-11-01 | 7,515 | 0.9951 | 0.5943 | 0.2029 | 0.0134 | 0.513 | 0.898 | 1.301 | 2.9026 | 0.6785 | 0.6857 |
| Elo K=15 | updated after every match; outcome/goal mapping refit each season on the 4 prior seasons | 2023-08-04 to 2026-11-01 | 7,515 | 0.9988 | 0.5967 | 0.2040 | 0.0072 | 0.510 | 0.906 | 1.325 | 2.9228 | 0.6899 | 0.6916 |
| Dixon-Coles | time-weighted matches from the previous ~3 years, refit weekly | 2023-08-04 to 2026-11-01 | 7,515 | 0.9990 | 0.5969 | 0.2042 | 0.0123 | 0.509 | 0.901 | 1.298 | 2.9100 | 0.6810 | 0.6878 |
| DC + shot-xG 25% | time-weighted matches from the previous ~3 years, refit weekly | 2023-08-04 to 2026-11-01 | 7,515 | 0.9993 | 0.5970 | 0.2043 | 0.0182 | 0.513 | 0.892 | 1.295 | 2.9146 | 0.6860 | 0.6909 |
| Poisson (Maher) | time-weighted matches from the previous ~3 years, refit weekly | 2023-08-04 to 2026-11-01 | 7,515 | 0.9998 | 0.5972 | 0.2043 | 0.0159 | 0.509 | 0.901 | 1.298 | 2.9101 | 0.6810 | 0.6882 |
| Frequency baseline | previous 5 seasons of the same league | 2023-08-04 to 2026-11-01 | 7,515 | 1.0743 | 0.6501 | 0.2298 | 0.0030 | 0.434 | 0.964 | 1.321 | 3.0137 | 0.6893 | 0.6893 |

## Seven leagues added after the model freeze (generalisation test)

| Model | Training window | Test period | Matches | Log loss | Brier | RPS | ECE | Accuracy | Goal MAE | Total-goals MAE | Score NLL | O2.5 log loss | BTTS log loss |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Bookmaker closing odds (benchmark) | market closing odds (benchmark, not a model) | 2012-07-28 to 2026-11-01 | 26,176 | 0.9583 | 0.5693 | 0.1927 | 0.0080 | 0.537 | — | — | — | — | — |
| Learned ensemble (champion) | own-country ratings walk-forward; shared models trained on the original divisions only | 2012-07-28 to 2026-11-01 | 26,176 | 0.9751 | 0.5805 | 0.1978 | 0.0103 | 0.525 | 0.901 | 1.304 | 2.8908 | 0.6794 | 0.6879 |
| LightGBM features | own-country ratings walk-forward; shared models trained on the original divisions only | 2012-07-28 to 2026-11-01 | 26,176 | 0.9772 | 0.5819 | 0.1984 | 0.0110 | 0.524 | 0.901 | 1.310 | 2.8984 | 0.6824 | 0.6895 |
| Elo K=15 | own-country ratings walk-forward; shared models trained on the original divisions only | 2012-07-28 to 2026-11-01 | 26,176 | 0.9795 | 0.5833 | 0.1991 | 0.0096 | 0.523 | 0.910 | 1.327 | 2.9084 | 0.6867 | 0.6924 |
| Dixon-Coles | own-country ratings walk-forward; shared models trained on the original divisions only | 2012-07-28 to 2026-11-01 | 26,176 | 0.9813 | 0.5845 | 0.1996 | 0.0107 | 0.520 | 0.912 | 1.316 | 2.9090 | 0.6862 | 0.6928 |
| Poisson (Maher) | own-country ratings walk-forward; shared models trained on the original divisions only | 2012-07-28 to 2026-11-01 | 26,176 | 0.9815 | 0.5846 | 0.1996 | 0.0141 | 0.520 | 0.912 | 1.316 | 2.9090 | 0.6861 | 0.6932 |
| Frequency baseline | previous 5 seasons of the same league | 2012-07-28 to 2026-11-01 | 26,176 | 1.0698 | 0.6470 | 0.2295 | 0.0047 | 0.443 | 0.980 | 1.325 | 3.0274 | 0.6888 | 0.6902 |

| Comparison (A vs B) | Metric | n | Mean A − B | 95% CI | Verdict |
|---|---|---:|---:|---|---|
| Dixon-Coles vs Learned ensemble (champion) | log_loss | 26,176 | +0.0062 | +0.0047 .. +0.0078 | B better |
| Dixon-Coles vs Learned ensemble (champion) | rps | 26,176 | +0.0018 | +0.0014 .. +0.0023 | B better |
| Elo K=15 vs Learned ensemble (champion) | log_loss | 26,176 | +0.0044 | +0.0034 .. +0.0055 | B better |
| Elo K=15 vs Learned ensemble (champion) | rps | 26,176 | +0.0013 | +0.0010 .. +0.0017 | B better |
| LightGBM features vs Learned ensemble (champion) | log_loss | 26,176 | +0.0022 | +0.0012 .. +0.0031 | B better |
| LightGBM features vs Learned ensemble (champion) | rps | 26,176 | +0.0006 | +0.0003 .. +0.0008 | B better |
| Learned ensemble (champion) vs Bookmaker closing odds (benchmark) | log_loss | 26,176 | +0.0168 | +0.0144 .. +0.0192 | B better |
| Learned ensemble (champion) vs Bookmaker closing odds (benchmark) | rps | 26,176 | +0.0051 | +0.0044 .. +0.0059 | B better |

## Paired comparisons, validation

| Comparison (A vs B) | Metric | n | Mean A − B | 95% CI | Verdict |
|---|---|---:|---:|---|---|
| Frequency baseline vs Elo K=15 | log_loss | 26,057 | +0.0721 | +0.0633 .. +0.0812 | B better |
| Frequency baseline vs Elo K=15 | rps | 26,057 | +0.0246 | +0.0216 .. +0.0277 | B better |
| Frequency baseline vs Elo K=15 | score_nll | 26,057 | +0.0993 | +0.0860 .. +0.1131 | B better |
| Elo K=15 vs Poisson (Maher) | log_loss | 26,057 | -0.0001 | -0.0015 .. +0.0012 | inconclusive |
| Elo K=15 vs Poisson (Maher) | rps | 26,057 | +0.0001 | -0.0003 .. +0.0004 | inconclusive |
| Elo K=15 vs Poisson (Maher) | score_nll | 26,057 | +0.0061 | +0.0023 .. +0.0101 | B better |
| Poisson (Maher) vs Dixon-Coles | log_loss | 26,057 | +0.0004 | -0.0000 .. +0.0009 | inconclusive |
| Poisson (Maher) vs Dixon-Coles | rps | 26,057 | +0.0000 | -0.0000 .. +0.0001 | inconclusive |
| Poisson (Maher) vs Dixon-Coles | score_nll | 26,057 | +0.0004 | -0.0001 .. +0.0010 | inconclusive |
| Dixon-Coles vs DC + shot-xG 25% | log_loss | 26,057 | -0.0006 | -0.0012 .. +0.0000 | inconclusive |
| Dixon-Coles vs DC + shot-xG 25% | rps | 26,057 | -0.0001 | -0.0002 .. +0.0001 | inconclusive |
| Dixon-Coles vs DC + shot-xG 25% | score_nll | 26,057 | -0.0011 | -0.0026 .. +0.0005 | inconclusive |
| DC + shot-xG 25% vs LightGBM features | log_loss | 26,057 | +0.0036 | +0.0022 .. +0.0051 | B better |
| DC + shot-xG 25% vs LightGBM features | rps | 26,057 | +0.0011 | +0.0006 .. +0.0015 | B better |
| DC + shot-xG 25% vs LightGBM features | score_nll | 26,057 | +0.0065 | +0.0040 .. +0.0091 | B better |
| Dixon-Coles vs LightGBM stacked on DC | log_loss | 26,057 | +0.0034 | +0.0023 .. +0.0047 | B better |
| Dixon-Coles vs LightGBM stacked on DC | rps | 26,057 | +0.0011 | +0.0007 .. +0.0015 | B better |
| Dixon-Coles vs LightGBM stacked on DC | score_nll | 26,057 | +0.0077 | +0.0056 .. +0.0097 | B better |
| Dixon-Coles vs Learned ensemble (champion) | log_loss | 26,057 | +0.0036 | +0.0028 .. +0.0045 | B better |
| Dixon-Coles vs Learned ensemble (champion) | rps | 26,057 | +0.0011 | +0.0009 .. +0.0014 | B better |
| Dixon-Coles vs Learned ensemble (champion) | score_nll | 26,057 | +0.0098 | +0.0081 .. +0.0115 | B better |
| DC + shot-xG 25% vs Learned ensemble (champion) | log_loss | 26,057 | +0.0042 | +0.0032 .. +0.0053 | B better |
| DC + shot-xG 25% vs Learned ensemble (champion) | rps | 26,057 | +0.0012 | +0.0010 .. +0.0015 | B better |
| DC + shot-xG 25% vs Learned ensemble (champion) | score_nll | 26,057 | +0.0109 | +0.0090 .. +0.0127 | B better |
| Multinomial logistic vs Dixon-Coles | log_loss | 26,057 | -0.0030 | -0.0046 .. -0.0014 | A better |
| Multinomial logistic vs Dixon-Coles | rps | 26,057 | -0.0011 | -0.0015 .. -0.0007 | A better |
| Learned ensemble (champion) vs Bookmaker closing odds (benchmark) | log_loss | 26,053 | +0.0136 | +0.0115 .. +0.0156 | B better |
| Learned ensemble (champion) vs Bookmaker closing odds (benchmark) | rps | 26,053 | +0.0042 | +0.0036 .. +0.0048 | B better |

## Paired comparisons, final test

| Comparison (A vs B) | Metric | n | Mean A − B | 95% CI | Verdict |
|---|---|---:|---:|---|---|
| Elo K=15 vs Learned ensemble (champion) | log_loss | 7,515 | +0.0039 | +0.0026 .. +0.0051 | B better |
| Elo K=15 vs Learned ensemble (champion) | rps | 7,515 | +0.0012 | +0.0008 .. +0.0016 | B better |
| Elo K=15 vs Learned ensemble (champion) | score_nll | 7,515 | +0.0232 | +0.0162 .. +0.0318 | B better |
| Dixon-Coles vs Learned ensemble (champion) | log_loss | 7,515 | +0.0041 | +0.0022 .. +0.0060 | B better |
| Dixon-Coles vs Learned ensemble (champion) | rps | 7,515 | +0.0014 | +0.0009 .. +0.0020 | B better |
| Dixon-Coles vs Learned ensemble (champion) | score_nll | 7,515 | +0.0104 | +0.0065 .. +0.0142 | B better |
| Poisson (Maher) vs Dixon-Coles | log_loss | 7,515 | +0.0008 | +0.0002 .. +0.0015 | B better |
| Poisson (Maher) vs Dixon-Coles | rps | 7,515 | +0.0001 | +0.0000 .. +0.0001 | B better |
| Poisson (Maher) vs Dixon-Coles | score_nll | 7,515 | +0.0001 | -0.0004 .. +0.0007 | inconclusive |
| DC + shot-xG 25% vs Learned ensemble (champion) | log_loss | 7,515 | +0.0044 | +0.0026 .. +0.0062 | B better |
| DC + shot-xG 25% vs Learned ensemble (champion) | rps | 7,515 | +0.0015 | +0.0009 .. +0.0021 | B better |
| DC + shot-xG 25% vs Learned ensemble (champion) | score_nll | 7,515 | +0.0150 | +0.0099 .. +0.0210 | B better |
| LightGBM features vs Learned ensemble (champion) | log_loss | 7,515 | +0.0002 | -0.0006 .. +0.0011 | inconclusive |
| LightGBM features vs Learned ensemble (champion) | rps | 7,515 | +0.0001 | -0.0001 .. +0.0003 | inconclusive |
| LightGBM features vs Learned ensemble (champion) | score_nll | 7,515 | +0.0030 | +0.0020 .. +0.0040 | B better |
| Learned ensemble (champion) vs Bookmaker closing odds (benchmark) | log_loss | 7,515 | +0.0176 | +0.0148 .. +0.0208 | B better |
| Learned ensemble (champion) vs Bookmaker closing odds (benchmark) | rps | 7,515 | +0.0054 | +0.0045 .. +0.0064 | B better |

## Champion by league, final test

| League | Matches | Log loss | RPS | ECE | Accuracy | Goal MAE |
|---|---:|---:|---:|---:|---:|---:|
| D1 | 990 | 0.9822 | 0.1990 | 0.0303 | 0.518 | 0.989 |
| E0 | 1,229 | 0.9869 | 0.2020 | 0.0194 | 0.516 | 0.930 |
| E1 | 1,812 | 1.0403 | 0.2168 | 0.0096 | 0.460 | 0.862 |
| F1 | 998 | 0.9944 | 0.2069 | 0.0189 | 0.529 | 0.951 |
| I1 | 1,238 | 0.9754 | 0.1928 | 0.0332 | 0.534 | 0.850 |
| SP1 | 1,248 | 0.9668 | 0.1928 | 0.0382 | 0.543 | 0.846 |

## Champion calibration, final test

| Top-outcome probability | Matches | Mean predicted | Hit rate | Log loss |
|---|---:|---:|---:|---:|
| (0.0, 0.4] | 1,214 | 0.381 | 0.358 | 1.0936 |
| (0.4, 0.5] | 3,014 | 0.447 | 0.452 | 1.0658 |
| (0.5, 0.6] | 1,771 | 0.545 | 0.543 | 0.9981 |
| (0.6, 0.7] | 1,023 | 0.644 | 0.678 | 0.8388 |
| (0.7, 1.0] | 493 | 0.760 | 0.799 | 0.6307 |

| Outcome | Bin | Matches | Mean predicted | Observed |
|---|---|---:|---:|---:|
| home | 0.1–0.2 | 418 | 0.163 | 0.117 |
| home | 0.2–0.3 | 1,062 | 0.253 | 0.235 |
| home | 0.3–0.4 | 1,619 | 0.353 | 0.343 |
| home | 0.4–0.5 | 1,984 | 0.447 | 0.449 |
| home | 0.5–0.6 | 1,180 | 0.546 | 0.535 |
| home | 0.6–0.7 | 802 | 0.645 | 0.690 |
| home | 0.7–0.8 | 312 | 0.741 | 0.782 |
| home | 0.8–0.9 | 105 | 0.833 | 0.848 |
| draw | 0.1–0.2 | 805 | 0.170 | 0.155 |
| draw | 0.2–0.3 | 6,587 | 0.256 | 0.268 |
| away | 0.0–0.1 | 280 | 0.075 | 0.061 |
| away | 0.1–0.2 | 1,354 | 0.156 | 0.123 |
| away | 0.2–0.3 | 2,127 | 0.254 | 0.247 |
| away | 0.3–0.4 | 1,836 | 0.344 | 0.337 |
| away | 0.4–0.5 | 1,030 | 0.447 | 0.457 |
| away | 0.5–0.6 | 591 | 0.544 | 0.560 |
| away | 0.6–0.7 | 221 | 0.639 | 0.638 |

## Ensemble weights by season

| Season | dixon_coles | dc_xg25 | elo_k15 | gbm_features | scale | ρ |
|---|---:|---:|---:|---:|---:|---:|
| 2012 | 0.00 | 0.42 | 0.25 | 0.33 | +0.020 | -0.074 |
| 2013 | 0.00 | 0.37 | 0.25 | 0.37 | +0.019 | -0.069 |
| 2014 | 0.12 | 0.21 | 0.21 | 0.45 | +0.012 | -0.053 |
| 2015 | 0.18 | 0.16 | 0.15 | 0.51 | +0.008 | -0.042 |
| 2016 | 0.37 | 0.00 | 0.13 | 0.50 | +0.001 | -0.049 |
| 2017 | 0.39 | 0.00 | 0.09 | 0.52 | +0.006 | -0.069 |
| 2018 | 0.24 | 0.15 | 0.09 | 0.52 | +0.015 | -0.069 |
| 2019 | 0.00 | 0.43 | 0.00 | 0.57 | +0.032 | -0.066 |
| 2020 | 0.00 | 0.36 | 0.00 | 0.64 | +0.029 | -0.054 |
| 2021 | 0.00 | 0.32 | 0.00 | 0.68 | +0.032 | -0.053 |
| 2022 | 0.00 | 0.23 | 0.00 | 0.77 | +0.030 | -0.050 |
| 2023 | 0.00 | 0.22 | 0.00 | 0.78 | +0.019 | -0.043 |
| 2024 | 0.00 | 0.25 | 0.00 | 0.75 | +0.020 | -0.042 |
| 2025 | 0.00 | 0.32 | 0.00 | 0.68 | +0.019 | -0.031 |
| 2026 | 0.00 | 0.34 | 0.01 | 0.65 | +0.030 | -0.036 |

## In-play model

Model `inplay-2026-11-03T15:00:46`; training/test split at 2016-02-01 (915 / 636 StatsBomb matches); average half ends 46.3' and 93.8'; second-half scale by tier {'1': 1.04, '2+': 1.04}.

Checkpoint evaluation on held-out matches (final-result log loss / 'another goal' log loss / 'another goal' ECE):

| Model | 1H 15' | 1H 30' | 2H 45' | 2H 60' | 2H 75' | 2H 85' |
|---|---:|---:|---:|---:|---:|---:|
| constant_rate | 0.920 / 0.305 / 0.023 | 0.890 / 0.411 / 0.030 | 0.766 / 0.517 / 0.056 | 0.665 / 0.647 / 0.047 | 0.513 / 0.698 / 0.063 | 0.291 / 0.551 / 0.028 |
| extrapolation | 1.620 / 0.756 / 0.373 | 1.438 / 0.754 / 0.289 | 1.072 / 0.718 / 0.222 | 0.877 / 0.813 / 0.200 | 0.625 / 0.782 / 0.166 | 0.334 / 0.578 / 0.075 |
| hazard | 0.919 / 0.304 / 0.024 | 0.887 / 0.411 / 0.030 | 0.760 / 0.511 / 0.048 | 0.652 / 0.640 / 0.017 | 0.503 / 0.690 / 0.033 | 0.286 / 0.549 / 0.005 |
| hazard_evidence | 0.923 / 0.305 / 0.021 | 0.889 / 0.410 / 0.024 | 0.759 / 0.506 / 0.033 | 0.652 / 0.638 / 0.019 | 0.503 / 0.692 / 0.031 | 0.287 / 0.550 / 0.009 |

Half-time forecasts on football-data.co.uk HT scores (2016/17–2022/23, all 11 divisions):

| Model | Matches | 1X2 log loss | P(2nd-half goal) predicted | observed | ECE |
|---|---:|---:|---:|---:|---:|
| constant_rate | 43,759 | 0.8266 | 0.737 | 0.779 | 0.042 |
| extrapolation | 43,759 | 1.1899 | 0.628 | 0.779 | 0.233 |
| hazard | 43,759 | 0.8244 | 0.757 | 0.779 | 0.022 |

| Comparison (A vs B) | Metric | n | Mean A − B | 95% CI | Verdict |
|---|---|---:|---:|---|---|
| hazard vs hazard_evidence | 1X2 log loss, held-out SB | 3,816 | -0.0007 | -0.0029 .. +0.0015 | inconclusive |
| constant_rate vs hazard | 1X2 log loss, held-out SB | 3,816 | +0.0060 | +0.0002 .. +0.0123 | B better |
| extrapolation vs hazard | 1X2 log loss, held-out SB | 3,816 | +0.3265 | +0.2722 .. +0.3814 | B better |
| constant_rate vs hazard | 1X2 log loss, half-time | 43,759 | +0.0022 | +0.0018 .. +0.0026 | B better |
| extrapolation vs hazard | 1X2 log loss, half-time | 43,759 | +0.3655 | +0.3556 .. +0.3758 | B better |

Learned multipliers (hazard model):

| Term | Multiplier | s.e. of log |
|---|---:|---:|
| t0 | ×0.758 | 0.059 |
| t1 | ×0.972 | 0.055 |
| t2 | ×0.903 | 0.059 |
| t3 | ×0.932 | 0.174 |
| t4 | ×0.937 | 0.061 |
| t5 | ×1.002 | 0.059 |
| t6 | ×1.030 | 0.074 |
| t7 | ×1.105 | 0.109 |
| lead1 | ×1.069 | 0.063 |
| lead2 | ×1.025 | 0.082 |
| trail1 | ×1.068 | 0.067 |
| trail2 | ×1.123 | 0.093 |
| lead_late | ×0.929 | 0.099 |
| trail_late | ×1.088 | 0.105 |
| man_up | ×1.758 | 0.081 |
| man_down | ×0.587 | 0.137 |

## Experiment log

### Does the model add information beyond opening odds?

- Hypothesis: If the model captures information the market misses, a log-linear blend of model and de-vigged opening odds (weights fitted on prior seasons only) will beat the opening odds alone.
- Result: Validation 2013-2022 (n=23678): opening market 0.9821, model 0.9938, blend 0.9822; market-blend diff -0.0000 (95% CI -0.0001..+0.0000); mean model weight 0.00.
- Decision: reject: no measurable information beyond the opening market

### Draw calibration of the champion

- Hypothesis: Poisson-family models are known to under-predict draws; check whether the Dixon-Coles dependence term and learned ensemble remove the bias.
- Result: Validation n=26057: mean predicted draw 0.255 vs observed 0.255 (gap +0.001). League gaps: D1 -0.004, E0 +0.008, E1 +0.002, F1 -0.001, I1 -0.007, SP1 +0.003
- Decision: no change needed

### Monte Carlo simulation vs exact scoreline enumeration

- Hypothesis: For a fixed pair of goal rates, Monte Carlo adds sampling error without adding information; exact enumeration should be faster and error-free. Monte Carlo is only worth it when mixing over uncertain inputs (lineups, scenarios).
- Result: Exact markets: 0.29 ms/match. Monte Carlo RMSE on P(home) at 10k sims up to 0.0058 (~0.6 pp) at 0.5 ms/match; 100k sims needed for <0.2 pp.
- Decision: adopt exact enumeration for single-scenario forecasts; use mixtures of exact matrices for scenario uncertainty

### GPU vs CPU for gradient boosting

- Hypothesis: The tabular training set (~70k rows, 26 features) is too small for GPU acceleration to pay off.
- Result: rows=67333, lightgbm_cpu_4threads_s=0.73, xgboost_cpu_s=4.22, xgboost_cuda_s=5.39
- Decision: no meaningful GPU benefit (speed-up x0.78); fastest is lightgbm_cpu_4threads_s; keep CPU training with 4 threads

### Spread calibration of the ensemble (challenger v2)

- Hypothesis: Validation reliability showed the geometric-mean ensemble is slightly under-confident (temperature exponent 1.066); learning a spread multiplier on the home-away log-rate difference (fitted per season on the previous three seasons) should sharpen forecasts while keeping a coherent score distribution.
- Result: Validation n=26057: log loss 0.9947 -> 0.9945 (diff +0.0002, CI +0.0001..+0.0004); ECE 0.0083 -> 0.0078; scoreline NLL diff +0.00001 (CI -0.00016..+0.00020). Fitted spread 0.97-1.05 (below 1 only in the 2020-21 empty-stadium seasons).
- Decision: reject for promotion: statistically detectable 1X2 gain is practically negligible and the scoreline criterion is not met

### Elo K-factor

- Hypothesis: A lower K (slower updates) should suit league football where single results are noisy.
- Result: Validation n=26057 log loss: K=15 0.9990, K=20 1.0001, K=30 1.0031.
- Decision: adopt K=15 as the Elo baseline

### Dixon-Coles low-score correction vs independent Poisson

- Hypothesis: The Dixon-Coles dependence term should improve draw and low-score probabilities.
- Result: Log loss diff (Poisson - DC) +0.0004 (95% CI -0.0000..+0.0009); scoreline NLL +0.0004 (95% CI -0.0001..+0.0010); ECE 0.0117 -> 0.0077.
- Decision: adopt (small but consistent calibration benefit; on the final test the log loss gain was +0.0008, CI +0.0002..+0.0015)

### Time-decay half-life for team strengths

- Hypothesis: Exponential down-weighting of old matches should beat both slower and faster decay.
- Result: Scoreline NLL: xi=0.0010 2.8939, xi=0.0019 2.8929, xi=0.0030 2.8957; ridge 3.0 2.8950.
- Decision: adopt xi=0.0019/day (half-life about one year) and ridge 1.0

### Shot-based expected-goals proxy as a response

- Hypothesis: Shots on target carry less noise than goals, so blending a shot-based xG proxy into the strength fit should improve forecasts.
- Result: Log loss: 0% 0.9984, 25% 0.9989, 50% 1.0022, 75% 1.0089; ECE rises to 0.0406 at 75%. Goal MAE improves slightly (0.902 -> 0.894).
- Decision: reject as a standalone model; retained as an ensemble component where it earns weight

### Gradient boosting on engineered features

- Hypothesis: LightGBM with Elo, rolling form, shot volumes, rest and promotion flags should capture non-linear effects the rating models miss.
- Result: Log loss 0.9953 vs DC 0.9984; DC-xG vs GBM diff +0.0036 (95% CI +0.0022..+0.0051). Stacked on DC rates: 0.9949.
- Decision: adopt as ensemble component

### Learned ensemble of goal-rate models

- Hypothesis: A geometric blend of component goal rates with weights learned on prior seasons should beat every component while keeping a coherent score distribution.
- Result: vs DC: log loss +0.0036 (95% CI +0.0028..+0.0045), scoreline NLL +0.0098 (95% CI +0.0081..+0.0115). vs bookmaker close: +0.0136 (95% CI +0.0115..+0.0156) (bookmaker better).
- Decision: adopt as champion

### Multinomial logistic regression baseline

- Hypothesis: A linear model on the same features gives a strong 1X2-only baseline.
- Result: Log loss 0.9954; logistic vs DC -0.0030 (95% CI -0.0046..-0.0014) (negative = logistic better).
- Decision: keep as research baseline only (no score distribution)

### Second-half goals before and after the 2023 stoppage-time change (analysis only)

- Hypothesis: Longer added time since 2023/24 should raise second-half goals relative to a model fitted on 2015/16 half lengths. This uses final-test seasons for diagnosis only, not for tuning.
- Result: Mean (actual - predicted) second-half goals: 2016-2022 +0.004, 2023-2026 -0.001. Per season: 2016: 1.60 vs 1.49, 2017: 1.50 vs 1.53, 2018: 1.58 vs 1.53, 2019: 1.55 vs 1.54, 2020: 1.48 vs 1.50, 2021: 1.53 vs 1.53, 2022: 1.46 vs 1.55, 2023: 1.57 vs 1.58, 2024: 1.51 vs 1.57, 2025: 1.53 vs 1.53, 2026: 1.69 vs 1.62
- Decision: no era effect detected

### Ensemble including the stacked LightGBM model

- Hypothesis: The stacked LightGBM (features plus Dixon-Coles rates) was the strongest single validation model; adding it to the ensemble components should improve the ensemble.
- Result: Validation n=26057: log loss 0.9947 -> 0.9947 (diff -0.0000, CI -0.0002..+0.0001); scoreline NLL 2.8832 -> 2.8833 (diff -0.0001, CI -0.0005..+0.0001).
- Decision: reject: no credible improvement over the current ensemble

### Forecast quality under lineup uncertainty

- Hypothesis: Team-only forecasts implicitly assume a typical lineup, so they should be worse when the confirmed XI deviates strongly from the team's usual XI; an XI adjustment should help most there.
- Result: usual XIs (lowest third of deviation): n=244, team-only LL 0.9768, XI-adjusted 0.9760; moderate rotation: n=243, team-only LL 0.9600, XI-adjusted 0.9600; heavy rotation (top third): n=243, team-only LL 0.9630, XI-adjusted 0.9609
- Decision: analysis only (small samples; see the XI experiment for the adoption decision)

### Confirmed starting XI quality vs team-only forecasts

- Hypothesis: If confirmed lineups carry information beyond team-level ratings, adjusting pre-match goal rates by the starting XI's prior xG+xA per 90 (shrunk toward positional means, relative to the team's recent XIs) will improve held-out forecasts.
- Result: Tested on 730 held-out 2015/16 matches (fitted on 661 earlier ones). The starting XI's prior xG+xA index, relative to the team's usual XI, gets coefficient +0.14 on log goal rate; 1X2 log loss change +0.0009 (95% CI -0.0001..+0.0019), scoreline NLL change +0.0015 (CI -0.0003..+0.0031). Not adopted: the evidence is not strong enough on this sample, so forecasts stay team-level.
- Decision: reject (insufficient evidence)

### Dynamic ratings: online Poisson and pi-ratings

- Hypothesis: Match-by-match rating updates (online Poisson gradient steps; pi-ratings of Constantinou and Fenton 2013) adapt faster than weekly refits with a one-year half-life and should improve forecasts, either alone, inside the ensemble or as LightGBM features.
- Result: Validation (n=26057): online Poisson lr 0.015 log loss 0.9971 / score NLL 2.8913 vs Dixon-Coles 0.9984 / 2.8929 (best of 7 settings, so slightly optimistic); pi-ratings best 0.9997 (worse than Elo K=15 0.9990). Ensemble with online Poisson: log loss diff +0.0001 (CI -0.0000..+0.0002); with pi-ratings too: +0.0000. LightGBM with pi and online-Poisson features: log loss 0.9953 -> 0.9947 (diff +0.0006, CI +0.0001..+0.0012), score NLL diff +0.0021 (CI +0.0013..+0.0029).
- Decision: adopt GBM features

### Ensemble built on the improved LightGBM

- Hypothesis: If LightGBM with pi-rating and online-Poisson features is better on its own, an ensemble using it (with or without the online Poisson component) should beat the current champion.
- Result: Validation n=26,057: ensemble with LightGBM v2 log loss 0.9946 vs champion 0.9947 (diff +0.0001, CI -0.0002..+0.0004), score NLL diff -0.0002 (CI -0.0006..+0.0002); adding online Poisson as well: +0.0001 (CI -0.0002..+0.0004). LightGBM v2 takes 93% of the blend weight.
- Decision: reject: the champion already captures this information; no promotion

### Generalisation to seven leagues added after the model freeze

- Hypothesis: If the champion captures general football structure rather than quirks of the leagues it was selected on, its ranking should hold on divisions never used for model selection, ensemble weighting or LightGBM training (own-country ratings only, walk-forward).
- Result: n=26,176 matches (Eredivisie, Primeira Liga, Pro League, Scottish Premiership and Championship, Super Lig, Super League Greece; 2012/13 onwards). Log loss: ensemble 0.9751, LightGBM 0.9772, Elo 0.9795, Dixon-Coles 0.9813, frequency 1.0698, bookmaker close 0.9583. Ensemble vs DC +0.0062 (CI +0.0047..+0.0078); vs LightGBM +0.0022 (CI +0.0012..+0.0031); bookmaker better by 0.0168. Core predictions unchanged (max rate change 0.0).
- Decision: confirmed: ranking generalises; leagues added to coverage

### In-play short-horizon and next-goal calibration (held-out replay)

- Hypothesis: The hazard model's 10-minute goal probabilities and next-scorer probabilities should be calibrated on held-out StatsBomb matches, beating a constant base rate.
- Result: 10176 checkpoints every 5 minutes: P(goal in next 10') mean 0.247 vs observed 0.250, ECE 0.004, log loss 0.5595 vs constant base rate 0.5628; next-goal (home/away/none) log loss 0.9360, accuracy 0.546; leader goes on to win: predicted 0.816 vs observed 0.809 over 5303 checkpoints with a lead, ECE 0.016, Brier 0.1299.
- Decision: keep (calibrated)

### Reproducible ensemble weight fitting

- Hypothesis: The Nelder-Mead fit of ensemble weights can stop at different local optima when only the floating-point summation order of the training rows changes; deterministic ordering plus multi-start L-BFGS should make the fit reproducible without hurting accuracy.
- Result: Season 2017 example: Nelder-Mead objective 38,587.30 vs multi-start L-BFGS 38,585.69 (different weights). New fit is identical after shuffling input rows (max rate difference 0.0). Validation log loss diff +0.00005 (CI -0.00004..+0.00014), score NLL +0.00022 (CI -0.00009..+0.00059); final-test predictions unchanged (same weights for 2023-2026).
- Decision: adopt for reproducibility as ensemble@v3 (accuracy unchanged)

### Provider xG and field tilt beyond the champion (StatsBomb 2015/16)

- Hypothesis: Teams whose recent StatsBomb xG exceeds their goals (for and against) have been unlucky and should regress upwards; adjusting the champion's goal rates by the log ratio of recent xG to recent goals (and by field-tilt difference) should improve held-out forecasts.
- Result: Fitted on 587 matches before 2016-01-15, tested on 730. xG/goals adjustment: coefficient +0.21, log loss 0.9666 -> 0.9673 (diff -0.0007, CI -0.0039..+0.0025), score NLL diff -0.0057 (CI -0.0121..+0.0008). Adding field tilt (coefficient +0.06): log loss diff +0.0001 (CI -0.0031..+0.0035).
- Decision: not adopted: no credible gain on one season of provider xG (the champion already uses shots on target)

### Full reproduction from raw data

- Hypothesis: An isolated run (separate database copy and cache directory) that recomputes every walk-forward model from the raw match table should reproduce the production validation results exactly.
- Result: Recomputed 17 models from scratch (about 50 minutes at the 4-thread budget). Champion ensemble: identical predictions on all 26,057 validation and 6,912 completed final-test matches (max rate difference 0.0). 15 of 17 models identical; Elo K=20/K=30 differed by up to 5e-4 log loss because their cached predictions dated from the first run and Elo outcome mappings were fitted with an order-sensitive Nelder-Mead. After switching those fits to sorted inputs with tight L-BFGS (order-independent, verified by shuffling) all 17 models agree within 2e-8.
- Decision: adopt order-independent fitting; pipeline reproducible from raw data

### Rolling league home-advantage feature

- Hypothesis: Home advantage is not constant (it collapsed in the 2020/21 empty-stadium season); giving LightGBM the league's rolling home/away goal ratio and goal level over its previous 380 matches should let it adapt within seasons.
- Result: Validation n=26,057: LightGBM log loss 0.9953 -> 0.9953 (diff -0.0000, CI -0.0003..+0.0003), score NLL diff +0.0001 (CI -0.0004..+0.0005). 2020/21 season: log loss 1.0087 -> 1.0083; mean predicted home win 0.437 -> 0.436 vs observed 0.402.
- Decision: reject: no credible gain

### Second-half rate recalibration from half-time data

- Hypothesis: The hazard model (fitted on StatsBomb 2015/16 top-flight matches) mis-states second-half scoring at half-time; a second-half rate scale per tier group, fitted on 2012-2015 football-data HT/FT results, will improve calibration on 2016-2022.
- Result: tier 1: scale x1.04 (train n=5000), test n=8000: P(2nd-half goal) 0.769 -> 0.782 vs observed 0.788, ECE 0.021 -> 0.008, another-goal LL +0.0011 (CI +0.0004..+0.0017), 1X2 LL +0.0007 (CI +0.0003..+0.0011) adopted; tier 2+: scale x1.04 (train n=5000), test n=8000: P(2nd-half goal) 0.742 -> 0.755 vs observed 0.759, ECE 0.019 -> 0.006, another-goal LL +0.0007 (CI +0.0000..+0.0014), 1X2 LL +0.0006 (CI +0.0002..+0.0010) adopted
- Decision: adopt per-tier scales where credible: tier 1 x1.04, tier 2+ x1.04

### Favourite under-confidence: global spread challenger (re-test on all leagues)

- Hypothesis: The Learning weakness report flags under-confident favourites (home favourites at 70-100%: 77.8% predicted vs 81.5% observed, z=6.4) and slightly under-predicted draws. A learned spread on the home-away log-rate difference, fitted each season on the previous three, should correct it.
- Result: Validation, six primary leagues: 1X2 log loss +0.0002 (CI +0.0001..+0.0004), RPS +0.00005 (CI +0.00001..+0.00009), scoreline NLL +0.00001 (CI -0.00017..+0.00019); all 18 divisions similar; no primary league significantly worse. Favourites at 70%+: predicted 77.2% -> 77.5% vs observed 81.2%; class-wise ECE 0.0088 -> 0.0083.
- Decision: not promoted: the gain is tiny, the scoreline criterion is not met and the favourite gap remains

### Favourite under-confidence: gap-dependent sharpening challenger

- Hypothesis: If lopsided matches are under-confident because large expected-goal gaps are understated, sharpening that grows with the gap (half-difference x (a + b|half-difference|), fitted each season on the previous three) should close the favourite gap.
- Result: Fitted b is negative in every recent season (e.g. 2025: a=1.067, b=-0.139): the goal likelihood prefers less sharpening for lopsided matches. Validation log loss +0.00007 (CI -0.00009..+0.00025), scoreline NLL -0.00017 (CI -0.00041..+0.00006); favourites at 70%+ still 77.6% predicted vs 81.2% observed.
- Decision: rejected: the favourite under-confidence comes from the shape of the score distribution (outcome given expected goals), not from the expected-goal gap; a fix needs a different score model (e.g. dispersion or dependence that varies with strength) and is recorded as future work
