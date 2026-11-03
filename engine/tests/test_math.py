import numpy as np
import pandas as pd
import pytest

from fie import metrics
from fie.inplay import InPlayModel
from fie.ratings import StrengthModel
from fie.scoreline import check_consistency, dixon_coles_matrix, markets, poisson_matrix


@pytest.mark.parametrize("lh,la,rho", [(1.5, 1.1, 0.0), (2.8, 0.4, -0.1), (0.6, 0.7, 0.12), (4.0, 3.5, -0.05)])
def test_markets_consistent_with_matrix(lh, la, rho):
    m = dixon_coles_matrix(lh, la, rho)
    mk = markets(m)
    assert check_consistency(m, mk) == []
    assert (m >= 0).all()
    assert abs(mk["p_home"] + mk["p_draw"] + mk["p_away"] - 1) < 1e-9
    for line, v in mk["over_under"].items():
        assert abs(v["over"] + v["under"] - 1) < 1e-12
    assert mk["btts"] <= min(mk["p_home_scores"], mk["p_away_scores"]) + 1e-12
    assert mk["p_goal_each_half"] <= min(mk["p_goal_first_half"], mk["p_goal_second_half"]) + 1e-12
    assert abs(mk["home_clean_sheet"] - (1 - mk["p_away_scores"])) < 1e-12


def test_dixon_coles_preserves_mass_and_reduces_to_poisson():
    m0 = poisson_matrix(1.3, 1.1)
    m1 = dixon_coles_matrix(1.3, 1.1, 0.0)
    assert np.allclose(m0 / m0.sum(), m1)
    m2 = dixon_coles_matrix(1.3, 1.1, -0.1)
    assert m2[0, 0] > m1[0, 0] and m2[1, 1] > m1[1, 1]


def test_metrics_known_values():
    p = np.array([[1.0, 0.0, 0.0], [1 / 3, 1 / 3, 1 / 3]])
    y = np.array([0, 2])
    assert metrics.brier_rows(p, y)[0] == 0
    assert np.isclose(metrics.log_loss_rows(p, y)[1], np.log(3))
    assert np.isclose(metrics.rps_rows(np.array([[0.0, 0.0, 1.0]]), np.array([0]))[0], 1.0)
    d = metrics.devig([2.0], [3.5], [4.0])
    assert np.isclose(d.sum(), 1.0)


def _synthetic(n_teams=12, seasons=3, seed=0):
    rng = np.random.default_rng(seed)
    att = rng.normal(0, 0.3, n_teams)
    dfn = rng.normal(0, 0.3, n_teams)
    rows = []
    date = pd.Timestamp("2015-08-01")
    for s in range(seasons):
        for h in range(n_teams):
            for a in range(n_teams):
                if h == a:
                    continue
                lh = np.exp(0.15 + 0.25 + att[h] - dfn[a])
                la = np.exp(0.15 + att[a] - dfn[h])
                rows.append({"date": date, "home": h, "away": a, "hg": rng.poisson(lh), "ag": rng.poisson(la), "tier": 1})
                date += pd.Timedelta(hours=10)
    return pd.DataFrame(rows), att, dfn


def test_strength_model_recovers_parameters():
    frame, att, dfn = _synthetic(seasons=6)
    model = StrengthModel(xi=0.0, ridge=0.01, window_days=100000).fit(frame, frame["date"].max() + pd.Timedelta(days=1))
    a, d, mu, h, rho = model.params
    est = np.array([a[model.team_index[t]] for t in range(len(att))])
    assert np.corrcoef(est, att)[0, 1] > 0.85
    assert abs(h[0] - 0.25) < 0.08
    assert abs(rho) < 0.1


def test_strength_model_ignores_future_matches():
    frame, *_ = _synthetic()
    cutoff = frame["date"].iloc[len(frame) // 2]
    m1 = StrengthModel(xi=0.002).fit(frame, cutoff)
    shuffled = frame.copy()
    future = shuffled["date"] >= cutoff
    shuffled.loc[future, ["hg", "ag"]] = 9
    m2 = StrengthModel(xi=0.002).fit(shuffled, cutoff)
    assert np.allclose(m1.params[0], m2.params[0])


def _toy_inplay():
    m = InPlayModel()
    m.columns = m.feature_names()
    m.beta = np.zeros(len(m.columns))
    m.se = np.zeros(len(m.columns))
    return m


def test_inplay_probabilities_valid_and_monotone_in_time():
    m = _toy_inplay()
    base = {"period": 2, "home_goals": 1, "away_goals": 0, "reds_home": 0, "reds_away": 0,
            "xg_home": None, "xg_away": None, "exp_home_so_far": None, "exp_away_so_far": None, "recent_xg_home": None, "recent_xg_away": None}
    early = m.forecast({**base, "clock": 50.0}, 1.5 / 94, 1.1 / 94)
    late = m.forecast({**base, "clock": 85.0}, 1.5 / 94, 1.1 / 94)
    for f in (early, late):
        assert abs(f["p_home"] + f["p_draw"] + f["p_away"] - 1) < 1e-9
        assert 0 <= f["p_another_goal"] <= 1
        assert f["p_goal_5"] <= f["p_goal_10"] <= f["p_goal_15"] <= f["p_another_goal"] + 1e-9
        assert f["matrix_full"][0, :].sum() < 1e-12
    assert late["p_another_goal"] < early["p_another_goal"]
    assert late["p_home"] > early["p_home"]
    assert abs(early["exp_final_total"] - 1 - early["exp_remaining"]) < 1e-9


def test_inplay_goal_changes_score_state():
    m = _toy_inplay()
    s = {"period": 2, "clock": 60.0, "home_goals": 0, "away_goals": 0, "reds_home": 0, "reds_away": 0,
         "xg_home": None, "xg_away": None, "exp_home_so_far": None, "exp_away_so_far": None, "recent_xg_home": None, "recent_xg_away": None}
    level = m.forecast(s, 1.4 / 94, 1.2 / 94)
    ahead = m.forecast({**s, "home_goals": 1}, 1.4 / 94, 1.2 / 94)
    assert ahead["p_home"] > level["p_home"]
    assert ahead["p_away"] < level["p_away"]


def test_inplay_red_card_effect_direction():
    m = _toy_inplay()
    m.beta[m.columns.index("man_up")] = 0.3
    m.beta[m.columns.index("man_down")] = -0.3
    s = {"period": 2, "clock": 55.0, "home_goals": 0, "away_goals": 0, "reds_home": 0, "reds_away": 0,
         "xg_home": None, "xg_away": None, "exp_home_so_far": None, "exp_away_so_far": None, "recent_xg_home": None, "recent_xg_away": None}
    a = m.forecast(s, 1.3 / 94, 1.3 / 94)
    b = m.forecast({**s, "reds_away": 1}, 1.3 / 94, 1.3 / 94)
    assert b["p_home"] > a["p_home"]


def test_score_shape_families_preserve_means_and_nest_dixon_coles():
    from fie import shapes
    lh, la = np.array([1.6, 0.7, 2.4]), np.array([1.1, 1.9, 0.5])
    g = np.arange(shapes.G)
    base = shapes.dixon_coles(lh, la, -0.05)
    for i in range(3):
        assert np.allclose(base[i], dixon_coles_matrix(lh[i], la[i], -0.05))
    assert np.allclose(shapes.bivariate(lh, la, -0.05, 0.0), base, atol=1e-8)
    assert np.allclose(shapes.com_poisson(lh, la, -0.05, 1.0, 1.0), base, atol=1e-4)
    for m in (shapes.bivariate(lh, la, 0.0, 0.3), shapes.com_poisson(lh, la, 0.0, 1.3, 1.2)):
        assert np.allclose(m.sum(axis=(1, 2)), 1.0)
        assert np.allclose(m.sum(axis=2) @ g, lh, atol=1e-3) and np.allclose(m.sum(axis=1) @ g, la, atol=1e-3)
    p = shapes.outcome(shapes.diagonal_gap(lh, la, 0.0, 0.2, 0.0))
    assert np.allclose(p.sum(axis=1), 1.0) and (p[:, 1] > shapes.outcome(shapes.dixon_coles(lh, la, 0.0))[:, 1]).all()
