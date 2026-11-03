import numpy as np
import pytest

from fie import value
from fie.scoreline import dixon_coles_matrix


def test_margin_and_devig():
    prices = [2.0, 3.5, 4.0]
    assert value.margin(prices) == pytest.approx(1 / 2 + 1 / 3.5 + 1 / 4 - 1)
    for method in ("power", "proportional"):
        p = value.devig(prices, method)
        assert p.sum() == pytest.approx(1.0)
        assert (p > 0).all() and p[0] > p[1] > p[2]


@pytest.mark.parametrize("line,hg,ag,expected", [
    (-0.5, 1, 0, 0.9), (-0.5, 0, 0, -1.0),
    (0.0, 1, 1, 0.0), (0.0, 2, 1, 0.9),
    (-0.25, 1, 1, -0.5), (-0.25, 2, 1, 0.9),
    (-0.75, 1, 0, 0.45), (-0.75, 2, 0, 0.9), (-0.75, 1, 1, -1.0),
    (0.25, 1, 1, 0.45), (0.25, 0, 1, -1.0),
    (-1.0, 1, 0, 0.0), (-1.0, 2, 0, 0.9),
    (1.5, 0, 1, 0.9), (1.5, 0, 2, -1.0),
])
def test_asian_handicap_settlement_home(line, hg, ag, expected):
    assert value.settle("ah", line, "home", 1.9, hg, ag) == pytest.approx(expected)


def test_asian_handicap_sides_mirror():
    assert value.settle("ah", -0.25, "away", 1.9, 1, 1) == pytest.approx(0.45)
    assert value.settle("ah", -0.75, "away", 2.0, 1, 0) == pytest.approx(-0.5)


def test_totals_and_1x2_settlement():
    assert value.settle("ou", 2.5, "over", 2.1, 2, 1) == pytest.approx(1.1)
    assert value.settle("ou", 2.5, "under", 1.8, 2, 1) == -1.0
    assert value.settle("1x2", 0.0, "draw", 3.4, 0, 0) == pytest.approx(2.4)


def test_fair_odds_have_zero_expected_value():
    m = dixon_coles_matrix(1.6, 1.1, -0.05)
    for market, line, sel in (("1x2", 0.0, "home"), ("ou", 2.5, "over"), ("ah", -0.25, "home"), ("ah", -0.75, "away"), ("ah", 0.0, "home")):
        be = value.break_even_odds(m, market, line, sel)
        assert value.expected_return(m, market, line, sel, be) == pytest.approx(0.0, abs=1e-9)


def test_outcome_distribution_sums_to_one():
    m = dixon_coles_matrix(1.3, 1.2, -0.08)
    for line in (-1.75, -0.25, 0.0, 0.5, 1.25):
        d = value.outcome_distribution(m / m.sum(), "ah", line, "home")
        assert d["win"] + d["push"] + d["lose"] == pytest.approx(1.0)


def test_evaluate_matches_matrix_probabilities():
    out = value.evaluate(1.5, 1.0, -0.05, "1x2", 0.0, {"home": 2.0, "draw": 3.6, "away": 4.2})
    m = dixon_coles_matrix(1.5, 1.0, -0.05)
    home = next(s for s in out["selections"] if s["selection"] == "home")
    assert home["model_win"] == pytest.approx(np.tril(m, -1).sum())
    assert home["ev"] == pytest.approx(home["model_win"] * 2.0 - 1.0)
    assert out["margin"] > 0
