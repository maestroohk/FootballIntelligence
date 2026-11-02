import numpy as np
from scipy.stats import poisson

MAX_GOALS = 10
FIRST_HALF_SHARE = 0.445


def poisson_matrix(lh: float, la: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    goals = np.arange(max_goals + 1)
    return np.outer(poisson.pmf(goals, lh), poisson.pmf(goals, la))


def dc_tau(lh, la, rho):
    return np.array([[1 - lh * la * rho, 1 + lh * rho], [1 + la * rho, 1 - rho]])


def dixon_coles_matrix(lh: float, la: float, rho: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    m = poisson_matrix(lh, la, max_goals)
    m[:2, :2] *= dc_tau(lh, la, rho)
    m = np.clip(m, 0.0, None)
    return m / m.sum()


def normalise(m: np.ndarray) -> np.ndarray:
    m = np.clip(m, 0.0, None)
    return m / m.sum()


def outcome_probs(m: np.ndarray):
    return float(np.tril(m, -1).sum()), float(np.trace(m)), float(np.triu(m, 1).sum())


def total_goal_dist(m: np.ndarray) -> np.ndarray:
    n = m.shape[0]
    out = np.zeros(2 * n - 1)
    for i in range(n):
        out[i:i + n] += m[i]
    return out


def markets(m: np.ndarray, first_half_share: float = FIRST_HALF_SHARE, top_n: int = 8) -> dict:
    m = normalise(m)
    goals = np.arange(m.shape[0])
    ph, pd, pa = outcome_probs(m)
    eh = float((m.sum(axis=1) * goals).sum())
    ea = float((m.sum(axis=0) * goals).sum())
    totals = total_goal_dist(m)
    n = np.arange(len(totals))
    p_first = float((totals * (1 - (1 - first_half_share) ** n)).sum())
    p_second = float((totals * (1 - first_half_share ** n)).sum())
    p_both_halves = float((totals[1:] * (1 - first_half_share ** n[1:] - (1 - first_half_share) ** n[1:])).sum())
    flat = [(float(m[i, j]), i, j) for i in range(m.shape[0]) for j in range(m.shape[1])]
    flat.sort(reverse=True)
    p_home_scores = float(1 - m[0, :].sum())
    p_away_scores = float(1 - m[:, 0].sum())
    lines = {}
    for line in (0.5, 1.5, 2.5, 3.5, 4.5):
        over = float(totals[n > line].sum())
        lines[str(line)] = {"over": over, "under": 1 - over}
    entropy = float(-(np.array([ph, pd, pa]) * np.log(np.clip([ph, pd, pa], 1e-12, 1))).sum())
    return {
        "p_home": ph, "p_draw": pd, "p_away": pa,
        "exp_home": eh, "exp_away": ea, "exp_total": eh + ea, "exp_diff": eh - ea,
        "p_home_scores": p_home_scores, "p_away_scores": p_away_scores,
        "btts": float(m[1:, 1:].sum()),
        "home_clean_sheet": float(m[:, 0].sum()), "away_clean_sheet": float(m[0, :].sum()),
        "over_under": lines,
        "p_goal_first_half": p_first, "p_goal_second_half": p_second, "p_goal_each_half": p_both_halves,
        "top_scores": [{"home": i, "away": j, "p": p} for p, i, j in flat[:top_n]],
        "total_goals": [float(x) for x in totals[:9]],
        "outcome_entropy": entropy,
        "max_outcome_p": max(ph, pd, pa),
        "matrix": np.round(m[:7, :7], 6).tolist(),
    }


def check_consistency(m: np.ndarray, mk: dict, tol: float = 1e-9) -> list:
    errors = []
    if (m < -tol).any():
        errors.append("negative probability")
    if abs(m.sum() - 1) > 1e-6:
        errors.append("matrix does not sum to 1")
    if abs(mk["p_home"] + mk["p_draw"] + mk["p_away"] - 1) > 1e-6:
        errors.append("1X2 does not sum to 1")
    over25 = sum(m[i, j] for i in range(m.shape[0]) for j in range(m.shape[1]) if i + j >= 3)
    if abs(over25 - mk["over_under"]["2.5"]["over"]) > 1e-6:
        errors.append("over 2.5 inconsistent with matrix")
    if abs((1 - m[0, 0]) - mk["over_under"]["0.5"]["over"]) > 1e-6:
        errors.append("over 0.5 inconsistent with matrix")
    return errors
