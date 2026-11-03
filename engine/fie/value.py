import numpy as np

from .scoreline import dixon_coles_matrix

SELECTIONS = {"1x2": ("home", "draw", "away"), "ou": ("over", "under"), "ah": ("home", "away")}


def margin(prices):
    return float(sum(1.0 / p for p in prices) - 1.0)


def devig(prices, method="power"):
    inv = np.array([1.0 / p for p in prices], dtype=float)
    if method == "proportional":
        return inv / inv.sum()
    lo, hi = 0.5, 4.0
    for _ in range(80):
        k = (lo + hi) / 2
        if (inv ** k).sum() > 1:
            lo = k
        else:
            hi = k
    p = inv ** ((lo + hi) / 2)
    return p / p.sum()


def _half_lines(line):
    frac = abs(line * 4) % 2
    if frac == 1:
        return [line - 0.25, line + 0.25]
    return [line]


def ah_return(home_goals, away_goals, line, selection, odds):
    parts = _half_lines(line)
    total = 0.0
    for part in parts:
        margin_ = (home_goals - away_goals + part) if selection == "home" else (away_goals - home_goals - part)
        if margin_ > 0:
            total += odds - 1.0
        elif margin_ < 0:
            total -= 1.0
    return total / len(parts)


def settle(market, line, selection, odds, home_goals, away_goals):
    if market == "1x2":
        won = {"home": home_goals > away_goals, "draw": home_goals == away_goals, "away": home_goals < away_goals}[selection]
        return odds - 1.0 if won else -1.0
    if market == "ou":
        total = home_goals + away_goals
        if total == line:
            return 0.0
        won = total > line if selection == "over" else total < line
        return odds - 1.0 if won else -1.0
    if market == "ah":
        return ah_return(home_goals, away_goals, line, selection, odds)
    raise ValueError(market)


def outcome_distribution(matrix, market, line, selection):
    n = matrix.shape[0]
    win = push = 0.0
    lose = 0.0
    for h in range(n):
        for a in range(n):
            p = matrix[h, a]
            r = settle(market, line, selection, 2.0, h, a)
            if r > 0.5:
                win += p
            elif r > 0:
                win += p / 2
                push += p / 2
            elif r == 0:
                push += p
            elif r > -0.75:
                push += p / 2
                lose += p / 2
            else:
                lose += p
    return {"win": win, "push": push, "lose": lose}


def expected_return(matrix, market, line, selection, odds):
    n = matrix.shape[0]
    return float(sum(matrix[h, a] * settle(market, line, selection, odds, h, a) for h in range(n) for a in range(n)))


def break_even_odds(matrix, market, line, selection):
    d = outcome_distribution(matrix, market, line, selection)
    return (d["win"] + d["lose"]) / d["win"] if d["win"] > 0 else float("inf")


def evaluate(lh, la, rho, market, line, prices):
    m = dixon_coles_matrix(lh, la, rho)
    m = m / m.sum()
    sels = SELECTIONS[market]
    fair = devig([prices[s] for s in sels])
    out = []
    for s, f in zip(sels, fair):
        d = outcome_distribution(m, market, line, s)
        out.append({"selection": s, "odds": prices[s], "model_win": d["win"], "model_push": d["push"],
                    "implied": 1.0 / prices[s], "market_prob": float(f), "break_even_odds": break_even_odds(m, market, line, s),
                    "ev": expected_return(m, market, line, s, prices[s])})
    return {"market": market, "line": line, "margin": margin([prices[s] for s in sels]), "selections": out}
