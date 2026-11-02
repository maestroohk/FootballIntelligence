import math

import numpy as np

from .scoreline import markets

TIME_BINS = [(1, 0, 15), (1, 15, 30), (1, 30, 45), (1, 45, 99), (2, 45, 60), (2, 60, 75), (2, 75, 90), (2, 90, 199)]
MAX_GOALS = 12

BASE_FEATURES = ["lead1", "lead2", "trail1", "trail2", "lead_late", "trail_late", "man_up", "man_down"]
EVIDENCE_FEATURES = ["xg_ratio", "recent_xg"]


def time_bin(period, clock):
    for k, (p, lo, hi) in enumerate(TIME_BINS):
        if p == period and lo <= clock < hi:
            return k
    return 3 if period == 1 else 7


def state_features(diff, late, reds_own, reds_opp, xg_so_far=None, expected_so_far=None, recent_xg=None, use_evidence=False):
    f = {
        "lead1": float(diff == 1), "lead2": float(diff >= 2), "trail1": float(diff == -1), "trail2": float(diff <= -2),
        "lead_late": float(diff > 0 and late), "trail_late": float(diff < 0 and late),
        "man_up": float(reds_opp - reds_own >= 1), "man_down": float(reds_own - reds_opp >= 1),
    }
    if use_evidence:
        f["xg_ratio"] = math.log((xg_so_far + 0.3) / (expected_so_far + 0.3)) if xg_so_far is not None else 0.0
        f["recent_xg"] = min(recent_xg or 0.0, 1.5)
    return f


def fit_poisson_glm(x, y, offset, l2=1.0, iters=50):
    beta = np.zeros(x.shape[1])
    pen = np.full(x.shape[1], l2)
    for _ in range(iters):
        eta = x @ beta + offset
        mu = np.exp(eta)
        grad = x.T @ (y - mu) - pen * beta
        hess = (x * mu[:, None]).T @ x + np.diag(pen)
        step = np.linalg.solve(hess, grad)
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    eta = x @ beta + offset
    mu = np.exp(eta)
    cov = np.linalg.inv((x * mu[:, None]).T @ x + np.diag(pen))
    return beta, np.sqrt(np.diag(cov))


class InPlayModel:
    def __init__(self, use_evidence=False, mode="hazard"):
        self.use_evidence = use_evidence
        self.mode = mode
        self.beta = None
        self.columns = None
        self.half_end = (47.0, 94.5)
        self.second_half_scale = 1.0

    def feature_names(self):
        return [f"t{k}" for k in range(len(TIME_BINS))] + BASE_FEATURES + (EVIDENCE_FEATURES if self.use_evidence else [])

    def design(self, rows):
        names = self.feature_names()
        x = np.zeros((len(rows), len(names)))
        for i, r in enumerate(rows):
            x[i, r["bin"]] = 1.0
            for j, name in enumerate(names[len(TIME_BINS):]):
                x[i, len(TIME_BINS) + j] = r["f"].get(name, 0.0)
        return x

    def fit(self, panel, half_end=None):
        rows = panel["rows"]
        x = self.design(rows)
        y = np.array([r["goals"] for r in rows], float)
        offset = np.array([math.log(r["lambda_pm"]) for r in rows])
        self.beta, self.se = fit_poisson_glm(x, y, offset)
        self.columns = self.feature_names()
        if half_end:
            self.half_end = half_end
        return self

    def coefficients(self):
        return [{"name": n, "beta": float(b), "se": float(s), "multiplier": float(math.exp(b))} for n, b, s in zip(self.columns, self.beta, self.se)]

    def rate(self, lambda_pm, period, clock, f):
        if period == 2:
            lambda_pm = lambda_pm * self.second_half_scale
        if self.mode == "constant":
            return lambda_pm
        z = self.beta[time_bin(period, clock)]
        for j, name in enumerate(self.columns[len(TIME_BINS):]):
            z += self.beta[len(TIME_BINS) + j] * f.get(name, 0.0)
        return lambda_pm * math.exp(z)

    def remaining_grid(self, period, clock, minutes_end=None):
        h1, h2 = minutes_end or self.half_end
        steps = []
        if period == 1:
            t = clock
            end1 = max(h1, clock + 0.25)
            while t < end1 - 1e-9:
                dt = min(1.0, end1 - t)
                steps.append((1, t, dt))
                t += dt
            t = 45.0
        else:
            t = clock
        end2 = max(h2, t + 0.25)
        while t < end2 - 1e-9:
            dt = min(1.0, end2 - t)
            steps.append((2, t, dt))
            t += dt
        return steps

    def forecast(self, state, lambda_home_pm, lambda_away_pm, half_end=None):
        hg, ag = state["home_goals"], state["away_goals"]
        steps = [] if state.get("finished") else self.remaining_grid(state["period"], state["clock"], half_end)
        n = MAX_GOALS + 1
        dist = np.zeros((n, n))
        dist[min(hg, MAX_GOALS), min(ag, MAX_GOALS)] = 1.0
        ev = {k: state.get(k) for k in ("xg_home", "xg_away", "exp_home_so_far", "exp_away_so_far", "recent_xg_home", "recent_xg_away")}
        rate_cache = {}

        def rates(diff, period, clock):
            key = (diff, time_bin(period, clock), clock >= 70)
            if key not in rate_cache:
                late = clock >= 70
                fh = state_features(diff, late, state["reds_home"], state["reds_away"], ev["xg_home"], ev["exp_home_so_far"], ev["recent_xg_home"], self.use_evidence)
                fa = state_features(-diff, late, state["reds_away"], state["reds_home"], ev["xg_away"], ev["exp_away_so_far"], ev["recent_xg_away"], self.use_evidence)
                rate_cache[key] = (self.rate(lambda_home_pm, period, clock, fh), self.rate(lambda_away_pm, period, clock, fa))
            return rate_cache[key]

        hi = np.arange(n)[:, None]
        ai = np.arange(n)[None, :]
        diffs = np.clip(hi - ai, -3, 3)
        no_goal_curve = []
        total_rate_now = None
        cum_rate = 0.0
        for period, clock, dt in steps:
            rh = np.zeros((n, n))
            ra = np.zeros((n, n))
            for d in range(-3, 4):
                mask = diffs == d
                rh[mask], ra[mask] = rates(d, period, clock)
            cur_d = int(np.clip(hg - ag, -3, 3))
            r_now = rates(cur_d, period, clock)
            if total_rate_now is None:
                total_rate_now = r_now
            cum_rate += (r_now[0] + r_now[1]) * dt
            no_goal_curve.append((clock + dt, math.exp(-cum_rate)))
            tot = rh + ra
            p_none = np.exp(-tot * dt)
            with np.errstate(invalid="ignore", divide="ignore"):
                share_h = np.where(tot > 0, rh / tot, 0.0)
            p_any = 1 - p_none
            new = dist * p_none
            new[1:, :] += (dist * p_any * share_h)[:-1, :]
            new[:, 1:] += (dist * p_any * (1 - share_h))[:, :-1]
            new[-1, :] += (dist * p_any * share_h)[-1, :]
            new[:, -1] += (dist * p_any * (1 - share_h))[:, -1]
            dist = new
        dist /= dist.sum()
        mk = markets(dist)
        now_clock = state["clock"]

        def p_goal_within(k):
            for c, s in no_goal_curve:
                if c >= now_clock + k - 1e-9:
                    return 1 - s
            return 1 - (no_goal_curve[-1][1] if no_goal_curve else 1.0)

        p_another = 1 - (no_goal_curve[-1][1] if no_goal_curve else 1.0)
        rh_now, ra_now = total_rate_now or (0.0, 0.0)
        share_home_next = rh_now / (rh_now + ra_now) if rh_now + ra_now > 0 else 0.5
        exp_final_total = mk["exp_total"]
        lead = hg - ag
        if lead > 0:
            hold, comeback_win, comeback_avoid_defeat = mk["p_home"], mk["p_away"], mk["p_away"] + mk["p_draw"]
        elif lead < 0:
            hold, comeback_win, comeback_avoid_defeat = mk["p_away"], mk["p_home"], mk["p_home"] + mk["p_draw"]
        else:
            hold = comeback_win = comeback_avoid_defeat = None
        return {
            **mk,
            "p_another_goal": p_another,
            "p_goal_5": p_goal_within(5), "p_goal_10": p_goal_within(10), "p_goal_15": p_goal_within(15),
            "exp_remaining": exp_final_total - hg - ag,
            "exp_final_total": exp_final_total,
            "p_next_home": p_another * share_home_next, "p_next_away": p_another * (1 - share_home_next), "p_no_more": 1 - p_another,
            "p_leader_holds": hold, "p_comeback_win": comeback_win, "p_comeback_avoid_defeat": comeback_avoid_defeat,
            "rate_home_per90": rh_now * 90, "rate_away_per90": ra_now * 90,
            "minutes_remaining_est": sum(s[2] for s in steps),
            "matrix_full": dist,
        }

    def to_json(self):
        return {"use_evidence": self.use_evidence, "mode": self.mode, "columns": self.columns,
                "beta": None if self.beta is None else [float(b) for b in self.beta],
                "se": None if self.beta is None else [float(s) for s in self.se], "half_end": list(self.half_end)}

    @classmethod
    def from_json(cls, d):
        m = cls(use_evidence=d["use_evidence"], mode=d["mode"])
        m.columns = d["columns"]
        m.beta = None if d["beta"] is None else np.array(d["beta"])
        m.se = None if d["se"] is None else np.array(d["se"])
        m.half_end = tuple(d["half_end"])
        return m


def per_minute_baseline(lambda_match, expected_minutes=94.0):
    return lambda_match / expected_minutes


def extrapolation_forecast(state, lambda_home, lambda_away, half_end=(47.0, 94.5)):
    elapsed = state["clock"] if state["period"] == 2 else min(state["clock"], 45)
    elapsed = max(elapsed, 1.0)
    remaining = max(half_end[1] - (state["clock"] if state["period"] == 2 else 45 + (half_end[0] - state["clock"])), 0.5)
    if state["period"] == 1:
        remaining = (half_end[0] - state["clock"]) + (half_end[1] - 45)
    rh = max(state["home_goals"] / elapsed, 0.2 * lambda_home / 94)
    ra = max(state["away_goals"] / elapsed, 0.2 * lambda_away / 94)
    from scipy.stats import poisson
    g = np.arange(MAX_GOALS + 1)
    ph = poisson.pmf(g, rh * remaining)
    pa = poisson.pmf(g, ra * remaining)
    m = np.zeros((MAX_GOALS + 1, MAX_GOALS + 1))
    for i in range(MAX_GOALS + 1):
        for j in range(MAX_GOALS + 1):
            hi, ai = state["home_goals"] + i, state["away_goals"] + j
            if hi <= MAX_GOALS and ai <= MAX_GOALS:
                m[hi, ai] += ph[i] * pa[j]
    m /= m.sum()
    mk = markets(m)
    mk["p_another_goal"] = 1 - ph[0] * pa[0]
    return mk
