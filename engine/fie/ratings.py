import numpy as np
from scipy.optimize import minimize

from .scoreline import dixon_coles_matrix, outcome_probs


class StrengthModel:
    def __init__(self, xi=0.0019, ridge=1.0, dixon_coles=True, window_days=1100, response="goals", xg_weight=0.0):
        self.xi = xi
        self.ridge = ridge
        self.dixon_coles = dixon_coles
        self.window_days = window_days
        self.response = response
        self.xg_weight = xg_weight
        self.params = None
        self.team_index = {}
        self.n_tiers = 1

    def _unpack(self, theta, n, t):
        a = theta[:n]
        d = theta[n:2 * n]
        mu = theta[2 * n:2 * n + t]
        h = theta[2 * n + t:2 * n + 2 * t]
        rho = theta[-1] if self.dixon_coles else 0.0
        return a, d, mu, h, rho

    def fit(self, frame, as_of):
        age = (as_of - frame["date"]).dt.days.to_numpy()
        keep = (age > 0) & (age <= self.window_days)
        f = frame.loc[keep]
        age = age[keep]
        teams = np.unique(np.concatenate([f["home"].to_numpy(), f["away"].to_numpy()]))
        old_index, old_params = self.team_index, self.params
        self.team_index = {t: i for i, t in enumerate(teams)}
        hi = np.array([self.team_index[t] for t in f["home"]])
        ai = np.array([self.team_index[t] for t in f["away"]])
        tier = f["tier"].to_numpy() - 1
        self.n_tiers = t = int(max(3, tier.max() + 1))
        n = len(teams)
        w = np.exp(-self.xi * age)
        hg = f["hg"].to_numpy(float)
        ag = f["ag"].to_numpy(float)
        yh, ya = hg, ag
        if self.xg_weight > 0:
            xh = f["xgp_h"].to_numpy(float)
            xa = f["xgp_a"].to_numpy(float)
            ok = ~(np.isnan(xh) | np.isnan(xa))
            yh = np.where(ok, (1 - self.xg_weight) * hg + self.xg_weight * np.nan_to_num(xh), hg)
            ya = np.where(ok, (1 - self.xg_weight) * ag + self.xg_weight * np.nan_to_num(xa), ag)
        c00 = (hg == 0) & (ag == 0)
        c01 = (hg == 0) & (ag == 1)
        c10 = (hg == 1) & (ag == 0)
        c11 = (hg == 1) & (ag == 1)
        ridge = self.ridge
        dc = self.dixon_coles

        def objective(theta):
            a, d, mu, h, rho = self._unpack(theta, n, t)
            eh = mu[tier] + h[tier] + a[hi] - d[ai]
            ea = mu[tier] + a[ai] - d[hi]
            lh = np.exp(eh)
            la = np.exp(ea)
            ll = w * (yh * eh - lh + ya * ea - la)
            g_eh = w * (yh - lh)
            g_ea = w * (ya - la)
            g_rho = 0.0
            if dc:
                t00 = 1 - lh * la * rho
                t01 = 1 + lh * rho
                t10 = 1 + la * rho
                t11 = 1 - rho
                tau = np.ones_like(lh)
                tau = np.where(c00, t00, tau)
                tau = np.where(c01, t01, tau)
                tau = np.where(c10, t10, tau)
                tau = np.where(c11, t11, tau)
                if (tau <= 0).any():
                    return 1e12, np.zeros_like(theta)
                ll = ll + w * np.log(tau)
                g_eh = g_eh + w * np.where(c00, -lh * la * rho / t00, 0) + w * np.where(c01, lh * rho / t01, 0)
                g_ea = g_ea + w * np.where(c00, -lh * la * rho / t00, 0) + w * np.where(c10, la * rho / t10, 0)
                g_rho = float((w * (np.where(c00, -lh * la / t00, 0) + np.where(c01, lh / t01, 0)
                                    + np.where(c10, la / t10, 0) + np.where(c11, -1 / t11, 0))).sum())
            nll = -ll.sum() + ridge * ((a ** 2).sum() + (d ** 2).sum())
            ga = np.bincount(hi, g_eh, n) + np.bincount(ai, g_ea, n)
            gd = -np.bincount(ai, g_eh, n) - np.bincount(hi, g_ea, n)
            gmu = np.bincount(tier, g_eh + g_ea, t)
            gh = np.bincount(tier, g_eh, t)
            grad = -np.concatenate([ga, gd, gmu, gh, [g_rho] if dc else []])
            grad[:n] += 2 * ridge * a
            grad[n:2 * n] += 2 * ridge * d
            return nll, grad

        theta0 = np.zeros(2 * n + 2 * t + (1 if dc else 0))
        theta0[2 * n:2 * n + t] = 0.25
        theta0[2 * n + t:2 * n + 2 * t] = 0.25
        if old_params is not None:
            oa, od, omu, oh, orho = old_params
            for team, i in self.team_index.items():
                j = old_index.get(team)
                if j is not None:
                    theta0[i] = oa[j]
                    theta0[n + i] = od[j]
            theta0[2 * n:2 * n + t] = omu[:t]
            theta0[2 * n + t:2 * n + 2 * t] = oh[:t]
            if dc:
                theta0[-1] = orho
        bounds = [(None, None)] * (2 * n + 2 * t) + ([(-0.3, 0.3)] if dc else [])
        res = minimize(objective, theta0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 500})
        self.params = self._unpack(res.x, n, t)
        self.fit_info = {"n_matches": int(len(f)), "n_teams": n, "converged": bool(res.success), "nll": float(res.fun)}
        return self

    def rates(self, home, away, tier, new_team_tier_prior=None):
        a, d, mu, h, rho = self.params
        out = []
        for ht, at, tr in zip(home, away, tier):
            tr = int(tr) - 1
            i = self.team_index.get(ht)
            j = self.team_index.get(at)
            prior_a, prior_d = (new_team_tier_prior or {}).get(tr, (0.0, 0.0))
            ah, dh = (a[i], d[i]) if i is not None else (prior_a, prior_d)
            aa, da = (a[j], d[j]) if j is not None else (prior_a, prior_d)
            lh = np.exp(mu[tr] + h[tr] + ah - da)
            la = np.exp(mu[tr] + aa - dh)
            out.append((lh, la, i is not None and j is not None))
        return np.array([o[0] for o in out]), np.array([o[1] for o in out]), np.array([o[2] for o in out]), rho

    def tier_priors(self, frame):
        a, d, *_ = self.params
        priors = {}
        if not self.team_index:
            return priors
        for tr in range(self.n_tiers):
            teams = set(frame.loc[frame["tier"] == tr + 1, "home"]) & set(self.team_index)
            if teams:
                idx = [self.team_index[x] for x in teams]
                priors[tr] = (float(np.percentile(a[idx], 20)), float(np.percentile(d[idx], 20)))
        return priors

    def team_table(self, names=None):
        a, d, mu, h, rho = self.params
        rows = []
        for team, i in self.team_index.items():
            rows.append({"team_id": int(team), "name": (names or {}).get(team, str(team)), "attack": float(a[i]), "defence": float(d[i])})
        return sorted(rows, key=lambda r: -(r["attack"] + r["defence"]))


class Elo:
    def __init__(self, k=20.0, home=65.0, tier_gap=120.0, goal_scaling=True, season_regress=0.1):
        self.k = k
        self.home = home
        self.tier_gap = tier_gap
        self.goal_scaling = goal_scaling
        self.season_regress = season_regress
        self.ratings = {}
        self.team_season = {}

    def initial(self, tier):
        return 1500.0 - self.tier_gap * (tier - 1)

    def rating(self, team, tier):
        if team not in self.ratings:
            self.ratings[team] = self.initial(tier)
        return self.ratings[team]

    def season_start(self, team, season, tier):
        r = self.rating(team, tier)
        last = self.team_season.get(team)
        if last is not None and last != season:
            r = r + self.season_regress * (self.initial(tier) - r)
            self.ratings[team] = r
        self.team_season[team] = season

    def expected(self, rh, ra):
        return 1 / (1 + 10 ** (-(rh + self.home - ra) / 400))

    def update(self, home, away, hg, ag, tier, season):
        self.season_start(home, season, tier)
        self.season_start(away, season, tier)
        rh, ra = self.rating(home, tier), self.rating(away, tier)
        e = self.expected(rh, ra)
        s = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        margin = abs(hg - ag)
        mult = 1.0
        if self.goal_scaling and margin > 1:
            mult = 1.5 if margin == 2 else (11 + margin) / 8
        delta = self.k * mult * (s - e)
        self.ratings[home] = rh + delta
        self.ratings[away] = ra - delta
        return rh, ra


def ordered_logit_fit(diff, y, weights=None):
    diff = np.asarray(diff, float) / 400
    y = np.asarray(y)
    w = np.ones(len(y)) if weights is None else np.asarray(weights)
    order = np.lexsort((y, diff))
    diff, y, w = diff[order], y[order], w[order]

    def nll(theta):
        b, c1, gap = theta
        c2 = c1 + abs(gap)
        z = b * diff
        p_away = 1 / (1 + np.exp(-(c1 - z)))
        p_away_or_draw = 1 / (1 + np.exp(-(c2 - z)))
        p = np.column_stack([1 - p_away_or_draw, p_away_or_draw - p_away, p_away])
        return -(w * np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))).sum()

    res = minimize(nll, np.array([2.0, -1.0, 1.0]), method="L-BFGS-B", options={"ftol": 1e-13, "gtol": 1e-9, "maxiter": 2000})
    return res.x


def ordered_logit_predict(theta, diff):
    b, c1, gap = theta
    c2 = c1 + abs(gap)
    z = b * np.asarray(diff, float) / 400
    p_away = 1 / (1 + np.exp(-(c1 - z)))
    p_away_or_draw = 1 / (1 + np.exp(-(c2 - z)))
    return np.column_stack([1 - p_away_or_draw, p_away_or_draw - p_away, p_away])


def poisson_link_fit(diff, goals):
    x = np.asarray(diff, float) / 400
    y = np.asarray(goals, float)
    order = np.lexsort((y, x))
    x, y = x[order], y[order]

    def nll(theta):
        eta = theta[0] + theta[1] * x
        return -(y * eta - np.exp(eta)).sum(), -np.array([(y - np.exp(eta)).sum(), ((y - np.exp(eta)) * x).sum()])

    res = minimize(nll, np.array([0.3, 0.5]), jac=True, method="L-BFGS-B")
    return res.x


def matrix_probs(lh, la, rho):
    out = np.empty((len(lh), 3))
    for k in range(len(lh)):
        out[k] = outcome_probs(dixon_coles_matrix(lh[k], la[k], rho))
    return out


class PiRatings:
    def __init__(self, lam=0.035, gamma=0.7, b=10.0, c=3.0):
        self.lam, self.gamma, self.b, self.c = lam, gamma, b, c
        self.home = {}
        self.away = {}

    def _gd(self, r):
        return np.sign(r) * (self.b ** (abs(r) / self.c) - 1)

    def expected_gd(self, home, away):
        return self._gd(self.home.get(home, 0.0)) - self._gd(self.away.get(away, 0.0))

    def update(self, home, away, hg, ag):
        rh, ra = self.home.get(home, 0.0), self.away.get(away, 0.0)
        err = (hg - ag) - (self._gd(rh) - self._gd(ra))
        psi = self.c * np.log10(1 + abs(err)) * np.sign(err)
        dh, da = psi * self.lam, -psi * self.lam
        self.home[home] = rh + dh
        self.away[home] = self.away.get(home, 0.0) + dh * self.gamma
        self.away[away] = ra + da
        self.home[away] = self.home.get(away, 0.0) + da * self.gamma


class OnlinePoisson:
    def __init__(self, lr=0.04, lr_level=0.005, tier_gap=0.25):
        self.lr, self.lr_level, self.tier_gap = lr, lr_level, tier_gap
        self.att = {}
        self.dfn = {}
        self.mu = {}
        self.ha = {}

    def _init(self, team, tier):
        if team not in self.att:
            self.att[team] = -self.tier_gap * (tier - 1) / 2
            self.dfn[team] = -self.tier_gap * (tier - 1) / 2

    def rates(self, home, away, tier):
        self._init(home, tier)
        self._init(away, tier)
        mu = self.mu.get(tier, 0.25)
        ha = self.ha.get(tier, 0.25)
        return np.exp(mu + ha + self.att[home] - self.dfn[away]), np.exp(mu + self.att[away] - self.dfn[home])

    def update(self, home, away, tier, hg, ag):
        lh, la = self.rates(home, away, tier)
        eh, ea = hg - lh, ag - la
        self.att[home] += self.lr * eh
        self.dfn[away] -= self.lr * eh
        self.att[away] += self.lr * ea
        self.dfn[home] -= self.lr * ea
        self.mu[tier] = self.mu.get(tier, 0.25) + self.lr_level * (eh + ea) / 2
        self.ha[tier] = self.ha.get(tier, 0.25) + self.lr_level * (eh - ea) / 2
