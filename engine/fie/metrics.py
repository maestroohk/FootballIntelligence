import numpy as np

EPS = 1e-12


def outcome_index(hg, ag):
    hg, ag = np.asarray(hg), np.asarray(ag)
    return np.where(hg > ag, 0, np.where(hg == ag, 1, 2))


def one_hot(y, k=3):
    out = np.zeros((len(y), k))
    out[np.arange(len(y)), y] = 1
    return out


def log_loss_rows(p, y):
    return -np.log(np.clip(p[np.arange(len(y)), y], EPS, 1))


def brier_rows(p, y):
    return ((p - one_hot(y, p.shape[1])) ** 2).sum(axis=1)


def rps_rows(p, y):
    cp = np.cumsum(p, axis=1)[:, :-1]
    co = np.cumsum(one_hot(y, p.shape[1]), axis=1)[:, :-1]
    return ((cp - co) ** 2).sum(axis=1) / (p.shape[1] - 1)


def reliability(prob, hit, bins=10):
    prob, hit = np.asarray(prob), np.asarray(hit)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(prob, edges) - 1, 0, bins - 1)
    out = []
    for b in range(bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        out.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": int(mask.sum()),
                    "mean_p": float(prob[mask].mean()), "freq": float(hit[mask].mean())})
    return out


def ece(prob, hit, bins=10):
    rel = reliability(prob, hit, bins)
    n = sum(r["n"] for r in rel)
    return float(sum(r["n"] * abs(r["mean_p"] - r["freq"]) for r in rel) / max(n, 1))


def classwise_ece(p, y, bins=10):
    oh = one_hot(y, p.shape[1])
    return float(np.mean([ece(p[:, k], oh[:, k], bins) for k in range(p.shape[1])]))


def goal_rows(lh, la, hg, ag, score_p=None):
    out = {
        "goal_mae_rows": (np.abs(np.asarray(lh) - hg) + np.abs(np.asarray(la) - ag)) / 2,
        "total_mae_rows": np.abs(np.asarray(lh) + np.asarray(la) - (np.asarray(hg) + np.asarray(ag))),
    }
    if score_p is not None:
        out["score_nll_rows"] = -np.log(np.clip(score_p, EPS, 1))
    return out


def summarise(p, y, lh=None, la=None, hg=None, ag=None, score_p=None, over25=None):
    p = np.asarray(p, dtype=float)
    y = np.asarray(y)
    res = {
        "n": int(len(y)),
        "log_loss": float(log_loss_rows(p, y).mean()),
        "brier": float(brier_rows(p, y).mean()),
        "rps": float(rps_rows(p, y).mean()),
        "ece": classwise_ece(p, y),
        "accuracy": float((p.argmax(axis=1) == y).mean()),
        "mean_p_home": float(p[:, 0].mean()), "mean_p_draw": float(p[:, 1].mean()),
        "freq_home": float((y == 0).mean()), "freq_draw": float((y == 1).mean()),
    }
    if lh is not None and hg is not None:
        g = goal_rows(lh, la, hg, ag, score_p)
        res["goal_mae"] = float(g["goal_mae_rows"].mean())
        res["total_goals_mae"] = float(g["total_mae_rows"].mean())
        if score_p is not None:
            res["score_nll"] = float(g["score_nll_rows"].mean())
    if over25 is not None and hg is not None:
        hit = (np.asarray(hg) + np.asarray(ag)) > 2.5
        o = np.clip(np.asarray(over25), EPS, 1 - EPS)
        res["over25_log_loss"] = float(-(hit * np.log(o) + (1 - hit) * np.log(1 - o)).mean())
        res["over25_ece"] = ece(o, hit)
    return res


def reliability_all(p, y, bins=10):
    oh = one_hot(y, 3)
    return {name: reliability(p[:, k], oh[:, k], bins) for k, name in enumerate(["home", "draw", "away"])}


def paired_bootstrap(loss_a, loss_b, n_boot=2000, seed=7, block=None):
    rng = np.random.default_rng(seed)
    diff = np.asarray(loss_a) - np.asarray(loss_b)
    n = len(diff)
    if block is not None:
        groups = np.asarray(block)
        uniq = np.unique(groups)
        idx_by = {g: np.where(groups == g)[0] for g in uniq}
        sums = np.array([diff[idx_by[g]].sum() for g in uniq])
        counts = np.array([len(idx_by[g]) for g in uniq])
        boots = []
        for _ in range(n_boot):
            pick = rng.integers(0, len(uniq), len(uniq))
            boots.append(sums[pick].sum() / counts[pick].sum())
        boots = np.array(boots)
    else:
        boots = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return {
        "mean_diff": float(diff.mean()),
        "ci_low": float(np.percentile(boots, 2.5)),
        "ci_high": float(np.percentile(boots, 97.5)),
        "p_b_better": float((boots > 0).mean()),
        "n": int(n),
    }


def devig(odds_h, odds_d, odds_a, method="power"):
    inv = np.column_stack([1 / np.asarray(odds_h, float), 1 / np.asarray(odds_d, float), 1 / np.asarray(odds_a, float)])
    if method == "proportional":
        return inv / inv.sum(axis=1, keepdims=True)
    out = np.empty_like(inv)
    for r in range(len(inv)):
        lo, hi = 0.5, 3.0
        for _ in range(60):
            k = (lo + hi) / 2
            s = (inv[r] ** k).sum()
            if s > 1:
                lo = k
            else:
                hi = k
        out[r] = inv[r] ** k
        out[r] /= out[r].sum()
    return out
