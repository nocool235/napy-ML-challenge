"""
Turn pair probabilities into the per-S1 match lists that are scored.

Two rules, both tuned on out-of-fold predictions against the real metric
(per-entity F0.5, macro, singletons included):

  thr   : keep candidates with p >= t
  expF  : per S1, choose the top-k set that maximises expected F0.5.
          For a chosen set of size k with TP true positives and T true matches overall,
          F0.5 = 1.25*TP / (0.25*T + k)  (k>0);  predicting empty scores 1 iff T == 0.
          E[empty] = prod(1-p);  E[top-k] ~= 1.25*sum_topk(p) / (0.25*sum_all(p) + k).
          `alpha` sharpens/softens probabilities (p**alpha) as a cheap calibration knob.

Optional one-to-one constraint: S1 is deduplicated, so each S2/S3 record may be
assigned only to its highest-probability S1 record.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import THRESH_GRID
from metric import macro_f05


def one_to_one(i: np.ndarray, j: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Boolean mask: True where this S1 is the best (max p) S1 for candidate j."""
    best = pd.Series(p).groupby(j).transform("max").values
    return p >= best - 1e-12


def rule_threshold(i, j, p, t, o2o=True):
    keep = p >= t
    if o2o:
        keep &= one_to_one(i, j, p)
    return keep


def rule_expected_f(i, j, p, alpha=1.0, o2o=True):
    q = np.clip(p, 1e-6, 1 - 1e-6) ** alpha
    elig = one_to_one(i, j, p) if o2o else np.ones(len(p), bool)
    df = pd.DataFrame({"i": i, "q": q, "e": elig, "pos": np.arange(len(p))})
    df["qe"] = np.where(df["e"], df["q"], 0.0)
    df = df.sort_values(["i", "qe"], ascending=[True, False])
    g = df.groupby("i")
    df["k"] = g.cumcount() + 1
    df["cum"] = g["qe"].cumsum()
    tot = g["q"].transform("sum")                     # expected number of true matches
    df["ef"] = 1.25 * df["cum"] / (0.25 * tot + df["k"])
    df.loc[~df["e"], "ef"] = -1.0                        # ineligible rows can't be selected
    df["lq"] = np.log1p(-df["q"])
    e0 = np.exp(df.groupby("i")["lq"].transform("sum"))   # P(no true match among candidates)
    best_k = df.assign(ef_max=df.groupby("i")["ef"].transform("max"))
    kstar = best_k.loc[best_k["ef"] == best_k["ef_max"]].groupby("i")["k"].min()
    df["kstar"] = df["i"].map(kstar)
    df["take"] = (df["k"] <= df["kstar"]) & (df["ef"].groupby(df["i"]).transform("max") > e0) & df["e"]
    keep = np.zeros(len(p), bool)
    keep[df.loc[df["take"], "pos"].values] = True
    return keep


def to_pred(i, j, keep, s1_ids, c_ids) -> dict:
    pred = {}
    for a, b in zip(i[keep], j[keep]):
        pred.setdefault(s1_ids[a], set()).add(c_ids[b])
    return pred


def tune(i, j, p, truth, s1_ids, c_ids, verbose=True) -> tuple[dict, pd.DataFrame]:
    rows = []
    for o2o in (True, False):
        for t in THRESH_GRID:
            keep = rule_threshold(i, j, p, t, o2o)
            rows.append({"rule": "thr", "param": t, "o2o": o2o,
                         "f05": macro_f05(to_pred(i, j, keep, s1_ids, c_ids), truth, s1_ids)})
        for a in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0):
            keep = rule_expected_f(i, j, p, a, o2o)
            rows.append({"rule": "expF", "param": a, "o2o": o2o,
                         "f05": macro_f05(to_pred(i, j, keep, s1_ids, c_ids), truth, s1_ids)})
    res = pd.DataFrame(rows).sort_values("f05", ascending=False)
    best = res.iloc[0].to_dict()
    if verbose:
        print("[decide] top rules on OOF:\n" + res.head(8).to_string(index=False))
    return best, res


def apply(best: dict, i, j, p) -> np.ndarray:
    if best["rule"] == "thr":
        return rule_threshold(i, j, p, best["param"], best["o2o"])
    return rule_expected_f(i, j, p, best["param"], best["o2o"])


# ------------------------------------------------------------------ vectorised versions (full scale)
# Same rules as above, in numpy over millions of pairs: no per-S1 Python loops, no dicts of sets.
# i / j are integer row indices (S1 / candidate); scores come from per_s1_f05 with bincount.

def per_s1_f05(keep_i: np.ndarray, keep_y: np.ndarray, n_s1: int, n_true: np.ndarray) -> np.ndarray:
    """Exact per-S1 F0.5 = 1.25 TP / (0.25 T + P); an empty prediction scores 1 iff T == 0.
    n_true counts ALL true matches per S1, including ones retrieval never proposed."""
    P = np.bincount(keep_i, minlength=n_s1)
    TP = np.bincount(keep_i, weights=keep_y, minlength=n_s1)
    return np.where(P + n_true == 0, 1.0, 1.25 * TP / np.maximum(0.25 * n_true + P, 1e-9))


def o2o_mask(j: np.ndarray, p: np.ndarray) -> np.ndarray:
    """True for exactly one pair per candidate j: the one with the highest p (S1 is deduplicated)."""
    order = np.lexsort((-p, j))
    first = np.ones(len(order), bool)
    first[1:] = j[order][1:] != j[order][:-1]
    m = np.zeros(len(p), bool)
    m[order[first]] = True
    return m


def expf_mask(i: np.ndarray, p: np.ndarray, alpha: float = 1.0, elig: np.ndarray | None = None) -> np.ndarray:
    """Per S1: the top-k eligible pairs maximising E[F0.5] ~ 1.25 sum_topk(q) / (0.25 sum(q) + k), taken
    only if that beats E[empty] = prod(1 - q). q = p ** alpha."""
    q = np.clip(p, 1e-6, 1 - 1e-6) ** alpha
    elig = np.ones(len(p), bool) if elig is None else elig
    qe = np.where(elig, q, 0.0)
    order = np.lexsort((-qe, i))
    i_s, q_s, qe_s, e_s = i[order], q[order], qe[order], elig[order]
    starts = np.flatnonzero(np.r_[True, i_s[1:] != i_s[:-1]])
    grp = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(i_s)]))
    k = np.arange(len(i_s)) - starts[grp] + 1
    cs = np.cumsum(qe_s)
    cum = cs - (cs[starts] - qe_s[starts])[grp]
    tot = np.add.reduceat(q_s, starts)[grp]
    ef = np.where(e_s, 1.25 * cum / (0.25 * tot + k), -1.0)
    gmax = np.maximum.reduceat(ef, starts)[grp]
    kstar = np.minimum.reduceat(np.where(ef >= gmax - 1e-12, k, np.iinfo(np.int64).max), starts)[grp]
    e0 = np.exp(np.add.reduceat(np.log1p(-q_s), starts))[grp]
    m = np.zeros(len(p), bool)
    m[order[(k <= kstar) & (gmax > e0) & e_s]] = True
    return m


def rule_mask(rule: dict, i: np.ndarray, j: np.ndarray, p: np.ndarray) -> np.ndarray:
    elig = o2o_mask(j, p) if rule["o2o"] else np.ones(len(p), bool)
    if rule["rule"] == "thr":
        return elig & (p >= rule["param"])
    return expf_mask(i, p, rule["param"], elig)


def score_by_group(f: np.ndarray, s1_group: np.ndarray, weights: dict) -> dict:
    """Mean F0.5 per group (e.g. country) and the weighted mix over the groups present in `weights`."""
    out = {"all": float(f.mean())}
    for g in np.unique(s1_group):
        out[str(g)] = float(f[s1_group == g].mean())
    present = [g for g in weights if g in out]
    out["mix"] = sum(weights[g] * out[g] for g in present) / sum(weights[g] for g in present) if present else out["all"]
    return out


def tune_vec(i, j, p, y, n_true, s1_group, weights, verbose=True):
    """Grid over both rules x one-to-one on/off; picks the rule with the best weighted-mix F0.5."""
    n_s1 = len(n_true)
    elig = {True: o2o_mask(j, p), False: np.ones(len(p), bool)}
    rows = []

    def add(rule, param, o2o, keep):
        f = per_s1_f05(i[keep], y[keep], n_s1, n_true)
        rows.append({"rule": rule, "param": param, "o2o": o2o, **score_by_group(f, s1_group, weights)})

    for o2o in (True, False):
        for t in THRESH_GRID:
            add("thr", t, o2o, elig[o2o] & (p >= t))
        for a in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0):
            add("expF", a, o2o, expf_mask(i, p, a, elig[o2o]))
    res = pd.DataFrame(rows).sort_values("mix", ascending=False)
    if verbose:
        print("[decide] top rules:\n" + res.head(8).to_string(index=False), flush=True)
    return res.iloc[0].to_dict(), res


def tune_robust(table_groupcv: pd.DataFrame, table_loco: pd.DataFrame, france_weight: float = 0.15,
                loco_countries: tuple[str, ...] = ("india", "us"), verbose: bool = True):
    """Pick ONE rule, applied to every test record regardless of country (never a per-country branch -
    the rules explicitly forbid that). `table_groupcv` is a normal tune_vec() grid (in-distribution score
    for India/US, since those countries are seen in training). `table_loco` is a tune_vec() grid computed
    on leave-one-country-out OOF (train on one of `loco_countries`, score the other) - our only proxy for
    an unseen country like France. Its mean across `loco_countries` stands in for the France score a
    candidate rule would get. robust_score blends the two exactly as the real test country mix does:
    (1 - france_weight) x in-distribution mix + france_weight x LOCO-mean proxy for the unseen country.
    A rule tuned on group-CV alone can look great in-distribution while being a poor bet on a country the
    model has never seen (this is why stage 2 needed this check, not just its own group-CV score)."""
    loco = table_loco.copy()
    loco["loco_mean"] = loco[list(loco_countries)].mean(axis=1)
    m = table_groupcv.merge(loco[["rule", "param", "o2o", "loco_mean"]], on=["rule", "param", "o2o"], how="inner")
    if m.empty:
        raise ValueError("no (rule, param, o2o) rows in common between the two grids")
    m["robust_score"] = (1 - france_weight) * m["mix"] + france_weight * m["loco_mean"]
    m = m.sort_values("robust_score", ascending=False)
    if verbose:
        cols = ["rule", "param", "o2o", "mix", "loco_mean", "robust_score"]
        print("[decide] top rules by robust score:\n" + m[cols].head(8).to_string(index=False), flush=True)
        top_groupcv = table_groupcv.sort_values("mix", ascending=False).iloc[0]
        if (top_groupcv["rule"], top_groupcv["param"], top_groupcv["o2o"]) != tuple(m.iloc[0][["rule", "param", "o2o"]]):
            print(f"[decide] robust choice differs from the group-CV-only best "
                  f"({top_groupcv['rule']} {top_groupcv['param']} o2o={top_groupcv['o2o']}, "
                  f"mix={top_groupcv['mix']:.4f}) - chose the more France-robust rule instead", flush=True)
    return m.iloc[0].to_dict(), m
