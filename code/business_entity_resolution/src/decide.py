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
