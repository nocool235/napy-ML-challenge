"""Competition metric: per-Source-1 F0.5, macro-averaged, singletons included."""
from __future__ import annotations

import numpy as np


def f05(pred: set, true: set) -> float:
    if not true and not pred:
        return 1.0
    if not true or not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return 1.25 * p * r / (0.25 * p + r)


def macro_f05(pred: dict, truth: dict, s1_ids) -> float:
    return float(np.mean([f05(pred.get(k, set()), truth.get(k, set())) for k in s1_ids]))


def breakdown(pred: dict, truth: dict, s1_ids, groups: dict | None = None) -> dict:
    """Score split into singletons / matched entities and (optionally) by a grouping such as country."""
    rows = [(k, f05(pred.get(k, set()), truth.get(k, set())), len(truth.get(k, set())) == 0) for k in s1_ids]
    out = {
        "f05": float(np.mean([r[1] for r in rows])),
        "f05_singletons": float(np.mean([r[1] for r in rows if r[2]] or [np.nan])),
        "f05_matched": float(np.mean([r[1] for r in rows if not r[2]] or [np.nan])),
        "share_singletons": float(np.mean([r[2] for r in rows])),
    }
    if groups:
        for g in sorted(set(groups.values())):
            vals = [r[1] for r in rows if groups.get(r[0]) == g]
            out[f"f05_{g}"] = float(np.mean(vals)) if vals else float("nan")
    return out
