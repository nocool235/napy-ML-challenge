#!/usr/bin/env python3
"""Union recall of saved retrieval runs (work/bench/<name>.npz, same sample): each run contributes its own
top-K per query (by its own score); a true pair counts as found if any run has it in its top-K."""
import itertools
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
runs = sys.argv[1:]
data = {r: np.load(ROOT / "work" / "bench" / f"{r}.npz", allow_pickle=True) for r in runs}
first = data[runs[0]]
gt = pl.DataFrame({"i": first["gi"], "j": first["gj"]})
ctry = first["ctry"]
rank = {}
for r, d in data.items():
    R = pl.DataFrame({"j": d["j"], "i": d["i"], "s": d["s"]}).with_columns(
        r=pl.col("s").rank(method="ordinal", descending=True).over("j"))
    rank[r] = gt.join(R, on=["i", "j"], how="left")["r"].fill_null(10 ** 9).to_numpy()
    n_pairs = len(d["j"])
    print(f"{r:<28} pairs/query={n_pairs / len(np.unique(d['j'])):.1f}")
for n in range(1, len(runs) + 1):
    for combo in itertools.combinations(runs, n):
        best = np.min([rank[r] for r in combo], axis=0)
        cells = " ".join(f"R@{k}={np.mean(best <= k):.4f}" for k in (1, 10, 20, 50))
        india = f"india R@20={np.mean(best[ctry == 'india'] <= 20):.4f}"
        print(f"{'+'.join(combo):<70} {cells}  {india}")
