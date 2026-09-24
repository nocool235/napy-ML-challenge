"""
Candidate generation (blocking).

Several TF-IDF "views" of each record; for every view we take the top-K nearest
neighbours by cosine similarity in BOTH directions:
    forward : each S1 record  -> top-K records of S2 and top-K records of S3
    backward: each S2/S3 rec  -> top-K' records of S1
and union everything. Recall ceiling is what matters here - the matcher can only find
pairs that blocking produced. The ranks from blocking are kept as features.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from config import VIEWS, K_FWD, K_BWD


def fit_vectorizers(frames: list[pd.DataFrame]) -> dict:
    """Fit one TF-IDF vectoriser per view on all records (unsupervised, no labels used)."""
    vecs = {}
    for view, spec in VIEWS.items():
        corpus = pd.concat([f[spec["col"]] for f in frames]).fillna("").astype(str)
        v = TfidfVectorizer(dtype=np.float32, sublinear_tf=True, min_df=1, **spec["tfidf"])
        v.fit(corpus.values)
        vecs[view] = v
    return vecs


def transform(vecs: dict, df: pd.DataFrame) -> dict:
    return {view: vecs[view].transform(df[VIEWS[view]["col"]].fillna("").astype(str).values).tocsr()
            for view in vecs}


def topk_cosine(A: sparse.csr_matrix, B: sparse.csr_matrix, k: int, max_cells: int = 30_000_000):
    """For every row of A, the k most similar rows of B (rows are L2-normalised => dot = cosine).
    Returns flat arrays (a_idx, b_idx, sim, rank) with zero-similarity hits dropped."""
    nA, nB = A.shape[0], B.shape[0]
    if nA == 0 or nB == 0:
        return (np.empty(0, int),) * 2 + (np.empty(0, np.float32), np.empty(0, int))
    k = min(k, nB)
    chunk = max(16, max_cells // max(nB, 1))
    BT = B.T.tocsc()
    out_a, out_b, out_s, out_r = [], [], [], []
    for start in range(0, nA, chunk):
        S = (A[start:start + chunk] @ BT).toarray()
        idx = np.argpartition(-S, kth=k - 1, axis=1)[:, :k]
        sims = np.take_along_axis(S, idx, axis=1)
        order = np.argsort(-sims, axis=1)
        idx = np.take_along_axis(idx, order, axis=1)
        sims = np.take_along_axis(sims, order, axis=1)
        rows = np.repeat(np.arange(start, start + S.shape[0]), k)
        ranks = np.tile(np.arange(1, k + 1), S.shape[0])
        keep = sims.ravel() > 0
        out_a.append(rows[keep]); out_b.append(idx.ravel()[keep])
        out_s.append(sims.ravel()[keep]); out_r.append(ranks[keep])
    return (np.concatenate(out_a), np.concatenate(out_b), np.concatenate(out_s), np.concatenate(out_r))


def generate_candidates(s1: pd.DataFrame, cand: pd.DataFrame, X1: dict, XC: dict, verbose=True) -> pd.DataFrame:
    """s1 / cand are normalised frames (cand = S2 and S3 concatenated, reset index).
    Returns a pair frame: i (row in s1), j (row in cand) + blocking-rank features."""
    t0 = time.time()
    src = cand["source"].to_numpy(dtype=object)
    pieces = []
    for view in VIEWS:
        A, B = X1[view], XC[view]
        for s in ("S2", "S3"):
            cidx = np.where(src == s)[0]
            if len(cidx) == 0:
                continue
            # forward: S1 -> this source
            a, b, sim, r = topk_cosine(A, B[cidx], K_FWD[view])
            pieces.append(pd.DataFrame({"i": a, "j": cidx[b], f"rf_{view}": r}))
            # backward: this source -> S1
            b2, a2, sim2, r2 = topk_cosine(B[cidx], A, K_BWD[view])
            pieces.append(pd.DataFrame({"i": a2, "j": cidx[b2], f"rb_{view}": r2}))
    pairs = pd.concat(pieces, ignore_index=True)
    rank_cols = [c for c in pairs.columns if c.startswith(("rf_", "rb_"))]
    pairs = pairs.groupby(["i", "j"], as_index=False)[rank_cols].min()
    big = max(max(K_FWD.values()), max(K_BWD.values())) + 1
    pairs[rank_cols] = pairs[rank_cols].fillna(big).astype(np.float32)
    pairs["blk_best_rank"] = pairs[rank_cols].min(axis=1)
    pairs["blk_n_views"] = (pairs[rank_cols] < big).sum(axis=1).astype(np.float32)
    if verbose:
        print(f"[blocking] {len(pairs):,} pairs for {len(s1):,} S1 records "
              f"({len(pairs) / max(len(s1), 1):.1f}/S1) in {time.time() - t0:.1f}s")
    return pairs


def blocking_report(pairs: pd.DataFrame, s1: pd.DataFrame, cand: pd.DataFrame, truth: dict) -> dict:
    """Pair recall + oracle F0.5 (score if the matcher were perfect on these candidates)."""
    from metric import macro_f05
    s1_ids = s1["entity_id"].to_numpy(dtype=object)
    c_ids = cand["entity_id"].to_numpy(dtype=object)
    got = {}
    for i, j in zip(pairs["i"].values, pairs["j"].values):
        got.setdefault(s1_ids[i], set()).add(c_ids[j])
    n_true = sum(len(v) for v in truth.values())
    hit = sum(len(truth.get(k, set()) & got.get(k, set())) for k in truth)
    oracle = {k: truth.get(k, set()) & got.get(k, set()) for k in s1_ids}
    rep = {
        "pairs": len(pairs),
        "pairs_per_s1": len(pairs) / max(len(s1), 1),
        "pair_recall": hit / max(n_true, 1),
        "oracle_f05": macro_f05(oracle, truth, s1_ids),
        "reduction_ratio": 1 - len(pairs) / max(len(s1) * len(cand), 1),
    }
    # recall per view (which views are pulling their weight)
    big = max(max(K_FWD.values()), max(K_BWD.values())) + 1
    for c in [c for c in pairs.columns if c.startswith(("rf_", "rb_"))]:
        sub = pairs[pairs[c] < big]
        cnt = sum(1 for i, j in zip(sub["i"].values, sub["j"].values) if c_ids[j] in truth.get(s1_ids[i], ()))
        rep[f"recall_{c}"] = cnt / max(n_true, 1)
    return rep
