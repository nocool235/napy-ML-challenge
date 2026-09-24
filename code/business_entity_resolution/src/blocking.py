"""
Candidate generation (blocking), partitioned.

Every record gets a small set of blocking keys (all prefixed with country_norm):
    pc   first chars of the postcode                    (rare in this data: ~2% India, ~12% US)
    n    the rarest name_core tokens                    (typo on a common word doesn't matter)
    a    the rarest non-numeric address words
    s    prefix of name_nospace                         (handles: "maithansons.com", "@gurubuilders")
    #    the rarest address numbers with >= 2 digits    (non-Latin names: only the flat/plot no. is shared)
Records with none of these fall back to country + first chars of name_core, so nothing is dropped.

Within each key group we run the TF-IDF top-K cosine search per view, in BOTH directions:
    forward : each S1 record  -> top-K records of S2 and top-K records of S3
    backward: each S2/S3 rec  -> top-K' records of S1
and union everything, keeping the best (min) rank a pair got anywhere. Cost is
sum_g |S1_g| x |cand_g| instead of |S1| x |cand|. Tiny groups are batched and scored as explicit
pairs (one matmul call per tiny group would be all Python overhead).
The output contract is unchanged: pairs with i, j, rf_*/rb_* ranks, blk_best_rank, blk_n_views.
"""
from __future__ import annotations

import time
from collections import Counter

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from config import VIEWS, K_FWD, K_BWD, BLOCK_KEYS, TINY_GROUP_CELLS, TINY_BATCH_CELLS


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


def _row_cos(XA, XB, ia, ib, chunk=500_000):
    out = np.empty(len(ia), np.float32)
    for s in range(0, len(ia), chunk):
        out[s:s + chunk] = np.asarray(XA[ia[s:s + chunk]].multiply(XB[ib[s:s + chunk]]).sum(axis=1)).ravel()
    return out


# ----------------------------------------------------------------------------- keys
def record_keys(frames: list[pd.DataFrame]):
    """Blocking keys for each frame (token rarity counted over all frames, no labels).
    Returns per frame: (keys: list[list[str]], kind: np.ndarray of 'pc'|'tok'|'name_fb'|'empty_fb')."""
    cfg = BLOCK_KEYS
    nfreq, wfreq, numfreq = Counter(), Counter(), Counter()
    for f in frames:
        for x in f["name_core"].values:
            nfreq.update(set(x.split()))
        for x in f["addr_words"].values:
            wfreq.update(set(x.split()))
        for x in f["addr_nums"].values:
            numfreq.update(t for t in set(x.split()) if len(t) >= 2)

    def rarest(toks, freq, n):
        return sorted(toks, key=lambda t: (freq[t], t))[:n] if n else []

    out = []
    for f in frames:
        keys, kind = [], []
        cols = zip(f["country_norm"].values, f["addr_postcode"].values, f["name_core"].values,
                   f["addr_words"].values, f["name_nospace"].values, f["addr_nums"].values)
        for c, pc, nm, aw, ns, num in cols:
            ks = []
            if pc and cfg["pc"]:
                ks.append(f"{c}|pc|{pc.split()[0][:cfg['pc']]}")
            has_pc = bool(ks)
            ks += [f"{c}|n|{t}" for t in rarest(set(nm.split()), nfreq, cfg["name_rare"])]
            ks += [f"{c}|a|{t}" for t in rarest(set(aw.split()), wfreq, cfg["addr_rare"])]
            if cfg["nospace"] and len(ns) >= 4:
                ks.append(f"{c}|s|{ns[:cfg['nospace']]}")
            ks += [f"{c}|#|{t}" for t in rarest({t for t in num.split() if len(t) >= 2}, numfreq, cfg["num_rare"])]
            if ks:
                kind.append("pc" if has_pc else "tok")
            else:
                ks = [f"{c}|fb|{nm[:cfg['fallback_name']]}"]
                kind.append("name_fb" if nm else "empty_fb")
            keys.append(ks)
        out.append((keys, np.array(kind, dtype=object)))
    return out


def _explode(keys: list[list[str]]):
    rows = np.repeat(np.arange(len(keys)), [len(k) for k in keys])
    return rows, [k for ks in keys for k in ks]


# ----------------------------------------------------------------------------- candidates
def generate_candidates(s1: pd.DataFrame, cand: pd.DataFrame, X1: dict, XC: dict, verbose=True) -> pd.DataFrame:
    """s1 / cand are normalised frames (cand = S2 and S3 concatenated, reset index).
    Returns a pair frame: i (row in s1), j (row in cand) + blocking-rank features."""
    t0 = time.time()
    src = cand["source"].to_numpy(dtype=object)
    (k1, kind1), (kc, _) = record_keys([s1, cand])
    r1, f1 = _explode(k1)
    rc, fc = _explode(kc)
    codes, _ = pd.factorize(np.array(f1 + fc, dtype=object))
    c1, cc = codes[:len(f1)], codes[len(f1):]
    n_keys = codes.max() + 1 if len(codes) else 0
    n1, nc = np.bincount(c1, minlength=n_keys), np.bincount(cc, minlength=n_keys)
    cells = n1.astype(np.int64) * nc
    tiny = (cells > 0) & (cells <= TINY_GROUP_CELLS)
    big = cells > TINY_GROUP_CELLS
    t_keys = time.time() - t0

    pieces = {}  # rank column -> list of (i, j, rank)

    def add(col, i, j, r):
        pieces.setdefault(col, []).append((i.astype(np.int64), j.astype(np.int64), r.astype(np.float32)))

    # ---- big groups: per-group top-K matmul
    o1, oc = np.argsort(c1, kind="stable"), np.argsort(cc, kind="stable")
    s1_sorted, sc_sorted = c1[o1], cc[oc]
    for g in np.where(big)[0]:
        ia = r1[o1[np.searchsorted(s1_sorted, g):np.searchsorted(s1_sorted, g, "right")]]
        ic = rc[oc[np.searchsorted(sc_sorted, g):np.searchsorted(sc_sorted, g, "right")]]
        for view in VIEWS:
            Ag = X1[view][ia]
            for s in ("S2", "S3"):
                ics = ic[src[ic] == s]
                if len(ics) == 0:
                    continue
                Bg = XC[view][ics]
                a, b, _, r = topk_cosine(Ag, Bg, K_FWD[view])
                add(f"rf_{view}", ia[a], ics[b], r)
                b2, a2, _, r2 = topk_cosine(Bg, Ag, K_BWD[view])
                add(f"rb_{view}", ia[a2], ics[b2], r2)
    t_big = time.time() - t0 - t_keys

    # ---- tiny groups: explicit pairs in batches, ranked per S1 (forward) and per candidate (backward)
    E1 = pd.DataFrame({"g": c1, "i": r1})
    E1 = E1[tiny[E1["g"].values]]
    EC = pd.DataFrame({"g": cc, "j": rc})
    EC = EC[tiny[EC["g"].values]]
    tiny_codes = np.where(tiny)[0]
    batch_id = np.zeros(n_keys, np.int64)
    batch_id[tiny_codes] = np.cumsum(cells[tiny_codes]) // TINY_BATCH_CELLS
    E1["b"], EC["b"] = batch_id[E1["g"].values], batch_id[EC["g"].values]
    n_tiny_pairs = 0
    for bid, e1 in E1.groupby("b", sort=False):
        P = e1.merge(EC[EC["b"].values == bid], on="g")[["i", "j"]].drop_duplicates()
        n_tiny_pairs += len(P)
        i, j = P["i"].to_numpy(), P["j"].to_numpy()
        is_s3 = src[j] == "S3"
        for view in VIEWS:
            cos = _row_cos(X1[view], XC[view], i, j)
            R = pd.DataFrame({"i": i, "j": j, "s": is_s3, "cos": cos})
            R = R[R["cos"] > 0]
            fwd = R.groupby(["i", "s"])["cos"].rank(ascending=False, method="first")
            m = fwd.values <= K_FWD[view]
            add(f"rf_{view}", R["i"].values[m], R["j"].values[m], fwd.values[m])
            bwd = R.groupby("j")["cos"].rank(ascending=False, method="first")
            m = bwd.values <= K_BWD[view]
            add(f"rb_{view}", R["i"].values[m], R["j"].values[m], bwd.values[m])
    t_tiny = time.time() - t0 - t_keys - t_big

    frames = [pd.DataFrame({"i": np.concatenate([p[0] for p in v]), "j": np.concatenate([p[1] for p in v]),
                            col: np.concatenate([p[2] for p in v])}) for col, v in pieces.items()]
    pairs = pd.concat(frames, ignore_index=True)
    rank_cols = [f"r{d}_{v}" for v in VIEWS for d in ("f", "b") if f"r{d}_{v}" in pairs.columns]
    pairs = pairs.groupby(["i", "j"], as_index=False)[rank_cols].min()
    n_union = len(pairs)
    big_rank = max(max(K_FWD.values()), max(K_BWD.values())) + 1

    # ---- global re-rank: group ranks are local to each key group and an S1 sits in ~7 groups, so K
    # would multiply. Re-rank every view across the whole union (per S1 per source / per candidate)
    # and keep a pair only if its global rank is within K somewhere.
    i, j = pairs["i"].to_numpy(), pairs["j"].to_numpy()
    R = pd.DataFrame({"i": i, "j": j, "s": src[j] == "S3"})
    keep = np.zeros(len(pairs), bool)
    for view in VIEWS:
        R["c"] = _row_cos(X1[view], XC[view], i, j)
        pos = R["c"].to_numpy() > 0
        rf = np.where(pos, R.groupby(["i", "s"])["c"].rank(ascending=False, method="first").to_numpy(), big_rank)
        rb = np.where(pos, R.groupby("j")["c"].rank(ascending=False, method="first").to_numpy(), big_rank)
        rf = np.where(rf <= K_FWD[view], rf, big_rank)
        rb = np.where(rb <= K_BWD[view], rb, big_rank)
        pairs[f"rf_{view}"], pairs[f"rb_{view}"] = rf.astype(np.float32), rb.astype(np.float32)
        keep |= (rf < big_rank) | (rb < big_rank)
    del R
    pairs = pairs[keep].reset_index(drop=True)
    rank_cols = [f"r{d}_{v}" for v in VIEWS for d in ("f", "b")]
    pairs["blk_best_rank"] = pairs[rank_cols].min(axis=1)
    pairs["blk_n_views"] = (pairs[rank_cols] < big_rank).sum(axis=1).astype(np.float32)

    no_cand = len(s1) - pairs["i"].nunique()
    stats = {
        "s1_postcode_key": int((kind1 == "pc").sum()),
        "s1_token_keys_only": int((kind1 == "tok").sum()),
        "s1_name_prefix_fallback": int((kind1 == "name_fb").sum()),
        "s1_empty_fallback": int((kind1 == "empty_fb").sum()),
        "s1_no_candidates": int(no_cand),
        "keys_per_s1": round(len(f1) / max(len(s1), 1), 2),
        "groups_big": int(big.sum()), "groups_tiny": int(tiny.sum()),
        "cells_frac_of_full": float(cells.sum() / max(len(s1) * len(cand), 1)),
        "tiny_pairs": int(n_tiny_pairs),
        "union_pairs_per_s1": round(n_union / max(len(s1), 1), 1),
        "sec_keys": round(t_keys, 1), "sec_big": round(t_big, 1), "sec_tiny": round(t_tiny, 1),
    }
    pairs.attrs["blk_stats"] = stats
    if verbose:
        print(f"[blocking] {len(pairs):,} pairs for {len(s1):,} S1 records "
              f"({len(pairs) / max(len(s1), 1):.1f}/S1) in {time.time() - t0:.1f}s")
        print("[blocking] partition " + str(stats))
    return pairs


def blocking_report(pairs: pd.DataFrame, s1: pd.DataFrame, cand: pd.DataFrame, truth: dict) -> dict:
    """Pair recall + oracle F0.5 (score if the matcher were perfect on these candidates),
    plus key_recall: share of true pairs that share at least one blocking key (the partition's ceiling)."""
    from metric import macro_f05
    s1_ids = s1["entity_id"].to_numpy(dtype=object)
    c_ids = cand["entity_id"].to_numpy(dtype=object)
    got = {}
    for i, j in zip(pairs["i"].values, pairs["j"].values):
        got.setdefault(s1_ids[i], set()).add(c_ids[j])
    n_true = sum(len(v) for v in truth.values())
    hit = sum(len(truth.get(k, set()) & got.get(k, set())) for k in truth)
    oracle = {k: truth.get(k, set()) & got.get(k, set()) for k in s1_ids}
    (k1, _), (kc, _) = record_keys([s1, cand])
    pos_c = dict(zip(c_ids, range(len(c_ids))))
    shared = sum(bool(set(k1[a]) & set(kc[pos_c[m]])) for a, k in enumerate(s1_ids)
                 for m in truth.get(k, ()) if m in pos_c)
    rep = {
        "pairs": len(pairs),
        "pairs_per_s1": len(pairs) / max(len(s1), 1),
        "key_recall": shared / max(n_true, 1),
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
