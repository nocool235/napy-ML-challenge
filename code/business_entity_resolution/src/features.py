"""
Pairwise features for (S1 record, S2/S3 candidate).

Three families:
  1. similarity   - fuzzy string scores (rapidfuzz, vectorised), TF-IDF cosines per view,
                    token Jaccard, acronym / DBA handling
  2. structure    - postcode / house number / name-number agreement & conflict,
                    legal-form conflict, lengths, landmark flags, name frequency (chains)
  3. context      - how this candidate ranks among all candidates of the same S1 record,
                    and how this S1 ranks among all S1 records competing for the candidate.
                    (S1 is deduplicated => a candidate belongs to at most one S1.)
No feature uses the country label except `same_country`, so the model transfers to France.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

from config import VIEWS


def _cp(a, b, scorer, **kw):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32, **kw)


def _row_cos(XA, XB, ia, ib, chunk=200_000):
    out = np.empty(len(ia), np.float32)
    for s in range(0, len(ia), chunk):
        a = XA[ia[s:s + chunk]]
        b = XB[ib[s:s + chunk]]
        out[s:s + chunk] = np.asarray(a.multiply(b).sum(axis=1)).ravel()
    return out


def _jaccard(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(len(a), np.float32)
    for k, (x, y) in enumerate(zip(a, b)):
        sx, sy = set(x.split()), set(y.split())
        if sx or sy:
            out[k] = len(sx & sy) / len(sx | sy)
    return out


def _set_agree(a: np.ndarray, b: np.ndarray):
    """For space-separated token sets: (both_present, any_overlap, conflict)."""
    both = np.zeros(len(a), np.float32); match = np.zeros(len(a), np.float32); conf = np.zeros(len(a), np.float32)
    for k, (x, y) in enumerate(zip(a, b)):
        if x and y:
            both[k] = 1
            if set(x.split()) & set(y.split()):
                match[k] = 1
            else:
                conf[k] = 1
    return both, match, conf


def _best_alt(a: np.ndarray, b: np.ndarray, base: np.ndarray) -> np.ndarray:
    """max token_set_ratio over alternative names (DBA / parenthesised) - loop only where needed."""
    out = base.copy()
    need = np.array(["|" in x or "|" in y for x, y in zip(a, b)])
    for k in np.where(need)[0]:
        out[k] = max(fuzz.token_set_ratio(p, q) for p in a[k].split("|") for q in b[k].split("|"))
    return out


def pair_features(pairs: pd.DataFrame, s1: pd.DataFrame, cand: pd.DataFrame, X1: dict, XC: dict,
                  name_freq: dict) -> pd.DataFrame:
    t0 = time.time()
    i, j = pairs["i"].values, pairs["j"].values
    F = pairs.copy()

    def col(df, c, idx):
        return df[c].to_numpy(dtype=object)[idx]

    # ---- TF-IDF cosines for every view (not just the one that found the pair)
    for view in VIEWS:
        F[f"cos_{view}"] = _row_cos(X1[view], XC[view], i, j)

    # ---- names
    a, b = col(s1, "name_core", i), col(cand, "name_core", j)
    F["n_ratio"] = _cp(a, b, fuzz.ratio)
    F["n_partial"] = _cp(a, b, fuzz.partial_ratio)
    F["n_tsort"] = _cp(a, b, fuzz.token_sort_ratio)
    F["n_tset"] = _cp(a, b, fuzz.token_set_ratio)
    F["n_wratio"] = _cp(a, b, fuzz.WRatio)
    F["n_jw"] = _cp(a, b, JaroWinkler.normalized_similarity)
    F["n_lev"] = _cp(a, b, Levenshtein.distance)
    F["n_jacc"] = _jaccard(a, b)
    F["n_first_eq"] = np.array([x.split()[:1] == y.split()[:1] and bool(x) for x, y in zip(a, b)], np.float32)
    F["n_len1"] = np.fromiter(map(len, a), np.float32, len(a))
    F["n_len2"] = np.fromiter(map(len, b), np.float32, len(b))
    F["n_len_diff"] = np.abs(F["n_len1"] - F["n_len2"])
    F["n_empty_any"] = ((F["n_len1"] == 0) | (F["n_len2"] == 0)).astype(np.float32)
    ns_a, ns_b = col(s1, "name_nospace", i), col(cand, "name_nospace", j)
    F["n_nospace_ratio"] = _cp(ns_a, ns_b, fuzz.ratio)
    F["n_clean_tset"] = _cp(col(s1, "name_clean", i), col(cand, "name_clean", j), fuzz.token_set_ratio)
    ini_a, ini_b = col(s1, "name_initials", i), col(cand, "name_initials", j)
    F["n_acronym"] = np.array([(len(x) >= 2 and x == q) or (len(y) >= 2 and y == p)
                               for x, y, p, q in zip(ini_a, ini_b, ns_a, ns_b)], np.float32)
    F["n_alt_best"] = _best_alt(col(s1, "name_alts", i), col(cand, "name_alts", j), F["n_tset"].values)
    both, match, conf = _set_agree(col(s1, "name_nums", i), col(cand, "name_nums", j))
    F["n_nums_match"], F["n_nums_conflict"] = match, conf
    la, lb = col(s1, "name_legal", i), col(cand, "name_legal", j)
    both, match, conf = _set_agree(la, lb)
    F["n_legal_both"], F["n_legal_conflict"] = both, conf
    f1 = np.array([name_freq.get(x, 1) for x in a], np.float32)
    f2 = np.array([name_freq.get(x, 1) for x in b], np.float32)
    F["n_freq1"], F["n_freq2"] = np.log1p(f1), np.log1p(f2)

    # ---- addresses
    a, b = col(s1, "addr_clean", i), col(cand, "addr_clean", j)
    F["a_ratio"] = _cp(a, b, fuzz.ratio)
    F["a_partial"] = _cp(a, b, fuzz.partial_ratio)
    F["a_tsort"] = _cp(a, b, fuzz.token_sort_ratio)
    F["a_tset"] = _cp(a, b, fuzz.token_set_ratio)
    F["a_jacc"] = _jaccard(a, b)
    F["a_nolm_tset"] = _cp(col(s1, "addr_nolm", i), col(cand, "addr_nolm", j), fuzz.token_set_ratio)
    F["a_words_tset"] = _cp(col(s1, "addr_words", i), col(cand, "addr_words", j), fuzz.token_set_ratio)
    F["a_ntok1"] = s1["addr_ntok"].values[i].astype(np.float32)
    F["a_ntok2"] = cand["addr_ntok"].values[j].astype(np.float32)
    F["a_empty_any"] = ((F["a_ntok1"] == 0) | (F["a_ntok2"] == 0)).astype(np.float32)
    F["a_lm1"] = s1["addr_has_landmark"].values[i].astype(np.float32)
    F["a_lm2"] = cand["addr_has_landmark"].values[j].astype(np.float32)
    both, match, conf = _set_agree(col(s1, "addr_postcode", i), col(cand, "addr_postcode", j))
    F["pc_both"], F["pc_match"], F["pc_conflict"] = both, match, conf
    both, match, conf = _set_agree(col(s1, "addr_house", i), col(cand, "addr_house", j))
    F["hn_both"], F["hn_match"], F["hn_conflict"] = both, match, conf
    F["a_nums_jacc"] = _jaccard(col(s1, "addr_nums", i), col(cand, "addr_nums", j))

    # ---- misc
    F["same_country"] = (s1["country_norm"].to_numpy(dtype=object)[i] == cand["country_norm"].to_numpy(dtype=object)[j]).astype(np.float32)
    F["is_s3"] = (cand["source"].to_numpy(dtype=object)[j] == "S3").astype(np.float32)

    F = add_context(F, ["cos_name_addr", "cos_name", "cos_addr", "n_tset", "a_tset"])
    print(f"[features] {F.shape[1]} cols x {len(F):,} pairs in {time.time() - t0:.1f}s")
    return F


def add_context(F: pd.DataFrame, cols: list[str], prefix: str = "") -> pd.DataFrame:
    """Rank / gap-to-best of each score within the S1 group (i) and within the candidate group (j)."""
    for c in cols:
        for g, tag in (("i", "s1"), ("j", "c")):
            grp = F.groupby(g)[c]
            F[f"{prefix}{c}_rank_{tag}"] = grp.rank(ascending=False, method="min").astype(np.float32)
            F[f"{prefix}{c}_gap_{tag}"] = (grp.transform("max") - F[c]).astype(np.float32)
    F[f"{prefix}n_cand_s1"] = F.groupby("i")["j"].transform("size").astype(np.float32)
    F[f"{prefix}n_cand_c"] = F.groupby("j")["i"].transform("size").astype(np.float32)
    return F


def stage2_features(F: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    """Context built on stage-1 probabilities (competition between candidates / between S1s)."""
    G = F.copy()
    G["p1"] = p.astype(np.float32)
    for g, tag in (("i", "s1"), ("j", "c")):
        grp = G.groupby(g)["p1"]
        G[f"p1_rank_{tag}"] = grp.rank(ascending=False, method="first").astype(np.float32)
        G[f"p1_max_{tag}"] = grp.transform("max").astype(np.float32)
        G[f"p1_gap_{tag}"] = G[f"p1_max_{tag}"] - G["p1"]
        G[f"p1_sum_{tag}"] = grp.transform("sum").astype(np.float32)
        G[f"p1_n05_{tag}"] = (G["p1"] > 0.5).groupby(G[g]).transform("sum").astype(np.float32)
        # second-best in group (how contested is the top spot)
        second = G.loc[G[f"p1_rank_{tag}"] == 2].set_index(g)["p1"]
        G[f"p1_2nd_{tag}"] = G[g].map(second).fillna(0).astype(np.float32)
    return G


NON_FEATURES = {"i", "j", "y"}


def feature_cols(F: pd.DataFrame) -> list[str]:
    return [c for c in F.columns if c not in NON_FEATURES]
