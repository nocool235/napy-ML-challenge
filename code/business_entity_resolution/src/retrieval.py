"""Stream-1 retrieval: exact-key + ANN candidate generation.

This module is intentionally separate from the existing end-to-end blocking.py
pipeline.  It produces the Stream-1 candidate-pair contract.
"""

from __future__ import annotations

from collections import defaultdict

import faiss
import numpy as np
import pandas as pd
import polars as pl
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


# ---------------------------------------------------------------- full-scale hash-join candidates (polars)
# Key types, bit i of `kbits` set when key type i produced the pair. Every key is scoped by country_norm.
# Bump when candidate generation changes what it returns (keys, caps, TF-IDF view settings). It names the
# pair cache in match.py, so a bump forces retrieval to rerun; refactors that keep the output don't need one.
RETRIEVAL_VERSION = "r1"
HASH_KEY_TYPES = ("name", "street", "hskel", "skel")
HASH_CAP_S1 = 5     # key groups with more S1 records than this are dropped (chains / generic names)
HASH_CAP_C = 50     # ... or with more S2/S3 records than this


def rare_addr_words(s1: pl.DataFrame, cand: pl.DataFrame, k: int = 2, chunk: int = 2_000_000):
    """Each record's k rarest address words (document frequency over S1 + S2/S3 of the same country).
    s1 carries row index `i`, cand carries `j`. Two chunked passes keep peak memory low."""
    def words(df, idx, lo):
        return (df.slice(lo, chunk).select(idx, "country_norm", w=pl.col("addr_words").str.split(" "))
                  .explode("w").filter(pl.col("w").str.len_chars() > 1).unique([idx, "w"]))
    counts = None
    for df, idx in ((s1, "i"), (cand, "j")):
        for lo in range(0, len(df), chunk):
            c = words(df, idx, lo).group_by("country_norm", "w").len()
            counts = c if counts is None else pl.concat([counts, c]).group_by("country_norm", "w").agg(pl.col("len").sum())
    out = []
    for df, idx in ((s1, "i"), (cand, "j")):
        parts = [words(df, idx, lo).join(counts, on=["country_norm", "w"]).sort([idx, "len", "w"])
                   .group_by(idx, maintain_order=True).head(k).select(idx, "w") for lo in range(0, len(df), chunk)]
        out.append(pl.concat(parts))
    return out


def _hash_keys(df: pl.DataFrame, idx: str, rare: pl.DataFrame) -> pl.DataFrame:
    c = pl.col("country_norm")
    parts = [
        df.filter(pl.col("name_core").str.len_chars() >= 3).select(
            idx, key=pl.concat_str([c, pl.lit("|n|"), pl.col("name_core")]), kt=pl.lit(0, pl.UInt8)),
        df.filter(pl.col("addr_street") != "").select(
            idx, key=pl.concat_str([c, pl.lit("|s|"), pl.col("addr_street")]), kt=pl.lit(1, pl.UInt8)),
        df.filter((pl.col("addr_house") != "") & (pl.col("name_skel") != "")).select(
            idx, key=pl.concat_str([c, pl.lit("|h|"), pl.col("name_skel").str.split(" ").list.first(), pl.lit("|"),
                                    pl.col("addr_house")]), kt=pl.lit(2, pl.UInt8)),
        rare.join(df.select(idx, "country_norm", "name_skel"), on=idx).filter(pl.col("name_skel") != "").select(
            idx, key=pl.concat_str([c, pl.lit("|k|"), pl.col("name_skel"), pl.lit("|"), pl.col("w")]),
            kt=pl.lit(3, pl.UInt8)),
    ]
    return pl.concat(parts).with_columns(pl.col("key").hash()).unique()


def tfidf_candidates(s1: pl.DataFrame, cand: pl.DataFrame, col: str = "name_addr", k: int = 5,
                     max_df: float = 0.01, threads: int = 0, chunk: int = 250_000,
                     analyzer: str = "word", ngram_range: tuple[int, int] = (1, 1),
                     extra: pl.DataFrame | None = None, log=print) -> pl.DataFrame:
    """Exact sparse top-k retrieval, S2/S3 -> S1, inside each country.

    TF-IDF on `col` (`analyzer`/`ngram_range`: "word" for token overlap, "char_wb" for typo/spelling-drift
    tolerant n-grams); tokens in more than `max_df` of the country's records are dropped (they cost the
    most in the sparse product and discriminate least) and tokens seen once are dropped (they can't match).
    For each S2/S3 record, its k most similar S1 records (sparse_dot_topn, multi-threaded).
    `extra` (i, j) pairs - e.g. the hash-join candidates, or another view's output - are scored with the
    same cosine, so calling this again with `extra` set to a prior call's output unions the two views and
    gives every pair both views' scores. Returns (i, j, tf_cos, tf_rank); tf_rank is the S1's rank for that
    candidate, k + 1 when the pair only came from `extra`."""
    import os
    import time

    from sparse_dot_topn import sp_matmul_topn

    kw = {"token_pattern": r"[^ |]+"} if analyzer == "word" else {}
    threads = threads or os.cpu_count() or 1
    ctry1, ctryc = s1["country_norm"].to_numpy(), cand["country_norm"].to_numpy()
    local_i = np.full(len(s1), -1, np.int64)
    local_j = np.full(len(cand), -1, np.int64)
    parts = []
    for c in sorted(set(ctry1) & set(ctryc)):
        t0 = time.time()
        gi, gj = np.flatnonzero(ctry1 == c), np.flatnonzero(ctryc == c)
        local_i[gi], local_j[gj] = np.arange(len(gi)), np.arange(len(gj))
        vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, sublinear_tf=True, dtype=np.float32,
                              min_df=2, max_df=max_df, **kw)
        X = vec.fit_transform(s1[col].gather(gi).to_list() + cand[col].gather(gj).to_list()).tocsr()
        vec.stop_words_ = None          # sklearn keeps every pruned token; not needed, can be millions
        X1, XC = X[:len(gi)], X[len(gi):]
        B = X1.T.tocsr()
        rows, cols, vals, ranks = [], [], [], []
        for s in range(0, len(gj), chunk):
            C = sp_matmul_topn(XC[s:s + chunk], B, top_n=k, threshold=0.0, sort=True, n_threads=threads).tocsr()
            r = np.repeat(np.arange(C.shape[0]), np.diff(C.indptr))
            rows.append(gj[s + r]); cols.append(gi[C.indices]); vals.append(C.data.astype(np.float32))
            ranks.append((np.arange(C.nnz) - C.indptr[r] + 1).astype(np.uint8))
        top = pl.DataFrame({"i": np.concatenate(cols).astype(np.uint32), "j": np.concatenate(rows).astype(np.uint32),
                            "tf_cos": np.concatenate(vals), "tf_rank": np.concatenate(ranks)})
        parts.append(top)
        if extra is not None:
            e = extra.filter(pl.Series(ctry1[extra["i"].to_numpy()] == c))
            li, lj = local_i[e["i"].to_numpy()], local_j[e["j"].to_numpy()]
            cos = np.empty(len(li), np.float32)
            for s in range(0, len(li), 2_000_000):
                cos[s:s + 2_000_000] = np.asarray(
                    X1[li[s:s + 2_000_000]].multiply(XC[lj[s:s + 2_000_000]]).sum(axis=1)).ravel()
            parts.append(e.select(pl.col("i").cast(pl.UInt32), pl.col("j").cast(pl.UInt32),
                                  tf_cos=pl.Series(cos), tf_rank=pl.lit(0, pl.UInt8)))
        log(f"tfidf {c}: S1={len(gi):,} cand={len(gj):,} vocab={X.shape[1]:,} top-{k} pairs={len(top):,} "
            f"in {time.time() - t0:.0f}s")
        del X, X1, XC, B
    out = (pl.concat(parts).group_by("i", "j")
             .agg(pl.col("tf_cos").max(), pl.col("tf_rank").max())
             .with_columns(tf_rank=pl.when(pl.col("tf_rank") == 0).then(k + 1).otherwise(pl.col("tf_rank"))
                                     .cast(pl.UInt8)))
    return out.sort("i", "j")


def tfidf_pair_cosine(s1: pl.DataFrame, cand: pl.DataFrame, pairs: pl.DataFrame, col: str,
                      analyzer: str = "word", ngram: tuple[int, int] = (1, 1), min_df: int = 2,
                      chunk: int = 2_000_000) -> np.ndarray:
    """TF-IDF cosine of `col` between the two sides of each given pair (i, j). The vectoriser is fit per
    country on that country's S1 + S2/S3 text (unsupervised, no labels), so a shared rare token counts for
    much more than a shared common one. Scores only the given pairs; no top-k search."""
    kw = {"token_pattern": r"[^ |]+"} if analyzer == "word" else {}
    ctry1, ctryc = s1["country_norm"].to_numpy(), cand["country_norm"].to_numpy()
    pi, pj = pairs["i"].to_numpy(), pairs["j"].to_numpy()
    pc = ctry1[pi]
    local_i = np.full(len(s1), -1, np.int64)
    local_j = np.full(len(cand), -1, np.int64)
    out = np.zeros(len(pi), np.float32)
    for c in np.unique(pc):
        gi, gj = np.flatnonzero(ctry1 == c), np.flatnonzero(ctryc == c)
        local_i[gi], local_j[gj] = np.arange(len(gi)), np.arange(len(gj))
        vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, sublinear_tf=True, dtype=np.float32,
                              min_df=min_df, **kw)
        X = vec.fit_transform(s1[col].gather(gi).to_list() + cand[col].gather(gj).to_list()).tocsr()
        vec.stop_words_ = None
        X1, XC = X[:len(gi)], X[len(gi):]
        rows = np.flatnonzero(pc == c)
        li, lj = local_i[pi[rows]], local_j[pj[rows]]
        for s in range(0, len(rows), chunk):
            out[rows[s:s + chunk]] = np.asarray(
                X1[li[s:s + chunk]].multiply(XC[lj[s:s + chunk]]).sum(axis=1)).ravel()
        del X, X1, XC
    return out


def hash_candidates(s1: pl.DataFrame, cand: pl.DataFrame, cap_s1: int = HASH_CAP_S1,
                    cap_c: int = HASH_CAP_C) -> pl.DataFrame:
    """Capped hash-join candidates -> (i, j, kbits). s1 needs row index `i`, cand needs `j`, plus the
    country_norm / name_core / name_skel / addr_words / addr_street / addr_house columns."""
    r1, rc = rare_addr_words(s1, cand)
    k1, kc = _hash_keys(s1, "i", r1), _hash_keys(cand, "j", rc)
    g1 = k1.group_by("key").len().filter(pl.col("len") <= cap_s1).select("key")
    gc = kc.group_by("key").len().filter(pl.col("len") <= cap_c).select("key")
    return (k1.join(g1, on="key").join(kc.join(gc, on="key").drop("kt"), on="key")
              .select("i", "j", bit=pl.col("kt").replace_strict({0: 1, 1: 2, 2: 4, 3: 8}, return_dtype=pl.UInt8))
              .unique()
              .group_by("i", "j").agg(kbits=pl.col("bit").sum().cast(pl.UInt8))
              .sort("i", "j"))


# Initial experimental value. The task requires a cap but does not prescribe
# a numeric value. Benchmark this parameter on labelled training data.
DEFAULT_EXACT_KEY_CAP = 500


EXACT_KEY_SPECS = (
    ("name_postcode", ("country_norm", "name_core", "addr_postcode")),
    ("skeleton_postcode", ("country_norm", "name_skel", "addr_postcode")),
    ("postcode_house", ("country_norm", "addr_postcode", "addr_house")),
)


def _valid_key(values) -> bool:
    """Return False for keys containing missing/empty components."""
    return all(str(v).strip() for v in values)


def _build_lookup(
    cand: pd.DataFrame,
    columns: tuple[str, ...],
) -> dict[tuple[str, ...], list[int]]:
    """Build key -> candidate-row lookup."""
    lookup: dict[tuple[str, ...], list[int]] = defaultdict(list)

    for j, row in enumerate(cand[list(columns)].itertuples(index=False, name=None)):
        key = tuple(str(x).strip() for x in row)
        if _valid_key(key):
            lookup[key].append(j)

    return lookup


def exact_key_candidates(
    s1: pd.DataFrame,
    cand: pd.DataFrame,
    cap: int = DEFAULT_EXACT_KEY_CAP,
) -> pd.DataFrame:
    """Generate candidate pairs using the three required exact-key joins.

    Oversized key groups are skipped when:
        len(S1 group) * len(candidate group) > cap

    This prevents common keys from producing a Cartesian-product explosion.

    Returns the Stream-1 candidate columns:
        pair_id, s1_id, cand_id, source, country,
        retrieval_score, exact_key_match, ann_cosine
    """
    if cap < 1:
        raise ValueError("cap must be >= 1")

    required = {
        "entity_id",
        "source",
        "country",
        "country_norm",
        "name_core",
        "name_skel",
        "addr_postcode",
        "addr_house",
    }

    missing_s1 = required - set(s1.columns)
    missing_cand = required - set(cand.columns)

    if missing_s1:
        raise ValueError(f"S1 missing required columns: {sorted(missing_s1)}")
    if missing_cand:
        raise ValueError(f"candidate missing required columns: {sorted(missing_cand)}")

    # Store one entry per (S1 row, candidate row), with the set of exact keys
    # that produced it.
    matched_by: dict[tuple[int, int], set[str]] = defaultdict(set)

    skipped_groups = 0
    generated_pairs = 0

    for key_name, columns in EXACT_KEY_SPECS:
        s1_groups: dict[tuple[str, ...], list[int]] = defaultdict(list)

        for i, row in enumerate(
            s1[list(columns)].itertuples(index=False, name=None)
        ):
            key = tuple(str(x).strip() for x in row)
            if _valid_key(key):
                s1_groups[key].append(i)

        cand_lookup = _build_lookup(cand, columns)

        for key, s1_rows in s1_groups.items():
            cand_rows = cand_lookup.get(key)
            if not cand_rows:
                continue

            cells = len(s1_rows) * len(cand_rows)

            if cells > cap:
                skipped_groups += 1
                continue

            generated_pairs += cells

            for i in s1_rows:
                for j in cand_rows:
                    matched_by[(i, j)].add(key_name)

    if not matched_by:
        out = pd.DataFrame(
            columns=[
                "pair_id",
                "s1_id",
                "cand_id",
                "source",
                "country",
                "retrieval_score",
                "exact_key_match",
                "ann_cosine",
            ]
        )
        out.attrs["exact_key_stats"] = {
            "cap": cap,
            "key_specs": [name for name, _ in EXACT_KEY_SPECS],
            "groups_skipped_cap": skipped_groups,
            "pairs_generated_before_dedup": generated_pairs,
            "unique_pairs": 0,
        }
        return out

    rows = []
    for (i, j), key_names in matched_by.items():
        s1_id = str(s1.iloc[i]["entity_id"])
        cand_id = str(cand.iloc[j]["entity_id"])
        source = str(cand.iloc[j]["source"])
        country = str(s1.iloc[i]["country"])

        rows.append(
            {
                "pair_id": f"{s1_id}__{source}__{cand_id}",
                "s1_id": s1_id,
                "cand_id": cand_id,
                "source": source,
                "country": country,
                "retrieval_score": 1.0,
                "exact_key_match": 1,
                "ann_cosine": 0.0,
                "_exact_keys": ",".join(sorted(key_names)),
            }
        )

    out = pd.DataFrame(rows)

    # Internal diagnostic column is deliberately removed from the output
    # contract.
    out = out.drop(columns="_exact_keys")

    out.attrs["exact_key_stats"] = {
        "cap": cap,
        "key_specs": [name for name, _ in EXACT_KEY_SPECS],
        "groups_skipped_cap": skipped_groups,
        "pairs_generated_before_dedup": generated_pairs,
        "unique_pairs": len(out),
    }

    return out


def ann_hnsw_search(
    database_vectors: np.ndarray,
    query_vectors: np.ndarray,
    k: int,
    m: int = 32,
    ef_search: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Search normalized vectors with a FAISS HNSW inner-product index.

    Returns:
        scores: cosine-similarity scores, shape (n_queries, k)
        indices: database row indices, shape (n_queries, k)
    """
    database_vectors = np.asarray(database_vectors, dtype=np.float32)
    query_vectors = np.asarray(query_vectors, dtype=np.float32)

    if database_vectors.ndim != 2 or query_vectors.ndim != 2:
        raise ValueError("database_vectors and query_vectors must be 2D arrays")

    if database_vectors.shape[1] != query_vectors.shape[1]:
        raise ValueError("database and query vectors must have the same dimension")

    if len(database_vectors) == 0 or len(query_vectors) == 0:
        return (
            np.empty((len(query_vectors), 0), dtype=np.float32),
            np.empty((len(query_vectors), 0), dtype=np.int64),
        )

    k = min(int(k), len(database_vectors))
    if k <= 0:
        raise ValueError("k must be positive")

    database_vectors = np.ascontiguousarray(database_vectors)
    query_vectors = np.ascontiguousarray(query_vectors)

    faiss.normalize_L2(database_vectors)
    faiss.normalize_L2(query_vectors)

    index = faiss.IndexHNSWFlat(
        database_vectors.shape[1],
        int(m),
        faiss.METRIC_INNER_PRODUCT,
    )
    index.hnsw.efSearch = max(int(ef_search), k)
    index.add(database_vectors)

    scores, indices = index.search(query_vectors, k)

    return (
        scores.astype(np.float32, copy=False),
        indices.astype(np.int64, copy=False),
    )


def bidirectional_ann_pairs(
    s1_vectors: np.ndarray,
    cand_vectors: np.ndarray,
    k_forward: int = 6,
    k_backward: int = 2,
    m: int = 32,
    ef_search: int = 64,
) -> pd.DataFrame:
    """Run HNSW ANN search in both directions and union the retrieved pairs.

    Forward:  S1 -> candidate
    Backward: candidate -> S1

    Returned indices are local row indices into the input arrays.
    """
    s1_vectors = np.asarray(s1_vectors, dtype=np.float32)
    cand_vectors = np.asarray(cand_vectors, dtype=np.float32)

    if s1_vectors.ndim != 2 or cand_vectors.ndim != 2:
        raise ValueError("s1_vectors and cand_vectors must be 2D arrays")

    if s1_vectors.shape[1] != cand_vectors.shape[1]:
        raise ValueError("S1 and candidate vectors must have the same dimension")

    if len(s1_vectors) == 0 or len(cand_vectors) == 0:
        return pd.DataFrame(
            columns=["s1_idx", "cand_idx", "source", "ann_cosine"]
        )

    forward_k = min(int(k_forward), len(cand_vectors))
    backward_k = min(int(k_backward), len(s1_vectors))

    forward_dist, forward_idx = ann_hnsw_search(
        cand_vectors,
        s1_vectors,
        k=forward_k,
        m=m,
        ef_search=ef_search,
    )

    backward_dist, backward_idx = ann_hnsw_search(
        s1_vectors,
        cand_vectors,
        k=backward_k,
        m=m,
        ef_search=ef_search,
    )

    records = {}

    for i in range(forward_idx.shape[0]):
        for j in range(forward_idx.shape[1]):
            cand_idx = int(forward_idx[i, j])
            if cand_idx < 0:
                continue

            score = float(forward_dist[i, j])
            key = (int(i), cand_idx)

            records[key] = {
                "s1_idx": int(i),
                "cand_idx": cand_idx,
                "source": "ann_forward",
                "ann_cosine": score,
            }

    for j in range(backward_idx.shape[0]):
        for i in range(backward_idx.shape[1]):
            s1_idx = int(backward_idx[j, i])
            if s1_idx < 0:
                continue

            score = float(backward_dist[j, i])
            key = (s1_idx, int(j))

            if key in records:
                records[key]["source"] = "ann_both"
                records[key]["ann_cosine"] = max(
                    float(records[key]["ann_cosine"]),
                    score,
                )
            else:
                records[key] = {
                    "s1_idx": s1_idx,
                    "cand_idx": int(j),
                    "source": "ann_backward",
                    "ann_cosine": score,
                }

    if not records:
        return pd.DataFrame(
            columns=["s1_idx", "cand_idx", "source", "ann_cosine"]
        )

    return pd.DataFrame.from_records(
        list(records.values()),
        columns=["s1_idx", "cand_idx", "source", "ann_cosine"],
    )


def countrywise_ann_pairs(
    s1: pd.DataFrame,
    cand: pd.DataFrame,
    text_col: str = "name_addr",
    k_forward: int = 6,
    k_backward: int = 2,
    n_components: int = 64,
) -> pd.DataFrame:
    """Run bidirectional ANN retrieval independently within each country.

    Supports both pandas and Polars input DataFrames.
    Returned indices refer to the original S1 and candidate row positions.
    """
    required = {"country_norm", text_col}
    missing_s1 = required - set(s1.columns)
    missing_cand = required - set(cand.columns)

    if missing_s1:
        raise ValueError(f"S1 missing required columns: {sorted(missing_s1)}")
    if missing_cand:
        raise ValueError(f"candidate missing required columns: {sorted(missing_cand)}")

    # Convert only the columns needed by this function to pandas.
    # This keeps the existing TF-IDF/FAISS implementation unchanged.
    if hasattr(s1, "to_pandas"):
        s1_work = s1.select(["country_norm", text_col]).to_pandas()
    else:
        s1_work = s1[["country_norm", text_col]].copy()

    if hasattr(cand, "to_pandas"):
        cand_work = cand.select(["country_norm", text_col]).to_pandas()
    else:
        cand_work = cand[["country_norm", text_col]].copy()

    s1_work = s1_work.reset_index(drop=True)
    cand_work = cand_work.reset_index(drop=True)

    s1_work["country_norm"] = s1_work["country_norm"].astype(str)
    cand_work["country_norm"] = cand_work["country_norm"].astype(str)

    countries = sorted(
        set(s1_work["country_norm"])
        & set(cand_work["country_norm"])
    )

    rows = []

    for country in countries:
        s1_rows = np.flatnonzero(
            s1_work["country_norm"].to_numpy() == country
        )
        cand_rows = np.flatnonzero(
            cand_work["country_norm"].to_numpy() == country
        )

        if len(s1_rows) == 0 or len(cand_rows) == 0:
            continue

        s1_text = s1_work.iloc[s1_rows][text_col].reset_index(drop=True)
        cand_text = cand_work.iloc[cand_rows][text_col].reset_index(drop=True)

        all_text = pd.concat([cand_text, s1_text], ignore_index=True)

        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            lowercase=False,
            dtype=np.float32,
        )
        tfidf = vectorizer.fit_transform(all_text)

        max_components = min(
            n_components,
            max(1, tfidf.shape[0] - 1),
            max(1, tfidf.shape[1] - 1),
        )

        svd = TruncatedSVD(
            n_components=max_components,
            random_state=42,
        )

        dense = svd.fit_transform(tfidf).astype(np.float32)
        faiss.normalize_L2(dense)

        cand_vectors = dense[: len(cand_rows)]
        s1_vectors = dense[len(cand_rows):]

        pairs = bidirectional_ann_pairs(
            s1_vectors,
            cand_vectors,
            k_forward=k_forward,
            k_backward=k_backward,
        )

        if pairs.empty:
            continue

        # Re-score the ANN-retrieved pairs using exact cosine in the
        # original per-country TF-IDF space. The ANN/SVD score is used
        # only for retrieval; ann_cosine stores the exact re-score.
        s1_local_idx = pairs["s1_idx"].to_numpy(dtype=np.int64)
        cand_local_idx = pairs["cand_idx"].to_numpy(dtype=np.int64)

        s1_matrix = tfidf[len(cand_rows):]
        cand_matrix = tfidf[:len(cand_rows)]

        exact_scores = np.asarray(
            s1_matrix[s1_local_idx].multiply(
                cand_matrix[cand_local_idx]
            ).sum(axis=1)
        ).ravel().astype(np.float32)

        pairs["ann_cosine"] = exact_scores

        pairs["s1_idx"] = pairs["s1_idx"].map(
            lambda x: int(s1_rows[int(x)])
        )
        pairs["cand_idx"] = pairs["cand_idx"].map(
            lambda x: int(cand_rows[int(x)])
        )
        pairs["country_norm"] = country

        rows.append(pairs)

    if not rows:
        return pd.DataFrame(
            columns=[
                "s1_idx",
                "cand_idx",
                "source",
                "ann_cosine",
                "country_norm",
            ]
        )

    return pd.concat(rows, ignore_index=True)


def ann_candidates(
    s1: pd.DataFrame,
    cand: pd.DataFrame,
    text_col: str = "name_addr",
    k_forward: int = 6,
    k_backward: int = 2,
    n_components: int = 64,
) -> pd.DataFrame:
    """Wrap countrywise_ann_pairs() into the shared Stream-1 candidate contract.

    countrywise_ann_pairs() returns local row positions and an ANN-direction
    label in `source` (ann_forward/ann_backward/ann_both); this adapts that
    into pair_id/s1_id/cand_id/source(S2 or S3)/country/retrieval_score/
    exact_key_match/ann_cosine, so its output is union-compatible with
    exact_key_candidates().
    """
    required = {"entity_id", "source", "country"}
    missing_s1 = required - set(s1.columns)
    missing_cand = required - set(cand.columns)

    if missing_s1:
        raise ValueError(f"S1 missing required columns: {sorted(missing_s1)}")
    if missing_cand:
        raise ValueError(f"candidate missing required columns: {sorted(missing_cand)}")

    contract_cols = [
        "pair_id", "s1_id", "cand_id", "source", "country",
        "retrieval_score", "exact_key_match", "ann_cosine",
    ]

    raw = countrywise_ann_pairs(
        s1, cand, text_col=text_col,
        k_forward=k_forward, k_backward=k_backward, n_components=n_components,
    )
    if raw.empty:
        return pd.DataFrame(columns=contract_cols)

    s1 = s1.reset_index(drop=True)
    cand = cand.reset_index(drop=True)
    s1_ids = s1["entity_id"].to_numpy(dtype=object)
    cand_ids = cand["entity_id"].to_numpy(dtype=object)
    cand_source = cand["source"].to_numpy(dtype=object)
    s1_country = s1["country"].to_numpy(dtype=object)

    s1_idx = raw["s1_idx"].to_numpy(dtype=np.int64)
    cand_idx = raw["cand_idx"].to_numpy(dtype=np.int64)

    out = pd.DataFrame({
        "s1_id": s1_ids[s1_idx],
        "cand_id": cand_ids[cand_idx],
        "source": cand_source[cand_idx],
        "country": s1_country[s1_idx],
        "retrieval_score": raw["ann_cosine"].to_numpy(dtype=np.float32),
        "exact_key_match": np.zeros(len(raw), dtype=np.int8),
        "ann_cosine": raw["ann_cosine"].to_numpy(dtype=np.float32),
    })
    out["pair_id"] = out["s1_id"] + "__" + out["source"] + "__" + out["cand_id"]
    return out[contract_cols]


def union_candidates(exact_df: pd.DataFrame, ann_df: pd.DataFrame) -> pd.DataFrame:
    """Union exact-key and ANN candidate pairs into one Stream-1 contract table.

    One row per (s1_id, cand_id): exact_key_match is 1 if either source found
    it via an exact key, retrieval_score/ann_cosine take the max across
    whichever source(s) retrieved the pair.
    """
    contract_cols = [
        "pair_id", "s1_id", "cand_id", "source", "country",
        "retrieval_score", "exact_key_match", "ann_cosine",
    ]
    agg_cols = ["s1_id", "cand_id", "source", "country",
                "retrieval_score", "exact_key_match", "ann_cosine"]

    both = pd.concat([exact_df[agg_cols], ann_df[agg_cols]], ignore_index=True)
    if both.empty:
        return pd.DataFrame(columns=contract_cols)

    agg = (
        both.groupby(["s1_id", "cand_id"], as_index=False)
        .agg(
            source=("source", "first"),
            country=("country", "first"),
            retrieval_score=("retrieval_score", "max"),
            exact_key_match=("exact_key_match", "max"),
            ann_cosine=("ann_cosine", "max"),
        )
    )
    agg["pair_id"] = agg["s1_id"] + "__" + agg["source"] + "__" + agg["cand_id"]
    return agg[contract_cols]


def smoke_test() -> str:
    """Confirm that the retrieval module imports correctly."""
    return "retrieval.py OK"


if __name__ == "__main__":
    print(smoke_test())
