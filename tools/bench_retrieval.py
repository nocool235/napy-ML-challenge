#!/usr/bin/env python3
"""
Retrieval benchmark on a train sample of fraction --frac (S1 sampled, their true S2/S3 kept, other
S2/S3 sampled at the same rate): recall of the true S1 for S2/S3 -> S1 queries, inside each country.

    --method sparse : char n-gram TF-IDF, n-grams with df > max_df * n_docs dropped, rows re-normalised,
                      exact top-K with sparse_dot_topn
    --method hnsw   : TruncatedSVD of the same TF-IDF, faiss HNSW (inner product), top-K', then re-scored
                      with the exact sparse cosine and cut to K

Each method in its own process (faiss and sparse_dot_topn bundle clashing OpenMP runtimes on macOS).

    python tools/bench_retrieval.py --work-dir work --data-dir <dataset> --frac 0.05 --method sparse
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as l2norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution" / "src"))

import prep  # noqa: E402
from utils import log, read_tsv  # noqa: E402

KS = (1, 3, 10, 20, 30, 50)


def sample(norm: Path, data_dir: Path, frac: float, seed: int = 0, view: str = "name_addr"):
    cols = ["entity_id", "country_norm", view]
    s1 = prep.load(norm, "train", ("source1",), cols)
    cand = prep.load(norm, "train", ("source2", "source3"), cols)
    gt = (read_tsv(data_dir / "train" / "train_ground_truth.tsv")
            .with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
            .filter(pl.col("matched_entity_ids").str.len_chars() > 0)
            .select(s1_id="source1_entity_id", c_id="matched_entity_ids"))
    rng = np.random.default_rng(seed)
    s1 = s1.filter(pl.Series(rng.random(len(s1)) < frac)).with_row_index("i")
    gt = gt.join(s1.select("i", s1_id="entity_id"), on="s1_id")
    keep_c = pl.Series(rng.random(len(cand)) < frac) | cand["entity_id"].is_in(gt["c_id"].implode())
    cand = cand.filter(keep_c).with_row_index("j")
    gt = gt.join(cand.select("j", c_id="entity_id"), on="c_id").select("i", "j")
    return s1, cand, gt


def tfidf(s1, cand, view, max_df, analyzer="char"):
    if analyzer == "char":
        v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), sublinear_tf=True, dtype=np.float32, min_df=2)
    else:
        v = TfidfVectorizer(analyzer="word", token_pattern=r"[^ |]+", sublinear_tf=True, dtype=np.float32, min_df=1)
    X = v.fit_transform(pl.concat([s1[view], cand[view]]).to_list()).tocsr()
    if max_df:
        df = np.bincount(X.indices, minlength=X.shape[1])
        keep = np.where(df <= max_df * X.shape[0])[0]
        X = l2norm(X[:, keep], copy=False).tocsr()
    return X[:len(s1)], X[len(s1):]


def recall_table(found_rank: np.ndarray) -> dict:
    """found_rank: rank (1-based) of the true S1 for each true pair, 0 = not retrieved."""
    return {f"R@{k}": float(np.mean((found_rank > 0) & (found_rank <= k))) for k in KS}


def run_sparse(X1, XC, s1c, cc, K, threads):
    from sparse_dot_topn import sp_matmul_topn
    rows, cols, vals = [], [], []
    for c in np.unique(s1c):
        i_idx, j_idx = np.where(s1c == c)[0], np.where(cc == c)[0]
        B = X1[i_idx].T.tocsr()
        for s in range(0, len(j_idx), 50_000):
            jj = j_idx[s:s + 50_000]
            C = sp_matmul_topn(XC[jj], B, top_n=K, threshold=0.0, n_threads=threads).tocoo()
            rows.append(jj[C.row]); cols.append(i_idx[C.col]); vals.append(C.data)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)


def run_hnsw(X1, XC, s1c, cc, K, threads, dim, ef, M):
    import faiss
    from sklearn.decomposition import TruncatedSVD
    faiss.omp_set_num_threads(threads)
    t = time.time()
    fit_rows = np.random.default_rng(1).choice(X1.shape[0] + XC.shape[0], min(200_000, X1.shape[0] + XC.shape[0]), replace=False)
    Xall = sparse.vstack([X1, XC]).tocsr()
    svd = TruncatedSVD(n_components=dim, algorithm="randomized", n_iter=4, random_state=0).fit(Xall[fit_rows])
    V1 = l2norm(svd.transform(X1)).astype(np.float32)
    VC = l2norm(svd.transform(XC)).astype(np.float32)
    log(f"hnsw: svd dim={dim} fit+transform {time.time() - t:.0f}s")
    rows, cols, vals = [], [], []
    tb = tq = 0.0
    for c in np.unique(s1c):
        i_idx, j_idx = np.where(s1c == c)[0], np.where(cc == c)[0]
        t = time.time()
        index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 80
        index.add(V1[i_idx])
        tb += time.time() - t
        t = time.time()
        index.hnsw.efSearch = ef
        D, I = index.search(VC[j_idx], K)
        tq += time.time() - t
        ok = I >= 0
        rows.append(np.repeat(j_idx, K)[ok.ravel()]); cols.append(i_idx[I[ok]]); vals.append(D[ok])
    log(f"hnsw: build {tb:.0f}s, query {tq:.0f}s")
    j, i = np.concatenate(rows), np.concatenate(cols)
    # re-score with the exact sparse cosine
    exact = np.empty(len(j), np.float32)
    for s in range(0, len(j), 1_000_000):
        exact[s:s + 1_000_000] = np.asarray(XC[j[s:s + 1_000_000]].multiply(X1[i[s:s + 1_000_000]]).sum(axis=1)).ravel()
    return j, i, exact


def ranks_of_truth(j, i, score, gt: pl.DataFrame, n_c: int) -> np.ndarray:
    """rank of the true S1 among each query's retrieved S1s (by score), 0 if absent."""
    R = pl.DataFrame({"j": j, "i": i, "s": score}).with_columns(
        r=pl.col("s").rank(method="ordinal", descending=True).over("j"))
    m = gt.join(R, on=["i", "j"], how="left").with_columns(pl.col("r").fill_null(0))
    return m["r"].to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", default=str(ROOT / "work"))
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--frac", type=float, default=0.05)
    ap.add_argument("--method", choices=["sparse", "hnsw"], required=True)
    ap.add_argument("--view", default="name_addr")
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--max-df", type=float, default=0.0, help="sparse: drop n-grams with df > max_df * n_docs")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--ef", type=int, default=128)
    ap.add_argument("--M", type=int, default=32)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--analyzer", choices=["char", "word"], default="char")
    ap.add_argument("--save", default="", help="name: save retrieved pairs to work/bench/<name>.npz")
    ap.add_argument("--out", default=str(ROOT / "work" / "bench_retrieval.jsonl"))
    a = ap.parse_args()
    norm = prep.norm_dir(Path(a.work_dir))
    t0 = time.time()
    s1, cand, gt = sample(norm, Path(a.data_dir), a.frac, view=a.view)
    log(f"sample frac={a.frac}: S1={len(s1):,} cand={len(cand):,} true pairs={len(gt):,}")
    t = time.time()
    X1, XC = tfidf(s1, cand, a.view, a.max_df if a.method == "sparse" else 0.0, a.analyzer)
    t_vec = time.time() - t
    s1c, cc = s1["country_norm"].to_numpy(), cand["country_norm"].to_numpy()
    t = time.time()
    if a.method == "sparse":
        j, i, sc = run_sparse(X1, XC, s1c, cc, a.k, a.threads)
    else:
        j, i, sc = run_hnsw(X1, XC, s1c, cc, a.k, a.threads, a.dim, a.ef, a.M)
    t_ret = time.time() - t
    r = ranks_of_truth(j, i, sc, gt, len(cand))
    if a.save:
        (ROOT / "work" / "bench").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(ROOT / "work" / "bench" / f"{a.save}.npz", j=j.astype(np.int32), i=i.astype(np.int32),
                            s=sc.astype(np.float32), gi=gt["i"].to_numpy().astype(np.int32),
                            gj=gt["j"].to_numpy().astype(np.int32), ctry=s1["country_norm"].to_numpy()[gt["i"].to_numpy()])
    res = {"method": a.method, "analyzer": a.analyzer, "frac": a.frac, "view": a.view, "k": a.k, "max_df": a.max_df, "dim": a.dim, "ef": a.ef,
           "M": a.M, "n_s1": len(s1), "n_cand": len(cand), "sec_vectorise": round(t_vec, 1),
           "sec_retrieve": round(t_ret, 1), "sec_total": round(time.time() - t0, 1), **recall_table(r)}
    ctry = s1["country_norm"].to_numpy()[gt["i"].to_numpy()]
    for c in np.unique(ctry):
        res[f"R@10_{c}"] = recall_table(r[ctry == c])["R@10"]
        res[f"R@30_{c}"] = recall_table(r[ctry == c])["R@30"]
    log(json.dumps(res))
    with open(a.out, "a") as fh:
        fh.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
