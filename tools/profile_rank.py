#!/usr/bin/env python3
"""
Step-0 retrieval profile at FULL scale: for a stratified sample of matched S2/S3 records, the rank
of their true S1 among ALL train S1 records under the pipeline's TF-IDF cosine
(char_wb n-grams, sublinear tf, smooth idf, l2) -> recall@k, plus each query's best wrong S1
(real full-scale hard negatives).

Exact brute force, made affordable by only scoring the sampled queries. IDF uses document
frequencies over all train records (S1+S2+S3); the vectoriser is hashed (2^22 buckets) so the
vocabulary never has to be held in memory. S1 is shuffled into equal chunks, so summing competitor
counts over a random subset of chunks gives the rank at a smaller corpus size (1%, 5%, 20% ...).

    python tools/profile_rank.py --data-dir ../student_resource/dataset --out reports/rank_profile.md
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize as l2norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "business_entity_resolution" / "src"))

NF = 2 ** 22
VIEWS = {"name_addr": (3, 4), "name": (2, 4), "addr": (3, 4)}  # same analyzers as config.VIEWS
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def hv(view):
    return HashingVectorizer(analyzer="char_wb", ngram_range=VIEWS[view], n_features=NF,
                             alternate_sign=False, norm=None, dtype=np.float32)


# ----------------------------------------------------------------------------- workers
def norm_chunk(args):
    """normalise a chunk -> (name_addr, name_core) strings + per-view document frequencies."""
    from normalize import name_fields, addr_fields
    names, addrs, keep = args
    nc = [name_fields(n)["name_core"] for n in names]
    ac = [addr_fields(a)["addr_clean"] for a in addrs]
    na = [(x + " | " + y).strip() for x, y in zip(nc, ac)]
    df = {}
    for view, txt in (("name_addr", na), ("name", nc), ("addr", ac)):
        X = hv(view).transform(txt)
        idx, cnt = np.unique(X.indices, return_counts=True)
        df[view] = (idx.astype(np.int32), cnt.astype(np.int32))
    return (na, nc, ac) if keep else None, df


_G = {}


def _init(g):
    _G.update(g)


def vec(view, txt):
    X = hv(view).transform(txt).tocsr()
    X.data = 1 + np.log(X.data)
    X.data *= _G["idf"][view][X.indices]
    return l2norm(X, copy=False)


def rank_chunk(args):
    """competitor counts for every query against one S1 chunk (true S1 excluded)."""
    rows, texts, ctry = args
    out = {}
    pos_of_true = {r: k for k, r in enumerate(rows)}
    for view in VIEWS:
        X = vec(view, texts[view])
        Q, cos_true, qc, true_row = _G["Q"][view], _G["cos_true"][view], _G["qctry"], _G["true_row"]
        n_q = Q.shape[0]
        same = np.zeros(n_q, np.int32); other = np.zeros(n_q, np.int32)
        best = np.full(n_q, -1.0, np.float32); best_row = np.full(n_q, -1, np.int64)
        for s in range(0, n_q, 500):
            S = (X @ Q[s:s + 500].T).toarray()
            for q in range(s, min(s + 500, n_q)):
                k = pos_of_true.get(true_row[q])
                if k is not None:
                    S[k, q - s] = -1.0
            beat = S >= (cos_true[s:s + 500] - 1e-6)[None, :]
            m = ctry[:, None] == qc[None, s:s + 500]
            same[s:s + 500] = (beat & m).sum(0); other[s:s + 500] = (beat & ~m).sum(0)
            Sm = np.where(m, S, -1.0)
            am = Sm.argmax(0)
            best[s:s + 500] = Sm[am, np.arange(Sm.shape[1])]
            best_row[s:s + 500] = np.asarray(rows)[am]
        out[view] = (same, other, best, best_row)
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="reports/rank_profile.md")
    ap.add_argument("--per-stratum", type=int, default=1000)
    ap.add_argument("--chunk", type=int, default=20000)
    ap.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    d = Path(a.data_dir) / "train"
    rd = lambda p: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)
    s1 = rd(d / "train_source1.tsv")
    cand = pl.concat([rd(d / "train_source2.tsv"), rd(d / "train_source3.tsv")])
    gt = rd(d / "train_ground_truth.tsv")
    ex = (gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
            .filter(pl.col("matched_entity_ids").str.len_chars() > 0)
            .select(s1_id="source1_entity_id", entity_id="matched_entity_ids"))
    log(f"loaded S1={len(s1):,} cand={len(cand):,} pairs={len(ex):,}")

    # stratified query sample: matched S2/S3 records per (source, country)
    qs = (cand.join(ex, on="entity_id").with_columns(src=pl.col("entity_id").str.slice(0, 2))
              .filter(pl.int_range(pl.len()).shuffle(seed=a.seed).over("src", "country") < a.per_stratum))
    s1_row = dict(zip(s1["entity_id"].to_list(), range(len(s1))))
    true_row = np.array([s1_row[x] for x in qs["s1_id"].to_list()], np.int64)
    log(f"queries: {qs.group_by('src', 'country').len().sort('src', 'country').rows()}")

    # normalise everything in parallel; df over all train records
    step = 100_000
    jobs = [(s1["business_name"][i:i + step].to_list(), s1["business_address"][i:i + step].to_list(), True)
            for i in range(0, len(s1), step)]
    n_s1_jobs = len(jobs)
    jobs += [(cand["business_name"][i:i + step].to_list(), cand["business_address"][i:i + step].to_list(), False)
             for i in range(0, len(cand), step)]
    df = {v: np.zeros(NF, np.int64) for v in VIEWS}
    s1_na, s1_nc, s1_ac = [], [], []
    with ProcessPoolExecutor(a.n_jobs) as pool:
        for k, (strs, dfp) in enumerate(pool.map(norm_chunk, jobs)):
            for v, (idx, cnt) in dfp.items():
                df[v][idx] += cnt
            if strs is not None:
                s1_na += strs[0]; s1_nc += strs[1]; s1_ac += strs[2]
            if k % 20 == 0:
                log(f"normalise {k + 1}/{len(jobs)} chunks")
    n_docs = len(s1) + len(cand)
    idf = {v: (np.log((1 + n_docs) / (1 + df[v])) + 1).astype(np.float32) for v in VIEWS}
    log(f"normalised {n_docs:,} records; S1 empty name_core: {np.mean([not x for x in s1_nc]):.4f}")

    from normalize import name_fields, addr_fields
    q_nc = [name_fields(n)["name_core"] for n in qs["business_name"].to_list()]
    q_ac = [addr_fields(y)["addr_clean"] for y in qs["business_address"].to_list()]
    q_na = [(x + " | " + y).strip() for x, y in zip(q_nc, q_ac)]
    _init({"idf": idf})
    q_txt = {"name_addr": q_na, "name": q_nc, "addr": q_ac}
    s1_txt = {"name_addr": s1_na, "name": s1_nc, "addr": s1_ac}
    Q = {v: vec(v, q_txt[v]) for v in VIEWS}
    T = {v: vec(v, [s1_txt[v][r] for r in true_row]) for v in VIEWS}
    cos_true = {v: np.asarray(Q[v].multiply(T[v]).sum(1)).ravel().astype(np.float32) for v in VIEWS}
    s1_ctry = s1["country"].to_numpy()
    g = {"idf": idf, "Q": Q, "cos_true": cos_true, "qctry": qs["country"].to_numpy(), "true_row": true_row}

    perm = np.random.default_rng(a.seed).permutation(len(s1))
    chunks = [perm[i:i + a.chunk] for i in range(0, len(s1), a.chunk)]
    rjobs = [(c.tolist(), {v: [s1_txt[v][r] for r in c] for v in VIEWS}, s1_ctry[c]) for c in chunks]
    n_q = len(qs)
    same = {v: np.zeros((len(chunks), n_q), np.int32) for v in VIEWS}
    other = {v: np.zeros((len(chunks), n_q), np.int32) for v in VIEWS}
    best = {v: np.full(n_q, -1.0, np.float32) for v in VIEWS}
    best_row = {v: np.full(n_q, -1, np.int64) for v in VIEWS}
    with ProcessPoolExecutor(a.n_jobs, initializer=_init, initargs=(g,)) as pool:
        for k, res in enumerate(pool.map(rank_chunk, rjobs)):
            for v, (sm, ot, b, br) in res.items():
                same[v][k], other[v][k] = sm, ot
                upd = b > best[v]
                best[v][upd], best_row[v][upd] = b[upd], br[upd]
            if k % 10 == 0:
                log(f"rank {k + 1}/{len(chunks)} chunks")

    # ------------------------------------------------------------------ report
    src = qs["src"].to_numpy(); qc = qs["country"].to_numpy()
    strata = [("all", np.ones(n_q, bool))] + [(f"{s}/{c}", (src == s) & (qc == c))
                                             for s in ("S2", "S3") for c in sorted(set(qc))]
    ks = (1, 3, 5, 10, 30, 100)
    md = ["# Retrieval profile: rank of the true S1 for S2/S3 records (train, exact TF-IDF cosine)\n",
          f"{n_q:,} matched S2/S3 queries ({a.per_stratum} per source x country) against all {len(s1):,} train S1. "
          "Rank counts ties against us. `same-country` = only S1 records with the query's country compete; "
          "`global` = all S1 compete. Smaller scales = random subsets of S1 (true S1 always kept).\n"]
    rng = np.random.default_rng(a.seed + 1)
    for v in VIEWS:
        md.append(f"\n## View `{v}`\n")
        md.append(f"- cos(query, true S1): mean {cos_true[v].mean():.3f}, p10 {np.quantile(cos_true[v], .1):.3f}, "
                  f"share zero (no shared n-gram, e.g. non-Latin name) {np.mean(cos_true[v] == 0):.4f}")
        md.append(f"- competitors from another country with cos >= true: mean {other[v].sum(0).mean():.2f} per query, "
                  f"share of queries with any {np.mean(other[v].sum(0) > 0):.4f}\n")
        md.append("| scale | stratum | " + " | ".join(f"R@{k}" for k in ks) + " | global R@1 | global R@10 | median rank |")
        md.append("|---|---|" + "---|" * (len(ks) + 3))
        for frac in (0.01, 0.05, 0.2, 1.0):
            n_c = max(1, round(frac * len(chunks)))
            pick = np.sort(rng.choice(len(chunks), n_c, replace=False)) if frac < 1 else np.arange(len(chunks))
            r_same = 1 + same[v][pick].sum(0)
            r_glob = r_same + other[v][pick].sum(0)
            for name, m in strata:
                cells = [f"{np.mean(r_same[m] <= k):.4f}" for k in ks]
                md.append(f"| {frac:.0%} | {name} | " + " | ".join(cells) +
                          f" | {np.mean(r_glob[m] <= 1):.4f} | {np.mean(r_glob[m] <= 10):.4f} | {int(np.median(r_same[m]))} |")
    np.savez_compressed(Path(a.out).with_suffix(".npz"), src=src, qc=qc,
                        **{f"same_{v}": same[v] for v in VIEWS}, **{f"other_{v}": other[v] for v in VIEWS},
                        **{f"cos_true_{v}": cos_true[v] for v in VIEWS})
    md.append("\n## Union of views (true S1 found if its same-country rank <= K in ANY view)\n")
    md.append("| scale | stratum | views | R@1 | R@3 | R@10 | R@30 | R@100 |")
    md.append("|---|---|---|---|---|---|---|---|")
    for frac in (0.01, 0.2, 1.0):
        n_c = max(1, round(frac * len(chunks)))
        pick = np.sort(rng.choice(len(chunks), n_c, replace=False)) if frac < 1 else np.arange(len(chunks))
        rk = {v: 1 + same[v][pick].sum(0) for v in VIEWS}
        for combo in (("name_addr",), ("name_addr", "name"), ("name_addr", "addr"), ("name_addr", "name", "addr")):
            r = np.min([rk[v] for v in combo], axis=0)
            for name, m in strata:
                md.append(f"| {frac:.0%} | {name} | {'+'.join(combo)} | " +
                          " | ".join(f"{np.mean(r[m] <= k):.4f}" for k in (1, 3, 10, 30, 100)) + " |")
    # hard negatives: best same-country wrong S1 that beats (or ties) the true S1, full scale
    v = "name_addr"
    r_full = 1 + same[v].sum(0)
    md.append("\n## Full-scale hard negatives (`name_addr`): queries whose best wrong S1 outranks the true S1\n")
    md.append(f"Share of queries with rank > 1: {np.mean(r_full > 1):.4f}\n")
    s1n, s1a = s1["business_name"].to_list(), s1["business_address"].to_list()
    qn, qa = qs["business_name"].to_list(), qs["business_address"].to_list()
    for c in sorted(set(qc)):
        idx = np.where((qc == c) & (r_full > 1))[0]
        idx = rng.choice(idx, min(30, len(idx)), replace=False) if len(idx) else idx
        md.append(f"\n### {c} ({np.sum((qc == c) & (r_full > 1))} of {np.sum(qc == c)} queries)\n")
        md.append("| q src | rank | cos true / wrong | query (S2/S3) | true S1 | best wrong S1 |")
        md.append("|---|---|---|---|---|---|")
        for q in idx:
            t, w = true_row[q], best_row[v][q]
            cell = lambda n, ad: f"{n} // {ad}".replace("|", "/")[:110]
            md.append(f"| {src[q]} | {r_full[q]} | {cos_true[v][q]:.2f} / {best[v][q]:.2f} | {cell(qn[q], qa[q])} | "
                      f"{cell(s1n[t], s1a[t])} | {cell(s1n[w], s1a[w])} |")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
