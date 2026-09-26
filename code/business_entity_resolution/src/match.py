"""
Full-scale matcher, on the parquet cache written by prep.py (polars throughout):

    candidates (retrieval.py)  ->  pair features  ->  LightGBM, out-of-fold over all train S1
      ->  decision rule tuned on the exact macro F0.5 (decide.tune_vec)  ->  output/*.tsv

Train and test go through identical code. candidate_pairs.tsv is exactly the pair set the final
model scored. Features are cached per split under <work>/match-<hash>/ and reused with --resume.

    python src/match.py --data-dir ../../dataset --work-dir ../../cache/real_prep --out-dir ../../output
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cascade_pruner  # noqa: E402
import decide  # noqa: E402
import prep  # noqa: E402
import retrieval  # noqa: E402
from utils import Progress, code_hash, log, read_tsv  # noqa: E402

SRC = Path(__file__).resolve().parent
FEATURE_VERSION = "f2"      # bump whenever a feature definition below changes
SEED = 42
MIX = {"india": 0.47, "us": 0.38}   # test country mix for the two train countries (France has no labels)
COLS = ["entity_id", "source", "country_norm", "name_core", "name_skel", "name_nospace", "name_ind",
        "addr_clean", "addr_nolm", "addr_words", "addr_street", "addr_house", "addr_postcode", "addr_ntok",
        "name_addr"]
LGB_PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_child_samples=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  max_bin=255, verbose=-1, seed=SEED)

STRING_FEATURES = [  # (feature, normalised column, rapidfuzz scorer)
    ("n_tset", "name_core", fuzz.token_set_ratio),
    ("n_tsort", "name_core", fuzz.token_sort_ratio),
    ("n_ratio", "name_core", fuzz.ratio),
    ("n_partial", "name_core", fuzz.partial_ratio),
    ("n_jw", "name_core", JaroWinkler.normalized_similarity),
    ("n_skel_tset", "name_skel", fuzz.token_set_ratio),
    ("n_nospace_ratio", "name_nospace", fuzz.ratio),
    ("a_tset", "addr_clean", fuzz.token_set_ratio),
    ("a_tsort", "addr_clean", fuzz.token_sort_ratio),
    ("a_ratio", "addr_clean", fuzz.ratio),
    ("a_partial", "addr_clean", fuzz.partial_ratio),
    ("a_nolm_tset", "addr_nolm", fuzz.token_set_ratio),
    ("a_words_tset", "addr_words", fuzz.token_set_ratio),
]
CONTEXT_COLS = ["base", "n_tset", "a_tset", "n_skel_tset"]
# retrieval-time columns carried into the matcher when TF-IDF retrieval + pruner are on
RETRIEVAL_COLS = ["tf_cos", "tf_rank", "n_keys", "tf_max_j", "tf_gap_j", "tf_rank_i", "tf_gap_i",
                  "n_ret_i", "n_ret_j", "p_prune"]


# ----------------------------------------------------------------------------- data
def load(norm: Path, split: str):
    s1 = prep.load(norm, split, ("source1",), COLS).with_row_index("i")
    cand = prep.load(norm, split, ("source2", "source3"), COLS).with_row_index("j")
    return s1, cand


def load_truth(data_dir: Path, s1: pl.DataFrame, cand: pl.DataFrame) -> pl.DataFrame:
    """All true (i, j) pairs of train, as row indices."""
    return (read_tsv(data_dir / "train" / "train_ground_truth.tsv")
            .with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
            .filter(pl.col("matched_entity_ids").str.len_chars() > 0)
            .join(s1.select("i", "entity_id"), left_on="source1_entity_id", right_on="entity_id")
            .join(cand.select("j", "entity_id"), left_on="matched_entity_ids", right_on="entity_id")
            .select("i", "j"))


# ----------------------------------------------------------------------------- features
def string_features(pairs: pl.DataFrame, s1: pl.DataFrame, cand: pl.DataFrame, batch: int = 2_000_000) -> pl.DataFrame:
    """rapidfuzz scores for every pair, in batches (cpdist is vectorised and multi-threaded)."""
    i, j = pairs["i"].to_numpy(), pairs["j"].to_numpy()
    out = {name: np.empty(len(i), np.float32) for name, _, _ in STRING_FEATURES}
    cols = sorted({c for _, c, _ in STRING_FEATURES})
    prog = Progress(len(i), "string features", every=30)
    for s in range(0, len(i), batch):
        ii, jj = pl.Series(i[s:s + batch]), pl.Series(j[s:s + batch])
        a = {c: s1[c].gather(ii).to_list() for c in cols}
        b = {c: cand[c].gather(jj).to_list() for c in cols}
        for name, c, scorer in STRING_FEATURES:
            out[name][s:s + batch] = process.cpdist(a[c], b[c], scorer=scorer, workers=-1, dtype=np.float32)
        prog.update(len(ii))
    out["n_jw"] *= 100
    return pl.DataFrame(out)


def pair_features(pairs: pl.DataFrame, s1: pl.DataFrame, cand: pl.DataFrame) -> pl.DataFrame:
    """pairs (i, j, kbits) -> pairs + all model features. No feature reads the country label except
    through same-country retrieval, so the model can be applied to France unchanged."""
    t0 = time.time()
    I, J = pairs["i"], pairs["j"]

    # name frequency within a country over S1 + S2/S3: a very common name (a chain) is weak evidence
    freq = (pl.concat([s1.select("country_norm", "name_core"), cand.select("country_norm", "name_core")])
              .group_by("country_norm", "name_core").len())
    nf1 = s1.join(freq, on=["country_norm", "name_core"], how="left").sort("i")["len"]
    nf2 = cand.join(freq, on=["country_norm", "name_core"], how="left").sort("j")["len"]

    g1 = lambda c: s1[c].gather(I)       # noqa: E731
    g2 = lambda c: cand[c].gather(J)     # noqa: E731
    raw = pl.DataFrame({
        "i": I, "j": J, "kbits": pairs["kbits"],
        "pc1": g1("addr_postcode"), "pc2": g2("addr_postcode"),
        "hn1": g1("addr_house"), "hn2": g2("addr_house"),
        "st1": g1("addr_street"), "st2": g2("addr_street"),
        "n_len1": g1("name_core").str.len_chars(), "n_len2": g2("name_core").str.len_chars(),
        "a_ntok1": g1("addr_ntok"), "a_ntok2": g2("addr_ntok"),
        "n_ind1": g1("name_ind"), "n_ind2": g2("name_ind"),
        "src": g2("source"),
        "n_freq1": nf1.gather(I), "n_freq2": nf2.gather(J),
    })
    both = lambda a, b: (pl.col(a) != "") & (pl.col(b) != "")   # noqa: E731
    F = raw.select(
        "i", "j",
        *[((pl.col("kbits") // (1 << b)) % 2).cast(pl.Int8).alias(f"k_{name}")
          for b, name in enumerate(retrieval.HASH_KEY_TYPES)],
        pc_both=both("pc1", "pc2").cast(pl.Int8),
        pc_match=(both("pc1", "pc2") & (pl.col("pc1") == pl.col("pc2"))).cast(pl.Int8),
        pc_conflict=(both("pc1", "pc2") & (pl.col("pc1") != pl.col("pc2"))).cast(pl.Int8),
        hn_both=both("hn1", "hn2").cast(pl.Int8),
        hn_match=(both("hn1", "hn2") & (pl.col("hn1") == pl.col("hn2"))).cast(pl.Int8),
        hn_conflict=(both("hn1", "hn2") & (pl.col("hn1") != pl.col("hn2"))).cast(pl.Int8),
        st_match=(both("st1", "st2") & (pl.col("st1") == pl.col("st2"))).cast(pl.Int8),
        n_len1=pl.col("n_len1").cast(pl.Float32), n_len2=pl.col("n_len2").cast(pl.Float32),
        a_ntok1=pl.col("a_ntok1").cast(pl.Float32), a_ntok2=pl.col("a_ntok2").cast(pl.Float32),
        n_ind1=pl.col("n_ind1").cast(pl.Int8), n_ind2=pl.col("n_ind2").cast(pl.Int8),
        n_empty=((pl.col("n_len1") == 0) | (pl.col("n_len2") == 0)).cast(pl.Int8),
        a_empty=((pl.col("a_ntok1") == 0) | (pl.col("a_ntok2") == 0)).cast(pl.Int8),
        is_s3=(pl.col("src") == "S3").cast(pl.Int8),
        n_freq1=pl.col("n_freq1").cast(pl.Float32).log1p(), n_freq2=pl.col("n_freq2").cast(pl.Float32).log1p(),
    )
    del raw
    retrieval_cols = [c for c in RETRIEVAL_COLS if c in pairs.columns]
    F = pl.concat([F, pairs.select(pl.col(retrieval_cols).cast(pl.Float32)), string_features(pairs, s1, cand)],
                  how="horizontal")

    # crude-baseline score as a strong prior, then context: how this pair ranks among the S1's candidates
    # (i) and among the S1s competing for the same candidate (j) - S1 is deduplicated, so each candidate
    # has at most one true owner.
    name_best = pl.max_horizontal("n_tset", "n_skel_tset")
    F = F.with_columns(base=pl.when(pl.col("a_empty") == 1).then(name_best * 0.85)
                               .otherwise(0.4 * name_best + 0.6 * pl.col("a_tset")) / 100)
    ctx = []
    for c in CONTEXT_COLS:
        for g in ("i", "j"):
            ctx += [pl.col(c).rank("min", descending=True).over(g).cast(pl.Float32).alias(f"{c}_rank_{g}"),
                    (pl.col(c).max().over(g) - pl.col(c)).cast(pl.Float32).alias(f"{c}_gap_{g}")]
    ctx += [pl.len().over("i").cast(pl.Float32).alias("n_cand_i"), pl.len().over("j").cast(pl.Float32).alias("n_cand_j")]
    F = F.with_columns(ctx)
    log(f"features: {F.width - 2} columns x {len(F):,} pairs in {time.time() - t0:.0f}s")
    return F


def candidates(s1: pl.DataFrame, cand: pl.DataFrame, split: str, cache: Path, tfidf_k: int,
               gt: pl.DataFrame | None, prune_recall: float = 0.997) -> pl.DataFrame:
    """Hash-join keys, plus (tfidf_k > 0) TF-IDF top-k retrieval and the cascade pruner.
    The pruner is fit on train (needs gt) and saved in the cache dir; test loads it."""
    t0 = time.time()
    h = retrieval.hash_candidates(s1, cand)
    log(f"{split}: hash keys -> {len(h):,} pairs ({len(h) / len(s1):.1f}/S1) in {time.time() - t0:.0f}s")
    if tfidf_k <= 0:
        return h
    u = (retrieval.tfidf_candidates(s1, cand, k=tfidf_k, extra=h.select("i", "j"), log=log)
         .join(h, on=["i", "j"], how="left").with_columns(pl.col("kbits").fill_null(0)))
    P = cascade_pruner.pruner_features(u)
    del u, h
    log(f"{split}: retrieval union -> {len(P):,} pairs ({len(P) / len(s1):.1f}/S1) in {time.time() - t0:.0f}s")
    if gt is not None:
        y = P.join(gt.with_columns(y=pl.lit(1, pl.Int8)), on=["i", "j"], how="left")["y"].fill_null(0).to_numpy()
        log(f"{split}: recall before pruning {y.sum() / len(gt):.4f}")
        pruner = cascade_pruner.Pruner()
        pruner.fit(P, y, len(s1), target_recall=prune_recall, log=log)
        pruner.save(cache)
    else:
        pruner = cascade_pruner.Pruner.load(cache)
    p = pruner.score(P)
    keep = p >= pruner.threshold
    P = P.with_columns(p_prune=pl.Series(p)).filter(pl.Series(keep))
    if gt is not None:
        log(f"{split}: recall after pruning {y[keep].sum() / len(gt):.4f}")
    log(f"{split}: pruned -> {len(P):,} pairs ({len(P) / len(s1):.1f}/S1), kept {keep.mean():.1%}")
    return P.rename({"n_i": "n_ret_i", "n_j": "n_ret_j"}).drop([f"k_{n}" for n in retrieval.HASH_KEY_TYPES])


def build(norm: Path, split: str, cache: Path, resume: bool, tfidf_k: int, data_dir: Path,
          prune_recall: float = 0.997):
    """Load a split, generate candidates and features (cached as parquet)."""
    s1, cand = load(norm, split)
    fp = cache / f"{split}_features.parquet"
    if resume and fp.exists():
        log(f"{split}: loading cached features {fp}")
        return s1, cand, pl.read_parquet(fp)
    gt = load_truth(data_dir, s1, cand) if split == "train" else None
    pairs = candidates(s1, cand, split, cache, tfidf_k, gt, prune_recall)
    F = pair_features(pairs, s1, cand)
    cache.mkdir(parents=True, exist_ok=True)
    tmp = fp.with_suffix(".tmp")
    F.write_parquet(tmp)
    tmp.rename(fp)
    return s1, cand, F


def feature_cols(F: pl.DataFrame) -> list[str]:
    return [c for c in F.columns if c not in ("i", "j", "y")]


# ----------------------------------------------------------------------------- model
def _fit(X, y, tr, va, names, rounds=3000):
    dtr = lgb.Dataset(X[tr], y[tr], feature_name=names, free_raw_data=True)
    dva = lgb.Dataset(X[va], y[va], reference=dtr)
    return lgb.train(LGB_PARAMS, dtr, rounds, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(50, verbose=False)])


def train_oof(X, y, i, n_s1, folds: int, frac: float, names: list[str]):
    """Out-of-fold probabilities for EVERY train pair (folds grouped by S1). Each fold's model trains on
    a `frac` sample of the other folds' S1 records (all their pairs), with 10% of that sample held out
    for early stopping - so OOF covers all S1s and the one-to-one competition between them is intact."""
    rng = np.random.default_rng(SEED)
    fold = (rng.permutation(n_s1) % folds)[i]
    samp = (rng.random(n_s1) < frac)[i]
    es = (rng.random(n_s1) < 0.1)[i]
    oof = np.zeros(len(y), np.float32)
    iters, imps = [], []
    for k in range(folds):
        t0 = time.time()
        m = _fit(X, y, (fold != k) & samp & ~es, (fold != k) & samp & es, names)
        va = fold == k
        oof[va] = m.predict(X[va], num_iteration=m.best_iteration)
        iters.append(m.best_iteration)
        imps.append(m.feature_importance("gain"))
        log(f"fold {k}: best_iter={m.best_iteration} logloss={m.best_score['valid_0']['binary_logloss']:.4f} "
            f"({time.time() - t0:.0f}s)")
    final = _fit(X, y, samp & ~es, samp & es, names)
    log(f"final model: best_iter={final.best_iteration}")
    imp = sorted(zip(names, np.mean(imps, axis=0)), key=lambda t: -t[1])
    return oof, final, imp


# ----------------------------------------------------------------------------- outputs
def write_lists(path: Path, s1: pl.DataFrame, cand: pl.DataFrame, i: np.ndarray, j: np.ndarray, col: str):
    agg = (pl.DataFrame({"i": i, "j": j}).with_columns(pl.col("i").cast(pl.UInt32), pl.col("j").cast(pl.UInt32))
             .join(cand.select("j", cid="entity_id"), on="j")
             .group_by("i").agg(pl.col("cid").sort().str.join(",")))
    (s1.select("i", source1_entity_id="entity_id").join(agg, on="i", how="left")
       .sort("i").select("source1_entity_id", pl.col("cid").fill_null("").alias(col))
       .write_csv(path, separator="\t", quote_style="never"))


def structure(s1: pl.DataFrame, cand: pl.DataFrame, ki: np.ndarray, kj: np.ndarray) -> dict:
    """Predicted structure per country (train truth: 5.6% singletons, 3.46 matches/S1)."""
    ctry, cc = s1["country_norm"].to_numpy(), cand["country_norm"].to_numpy()
    n_per = np.bincount(ki, minlength=len(s1))
    assigned = np.zeros(len(cand), bool)
    assigned[kj] = True
    return {c: {"s1": int((ctry == c).sum()), "pred_singleton_rate": round(float((n_per[ctry == c] == 0).mean()), 4),
                "mean_matches_per_s1": round(float(n_per[ctry == c].mean()), 3),
                "share_s2s3_unassigned": round(float(1 - assigned[cc == c].mean()), 4)} for c in np.unique(ctry)}


# ----------------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../../dataset")
    ap.add_argument("--work-dir", default="../../work")
    ap.add_argument("--out-dir", default="../../output")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--train-frac", type=float, default=0.3,
                    help="share of train S1 records each model trains on (OOF still covers every S1)")
    ap.add_argument("--tfidf-k", type=int, default=10,
                    help="TF-IDF top-k S1 per S2/S3 record, unioned with hash keys, then pruned; 0 = hash keys only")
    ap.add_argument("--prune-recall", type=float, default=0.997,
                    help="share of retrieved true pairs the cascade pruner must keep (train holdout)")
    ap.add_argument("--resume", action="store_true", help="reuse cached features for this code version")
    ap.add_argument("--validate-only", action="store_true", help="train + OOF only, no test outputs")
    a = ap.parse_args()

    t0 = time.time()
    data_dir, out = Path(a.data_dir), Path(a.out_dir)
    dbg = out / "debug"
    dbg.mkdir(parents=True, exist_ok=True)
    norm = prep.norm_dir(Path(a.work_dir))
    cache = Path(a.work_dir) / f"match-{code_hash(SRC / 'retrieval.py', SRC / 'cascade_pruner.py', extra=f"{FEATURE_VERSION}|{norm.name}|k{a.tfidf_k}|r{a.prune_recall}")}"
    log(f"norm cache {norm}, feature cache {cache}")

    # ---------------- train
    s1, cand, F = build(norm, "train", cache, a.resume, a.tfidf_k, data_dir, a.prune_recall)
    gt = load_truth(data_dir, s1, cand)
    n_true = np.bincount(gt["i"].to_numpy(), minlength=len(s1))
    F = F.join(gt.with_columns(y=pl.lit(1, pl.Int8)), on=["i", "j"], how="left").with_columns(pl.col("y").fill_null(0))
    i, j, y = F["i"].to_numpy(), F["j"].to_numpy(), F["y"].to_numpy().astype(np.float32)
    ctry = s1["country_norm"].to_numpy()
    recall = float(y.sum() / len(gt))
    oracle = decide.score_by_group(decide.per_s1_f05(i[y == 1], y[y == 1], len(s1), n_true), ctry, MIX)
    log(f"train: pairs={len(F):,} positives={int(y.sum()):,} pair recall={recall:.4f} "
        f"oracle F0.5 (perfect matcher on these candidates) mix={oracle['mix']:.4f}")

    cols = feature_cols(F)
    X = F.select(pl.col(cols).cast(pl.Float32)).to_numpy()
    del F
    oof, model, imp = train_oof(X, y, i, len(s1), a.folds, a.train_frac, cols)
    del X
    best, table = decide.tune_vec(i, j, oof.astype(np.float64), y, n_true, ctry, MIX)
    table.to_csv(dbg / "decision_rules.csv", index=False)
    log(f"OOF best rule: {best}")
    with open(dbg / "importance.tsv", "w") as fh:
        fh.writelines(f"{n}\t{g:.0f}\n" for n, g in imp)
    metrics = {"pair_recall": recall, "oracle": oracle, "oof": best, "n_pairs": len(i),
               "train_frac": a.train_frac, "folds": a.folds, "tfidf_k": a.tfidf_k, "prune_recall": a.prune_recall, "feature_version": FEATURE_VERSION}
    json.dump(metrics, open(dbg / "metrics_train.json", "w"), indent=2, default=float)
    model.save_model(str(dbg / "model.txt"))
    if a.validate_only:
        log(f"done (validate-only) in {time.time() - t0:.0f}s")
        return
    del s1, cand, i, j, y, oof

    # ---------------- test
    s1, cand, F = build(norm, "test", cache, a.resume, a.tfidf_k, data_dir, a.prune_recall)
    ti, tj = F["i"].to_numpy(), F["j"].to_numpy()
    tp = model.predict(F.select(pl.col(cols).cast(pl.Float32)).to_numpy(), num_iteration=model.best_iteration)
    del F
    keep = decide.rule_mask(best, ti, tj, tp)
    out.mkdir(parents=True, exist_ok=True)
    write_lists(out / "matching_results.tsv", s1, cand, ti[keep], tj[keep], "matched_entity_ids")
    write_lists(out / "candidate_pairs.tsv", s1, cand, ti, tj, "candidate_entity_ids")
    pl.DataFrame({"i": ti, "j": tj, "p": tp.astype(np.float32)}).write_parquet(dbg / "test_probs.parquet")
    st = structure(s1, cand, ti[keep], tj[keep])
    for c, v in st.items():
        log(f"test {c}: {v}")
    metrics["test_structure"] = st
    json.dump(metrics, open(dbg / "metrics_train.json", "w"), indent=2, default=float)
    log(f"wrote {out / 'matching_results.tsv'} in {time.time() - t0:.0f}s total")


if __name__ == "__main__":
    main()
