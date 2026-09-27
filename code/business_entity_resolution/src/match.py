"""
Full-scale matcher, on the parquet cache written by prep.py (polars throughout):

    candidates (retrieval.py)  ->  pair features  ->  LightGBM, out-of-fold over all train S1
      ->  decision rule tuned on the exact macro F0.5 (decide.tune_vec)  ->  output/*.tsv

Train and test go through identical code. candidate_pairs.tsv is exactly the pair set the final
model scored. Candidate pairs (+ the pruner) are cached under <work>/pairs-<hash>/ and features under
<work>/feat-<hash>/, each keyed on the exact source that produced it, and reused automatically
(--no-cache recomputes). --models lgb,xgb --device cuda trains XGBoost on the GPU alongside LightGBM.

    python src/match.py --data-dir ../../dataset --work-dir ../../cache/real_prep --out-dir ../../output
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
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
import stage2  # noqa: E402
from utils import Progress, log, read_tsv  # noqa: E402

SRC = Path(__file__).resolve().parent
FEATURE_VERSION = "f5"      # bump whenever a feature definition below changes
SEED = 42
MIX = {"india": 0.47, "us": 0.38}   # test country mix for the two train countries (France has no labels)
COLS = ["entity_id", "source", "country_norm", "name_core", "name_skel", "name_nospace", "name_ind",
        "addr_clean", "addr_nolm", "addr_words", "addr_street", "addr_house", "addr_postcode", "addr_ntok",
        "name_addr", "addr_has_landmark", "addr_nums", "name_nums", "name_initials", "name_legal"]
LGB_PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_child_samples=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  max_bin=255, verbose=-1, seed=SEED)
XGB_PARAMS = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", grow_policy="lossguide",
                  max_leaves=127, max_depth=0, eta=0.08, subsample=0.8, colsample_bytree=0.8, min_child_weight=1,
                  reg_lambda=1.0, max_bin=256, seed=SEED)   # mirrors LGB_PARAMS: leaf-wise, 127 leaves

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
    # the same after dropping source-noise tokens (see source_noise_tokens)
    ("n_dn_tset", "name_dn", fuzz.token_set_ratio),
    ("n_dn_tsort", "name_dn", fuzz.token_sort_ratio),
    ("n_dn_ratio", "name_dn", fuzz.ratio),
    ("a_dn_tset", "addr_dn", fuzz.token_set_ratio),
]
# A token one source writes far more often than another, within a country, is formatting noise of that
# source (France: "Participations", "Holding" appended to names; India: transliterations like "limitet";
# US: "Midtown", "Riverside" prefixes) - found from the data, the same rule for every country, no labels.
NOISE_MIN_RATE = 0.001   # used by >= 0.1% of one source's records in the country ...
NOISE_RATIO = 5.0        # ... and >= 5x as often as by another source
# (feature, S1 column, candidate column, scorer): name vs the other record's address, for field swaps
CROSS_FEATURES = [
    ("swap_n1a2", "name_core", "addr_clean", fuzz.token_set_ratio),
    ("swap_a1n2", "addr_clean", "name_core", fuzz.token_set_ratio),
]
# (feature, column, analyzer, ngram range): per-country TF-IDF cosine of one field, scored on the pairs
PAIR_TFIDF = [
    ("n_idf_cos", "name_core", "word", (1, 1)),
    ("a_idf_cos", "addr_clean", "word", (1, 1)),
    ("n_char_cos", "name_core", "char_wb", (3, 3)),
]
CONTEXT_COLS = ["base", "n_tset", "a_tset", "n_skel_tset", "n_idf_cos", "n_dn_tset"]
# retrieval-time columns carried into the matcher when TF-IDF retrieval + pruner are on
RETRIEVAL_COLS = ["tf_cos", "tf_rank", "n_keys", "tf_max_j", "tf_gap_j", "tf_rank_i", "tf_gap_i",
                  "n_ret_i", "n_ret_j", "p_prune"] + [
    f"{c}_{suf}" for suf in ("char", "addr", "ns")
    for c in ("tf_cos", "tf_rank", "tf_max_j", "tf_gap_j", "tf_rank_i", "tf_gap_i")
]


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
    specs = [(name, c, c, scorer) for name, c, scorer in STRING_FEATURES] + CROSS_FEATURES
    out = {name: np.empty(len(i), np.float32) for name, _, _, _ in specs}
    cols = sorted({c for _, c1, c2, _ in specs for c in (c1, c2)})
    prog = Progress(len(i), "string features", every=30)
    for s in range(0, len(i), batch):
        ii, jj = pl.Series(i[s:s + batch]), pl.Series(j[s:s + batch])
        a = {c: s1[c].gather(ii).to_list() for c in cols}
        b = {c: cand[c].gather(jj).to_list() for c in cols}
        for name, c1, c2, scorer in specs:
            out[name][s:s + batch] = process.cpdist(a[c1], b[c2], scorer=scorer, workers=-1, dtype=np.float32)
        prog.update(len(ii))
    out["n_jw"] *= 100
    return pl.DataFrame(out)


def source_noise_tokens(s1: pl.DataFrame, cand: pl.DataFrame, col: str) -> dict[str, list[str]]:
    """country -> tokens whose per-record rate differs >= NOISE_RATIO x between two sources (S1/S2/S3) of
    that country, with the higher rate >= NOISE_MIN_RATE. A source missing a token entirely has rate 0."""
    recs = pl.concat([s1.select("source", "country_norm", col), cand.select("source", "country_norm", col)])
    docs = recs.group_by("country_norm", "source").len().rename({"len": "docs"})
    n_src = docs.group_by("country_norm").len().rename({"len": "n_src"})
    rates = (recs.with_row_index("d").select("d", "source", "country_norm", tok=pl.col(col).str.split(" "))
                 .explode("tok").filter(pl.col("tok").str.len_chars() > 0).unique(["d", "tok"])
                 .group_by("country_norm", "source", "tok").len()
                 .join(docs, on=["country_norm", "source"]).with_columns(rate=pl.col("len") / pl.col("docs")))
    noise = (rates.group_by("country_norm", "tok").agg(hi=pl.col("rate").max(), lo=pl.col("rate").min(), ns=pl.len())
                  .join(n_src, on="country_norm")
                  .with_columns(lo=pl.when(pl.col("ns") < pl.col("n_src")).then(0.0).otherwise(pl.col("lo")))
                  .filter((pl.col("hi") >= NOISE_MIN_RATE) & ((pl.col("hi") + 1e-4) / (pl.col("lo") + 1e-4) >= NOISE_RATIO)))
    return {c: g["tok"].to_list() for (c,), g in noise.group_by("country_norm")}


def _denoised(col: str, noise: dict[str, list[str]]) -> pl.Expr:
    """`col` with the country's noise tokens removed."""
    expr = pl.col(col)
    for c, toks in noise.items():
        cleaned = pl.col(col).str.split(" ").list.eval(pl.element().filter(~pl.element().is_in(toks))).list.join(" ")
        expr = pl.when(pl.col("country_norm") == c).then(cleaned).otherwise(expr)
    return expr


def _set_overlap(a: str, b: str, prefix: str) -> dict:
    """Space-separated token sets on both sides (e.g. every number in the address): both present,
    Jaccard overlap, and conflict (both present, nothing shared)."""
    both = (pl.col(a) != "") & (pl.col(b) != "")
    sa, sb = pl.col(a).str.split(" ").list.unique(), pl.col(b).str.split(" ").list.unique()
    inter = sa.list.set_intersection(sb).list.len()
    union = sa.list.set_union(sb).list.len()
    return {f"{prefix}_both": both.cast(pl.Int8),
            f"{prefix}_jacc": pl.when(both).then(inter / union).otherwise(0.0).cast(pl.Float32),
            f"{prefix}_conflict": (both & (inter == 0)).cast(pl.Int8)}


def pair_features(pairs: pl.DataFrame, s1: pl.DataFrame, cand: pl.DataFrame) -> pl.DataFrame:
    """pairs (i, j, kbits) -> pairs + all model features. No feature reads the country label except
    through same-country retrieval, so the model can be applied to France unchanged."""
    t0 = time.time()
    I, J = pairs["i"], pairs["j"]
    ntok = lambda c: pl.col(c).str.split(" ").list.len()   # noqa: E731
    for col, new in (("name_core", "name_dn"), ("addr_clean", "addr_dn")):
        noise = source_noise_tokens(s1, cand, col)
        log(f"source-noise tokens in {col}: " + ", ".join(f"{c} {len(v)} (e.g. {' '.join(v[:6])})" for c, v in noise.items()))
        s1 = s1.with_columns(_denoised(col, noise).alias(new))
        cand = cand.with_columns(_denoised(col, noise).alias(new))
    s1 = s1.with_columns(n_noise=(ntok("name_core") - ntok("name_dn")).cast(pl.Int8))
    cand = cand.with_columns(n_noise=(ntok("name_core") - ntok("name_dn")).cast(pl.Int8))

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
        "lm1": g1("addr_has_landmark"), "lm2": g2("addr_has_landmark"),
        "an1": g1("addr_nums"), "an2": g2("addr_nums"),
        "nn1": g1("name_nums"), "nn2": g2("name_nums"),
        "ini1": g1("name_initials"), "ini2": g2("name_initials"),
        "ns1": g1("name_nospace"), "ns2": g2("name_nospace"),
        "nz1": g1("n_noise"), "nz2": g2("n_noise"),
        "lg1": g1("name_legal"), "lg2": g2("name_legal"),
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
        lm1=pl.col("lm1").cast(pl.Int8), lm2=pl.col("lm2").cast(pl.Int8),
        lm_any=((pl.col("lm1") + pl.col("lm2")) > 0).cast(pl.Int8),
        **_set_overlap("an1", "an2", "a_nums"), **_set_overlap("nn1", "nn2", "n_nums"),
        **_set_overlap("lg1", "lg2", "legal"),
        # one name is the other's initials ("Europeen Gestion" / "EG")
        n_acronym=(((pl.col("ini1").str.len_chars() >= 2) & (pl.col("ns2") == pl.col("ini1")))
                   | ((pl.col("ini2").str.len_chars() >= 2) & (pl.col("ns1") == pl.col("ini2")))).cast(pl.Int8),
        n_noise1=pl.col("nz1").cast(pl.Float32), n_noise2=pl.col("nz2").cast(pl.Float32),
    )
    del raw
    retrieval_cols = [c for c in RETRIEVAL_COLS if c in pairs.columns]
    F = pl.concat([F, pairs.select(pl.col(retrieval_cols).cast(pl.Float32)), string_features(pairs, s1, cand)],
                  how="horizontal")
    for name, c, analyzer, ngram in PAIR_TFIDF:
        t1 = time.time()
        F = F.with_columns(pl.Series(name, retrieval.tfidf_pair_cosine(s1, cand, pairs, c, analyzer, ngram)))
        log(f"feature {name} ({c}, {analyzer} {ngram}) in {time.time() - t1:.0f}s")

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


# extra retrieval views beyond the base word view: name (col, analyzer, ngram_range). Each is unioned in
# with the same extra-chaining trick as retrieval.tfidf_candidates itself (its `extra` becomes everything
# retrieved so far, so calling it again scores the whole running union with the new view's cosine too).
EXTRA_VIEWS = {
    # char n-grams on the core name: typo/spelling-drift matches the word view tokenises past (68% of
    # missed true pairs have high char-level similarity despite being missed at the word level)
    "char": ("name_core", "char_wb", (3, 4)),
    # address-only: catches unrelated-name-same-address misses (~9% of misses) and empty-address dilution
    # (~42% of misses have one empty address, which starves the combined name+address view of signal)
    "addr": ("addr_clean", "word", (1, 1)),
    # char n-grams on the no-space name: run-together web-handle names ("revitup.com" vs "Rev It Up
    # Pilates") that word boundaries in the plain char view still miss
    "ns": ("name_nospace", "char_wb", (3, 4)),
}


def candidates(s1: pl.DataFrame, cand: pl.DataFrame, split: str, pcache: Path, tfidf_k: int,
               gt: pl.DataFrame | None, prune_recall: float, use_cache: bool = True,
               extra_k: dict[str, int] | None = None, extra_max_df: dict[str, float] | None = None) -> pl.DataFrame:
    """Hash-join keys, plus (tfidf_k > 0) word TF-IDF top-k retrieval, plus any EXTRA_VIEWS with a positive
    k in `extra_k` (see EXTRA_VIEWS for what each catches). Then the cascade pruner. The pruner is fit on
    train (needs gt); pruner and pairs are saved in the pair cache, test loads the pruner."""
    extra_k = extra_k or {}
    extra_max_df = extra_max_df or {}
    fp = pcache / f"{split}_pairs.parquet"
    if use_cache and fp.exists():
        log(f"{split}: loading cached candidate pairs {fp}")
        return pl.read_parquet(fp)
    t0 = time.time()
    h = retrieval.hash_candidates(s1, cand)
    log(f"{split}: hash keys -> {len(h):,} pairs ({len(h) / len(s1):.1f}/S1) in {time.time() - t0:.0f}s")
    if tfidf_k <= 0:
        P = h
    else:
        u = retrieval.tfidf_candidates(s1, cand, k=tfidf_k, extra=h.select("i", "j"), log=log)
        fills = [pl.col("tf_cos").fill_null(0.0), pl.col("tf_rank").fill_null(float(tfidf_k + 1))]
        for suffix, (col, analyzer, ngram) in EXTRA_VIEWS.items():
            k = extra_k.get(suffix, 0)
            if k <= 0:
                continue
            v = retrieval.tfidf_candidates(s1, cand, col=col, analyzer=analyzer, ngram_range=ngram,
                                           max_df=extra_max_df.get(suffix, 0.01), k=k,
                                           extra=u.select("i", "j"), log=log)
            u = (v.rename({"tf_cos": f"tf_cos_{suffix}", "tf_rank": f"tf_rank_{suffix}"})
                  .join(u, on=["i", "j"], how="left").with_columns(fills))
            fills = fills + [pl.col(f"tf_cos_{suffix}").fill_null(0.0),
                             pl.col(f"tf_rank_{suffix}").fill_null(float(k + 1))]
            log(f"{split}: + {suffix} view -> {len(u):,} pairs ({len(u) / len(s1):.1f}/S1) in {time.time() - t0:.0f}s")
        u = u.join(h, on=["i", "j"], how="left").with_columns(pl.col("kbits").fill_null(0))
        P = cascade_pruner.pruner_features(u)
        del u, h
        log(f"{split}: retrieval union -> {len(P):,} pairs ({len(P) / len(s1):.1f}/S1) in {time.time() - t0:.0f}s")
        if gt is not None:
            y = P.join(gt.with_columns(y=pl.lit(1, pl.Int8)), on=["i", "j"], how="left")["y"].fill_null(0).to_numpy()
            log(f"{split}: recall before pruning {y.sum() / len(gt):.4f}")
            pruner = cascade_pruner.Pruner()
            pruner.fit(P, y, len(s1), target_recall=prune_recall, log=log)
            pruner.save(pcache)
        else:
            pruner = cascade_pruner.Pruner.load(pcache)
        p = pruner.score(P)
        keep = p >= pruner.threshold
        P = P.with_columns(p_prune=pl.Series(p)).filter(pl.Series(keep))
        if gt is not None:
            log(f"{split}: recall after pruning {y[keep].sum() / len(gt):.4f}")
        log(f"{split}: pruned -> {len(P):,} pairs ({len(P) / len(s1):.1f}/S1), kept {keep.mean():.1%}")
        P = P.rename({"n_i": "n_ret_i", "n_j": "n_ret_j"}).drop([f"k_{n}" for n in retrieval.HASH_KEY_TYPES])
    _write(P, fp)
    return P


def _write(df: pl.DataFrame, fp: Path):
    fp.parent.mkdir(parents=True, exist_ok=True)
    tmp = fp.with_suffix(".tmp")
    df.write_parquet(tmp)
    tmp.replace(fp)           # atomic: a half-written file never looks complete


def build(norm: Path, split: str, pcache: Path, fcache: Path, use_cache: bool, tfidf_k: int, data_dir: Path,
          prune_recall: float, extra_k: dict[str, int] | None = None, extra_max_df: dict[str, float] | None = None):
    """Load a split; candidate pairs and features each come from their own cache when present."""
    s1, cand = load(norm, split)
    fp = fcache / f"{split}_features.parquet"
    if use_cache and fp.exists():
        log(f"{split}: loading cached features {fp}")
        return s1, cand, pl.read_parquet(fp)
    gt = load_truth(data_dir, s1, cand) if split == "train" else None
    pairs = candidates(s1, cand, split, pcache, tfidf_k, gt, prune_recall, use_cache, extra_k, extra_max_df)
    F = pair_features(pairs, s1, cand)
    _write(F, fp)
    return s1, cand, F


def _key(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:10]


def cache_dirs(work: Path, norm: Path, tfidf_k: int, prune_recall: float, extra_k: dict[str, int] | None = None,
               extra_max_df: dict[str, float] | None = None) -> tuple[Path, Path]:
    """Cache names from explicit versions + every setting that changes the output. Versions are bumped by
    hand when behaviour changes (RETRIEVAL_VERSION, PRUNER_VERSION, FEATURE_VERSION); hashing function source
    instead invalidated valid caches whenever a default-preserving parameter was added."""
    extra_k, extra_max_df = extra_k or {}, extra_max_df or {}
    extras = "|".join(f"{s}k{extra_k.get(s, 0)}df{extra_max_df.get(s, 0.01)}" for s in sorted(EXTRA_VIEWS))
    pair_key = _key(f"{retrieval.RETRIEVAL_VERSION}|{cascade_pruner.PRUNER_VERSION}|{norm.name}|k{tfidf_k}|"
                    f"r{prune_recall}|{retrieval.HASH_CAP_S1},{retrieval.HASH_CAP_C}|{extras}")
    feat_key = _key(f"{pair_key}|{FEATURE_VERSION}|{COLS}|{[f[:2] for f in STRING_FEATURES]}|"
                    f"{[f[:3] for f in CROSS_FEATURES]}|{PAIR_TFIDF}|{CONTEXT_COLS}|{RETRIEVAL_COLS}|"
                    f"{NOISE_MIN_RATE}|{NOISE_RATIO}")
    return work / f"pairs-{pair_key}", work / f"feat-{feat_key}"


def feature_cols(F: pl.DataFrame) -> list[str]:
    return [c for c in F.columns if c not in ("i", "j", "y")]


# ----------------------------------------------------------------------------- model
class Model:
    """A fitted LightGBM or XGBoost booster behind one interface."""

    def __init__(self, kind: str, booster, seconds: float, names: list[str]):
        self.kind, self.booster, self.seconds, self.names = kind, booster, seconds, names

    @property
    def best_iteration(self) -> int:
        return int(self.booster.best_iteration)

    @property
    def logloss(self) -> float:
        if self.kind == "lgb":
            return float(self.booster.best_score["valid_0"]["binary_logloss"])
        return float(self.booster.best_score)

    def predict(self, X: np.ndarray, chunk: int = 2_000_000) -> np.ndarray:
        """In row chunks: XGBoost holds its input as CSR (~8 bytes per value), so one 18M x 102 call needs ~15 GB
        and on a 16 GB GPU spills into shared memory and crawls (it stalled v5's test phase)."""
        out = np.empty(len(X), np.float32)
        for s in range(0, len(X), chunk):
            if self.kind == "lgb":
                out[s:s + chunk] = self.booster.predict(X[s:s + chunk], num_iteration=self.best_iteration)
            else:
                import xgboost as xgb
                out[s:s + chunk] = self.booster.predict(xgb.DMatrix(X[s:s + chunk], feature_names=self.names),
                                                        iteration_range=(0, self.best_iteration + 1))
        return out

    def importance(self, names: list[str]) -> np.ndarray:
        if self.kind == "lgb":
            return self.booster.feature_importance("gain")
        g = self.booster.get_score(importance_type="total_gain")
        return np.array([g.get(n, 0.0) for n in names])

    def save(self, d: Path):
        if self.kind == "lgb":
            self.booster.save_model(str(d / "model_lgb.txt"))
        else:
            self.booster.save_model(str(d / "model_xgb.json"))


def fit_model(kind: str, device: str, X, y, tr, va, names, rounds: int = 3000) -> Model:
    """kind 'lgb' (LightGBM, CPU) or 'xgb' (XGBoost, `device` cpu or cuda); early stopping on `va`."""
    t0 = time.time()
    if kind == "lgb":
        dtr = lgb.Dataset(X[tr], y[tr], feature_name=names, free_raw_data=True)
        dva = lgb.Dataset(X[va], y[va], reference=dtr)
        b = lgb.train(LGB_PARAMS, dtr, rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
    else:
        import xgboost as xgb     # optional dependency: only needed when --models includes xgb
        dtr = xgb.QuantileDMatrix(X[tr], y[tr], max_bin=XGB_PARAMS["max_bin"], feature_names=names)
        dva = xgb.QuantileDMatrix(X[va], y[va], ref=dtr, feature_names=names)
        b = xgb.train({**XGB_PARAMS, "device": device}, dtr, rounds, evals=[(dva, "va")],
                      early_stopping_rounds=50, verbose_eval=False)
    return Model(kind, b, time.time() - t0, names)


def train_oof(X, y, i, n_s1, folds: int, frac: float, names: list[str], kinds: list[str], device: str,
              fold_of_s1: np.ndarray | None = None, final: bool = True):
    """Out-of-fold probabilities for EVERY train pair (folds grouped by S1), per model kind. Each fold's
    model trains on a `frac` sample of the other folds' S1 records (all their pairs), with 10% of that
    sample held out for early stopping - so OOF covers all S1s and the one-to-one competition between
    them is intact. With two kinds, both train at once (LightGBM on CPU, XGBoost on the GPU).
    fold_of_s1 overrides the random folds (e.g. one fold per country); final=False skips the final model."""
    rng = np.random.default_rng(SEED)
    fold_s1 = rng.permutation(n_s1) % folds if fold_of_s1 is None else fold_of_s1
    folds = int(fold_s1.max()) + 1
    fold = fold_s1[i]
    samp = (rng.random(n_s1) < frac)[i]
    es = (rng.random(n_s1) < 0.1)[i]
    oof = {k: np.zeros(len(y), np.float32) for k in kinds}
    imps = {k: [] for k in kinds}
    with ThreadPoolExecutor(len(kinds)) as ex:
        for f in range(folds):
            t0 = time.time()
            tr, va, pr = (fold != f) & samp & ~es, (fold != f) & samp & es, fold == f
            futs = {k: ex.submit(fit_model, k, device, X, y, tr, va, names) for k in kinds}
            for k, fu in futs.items():
                m = fu.result()
                oof[k][pr] = m.predict(X[pr])
                imps[k].append(m.importance(names))
                log(f"fold {f} {k}: best_iter={m.best_iteration} logloss={m.logloss:.4f} ({m.seconds:.0f}s)")
            log(f"fold {f}: {time.time() - t0:.0f}s")
        finals = {}
        if final:
            futs = {k: ex.submit(fit_model, k, device, X, y, samp & ~es, samp & es, names) for k in kinds}
            finals = {k: fu.result() for k, fu in futs.items()}
    for k, m in finals.items():
        log(f"final {k}: best_iter={m.best_iteration} ({m.seconds:.0f}s)")
    imp = {k: sorted(zip(names, np.mean(v, axis=0)), key=lambda t: -t[1]) for k, v in imps.items()}
    return oof, finals, imp


def logloss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p.astype(np.float64), 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def blend_weight(oof: dict, y: np.ndarray) -> float:
    """Weight on the first model kind (1.0 with a single kind), chosen on OOF log-loss - a fast proxy;
    the decision rule is then tuned once on the blend."""
    kinds = list(oof)
    if len(kinds) == 1:
        return 1.0
    scores = {w: logloss(blend(oof, w), y) for w in np.round(np.linspace(0, 1, 11), 2)}
    log("blend log-loss by weight on " + kinds[0] + ": " + ", ".join(f"{w}:{s:.5f}" for w, s in scores.items()))
    return float(min(scores, key=scores.get))


def blend(ps: dict, w: float) -> np.ndarray:
    kinds = list(ps)
    return ps[kinds[0]] if len(kinds) == 1 else w * ps[kinds[0]] + (1 - w) * ps[kinds[1]]


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


# ----------------------------------------------------------------------------- France proxy
SHIPPED_RULE = {"rule": "expF", "param": 1.0, "o2o": True}   # the rule every group-CV run has chosen so far


def leave_one_country_out(X, y, i, j, s1, n_true, ctry, cols, kinds, a, dbg: Path):
    """Train on one country, score the other (each train country takes a turn as the unseen one).
    Reported under the shipped rule (honest: France gets a rule tuned on other countries) and under the
    rule re-tuned on these predictions (optimistic upper bound)."""
    countries, code = np.unique(ctry, return_inverse=True)
    if len(countries) < 2:
        raise SystemExit("--cv country needs at least two countries in train")
    oof, _, _ = train_oof(X, y, i, len(s1), 0, a.train_frac, cols, kinds, a.device, fold_of_s1=code, final=False)
    p = blend(oof, blend_weight(oof, y)).astype(np.float64)
    m = decide.rule_mask(SHIPPED_RULE, i, j, p)
    shipped = decide.score_by_group(decide.per_s1_f05(i[m], y[m], len(s1), n_true), ctry, MIX)
    tuned, table = decide.tune_vec(i, j, p, y, n_true, ctry, MIX, verbose=False)
    loco = {"shipped_rule": shipped, "retuned_rule": tuned,
            "loco_mean_shipped": float(np.mean([shipped[str(c)] for c in countries])),
            "models": kinds, "train_frac": a.train_frac, "feature_version": FEATURE_VERSION}
    log(f"LOCO (each country scored by a model trained only on the other(s)), shipped rule: {shipped}")
    log(f"LOCO re-tuned rule (optimistic): {tuned}")
    log(f"LOCO mean under the shipped rule = {loco['loco_mean_shipped']:.4f}  <- France proxy")
    table.to_csv(dbg / "decision_rules_loco.csv", index=False)
    json.dump(loco, open(dbg / "metrics_loco.json", "w"), indent=2, default=float)


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
    ap.add_argument("--char-k", type=int, default=0,
                    help="char 3-4gram TF-IDF top-k per S2/S3 record on name_core (0 = disabled). "
                         "Catches typo/tokenisation misses the word view drops (see retrieval miss-analysis)")
    ap.add_argument("--char-max-df", type=float, default=0.01,
                    help="char view: drop n-grams seen in more than this share of a country's records")
    ap.add_argument("--addr-k", type=int, default=0,
                    help="word TF-IDF top-k on addr_clean alone (0 = disabled). Catches unrelated-name-"
                         "same-address misses and empty-address dilution in the combined name+address view")
    ap.add_argument("--addr-max-df", type=float, default=0.01, help="address view: max_df, as --char-max-df")
    ap.add_argument("--ns-char-k", type=int, default=0,
                    help="char 3-4gram TF-IDF top-k on name_nospace (0 = disabled). Catches run-together "
                         "web-handle names word boundaries in the plain char view still miss")
    ap.add_argument("--ns-char-max-df", type=float, default=0.01, help="name_nospace char view: max_df")
    ap.add_argument("--models", default="lgb",
                    help="comma list of lgb (LightGBM, CPU) and xgb (XGBoost); two kinds are blended")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="device for XGBoost")
    ap.add_argument("--no-cache", action="store_true", help="recompute candidate pairs and features")
    ap.add_argument("--stage2", action="store_true",
                    help="second-stage model on stage-1 OOF probabilities + sibling agreement (kept only if OOF improves)")
    ap.add_argument("--validate-only", action="store_true", help="train + OOF only, no test outputs")
    ap.add_argument("--cv", choices=["group", "country"], default="group",
                    help="country = leave-one-country-out (train on one country, score the other): the France proxy")
    a = ap.parse_args()
    kinds = [k.strip() for k in a.models.split(",") if k.strip()]
    if not kinds or set(kinds) - {"lgb", "xgb"} or len(set(kinds)) != len(kinds):
        ap.error("--models takes lgb, xgb or lgb,xgb")

    t0 = time.time()
    data_dir, out = Path(a.data_dir), Path(a.out_dir)
    dbg = out / "debug"
    dbg.mkdir(parents=True, exist_ok=True)
    norm = prep.norm_dir(Path(a.work_dir))
    extra_k = {"char": a.char_k, "addr": a.addr_k, "ns": a.ns_char_k}
    extra_max_df = {"char": a.char_max_df, "addr": a.addr_max_df, "ns": a.ns_char_max_df}
    pcache, fcache = cache_dirs(Path(a.work_dir), norm, a.tfidf_k, a.prune_recall, extra_k, extra_max_df)
    use_cache = not a.no_cache
    log(f"models {kinds} (xgb device {a.device}); norm {norm}; pair cache {pcache}; feature cache {fcache}; "
        f"extra views k={extra_k}")

    # ---------------- train
    s1, cand, F = build(norm, "train", pcache, fcache, use_cache, a.tfidf_k, data_dir, a.prune_recall,
                        extra_k, extra_max_df)
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
    if a.cv == "country":
        leave_one_country_out(X, y, i, j, s1, n_true, ctry, cols, kinds, a, dbg)
        log(f"done (leave-one-country-out) in {time.time() - t0:.0f}s")
        return
    t_model = time.time()
    oof, finals, imp = train_oof(X, y, i, len(s1), a.folds, a.train_frac, cols, kinds, a.device)
    w = blend_weight(oof, y)
    p_oof = blend(oof, w)
    log(f"models trained in {time.time() - t_model:.0f}s; blend weight on {kinds[0]} = {w}; "
        + ", ".join(f"{k} logloss {logloss(v, y):.5f}" for k, v in oof.items())
        + (f", blend logloss {logloss(p_oof, y):.5f}" if len(kinds) > 1 else ""))
    best, table = decide.tune_vec(i, j, p_oof.astype(np.float64), y, n_true, ctry, MIX)
    table.to_csv(dbg / "decision_rules.csv", index=False)
    # out-of-fold probabilities for error analysis (tools/oof_errors.py)
    pl.DataFrame({"i": i, "j": j, "y": y.astype(np.int8), "p": p_oof.astype(np.float32)}).write_parquet(dbg / "train_oof.parquet")
    log(f"OOF best rule: {best}")
    per_model = {}
    if len(kinds) > 1:   # each model alone, under the blend's rule
        for k, v in oof.items():
            m = decide.rule_mask(best, i, j, v)
            per_model[k] = decide.score_by_group(decide.per_s1_f05(i[m], y[m], len(s1), n_true), ctry, MIX)
            log(f"OOF {k} alone (same rule): {per_model[k]}")
    for k, v in imp.items():
        with open(dbg / f"importance_{k}.tsv", "w") as fh:
            fh.writelines(f"{n}\t{g:.0f}\n" for n, g in v)
    metrics = {"pair_recall": recall, "oracle": oracle, "oof": best, "oof_per_model": per_model,
               "models": kinds, "device": a.device, "blend_weight": w, "n_pairs": len(i),
               "train_frac": a.train_frac, "folds": a.folds, "tfidf_k": a.tfidf_k, "prune_recall": a.prune_recall,
               "extra_k": extra_k, "extra_max_df": extra_max_df, "feature_version": FEATURE_VERSION}
    json.dump(metrics, open(dbg / "metrics_train.json", "w"), indent=2, default=float)
    for m in finals.values():
        m.save(dbg)

    # ---------------- stage 2: context over stage-1 OOF probabilities (same folds, so no leakage)
    rule, finals2, w2, cols2 = best, None, None, None
    if a.stage2:
        t2 = time.time()
        G = stage2.stage2_features(pl.DataFrame({"i": i, "j": j}), p_oof, cand)
        cols2 = cols + G.columns
        X = np.hstack([X, G.to_numpy().astype(np.float32)])
        del G
        log(f"stage 2: {len(cols2) - len(cols)} extra features in {time.time() - t2:.0f}s")
        oof2, finals2, imp2 = train_oof(X, y, i, len(s1), a.folds, a.train_frac, cols2, kinds, a.device)
        w2 = blend_weight(oof2, y)
        best2, table2 = decide.tune_vec(i, j, blend(oof2, w2).astype(np.float64), y, n_true, ctry, MIX)
        table2.to_csv(dbg / "decision_rules_stage2.csv", index=False)
        for k, v in imp2.items():
            with open(dbg / f"importance_stage2_{k}.tsv", "w") as fh:
                fh.writelines(f"{n}\t{g:.0f}\n" for n, g in v)
        used = best2["mix"] > best["mix"]
        metrics["stage2"] = {"oof": best2, "blend_weight": w2, "used": used, "seconds": time.time() - t2}
        log(f"stage 2 OOF best rule: {best2} -> {'USING stage 2' if used else 'not better, keeping stage 1'}")
        if used:
            rule = best2
            (dbg / "stage2").mkdir(exist_ok=True)
            for m in finals2.values():
                m.save(dbg / "stage2")
        else:
            finals2 = None
        json.dump(metrics, open(dbg / "metrics_train.json", "w"), indent=2, default=float)
    del X
    if a.validate_only:
        log(f"done (validate-only) in {time.time() - t0:.0f}s")
        return
    del s1, cand, i, j, y, oof, p_oof

    # ---------------- test
    s1, cand, F = build(norm, "test", pcache, fcache, use_cache, a.tfidf_k, data_dir, a.prune_recall,
                        extra_k, extra_max_df)
    ti, tj = F["i"].to_numpy(), F["j"].to_numpy()
    Xt = F.select(pl.col(cols).cast(pl.Float32)).to_numpy()
    del F
    tp = blend({k: m.predict(Xt) for k, m in finals.items()}, w)
    if finals2 is not None:
        Gt = stage2.stage2_features(pl.DataFrame({"i": ti, "j": tj}), tp, cand)
        Xt = np.hstack([Xt, Gt.to_numpy().astype(np.float32)])
        del Gt
        tp = blend({k: m.predict(Xt) for k, m in finals2.items()}, w2)
    del Xt
    keep = decide.rule_mask(rule, ti, tj, tp)
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
