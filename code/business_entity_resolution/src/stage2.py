"""
Second-stage features: context over the stage-1 match probabilities.

Stage 1 scores every (S1, candidate) pair on its own. Stage 2 adds, for each pair:
  * within its S1 record:   rank of p, best p, gap to the best, runner-up, total p mass, pairs above 0.5,
                            this pair's share of the mass (how many matches does this S1 really have?)
  * within its candidate:   best p over the competing S1 records, and this pair's margin over the best
                            OTHER S1 (S1 is deduplicated, so a candidate has at most one true owner)
  * sibling agreement:      the S1's other likely candidates (its top 3 by p) should be copies of the
                            same business - how similar is this candidate's name / address / house number /
                            postcode to theirs (max, mean, p-weighted mean)

On train, p must be out-of-fold stage-1 probabilities (otherwise stage 2 learns from leaked labels);
on test, p comes from the final stage-1 model(s).
"""
from __future__ import annotations

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

N_SIBLINGS = 3


def prob_context(pairs: pl.DataFrame, p: np.ndarray) -> pl.DataFrame:
    """pairs (i, j) + stage-1 p -> probability-context features, same row order."""
    D = pairs.select("i", "j").with_columns(p1=pl.Series(p.astype(np.float32)))
    top1_j = pl.col("p1").max().over("j")
    top2_j = pl.col("p1").sort(descending=True).slice(1, 1).first().over("j")
    return D.with_columns(
        p1_rank_i=pl.col("p1").rank("ordinal", descending=True).over("i").cast(pl.Float32),
        p1_max_i=pl.col("p1").max().over("i"),
        p1_gap_i=pl.col("p1").max().over("i") - pl.col("p1"),
        p1_2nd_i=pl.col("p1").sort(descending=True).slice(1, 1).first().over("i").fill_null(0.0),
        p1_sum_i=pl.col("p1").sum().over("i"),
        p1_n05_i=(pl.col("p1") > 0.5).sum().over("i").cast(pl.Float32),
        p1_max_j=top1_j,
        p1_n05_j=(pl.col("p1") > 0.5).sum().over("j").cast(pl.Float32),
        # margin over the best competing S1: if this pair is the candidate's best, compare with the runner-up
        p1_margin_j=pl.col("p1") - pl.when(pl.col("p1") >= top1_j).then(top2_j.fill_null(0.0)).otherwise(top1_j),
    ).with_columns(
        p1_share_i=pl.col("p1") / pl.col("p1_sum_i").clip(1e-6),
    ).drop("i", "j")


def sibling_features(pairs: pl.DataFrame, p: np.ndarray, cand: pl.DataFrame, batch: int = 4_000_000) -> pl.DataFrame:
    """For each pair (i, j): similarity of candidate j to the other top-N_SIBLINGS candidates of S1 i."""
    D = pairs.select("i", "j").with_row_index("r").with_columns(p1=pl.Series(p.astype(np.float32)))
    sib = (D.with_columns(rk=pl.col("p1").rank("ordinal", descending=True).over("i"))
             .filter(pl.col("rk") <= N_SIBLINGS).select("i", js="j", ps="p1"))
    X = D.join(sib, on="i").filter(pl.col("j") != pl.col("js"))
    j, js = X["j"], X["js"]
    feats = {}
    for name, col, scorer in (("sib_addr", "addr_clean", fuzz.token_set_ratio),
                              ("sib_name", "name_core", fuzz.token_set_ratio)):
        a, b = cand[col].gather(j), cand[col].gather(js)
        out = np.empty(len(X), np.float32)
        for s in range(0, len(X), batch):
            out[s:s + batch] = process.cpdist(a.slice(s, batch).to_list(), b.slice(s, batch).to_list(),
                                              scorer=scorer, workers=-1, dtype=np.float32)
        feats[name] = out
    for name, col in (("sib_hn", "addr_house"), ("sib_pc", "addr_postcode")):
        a, b = cand[col].gather(j), cand[col].gather(js)
        feats[name] = ((a != "") & (b != "") & (a == b)).cast(pl.Float32).to_numpy()
        feats[name + "_conf"] = ((a != "") & (b != "") & (a != b)).cast(pl.Float32).to_numpy()
    X = X.select("r", "ps").with_columns(**{k: pl.Series(v) for k, v in feats.items()})
    agg = X.group_by("r").agg(
        n_sib=pl.len().cast(pl.Float32),
        sib_addr_max=pl.col("sib_addr").max(), sib_addr_mean=pl.col("sib_addr").mean(),
        sib_addr_wmean=(pl.col("sib_addr") * pl.col("ps")).sum() / pl.col("ps").sum().clip(1e-6),
        sib_name_max=pl.col("sib_name").max(), sib_name_mean=pl.col("sib_name").mean(),
        sib_hn_match=pl.col("sib_hn").max(), sib_hn_conf=pl.col("sib_hn_conf").mean(),
        sib_pc_match=pl.col("sib_pc").max(), sib_pc_conf=pl.col("sib_pc_conf").mean(),
    )
    out = D.select("r").join(agg, on="r", how="left").sort("r").drop("r")
    # no siblings (the S1 has a single candidate): -1 marks "not applicable", distinct from "disagrees"
    return out.with_columns(pl.col("n_sib").fill_null(0.0), pl.all().exclude("n_sib").fill_null(-1.0).cast(pl.Float32))


def stage2_features(pairs: pl.DataFrame, p: np.ndarray, cand: pl.DataFrame) -> pl.DataFrame:
    """All stage-2 columns for `pairs` (same row order): probability context + sibling agreement."""
    return pl.concat([prob_context(pairs, p), sibling_features(pairs, p, cand)], how="horizontal")
