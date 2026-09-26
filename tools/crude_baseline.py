#!/usr/bin/env python3
"""
Crude no-ML baseline: exact hash-join keys + one similarity threshold. It exists to prove the output
format, the validator and the portal end to end and to get a first LB number; it is independent of
the main pipeline and only reads the normalised parquet cache written by src/prep.py.

Keys (always inside one country; key groups larger than the caps are dropped):
    name   : exact name_core
    skel   : name skeleton + one of the record's 2 rarest address words
    street : house number + street word
    hskel  : first name-skeleton token + house number
Score = w * name_sim + (1 - w) * addr_sim (rapidfuzz token_set_ratio; name_sim also tries skeletons);
if either address is empty the score is name_sim * EMPTY_ADDR_FACTOR. Every S2/S3 record goes to its
best-scoring S1 and is kept if score >= t. (w, t) are tuned on train against macro F0.5.

    python tools/crude_baseline.py --work-dir work --data-dir <dataset> --out-dir output_crude
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution" / "src"))

import prep  # noqa: E402
from metric import f05  # noqa: E402
from utils import log, read_tsv  # noqa: E402

COLS = ["entity_id", "country_norm", "name_core", "name_skel", "addr_clean", "addr_words", "addr_street", "addr_house"]
CAP_S1, CAP_C = 5, 50
EMPTY_ADDR_FACTOR = 0.85
W_GRID = [0.4, 0.5, 0.6, 0.7]
T_GRID = [round(x, 3) for x in np.arange(0.50, 0.97, 0.01)]


def keys_for(df: pl.DataFrame, idx: str, rare: pl.DataFrame) -> pl.DataFrame:
    c = pl.col("country_norm")
    k_name = df.filter(pl.col("name_core").str.len_chars() >= 3).select(
        idx, key=pl.concat_str([c, pl.lit("|n|"), pl.col("name_core")]))
    k_street = df.filter(pl.col("addr_street") != "").select(
        idx, key=pl.concat_str([c, pl.lit("|s|"), pl.col("addr_street")]))
    k_hskel = df.filter((pl.col("addr_house") != "") & (pl.col("name_skel") != "")).select(
        idx, key=pl.concat_str([c, pl.lit("|h|"), pl.col("name_skel").str.split(" ").list.first(), pl.lit("|"),
                                pl.col("addr_house")]))
    k_skel = (rare.join(df.select(idx, "country_norm", "name_skel"), on=idx)
                .filter(pl.col("name_skel") != "")
                .select(idx, key=pl.concat_str([c, pl.lit("|k|"), pl.col("name_skel"), pl.lit("|"), pl.col("w")])))
    return pl.concat([k_name, k_street, k_hskel, k_skel]).with_columns(pl.col("key").hash()).unique()


def rare_words(s1: pl.DataFrame, cand: pl.DataFrame, k: int = 2, chunk: int = 2_000_000):
    """each record's k rarest address words (document frequency over S1 + S2/S3 of the same country).
    Two chunked passes (global counts, then per-record top-k) keep peak memory low."""
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


def candidate_pairs(s1: pl.DataFrame, cand: pl.DataFrame) -> pl.DataFrame:
    r1, rc = rare_words(s1, cand)
    k1, kc = keys_for(s1, "i", r1), keys_for(cand, "j", rc)
    g1 = k1.group_by("key").len().filter(pl.col("len") <= CAP_S1).select("key")
    gc = kc.group_by("key").len().filter(pl.col("len") <= CAP_C).select("key")
    pairs = k1.join(g1, on="key").join(kc.join(gc, on="key"), on="key").select("i", "j").unique()
    return pairs.sort("i", "j")


def similarities(pairs: pl.DataFrame, s1: pl.DataFrame, cand: pl.DataFrame, batch: int = 3_000_000):
    i, j = pairs["i"].to_numpy(), pairs["j"].to_numpy()
    name = np.empty(len(i), np.float32); addr = np.empty(len(i), np.float32); empty = np.empty(len(i), bool)
    for s in range(0, len(i), batch):
        ii, jj = pl.Series(i[s:s + batch]), pl.Series(j[s:s + batch])
        g = lambda c: (s1[c].gather(ii).to_list(), cand[c].gather(jj).to_list())
        n1 = process.cpdist(*g("name_core"), scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
        n2 = process.cpdist(*g("name_skel"), scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
        a, b = g("addr_clean")
        ad = process.cpdist(a, b, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
        name[s:s + batch] = np.maximum(n1, n2) / 100
        addr[s:s + batch] = ad / 100
        empty[s:s + batch] = (np.array([x == "" for x in a]) | np.array([x == "" for x in b]))
    return name, addr, empty


def best_per_candidate(j, name, addr, empty, w):
    """row of the best-scoring S1 for every candidate j, and the score array."""
    score = np.where(empty, name * EMPTY_ADDR_FACTOR, w * name + (1 - w) * addr)
    order = np.lexsort((-score, j))                      # per candidate j, best S1 first
    first = np.ones(len(order), bool)
    first[1:] = j[order][1:] != j[order][:-1]
    return order[first], score


def decide(i, j, name, addr, empty, w, t):
    best, score = best_per_candidate(j, name, addr, empty, w)
    keep = best[score[best] >= t]
    decide.last_rows = keep
    return i[keep], j[keep]


def y_of(pi, pj, i, j, y):
    return y[decide.last_rows]


def per_s1_f05(keep_i, keep_y, n_s1, n_true):
    """exact per-S1 F0.5 from kept pairs: F = 1.25 TP / (0.25 T + P); empty prediction scores 1 iff T == 0."""
    P = np.bincount(keep_i, minlength=n_s1)
    TP = np.bincount(keep_i, weights=keep_y, minlength=n_s1)
    f = np.where(P + n_true == 0, 1.0, 1.25 * TP / np.maximum(0.25 * n_true + P, 1e-9))
    return f


def load(norm, split):
    s1 = prep.load(norm, split, ("source1",), COLS).with_row_index("i")
    cand = prep.load(norm, split, ("source2", "source3"), COLS).with_row_index("j")
    return s1, cand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", default=str(ROOT / "work"))
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", default=str(ROOT / "output_crude"))
    a = ap.parse_args()
    norm = prep.norm_dir(Path(a.work_dir))
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    # ---------------- train: tune (w, t)
    s1, cand = load(norm, "train")
    gt = (read_tsv(Path(a.data_dir) / "train" / "train_ground_truth.tsv")
            .with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
            .filter(pl.col("matched_entity_ids").str.len_chars() > 0)
            .join(s1.select("i", "entity_id"), left_on="source1_entity_id", right_on="entity_id")
            .join(cand.select("j", "entity_id"), left_on="matched_entity_ids", right_on="entity_id")
            .select("i", "j", y=pl.lit(1, pl.Int8)))
    n_true = np.bincount(gt["i"].to_numpy(), minlength=len(s1))
    pairs = candidate_pairs(s1, cand).join(gt, on=["i", "j"], how="left").with_columns(pl.col("y").fill_null(0))
    i, j, y = pairs["i"].to_numpy(), pairs["j"].to_numpy(), pairs["y"].to_numpy().astype(np.float64)
    log(f"train: {len(pairs):,} candidate pairs ({len(pairs) / len(s1):.1f}/S1), pair recall {y.sum() / len(gt):.4f}")
    name, addr, empty = similarities(pairs, s1, cand)
    log("train: similarities done")
    ctry = s1["country_norm"].to_numpy()
    mix = {"india": 0.47, "us": 0.38}
    best, table = None, []
    for w in W_GRID:
        best_rows, score = best_per_candidate(j, name, addr, empty, w)
        for t in T_GRID:
            keep = best_rows[score[best_rows] >= t]
            f = per_s1_f05(i[keep], y[keep], len(s1), n_true)
            per = {c: float(f[ctry == c].mean()) for c in np.unique(ctry)}
            weighted = sum(mix.get(c, 0) * v for c, v in per.items()) / sum(mix.get(c, 0) for c in per)
            table.append({"w": w, "t": t, **per, "all": float(f.mean()), "test_mix": weighted})
            if best is None or weighted > best["test_mix"]:
                best = table[-1]
        log(f"tuned w={w}: best so far {best}")
    pl.DataFrame(table).write_csv(out / "crude_tuning.csv")
    log(f"chosen {best}")
    del s1, cand, pairs, name, addr, empty

    # ---------------- test
    s1, cand = load(norm, "test")
    pairs = candidate_pairs(s1, cand)
    log(f"test: {len(pairs):,} candidate pairs ({len(pairs) / len(s1):.1f}/S1)")
    name, addr, empty = similarities(pairs, s1, cand)
    i, j = pairs["i"].to_numpy(), pairs["j"].to_numpy()
    pi, pj = decide(i, j, name, addr, empty, best["w"], best["t"])
    s1_ids, c_ids = s1["entity_id"].to_list(), cand["entity_id"].to_numpy()

    def write(path, col, ii, jj):
        lists = {}
        for x, y in zip(ii, jj):
            lists.setdefault(x, []).append(c_ids[y])
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(f"source1_entity_id\t{col}\n")
            for r, sid in enumerate(s1_ids):
                fh.write(f"{sid}\t{','.join(sorted(set(lists.get(r, ()))))}\n")

    write(out / "matching_results.tsv", "matched_entity_ids", pi, pj)
    write(out / "candidate_pairs.tsv", "candidate_entity_ids", i, j)
    # structure check against train's generator settings (5.6% singletons, 3.46 matches/S1, 26% unassigned)
    ctry = s1["country_norm"].to_numpy(); cc = cand["country_norm"].to_numpy()
    n_per = np.bincount(pi, minlength=len(s1_ids))
    assigned = np.zeros(len(c_ids), bool); assigned[pj] = True
    stats = {c: {"s1": int((ctry == c).sum()), "pred_singleton_rate": float((n_per[ctry == c] == 0).mean()),
                 "mean_matches_per_s1": float(n_per[ctry == c].mean()),
                 "share_s2s3_unassigned": float(1 - assigned[cc == c].mean())} for c in np.unique(ctry)}
    json.dump({"chosen": best, "test_structure": stats}, open(out / "crude_summary.json", "w"), indent=2)
    for c, v in stats.items():
        log(f"test {c}: {v}")
    log(f"wrote {out / 'matching_results.tsv'}")


if __name__ == "__main__":
    main()
