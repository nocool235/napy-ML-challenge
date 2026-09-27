#!/usr/bin/env python3
"""
Spot-check a match.py test output for one country (France has no labels, so this is how we look at it):

  1. confidence profile per country: how many pairs sit in the uncertain middle, how often an S1's best
     candidate is below 0.5, pairs per S1 - is the model less sure on this country than on the others?
  2. token asymmetry between sources: normalised tokens that S1 / S2 / S3 use at very different rates.
     A variant only one source writes ("av" vs "avenue") that normalize.py doesn't unify shows up here.
  3. borderline examples with raw text: kept pairs just above the rule, dropped pairs just below it,
     and candidates two S1 records compete for.

    python tools/spot_check.py --work-dir cache/real_prep --out-dir output_v2 --country france
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution" / "src"))

import decide  # noqa: E402
import prep  # noqa: E402

COLS = ["entity_id", "country_norm", "source", "business_name", "business_address", "name_core", "addr_clean"]


def confidence_profile(P: pl.DataFrame) -> pl.DataFrame:
    per_s1 = P.group_by("i", "country_norm").agg(pmax=pl.col("p").max(), n=pl.len(), kept=pl.col("keep").sum())
    a = P.group_by("country_norm").agg(
        pairs=pl.len(), mid_share=((pl.col("p") > 0.2) & (pl.col("p") < 0.8)).mean(),
        above_05=(pl.col("p") > 0.5).mean())
    b = per_s1.group_by("country_norm").agg(
        s1_with_pairs=pl.len(), pairs_per_s1=pl.col("n").mean(), best_below_05=(pl.col("pmax") < 0.5).mean(),
        mean_best_p=pl.col("pmax").mean(), kept_per_s1=pl.col("kept").mean())
    return a.join(b, on="country_norm").sort("country_norm")


def token_asymmetry(recs: pl.DataFrame, col: str, min_rate: float = 0.003, top: int = 25) -> pl.DataFrame:
    """Document rate of each token per source; tokens common in one source but rare in another."""
    n_src = recs.group_by("source").len()
    t = (recs.with_row_index("d").select("d", "source", tok=pl.col(col).str.split(" ")).explode("tok")
             .filter(pl.col("tok").str.len_chars() > 0).unique(["d", "tok"])      # once per record, not globally
             .group_by("source", "tok").len().join(n_src, on="source", suffix="_docs")
             .with_columns(rate=pl.col("len") / pl.col("len_docs"))
             .pivot(on="source", index="tok", values="rate").fill_null(0.0))
    srcs = [c for c in t.columns if c != "tok"]
    return (t.with_columns(hi=pl.max_horizontal(srcs), lo=pl.min_horizontal(srcs))
             .filter(pl.col("hi") >= min_rate)
             .with_columns(ratio=(pl.col("hi") + 1e-4) / (pl.col("lo") + 1e-4))
             .sort("ratio", descending=True).head(top))


def show(rows: pl.DataFrame, s1: pl.DataFrame, cand: pl.DataFrame, title: str, n: int, seed: int = 0):
    print(f"\n### {title} ({len(rows):,} pairs; showing {min(n, len(rows))})")
    if len(rows) == 0:
        return
    rows = rows.sample(min(n, len(rows)), seed=seed)
    R = (rows.join(s1.select("i", n1="business_name", a1="business_address"), on="i")
             .join(cand.select("j", n2="business_name", a2="business_address", src="source"), on="j"))
    for r in R.iter_rows(named=True):
        print(f"- p={r['p']:.2f} kept={r['keep']}  S1: {r['n1']} | {r['a1']}")
        print(f"  {r['src']}: {r['n2']} | {r['a2']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", default=str(ROOT / "cache" / "real_prep"))
    ap.add_argument("--out-dir", required=True, help="a match.py output dir (needs debug/test_probs.parquet)")
    ap.add_argument("--country", default="france")
    ap.add_argument("--n", type=int, default=12, help="examples per section")
    a = ap.parse_args()

    out = Path(a.out_dir)
    rule = json.loads((out / "debug" / "metrics_train.json").read_text())["oof"]
    norm = prep.norm_dir(Path(a.work_dir))
    s1 = prep.load(norm, "test", ("source1",), COLS).with_row_index("i")
    cand = prep.load(norm, "test", ("source2", "source3"), COLS).with_row_index("j")
    P = pl.read_parquet(out / "debug" / "test_probs.parquet")
    i, j, p = P["i"].to_numpy(), P["j"].to_numpy(), P["p"].to_numpy().astype(np.float64)
    P = P.with_columns(keep=pl.Series(decide.rule_mask(rule, i, j, p))).join(s1.select("i", "country_norm"), on="i")

    print(f"# Spot-check of {out} (rule {rule['rule']} {rule['param']} o2o={rule['o2o']})")
    print("\n## 1. Confidence profile per country")
    print(confidence_profile(P).to_pandas().round(4).to_string(index=False))

    c = a.country
    s1c, candc = s1.filter(pl.col("country_norm") == c), cand.filter(pl.col("country_norm") == c)
    recs = pl.concat([s1c.select("source", "name_core", "addr_clean"), candc.select("source", "name_core", "addr_clean")])
    for col in ("addr_clean", "name_core"):
        print(f"\n## 2. {c}: most source-asymmetric normalised tokens in {col} (document rate per source)")
        print(token_asymmetry(recs, col).to_pandas().round(4).to_string(index=False))

    Pc = P.filter(pl.col("country_norm") == c)
    print(f"\n## 3. {c}: borderline examples")
    show(Pc.filter(pl.col("keep") & (pl.col("p") < 0.65)), s1c, candc, "kept, but only just (p < 0.65)", a.n)
    show(Pc.filter(~pl.col("keep") & (pl.col("p") > 0.3)), s1c, candc, "dropped, but close (p > 0.3)", a.n, 1)
    contested = Pc.filter(pl.col("p") > 0.3).group_by("j").len().filter(pl.col("len") > 1).select("j")
    show(Pc.join(contested, on="j").filter(pl.col("p") > 0.3).sort("j"), s1c, candc,
         "candidates two S1 records both claim (p > 0.3)", a.n, 2)


if __name__ == "__main__":
    main()
