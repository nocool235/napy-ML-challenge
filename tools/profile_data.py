#!/usr/bin/env python3
"""
Step-0 profile of the FULL dataset: sizes, missingness and noise patterns per source/country,
ground-truth structure, cross-country pairs, and examples (true pairs, France formats).
Polars; one split in memory at a time.

    python tools/profile_data.py --data-dir ../student_resource/dataset --out reports/data_profile.md
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import polars as pl

INDIC = r"[\x{0900}-\x{0DFF}]"  # Devanagari .. Sinhala (Hindi, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam)
LEGAL = r"llc|inc|ltd|limited|pvt|private|corp|corporation|company|co|llp|plc|pllc|sarl|sas|sasu|sci|eurl|sa|snc|gmbh"
NAME_PATTERNS = {
    "name_indic_script": INDIC,
    "name_handle_or_url": r"(?i)(^\s*@|www\.|\.(com|in|net|org|co|biz|info|fr)\b)",
    "name_leading_junk": r"^\s*[^\p{L}\p{N}\s]",
    "name_pipe": r"\|",
    "name_hash_number": r"#\s?\d",
    "name_brackets": r"[\(\)\[\]]",
    "name_legal_form": rf"(?i)\b({LEGAL})\b",
    "name_legal_form_first": rf"(?i)^\W*({LEGAL})\b",
    "name_dba": r"(?i)\b(dba|d/b/a|t/a|aka|trading as)\b",
    "name_digit_inside_word": r"\p{L}\d\p{L}",
    "name_accented_latin": r"[À-ÖØ-öø-ÿ]",
}
ADDR_PATTERNS = {
    "addr_literal_null": r"(?i)\bnull\b",
    "addr_indic_script": INDIC,
    "addr_5digit": r"\b\d{5}\b",
    "addr_6digit_or_3sp3": r"\b\d{6}\b|\b\d{3}\s\d{3}\b",
    "addr_zip_plus4": r"\b\d{5}-\d{4}\b",
    "addr_landmark": r"(?i)\b(near|opp|opposite|behind|beside|next to|nr)\b",
    "addr_po_box": r"(?i)\bp\.?\s?o\.?\s?box\b",
    "addr_unit_suite_flat": r"(?i)\b(unit|suite|ste|apt|apartment|flat|floor)\b|#",
    "addr_starts_with_digit": r"^\s*\d",
    "addr_accented_latin": r"[À-ÖØ-öø-ÿ]",
    "addr_cedex_bis_ter": r"(?i)\b(cedex|bis|ter)\b",
}


def read(p: Path) -> pl.DataFrame:
    return pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)


def md_table(df: pl.DataFrame, float_fmt="{:.3f}") -> str:
    cols = df.columns
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.iter_rows():
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append(float_fmt.format(v))
            else:
                cells.append(str(v).replace("|", "\\|").replace("\n", " "))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def pattern_stats(df: pl.DataFrame, split: str) -> pl.DataFrame:
    n, a = pl.col("business_name"), pl.col("business_address")
    ex = [pl.len().alias("n"),
          (n.str.strip_chars() == "").mean().alias("name_empty"),
          (a.str.strip_chars() == "").mean().alias("addr_empty")]
    ex += [n.str.contains(p).mean().alias(k) for k, p in NAME_PATTERNS.items()]
    ex += [((n == n.str.to_uppercase()) & n.str.contains(r"[A-Z]")).mean().alias("name_all_caps"),
           ((n == n.str.to_lowercase()) & n.str.contains(r"[a-z]")).mean().alias("name_all_lower")]
    ex += [a.str.contains(p).mean().alias(k) for k, p in ADDR_PATTERNS.items()]
    ex += [((a == a.str.to_uppercase()) & a.str.contains(r"[A-Z]")).mean().alias("addr_all_caps"),
           (a.str.count_matches(",") + 1).mean().alias("addr_comma_parts"),
           n.str.len_chars().mean().alias("name_len"), a.str.len_chars().mean().alias("addr_len")]
    return (df.with_columns(pl.col("entity_id").str.slice(0, 2).alias("source"))
              .group_by(["source", "country"]).agg(ex)
              .with_columns(pl.lit(split).alias("split")).sort(["split", "source", "country"]))


def transpose_stats(st: pl.DataFrame) -> pl.DataFrame:
    keys = [f"{s}/{src}/{c}" for s, src, c in zip(st["split"], st["source"], st["country"])]
    metrics = [c for c in st.columns if c not in ("split", "source", "country")]
    rows = {"metric": metrics}
    for k, r in zip(keys, st.iter_rows(named=True)):
        rows[k] = [float(r[m]) for m in metrics]
    return pl.DataFrame(rows)


def top_tokens(s: pl.Series, k=50) -> str:
    toks = (s.str.to_lowercase().str.replace_all(r"[^\p{L}\p{N}]+", " ").str.strip_chars()
             .str.split(" ").explode().drop_nulls())
    toks = toks.filter(toks.str.len_chars() > 0)
    vc = toks.value_counts(sort=True).head(k)
    return ", ".join(f"{t} ({c:,})" for t, c in vc.iter_rows())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="reports/data_profile.md")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    d, out = Path(args.data_dir), Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    md, t0 = ["# Data profile (full dataset)\n"], time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    stats, sizes = [], []
    for split in ("train", "test"):
        frames = {}
        for src in ("source1", "source2", "source3"):
            f = read(d / split / f"{split}_{src}.tsv")
            frames[src] = f
            sizes.append(f.group_by("country").len().with_columns(split=pl.lit(split), source=pl.lit(src)))
            stats.append(pattern_stats(f, split))
            log(f"{split}/{src}: {len(f):,} rows")
        if split == "train":
            s1, cand = frames["source1"], pl.concat([frames["source2"], frames["source3"]])
            gt = read(d / "train" / "train_ground_truth.tsv")
            ex = (gt.with_columns(pl.col("matched_entity_ids").str.split(","))
                    .explode("matched_entity_ids")
                    .filter(pl.col("matched_entity_ids").str.len_chars() > 0)
                    .select(s1="source1_entity_id", m="matched_entity_ids"))
            md.append("## Ground truth structure (train)\n")
            n_s1 = len(s1)
            dup_m = int(ex["m"].is_duplicated().sum())
            md.append(f"- S1 rows: {n_s1:,}; GT rows: {len(gt):,}; S1 ids missing from GT: "
                      f"{len(s1.join(gt, left_on='entity_id', right_on='source1_entity_id', how='anti')):,}; "
                      f"GT ids not in S1: {len(gt.join(s1, left_on='source1_entity_id', right_on='entity_id', how='anti')):,}")
            md.append(f"- True pairs: {len(ex):,}; distinct matched S2/S3 ids: {ex['m'].n_unique():,}; "
                      f"rows whose S2/S3 id appears under >1 S1: {dup_m:,}")
            md.append(f"- Matched ids not present in S2/S3 files: {len(ex.join(cand, left_on='m', right_on='entity_id', how='anti')):,}")
            per = (s1.select("entity_id", "country")
                     .join(ex.group_by("s1").agg(n=pl.len(), n_s2=pl.col("m").str.starts_with("S2-").sum(),
                                                 n_s3=pl.col("m").str.starts_with("S3-").sum()),
                           left_on="entity_id", right_on="s1", how="left").fill_null(0))
            summ = per.group_by("country").agg(
                s1=pl.len(), singleton_rate=(pl.col("n") == 0).mean(), mean_matches=pl.col("n").mean(),
                mean_s2=pl.col("n_s2").mean(), mean_s3=pl.col("n_s3").mean(),
                share_with_s2=(pl.col("n_s2") > 0).mean(), share_with_s3=(pl.col("n_s3") > 0).mean(),
                p50=pl.col("n").median(), p99=pl.col("n").quantile(0.99), max=pl.col("n").max()).sort("country")
            md.append("\n### Matches per S1\n" + md_table(summ))
            log("GT per-S1 stats done")
            dist = (per.with_columns(k=pl.when(pl.col("n") >= 10).then(pl.lit("10+")).otherwise(pl.col("n").cast(pl.String)))
                       .group_by("country", "k").len().with_columns(share=pl.col("len") / pl.col("len").sum().over("country"))
                       .pivot(on="country", index="k", values="share").sort(pl.col("k").str.replace("10\\+", "99").cast(pl.Int32)))
            md.append("\n### Distribution of #matches per S1 (share of S1)\n" + md_table(dist))
            combo = (per.group_by("n_s2", "n_s3").len().with_columns(share=pl.col("len") / n_s1)
                        .sort("len", descending=True).head(15))
            md.append("\n### Most common (n_S2, n_S3) combinations\n" + md_table(combo))
            # cross-country pairs
            cc = (ex.join(s1.select("entity_id", c1="country"), left_on="s1", right_on="entity_id")
                    .join(cand.select("entity_id", c2="country"), left_on="m", right_on="entity_id"))
            md.append(f"\n- True pairs with different country labels: {(cc['c1'] != cc['c2']).sum():,} "
                      f"of {len(cc):,} ({(cc['c1'] != cc['c2']).mean():.5%})")
            md.append("\n" + md_table(cc.group_by("c1", "c2").len().sort("len", descending=True)))
            # S2/S3 records that match nothing
            matched = ex.select(entity_id=pl.col("m").unique(), matched=pl.lit(True))
            um = (cand.select("entity_id", "country").join(matched, on="entity_id", how="left")
                      .with_columns(source=pl.col("entity_id").str.slice(0, 2), matched=pl.col("matched").fill_null(False))
                      .group_by("source", "country").agg(n=pl.len(), share_matched=pl.col("matched").mean())
                      .sort("source", "country"))
            md.append("\n### S2/S3 records that belong to some S1\n" + md_table(um))
            log("cross-country + unmatched done")
            # example true pairs per country
            md.append("\n## 30 random true pairs per country\n")
            full = (ex.join(s1.select("entity_id", "country", n1="business_name", a1="business_address"), left_on="s1", right_on="entity_id")
                      .join(cand.select("entity_id", n2="business_name", a2="business_address"), left_on="m", right_on="entity_id"))
            for c in sorted(full["country"].unique().to_list()):
                smp = full.filter(pl.col("country") == c).sample(30, seed=args.seed).select(
                    src=pl.col("m").str.slice(0, 2), s1_name="n1", s1_addr="a1", cand_name="n2", cand_addr="a2")
                md.append(f"\n### {c}\n" + md_table(smp))
            del s1, cand, gt, ex, per, cc, full
            log("train examples done")
        else:
            fr = pl.concat([f.filter(pl.col("country") == "France") for f in frames.values()])
            md.append("\n## France (test only, unlabelled)\n")
            md.append(f"- France records: {len(fr):,} ({fr.group_by(pl.col('entity_id').str.slice(0, 2)).len().sort('entity_id').rows()})")
            md.append(f"- Top address tokens: {top_tokens(fr['business_address'], 70)}")
            md.append(f"- Top name tokens: {top_tokens(fr['business_name'], 70)}")
            for src in ("S1", "S2", "S3"):
                smp = fr.filter(pl.col("entity_id").str.starts_with(src)).sample(12, seed=args.seed).select(
                    "entity_id", "business_name", "business_address")
                md.append(f"\n### France examples, {src}\n" + md_table(smp))
            for c in ("US", "India"):
                s = pl.concat([f.filter(pl.col("country") == c) for f in frames.values()])
                md.append(f"- Test {c} top address tokens: {top_tokens(s['business_address'], 40)}")
        del frames
        log("split done")

    md.insert(1, "## Rows per split / source / country\n" + md_table(
        pl.concat(sizes).pivot(on="country", index=["split", "source"], values="len").sort("split", "source", descending=[True, False])) + "\n")
    md.append("\n## Missingness and noise patterns (share of records)\n" + md_table(transpose_stats(pl.concat(stats))))
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
