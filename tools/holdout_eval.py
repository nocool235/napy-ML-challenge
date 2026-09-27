"""
Honest before/after check for test-time changes (normalisation, retrieval) scored with an existing run's
FINAL models, without retraining.

A run's final models train on a random `train_frac` share of train S1 records (match.train_oof). The other
S1 records are never seen by them, so scoring those S1s with the final models (stage 1 -> stage 2 -> the
run's rule, exactly as on test) is an out-of-sample estimate. Everything is scoped by country, so one
country can be rebuilt alone at full density.

  baseline (the run's cached train features):
    python tools/holdout_eval.py --run-dir output_v5 --norm-dir cache/real_prep/norm-0c9c1740ce \
        --pair-cache cache/real_prep/pairs-460b9301bf --feat-cache cache/real_prep/feat-e26f60588d --countries india
  variant (rebuild candidates + features from another normalisation cache, same retrieval settings/pruner):
    python tools/holdout_eval.py --run-dir output_v5 --norm-dir cache/real_prep/norm-<new> \
        --pair-cache cache/real_prep/pairs-460b9301bf --rebuild --countries india --work work_eval
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

SRC = Path(__file__).resolve().parents[1] / "code" / "business_entity_resolution" / "src"
sys.path.insert(0, str(SRC))
import decide  # noqa: E402
import match  # noqa: E402
import predict_test  # noqa: E402
import stage2  # noqa: E402
from build_token_maps import final_train_s1  # noqa: E402
from utils import log  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--norm-dir", required=True)
    ap.add_argument("--pair-cache", required=True, help="the run's pairs-<key> dir (pruner)")
    ap.add_argument("--feat-cache", default="", help="the run's feat-<key> dir (baseline: cached train features)")
    ap.add_argument("--rebuild", action="store_true", help="rebuild candidates + features from --norm-dir")
    ap.add_argument("--countries", default="india")
    ap.add_argument("--work", default="work_eval")
    ap.add_argument("--drop-s1-frac", type=float, default=0.0,
                    help="remove this share of S1 records but keep all S2/S3 (their records become orphans, as in test: "
                         "5.75 S2/S3 per S1 vs train's 4.68 = train with ~20%% of S1 removed)")
    ap.add_argument("--rule-grid", action="store_true", help="also score a grid of decision rules on the same p")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    t0 = time.time()
    run, norm = Path(a.run_dir), Path(a.norm_dir)
    m = json.load(open(run / "debug" / "metrics_train.json"))
    kinds, w, rule = m["models"], m["blend_weight"], m["oof"]
    extra_k = m.get("extra_k") or {"char": m.get("char_k", 0)}
    extra_max_df = m.get("extra_max_df") or {"char": m.get("char_max_df", 0.01)}
    models1, models2, w2 = predict_test.load_models(run / "debug", kinds), None, None
    st2 = m.get("stage2")
    if st2 and st2["used"]:
        models2, w2, rule = predict_test.load_models(run / "debug" / "stage2", kinds), st2["blend_weight"], st2["oof"]
    countries = [c.strip() for c in a.countries.split(",") if c.strip()]

    s1, cand = match.load(norm, "train")
    holdout_full = ~final_train_s1(len(s1), frac=m["train_frac"], folds=m["folds"])
    gt_full = match.load_truth(Path(a.data_dir), s1, cand)
    if a.drop_s1_frac > 0:
        if not a.rebuild:
            raise SystemExit("--drop-s1-frac needs --rebuild (retrieval and context must not see the dropped S1s)")
        drop = np.random.default_rng(123).random(len(s1)) < a.drop_s1_frac
        s1 = s1.filter(pl.Series(~drop))
        log(f"dropped {int(drop.sum()):,} S1 records ({a.drop_s1_frac:.0%}); their S2/S3 records stay as orphans")
    if a.rebuild:
        s1 = s1.filter(pl.col("country_norm").is_in(countries)).rename({"i": "i0"}).with_row_index("i")
        cand = cand.filter(pl.col("country_norm").is_in(countries)).rename({"j": "j0"}).with_row_index("j")
        gt = (gt_full.join(s1.select("i", "i0"), left_on="i", right_on="i0").select(pl.col("i_right").alias("i"), "j")
                     .join(cand.select("j", "j0"), left_on="j", right_on="j0").select("i", pl.col("j_right").alias("j")))
        holdout = holdout_full[s1["i0"].to_numpy()]
        pc = Path(a.work) / "pairs"
        pc.mkdir(parents=True, exist_ok=True)
        for f in ("pruner.txt", "pruner.json"):
            shutil.copy2(Path(a.pair_cache) / f, pc / f)
        pairs = match.candidates(s1, cand, "train", pc, m["tfidf_k"], None, m["prune_recall"], False,
                                 extra_k, extra_max_df)
        F = match.pair_features(pairs, s1, cand)
        del pairs
    else:
        keep_i = s1.filter(pl.col("country_norm").is_in(countries))["i"]
        F = pl.scan_parquet(Path(a.feat_cache) / "train_features.parquet").filter(
            pl.col("i").is_in(keep_i.implode())).collect()
        gt = gt_full.filter(pl.col("i").is_in(keep_i.implode()))
        holdout = holdout_full
    in_c = np.isin(s1["country_norm"].to_numpy(), countries)
    ti, tj = F["i"].to_numpy(), F["j"].to_numpy()
    y = (F.select("i", "j").join(gt.with_columns(y=pl.lit(1, pl.Int8)), on=["i", "j"], how="left")["y"]
         .fill_null(0).to_numpy())
    p = predict_test.predict(models1, F, w)
    if models2 is not None:
        G = stage2.stage2_features(pl.DataFrame({"i": ti, "j": tj}), p, cand)
        p = predict_test.predict(models2, pl.concat([F, G], how="horizontal"), w2)
        del G
    del F
    keep = decide.rule_mask(rule, ti, tj, p)
    n_true = np.bincount(gt["i"].to_numpy(), minlength=len(s1))
    f = decide.per_s1_f05(ti[keep], y[keep], len(s1), n_true)
    oracle = decide.per_s1_f05(ti[y == 1], y[y == 1], len(s1), n_true)
    sel = holdout & in_c
    ret = np.bincount(ti[y == 1], minlength=len(s1))
    res = {"countries": countries, "rebuild": a.rebuild, "norm": str(norm), "holdout_s1": int(sel.sum()),
           "pairs": int(len(ti)), "pairs_per_s1": round(len(ti) / int(in_c.sum()), 3),
           "recall_holdout": round(float(ret[sel].sum() / max(n_true[sel].sum(), 1)), 5),
           "oracle_f05_holdout": round(float(oracle[sel].mean()), 5),
           "f05_holdout": round(float(f[sel].mean()), 5),
           "f05_holdout_singletons": round(float(f[sel & (n_true == 0)].mean()), 5),
           "f05_holdout_matched": round(float(f[sel & (n_true > 0)].mean()), 5),
           "f05_train_part (in-sample)": round(float(f[~holdout & in_c].mean()), 5),
           "seconds": round(time.time() - t0)}
    if a.rule_grid:
        grid = ([{"rule": "expF", "param": x, "o2o": True} for x in (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0)]
                + [{"rule": "thr", "param": x, "o2o": True} for x in (0.5, 0.6, 0.67, 0.7, 0.8, 0.9, 0.95)])
        res["rules"] = []
        for r in grid:
            k = decide.rule_mask(r, ti, tj, p)
            fr = decide.per_s1_f05(ti[k], y[k], len(s1), n_true)
            res["rules"].append({**r, "f05_holdout": round(float(fr[sel].mean()), 5), "pairs_kept": int(k.sum()),
                                 "precision": round(float(y[k].mean()), 5)})
            log(f"rule {r}: F0.5 holdout {fr[sel].mean():.5f}, kept {k.sum():,}, pair precision {y[k].mean():.4f}")
    log(json.dumps(res))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
