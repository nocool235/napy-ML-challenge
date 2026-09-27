"""
Post-processing on a finished run's test probabilities. No retraining.

--em-prior: label-shift correction (Saerens, Latinne & Decaestecker, 2002). Test has more S2/S3 records that
match nothing than train, so the share of true pairs among test candidates can be lower than the share the
model was trained on. Per country, EM estimates the test positive rate from the model's own test
probabilities, starting from the training rate. Each probability is then re-weighted by the ratio of the two
priors, and the run's decision rule is applied unchanged. The same procedure runs for every country
(estimates are per country, like every other statistic in the pipeline; nothing branches on a country name).
The EM estimate is trustworthy only if the model is calibrated: on v5's out-of-fold probabilities it recovers
each train country's true positive rate within 0.002.

  python postprocess.py --run-dir output_v5 --norm-dir cache/real_prep/norm-0c9c1740ce --out-dir output_v5_em --em-prior
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import decide  # noqa: E402
import match  # noqa: E402
import prep  # noqa: E402
from utils import log  # noqa: E402


def em_prior(p: np.ndarray, pi_tr: float, iters: int = 500, tol: float = 1e-8) -> tuple[float, np.ndarray]:
    """EM estimate of the positive rate behind probabilities `p` from a model trained at rate `pi_tr`;
    returns (estimated rate, re-weighted probabilities)."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    pi = pi_tr
    for _ in range(iters):
        a = (pi / pi_tr) * p
        q = a / (a + ((1 - pi) / (1 - pi_tr)) * (1 - p))
        new = float(q.mean())
        done = abs(new - pi) < tol
        pi = new
        if done:
            break
    a = (pi / pi_tr) * p
    return pi, a / (a + ((1 - pi) / (1 - pi_tr)) * (1 - p))


def house_conflict_rate(feat: Path, probs: pl.DataFrame, ctry_of_i: np.ndarray, min_p: float = 0.9) -> dict:
    """Per country: share of confident mutual-best pairs (p >= min_p, best S1 for the record and best record for
    the S1) whose house numbers conflict, among pairs where both records carry one. Unsupervised: uses only the
    model's own probabilities, so it runs the same way on any split and any country."""
    F = pl.read_parquet(feat, columns=["i", "j", "hn_both", "hn_conflict"])
    if not ((F["i"] == probs["i"]).all() and (F["j"] == probs["j"]).all()):
        raise ValueError(f"{feat} rows do not align with the probabilities")
    d = F.with_columns(p=probs["p"], country=pl.Series(ctry_of_i[F["i"].to_numpy()]))
    bi = d.sort("p", descending=True).group_by("i", maintain_order=True).first().select("i", bj="j")
    bj = d.sort("p", descending=True).group_by("j", maintain_order=True).first().select("j", bi_="i")
    mb = (d.join(bi, on="i").filter(pl.col("j") == pl.col("bj")).join(bj, on="j").filter(pl.col("i") == pl.col("bi_"))
           .filter((pl.col("p") >= min_p) & (pl.col("hn_both") == 1)))
    return {r[0]: (float(r[1]), int(r[2]))
            for r in mb.group_by("country").agg(pl.col("hn_conflict").mean(), pl.len()).iter_rows()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="finished run: debug/test_probs.parquet, debug/metrics_train.json, "
                                                     "debug/train_oof.parquet, candidate_pairs.tsv")
    ap.add_argument("--norm-dir", required=True, help="normalisation cache the run used (row order of i/j)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--probs-dir", default="", help="where debug/test_probs.parquet and candidate_pairs.tsv come from "
                                                    "(e.g. a predict_test.py rescore); default: --run-dir")
    ap.add_argument("--em-prior", action="store_true", help="apply the per-country EM prior-shift correction")
    ap.add_argument("--house-lr", default="", help="train feature cache (feat-<key>/): where a country's confident "
                    "pairs show house-number conflicts less often than any training country did, conflicts are "
                    "stronger evidence than the model learned; scale conflicting pairs' odds by that ratio")
    ap.add_argument("--test-feat", default="", help="test feature cache file (with --house-lr)")
    ap.add_argument("--house-lr-scale", type=float, default=1.0,
                    help="multiply the house-number odds factor where it applies (< 1 = stronger); the rate ratio is "
                         "a conservative estimate since confident pairs under-count conflicts by 0.4-0.6x in train")
    ap.add_argument("--rule", default="", help="override the run's decision rule, e.g. expF:1.5 or thr:0.8 (one-to-one "
                                               "kept); one global rule for every record, chosen offline")
    a = ap.parse_args()

    run, out = Path(a.run_dir), Path(a.out_dir)
    m = json.load(open(run / "debug" / "metrics_train.json"))
    st2 = m.get("stage2")
    rule = st2["oof"] if st2 and st2["used"] else m["oof"]
    rule0 = rule
    if a.rule:
        kind, param = a.rule.split(":")
        rule = {"rule": kind, "param": float(param), "o2o": True}
    src = Path(a.probs_dir) if a.probs_dir else run
    tp = pl.read_parquet(src / "debug" / "test_probs.parquet")
    ti, tj, p = tp["i"].to_numpy(), tp["j"].to_numpy(), tp["p"].to_numpy()   # float32, as match.py decides
    s1, cand = match.load(Path(a.norm_dir), "test")
    if len(s1) <= ti.max() or len(cand) <= tj.max():
        raise SystemExit("test_probs indices exceed the norm cache: wrong --norm-dir")
    ctry = s1["country_norm"].to_numpy()[ti]
    # the rate the final models were trained at: every candidate pair of a random share of train S1s
    pi_tr = float(pl.read_parquet(run / "debug" / "train_oof.parquet", columns=["y"])["y"].mean())
    q, report = p.copy(), {}
    if a.em_prior:
        for c in np.unique(ctry):
            mk = ctry == c
            pi_c, qc = em_prior(p[mk], pi_tr)
            q[mk] = qc.astype(np.float32)
            report[str(c)] = {"train_rate": round(pi_tr, 4), "mean_p": round(float(p[mk].mean()), 4),
                              "em_test_rate": round(pi_c, 4)}
            log(f"{c}: train positive rate {pi_tr:.4f}, mean test p {p[mk].mean():.4f}, EM test rate {pi_c:.4f}")
    if a.house_lr:
        oof = pl.read_parquet(run / "debug" / "train_oof.parquet")
        s1tr = prep.load(Path(a.norm_dir), "train", ("source1",), ["country_norm"])["country_norm"].to_numpy()
        r_train = house_conflict_rate(Path(a.house_lr) / "train_features.parquet", oof, s1tr)
        r_test = house_conflict_rate(Path(a.test_feat), tp, s1["country_norm"].to_numpy())
        MIN_N, LR_MIN = 1000, 0.05          # trust a rate only from >= 1000 confident pairs; never zero out
        floor_c, (floor, _) = min(((c, v) for c, v in r_train.items() if v[1] >= MIN_N), key=lambda cv: cv[1][0])
        Fh = pl.read_parquet(a.test_feat, columns=["hn_conflict"])["hn_conflict"].to_numpy() == 1
        for c in np.unique(ctry):
            r_c, n_c = r_test.get(str(c), (floor, 0))
            lr = 1.0 if n_c < MIN_N else min(1.0, r_c / floor)
            lr = max(LR_MIN, lr * a.house_lr_scale) if lr < 1.0 else lr
            mk = (ctry == c) & Fh
            odds = q[mk].astype(np.float64) / np.clip(1 - q[mk].astype(np.float64), 1e-9, None) * lr
            q[mk] = (odds / (1 + odds)).astype(np.float32)
            report.setdefault(str(c), {}).update({"house_conflict_rate": round(r_c, 4), "confident_pairs": n_c,
                                                  "house_lr": round(lr, 4)})
            log(f"{c}: confident-pair house-conflict rate {r_c:.4f} (n={n_c:,}) vs lowest train {floor:.4f} "
                f"({floor_c}) -> odds x{lr:.3f} on {mk.sum():,} conflicting pairs")
    keep0, keep = decide.rule_mask(rule0, ti, tj, p), decide.rule_mask(rule, ti, tj, q)
    before, after = match.structure(s1, cand, ti[keep0], tj[keep0]), match.structure(s1, cand, ti[keep], tj[keep])
    for c in after:
        log(f"test {c}: before {before[c]}\n{'':>22}after  {after[c]}")
    log(f"pairs kept: before {keep0.sum():,}, after {keep.sum():,} ({keep.sum() - keep0.sum():+,}); "
        f"rule {{{rule['rule']}, {rule['param']}, o2o={rule['o2o']}}}")
    (out / "debug").mkdir(parents=True, exist_ok=True)
    match.write_lists(out / "matching_results.tsv", s1, cand, ti[keep], tj[keep], "matched_entity_ids")
    match.write_lists(out / "debug" / "before_matching_results.tsv", s1, cand, ti[keep0], tj[keep0], "matched_entity_ids")
    shutil.copy2(src / "candidate_pairs.tsv", out / "candidate_pairs.tsv")     # same candidates, same file
    pl.DataFrame({"i": ti, "j": tj, "p": q.astype(np.float32)}).write_parquet(out / "debug" / "test_probs.parquet")
    json.dump({"em": report, "rule": rule, "before": before, "after": after, "run": str(run)},
              open(out / "debug" / "postprocess.json", "w"), indent=2, default=float)
    log(f"wrote {out / 'matching_results.tsv'}")


if __name__ == "__main__":
    main()
