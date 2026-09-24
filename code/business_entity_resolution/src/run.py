"""
End-to-end pipeline:  data -> normalise -> blocking -> features -> 2-stage LightGBM
                      -> decision rule tuned on OOF F0.5 -> output/*.tsv

Usage (from code/business_entity_resolution/):
    python src/run.py --data-dir ../../dataset --out-dir ../../output
    python src/run.py --data-dir ../../dataset --cv country      # leave-one-country-out check
    python src/run.py --data-dir ../../dataset --validate-only   # train CV only, no test outputs
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from blocking import fit_vectorizers, transform, generate_candidates, blocking_report  # noqa: E402
from decide import tune, apply, to_pred  # noqa: E402
from features import pair_features, stage2_features, feature_cols  # noqa: E402
from metric import breakdown  # noqa: E402
from model import make_folds, cv_oof, fit_full  # noqa: E402
from normalize import normalize_frame  # noqa: E402


# ----------------------------------------------------------------------------- io
def read_tsv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, quoting=3)


def load_split(data_dir: Path, split: str):
    d = data_dir / split
    s1 = read_tsv(d / f"{split}_source1.tsv")
    s2 = read_tsv(d / f"{split}_source2.tsv")
    s3 = read_tsv(d / f"{split}_source3.tsv")
    truth = None
    gt = d / f"{split}_ground_truth.tsv"
    if gt.exists():
        g = read_tsv(gt)
        truth = {r.source1_entity_id: {x.strip() for x in r.matched_entity_ids.split(",") if x.strip()}
                 for r in g.itertuples()}
    return s1, pd.concat([s2, s3], ignore_index=True), truth


def write_lists(path: Path, s1_ids, lists: dict, col: str):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"source1_entity_id\t{col}\n")
        for k in s1_ids:
            fh.write(f"{k}\t{','.join(sorted(lists.get(k, ())))}\n")


def self_validate(match_p: Path, cand_p: Path, s1_ids, c_ids):
    """Mirror of the official rules (the official utils/validate_submission.py is the authority)."""
    valid_c, s1_set = set(c_ids), set(s1_ids)
    issues = []
    for p, col in ((match_p, "matched_entity_ids"), (cand_p, "candidate_entity_ids")):
        df = read_tsv(p)
        if list(df.columns) != ["source1_entity_id", col]:
            issues.append(f"{p.name}: bad header {list(df.columns)}")
        if df["source1_entity_id"].duplicated().any():
            issues.append(f"{p.name}: duplicate S1 rows")
        if set(df["source1_entity_id"]) != s1_set:
            issues.append(f"{p.name}: S1 id set differs from test_source1")
        for v in df[col]:
            ids = [x for x in v.split(",") if x]
            if len(ids) != len(set(ids)):
                issues.append(f"{p.name}: duplicate id in list"); break
            if any(x not in valid_c for x in ids):
                issues.append(f"{p.name}: unknown / non S2-S3 id"); break
    m, c = read_tsv(match_p), read_tsv(cand_p)
    cm = dict(zip(c["source1_entity_id"], c["candidate_entity_ids"].map(lambda v: set(filter(None, v.split(","))))))
    if any(not set(filter(None, v.split(","))) <= cm.get(k, set()) for k, v in zip(m["source1_entity_id"], m["matched_entity_ids"])):
        issues.append("matches not a subset of candidates")
    print("[validate] PASS" if not issues else "[validate] ISSUES:\n  " + "\n  ".join(issues))
    return not issues


# ----------------------------------------------------------------------------- stages
def build_split(split, s1_raw, c_raw, vecs, name_freq, cache: Path | None):
    cp = cache / f"{split}_pairs.pkl" if cache else None
    if cp and cp.exists():
        print(f"[cache] loading {cp}")
        return pickle.load(open(cp, "rb"))
    s1, cand = normalize_frame(s1_raw), normalize_frame(c_raw)
    X1, XC = transform(vecs, s1), transform(vecs, cand)
    pairs = generate_candidates(s1, cand, X1, XC)
    F = pair_features(pairs, s1, cand, X1, XC, name_freq)
    out = (s1, cand, pairs, F)
    if cp:
        cp.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(out, open(cp, "wb"), protocol=4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--cache-dir", default=str(config.CACHE_DIR))
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--cv", choices=["group", "country"], default="group")
    ap.add_argument("--validate-only", action="store_true", help="CV on train only; skip test")
    ap.add_argument("--no-stage2", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dbg = out_dir / "debug"; dbg.mkdir(exist_ok=True)
    cache = None if args.no_cache else Path(args.cache_dir)

    tr_s1, tr_c, truth = load_split(data_dir, "train")
    te_s1, te_c, _ = (None, None, None) if args.validate_only else load_split(data_dir, "test")
    print(f"[data] train S1={len(tr_s1):,} S2+S3={len(tr_c):,}"
          + ("" if te_s1 is None else f" | test S1={len(te_s1):,} S2+S3={len(te_c):,}"))
    print("[data] train countries:", tr_s1["country"].value_counts().to_dict())
    if te_s1 is not None:
        print("[data] test countries:", te_s1["country"].value_counts().to_dict())

    # sanity: does any S2/S3 id appear under two S1 records? (justifies the one-to-one rule)
    all_m = [x for v in truth.values() for x in v]
    print(f"[data] matched ids={len(all_m):,}, unique={len(set(all_m)):,}, "
          f"singletons={np.mean([len(v) == 0 for v in truth.values()]):.1%}")

    # vectorisers are fitted unsupervised on every record we have (no labels involved)
    vec_p = cache / "vecs.pkl" if cache else None
    if vec_p and vec_p.exists():
        vecs, name_freq = pickle.load(open(vec_p, "rb"))
    else:
        frames = [normalize_frame(f) for f in (tr_s1, tr_c, te_s1, te_c) if f is not None]
        vecs = fit_vectorizers(frames)
        name_freq = pd.concat([f["name_core"] for f in frames]).value_counts().to_dict()
        if vec_p:
            vec_p.parent.mkdir(parents=True, exist_ok=True)
            pickle.dump((vecs, name_freq), open(vec_p, "wb"))

    # ---------------- train
    s1, cand, pairs, F = build_split("train", tr_s1, tr_c, vecs, name_freq, cache)
    s1_ids, c_ids = s1["entity_id"].to_numpy(dtype=object), cand["entity_id"].to_numpy(dtype=object)
    truth = {k: truth.get(k, set()) for k in s1_ids}
    rep = blocking_report(pairs, s1, cand, truth)
    print("[blocking] " + json.dumps({k: round(v, 4) for k, v in rep.items()}))

    i, j = F["i"].values, F["j"].values
    y = np.array([c_ids[b] in truth[s1_ids[a]] for a, b in zip(i, j)], np.int8)
    print(f"[train] pairs={len(y):,} positives={y.sum():,} ({y.mean():.2%})")

    fold = make_folds(s1_ids[i], args.cv, s1["country"].to_numpy(dtype=object)[i])
    cols = feature_cols(F)
    oof1, n1, imp1 = cv_oof(F[cols], y, fold, "stage1")
    imp1.to_csv(dbg / "importance_stage1.csv")

    groups = dict(zip(s1_ids, s1["country"].to_numpy(dtype=object)))
    best1, _ = tune(i, j, oof1, truth, s1_ids, c_ids, verbose=False)
    print(f"[stage1] OOF best rule {best1}")
    p_oof, best, stage2 = oof1, best1, not args.no_stage2
    if stage2:
        G = stage2_features(F, oof1)
        cols2 = feature_cols(G)
        oof2, n2, imp2 = cv_oof(G[cols2], y, fold, "stage2")
        imp2.to_csv(dbg / "importance_stage2.csv")
        best2, table = tune(i, j, oof2, truth, s1_ids, c_ids)
        print(f"[stage2] OOF best rule {best2}")
        if best2["f05"] >= best1["f05"]:
            p_oof, best = oof2, best2
        else:
            stage2 = False
            print("[stage2] not better than stage1 -> using stage1")
        table.to_csv(dbg / "decision_rules.csv", index=False)

    keep = apply(best, i, j, p_oof)
    pred = to_pred(i, j, keep, s1_ids, c_ids)
    bd = breakdown(pred, truth, s1_ids, groups)
    print("[OOF] " + json.dumps({k: round(v, 4) for k, v in bd.items()}))

    # error dump for analysis (false positives / false negatives, with raw text)
    E = pd.DataFrame({"s1": s1_ids[i], "cand": c_ids[j], "p": p_oof, "y": y, "pred": keep.astype(int)})
    E = E[(E["y"] != E["pred"])]
    E = E.merge(s1[["entity_id", "business_name", "business_address", "country"]].add_prefix("s1_"),
                left_on="s1", right_on="s1_entity_id").merge(
        cand[["entity_id", "business_name", "business_address", "country"]].add_prefix("c_"),
        left_on="cand", right_on="c_entity_id")
    E.drop(columns=["s1_entity_id", "c_entity_id"]).sort_values("p", ascending=False).to_csv(
        dbg / "oof_errors.tsv", sep="\t", index=False)
    # true matches that blocking never proposed
    got = set(zip(s1_ids[i], c_ids[j]))
    miss = [(k, m) for k, v in truth.items() for m in v if (k, m) not in got]
    pd.DataFrame(miss, columns=["s1", "cand"]).to_csv(dbg / "blocking_misses.tsv", sep="\t", index=False)

    metrics = {"blocking": rep, "oof": bd, "rule": best, "stage2": stage2, "cv": args.cv}
    json.dump(metrics, open(dbg / "metrics_train.json", "w"), indent=2, default=float)
    if args.validate_only:
        print(f"[done] validate-only in {time.time() - t0:.0f}s")
        return

    # ---------------- test
    ts1, tcand, tpairs, TF = build_split("test", te_s1, te_c, vecs, name_freq, cache)
    ts1_ids, tc_ids = ts1["entity_id"].to_numpy(dtype=object), tcand["entity_id"].to_numpy(dtype=object)
    ti, tj = TF["i"].values, TF["j"].values
    m1 = fit_full(F[cols], y, n1)
    tp = m1.predict(TF[cols])
    if stage2:
        m2 = fit_full(G[cols2], y, n2)
        tp = m2.predict(stage2_features(TF, tp)[cols2])
    tkeep = apply(best, ti, tj, tp)
    tpred = to_pred(ti, tj, tkeep, ts1_ids, tc_ids)
    tcands = to_pred(ti, tj, np.ones(len(ti), bool), ts1_ids, tc_ids)

    mp, cp = out_dir / "matching_results.tsv", out_dir / "candidate_pairs.tsv"
    write_lists(mp, ts1_ids, tpred, "matched_entity_ids")
    write_lists(cp, ts1_ids, tcands, "candidate_entity_ids")
    pd.DataFrame({"s1": ts1_ids[ti], "cand": tc_ids[tj], "p": tp}).to_csv(dbg / "test_probs.tsv", sep="\t", index=False)
    ok = self_validate(mp, cp, ts1_ids, tc_ids)

    share_empty = 1 - len(tpred) / len(ts1_ids)
    by_c = pd.Series({k: len(tpred.get(k, ())) > 0 for k in ts1_ids}).groupby(ts1["country"].to_numpy(dtype=object)).mean()
    print(f"[test] S1 with >=1 match: {1 - share_empty:.1%} (train truth: "
          f"{1 - bd['share_singletons']:.1%}); by country: {by_c.round(3).to_dict()}")
    # submission log - keep one line per leaderboard upload
    with open(out_dir / "submission_log.tsv", "a") as fh:
        fh.write("\t".join(map(str, [time.strftime("%Y-%m-%d %H:%M"), args.cv, stage2, best["rule"],
                                     best["param"], best["o2o"], round(bd["f05"], 4), round(rep["pair_recall"], 4),
                                     ok, "LB=?"])) + "\n")
    print(f"[done] wrote {mp} and {cp} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
