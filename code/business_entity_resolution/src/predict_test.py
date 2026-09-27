"""
Test-only scoring with an already-trained run. No training.

Loads <run>/debug/metrics_train.json (decision rule, blend weights, retrieval settings, feature version) and
the saved stage-1 models (plus the stage-2 models when the run kept stage 2). It builds the test candidates
and features with the retrieval settings the run trained with, or loads them from the run's caches. Then it
scores them, applies the run's rule, and writes matching_results.tsv, candidate_pairs.tsv and per-pair
probabilities (debug/test_probs.parquet by row index; debug/test_probs_ids.parquet with IDs and country).

--countries scores only those countries. Retrieval, the pruner's context features, the pair features and
the decision rule are all scoped by country, so a country scored alone gets the same rows it gets inside a
full run. With --splice-into, those countries' rows replace the same S1 rows of an existing submission;
every other line is copied byte for byte.

  python predict_test.py --run-dir output_v5 --norm-dir cache/real_prep/norm-0c9c1740ce --out-dir output_v5_rescore
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import decide  # noqa: E402
import match  # noqa: E402
import stage2  # noqa: E402
from utils import log  # noqa: E402


def load_models(d: Path, kinds: list[str]) -> dict:
    """{kind: (booster, feature names in training order)} from a run's debug dir or its stage2/ subdir."""
    out = {}
    for k in kinds:
        if k == "lgb":
            b = lgb.Booster(model_file=str(d / "model_lgb.txt"))   # saved at its best iteration
            out[k] = (b, b.feature_name())
        else:
            import xgboost as xgb
            b = xgb.Booster()
            b.load_model(str(d / "model_xgb.json"))                # keeps best_iteration as an attribute
            out[k] = (b, list(b.feature_names))
    return out


def predict(models: dict, F: pl.DataFrame, w: float, chunk: int = 2_000_000) -> np.ndarray:
    """Blend of the models' probabilities, in row chunks: one 18M-row XGBoost call on the GPU needs ~9 GB of
    VRAM for the matrix alone and spills into (very slow) shared memory on a 16 GB card."""
    ps = {}
    for k, (b, names) in models.items():
        missing = [n for n in names if n not in F.columns]
        if missing:
            raise ValueError(f"{k} model expects features the frame lacks: {missing[:10]}")
        out = np.empty(len(F), np.float32)
        for s in range(0, len(F), chunk):
            X = F.slice(s, chunk).select(pl.col(names).cast(pl.Float32)).to_numpy()
            if k == "lgb":
                out[s:s + len(X)] = b.predict(X)
            else:
                import xgboost as xgb
                out[s:s + len(X)] = b.predict(xgb.DMatrix(X, feature_names=names),
                                              iteration_range=(0, b.best_iteration + 1))
            del X
        ps[k] = out
    return match.blend(ps, w)


def coverage(s1: pl.DataFrame, cand: pl.DataFrame, ti: np.ndarray, tj: np.ndarray) -> dict:
    """Candidate-set shape per country: S1 with >= 1 candidate, candidates per S2/S3 record."""
    ctry, cc = s1["country_norm"].to_numpy(), cand["country_norm"].to_numpy()
    n_i = np.bincount(ti, minlength=len(s1))
    n_j = np.bincount(tj, minlength=len(cand))
    return {c: {"s1_with_candidates": round(float((n_i[ctry == c] > 0).mean()), 4),
                "candidates_per_s2s3": round(float(n_j[cc == c].mean()), 3)} for c in np.unique(ctry)}


def splice(base: Path, new: Path, out: Path):
    """Replace base's lines whose first field (source1_entity_id) appears in `new`; copy every other line."""
    repl = {}
    for line in new.read_bytes().split(b"\n")[1:]:
        if line:
            repl[line.split(b"\t", 1)[0]] = line
    lines = base.read_bytes().split(b"\n")
    hits = 0
    for k in range(1, len(lines)):
        key = lines[k].split(b"\t", 1)[0]
        if lines[k] and key in repl:
            lines[k] = repl[key]
            hits += 1
    if hits != len(repl):
        raise ValueError(f"{new.name}: {len(repl) - hits} scored S1 ids not found in {base}")
    out.write_bytes(b"\n".join(lines))
    log(f"spliced {hits:,} rows into {out} (all other lines copied from {base})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="the trained run's output dir (debug/metrics_train.json, models)")
    ap.add_argument("--norm-dir", required=True,
                    help="normalisation cache the run trained on (named in its log), e.g. cache/real_prep/norm-0c9c1740ce")
    ap.add_argument("--pair-cache", default="", help="override the pairs-<key> dir (default: derived like match.py)")
    ap.add_argument("--feat-cache", default="", help="override the feat-<key> dir (default: derived like match.py)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--countries", default="", help="comma list: score only these countries")
    ap.add_argument("--splice-into", default="", help="dir whose matching_results.tsv / candidate_pairs.tsv get "
                                                      "the scored countries' rows; other lines kept byte for byte")
    ap.add_argument("--no-cache", action="store_true", help="rebuild test pairs and features even if cached")
    ap.add_argument("--stage1-only", action="store_true", help="ignore the run's stage 2: stage-1 models and rule")
    a = ap.parse_args()

    t0 = time.time()
    run, norm, out = Path(a.run_dir), Path(a.norm_dir), Path(a.out_dir)
    m = json.load(open(run / "debug" / "metrics_train.json"))
    kinds, w, rule = m["models"], m["blend_weight"], m["oof"]
    if "extra_k" in m:
        extra_k, extra_max_df = m["extra_k"], m["extra_max_df"]
    else:   # runs started before EXTRA_VIEWS recorded only the char view (their cache keys differ too: pass
        extra_k = {"char": m.get("char_k", 0)}          # --pair-cache/--feat-cache from the run's log)
        extra_max_df = {"char": m.get("char_max_df", 0.01)}
    models1 = load_models(run / "debug", kinds)
    models2, w2 = None, None
    st2 = m.get("stage2")
    if st2 and st2["used"] and not a.stage1_only:
        models2, w2, rule = load_models(run / "debug" / "stage2", kinds), st2["blend_weight"], st2["oof"]
    pcache, fcache = match.cache_dirs(norm.parent, norm, m["tfidf_k"], m["prune_recall"], extra_k, extra_max_df)
    pcache = Path(a.pair_cache) if a.pair_cache else pcache
    fcache = Path(a.feat_cache) if a.feat_cache else fcache
    if not (pcache / "pruner.json").exists():
        raise SystemExit(f"no pruner in {pcache}: wrong --norm-dir or cache override?")
    log(f"run {run}: models {kinds} (w={w}), stage 2 {'used (w2=' + str(w2) + ')' if models2 else 'not used'}, "
        f"rule {{{rule['rule']}, {rule['param']}, o2o={rule['o2o']}}}; pairs {pcache}; features {fcache}")

    s1, cand = match.load(norm, "test")
    countries = [c.strip() for c in a.countries.split(",") if c.strip()]
    if countries:
        s1 = s1.filter(pl.col("country_norm").is_in(countries)).drop("i").with_row_index("i")
        cand = cand.filter(pl.col("country_norm").is_in(countries)).drop("j").with_row_index("j")
        log(f"scoring {countries} only: S1={len(s1):,} S2/S3={len(cand):,}")
    use_cache = not a.no_cache and not countries        # the run's caches hold every country, row-indexed
    work = out / "cache"
    if use_cache and (fcache / "test_features.parquet").exists():
        log(f"loading cached test features {fcache / 'test_features.parquet'}")
        F = pl.read_parquet(fcache / "test_features.parquet")
    else:
        if m["feature_version"] != match.FEATURE_VERSION:     # rebuilt features must be the ones it trained on
            raise SystemExit(f"run trained on features {m['feature_version']}, code builds {match.FEATURE_VERSION}")
        if use_cache and (pcache / "test_pairs.parquet").exists():
            log(f"loading cached test pairs {pcache / 'test_pairs.parquet'}")
            pairs = pl.read_parquet(pcache / "test_pairs.parquet")
        else:
            pc = work / "pairs"                     # never write into the run's own cache
            pc.mkdir(parents=True, exist_ok=True)
            for f in ("pruner.txt", "pruner.json"):
                shutil.copy2(pcache / f, pc / f)
            pairs = match.candidates(s1, cand, "test", pc, m["tfidf_k"], None, m["prune_recall"], False,
                                     extra_k, extra_max_df)
        F = match.pair_features(pairs, s1, cand)
        del pairs
        match._write(F, work / "test_features.parquet")
    ti, tj = F["i"].to_numpy(), F["j"].to_numpy()
    tp = predict(models1, F, w)
    if models2 is not None:
        G = stage2.stage2_features(pl.DataFrame({"i": ti, "j": tj}), tp, cand)
        tp = predict(models2, pl.concat([F, G], how="horizontal"), w2)
        del G
    del F
    keep = decide.rule_mask(rule, ti, tj, tp)

    dst = out / "subset" if a.splice_into else out
    (dst / "debug").mkdir(parents=True, exist_ok=True)
    match.write_lists(dst / "matching_results.tsv", s1, cand, ti[keep], tj[keep], "matched_entity_ids")
    match.write_lists(dst / "candidate_pairs.tsv", s1, cand, ti, tj, "candidate_entity_ids")
    pl.DataFrame({"i": ti, "j": tj, "p": tp}).write_parquet(dst / "debug" / "test_probs.parquet")
    (pl.DataFrame({"i": ti.astype(np.uint32), "j": tj.astype(np.uint32), "p": tp})
       .join(s1.select("i", s1_id="entity_id", country="country_norm"), on="i")
       .join(cand.select("j", cand_id="entity_id"), on="j")
       .select("s1_id", "cand_id", "country", "p").write_parquet(dst / "debug" / "test_probs_ids.parquet"))
    st, cov = match.structure(s1, cand, ti[keep], tj[keep]), coverage(s1, cand, ti, tj)
    base_st = m.get("test_structure", {})
    for c in st:
        log(f"test {c}: {{**{st[c]}, **{cov[c]}}}" + (f"  | run's own: {base_st[c]}" if c in base_st else ""))
    json.dump({"structure": st, "coverage": cov, "rule": rule, "run": str(run)},
              open(dst / "debug" / "rescore.json", "w"), indent=2, default=float)
    if a.splice_into:
        base = Path(a.splice_into)
        for fn in ("matching_results.tsv", "candidate_pairs.tsv"):
            splice(base / fn, dst / fn, out / fn)
    log(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
