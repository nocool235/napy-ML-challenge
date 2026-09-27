# Amazon ML Challenge 2026 — Business Entity Resolution

Match Source 2 / Source 3 business records to the Source 1 record describing the same business, across US,
India and (test-only) France. Scored as macro `F0.5` per S1 record. Engineering detail, the flowchart and the
bottleneck list are in **[`ARCHITECTURE.md`](ARCHITECTURE.md)**.

**Deadline: Sunday 27 Sep 2026, 23:59 IST.**

> Reflects the working tree as of 2026-09-27. This README and `ARCHITECTURE.md` are updated together whenever
> the code changes; see the changelog at the bottom.

## Current state (27 Sep, 22:15 IST): final pipeline

Team **napy** (Adarsh Bhatt, Prakhar Saxena, Yash Goel, Nakul Verma). Single main line; the final pipeline is
`src/match.py` + `src/postprocess.py`:

```
prep.py        normalise → parquet cache (Indic transliteration, skeletons, canonical names/addresses)
match.py       retrieval.py       hash-join keys + word TF-IDF top-10 + name char n-gram TF-IDF top-8, per country
               cascade_pruner.py  LightGBM on retrieval features; stricter cut (score ≥ 0.01): 7.8 candidates per S1 on test
               match.py           102 pair features → XGBoost on all train entities (5-fold OOF) → stage 2 (+21 features)
postprocess.py expected-F0.5 rule, one-to-one, α 1.5 (chosen on a test-like train rebuild)
               + house-number likelihood correction where a country's noise rate is below every training country's
               → output/matching_results.tsv, output/candidate_pairs.tsv
```

| Upload | Files | OOF test-mix | Public LB | What changed |
|---|---|---|---|---|
| 1 | `output_v2/` | 0.949 | 0.9365 | hash keys + word TF-IDF + pruner, LightGBM |
| 2 | `output_v3/` | 0.957 | 0.947 | de-noised features (source-noise tokens) |
| 3 | `output_v5/` | 0.975 | 0.952557 | char n-gram view, legal-form feature, stage 2, LightGBM+XGBoost |
| 4 | `output_v5_fix15_hlr/` | – | **0.958** | float64 rule fix, α 1.5, France house-number correction |
| 5 | `output_v6_fix15_hlr/` | 0.975 (lean set) | [pending] | v6: 27% fewer candidates, XGBoost on all entities |

"Test-mix" is the train out-of-fold score with India and US weighted as in the test set (0.47/0.38). France has no
labels, so the leaderboard is the only France measurement. Details: `EXPERIMENTS.md`; methodology:
`Documentation_template.md`; engineering: `ARCHITECTURE.md`; upload log: `submission_log.tsv`.

## Reproduce

```bash
cd code/business_entity_resolution && pip install -r requirements.txt && cd ../..
# dataset/ (organiser data) and utils/validate_submission.py (from the student_resource zip) must be in place

# 1. normalise once (resumable; ~12 min)
python code/business_entity_resolution/src/prep.py --data-dir dataset --work-dir cache/real_prep

# 2. main pipeline: train OOF + test outputs (step 2 settings are the defaults)
python code/business_entity_resolution/src/match.py --data-dir dataset --work-dir cache/real_prep --out-dir output
#    --tfidf-k 0          hash keys only (the step-1 run)
#    --char-k/--addr-k/--ns-char-k   extra retrieval views (name char n-grams / addr words / no-space name), off by default
#    --prune-recall R     share of retrieved true pairs the pruner keeps (default 0.997)
#    --models lgb,xgb     also train XGBoost and blend (needs `pip install xgboost`)
#    --device cuda        run XGBoost on the GPU (CPU is the default)
#    --stage2             second-stage model (kept only if OOF improves; NOT validated on LOCO, see EXPERIMENTS.md)
#    --cv country         leave-one-country-out France proxy (no test outputs)
#    --validate-only      train + OOF only;  --no-cache  recompute pairs and features
# pairs and features are cached under cache/real_prep/pairs-*/ and feat-*/, keyed on hand-bumped version
# constants (not a source hash) plus config, and reused automatically

# 2b. THE FINAL SUBMISSION (v6): lean candidate set, XGBoost on all entities, then post-processing
python code/business_entity_resolution/src/match.py --data-dir dataset --work-dir cache/real_prep --out-dir output_v6 \
    --tfidf-k 10 --char-k 8 --char-max-df 0.01 --prune-threshold 0.01 --models xgb --device cuda --train-frac 1.0 --stage2
python code/business_entity_resolution/src/postprocess.py --run-dir output_v6 --norm-dir cache/real_prep/norm-<key> \
    --out-dir output --rule expF:1.5 --house-lr cache/real_prep/feat-<key> \
    --test-feat cache/real_prep/feat-<key>/test_features.parquet
#    <key>: the cache names match.py prints in its first log line
#    --rule: one global decision rule (chosen on a train rebuild with 20% of S1 removed, like test)
#    --house-lr: per-country house-number likelihood correction, only below the lowest training-country rate

# 3. validate, then package
python utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
python tools/package_submission.py --team <TEAMNAME> --doc Documentation_template.md --output-dir output

# fallback: crude no-ML baseline (6 min)
python tools/crude_baseline.py --work-dir cache/real_prep --data-dir dataset --out-dir output_crude
```

On Windows set `PYTHONIOENCODING=utf-8`, or a console print of a polars table can crash.

## Repo layout

```
code/business_entity_resolution/src/
  prep.py, normalize.py, translit.py, utils.py   normalisation → parquet cache
  retrieval.py        hash-join + TF-IDF candidates (plus experimental FAISS ANN, unused)
  cascade_pruner.py   trained candidate pruner
  match.py            MAIN pipeline driver: features, LightGBM/XGBoost OOF + blend, decision, outputs
  stage2.py           second-stage features over stage-1 probabilities (--stage2)
  decide.py, metric.py   decision rules (dict-based + vectorised) and the exact metric
  run.py, blocking.py, features.py, model.py, config.py   legacy sample-scale pipeline
  synthetic.py        dev-only fake data
tools/                crude baseline, retrieval benchmarks, profiling, submission packager
EXPERIMENTS.md        one row per change; only `full` rows predict the leaderboard
ARCHITECTURE.md       flowchart, tech stack, modules, bottlenecks
```

Gitignored: `dataset/`, `utils/`, `cache/`, `work/`, `output*/`.

## Next

Per the agreed plan (`ARCHITECTURE.md` §7):
1. GPU XGBoost behind a flag, plus a separate pair cache.
2. A cheap feature batch, then submit.
3. A second-stage model on the GPU.
4. In parallel: France proxy mode, retrieval-miss analysis, and a lighter reproduction mode for the organisers' 16-CPU limit.

## Rule compliance
No external data, APIs or geocoding; every dictionary is hand-written. LightGBM (MIT), scikit-learn (BSD-3),
rapidfuzz (MIT), sparse_dot_topn (Apache-2.0) and faiss-cpu (MIT), all pinned. CPU only, no pretrained models.
Country is used only to scope retrieval to same-country pairs; no feature or branch reads the label, so the pipeline
runs on France unchanged.

## Changelog (this session)
| Time (IST) | Change |
|---|---|
| 09-27 ~22:15 | Uploads 4–5: `output_v5_fix15_hlr` = **0.958**; v6 uploaded (lean candidates: 7.8 per S1, −27%; XGBoost on 100% of train entities; OOF 0.9751). `match.py --prune-threshold`, `--rounds`, cache overrides; house correction guarded. Docs updated for packaging (team napy). |
| 09-27 ~20:25 | v5 LB **0.952557** (rank 2293). Added `predict_test.py` (test-only scoring; chunked prediction), `postprocess.py` (`--rule`, `--house-lr`, `--em-prior`), token maps (`build_token_maps.py`, `prep --token-maps`) and `tools/holdout_eval.py` (final-model holdout, orphan simulation, rule grid). Fixed a float32 precision bug in `decide.expf_mask` (+0.0005 OOF) and chunked `match.Model.predict` (the monolithic GPU prediction stalled). Details in `EXPERIMENTS.md` rows 7–8. |
| 09-27 ~16:40 | f5 legal-conflict feature, generalised `EXTRA_VIEWS` retrieval (char/addr/name-no-space, each configurable), `decide.tune_robust` (picks one rule robust to an unseen country — never a per-country branch), version-keyed caching (fixed a critical cache-invalidation bug), B15 normalisation fixes shipped for real, stage2 found NOT to transfer via LOCO (not shipped). `output_v5` (all of the above + stage2) running. Full state in `submission_log.tsv`, `EXPERIMENTS.md`, `ARCHITECTURE.md`. |
| 09-27 ~01:20 | Rewrote this README from a stub; created `ARCHITECTURE.md`. |
| 09-27 01:37 | Crude baseline re-run → `output_crude/`. |
| 09-27 ~04:20 | Added GPU XGBoost + blending, split pair/feature caches, feature batch f3, `stage2.py`. v2b and v3 runs in progress. |
| 09-27 03:41 | Step 2 OOF 0.949 (India 0.936, US 0.966); next steps are in `ARCHITECTURE.md` §7. |
| 09-27 ~02:50 | New main pipeline `match.py` (+ `retrieval.hash_candidates`, vectorised `decide.py`). Step 1 → `output_v1/` (0.876, validator PASS). TF-IDF retrieval + trained pruner added; step 2 running → `output_v2/`. Pinned sparse_dot_topn and faiss-cpu. |
