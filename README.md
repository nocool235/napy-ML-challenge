# Amazon ML Challenge 2026 — Business Entity Resolution

Match Source 2 / Source 3 business records to the Source 1 record describing the same business, across US,
India and (test-only) France. Scored as macro `F0.5` per S1 record. Engineering detail, the flowchart and the
bottleneck list are in **[`ARCHITECTURE.md`](ARCHITECTURE.md)**.

**Deadline: Sunday 27 Sep 2026, 23:59 IST.**

> Reflects the working tree as of 2026-09-27. This README and `ARCHITECTURE.md` are updated together whenever
> the code changes; see the changelog at the bottom.

## Current state

This repo is the team's single main line. The main pipeline is `src/match.py`:

```
prep.py (normalise → parquet cache)
  → retrieval.py   hash-join keys + TF-IDF top-10 per S2/S3 record (per country)
  → cascade_pruner.py   LightGBM on retrieval features, keeps 99.7% of retrieved true pairs
  → match.py   64 pair features → LightGBM (5-fold out-of-fold) → expected-F0.5 decision rule
  → output/matching_results.tsv, output/candidate_pairs.tsv
```

| Submission | Train OOF test-mix F0.5 | Validator | Notes |
|---|---|---|---|
| `output_crude/` | 0.823 | – | No-ML fallback (`tools/crude_baseline.py`) |
| `output_v1/` | **0.876** | PASS | `match.py --tfidf-k 0`: hash-key candidates only (recall 0.81, ceiling 0.904) |
| `output_v2/` | **0.949** | test outputs being written | `match.py --tfidf-k 10`: TF-IDF retrieval + pruner (recall 0.965, ceiling 0.985) |

"Test-mix" is the train out-of-fold score with India and US weighted as in the test set (0.47/0.38). France has no
labels, so it is not in this number. The leaderboard score is the real check.

## Reproduce

```bash
cd code/business_entity_resolution && pip install -r requirements.txt && cd ../..
# dataset/ (organiser data) and utils/validate_submission.py (from the student_resource zip) must be in place

# 1. normalise once (resumable; ~12 min)
python code/business_entity_resolution/src/prep.py --data-dir dataset --work-dir cache/real_prep

# 2. main pipeline: train OOF + test outputs (step 2 settings are the defaults)
python code/business_entity_resolution/src/match.py --data-dir dataset --work-dir cache/real_prep --out-dir output
#    --tfidf-k 0        hash keys only (the step-1 run)
#    --prune-recall R   share of retrieved true pairs the pruner keeps (default 0.997)
#    --validate-only    train + OOF only;  --resume  reuse cached features

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
  match.py            MAIN pipeline driver: features, LightGBM OOF, decision, outputs
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
| 09-27 ~01:20 | Rewrote this README from a stub; created `ARCHITECTURE.md`. |
| 09-27 01:37 | Crude baseline re-run → `output_crude/`. |
| 09-27 03:41 | Step 2 OOF 0.949 (India 0.936, US 0.966); next steps are in `ARCHITECTURE.md` §7. |
| 09-27 ~02:50 | New main pipeline `match.py` (+ `retrieval.hash_candidates`, vectorised `decide.py`). Step 1 → `output_v1/` (0.876, validator PASS). TF-IDF retrieval + trained pruner added; step 2 running → `output_v2/`. Pinned sparse_dot_topn and faiss-cpu. |
