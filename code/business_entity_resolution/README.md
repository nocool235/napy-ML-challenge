# Business Entity Resolution: Amazon ML Challenge 2026

For every Source 1 record, this finds the Source 2 / Source 3 records of the same business. Scoring is macro F0.5 per Source 1 record.

```
prep.py        normalise every record once (Indic transliteration, phonetic skeletons, canonical names/addresses)
                 -> parquet cache
match.py       retrieval.py       hash-join keys + exact TF-IDF top-10 per S2/S3 record, per country
               cascade_pruner.py  LightGBM on retrieval features, keeps 99.7% of retrieved true pairs
               match.py           ~90 pair features -> LightGBM (5-fold out-of-fold) -> decision rule tuned on
                                  the exact macro F0.5 (decide.py) -> output/*.tsv
```

## Reproduce

Python 3.11 or later. Run everything from this folder. The organisers' `dataset/` must have the starter-kit layout
(`dataset/train/train_source1.tsv` … `dataset/test/test_source3.tsv`).

```bash
pip install -r requirements.txt

# 1. normalise once (3-12 min on 28 cores; resumable: finished parts are skipped)
python src/prep.py --data-dir ../../dataset --work-dir ../../work

# 2. retrieval (hash keys + word TF-IDF top-10 + name char n-gram TF-IDF top-8), cascade pruner, 102 pair
#    features, LightGBM + XGBoost 5-fold out-of-fold, stage 2, test candidates and features (~3 h on 28 cores + GPU;
#    --device cpu also works). Its first log line names the norm-/pairs-/feat- cache directories used below.
python src/match.py --data-dir ../../dataset --work-dir ../../work --out-dir ../../run \
    --tfidf-k 10 --char-k 8 --char-max-df 0.01 --models lgb,xgb --device cuda --stage2

# 3. final candidate set: stricter cascade-pruner cut (7.8 candidates per S1 on test), then score it (~13 min)
python src/predict_test.py --run-dir ../../run --norm-dir ../../work/norm-<key> --pair-cache ../../work/pairs-<key> \
    --feat-cache ../../work/feat-<key> --prune-threshold 0.01 --out-dir ../../scored

# 4. decision rule (expected-F0.5, one-to-one, alpha 1.5) + house-number likelihood correction (~1 min)
python src/postprocess.py --run-dir ../../run --probs-dir ../../scored --norm-dir ../../work/norm-<key> \
    --out-dir ../../output --rule expF:1.5 --house-lr ../../work/feat-<key> \
    --test-feat ../../scored/cache/test_features.parquet

# 5. check the files
python ../../utils/validate_submission.py --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv --test-dir ../../dataset/test
```

Steps 3–4 are what produced the submitted files. `match.py` alone (step 2) also writes a complete submission
into `run/`, using the run's own out-of-fold-tuned rule and the looser 99.7% pruner cut.

On Windows, set `PYTHONIOENCODING=utf-8` first.

`match.py` writes:
- `output/matching_results.tsv`: the leaderboard file.
- `output/candidate_pairs.tsv`: exactly the pairs the final model scored.
- `output/debug/`:
  - `metrics_train.json`: recall, oracle F0.5, out-of-fold F0.5 per country, the chosen rule and test structure;
  - `decision_rules.csv`: every rule tried, with its out-of-fold score;
  - `importance_*.tsv`: feature importances;
  - the model files and `test_probs.parquet`.

### Flags (`match.py`)
| flag | default | use |
|---|---|---|
| `--tfidf-k` | 10 | word-view TF-IDF top-k per S2/S3 record. `0` means hash-join keys only (no pruner) |
| `--char-k`, `--addr-k`, `--ns-char-k` | off | top-k for the extra retrieval views (name char n-grams, address words, no-space-name char n-grams); each independent, unioned onto the hash+word candidates |
| `--prune-recall` | 0.997 | Share of retrieved true pairs the cascade pruner must keep |
| `--models` | `lgb` | `lgb`, `xgb` or `lgb,xgb`. Two models are blended, with the weight chosen on out-of-fold log-loss |
| `--device` | `cpu` | `cuda` runs XGBoost on a GPU (optional; not needed to reproduce) |
| `--folds`, `--train-frac` | 5, 0.3 | Out-of-fold folds, and the share of train S1 records each model trains on |
| `--stage2` | – | Second-stage model over stage-1 probabilities and sibling agreement; used only if it improves out-of-fold |
| `--cv` | `group` | `country` = leave-one-country-out (train on one country, score the other), a proxy for an unseen country |
| `--validate-only` | – | Train and out-of-fold only; no test outputs |
| `--no-cache` | – | Recompute candidate pairs and features |

Candidate pairs (and the pruner) are cached in `<work>/pairs-<key>/`, and features in `<work>/feat-<key>/`. The key
is a hand-bumped version constant per module (`RETRIEVAL_VERSION`, `PRUNER_VERSION`, `FEATURE_VERSION`) plus the
config values that affect output, not a source hash — so a rerun reuses the cache whenever the relevant version
hasn't changed, regardless of unrelated edits elsewhere in the file.

Extra retrieval views beyond hash keys + word TF-IDF are configurable and independent: `--char-k`/`--char-max-df`
(name character 3–4-grams), `--addr-k`/`--addr-max-df` (address words), `--ns-char-k`/`--ns-char-max-df` (name with
spaces stripped, character n-grams). Any combination can be on or off; each adds its own columns to the pruner and
feature set automatically.

## Source layout
| module | role |
|---|---|
| `prep.py`, `normalize.py`, `translit.py` | Normalisation into a parquet cache keyed on a hash of the normalisation code |
| `retrieval.py` | `hash_candidates` (capped hash-join keys), `tfidf_candidates` (exact sparse top-k), `tfidf_pair_cosine` |
| `cascade_pruner.py` | Retrieval-time features and the LightGBM pruner |
| `match.py` | Main driver: pair features, LightGBM/XGBoost out-of-fold training, blend, decision, outputs |
| `stage2.py` | Second-stage features: probability context within each S1 / candidate, sibling agreement |
| `predict_test.py` | Test-only scoring with a finished run's saved models and rule; optional per-country rescoring and splicing |
| `postprocess.py` | `--em-prior`: per-country label-shift (EM) correction of test probabilities, then the same rule |
| `build_token_maps.py` | Optional token maps for normalisation: glued-word splits (unsupervised) and label-mined spellings |
| `decide.py`, `metric.py` | Vectorised decision rules and the exact competition metric |
| `utils.py` | Logging with ETA, hashing, TSV reading |
| `run.py`, `blocking.py`, `features.py`, `model.py`, `config.py` | **Legacy, not used by the final pipeline:** the earlier sample-scale prototype, kept for reference |

## Rule compliance
- No external data, APIs or geocoding. Every dictionary is hand-written in `normalize.py` / `translit.py`, and the phonetic-skeleton maps are learned from train Source 1 only.
- TF-IDF vectorisers and source-noise token lists are fit on the provided records without labels.
- Models: LightGBM (MIT); optional XGBoost (Apache-2.0). String metrics: rapidfuzz (MIT). Sparse top-k: sparse_dot_topn (Apache-2.0). No pretrained model; everything runs on CPU.
- Country is treated as an open set of labels. It only scopes retrieval and statistics to same-country records; no rule or feature branches on a specific country, so France runs through unchanged.
- Per-country statistics are computed the same way for every country, from unlabelled data. They are the TF-IDF vocabularies, the source-noise tokens and, in `postprocess.py --house-lr`, the house-number conflict rate among confident pairs. `--house-lr` only acts where a country's rate is below every training country's, which here happens to be France; it never names a country.
