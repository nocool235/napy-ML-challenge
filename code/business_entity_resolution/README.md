# Business Entity Resolution: Amazon ML Challenge 2026

This pipeline finds every Source 2 and Source 3 record that belongs to each Source 1 entity. Scoring is macro F0.5 per Source 1 entity.

```
raw TSVs ─► normalize.py ─► blocking.py ─► features.py ─► model.py (LightGBM, 2 stages) ─► decide.py ─► output/*.tsv
             canonical       TF-IDF kNN     fuzzy + TF-IDF    grouped CV → OOF probs          rule tuned on
             names/addrs     5 views,       + postcode/house  stage 2 = stage 1 + competition  OOF macro F0.5
                             both directions + context ranks   features over OOF probs         (+ one-to-one)
```

## Reproduce

Python 3.11. Run all commands from `code/business_entity_resolution/`.

```bash
pip install -r requirements.txt
# dataset/ has the layout from the starter kit: dataset/train/train_source1.tsv ... dataset/test/test_source3.tsv
python src/run.py --data-dir ../../dataset --out-dir ../../output
python3 ../../utils/validate_submission.py --matching ../../output/matching_results.tsv \
        --candidate ../../output/candidate_pairs.tsv --test-dir ../../dataset/test
```

The run writes these files:
- `output/matching_results.tsv`: the file you upload to the leaderboard.
- `output/candidate_pairs.tsv`: the exact candidate set the model scored.
- `output/debug/`:
  - `metrics_train.json`: blocking recall, oracle F0.5, and OOF F0.5 split by country, singletons and matched entities.
  - `oof_errors.tsv`: out-of-fold false positives and false negatives with the raw text. **Read this after every change.**
  - `blocking_misses.tsv`: true pairs that blocking never proposed. These cap your recall.
  - `decision_rules.csv`: the OOF score of every threshold and decision rule that was tried.
  - `importance_stage*.csv`: feature importances.
- `output/submission_log.tsv`: one line per run. Add the leaderboard (LB) score by hand after each upload.

### Flags
| flag | use |
|---|---|
| `--validate-only` | Runs train CV only. Use it for fast iteration on features or blocking. |
| `--cv country` | Leave-one-country-out CV: trains on US and scores India, then the reverse. It is the best offline proxy for France. |
| `--no-stage2` | Uses the stage-1 model only. |
| `--no-cache` | Recomputes everything. **Delete `cache/` or pass this flag after you change `normalize.py`, `blocking.py`, `features.py` or `config.py` views/K**. The cache stores pairs and features. |

To smoke-test without the real data, run `python src/synthetic.py --out ../../synthetic_dataset`. It writes fake data in the same layout, with a France slice in the test set. The synthetic data is much easier than the real data, so its scores mean nothing.

## Where to change things
- `config.py`: blocking views, K per view, LightGBM parameters and the threshold grid.
- `normalize.py`: the abbreviation, legal-suffix, state and landmark dictionaries. Most gains come from here, driven by `oof_errors.tsv`.
- `features.py`: add new pair features to `pair_features()`. New columns are picked up automatically.
- `decide.py`: the decision rules. `thr` keeps candidates with p ≥ t. `expF` keeps the top-k set with the highest expected F0.5. Both have an optional one-to-one constraint, which uses the fact that S1 is deduplicated.

## Rule compliance
- There are no external lookups, APIs or geocoding. Every dictionary is hand-written in `normalize.py`.
- The TF-IDF vectorisers are fitted on the text of the train and test records without labels.
- Models are LightGBM (MIT) and scikit-learn (BSD-3). rapidfuzz (MIT) provides the string metrics. There is no pretrained model and everything runs on CPU.
- Country is used only as a `same_country` feature. The pipeline accepts any country label.
