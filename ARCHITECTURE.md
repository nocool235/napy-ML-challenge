# Architecture: Amazon ML Challenge 2026 — Business Entity Resolution

*Verified against the working tree on 2026-09-27 (deadline day). Updated in the same session every time
the code changes; see the changelog at the bottom.*

---

## 1. What this system does

Given three noisy business-record sources (S1 = deduplicated reference, S2/S3 = records to attach to it),
output, for every test S1 record, the S2/S3 records describing the same business. Score: macro `F0.5` per S1
record (precision weighted over recall; a correct empty prediction for a singleton scores 1.0).

As of this session the repo is a single-team codebase with **one main pipeline**, `src/match.py`. Everything
else is either the shared normalisation stage, a fallback, or legacy code.

| Pipeline | Entry point | Status |
|---|---|---|
| **Main: `match.py`** | `src/match.py` | Full scale, polars, reads the `prep.py` parquet cache. Retrieval (hash keys + TF-IDF top-k) → cascade pruner → 64 pair features → LightGBM out-of-fold → vectorised expected-F0.5 decision rule → both TSVs. `--tfidf-k 0` reproduces the hash-keys-only "step 1" run. |
| Fallback: crude baseline | `tools/crude_baseline.py` | No-ML hash-join + one global threshold. Valid, full scale, test-mix ≈ 0.823. Kept as the safety net. |
| Legacy: `run.py` | `src/run.py` → `blocking.py` → `features.py` → `model.py` | Pandas, in-memory, never run above a 1% sample. Superseded by `match.py`; `decide.py`/`metric.py` are shared. |

---

## 2. Tech stack

| Layer | Choice | Notes |
|---|---|---|
| Language | Python 3.11 (dev box ran 3.12) | |
| I/O, joins, group-by features | **polars** 1.44.2 | Every full-scale stage. All TSV columns read as strings, no quote handling |
| Numerics | numpy 2.4.4, scipy 1.17.1 | Sparse matrices, vectorised decision rules |
| TF-IDF | scikit-learn 1.8.0 `TfidfVectorizer` | Word view on `name_addr`, per country, df-capped |
| Exact sparse top-k | **sparse_dot_topn 1.2.0** (Apache-2.0) | Multi-threaded `sp_matmul_topn`, S2/S3 → S1 |
| ANN (experimental, not in main pipeline) | faiss-cpu 1.15.1 (MIT) | `retrieval.countrywise_ann_pairs` / `ann_candidates` |
| String similarity | rapidfuzz 3.14.6 | 13 scorers via `process.cpdist` (vectorised, all cores) |
| Models | LightGBM 4.7.0 (MIT) | Cascade pruner + matcher; CPU only, no pretrained weights |
| Storage | Parquet (zstd) | Normalisation cache and per-split feature cache, both code-hashed |
| Parallelism | `ProcessPoolExecutor` with **spawn** (prep); library threads elsewhere | spawn avoids the polars fork deadlock on Linux |

All dependencies are pinned in `code/business_entity_resolution/requirements.txt` (sparse_dot_topn and
faiss-cpu added this session). Everything is MIT/BSD/Apache and CPU-only.

---

## 3. Flowchart

```mermaid
flowchart TD
    RAW["dataset/*.tsv<br/>train + test, source1/2/3, train ground truth"]

    subgraph PREP["prep.py (done, full scale)"]
        P1["normalize.py + translit.py<br/>Indic transliteration, skeletons,<br/>canonical name / address fields"]
        P2["norm-&lt;hash&gt;/split_source/part-*.parquet<br/>+ _SUCCESS markers"]
        P1 --> P2
    end
    RAW --> P1

    subgraph MAIN["match.py (main pipeline, identical code for train and test)"]
        R1["retrieval.hash_candidates<br/>4 capped hash-join keys, per country<br/>(i, j, kbits)"]
        R2["retrieval.tfidf_candidates<br/>word TF-IDF on name_addr, df ≤ 1%<br/>top-10 S1 per S2/S3 (sparse_dot_topn)"]
        U["union + retrieval features<br/>tf_cos, tf_rank, key flags,<br/>per-S1 / per-candidate ranks and gaps"]
        PR["cascade_pruner.Pruner<br/>LightGBM on retrieval features<br/>keep 99.7% of retrieved true pairs"]
        FE["pair_features<br/>13 rapidfuzz scores, postcode / house / street<br/>agreement, name frequency, context ranks"]
        M["LightGBM, 5-fold out-of-fold by S1<br/>each fold trains on 30% of the other folds' S1"]
        D["decide.tune_vec<br/>threshold vs expected-F0.5 top-k,<br/>one-to-one on/off, on the exact metric"]
        O["output/matching_results.tsv<br/>output/candidate_pairs.tsv (= the pairs the model scored)"]
        R1 --> U
        R2 --> U
        U --> PR --> FE --> M --> D --> O
    end
    P2 --> R1
    P2 --> R2

    subgraph FALL["tools/crude_baseline.py (fallback)"]
        C1["same hash keys, rapidfuzz score,<br/>one global threshold"]
    end
    P2 --> C1

    O --> V["utils/validate_submission.py<br/>→ tools/package_submission.py"]
```

---

## 4. Modules

### Normalisation (unchanged this session)
- **`translit.py`** — one ISCII-offset table for nine Brahmic scripts; `skeleton()` consonant classes so English
  and transliterated spellings collide.
- **`normalize.py`** — symmetric canonical dictionaries, country-agnostic; ~19 derived fields per record.
- **`prep.py`** — parallel normalisation into `norm-<sha1(normalize.py, translit.py, v2)>/`. Built at full scale in
  `cache/real_prep/norm-c401d63ff2/`, and its hash matches the current code.

### Retrieval: `retrieval.py`
- `hash_candidates(s1, cand)` — polars port of the crude baseline's four keys (exact name; skeleton + one of the two
  rarest address words; house number + street word; first skeleton token + house number), each scoped by
  country, groups capped at ≤5 S1 / ≤50 candidates. Returns `(i, j, kbits)`, where `kbits` records which key(s) fired.
- `tfidf_candidates(s1, cand, k=10, max_df=0.01, extra=...)` — per country: word TF-IDF on `name_addr` (tokens in
  more than 1% of records or seen only once are dropped), then exact top-k S1 for every S2/S3 record through
  `sparse_dot_topn`. The `extra` pairs (the hash candidates) get the same cosine, so every pair has `tf_cos` and
  `tf_rank` (k+1 when TF-IDF didn't rank it).
- Experimental, unused by `match.py`: `exact_key_candidates` (pandas, contract schema), `ann_hnsw_search`,
  `bidirectional_ann_pairs`, `countrywise_ann_pairs`, `ann_candidates`, `union_candidates`.

### Cascade pruner: `cascade_pruner.py`
LightGBM (63 leaves, 300 rounds) on 14 retrieval-time features: TF-IDF cosine and rank, key flags and count,
per-candidate best cosine and gap, per-S1 rank and gap, and pairs per S1 / per candidate. It trains on 20% of train
S1 and picks the threshold on a disjoint 10% so that 99.7% of retrieved true pairs survive. The model and threshold
are saved next to the feature cache, and test reuses them. The fit logs the kept share at 99.9/99.8/99.5/99% for tuning.

### Matcher: `match.py`
- **Features (64 with retrieval on, 54 without):** 13 rapidfuzz scores on name core, skeleton, no-space name,
  address, landmark-free address and address words; postcode, house and street agree/conflict; lengths; empty flags;
  Indic flags; `is_s3`; name frequency per country (a chain signal); the crude score `base`; rank and gap-to-best of
  `base`/`n_tset`/`a_tset`/`n_skel_tset` within the S1 and within the candidate; candidate counts; and the carried
  retrieval features plus the pruner's probability. Nothing reads the country label directly.
- **Training:** 5 folds grouped by S1 over **all** train pairs, so the one-to-one competition between S1s is intact.
  Each fold's model trains on a 30% sample of the other folds' S1 records (10% of that held out for early stopping).
  The final model uses the same sampling over all folds.
- **Decision:** `decide.tune_vec` tries a global threshold and the expected-F0.5 top-k rule, each with and without
  one-to-one, scored with the exact per-S1 F0.5 (all true matches counted, including ones retrieval never found),
  and picks the best test-mix score (`0.47·India + 0.38·US`, renormalised).
- **Outputs:** `matching_results.tsv`, `candidate_pairs.tsv` (every pair the model scored), and `debug/`:
  `metrics_train.json`, `decision_rules.csv`, `importance.tsv`, `model.txt`, `test_probs.parquet`.
- **Caching:** features per split go to `<work>/match-<hash(retrieval.py, cascade_pruner.py, version, k, prune target)>/`,
  reused with `--resume`.

### Decision and metric: `decide.py`, `metric.py`
The dict-based rules are still there for the legacy `run.py`. The new vectorised versions (`per_s1_f05`, `o2o_mask`,
`expf_mask`, `rule_mask`, `tune_vec`) were checked against them on random data: zero mismatches, identical metric.

### Tools (not shipped)
`crude_baseline.py` (fallback), `bench_retrieval.py` / `bench_union.py` (retrieval benchmarks), `profile_*.py`,
and `package_submission.py` (zip builder: point `--output-dir` at the chosen `output_*/`).

---

## 5. Results (full scale, train out-of-fold; test-mix = India/US reweighted to the test mix)

| Run | Candidate recall | Ceiling (perfect matcher) | India | US | Test-mix | Runtime |
|---|---|---|---|---|---|---|
| Crude baseline (`output_crude/`) | 0.808 | 0.904 | 0.800 | 0.852 | 0.823 | 6 min |
| Step 1: `match.py --tfidf-k 0` (`output_v1/`) | 0.808 | 0.904 | 0.833 | 0.929 | **0.876** | 21 min |
| Step 2: `match.py --tfidf-k 10` (`output_v2/`) | 0.968 → 0.965 after pruning | 0.985 | 0.936 | 0.966 | **0.949** | train ≈ 51 min; test outputs being written |

Step 2, train side, full scale:
- **Retrieval:** TF-IDF top-10 took 6 min for India (883k S1 × 4.1M candidates) and 12 min for US (1.32M × 6.2M) on 28 cores. The union with hash keys has 110.1M pairs (49.9 per S1) and finds 0.968 of all true matches.
- **Pruner:** 6.5× cut to 17.0M pairs (7.7 per S1) at 99.7% of retrieved true pairs, for final recall 0.965. The held-out tradeoff: keeping 99.9% keeps 20.3% of pairs, 99.8% → 17.2%, 99.5% → 13.5%, 99% → 11.3%.
- **Model:** 5 folds at 165–287 s each (best iteration 1.6k–2.3k), then the final model. Training is now the longest stage (~23 min).
- **Decision:** expected-F0.5 with one-to-one, α = 1.0. The best plain threshold (p ≥ 0.65 with one-to-one) is only 0.0006 behind.
- **Where the gain came from:** recall. India rose 0.10 (0.833 → 0.936) and US 0.04. About 0.036 remains between the model and the ceiling; that gap is now the matcher's to close.

---

## 6. Bottlenecks and suggestions

| ID | Bottleneck | Status / suggestion |
|---|---|---|
| B1 | No current valid submission | **Done.** `output_crude/` and `output_v1/` both pass the official validator. |
| B2 | Retrieval and pruner not wired to anything | **Done.** `match.py` runs retrieval → pruner → features → model → decision. |
| B3 | faiss / sparse_dot_topn not pinned | **Done.** Both pinned in `requirements.txt`. |
| B4 | Legacy `run.py` unproven at scale | **Superseded** by `match.py` (full-scale, polars). Keep `run.py` out of the submission or document it as legacy. |
| B5 | Candidate recall caps F0.5 | **Largely addressed.** Hash keys alone: 0.808 recall, 0.904 ceiling. With TF-IDF top-10 + pruner: 0.965 recall, 0.985 ceiling. Next: analyse the remaining 3.2% of misses, and add a char n-gram or name-only view if they're mostly typos or Indic spellings. |
| B6 | TF-IDF retrieval is slow: 19 min on train, 28 cores (India 6 + US 12) | Cost grows roughly quadratically with country size. Options: lower `max_df` (0.005), or a GPU sparse top-k (torch/CUDA; high engineering risk). |
| B7 | Memory: 110M pairs before pruning; features + model matrix ≈ 2× the pruned frame | The pruner keeps it bounded (17M pairs); `--prune-recall` trades recall for memory. Peak RAM ≈ 9–10 GB. |
| B12 | **Reproduction runtime on the organisers' 16 CPUs (3 h limit).** The full step-2 run is ≈ 1.5 h on 28 cores, of which ~23 min is 5-fold out-of-fold training that exists only to choose the decision rule. | Add a reproduction mode that trains the final model once and reuses the saved rule. Time a run capped at 16 threads before the final package. |
| B13 | Feature cache and retrieval are coupled: any feature change reruns TF-IDF retrieval (~40 min, train + test) | Cache the pruned pairs separately (keyed on retrieval + pruner code only), so a feature change costs ~2 min plus training. |
| B14 | Idle GPU (RTX 4080 SUPER, 16 GB); LightGBM trains on CPU (~23 min per run) | XGBoost with CUDA (`device="cuda"`, Apache-2.0 pip wheel) as an optional accelerator and ensemble member, behind a flag. The CPU stays the default for reproduction. |
| B8 | France is unlabelled; decisions are tuned on India/US only | Add a leave-one-country-out run (train US → score India and the reverse) as the France proxy, and spot-check France predictions (step 1 predicts France at 3.18 matches/S1, 4.7% singletons). |
| B9 | No automated tests | The vectorised decision rules have a parity check against the originals (run ad hoc this session). A small `tests/` with that check and normalisation examples would catch regressions cheaply. |
| B10 | Stale code paths (`exact_key_candidates`, the ANN functions, `run.py`, `blocking.py`, `features.py`, `model.py`) | They ship in the zip as-is. Either delete them before packaging or keep them clearly marked experimental/legacy (this document does). |
| B11 | Orphaned `output/pairs/candidate_pairs.parquet` (19,838 rows, old contract) | Delete, since nothing reads it. |

---

## 7. Plan (agreed 09-27 ~03:40)

1. Commit and push the current state.
2. GPU XGBoost behind a flag, plus a separate pruned-pair cache (B13, B14).
3. Cheap feature batch: landmark flags, all-address-number comparison, name↔address field swap, separate name and address rare-word (IDF) overlap. Check the gain per country.
4. One run with both models; validate; submit. (Step 2 gets submitted as soon as it validates, as the leaderboard anchor.)
5. Second-stage model: a second out-of-fold training round on stage-1 probabilities plus their context (gap to the S1's best, runner-up probability for the candidate, probability mass per S1, sibling agreement), run on the GPU. In parallel: France proxy mode (train on one country, score the other), the retrieval-miss analysis, and the reproduction mode (B12).
6. Deliverables by ~20:00: methodology doc, submission log, packaged zip.

---

## Changelog (this session)

| Time (IST) | Change |
|---|---|
| 09-27 ~01:20 | First write-up: three-pipeline split, integration gaps, bottlenecks B1–B9. No code changed. |
| 09-27 01:37 | Crude baseline re-run at full scale → `output_crude/` (test-mix 0.823). `utils/validate_submission.py` extracted from the organiser zip. |
| 09-27 ~02:05 | `retrieval.py`: restored the ANN functions (removed by accident), added `ann_candidates` and `union_candidates` (contract schema). |
| 09-27 ~02:25 | Team plan dropped; this repo is the single main line. Added `retrieval.hash_candidates` (polars), vectorised rules in `decide.py`, and new `match.py`. Step 1 → `output_v1/`, OOF test-mix 0.876, validator PASS. |
| 09-27 ~02:45 | Added `retrieval.tfidf_candidates` (sparse_dot_topn) and rewrote `cascade_pruner.py` as a trained LightGBM pruner. Wired both into `match.py` (`--tfidf-k`, `--prune-recall`). Pinned sparse_dot_topn and faiss-cpu. Step 2 full run started → `output_v2/`. |
| 09-27 03:41 | Step 2 train side done: recall 0.968 → 0.965 after pruning, ceiling 0.985, **OOF test-mix 0.949** (India 0.936, US 0.966). Test outputs being written. GPU confirmed idle; added B12–B14 and the agreed plan (§7). |
