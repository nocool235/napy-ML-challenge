# Architecture: Amazon ML Challenge 2026 — Business Entity Resolution

*Verified against the working tree on 2026-09-27 22:15 IST (deadline day; final pipeline = v6). Team napy. Updated in the same session every time
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
| **Main: `match.py` + `postprocess.py`** | `src/match.py`, `src/postprocess.py` | Final (upload 4, LB 0.958). Retrieval: hash keys + word TF-IDF top-10 + name char n-gram TF-IDF top-8. Cascade pruner at 99.7% of retrieved true pairs (10.8 candidates per S1 on test). 102 pair features (`f5`). LightGBM + XGBoost, 5-fold OOF, then stage 2 (+21 features). `postprocess.py` applies the rule (expF α 1.5, one-to-one) and the house-number correction. |
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
| Models | LightGBM 4.7.0 (MIT); XGBoost 3.4.1 (Apache-2.0, optional) | Cascade pruner + matcher. `--models lgb,xgb --device cuda` trains XGBoost on the RTX 4080 alongside LightGBM on CPU and blends them. The CPU is the default. No pretrained weights. |
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
        R2["retrieval.tfidf_candidates<br/>word TF-IDF on name_addr (top-10) +<br/>char 3-4-grams on name_core (top-8), df ≤ 1%<br/>exact sparse_dot_topn, per country"]
        U["union + retrieval features<br/>tf_cos, tf_rank, key flags,<br/>per-S1 / per-candidate ranks and gaps"]
        PR["cascade_pruner.Pruner<br/>LightGBM on 19 retrieval features<br/>keep 99.7% of retrieved true pairs<br/>(10.8 per S1 on test; --prune-threshold for smaller sets)"]
        FE["pair_features<br/>13 rapidfuzz scores, postcode / house / street<br/>agreement, name frequency, context ranks"]
        M["LightGBM (CPU) + XGBoost (GPU), 5-fold OOF by S1<br/>+ stage 2: 21 features over stage-1 probabilities"]
        D["decide.tune_vec (OOF) + postprocess.py<br/>expected-F0.5, one-to-one, α 1.5 (orphan rebuild);<br/>house-number likelihood correction per country"]
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
- `tfidf_candidates(s1, cand, col, k, max_df, analyzer, ngram_range, extra=...)` — per country: TF-IDF on `col`
  (tokens in more than `max_df` of records or seen only once are dropped), then exact top-k S1 for every S2/S3
  record through `sparse_dot_topn`. `analyzer`/`ngram_range` select word tokens or `char_wb` n-grams (typo/spelling
  tolerant). Passing a prior call's `(i, j)` as `extra` scores those pairs too and gives every pair both views'
  scores — chaining calls this way unions any number of views into one candidate set.
- **`EXTRA_VIEWS` (in `match.py`, generalises what used to be one hardcoded char view):** `char` (name_core,
  char 3–4-grams), `addr` (addr_clean, word), `ns` (name_nospace, char 3–4-grams). Each is on/off and independently
  sized via `--char-k`/`--addr-k`/`--ns-char-k` (and matching `--*-max-df` flags); `candidates()` loops over
  whichever are configured and chains them onto the hash+word union via `extra`. `cascade_pruner.py` and the
  `RETRIEVAL_COLS` feature list read whichever per-view columns are actually present, so views can be added or
  dropped from a run without touching either file.
- `tfidf_pair_cosine(s1, cand, pairs, col, ...)` — TF-IDF cosine for a *given* set of pairs (no top-k search); used
  both as a pair feature and for post-hoc diagnostics (e.g. scoring a pair retrieval never found).
- Experimental, unused by `match.py`: `exact_key_candidates` (pandas, contract schema), `ann_hnsw_search`,
  `bidirectional_ann_pairs`, `countrywise_ann_pairs`, `ann_candidates`, `union_candidates`.

### Cascade pruner: `cascade_pruner.py`
LightGBM (63 leaves, 300 rounds) on 14 retrieval-time features: TF-IDF cosine and rank, key flags and count,
per-candidate best cosine and gap, per-S1 rank and gap, and pairs per S1 / per candidate. It trains on 20% of train
S1 and picks the threshold on a disjoint 10% so that 99.7% of retrieved true pairs survive. The model and threshold
are saved next to the feature cache, and test reuses them. The fit logs the kept share at 99.9/99.8/99.5/99% for tuning.

### Matcher: `match.py`
- **Features (93 at f4/v3; +1 at f5): `FEATURE_VERSION` bumps whenever a definition changes** (`f2`→`f4`→`f5`,
  it names the feature cache so a bump forces a rebuild, nothing else does). 13 rapidfuzz scores on name core,
  skeleton, no-space name, address, landmark-free address and address words; postcode, house and street
  agree/conflict; lengths; empty flags; Indic flags; `is_s3`; name frequency per country (a chain signal); the
  crude score `base`; rank and gap-to-best of `base`/`n_tset`/`a_tset`/`n_skel_tset` within the S1 and within the
  candidate; candidate counts; and the carried retrieval features (per configured view) plus the pruner's probability.
  - **f3** added: landmark flags; overlap, Jaccard and conflict over *all* address numbers and over numbers in the
    name; field swap (name vs the other record's address); per-country TF-IDF cosines for name words, address
    words and name character 3-grams (`retrieval.tfidf_pair_cosine`); context over the name cosine.
  - **f4** added: data-driven per-country **source-noise tokens** (`source_noise_tokens`/`_denoised`) — tokens one
    source uses far more than another (e.g. French "Participations"/"Holding", Indian unmapped transliterations),
    detected unsupervised from the provided records, with similarity features recomputed on the de-noised text.
    This is the largest single validated gain so far: LOCO mean 0.9110 → 0.9227, and +0.011 real LB (v2→v3, part
    of a 0.9365→0.947 jump that also included other f4/f3 features).
  - **f5** added: legal-form-conflict — extracted legal suffixes (Ltd/LLP/Pvt/GmbH/SARL/...) compared between the
    two records (`name_legal`, via `_set_overlap`). Found missing during a false-positive audit; suffix conflicts
    are a strong precision signal (~50× more common among false positives than true matches in that audit).

  Nothing reads the country label directly.
- **Training:** 5 folds grouped by S1 over **all** train pairs, so the one-to-one competition between S1s is intact.
  Each fold's model trains on a 30% sample of the other folds' S1 records (10% of that held out for early stopping).
  The final model uses the same sampling over all folds.
- **Decision:** `decide.tune_vec` tries a global threshold and the expected-F0.5 top-k rule, each with and without
  one-to-one, scored with the exact per-S1 F0.5 (all true matches counted, including ones retrieval never found),
  and picks the best test-mix score (`0.47·India + 0.38·US`, renormalised).
- **Outputs:** `matching_results.tsv`, `candidate_pairs.tsv` (every pair the model scored), and `debug/`:
  `metrics_train.json`, `decision_rules.csv`, `importance.tsv`, `model.txt`, `test_probs.parquet`.
- **Caching:** candidate pairs and the pruner go to `<work>/pairs-<key>/`; features go to `<work>/feat-<key>/`.
  The key is **not** a source hash — it's `RETRIEVAL_VERSION` (`retrieval.py`) / `PRUNER_VERSION` (`cascade_pruner.py`)
  / `FEATURE_VERSION` (`match.py`) plus the config values that affect output (k, max_df per view, prune-recall).
  These are hand-bumped when a change actually alters what's returned; a source-hash scheme was tried first but was
  found to silently invalidate a cache from *any* edit to a relevant file — including a backward-compatible signature
  change (new optional param, existing call sites unaffected) — which once cost a ~45-minute unplanned retrieval
  rebuild mid-run. Both caches are reused automatically; `--no-cache` recomputes. A feature-only change now costs
  minutes, not a 40-min retrieval rerun.
- **Models:** `fit_model(kind, device, …)` wraps LightGBM and XGBoost behind one interface. With two kinds, each fold
  trains both in parallel threads, so a fold costs the slower of the two (LightGBM ~190 s; XGBoost on GPU ~57 s).
  The blend weight is chosen on OOF log-loss, and the decision rule is tuned once on the blend.
  Each model's score alone, under the same rule, is logged for comparison.

### Second stage: `stage2.py` (wired into `match.py` behind `--stage2`)
Context over the stage-1 probability *p*:
- **Within the S1:** rank, best, gap to best, runner-up, total mass, pairs above 0.5, and this pair's share of the mass.
- **Within the candidate:** the best competing S1, and the margin over the best *other* S1.
- **Sibling agreement:** compared with the S1's other top-3 candidates: name and address similarity (max, mean,
  p-weighted mean), and house-number and postcode agreement and conflict.

On train it uses out-of-fold *p*, to avoid leakage; on test it uses the final stage-1 model's *p*.
Stage 2 trains with the **same folds** as stage 1 (same seed). It is kept only if its out-of-fold score beats stage 1;
otherwise the stage-1 model and rule are used. Its models are saved to `debug/stage2/`.
Tested end to end on synthetic data: both the "keep stage 1" branch and a forced "use stage 2" branch; output PASSes the validator.

**Measured on real f4 features (XGBoost only, GPU, matching v3's cache): group-CV +0.0096 mix — a real,
sizeable in-distribution gain. But leave-one-country-out on the same features: −0.0011 — it does *not* transfer
to a country never trained on.** Group-CV mixes random folds of countries the model has already seen, so it can
reward a model for learning country-relational calibration (e.g. "this probability distribution shape means
India") that has no counterpart in an unseen country like France. This is why a France-relevant change is judged
on the LOCO number, never group-CV alone — see `decide.tune_robust` below. Stage 2 has not shipped in any upload
for this reason; it needs either a fix that makes its signal country-agnostic or explicit sign-off to ship anyway.

### France proxy: `--cv country`
Leave-one-country-out. Train on US and score India, then the reverse, with no test outputs.
- Reported under the **shipped** rule (expected-F0.5, α = 1, one-to-one), which is how France is actually decided: by a rule tuned on other countries.
- Also reported under a rule re-tuned on the held-out predictions, as an optimistic upper bound.
- Written to `debug/metrics_loco.json`.

**Use:** compare feature sets on transfer to an unseen country before spending an upload.

### Decision and metric: `decide.py`, `metric.py`
The dict-based rules are still there for the legacy `run.py`. The new vectorised versions (`per_s1_f05`, `o2o_mask`,
`expf_mask`, `rule_mask`, `tune_vec`) were checked against them on random data: zero mismatches, identical metric.

**Precision fix (09-27 evening):** `expf_mask` builds per-S1 running sums as differences of one global `cumsum` over every pair. In float32 (the test path) that sum reaches millions and each step loses ~0.25, so it now always runs in float64. That's the dtype the rule is tuned in (`tune_vec` gets float64 OOF). On v5's OOF the fix is worth +0.00048 mix; on test, 34,147 decisions flip.

`decide.tune_robust(table_groupcv, table_loco, france_weight=0.15)` picks **one** rule for every test record —
never a per-country branch, which the competition rules forbid — by blending each candidate rule's group-CV mix
score with its LOCO mean: `robust_score = 0.85 × mix + 0.15 × loco_mean`. This exists so a feature/rule that looks
best in-distribution but is a bad bet on an unseen country (see stage 2, above) can be caught before an upload,
without ever conditioning the shipped rule on the country label itself. Tested on real v3 OOF data: in that case
the robust choice matched the group-CV-only choice (expF, α=1, o2o=True) — confirming it's a safe drop-in
replacement for `tune_vec` that costs nothing when the two agree.

### Final-push additions (09-27 evening): scoring and normalisation without retraining
- **`predict_test.py`**: test-only scoring with a finished run's saved models and rule (stage 1 → stage 2 → rule),
  no training. `--countries` rescored alone gives the same rows as inside a full run (everything is
  country-scoped). `--splice-into` replaces those countries' rows in an existing submission and copies every other
  line byte for byte. Runs launched before `EXTRA_VIEWS` recorded `char_k` and use older cache keys: pass
  `--pair-cache`/`--feat-cache` from the run's log.
- **`postprocess.py --em-prior`**: per-country label-shift correction on a run's test probabilities (EM,
  Saerens et al. 2002), then the run's rule unchanged. Same procedure for every country. It's valid only because
  the model is calibrated: on v5's OOF, EM recovers each train country's true positive rate within 0.002.
- **Token maps** (`normalize.TOKEN_MAPS`, off by default; `prep.py --token-maps`, which is part of the norm-cache
  key): token → replacement, applied right after `basic_clean`. Built by **`build_token_maps.py`**:
  - **B1, glued words (unsupervised):** a token of ≥ 6 letters absent from every S1 name/address is split into S1
    words (`nantesclub` → `nantes club`).
  - **B2, spellings mined from true train pairs** (`fyuchar` → `future`). `--mine-s1 final-train` mines only the
    S1s a run's final model trained on, keeping the rest an honest holdout.
- **`tools/holdout_eval.py`**: F0.5 of a run's *final* models on the ~70% of train S1 they never trained on, from
  cached features (baseline) or rebuilt from another norm cache (variant) — the before/after check for
  test-time-only changes.

### Tools (not shipped)
`crude_baseline.py` (fallback), `bench_retrieval.py` / `bench_union.py` (retrieval benchmarks), `profile_*.py`,
`holdout_eval.py` (above), and `package_submission.py` (zip builder: point `--output-dir` at the chosen `output_*/`).

---

## 5. Results (full scale; test-mix = train out-of-fold, India/US reweighted to the test mix)

| Run | Candidate recall | Ceiling | India | US | Test-mix | Public LB |
|---|---|---|---|---|---|---|
| Crude baseline (`output_crude/`) | 0.808 | 0.904 | 0.800 | 0.852 | 0.823 | – |
| v1: hash keys + LightGBM | 0.808 | 0.904 | 0.833 | 0.929 | 0.876 | – |
| v2: + word TF-IDF + pruner | 0.965 | 0.985 | 0.936 | 0.966 | 0.949 | 0.9365 |
| v3: + de-noised features | 0.965 | 0.985 | 0.947 | 0.969 | 0.957 | 0.947 |
| v5: + char view, legal feature, stage 2 | 0.971 | 0.987 | 0.968 | 0.983 | 0.975 | 0.952557 |
| **Final: v5 + float64 rule, α 1.5, house correction** | 0.971 | 0.987 | 0.968 | 0.983 | 0.975 | **0.958** |
| v6: lean cut (−27% candidates), XGBoost on 100% of entities | 0.968 | 0.986 | 0.968 | 0.984 | 0.975 | 0.955 (not kept) |

Full-scale facts for the final pipeline:
- **Retrieval (train):** hash keys give 16.4M pairs. The word view took 6 min for India and 13 min for US; the char view 10 and 50 min (the US figure was inflated by contention; about 22 min clean). The union has 178M pairs (80.7 per S1) and recall 0.974.
- **Pruner:** 99.7% of retrieved true pairs gives 18.7M pairs (8.5 per S1, recall 0.971). The final cut (score ≥ 0.01) gives 13.7M (6.2 per S1, recall 0.968); test has 13.6M (7.8 per S1).
- **Candidate-size curve** (India, held-out train entities, final models): each 10% fewer candidates costs ≈ 0.0005 F0.5, and the loss accelerates past a ~30% cut.
- **Model (v6):** XGBoost on the GPU, 130 s per fold at 100% of entities (the 3000-round cap was reached, so v7 tries 6000). Stage-1 OOF 0.9691, stage 2 0.9751.
- **Why the LB lagged OOF:** orphan records, since test ≈ train with 20% of S1 removed (costs ~0.002, favours α 1.5), and France house numbers being 6× more reliable than in training (corrected post hoc: +0.005 LB).

---

## 6. Bottlenecks and suggestions

| ID | Bottleneck | Status / suggestion |
|---|---|---|
| B1 | No current valid submission | **Done.** `output_crude/` and `output_v1/` both pass the official validator. |
| B2 | Retrieval and pruner not wired to anything | **Done.** `match.py` runs retrieval → pruner → features → model → decision. |
| B3 | faiss / sparse_dot_topn not pinned | **Done.** Both pinned in `requirements.txt`. |
| B4 | Legacy `run.py` unproven at scale | **Superseded** by `match.py` (full-scale, polars). Keep `run.py` out of the submission or document it as legacy. |
| B5 | Candidate recall caps F0.5 | **Addressed (final: 0.968 at the lean cut, 0.974 before pruning).** Char n-gram view shipped in v5. Hash keys alone: 0.808 recall, 0.904 ceiling. With TF-IDF top-10 + pruner: 0.965 recall, 0.985 ceiling. Next: analyse the remaining 3.2% of misses, and add a char n-gram or name-only view if they're mostly typos or Indic spellings. |
| B6 | TF-IDF retrieval is slow: 19 min on train, 28 cores (India 6 + US 12) | Cost grows roughly quadratically with country size. Options: lower `max_df` (0.005), or a GPU sparse top-k (torch/CUDA; high engineering risk). |
| B7 | Memory: 110M pairs before pruning; features + model matrix ≈ 2× the pruned frame | The pruner keeps it bounded (17M pairs); `--prune-recall` trades recall for memory. Peak RAM ≈ 9–10 GB. |
| B12 | **Reproduction runtime** (v6: ≈ 2.5 h on 28 cores + GPU; CPU-only XGBoost is slower). The full step-2 run is ≈ 1.5 h on 28 cores, of which ~23 min is 5-fold out-of-fold training that exists only to choose the decision rule. | Add a reproduction mode that trains the final model once and reuses the saved rule. Time a run capped at 16 threads before the final package. |
| B13 | Feature cache and retrieval were coupled | **Done.** Separate `pairs-`/`feat-` caches. Originally keyed on function source; rekeyed to hand-bumped version constants (see Matcher §Caching, above) after a source-hash false-invalidation cost an unplanned ~45-min retrieval rebuild mid-run. The v2 cache was migrated across both scheme changes, so no retrieval rerun was needed. |
| B14 | Idle GPU | **Done.** XGBoost with CUDA (`--models lgb,xgb --device cuda`): a fold takes 57 s on GPU vs 187 s for LightGBM on CPU, at almost the same log-loss (0.0696 vs 0.0692). **Decision rule for shipping:** XGBoost ships only if the blend beats LightGBM alone by ≥ 0.002 OOF *and* the CPU-only reproduction run fits comfortably inside 3 h on 16 threads. |
| B8 | France is unlabelled; decisions are tuned on India/US only | **Measured and partly addressed.** Upload #1 (v2) scored 0.9365 public, which implies **France ≈ 0.86–0.89** vs ~0.94–0.97 for India/US. `tools/spot_check.py` findings: <br/>• The model is *confident* on France (4.3% of pairs uncertain vs 6–7%), so the errors are confident mistakes. <br/>• House-number conflicts are **not** France-specific (train true matches also conflict: India 22%, US 13%). <br/>• **The main cause:** S2/S3 pad French names with words S1 never uses ("Participations", "Holding", "Distribution", "Développement", "Associés", "Groupe", 2–4% each). US has the same effect far more weakly ("Midtown", "Riverside"), and India has unmapped transliterations ("limitet", "venchars"). <br/>• **Fix, f4:** data-driven source-noise tokens (same rule for every country, no labels) and similarity features on the de-noised text. <br/>• **Still open:** the France proxy run (train on one country, score the other). |
| B15 | Normalisation gaps surfaced by the spot-check | **Done**, in `normalize.py`: `cours`→`cross` (was `crs`, which collided with the existing `crs`→`cross` entry so "Cours X" and "Crs X" never matched — single-pass dict lookup); `N°` (`_NDEG_RE`) stripped before `basic_clean`, which was leaving a stray `n` confusable with north→`n`; `q`→`quai`; `appartement`→`apt`; state codes `tg`/`dl` and native spellings `tamizhnatu`/`keralam`/`pashchimabang` added to `PHRASES`. Verified with concrete before/after test cases. India transliteration snapping (`limitet`, `infrastrakchar`, etc.) is still open — f4's de-noising covers it at feature level without needing exact dictionary entries. |
| B9 | No automated tests | The vectorised decision rules have a parity check against the originals (run ad hoc this session). A small `tests/` with that check and normalisation examples would catch regressions cheaply. |
| B10 | Stale code paths (`exact_key_candidates`, the ANN functions, `run.py`, `blocking.py`, `features.py`, `model.py`) | They ship in the zip as-is. Either delete them before packaging or keep them clearly marked experimental/legacy (this document does). |
| B11 | Orphaned `output/pairs/candidate_pairs.parquet` (19,838 rows, old contract) | Delete, since nothing reads it. |
| B16 | Retrieval union still misses ~3.2–3.5% of true pairs (detailed miss analysis: 68% high-char-similarity/word-tokenisation failures, 42% empty-address dilution, India 72% of misses vs 40% of candidate share) | Generalised retrieval to `EXTRA_VIEWS` (char n-gram, address word, name-no-space char n-gram — see Retrieval §, above), each independently configurable. char view alone validated in a smaller run; the full char+addr+ns config is `output_v5` (in progress). Also: a cheap diagnostic on the current miss set found 50.4% of missed true pairs (0.98% of *all* true pairs) have a retrieved sibling of the same S1 with a strong address match — a rough ceiling for what a sibling-propagation retrieval step could add. Deferred building it (~2h) pending v5's real recall numbers, since it looked like partial overlap with the char/addr views' gains. |

---

## 7. Final state (09-27 22:15)

The morning plan (§ above, steps 1–6) was completed. Stage 2 shipped: v5 onwards, OOF +0.006. The evening added:
1. A float64 fix in the decision rule. The first stage-2 run (v5) then went through test scoring with chunked prediction, because its monolithic GPU predict stalled.
2. A post-mortem of the v5 leaderboard gap. This produced the orphan rebuild and the house-number correction: +0.0054 LB.
3. A smaller candidate set, since the organisers rank it: the pruner cut 0.01.
4. v6: XGBoost on all train entities, on the lean set.

v6 scored 0.955 on the leaderboard, below 0.958, so the **final package is upload 4** (`output_v5_fix15_hlr/`). The current code reproduces it byte for byte from v5's saved probabilities. Packaging: `python tools/package_submission.py --team napy --doc Documentation_template.md --output-dir output_v5_fix15_hlr`.

---

## Changelog (this session)

| Time (IST) | Change |
|---|---|
| 09-27 ~22:50 | Upload 5 (v6) = 0.955, so the final package is upload 4 (0.958). Docs set to that configuration. |
| 09-27 ~22:15 | Upload 4 (v5 + post-processing) = **0.958**. v6 trains XGBoost on 100% of entities on the lean candidate set (pruner cut 0.01; −27% candidates): OOF 0.9751. Added `match.py --prune-threshold/--rounds/--norm-dir/--pair-cache/--feat-cache`. The house correction is now guarded (min 1000 confident pairs, factor ≥ 0.05). Docs refreshed for the final package (team napy). |
| 09-27 ~20:25 | v5 LB 0.952557 (+0.0056 vs +0.0175 OOF; France likely got worse). Fixed a float32 precision bug in `decide.expf_mask` (+0.0005 OOF) and chunked prediction in `match.Model.predict`. Added `predict_test.py`, `postprocess.py` (`--rule`, `--house-lr`, `--em-prior`), token maps (`build_token_maps.py`, `prep --token-maps`) and `tools/holdout_eval.py`. Findings: France's confident-pair house-conflict rate is 0.85% vs ≥ 5.1% in every training country, handled by the house-LR correction. Orphans (test ≈ train minus 20% of S1) cost India ~0.0018 and favour expF α 1.5. B1/B2 token maps are flat without a retrain. |
| 09-27 ~16:40 | Committed the tier-1/2 work below (f5 legal-conflict feature, generalised `EXTRA_VIEWS` retrieval, `decide.tune_robust`, version-keyed caching, B15 normalisation fixes, `stage2.py`'s LOCO no-transfer finding, sibling-propagation diagnostic). `output_v5` (char+addr+ns retrieval, B15, f5, stage2, lgb+xgb) running in the background. |
| 09-27 ~15:00–16:40 | Session 2: found and fixed a critical cache-invalidation bug (source-hash → version-constant keys, see B13). Added feature set **f5** (legal-form-conflict). Generalised the retrieval view architecture to `EXTRA_VIEWS` (char/addr/name-no-space, each independently configurable). Rewrote `normalize.py`'s B15 fixes for real (cours/crs collision, N°, q, appartement, state-code phrases). Measured `stage2.py` via leave-one-country-out: group-CV +0.0096 but LOCO −0.0011 — does not transfer, not shipped. Added `decide.tune_robust` as the compliant way to pick one rule that's robust to an unseen country. Ran a sibling-propagation diagnostic (Tier 4 idea): 0.98% of all true pairs have a retrieved sibling with a strong address match — deferred the full build pending v5's real numbers. |
| 09-27 ~01:20 | First write-up: three-pipeline split, integration gaps, bottlenecks B1–B9. No code changed. |
| 09-27 01:37 | Crude baseline re-run at full scale → `output_crude/` (test-mix 0.823). `utils/validate_submission.py` extracted from the organiser zip. |
| 09-27 ~02:05 | `retrieval.py`: restored the ANN functions (removed by accident), added `ann_candidates` and `union_candidates` (contract schema). |
| 09-27 ~02:25 | Team plan dropped; this repo is the single main line. Added `retrieval.hash_candidates` (polars), vectorised rules in `decide.py`, and new `match.py`. Step 1 → `output_v1/`, OOF test-mix 0.876, validator PASS. |
| 09-27 ~02:45 | Added `retrieval.tfidf_candidates` (sparse_dot_topn) and rewrote `cascade_pruner.py` as a trained LightGBM pruner. Wired both into `match.py` (`--tfidf-k`, `--prune-recall`). Pinned sparse_dot_topn and faiss-cpu. Step 2 full run started → `output_v2/`. |
| 09-27 ~04:45 | v2b (blend) OOF 0.9497, +0.0004 over v2: below the 0.002 bar, so XGBoost stays an accelerator. Stage 2 wired behind `--stage2`, and `--cv country` added; both smoke-tested end to end on synthetic data. Filled `Documentation_template.md` and rewrote the shipped `code/business_entity_resolution/README.md`. v3 (f4) running. |
| 09-27 ~04:30 | Upload #1 (v2) = **0.9365 public, rank 1974**; implies France ≈ 0.86–0.89. Added `tools/spot_check.py`. Found the source-added noise tokens behind France's gap (also present in US/India). Added feature set **f4** (93 features): de-noised name/address similarities, noise counts, an acronym feature. Logged uploads in `submission_log.tsv`. v3 now runs f4. |
| 09-27 ~04:20 | Step 2 validated (PASS) and handed over as upload #1. Added GPU XGBoost (`fit_model`, parallel per-fold training, OOF-log-loss blend), split `pairs-`/`feat-` caches with a one-time v2 migration, feature batch f3 (82 features), and `stage2.py` (unit-checked). Runs queued: v2b (v2 features, LightGBM + XGBoost) → v3 (f3, LightGBM + XGBoost). |
| 09-27 03:41 | Step 2 train side done: recall 0.968 → 0.965 after pruning, ceiling 0.985, **OOF test-mix 0.949** (India 0.936, US 0.966). Test outputs being written. GPU confirmed idle; added B12–B14 and the agreed plan (§7). |
