# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** napy
**Team Members:** Adarsh Bhatt, Prakhar Saxena, Yash Goel, Nakul Verma
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

For every Source 1 business we find its Source 2 / Source 3 records with a retrieve → prune → classify → decide pipeline.
1. **Normalise:** country-agnostic rules, including rule-based Indic-script transliteration and phonetic skeletons.
2. **Retrieve:** capped hash-join keys, plus exact sparse TF-IDF top-k search inside each country on two views: name + address words, and name character n-grams.
3. **Prune:** a LightGBM cascade pruner on retrieval-time features only. It cuts 178M retrieved train pairs to **8.5 per S1** (10.8 per S1 on test) and keeps 97.1% of all true pairs.
4. **Classify:** a LightGBM + XGBoost blend on 102 pair features. A second stage adds 21 features built from the first stage's probabilities.
5. **Decide:** a per-entity rule that maximises the expected F0.5 directly. It uses a one-to-one constraint and is tuned on a train rebuild that reproduces test's structure.

Two findings mattered most. At full scale, **candidate recall, not the matcher**, was the first bottleneck (0.81 → 0.97). After that, the gap between out-of-fold and leaderboard scores came from how **test differs from train**. We measured two such differences without labels and corrected for them: records whose entity is missing from S1, and a lower house-number noise rate in the unseen country. Public leaderboard: 0.9365 → **0.958**.

---

## 2. Methodology

### 2.1 Problem Analysis

- **Scale.** Train has 2.21M S1 and 10.32M S2+S3 records; test has 1.73M S1 and 9.97M S2+S3. Within-country all-pairs on test is ≈ 6.7 × 10¹² pairs.
- **Structure (train).** 5.6% of S1 are singletons, with 3.46 true matches per S1 on average. No S2/S3 record belongs to two S1 records (checked over 7.64M true pairs), and no true pair crosses countries.
- **Test differs from train in two measurable ways:**
  - **Orphan records.** Test has 5.75 S2/S3 records per S1 against 4.68 in train. Train with ≈ 20% of its S1 records removed reproduces both that ratio (5.85) and test's ≈ 40% share of records that match nothing. Those records' businesses are simply absent from S1, and the best remaining S1 is a wrong one.
  - **France is unlabelled** (15% of test S1) and its record generator is noisier in some fields and cleaner in others; see §2.2 item 5.
- **Noise we had to handle:**
  - abbreviations and legal-form variants (Pvt/Private, SARL/SAS/EURL);
  - DBA and trade names;
  - word order and typos, including glued words ("nantesclub");
  - landmark-style Indian addresses ("Near SBI ATM");
  - Indic script in 23.5% of S2 and 13.2% of S3 India names;
  - house numbers perturbed in 22% of India and 13% of US true pairs;
  - postcodes are rare (≈ 10% of US, ≈ 1% of India and France addresses).
- **Source-added tokens.** Each source systematically adds words the others don't use:
  - France: "Participations", "Holding", "Développement";
  - US: location prefixes ("Midtown", "Riverside");
  - India: unmapped transliterations ("limitet", "venchars").

### 2.2 Solution Strategy

**Approach type:** blocking + classifier. The pieces are hybrid retrieval, a trained cascade pruner, a two-stage gradient-boosted pair classifier and a metric-optimal decision rule.

**Core innovations:**
1. **Hybrid retrieval, measured at full scale.** Hash-join keys alone find 81% of true pairs. Exact sparse TF-IDF on a word view and a character n-gram view raise that to 97.4% before pruning.
2. **A trained cascade pruner** on cheap retrieval features only. It decides which pairs the expensive matcher ever sees, so it sets the candidate-set size.
3. **An expected-F0.5 decision rule tuned on the exact metric** over all 2.2M train entities. It counts every true match, including ones retrieval never found.
4. **Test-like validation without labels.** We rebuild train with 20% of S1 removed and all their records kept. This reproduces test's orphan structure, and we choose the decision rule on it.
5. **Per-country noise statistics computed the same way for every country.** Examples are the source-added tokens and the house-number conflict rate among confident pairs. The model trained on India/US only, so these let it adapt to France without labels and without any rule that names a country.

---

## 3. Candidate Generation (Blocking)

**Normalisation first** (`prep.py`, `normalize.py`, `translit.py`):
- **Canonical forms, identical for every country.** Every variant maps to one short form on both sides of a pair (street/str/st → st, private/pvt → pvt, avenue/av → ave, rue/r → rue).
- **Indic scripts** (nine, one ISCII-aligned table) are transliterated to Latin. Each word then snaps to the most frequent train-S1 word with the same phonetic skeleton. This lifted token-set name similarity on Indic true pairs from 6.9 to 90.4.
- **Output:** ≈ 20 derived fields per record: core name, legal forms, skeleton, initials, canonical address, landmark-free address, postcode, house number, street key. They go into a parquet cache.

**Blocking keys and retrieval** (`retrieval.py`; every key and index is built per country):
1. **Hash-join keys**, groups capped at ≤ 5 S1 and ≤ 50 candidates so chain names can't explode:
   - exact core name;
   - name skeleton + one of the two rarest address words;
   - house number + street word;
   - first skeleton token + house number.
2. **Exact sparse TF-IDF top-k**, candidate → S1 (each candidate has at most one true owner):
   - word view on "core name | canonical address", top-10;
   - character 3–4-gram view on the core name, top-8. This view catches typos and glued words the word view misses.
   - Terms in more than 1% of a country's records, or seen once, are dropped.
   - Search uses `sparse_dot_topn`, multi-threaded and exact.
3. **Union**, keeping each pair's retrieval features: cosines and ranks per view, which keys fired, per-S1 and per-candidate gaps.
4. **Cascade pruner** (`cascade_pruner.py`):
   - LightGBM on 19 retrieval-time features only (per view: cosine, rank, per-S1 and per-candidate gaps; key flags; pair counts);
   - pairs below the pruner-score cut are dropped before any pair feature is computed.

**Candidate pairs generated** (full scale):

| Stage | Train pairs | Pairs per S1 | Recall of all true train pairs | Test pairs (per S1) |
|---|---|---|---|---|
| Hash keys only | 16.4M | 7.4 | 0.808 | 14.5M (8.4) |
| Union with both TF-IDF views | 178.1M | 80.7 | 0.974 | – |
| **Final: pruner at 99.7% of retrieved positives** | **18.7M** | **8.5** | **0.971** | **18.6M (10.8)** |
| Evaluated alternative: stricter cut (score ≥ 0.01) | 13.7M | 6.2 | 0.968 | 13.6M (7.8) |

`output/candidate_pairs.tsv` holds exactly the final set above, the only pairs the matcher scores. That's 3 × 10⁻⁶ of the within-country all-pairs space. Test has more pairs per S1 than train because of its orphan records.

**How you ensured true matches were not lost:**
- Recall of *all* true train pairs, including ones never retrieved, is measured after every stage.
- The oracle F0.5 (a perfect matcher on our candidates) is reported for every run. At the 99.7% cut it is 0.987.
- We measured the candidate-set size vs F0.5 curve (Appendix B). A 27% smaller set (score ≥ 0.01, 7.8 per S1 on test) cost ≈ 0.0014 F0.5 on India held-out entities. On the leaderboard, together with a retrained model, it scored 0.955 against 0.958, so we kept the 99.7% cut.

---

## 4. Matching Model

**Features used** (`match.py`, 102 per pair; no feature reads the country label):
- **Name features:**
  - rapidfuzz token-set, token-sort, ratio, partial and Jaro-Winkler on the core name;
  - token-set on the phonetic skeleton; ratio on the no-space name;
  - per-country TF-IDF cosines on name words and name character 3-grams;
  - numbers inside the name; acronym match; name frequency within the country (a chain signal);
  - the same similarities recomputed after removing each country's **source-noise tokens**, found without labels as tokens one source uses far more often than another.
- **Address features:**
  - token-set / token-sort / ratio / partial on the canonical address, the landmark-free address and the address words;
  - per-country TF-IDF cosine on address words;
  - postcode, house-number and street agreement and conflict; overlap and conflict across all address numbers;
  - landmark flags; de-noised address similarity.
- **Legal form:** agreement / conflict between the extracted legal forms (Pvt Ltd vs LLP; SARL vs SAS).
- **Other:** field swap (name vs the other record's address), empty-field flags, lengths, Indic flag, source, and the retrieval features plus the pruner score.
- **Context:** rank and gap-to-best of key scores within the S1's candidates and within the S1 records competing for the same candidate.

**Model type:**
- **Stage 1:** LightGBM (MIT; 127 leaves, learning rate 0.08, CPU) and XGBoost (Apache-2.0; hist, GPU, the same leaf-wise shape), blended 50/50 (the weight is chosen on out-of-fold log-loss). Both use early stopping on 10% of the training entities.
  - Training is 5 folds grouped by S1 entity over all train pairs. Each fold's models train on 30% of the other folds' entities, so out-of-fold predictions cover every entity and the one-to-one competition is intact.
  - Evaluated alternative (v6): XGBoost alone on 100% of entities, on the smaller candidate set. Out-of-fold F0.5 improved by ≈ 0.002, but the leaderboard fell (0.955 vs 0.958). The most likely cause is the unseen country, where out-of-fold scores can't see, so we kept the blend.
- **Stage 2:** the same blend with 21 more features computed from stage-1 probabilities:
  - within the S1: rank, gap to best, runner-up, probability mass;
  - within the candidate: margin over the best competing S1;
  - agreement with the S1's other top candidates: name and address similarity, house-number and postcode agreement.
  - It uses the same folds as stage 1 (out-of-fold probabilities, no leakage) and raised out-of-fold F0.5 from 0.968 to 0.975.

**Threshold selection method:**
- Macro F0.5 per entity is optimised directly, on out-of-fold probabilities for all 2.2M train entities. Every true match counts, including ones retrieval never found.
- Two rule families are tried, each with and without one-to-one (each candidate goes only to its highest-probability S1):
  - a global threshold;
  - a per-entity expected-F0.5 rule: take the top-k candidates maximising 1.25·Σq_topk / (0.25·Σq + k) with q = p^α, or predict an empty list when Π(1 − q) is higher.
- **Chosen rule: expected-F0.5, one-to-one, α = 1.5.**
  - α = 1.0 is marginally best on ordinary out-of-fold data. α = 1.5 is best on the orphan rebuild of India (20% of S1 removed, all their records kept), which reproduces test's structure.
- The rule computes in float64. Test decisions originally ran in float32, which garbled per-entity sums; fixing that is worth +0.0005.

**Post-processing** (`postprocess.py --house-lr`; same procedure for every country):
- For each country we measure, without labels, how often house numbers conflict among its confident mutual-best pairs (p ≥ 0.9, both records carry a house number). Test values are India 15.8%, US 7.2%, France 0.85%; train values are 13.8% and 5.1%.
- Where a country's rate is **below the lowest rate of any training country**, conflicts are stronger evidence than the model learned. That country's conflicting pairs then have their odds multiplied by the ratio.
- Only France qualifies (× 0.167); India and US are untouched by construction.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** train out-of-fold uses India/US only (France has no labels); test-mix weights them 0.47 / 0.38.

| Version | Recall | Ceiling | India | US | Test-mix | Public LB |
|---|---|---|---|---|---|---|
| No-ML baseline (hash keys + one threshold) | 0.808 | 0.904 | 0.800 | 0.852 | 0.823 | – |
| + TF-IDF word view + pruner + LightGBM (v2) | 0.965 | 0.985 | 0.936 | 0.966 | 0.949 | 0.9365 |
| + de-noised features, LightGBM/XGBoost (v3) | 0.965 | 0.985 | 0.947 | 0.969 | 0.957 | 0.947 |
| + char n-gram view, legal-form feature, stage 2 (v5) | 0.971 | 0.987 | 0.968 | 0.983 | 0.975 | 0.9526 |
| **Final: v5 + float64 rule, α 1.5, house-number correction** | **0.971** | **0.987** | 0.968 | 0.983 | 0.975 | **0.958** |
| Alternative (v6): 27% fewer candidates, XGBoost on all entities | 0.968 | 0.986 | 0.968 | 0.984 | 0.975 | 0.955 |

- **Held-out check (India, 617k train entities the final models never saw):** 0.9683 at the 99.7% cut; the orphan rebuild costs 0.0018.
- **What moved the leaderboard, and why:** v5's out-of-fold gain (+0.018) produced only +0.006 on the leaderboard. Solving the public score for the unseen country (weights 0.47 / 0.38 / 0.15) implied France had dropped. Two unlabelled statistics explained it:
  - orphan records, which a more liberal rule wrongly assigns;
  - France's house numbers being almost never perturbed, so the model, which learned that conflicts are routine noise, kept wrong same-street pairs.
  - Correcting both gave +0.0054 on the leaderboard.
- **Common false positives (wrong merges):**
  - an orphan record assigned to the most similar remaining S1;
  - same name and street but a different house number where house numbers are reliable (France);
  - generic short names once legal forms are stripped.
- **Common false negatives (missed matches):**
  - true pairs never retrieved (3.2%): glued words, empty candidate addresses, generic names whose terms exceed the 1% document-frequency cap;
  - heavily garbled transliterations;
  - names reduced to initials.
- **Tried and not shipped:**
  - EM prior-shift correction: the model is calibrated at every level of candidate crowding, so it would double-count.
  - Glued-word splitting + label-mined spellings as a test-time normalisation: recall +0.2pp, but F0.5 flat without retraining.

---

## 6. Conclusion

- Measuring everything at full scale was the most valuable decision. Sample-scale results pointed at the matcher; full scale showed recall was the bottleneck.
- Once recall was fixed, the remaining leaderboard gap came from how test differs from train, not from model capacity. Unlabelled statistics measured that difference and informed corrections that were identical for every country: a test-like rebuild for choosing the rule, and per-country noise rates.
- A trained cascade pruner keeps the candidate set at 10.8 pairs per S1 on test, for about 3.5 true matches per S1, at 97.1% recall of all true train pairs. A 27% smaller set is one flag away (`--prune-threshold 0.01`) at a measured cost of ≈ 0.001–0.003.

---

## Appendix

### A. Code Artefacts

Code is in `code/business_entity_resolution/src/`; exact commands are in `code/business_entity_resolution/README.md`. Cache directory names (`norm-…`, `pairs-…`, `feat-…`) are printed in `match.py`'s first log line.

| Stage | Command | Output |
|---|---|---|
| Normalise (once) | `python src/prep.py --data-dir <dataset> --work-dir <work>` | parquet cache |
| Train, OOF, stage 2, test outputs | `python src/match.py --data-dir <dataset> --work-dir <work> --out-dir <run> --tfidf-k 10 --char-k 8 --char-max-df 0.01 --models lgb,xgb --device cuda --stage2` | models, rule, OOF, test probabilities, `candidate_pairs.tsv` |
| Decision + house-number correction | `python src/postprocess.py --run-dir <run> --norm-dir <work>/norm-… --out-dir output --rule expF:1.5 --house-lr <work>/feat-… --test-feat <work>/feat-…/test_features.parquet` | `output/matching_results.tsv`, `output/candidate_pairs.tsv` (identical to the submitted files, verified byte for byte) |

- **Modules:**
  - `normalize.py`, `translit.py`, `prep.py`: normalisation;
  - `retrieval.py`: keys and TF-IDF retrieval;
  - `cascade_pruner.py`;
  - `match.py`: features, models, driver;
  - `stage2.py`;
  - `decide.py`, `metric.py`: decision rules and the exact metric;
  - `postprocess.py`; `predict_test.py` (rescoring a finished run without training);
  - `build_token_maps.py`: optional, not used in the final run.
- **Dependencies:** LightGBM (MIT), XGBoost (Apache-2.0), scikit-learn, rapidfuzz, polars, sparse_dot_topn. No external data, APIs, geocoding or pretrained models. XGBoost runs on CPU if no GPU is present.
- **Reproducibility:** XGBoost's GPU training is not bit-for-bit deterministic, so a rerun reproduces the scores closely but not byte for byte.

### B. Additional Results

- **Runtime** (i7-14700, 28 threads, RTX 4080 Super):
  - normalisation ≈ 3–12 min (once);
  - train retrieval ≈ 80 min, mostly the character view;
  - stage 1 ≈ 30 min; stage 2 ≈ 17 min;
  - test retrieval + features ≈ 40 min; scoring and post-processing ≈ 7 min.
- **Candidate-set size vs F0.5** (India, held-out train entities, final models):

  | Pairs per S1 | F0.5 |
  |---|---|
  | 8.6 | 0.9679 |
  | 6.2 | 0.9665 |
  | 5.5 | 0.9650 |
  | 4.8 | 0.9617 |

  The loss accelerates past a ~30% cut. The submitted files use the 8.6 point (10.8 per S1 on test).

- **Predicted test structure (final submission):**

  | Country | Singleton rate | Matches per S1 |
  |---|---|---|
  | France | 5.2% | 3.34 |
  | India | 6.2% | 3.26 |
  | US | 5.5% | 3.40 |

  Train truth is 5.6% singletons and 3.46 matches per S1.
