# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [TBD — team name as registered]
**Team Members:** Adarsh Bhatt, Nakul, Prakhar, Yash [TBD — full names as registered]
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We match every Source 1 business to its Source 2 / Source 3 records with a retrieve–prune–classify–decide pipeline.
1. **Normalise:** country-agnostic rules, including rule-based Indic-script transliteration and phonetic skeletons.
2. **Retrieve:** capped hash-join keys, unioned with exact sparse TF-IDF top-10 search inside each country.
3. **Prune:** a LightGBM cascade pruner cuts the candidate pairs 6.5× while keeping 99.7% of the true pairs retrieval found.
4. **Classify:** LightGBM (optionally blended with XGBoost) scores the surviving pairs on ~90 features.
5. **Decide:** a per-entity decision rule maximises the expected macro F0.5 directly.

The key finding was that at full scale, candidate recall rather than the matcher was the binding constraint. Raising it from 0.81 to 0.965 lifted our out-of-fold F0.5 from 0.876 to 0.949.

---

## 2. Methodology

### 2.1 Problem Analysis

- **Scale.** Train has 2.21M S1 and 10.32M S2+S3 records; test has 1.73M S1 and 9.97M S2+S3. Comparing every S1 with every candidate would mean ≈17 trillion pairs on test, and ≈6.7 trillion even within the same country.
- **Structure (train).** 5.6% of S1 are singletons, with 3.46 true matches per S1 on average. No S2/S3 record belongs to two S1 records (checked over 7.64M true pairs), and no true pair crosses countries.
- **Country shift.** Train is US and India. Test adds France (15% of S1), which has no labels.
- **Noise we had to handle:**
  - abbreviations and legal-form variants (Pvt/Private, SARL/SAS/EURL);
  - DBA and trade names;
  - word-order changes and typos;
  - landmark-style Indian addresses ("Near SBI ATM");
  - postcodes are rare: in ~10% of US addresses and ~1% of India and France addresses;
  - **Indic script:** 23.5% of S2 and 13.2% of S3 India names.
- **Source-added tokens** (found during error analysis). Each source systematically adds words the others do not use:
  - France: S2/S3 append "Participations", "Holding", "Distribution", "Développement", "Groupe" to 2–4% of names each;
  - US: location prefixes ("Midtown", "Riverside");
  - India: English words transliterated back and never mapped ("limitet", "venchars").

### 2.2 Solution Strategy

**Approach Type:** Blocking + classifier (hybrid retrieval, a trained cascade pruner, gradient-boosted pair classifier, metric-optimal decision rule)

**Core Innovation:**
1. **Hybrid retrieval measured at full scale.** Hash-join keys alone found only 81% of true pairs (ceiling F0.5 0.904). Adding exact sparse TF-IDF top-10 retrieval raised that to 96.8%.
2. **A trained cascade pruner** keeps the pair volume tractable (110M → 17M train pairs).
3. **A vectorised expected-F0.5 decision rule** with a one-to-one constraint. It is tuned directly on the competition metric over all 2.2M train entities, not on pair-level accuracy.
4. **Unsupervised "source-noise" detection.** The same rule runs for every country, which is how the model copes with France without seeing it in training.

---

## 3. Candidate Generation (Blocking)

**Normalisation first** (`prep.py`, `normalize.py`, `translit.py`):
- **Canonical forms, no country branching.** Every variant of a token maps to one short canonical form, applied identically to both sides of a pair (street/str/st → st, private/pvt → pvt).
- **Indic scripts** (nine of them, one ISCII-aligned table) are transliterated to Latin. Each word then snaps to the most frequent train-S1 Latin word with the same phonetic skeleton ("कंसल्टेंसी" → "consultancy").

  This lifted name similarity on Indic-script true pairs from 6.9 to 90.4 (token-set).
- **Output:** ~20 derived fields per record: core name, legal forms, name skeleton, initials, canonical address, landmark-free address, postcode, house number, street key. They are written once to a parquet cache.

**Blocking keys used** (`retrieval.py`):
1. **Hash-join keys, per country**, each group capped at ≤5 S1 and ≤50 candidates so chain names don't explode:
   - exact core name;
   - name skeleton plus one of the record's two rarest address words;
   - house number plus street word;
   - first name-skeleton token plus house number.
2. **Exact sparse TF-IDF retrieval, per country:**
   - word TF-IDF on "core name | canonical address";
   - tokens in more than 1% of the country's records are dropped (they are expensive and uninformative), as are tokens seen once;
   - every S2/S3 record retrieves its top-10 S1 records by exact cosine (`sparse_dot_topn`, multi-threaded).

   We query in the candidate → S1 direction because each candidate has at most one true owner.
3. **Union** of both, with retrieval features kept (TF-IDF cosine and rank, which keys fired).
4. **Cascade pruner** (`cascade_pruner.py`):
   - LightGBM on 14 retrieval-time features: cosine, rank, key flags, and gaps and ranks within the S1 and within the candidate;
   - the threshold is chosen on held-out S1 records to keep 99.7% of retrieved true pairs.

**Candidate pairs generated** (full scale):

| Stage | Train pairs | Pairs per S1 | Recall of all true pairs | Test pairs |
|---|---|---|---|---|
| Hash keys only | 16.4M | 7.4 | 0.808 | 14.5M |
| Union with TF-IDF top-10 | 110.1M | 49.9 | 0.968 | 106.4M |
| **After cascade pruner** | **17.0M** | **7.7** | **0.965** | **16.8M** |

Reduction ratio vs. the same-country Cartesian product: ≈99.9999%. The pairs in `output/candidate_pairs.tsv` are exactly the post-pruner set the final model scores.

**How you ensured true matches were not lost:**
- Recall of *all* true train pairs, including ones never retrieved, is measured after every stage.
- The oracle F0.5, i.e. the score a perfect matcher would get on our candidates, is reported for every run. The final candidate set has oracle F0.5 = 0.985.
- The pruner's threshold is set against a recall target on held-out entities, not a fixed score.
- Its full recall-vs-reduction curve is logged: keeping 99.9% / 99.5% of retrieved true pairs keeps 20% / 13% of pairs.

---

## 4. Matching Model

**Features used** (`match.py`, ~90 per pair; nothing reads the country label):
- **Name features:**
  - rapidfuzz token-set, token-sort, ratio, partial ratio and Jaro-Winkler on the core name;
  - token-set on the phonetic skeleton, and ratio on the no-space name;
  - per-country TF-IDF cosine on name words and on name character 3-grams;
  - numbers inside the name (match / conflict);
  - initials (acronym) match;
  - name frequency within the country, as a chain signal;
  - the same similarities recomputed after removing each country's **source-noise tokens**, found without labels as tokens one source uses ≥5× more often than another.
- **Address features:**
  - token-set / token-sort / ratio / partial on the canonical address, the landmark-free address and the address words;
  - per-country TF-IDF cosine on address words;
  - postcode, house-number and street agreement and conflict;
  - overlap and conflict across *all* address numbers;
  - landmark flags;
  - de-noised address similarity.
- **Other:**
  - field-swap similarity (name vs. the other record's address);
  - empty-field flags, lengths, and the Indic-script flag;
  - source (S2 / S3);
  - the retrieval features and the pruner's probability.
- **Context:** rank and gap-to-best of key scores within the S1's candidates and within the S1 records competing for the same candidate. S1 is deduplicated, so each candidate has at most one true owner.

**Model type:**
- LightGBM binary classifier (MIT; 127 leaves, learning rate 0.08, early stopping).
- Optionally blended with XGBoost (Apache-2.0), with the blend weight chosen on out-of-fold log-loss.
- Training is 5-fold, grouped by S1 entity, over **all** train pairs, so the out-of-fold predictions cover every entity and the one-to-one competition between entities is preserved.
- Each fold's model trains on a 30% sample of the other folds' entities. The final model is trained the same way over all folds.

**Threshold selection method:** We optimise the macro F0.5 per entity directly, over two rule families, each with and without a one-to-one constraint (each candidate goes only to its highest-probability S1):
1. a global probability threshold;
2. a per-entity expected-F0.5 rule: take the top-k candidates maximising 1.25·Σp_topk / (0.25·Σp + k), and predict an empty list when P(no match) = Π(1−p) is higher.

Rules are scored on out-of-fold probabilities for all 2.2M train entities, counting every true match (including ones retrieval never found). The chosen rule is the expected-F0.5 rule with one-to-one.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** Train out-of-fold, reweighted to the test country mix (0.47 India / 0.38 US; France has no labels):

| Version | Candidate recall | Ceiling | India | US | Test-mix | Public LB |
|---|---|---|---|---|---|---|
| No-ML baseline (hash keys + one threshold) | 0.808 | 0.904 | 0.800 | 0.852 | 0.823 | – |
| Hash keys + LightGBM + expected-F0.5 rule | 0.808 | 0.904 | 0.833 | 0.929 | 0.876 | – |
| + TF-IDF retrieval + cascade pruner (v2) | 0.965 | 0.985 | 0.936 | 0.966 | 0.949 | 0.9365 |
| Final [TBD] | | | | | | |

- **Common false positives (wrong merges):**
  - different branches or businesses with the same generic name on the same street (chains);
  - candidates that two S1 records both claim with moderate probability, which the one-to-one constraint and competition features resolve only partly;
  - short, generic core names once legal forms are stripped.
- **Common false negatives (missed matches):**
  - true pairs whose house numbers disagree between sources (22% of India and 13% of US true matches with two house numbers);
  - names padded with source-added words;
  - names reduced to initials ("Europeen Gestion" vs "EG");
  - candidates with an empty address;
  - the 3.5% of true pairs retrieval never proposes, which cap recall.
- **France:** the public score implies France ≈ 0.86–0.89, below India/US. Spot-checks traced this mainly to the source-added name words, which motivated the de-noised features.

---

## 6. Conclusion

At full scale, recall rather than the matcher was the bottleneck. Our sample-scale experiments suggested the opposite, because hard negatives are about 100× sparser in a 1% sample; measuring everything at full scale was the most valuable decision we made. A hybrid of cheap exact keys and exact sparse TF-IDF retrieval, a trained pruner, a gradient-boosted matcher and a decision rule tuned on the exact metric took us from 0.823 to 0.949 out-of-fold. Everything is country-agnostic and label-free outside training, which is what lets the pipeline run on France unchanged.

---

## Appendix

### A. Code Artefacts

The code lives in `code/business_entity_resolution/`, with all source in `src/`. See `README.md` there for exact commands.

| Stage | Command / module | Output |
|---|---|---|
| Normalise (once) | `python src/prep.py --data-dir <dataset> --work-dir <work>` | parquet cache of normalised records |
| Everything else | `python src/match.py --data-dir <dataset> --work-dir <work> --out-dir <output>` | `output/matching_results.tsv`, `output/candidate_pairs.tsv`, `output/debug/` |

Modules: `normalize.py` and `translit.py` (normalisation), `retrieval.py` (hash keys and TF-IDF retrieval), `cascade_pruner.py`, `match.py` (features, models, driver), `decide.py` (decision rules), `metric.py` (exact metric), `utils.py`.
Models: LightGBM (MIT), optional XGBoost (Apache-2.0). No external data, APIs, geocoding or pretrained models; everything runs on CPU.

### B. Additional Results

- **Runtime** (full data, 28-core CPU): normalisation ≈12 min (once); the full `match.py` run ≈67 min (retrieval ≈19 min train + 10 min test; LightGBM 5-fold ≈23 min). Reproduction timing on 16 CPUs: [TBD].
- **Cascade pruner tradeoff** (held-out train entities): share of pairs kept at 99.9% / 99.8% / 99.5% / 99% of retrieved true pairs = 20.3% / 17.2% / 13.5% / 11.3%.
- **Predicted test structure** (v2): singleton rate 4.0% / 5.2% / 5.0% and 3.31 / 3.22 / 3.30 matches per S1 for France / India / US. Train truth is 5.6% and 3.46.
