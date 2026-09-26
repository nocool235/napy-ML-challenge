# Methodology draft (copy into the official Documentation_template.md)

Replace every `[..]` placeholder with your real numbers from `output/debug/metrics_train.json`.

## 1. Methodology overview
We treat the task as a two-stage entity-resolution problem. The first stage is high-recall candidate generation (blocking) and the second is a precision-oriented pairwise classifier. A decision layer then optimises the competition metric (per-entity F0.5, macro, singletons included) directly on out-of-fold predictions. The pipeline is country-agnostic, so it generalises to France, which appears only in the test set.

## 2. Normalisation
- We lowercase, strip accents (Unicode NFKD), map `&` to "and", and remove punctuation and dotted abbreviations (P.V.T. → pvt).
- **Symmetric canonical tokens**: every variant maps to one short form, and both sides of a pair use the same map. Examples: street/str/st/saint → st; avenue/av/ave → ave; private/pvt/pte → pvt; shree/shri/sri → sri. The dictionaries cover English, Indian and French variants.
- Names produce a *core* name with legal forms and stopwords removed. They also produce the legal-form set, initials (for acronyms), a no-space form, and alternative names split on DBA, trading-as and parentheses.
- Addresses produce a canonical address and a copy with landmark segments removed ("Near SBI ATM", "Opp. …"). They also produce postcodes (5 or 6 digits, with "110 001" joined and ZIP+4 truncated), the house number and numeric tokens. US and Indian state names and legacy Indian city names are canonicalised.
- No external data is used. All dictionaries are hand-written.

## 3. Candidate generation / blocking
- We use five TF-IDF views: name char 2–4-grams, address char 3–4-grams, name+address char 3–4-grams, name words, and postcode/house-number tokens plus name words.
- For each view we take the top-K cosine neighbours in **both directions**: from S1 to each of S2 and S3 separately, and from each S2/S3 record back to S1. We then take the union.
- On train, blocking reaches pair recall [..], oracle F0.5 [..] and [..] candidates per S1 (reduction ratio [..]).
- Blocking ranks per view are kept as features.

## 4. Features and model
- **Similarity** (rapidfuzz): ratio, partial, token-sort, token-set, WRatio, Jaro-Winkler and Levenshtein on the core name. Also no-space ratio, acronym match, best DBA-alternative match, token Jaccard, and the same family on the address (full, landmark-free, words only). TF-IDF cosine is computed for all five views.
- **Structure**: postcode, house-number and name-number agreement and conflict; legal-form conflict; lengths and missing fields; landmark flags; name frequency (for chain stores, where the same name means weak evidence); same_country; source.
- **Context**: rank and gap-to-best of the key scores within the S1 record's candidates and within the candidate's competing S1 records. S1 is deduplicated, so a candidate belongs to at most one S1.
- **Stage 1**: LightGBM binary classifier with 5-fold GroupKFold by S1 record, which gives out-of-fold probabilities.
- **Stage 2**: LightGBM on the stage-1 features plus probability-context features (rank, max, gap, second-best, sum, and count above 0.5, per S1 record and per candidate).
- Leave-one-country-out validation (train US → test India and the reverse) scores [..], compared with [..] for grouped CV. This is our France proxy.

## 5. Decision layer
- We tuned the rule on OOF against the exact metric. One rule is a global threshold. The other is the expected-F0.5 top-k rule: F0.5 = 1.25·TP / (0.25·|truth| + k), and predicting an empty set is worth P(no match) = Π(1 − p). Each rule was tried with and without the one-to-one constraint.
- Chosen: [rule, parameter], with OOF F0.5 [..] (singletons [..], matched entities [..]).

## 6. Experiments log
| # | change | OOF F0.5 | LOCO F0.5 | LB |
|---|---|---|---|---|
| 1 | baseline (this pipeline) | | | |

## 7. Licences and compute
LightGBM (MIT), scikit-learn (BSD-3) and rapidfuzz (MIT). There is no pretrained model. Everything trains on CPU in [..] minutes on [instance].
