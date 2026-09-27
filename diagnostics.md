# Diagnostics — `output_v3` (LB public 0.947)

> **Snapshot of the v3 run (LB 0.947), written 27 Sep ~17:00.** Later findings, including the float64 decision-rule fix, orphan records, France's house-number reliability and the final v6 pipeline, are in `EXPERIMENTS.md` (rows 7–9) and `Documentation_template.md`.

Scope: read-only analysis of the **v3** artifacts (feature_version `f4`, LightGBM+XGBoost blend, no
stage2 — the latest *complete* run; `output_v4_stage2` and `output_v5` are still running and excluded).
All numbers below are either pulled directly from `output_v3/debug/metrics_train.json`,
`output_v3/debug/decision_rules.csv`, `output_v3_run.log`, or computed this session with short,
single-purpose scripts (a few seconds to ~80s each) over the already-materialized cached parquet files —
no retrieval, training, or full-scale recomputation was run.

> **Proxy notice.** Items 3 and 5 need per-pair *model probabilities* that aren't saved for v3 (only the
> final blend's aggregate scores are in `metrics_train.json`). They use a precomputed **XGBoost-only f4
> OOF** (`oof1_xgb.npy`, 5-fold, 30% train-frac, GPU, identical 93 f4 features to v3) as a close proxy for
> v3's shipped LightGBM+XGBoost blend. v3's own log shows LightGBM-alone and XGBoost-alone OOF scores
> within **0.0001** of each other and of the blend at the shipped rule, so the proxy is tight — confirmed
> again below (§3: proxy's "all" score 0.9603 vs. shipped blend's 0.9605) — but it is **not** byte-identical
> to the shipped blend's actual probabilities. Item 2's "best cosine" is a **separate, non-model** proxy
> (see §2) — it does not use the OOF probabilities at all, since a missed pair has no retrieval-time score
> or model score by definition.

---

## 1. Retrieval: recall, sources, K/params

**Retrieval recipe (v3, from `retrieval.py` + the call in `match.py`, confirmed against
`metrics_train.json`):**
- Hash-join keys: 4 types — `name`, `street`, `hskel` (name-skeleton + house number), `skel` (rare address
  word + name-skeleton), each scoped by country. Oversized groups dropped: >5 S1 records or >50 S2/S3
  records per key (`HASH_CAP_S1=5`, `HASH_CAP_C=50`).
- TF-IDF word view: column `name_addr`, `analyzer="word"`, ngram (1,1), `min_df=2`, `max_df=0.01`, **top-k=10**
  (`tfidf_k` in `metrics_train.json`) per S2/S3 record, scored against S1 within each country.
- v3 predates this session's char/addr/name_nospace extra views (`EXTRA_VIEWS` in `match.py`) — v3's only
  retrieval channels are the hash keys and this one TF-IDF word view.
- A cascade pruner is then applied on the hash∪tfidf union, targeting **99.7% recall** (`prune_recall`).
  The cached `train_pairs.parquet` (16,979,401 rows, matches the run log exactly) is the **post-pruner**
  surviving set — the same set the model was trained on. All recall numbers below are therefore measured
  **after** pruning, matching the pipeline's own reported `pair_recall = 0.9650`. The run log doesn't show
  a separate pre-prune recall for v3 because its pairs were loaded from cache (cache hit skips those log
  lines) — a genuine limitation of this log, not fabricated.

**Recall of true train pairs, overall / by country / by source (S2 vs S3):**

| Split | True pairs | Found | Recall |
|---|---:|---:|---:|
| **Overall** | 7,638,365 | 7,370,698 | **96.50%** |
| US | 4,578,522 | 4,496,744 | 98.21% |
| India | 3,059,843 | 2,873,954 | 93.92% |
| S2 (all countries) | 3,693,619 | 3,579,978 | 96.92% |
| S3 (all countries) | 3,944,746 | 3,790,720 | 96.10% |
| US × S2 | 2,213,074 | 2,171,373 | 98.12% |
| US × S3 | 2,365,448 | 2,325,371 | 98.31% |
| India × S2 | 1,480,545 | 1,408,605 | 95.14% |
| India × S3 | 1,579,298 | 1,465,349 | 92.78% |

India is the weak spot (93.9% vs US 98.2%), and within India, S3 is worse than S2 (92.8% vs 95.1%).

**Which channel found each true pair (of the 7,370,698 found), overall / by country / by source:**

| Channel | Overall | India | US | S2 | S3 |
|---|---:|---:|---:|---:|---:|
| Both (hash **and** tfidf) | 6,109,695 (82.9%) | 2,110,982 | 3,998,713 | 3,004,961 | 3,104,734 |
| TF-IDF only | 1,201,851 (16.3%) | 713,408 | 488,443 | 552,126 | 649,725 |
| Hash only | 59,152 (0.8%) | 49,564 | 9,588 | 22,891 | 36,261 |

The TF-IDF word view is doing most of the unique work — hash-only finds are rare (0.8%) — and it's
disproportionately carrying India (713k tfidf-only finds there vs 488k in US).

**Hash key-type breakdown**, among the 6,168,847 true pairs found via *any* hash key (bits can overlap —
a pair can be found by more than one key type):

| Key type | Pairs hit | Share of hash-found | Pairs found by **only** this key |
|---|---:|---:|---:|
| `name` | 3,165,974 | 51.3% | 255,435 |
| `street` | 3,337,698 | 54.1% | 630,480 |
| `hskel` | 3,361,141 | 54.5% | 277,386 |
| `skel` (rare-word) | 4,226,555 | 68.5% | 641,737 |

`skel` (rare address word + name-skeleton) is the single strongest hash key, both in overall coverage and
in unique contribution (641,737 pairs no other hash key type would have found).

**Candidates per covered S2/S3 record** (mean # of S1 candidates retained per record that got ≥1
candidate, post-pruner — this is the retrieval density the matcher's features are built on):

| Split | Mean cand/record | Coverage (≥1 candidate) |
|---|---:|---:|
| Train overall | 1.65 | 99.44% (10,261,921 / 10,320,219) |
| Train — India | 1.75 | 98.71% |
| Train — US | 1.59 | 99.92% |
| Test overall | 1.70 | 99.01% (9,870,830 / 9,969,589) |
| Test — France | 2.54 | 98.76% |
| Test — India | 1.65 | 98.37% |
| Test — US | 1.45 | 99.89% |

Very lean candidate sets (the pruner is aggressive by design, targeting 99.7% *retained* recall at minimum
cost) — consistent with 0.6–1.6% of records getting zero candidates at all, which is a hard ceiling on
oracle recall independent of the matcher (see §3).

---

## 2. 40 true pairs retrieval missed entirely

**267,667 of 7,638,365 (3.50%)** true train pairs never appear in `train_pairs.parquet` at all — not
retrieved by hash or TF-IDF, or retrieved but then cut by the cascade pruner (the cached table is
post-pruner, so this 3.50% is "never made it to the model," not purely "never retrieved"). Breakdown:
India 185,889 (69.4% of misses), US 81,778 (30.6%) — consistent with India's lower §1 recall.

**Similarity metric used:** `rapidfuzz.fuzz.token_set_ratio` on the combined `name_addr` field (0–100
scale). This is one of the pipeline's own existing features (`STRING_FEATURES` in `match.py`), computed
directly pairwise on just the 40 sampled pairs — cheap and instant. We deliberately did **not** refit a
per-country TF-IDF vectorizer to get an exact `tf_cos`-comparable cosine: that would replay the same
corpus-scale computation retrieval itself does (fitting over the full India/US corpus), which is explicitly
out of scope while the v5 job owns the CPU/GPU.

Sample: 40 pairs, stratified proportionally by country (28 India, 12 US, seed=42). Mean similarity on this
sample is **84.3** (median 86.8, min 53.3, max 100) — surprisingly high. Many misses are near-exact
name/address matches that still fell through, most visibly because of (a) generic/common tokens capped by
`HASH_CAP` or excluded by `max_df=0.01` (chain-like names: "svc", "estate", "agro", "ctr"...), and (b) an
**empty candidate address**, which removes street/house hash keys and dilutes the combined `name_addr`
TF-IDF signal — visible in several rows below where `c_addr` is blank despite a near-identical name.

| # | Country | Src | S1 name | S1 addr | Cand name | Cand addr | Sim |
|---|---|---|---|---|---|---|---:|
| 1 | US | S2 | mohammed partners | 607 ct st rockwell city ia | mohammedpartners | 607 coudt st twin lakes township ia | 70.0 |
| 2 | US | S2 | rss ace markets | 152 carrie st houston tx | rssacemarkets | houston township 15 carrie st tx | 84.4 |
| 3 | US | S3 | cardiology partners | 5719 arlington park dr dallas tx | cardiology pmasrtdfers | *(empty)* | 62.5 |
| 4 | US | S3 | myrle ortiz dynamic dpc | 21 lake view dr queensbury ny | dynamicmyrleortiz | lake view dr queensbury ny | 91.1 |
| 5 | US | S3 | modern consumer dynamics | 1391 chadbourne dr davis il | modern consumer | *(empty)* | 100.0 |
| 6 | US | S3 | red pub | ma bridgewater 90 goodwater way | red pub ent | *(empty)* | 77.8 |
| 7 | US | S2 | family ctr | 1705 high ridge nelson county va | family cdnetr | *(empty)* | 63.2 |
| 8 | US | S2 | regional fellowship | 22407 lewes georgetown hwy georgetown | regional fellowship | *(empty)* | 100.0 |
| 9 | US | S2 | womens health specialists | 710 university st walla walla wa | womens health specialists ent | *(empty)* | 92.6 |
| 10 | US | S3 | chacon osornio | 1205 veterans way ste apt 413 carmel in | fayepyranyla | carmel veterans way in | 78.7 |
| 11 | US | S3 | shipley quantumsphere | cincinnati oh 17 bradford pl | zetacalodrex | 17b bradford pl cincinnati oh | 76.1 |
| 12 | US | S2 | kime safe health | 516 ct st dallas or | kimesafe | 16 ct st dallas or | 86.6 |
| 13 | India | S3 | curative bros | 3 4 rajendar illam vedammal ave dr subrayan nagar kodambakkam chennai tn | brosprivatecurative | c 3 4 chennai chennai tn | 59.3 |
| 14 | India | S3 | vijay tech | gj ahmedabad raj shops appartments b h sharma corner vishal nagar rd is anpur 2 ahmedabad | vijaytechnologies | 2 ahmedabad hq region ahmedabad gj | 53.3 |
| 15 | India | S3 | soln ayurveda soln | wb poddar nagar ps jadavpur kolkata kolkata howrah kolkata 7 9 | ayurvedasolutions | 90 7 9 kolkata wb | 60.4 |
| 16 | India | S2 | sai svc | 3 382 bommanahal bommanahal v mandal ananthapur ap | sai svc | *(empty)* | 100.0 |
| 17 | India | S3 | adj public school | 14 flr dr gopal das bhawan 28 barakhamaba rd delhi delhi | adj school svc | *(empty)* | 83.3 |
| 18 | India | S3 | hari builders | tc 24 3088 2 ushasandya bldg kowdiar devasom board rd kowdiar thiruvananthapuram thiruvananthapuram trivandrum kl | hari bildezhs | tc 24 3088 2 trivandrum kl | 88.0 |
| 19 | India | S2 | dynamic sys | c 4 2 ground flr safdarjungdevelopmentarea delhi s w delhi delhi | dynamic sys | c 4 2 s w delhi delhi delhi | 100.0 |
| 20 | India | S3 | royal engineering | 154 a veer savarkar nagar mini bypass rd p o izzatnagar bareilly up | royal engineering | a bareilly up | 100.0 |
| 21 | India | S3 | md commerce | g 5 6 sec 10 dwarka delhi s w delhi delhi | mdcommerce | g 5 6 sec 10 dwarka delhi s w delhi delhi | 94.1 |
| 22 | India | S3 | maa infrastructure | b3 mohanam wing 1 t ragamalika flats thiruvengadam st ra puram chennai tn | maa infrastructure | b3 chennai chennai tn | 100.0 |
| 23 | India | S3 | tirupati investments | ts hyderabad 20 2 82 hyderabad hussaini alam old kabutar khana | tirupati investments | 20 2 82 bangalore ts | 86.8 |
| 24 | India | S3 | intl advanced exports | 15 32 ground flr w patel nagar delhi w delhi delhi | internationaladvancedexports | 240 15 32 ground flr w patel nagar delhi w delhi delhi | 82.4 |
| 25 | India | S3 | all food | 117 52 perumal koil st rajankuppam ayanambakkam poonamallee tiruvallur poonamallee tiruvallur tn | al pet | 117 5 poonamallee tn | 81.6 |
| 26 | India | S3 | future marketing | 140 a 2 flr block saurabh vihar badarpur delhi s delhi delhi | fyuchar marketing | delhi s delhi delhi 40 a | 79.2 |
| 27 | India | S3 | life estate | level 8 umiya business bay tower 1 cessna business park kadubeesanahalli marthahali bangalore ka | life estate | level 4 8 bangalore ka | 97.1 |
| 28 | India | S2 | sai developers | pune city pune s 43 12 301 kudale skyone near sarisha society jambhe pimpri chinchwad mh | shyam developers | s 43 12 301 pune city pune mh | 92.5 |
| 29 | India | S3 | real consultancy | ste a 2 flr cnergy bldg appa saheb marathe marg mumbai mh | real ctr | ste a mumbai mumbai mh | 91.7 |
| 30 | India | S3 | white estate | astral 207 1 behind rajpath club off s g hwy ahmedabad daskroi ahmedabad gj | vhait estet | gj thaltej daskroi astral | 65.5 |
| 31 | India | S2 | guru shivam it | f 17 iind flr 213 harsha complex subhash chowk laxmi nagar delhi e delhi delhi | guru shivam aiti | f 17 divreportingcircle e delhi delhi | 68.4 |
| 32 | India | S3 | guru power | tn 15 madurai jawahar rd chokkikulam | guru power | *(empty)* | 100.0 |
| 33 | India | S2 | digital marketing | e 52 block nandgram tyagi mkt village morta ghaziabad ghaziabad up | digital marketing | e 52 ghaziabaad ghaziabad up | 87.1 |
| 34 | India | S3 | tech healthcare | 7 navjeevan chambers g 4 vinoba marg c scheme jaipur rj | tek healthcare | 7 c scheme jaipur rj | 94.3 |
| 35 | India | S3 | great agro | shiva shankarappa sampigehally bangalore s bangalore ka | great agro | *(empty)* | 100.0 |
| 36 | India | S3 | my construction | 1 n 48 49 old blk b 1 gali 19 kiran garden uttam nagar delhi w delhi delhi | my construction | *(empty)* | 100.0 |
| 37 | India | S3 | supreme agro | s 9 2ndfloor whitehousest marksroad bangalore ka | supreme agro | 3 s 9 bangalore ka | 96.9 |
| 38 | India | S2 | fortune power | 43 1 flr pkt 3 dda flats sec a 9 delhi n w delhi delhi | forchyun power | block a 904 43 delhi n w delhi delhi | 69.8 |
| 39 | India | S2 | skylar | 94 ugf shanti kunj main mustafil 24 killa 25 vasant kunj vill mehrauli sec 3 4 delhi delhi | stgynar | 94 delhi delhi | 71.4 |
| 40 | India | S3 | hyderabad svc | 8 2 269 5 20 apt 302 20 sagar society banjara hills hyderabad ts | hyderabad svc ent | *(empty)* | 86.7 |

(None of these 40 had a postcode on either side, so the postcode column was dropped from the table.)

---

## 3. Oracle vs OOF F0.5, per country and singleton vs matched

Oracle and "OOF all-groups" rows below are the **exact, shipped** numbers from `metrics_train.json`. The
singleton/matched split isn't in that file and is computed here: oracle via `decide.per_s1_f05` on
`y==1` (a perfect matcher keeping every true candidate pair), OOF via the same function on the **XGBoost
proxy** OOF probabilities run through the shipped rule (`expF`, α=1.25, o2o=True). Singleton = S1 records
with 0 true matches (123,247 of 2,206,821 train S1 = 5.58%, matching `structure()`'s own "train truth: 5.6%
singletons" comment); matched = the other 2,083,574 (94.42%).

| | India | US | Mix (0.47/0.38) | All |
|---|---:|---:|---:|---:|
| **Oracle — all S1** | 0.9770 | 0.9946 | 0.9848 | 0.9875 |
| **Oracle — singleton** | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Oracle — matched** | 0.9756 | 0.9942 | 0.9840 | 0.9868 |
| **OOF (shipped blend, exact) — all S1** | 0.9472 | 0.9695 | 0.9571 | 0.9605 |
| **OOF (XGB proxy) — all S1** | 0.9468 | 0.9693 | 0.9568 | 0.9603 |
| **OOF (XGB proxy) — singleton** | 0.9031 | 0.9533 | 0.9256 | 0.9332 |
| **OOF (XGB proxy) — matched** | 0.9493 | 0.9702 | 0.9587 | 0.9619 |

The proxy's "all S1" row (0.9603 mix... wait "all" column 0.9603) sits within 0.0002 of the shipped blend's
0.9605 — validates using it for the split below.

**Headline finding:** singletons are trivial for the oracle (1.0, by construction — predicting "no match"
scores 1 whenever there's truly nothing to match) but the *hardest* group for the actual model: OOF mix
F0.5 on singletons is **0.9256**, a **7.44pp** drop from oracle, vs. only a **2.53pp** drop for matched
entities (0.9840 → 0.9587). The oracle-to-OOF gap on singletons is ~3x the gap on matched entities — the
model is over-predicting matches for a meaningful share of true singletons, and that's the single largest
lever visible in this OOF/oracle comparison, larger than the India-vs-US gap itself.

---

## 4. Train vs test at inference

**Test-side numbers** are pulled as-is from `metrics_train.json`'s `test_structure` (identical to the run
log's `test <country>: {...}` lines) — computed on the **final, full-data-trained** LightGBM+XGBoost blend.

**Train-side numbers** are computed this session two ways:
- *Candidates per covered S2/S3 record* — a pure retrieval-table stat, from `train_pairs.parquet` /
  `test_pairs.parquet` directly (no model needed) — already shown in §1, repeated here for comparison.
- *Predicted singleton rate / mean matches per S1 / share unassigned* — these are **decision-rule-applied**
  stats (`match.structure()`), so train needs *some* per-pair probability to apply the rule to. There is no
  saved train-side blend probability for v3, so this uses the same **XGB OOF proxy** as §3, run through the
  identical shipped rule via `decide.rule_mask` (i.e., `match.structure()` called on the OOF-proxy-kept
  pairs) — flagged again here since it's the same proxy caveat as §3, applied to a new quantity.

| Country | Split | S1 count | Pred. singleton rate | Mean matches/S1 | Share S2/S3 unassigned | Cand/covered record |
|---|---|---:|---:|---:|---:|---:|
| France | Test (final blend) | 259,452 | 4.47% | 3.276 | 40.77% | 2.54 |
| India | Test (final blend) | 809,986 | 5.67% | 3.153 | 45.86% | 1.65 |
| US | Test (final blend) | 663,106 | 5.35% | 3.263 | 43.31% | 1.45 |
| India | Train (OOF, XGB proxy) | 883,188 | 6.17% | 3.120 | 33.33% | 1.75 |
| US | Train (OOF, XGB proxy) | 1,323,633 | 5.90% | 3.221 | 31.08% | 1.59 |

France has no train-side row (no labels; test-only, as expected — not a bug).

**Observations:**
- Predicted singleton rate is *higher* on train (6.17%/5.90%) than on test (5.67%/5.35%) for both
  countries — plausible given the weaker XGB-only proxy vs. the full blend, and OOF (5-fold held-out)
  vs. a single final model fit on all the data, but not fully separable from a genuine train/test shift
  with this data alone.
- The bigger gap: **share of S2/S3 unassigned is much lower on train (31–33%) than test (43–46%)** for
  both India and US — train leaves noticeably fewer candidate records unmatched. Candidate density is
  *not* the explanation (train and test candidates/record are close: 1.75 vs 1.65 India, 1.59 vs 1.45 US
  — train is if anything slightly *denser*, which would predict train having *more* unassigned headroom to
  fill, not less). This gap is flagged rather than explained: it could be a proxy/blend artifact (XGB-only
  OOF vs. the shipped LGB+XGB final model), an OOF-vs-final-fit calibration difference, or a genuine
  train/test distribution shift. **Worth a follow-up with the real blend's train OOF once available** —
  this diagnostic can't fully attribute it from what's on disk for v3.

**Confirmation that test used the identical retrieval path and decision rule as train (explicitly checked,
not assumed):**
1. **Code path**: `match.py`'s `build()` is called once for train (line 592) and once for test (line 674)
   with the same `a.tfidf_k`, `a.prune_recall`, `extra_k`/`extra_max_df` arguments — same function, same
   knobs, both runs of the same process.
2. **Decision rule**: the `rule` object applied to test (`decide.rule_mask(rule, ti, tj, tp)`, line 686) is
   literally the `best` dict `tune_vec` returned from train's OOF grid search (line 618/641) — not a
   separately-tuned or hand-set value. `metrics_train.json`'s `oof.rule/param/o2o` = `expF / 1.25 / True`
   matches the run log's `OOF best rule: {...}` line exactly.
3. **Reproduction (done this session)**: reloading `output_v3/debug/test_probs.parquet` (the actual test
   pair probabilities) and re-applying `decide.rule_mask({"rule":"expF","param":1.25,"o2o":True}, ...)`
   reproduces `metrics_train.json`'s `test_structure` numbers **exactly** (to the 4th decimal place shown),
   confirming test inference used byte-identical rule-application code, not a re-derived or approximated
   one.

---

## 5. False positives / false negatives (OOF XGB proxy)

Top 30 by probability in each direction (proxy caveat from the top of this report applies — this is the
XGB-only f4 OOF, not the shipped blend).

**30 false positives** (`y==0`, highest proxy probability, p ≈ 0.99998–0.99999): overwhelmingly
**US (26/30)**, India 4/30. Pattern: S1 and the negative candidate have essentially identical name+address
text, just with word order shuffled or a city/state clause moved (e.g. "griffin central, 2416 cadwallader
sonk rd cortland oh" vs "griffin central, oh 2416 cadwallader sonk rd bazetta twp"). These read as **true
near-duplicates** — plausibly different real branches of the same business, or label noise in ground truth
— rather than a model mistake; the model is arguably right that these *look* identical.

| p | Country | S1 name | S1 addr | Cand name | Cand addr |
|---:|---|---|---|---|---|
| 0.999994 | US | griffin central | 2416 cadwallader sonk rd cortland oh | griffin central | oh 2416 cadwallader sonk rd bazetta twp |
| 0.999994 | US | roos thompson clinic | 23289 aberdeen ct foley al | roos thompson clinic | 23289 aberdeen ct n a foley al |
| 0.999993 | US | houston pediatric valley ctr | 5311 mindy park ct houston tx | houstonpediatric | 5311 mindy park ct houston tx |
| 0.999992 | US | vogel family practice | 150 cushman dr farmington me | vogel family practice | 150 cushman dr farmington me |
| 0.999991 | US | better continental sys | 9210 heather dr tempe az | better continental sys | tempe 9210 heather dr az |
| 0.999990 | US | sarius | 1012 royal st george dr naperville il | sarius | 1012 royal st george dr naperville il |
| 0.999990 | US | bolden melanson | 1323 kelly trail main isabella | bolden melanson | 1323 kelly trl isabella main |
| 0.999989 | US | foot ankle continental medicine | 3458 bexvie ave columbus oh | foot ankle continental mldifcbne | 3458 bexvie ave columbus oh |
| 0.999989 | India | om investments | manish kumar lp 873 ladpur jhajjar hr | om investments | manish kumar lp 873 ladpur jhajjar hr |
| 0.999989 | US | slate | 1015 s ave independence mo | slate | 1015 s ave mo independence |
| 0.999987 | US | cws paloma | 4262 college ave corning ny | cws paloma | 4262 college ave corning ny |
| 0.999986 | US | temple presbyterian church | 1505 est ln lake forest il | temple presbyterian church | 1505 est ln il lake forest |
| 0.999986 | US | lloyd poore | 498 olde mill dr westerville oh | lloydpoore | 498 olde mill dr westerville oh |
| 0.999986 | India | foundation tech | a 203 ashok nagar delhi n delhi delhi | foundation tech | delhi a 203 ashok nagar delhi n delhi |
| 0.999985 | US | great guild | 40308 central ave phoenix az | great guild | 40308 central ave phoenix az |
| 0.999985 | US | lenet partners el paso county | 481 borrego rd el paso county tx | lenet partners el paso county | 481 borrego rd san elizario tx |
| 0.999985 | US | katerine covarrubias massage | 3444 slater rd ferndale wa | katerine covarrubias svc | 3444 slater rd ferndale wa |
| 0.999984 | US | dent tattoo | 59 bell st staten island ny | dent tattoo | 59 bell st staten island ny |
| 0.999984 | US | annamaria singleton cslm | 2519 30 pl tulsa ok | annamaria singleton cslm | 2519 30 pl ok tulsa |
| 0.999984 | US | dynamic broadcast products | 536 vesely ct eastlake oh | dynamic broadcast products | 536 vesely ct eastlake oh |
| 0.999984 | India | neelkanth seva samiti | 3483 1 narang colony trinagar delhi | neelkanth seva samiti | 3483 1 trinagar delhi delhi |
| 0.999982 | US | kerr college | 14 hudson st ste 3r worcester ma | kerr college | 14 hudson st worcester ste 3r ma |
| 0.999982 | US | behavioral health smart ctr | 2112 120 st blue island il | behavioral health smart ctr | blue island 2112 120 st il |
| 0.999981 | India | red foundation | rv 47 indus town village ratanpur rd huzur bhopal mp | red foundation | rv 47 indus town village ratanpur rd huzur bhopal mp |
| 0.999981 | US | lindsey prairie aerospace | 100 riviera dr babylon ny | prairie lindsey aerospace | 100 riviera dr babylon ny |
| 0.999980 | US | yee qdro katy | 4114 palmer meadow ct katy tx | yee qdro katy | tx 4114 palmer meadow ct katy |
| 0.999980 | US | boclaris | 613 main st canyonville or | boclaris | 613 main st canyonville or |
| 0.999979 | US | beth methodist church | brighton 2270 nelson rd tn | beth methodist church | 2270 nelson rd brighton tn |
| 0.999979 | US | global learning studios | 173 tusculum sq dr nashville tn | global learning studios | 173 tusculum sq dr nashville city tn |
| 0.999977 | US | delta capital | 403 meadow ln kings mountain nc | delta capital | 403 meadow ln kings mountain nc |

**30 false negatives** (`y==1`, lowest proxy probability, p ≈ 3e-6–8.6e-5): mostly **India (20/30)**, US
10/30. Pattern: the candidate's name is heavily garbled/obfuscated relative to S1 (looks like deliberately
injected noise: "bay long" → "8ay ctr", "tx lab" → "tx svc", "liberty college" → "dovahalo 39164"), and the
candidate's address is very often **blank**. With both fields degraded, the model genuinely has almost
nothing to match on — these read as hard, possibly adversarially-noised cases rather than a fixable model
bug.

| p | Country | S1 name | S1 addr | Cand name | Cand addr |
|---:|---|---|---|---|---|
| 0.0000030 | India | blue svc | 103 c3 bhimashankar nagar sec 19a nerul thane mh | servicesprivate | block b 882 7 103 c3 n a thane mh |
| 0.0000126 | India | classic soln | 1019 jlpl sec 82 sas nagar mohali pb | solutionsprivate | 1019 mohali pb |
| 0.0000187 | India | eluru biotech | g v sobha rani 4 62a satrampadu eluru town eluru w godavari ap | eluru svc eluruserv | *(empty)* |
| 0.0000226 | India | seven estate | 202 sec 15 indira nagar lucknow up | xylomirasyn one | lucknow up lucknow 202 sec 15 indira nagar |
| 0.0000238 | US | rays painting | 309 town green way reisterstown md | rays servlce | *(empty)* |
| 0.0000251 | India | gujarat finance | 61 ring rd lajpat nagar iii delhi delhi | phriavte gujarat finance | 61 eros plz faridabad hr delhi |
| 0.0000307 | India | jh foundation | 158 okhla ind ph i delhi s delhi delhi | jh svc | *(empty)* |
| 0.0000333 | US | zephyr | 52 elm st montgomery wv | zephyr ctr ent | *(empty)* |
| 0.0000393 | India | sumiran molecules | 171 c 1 picnic garden rd kolkata howrah wb | sri sumiran svc | *(empty)* |
| 0.0000415 | US | 24hr fitness | 105 wa st mt carroll il | 24hr partners | *(empty)* |
| 0.0000420 | US | continental network | 222 pelham st ste 201 methuen ma | pyrafluxdelta | 22 pelham st pmb 7481 methuen ma |
| 0.0000426 | US | bay long | wallkill ny 219 17k | 8ay ctr | 17k bloomingburg ny |
| 0.0000454 | US | liberty college | in 1136 whitcomb ave indianapolis ste apt | dovahalo 39164 | whitcomb ave indianapolis township in |
| 0.0000457 | India | lords tele | 252 gf haque manzi samanpura phulwari patna br | sri ectopyra | 252 gf patna urmila kunj br |
| 0.0000464 | India | samriddh partners | 8 flr k 1 sec v salt lake city block ep gp rdb blvd wb | nexectokelo | rdb blvd block ep gp salt lake kolkata wb |
| 0.0000497 | India | bright sons | 1 721 near marottichuvadu junction edappally kanayannur taluk ernakulam kl ernakulam kl | fluxavi | 1 721 ernakulam thattampady n h sventeen kl |
| 0.0000531 | US | select asset laboratories | bella vista 36 carroll dr ar | select asset svc selectas | *(empty)* |
| 0.0000534 | India | future soln | 10 old 15 3 flr 35 st saidapet kanchipuram tn | hpiyuchchar cholyushans limitet | c 10 old 15 3 flr 35 st thiruvallur saidapet tn |
| 0.0000537 | India | lh infertility | 51 new ashok nagar delhi w delhi delhi | lh ctr | *(empty)* |
| 0.0000542 | India | esteem poultries | 63 ram vihar delhi e delhi delhi | esteem svc | *(empty)* |
| 0.0000561 | India | umiya technologyprivate | 32 ezra st 13 ground flr kolkata howrah wb | sri umiya ctr | *(empty)* |
| 0.0000581 | India | consultancy onrise info public | 1557 a 2 gala 2 bldg a 1 flr ganraj complex bhiwandi thane mh | halolumcalo | block c 59 1557 a 2 greater mumbai mh |
| 0.0000616 | India | m s it | gf 103 maddilapalem visakhapatnam vishakhapatnam ap | sri itmsprivate | gf 10 visakhapatnam vishakhapatnam ap |
| 0.0000625 | India | nagpal investigation | 8 2 flr shivpuri colony manikonda rangareddy ts | sri nagpal svc 7895948886 | *(empty)* |
| 0.0000675 | US | pritchett fuel | 949 1 st ste 511 wa dc | pritchett partners | *(empty)* |
| 0.0000743 | India | belgaum refining | 703 kuvempu nagar hanuman nagar kle international school rd belgaum ka | belgaum svc | 703 belagavi ka |
| 0.0000746 | India | ace intl | gali 17 ph 6 shiv vihar delhi delhi delhi 855 | internationalace | 855 2 gautam buddha nagar delhi delhi |
| 0.0000795 | US | gaw virtus | 212 plains ind rd wallingford ct | 6aw partners 1647413411 | 212 plains ind rd pmb 4648 wallingford ct |
| 0.0000861 | US | secure park | 115 clark st ste 2 3 ia | secure svc | *(empty)* |
| 0.0000861 | India | tx lab | f 1 jangpura ext delhi delhi | tx svc | *(empty)* |

---

## 6. Decision rule: current thresholds and how they were tuned

**Tuning method** (`decide.tune_vec`, `code/business_entity_resolution/src/decide.py`): grid-searches two
rule families crossed with the one-to-one constraint, scored by the *exact* macro F0.5 metric
(`decide.per_s1_f05` — singletons included, empty prediction scores 1 iff truly empty) computed on the OOF
blend predictions (5-fold, 30% train-frac per fold). Scores are aggregated per country and combined into a
weighted `mix` (India 0.47 / US 0.38, renormalized to sum to 1 since only these two countries have train
labels) via `decide.score_by_group`; the grid cell with the best `mix` wins. Rule families:
- `thr`: a single global probability threshold *t* (91 values, `THRESH_GRID` = 0.05–0.95 step 0.01).
- `expF`: per-S1, picks the top-*k* candidate set maximizing *expected* F0.5 under a calibration exponent
  α ∈ {0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0} applied to the probabilities (`p**α`).

Each family × α/t is tried with the one-to-one constraint (`o2o`) both on and off → (91+7)×2 = **196**
combinations, matching `decision_rules.csv`'s 196 data rows exactly.

**Winner (shipped, from `metrics_train.json.oof` and the run log's `[decide] top rules:` block):**

| rule | param (α or t) | o2o | all | india | us | mix |
|---|---:|---|---:|---:|---:|---:|
| **expF** | **1.25** | **True** | 0.96054 | 0.94718 | 0.96946 | **0.95714** |
| expF | 1.00 | True | 0.96050 | 0.94724 | 0.96935 | 0.95713 |
| expF | 1.25 | False | 0.96026 | 0.94687 | 0.96920 | 0.95685 |
| expF | 1.00 | False | 0.96010 | 0.94682 | 0.96896 | 0.95672 |
| thr | 0.65 | True | 0.96010 | 0.94662 | 0.96909 | 0.95667 |
| expF | 1.50 | True | 0.96010 | 0.94662 | 0.96909 | 0.95666 |
| thr | 0.64 | True | 0.96009 | 0.94660 | 0.96909 | 0.95655 |
| thr | 0.63 | True | 0.96008 | 0.94655 | 0.96911 | 0.95664 |

Takeaways:
- The winner (`expF`, α=1.25, o2o=True) beats the runner-up (`expF`, α=1.0) by only **0.00001** on `mix` —
  essentially tied; either alpha would ship equivalently.
- The best `expF` rule beats the best `thr` rule (`thr`, t=0.65) by **0.00047** on `mix` — a real but small
  edge from `expF`'s adaptive per-S1 cardinality selection over a single global threshold.
- `o2o=True` (each S2/S3 record assigned to at most one S1) beats `o2o=False` at every matching grid cell
  in the top rows (e.g. `expF`/1.25: 0.95714 with vs. 0.95685 without) — worth roughly **0.03pp**,
  consistently.

---

## Items not fully answered

- **§4's train/test unassigned-rate gap** (31–33% train vs. 43–46% test) is reported but not fully
  explained — see the note in §4. Resolving it needs the real train-side blend OOF probabilities (not
  saved for v3) rather than the XGB-only proxy used here.
- All §3/§4/§5 "OOF" numbers involving per-pair probabilities use the XGB-only proxy, not v3's shipped
  LightGBM+XGBoost blend (see the proxy notice at the top). The proxy's aggregate score is confirmed within
  0.0002 of the shipped blend at the "all S1" level, but no per-pair validation against the true blend was
  possible (the true blend's per-pair OOF wasn't saved for this run).
