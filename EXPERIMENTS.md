# Experiments log

One row per change. Scores are macro F0.5 per S1 entity, singletons included.
- **OOF**: out-of-fold on train, grouped by S1.
- **LOCO**: leave-one-country-out. We train on US and score India, then the reverse, and report the mean. It is the France proxy.
- **test-mix**: `0.47 × India + 0.38 × US + 0.15 × LOCO`, the test set's country mix. Decisions are made on this number.
- **scale**: `1%` means the old random 1% sample. It is much easier than the real data, because hard negatives are about 100× sparser. Only rows marked `full` predict the leaderboard.

| # | date (IST) | change | scale | OOF all | US | India | singletons | matched | LOCO | test-mix | runtime | LB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 09-25 03:40 | starter pipeline (TF-IDF top-K blocking, 2-stage LightGBM, expF rule) | 1% | 0.9853 | 0.9913 | 0.9762 | 0.9716 | 0.9862 | – | – | 20 min on a Mac | – |
| 1 | 09-25 04:10 | partitioned rare-key blocking + global re-rank, K cut (34 pairs/S1) | 1% | 0.9851 | 0.9914 | 0.9755 | 0.9693 | 0.9860 | – | – | 3.4 min on a Mac | – |
| 2 | 09-25 07:00 | crude no-ML baseline: 4 hash-join keys + token-set threshold (w=0.4, t=0.77), best S1 per candidate | full | 0.8309 | 0.8515 | 0.7999 | – | – | n/a (rule) | 0.823 (India/US only) | 21 min on a Mac | pending |
| 2b | 09-27 01:37 | crude baseline re-run after the prep fork→spawn fix (same code) → `output_crude/` | full | 0.8309 | 0.8515 | 0.7999 | – | – | n/a (rule) | 0.823 (India/US only) | 6 min, 28-core Windows | – |
| 3 | 09-27 02:49 | `match.py` step 1: SAME hash-key candidates (recall 0.808, oracle 0.904) + 54 pair features + LightGBM out-of-fold (5 folds, 30% S1 per model) + vectorised expected-F0.5 rule with one-to-one → `output_v1/` | full | 0.8907 | 0.9294 | 0.8328 | – | – | – | 0.876 (India/US only) | 21 min | – |
| 4 | 09-27 03:41 | `match.py` step 2: hash keys ∪ word TF-IDF top-10 per S2/S3 (df ≤ 1%, sparse_dot_topn) → cascade pruner (99.7% of retrieved positives, 110M → 17.0M pairs) → 64 features → same LightGBM + rule. Recall 0.968 → 0.965, oracle 0.985 → `output_v2/` | full | 0.9540 | 0.9661 | 0.9358 | – | – | – | 0.949 (India/US only) | train ≈ 51 min, 28 cores | **0.9365** (public, upload #1) |
| 5 | 09-27 04:42 | v2's features + GPU XGBoost trained alongside LightGBM, 50/50 blend (weight on OOF log-loss). Log-loss 0.06981 (LGB) / 0.06997 (XGB) → 0.06922 blend. XGB alone = LGB alone at the same rule (0.9483 vs 0.9484) at ~3× the speed (50–80 s vs 150–210 s per fold). **Blend +0.0004 over v2: below the 0.002 bar, not uploaded.** → `output_v2b/` | full | 0.9543 | 0.9664 | 0.9362 | – | – | – | 0.9497 (India/US only) | training 23 min | – |
| 6 | 09-27 05:30 | feature set f4: de-noised name/address similarity (data-driven per-country source-noise token detection), all-address-number overlap, name↔address field swap, per-country TF-IDF cosines (name words, addr words, name char-3gram), landmark flags, acronym match. LightGBM. → `output_v3/` | full | 0.9596 | 0.9688 | 0.9460 | – | – | LOCO mean 0.9227 (f2 baseline: 0.9110) | 0.9571 (India/US only) | features ~13 min, training ~28 min | **0.947** (public, upload #2) |
| 7 | 09-27 15:40– (running) | f5 (+legal-form-conflict feature) + `EXTRA_VIEWS` retrieval (char n-gram + word view; addr/ns views coded but deferred to the next run) + B15 normalisation fixes + LightGBM/XGBoost blend + `--stage2` → `output_v5/`. Stage 2 kept only if it beats stage 1 OOF (measured separately: does not transfer via LOCO, see row above the bottleneck table in `ARCHITECTURE.md` §B16/§Second stage). | full | *running* | | | | | | | | not yet uploaded |

**Aside, measured on v3's cached f4 features (XGBoost only, GPU) but not part of any run above:** stage 2
(`stage2.py`) scored group-CV mix 0.9667 (+0.0096 over stage 1's 0.9571) but LOCO mean −0.0011 vs stage 1 —
it does not transfer to an unseen country and has not shipped in any upload. This is why `decide.tune_robust`
(blends group-CV and LOCO) exists: to catch this kind of in-distribution-only gain before spending an upload on it.

## Step 0 data facts (full data)
These are the full-data facts that drive the design. The raw examples are in `reports/`, which is gitignored.
- **Scale:** train has 2.21M S1 and 10.32M S2+S3 records. Test has 1.73M S1 (India 47%, US 38%, France 15%) and 9.97M S2+S3.
- **Uniqueness:** no S2/S3 record belongs to two S1s, out of 7.64M true pairs.
- **Country:** there are 0 cross-country pairs.
- **Structure:** 5.6% of S1 are singletons, the mean is 3.46 matches per S1, and 26% of S2/S3 records match nothing. US and India are identical on all three.
- **Scripts:** 23.5% of S2 India names and 13.2% of S3 India names are in Indic script.
- **Postcodes:** a postcode appears in about 10% of US addresses and about 1% of India and France addresses.
- **Retrieval ceiling:** exact TF-IDF cosine at full scale, S2/S3 → S1, taking the union of the name+addr, name and addr views: R@10 is 0.968 and R@100 is 0.982. India S2 plateaus at 0.955.
- **Transliteration:** Indic-name true pairs (1% sample) went from a name token-set similarity of 6.9 to 90.4, and from 6.7% to 83.9% of pairs at ≥ 80.
