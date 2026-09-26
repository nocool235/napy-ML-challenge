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
