# FINAL PUSH: Sun 27 Sep, deadline 23:59 IST

> **Outcome (22:15).** Step 0 was done. The test-only path was verified byte for byte. A2/A3 were skipped because the France normalisation was not broken. C1 EM prior shift was not shipped: the model is calibrated across crowding levels. Findings: a float64 decision-rule bug, orphan records (α 1.5), and France house-number reliability (post-hoc correction, +0.0054 LB → 0.958). Candidate-set size was cut 27% (pruner cut 0.01). v6 trains XGBoost on all entities. Details: `EXPERIMENTS.md`, `Documentation_template.md`.

Put this file and `final_push_snippets.py` in the repo root. Then tell Claude Code:
**"Read FINAL_PUSH.md and final_push_snippets.py. Do Step 0 now, then the Ground rules, then Workstream A."**

---

## Step 0: right now (v5 is mid-run; do NOT kill it)

Let v5 finish training (about 20 min). Meanwhile, light work only; check free RAM first.

1. **Model saving.** Does the running v5 process save the trained models + tuned decision rule to disk BEFORE its test phase? Where?
2. **Speed up the char-view top-k without changing its config.** Keep the same vectorizer, the same K and the same max_df.
   - Swap in a multithreaded exact top-k: `pip install sparse_dot_topn`, then `sp_matmul_topn(Q, S1.T, top_n=K, n_threads=16, sort=True)`.
   - Fallback: chunked matmul in a thread pool.
   - Verify on 50k US train queries: the candidate sets must be the same (ties aside). Report old vs new time.
3. **Add a test-only entrypoint:**
   1. load the saved models + rule
   2. build test candidates, reusing caches where the inputs are unchanged
   3. score, then decide
   4. write matching_results, candidate_pairs and test probs

   Put the fixes in this file behind flags, default off (France fold/profile first). A3 and B6a use this path.
4. **When training ends:**
   - If the models are saved AND item 2 is verified: kill v5 before its test retrieval, and run the test-only path with the fast top-k.
   - Otherwise: let v5 run to the end.
   - Either way: validate, then **Submission 1**.

Do not change char max_df or K for this run. Test must match train's retrieval.
Everything runs on one machine: i7-14700, 32 GB RAM, RTX 4080 Super 16 GB. The **Machine plan** section below says what may run at the same time.

---

## Why these fixes (the numbers)

**Where the leaderboard points come from.** The leaderboard is weighted by S1 count:
**0.468 × India + 0.383 × US + 0.150 × France.**

**France is the weak country.** v3 scored India 0.947 and US 0.970 on OOF. Its leaderboard score was 0.947, which implies **France ≈ 0.89**. A public team scored 0.984 on validation and 0.969 on the leaderboard, which fits the same France ≈ 0.89 pattern.

**Test has about 2× the non-matching records (distractors).** Test has 5.75 S2/S3 records per S1; train has 4.68. If the matches per S1 are the same, about 40% of test S2/S3 records are distractors versus 26% in train. More distractors means more false merges in test than OOF shows.

**Most retrieval misses are recoverable.** 3.5% of true train pairs never reach the model (India 6.1%). The 40 misses in diagnostics.md have an average similarity of 84. About 65% of them are fixable with these fixes:
- splitting glued words: "internationaladvancedexports" → international advanced exports
- a spelling dictionary mined from the train labels: "fyuchar" → future, "forchyun" → fortune
- name+city keys: "royal engineering / a bareilly up"
- keeping house-number blocks in addresses: "g 5 6", "c 4 2"

**What 0.99 needs.** All three countries at about 0.99. That means France fixed, India recall at about 99%, and precision that holds against the extra test distractors. There is no single switch.

---

## Machine plan (i7-14700 · 32 GB RAM · RTX 4080 Super 16 GB)

**RAM is the limit, not CPU or GPU.**

**Job sizes.**
- **Heavy:** anything touching all ~10M S2/S3 records at once. That means a full train+test run, retrieval over every country, or a sibling self-join over every record.
- **Light:**
  - post-processing on saved probability files (C1–C3)
  - the France-only re-score (A3)
  - 10%-sample measurements (B0–B4)

**How many at once.** At most **1 heavy + 1 light**. Check free RAM before starting anything; if less than 8 GB is free, wait.

**v4_stage2.** If v5 supersedes it, kill it now. Two full runs at once on 32 GB risks out-of-memory crashes.

**Memory habits.**
- Process **one country at a time**: India, US, France. Run `del` + `gc.collect()` between countries.
- Use polars lazy/streaming for explode and join.
- Never hold the full candidate table for all countries in pandas.

**Threads.** LightGBM uses `num_threads=16` when alone, 8 when a light job runs alongside. Leave about 4 threads for the OS and the other Claude Code session.

**GPU.**
- Use it for **XGBoost only**: `device="cuda", tree_method="hist"`.
- Keep VRAM at or below about 14 GB. The "16 GB shared" GPU memory is system RAM and very slow. If XGBoost spills into it, set `max_bin=128` or subsample.
- LightGBM stays on CPU.
- Move TF-IDF top-k to the GPU (CuPy, batches of about 50k queries) **only if** the Ground-rules report shows retrieval is more than 30% of runtime.
- A transformer re-ranker isn't worth it tonight. The remaining errors are ambiguity (twin S1 records, empty addresses), not string similarity.

**Windows (not WSL).** multiprocessing uses spawn: keep pools at 8 workers or fewer, under `if __name__ == "__main__":`. `segment()` and `mine_subst()` are fast enough single-process, about 1–2 min each.

**Disk.** Write intermediate parquet to the local SSD, not a OneDrive or network folder.

**Teammates.** They need no compute, so they can start now:
- fill `Documentation_template.md` from diagnostics.md and this file
- build the zip skeleton (`code/business_entity_resolution/src`, README with exact run commands)
- `pip freeze > requirements.txt` from the working env
- run the validator on every file before it's uploaded

---

## Ground rules (every workstream)

1. **Report status before changing anything** (runtime profile already known: char view ≈ 80 min on train):
   - v5's OOF F0.5 (all / india / us) and oracle, once training finishes
   - whether v4_stage2 is still running. If v5 supersedes it, say so; I may kill it to free RAM.
   - free RAM right now
2. **Put every change behind a flag, default off.** Bump the cache/version keys whenever normalization or retrieval changes, so no stale parquet cache gets reused.
3. **Save per-pair probabilities for every run** (s1_id, cand_id, country, p), for both OOF and test.
4. **Keep candidate_pairs.tsv complete.** Any pair added by any stage must also appear in candidate_pairs.tsv. Run `utils/validate_submission.py` before handing me any file.
5. **Measure retrieval changes cheaply and report them.** Use a 10% sample of train S2/S3 *queries* against the FULL S1 index. Report recall, candidates/record and runtime, per country.
6. **Stop starting new full runs at 21:30.**

---

## Workstream A: France (do first; re-score test only, no retraining)

**A1. Recon (10 min).**
- Print 30 raw France rows from each of test S1/S2/S3, next to the current normalizer's output.
- Count non-ASCII characters in the France raw text.
- Run the normalizer on `Société Générale SARL, 12 Rue de l'Église, 75002 Paris 2ème Cedex`. If the output is something like `soci t g n rale`, that's the bug.
- Is `country` a model feature?
- Is postcode extraction country-specific, so France gets none?

**A2. Fix normalization.**
- Call `fold()` from final_push_snippets.py FIRST in the normalizer, before any regex that drops non-[a-z0-9] characters. Do this for every country; it's harmless for plain ASCII text.
- Add a per-country *profile* dict with a default profile for unknown countries. The problem statement requires treating country as an open set.
- **France profile:**
  - stopwords, dropped from name keys and address keys: `de du des la le les l d et a au aux en sur sous pour par chez`
  - address abbreviations, mapped to the SAME canonical forms the US map already uses:
    - avenue/av → (US ave form)
    - boulevard/bd/bld/boul → (US blvd form)
    - place → pl, chemin/chem → ch, route → rte, impasse → imp, allee → all
    - faubourg/fg → fbg, residence → res, batiment → bat, appartement/appt → apt
    - saint → st, sainte → ste
    - drop: cedex, bis, ter, arrondissement/arr
  - legal forms, added to the legal-suffix list:
    `sarl sas sasu sa eurl sci snc scp scm selarl selas sel gie sca scop scic ei eirl earl gaec societe ste ets etablissements cie compagnie`
  - generic words mapped to existing canonicals: centre → (same as center), international(e) → intl, technologies → tech, solutions → soln, services → svc
- **Default profile:** postcode regex `\b\d{5,6}\b`.
- If `country` is a model feature: note it for B. Don't retrain here.

**A3. Re-score France only** with the newest finished trained model and its saved decision rule.
- Splice the France rows into that run's matching_results and candidate_pairs.
- India and US rows must stay byte-identical → `sub_A.tsv`.
- Report France before/after: coverage, candidates/record, predicted singleton rate, matches/S1, unassigned share.
- If artifacts aren't saved and re-scoring needs a retrain, skip A3. Fold A2 into v6 instead and tell me.

**A4. Mine France-specific noise from test itself** (no labels; high-confidence pairs only).
- Take France test pairs with p ≥ 0.98 where the record's best S1 is that S1, and that S1's top record is that record.
- Run `mine_subst()` on their token lists, separately for names and addresses. Print the top 60.
- Add the sensible ones to the France profile. Re-run A3 → `sub_A2.tsv`.

**→ Submission 2 = sub_A (or sub_A2).** Only France rows differ from Submission 1, so **ΔLB ÷ 0.15 = France gain**. That tells us how broken France was.

---

## Workstream C: post-processing on saved probabilities (minutes; no retraining)

**C1. EM prior shift** (`em_prior()`).
1. Calibration check: EM on OOF probabilities per country should return ≈ the train positive rate (within 0.01). If not, fit isotonic calibration on OOF first.
2. Run EM on test probabilities per country. France uses the pooled train prior.
3. Report estimated test prior vs train prior per country.
4. Push the adjusted probabilities through the SAME tuned rule (expF, α, o2o) → `sub_C1.tsv`.
5. Report the change in matches/S1 and singleton rate per country.

**C2. Sibling links** (`sibling_edges()`, per country, over S2∪S3).
- Fingerprints: normalized name+address tokens *before* dictionary mapping, plus numeric composites (`num_composites()`) and phone numbers (runs of ≥7 digits). Keep only tokens absent from S1 and shared by 2–6 records. These are generator artifacts: shared typos, made-up names, PMB/unit numbers, phones.
- On train, report:
  - link precision: P(same S1 | both endpoints matched)
  - how many of the retrieval-missed true pairs have a sibling that is matched correctly

**C3. Sibling propagation** (`propagate()`).
- Grid on OOF: t_src ∈ {0.9, 0.95, 0.99}, t_dst ∈ {0.1, 0.3, 0.5}, k ∈ {1, 2}.
- Score with the exact macro-F0.5 scorer; report mix/india/us.
- If ΔOOF ≥ +0.002: apply the best setting to test → `sub_C3.tsv`, and add the propagated pairs to candidate_pairs.tsv. Document it as a blocking stage: sibling expansion.

**→ Submission 3 = the Submission 2 file + C1 and/or C3.** C1 can't be validated on OOF, so the leaderboard is its test. If ΔLB > 0, keep it in v6.

---

## Workstream B: recall + precision → v6 full run (start as soon as A2 is committed)

**B0. Recon (15 min, on the newest finished run's candidates).**
1. Classify the missed true pairs per country:
   - candidate address empty
   - identical core name
   - glued token (≥7 chars, not in S1 vocab)
   - shared numeric tokens
   - made-up name
   - other
2. For the top 30 OOF false positives, print the candidate's TRUE S1 (raw text) next to the predicted S1 (raw text). Are they "twin" S1 records? What tells them apart (city, house number, suite, legal suffix)?
3. Per-S1 joint distribution of (#S2 matches, #S3 matches), top 15 combos. If there's a strong regularity, tell me before using it.

**B1. Split glued words** (`build_logp()` + `segment()`).
- Vocab = S1 name and address tokens (train+test).
- Apply to tokens of length ≥6 that are not in the vocab, on both sides, for names and addresses.
- Cache per unique token and use multiprocessing.
- Run it BEFORE the abbreviation map, so internationaladvancedexports → international advanced exports → intl advanced exports.

**B2. Spelling dictionary mined from train labels** (`mine_subst()`).
- Mine on true pairs of a random 50% of train S1s; measure on the other 50%. For the final run, mine on all of train.
- Names and addresses separately. Also catches digit-for-letter spellings (8ay → bay).
- Apply to S2/S3 tokens only. Print the top 50 mappings.

**B3. Phone numbers.** Move runs of ≥7 digits out of names into a `phone` field. Used by C2, and add a phone-equal feature.

**B4. Retrieval.** Keep each change only if recall goes up and candidates/record goes up by at most 15%.
1. **TF-IDF tokenizer.** Use `token_pattern=r"(?u)\b\w+\b"`. The default pattern drops "5", "g" and "a", so Indian house numbers vanish. Also add `num_composites()` tokens to the address text.
2. **Hash keys** `c:core|loc` and `r:rare|loc`, via `name_loc_keys()` + `key_pairs()`, cap 5 S1 per key:
   - core = name with legal/generic words removed, tokens sorted
   - rare = the 2 rarest core tokens
   - loc = city/district tokens with S1 df ≥ 20
   - also add the key `composite|state`
3. **Name-only char n-gram view** (K=10) for S2/S3 records whose address is empty or has ≤2 tokens.
4. **Settings:** TF-IDF K 10 → 20; pruner target 99.7% → 99.9%.

**B5. New features.**
- `s1_core_name_df`: how many S1 in the country share the core name, and how many share core name + locality. This is the ambiguity signal for empty-address records.
- `cand_name_oov_frac`: share of candidate name tokens not in the S1 vocab (made-up names → rely on address).
- numeric-token Jaccard; composite-number exact match; locality match; phone equal.
- competition features: similarity margin vs the record's best *other* candidate; rank among the record's candidates; count of candidate S1s sharing the same core name (twins).
- Drop `country` as a model feature if present.

**B6a: v6-lite. The default: re-score only, no retraining.**
1. With A2 (+A4) and B1–B4 turned on, rebuild the TEST candidates for all countries, one country at a time.
2. Re-score them with the newest trained model and its saved rule, using the same path as A3. Only B5's *new* features need a retrain; B1–B4 just change the inputs and candidates, so the existing model can score them.
3. **Check before shipping.** On train, score only the candidate pairs that are **new** compared with the old candidate set. The model never trained on those pairs, so this is an honest out-of-sample check. Report precision and recall on them at the rule's operating point.
4. Ship as v6-lite only if the new pairs' precision is ≥ 0.90. Then apply whatever won in Submission 3 (C1/C3).

**B6b: v6-full (retrain).** Do this **only if** the Ground-rules report says a full run takes ≤ 2 h on this machine.
- Contents: A2 (+A4) + B1–B5 (+ stage 2 if the newest run has it).
- Report OOF all/india/us + oracle + runtime.
- It must launch by 19:45. If it can't finish by about 22:15, v6-lite is the final candidate.

---

## Submission plan (4 left)

| # | When | File | What it tells us |
|---|------|------|------------------|
| 1 | as soon as v5 validates | v5 as-is | safety score |
| 2 | ~19:00 | v5 + France fix (France rows only) | France gain = ΔLB / 0.15 |
| 3 | ~20:00 | #2 + EM prior shift and/or sibling propagation | whether the test-distractor fix pays |
| 4 | **23:00, final pick** | v6 (lite or full) + winners of #2/#3, *only if* its checks pass (v6-lite: new-pair precision ≥ 0.90; v6-full: OOF > v5 OOF); otherwise the best of #1–#3 | — |

Check on the portal whether the private leaderboard uses your **last** upload or lets you **select** one. If it's last-upload, #4 must be the best known file.

## Timeline (IST)

| Time | What |
|---|---|
| 17:50 | Step 0 (fast top-k + test-only path) and A1/A2, all light, while v5 trains |
| ~18:10 | v5 training done → Step 0 item 4 decision → v5 test results (~18:30 with fast top-k, ~19:30 without) → Submission 1 |
| then | A3 France re-score, plus C1–C3 in a second Claude Code session (both light) |
| 18:30 | B0–B4 on 10% samples (light, second session) |
| ~19:00 | Submission 2 |
| ~20:00 | Submission 3 |
| ≤19:45 | B6b launch (only if a full run takes ≤ 2 h); otherwise B6a v6-lite after Submission 3 |
| 21:30 | no new heavy jobs |
| 22:30 | build the zip (teammates have it prepared) |
| 23:30 | final upload + zip done |

The zip contains:
- `output/matching_results.tsv` and `output/candidate_pairs.tsv`
- `code/business_entity_resolution/src/`, plus README with exact run commands and `requirements.txt`
- the filled `Documentation_template.md`

Candidate-set size also counts in the final ranking, so keep the pruner.

**Not doing:** anything based on entity_id numbering or file row order. Top packages are code-reviewed; don't risk disqualification.
