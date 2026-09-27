# Holdout evaluations (27 Sep, evening)

Written by `tools/holdout_eval.py`. Each file scores v5's **final** models on the ~617k India train S1 records those
models never trained on (70% of train S1). The numbers are quoted in `EXPERIMENTS.md` and `Documentation_template.md`.

| File | What it measures |
|---|---|
| `holdout_india_baseline.json` | v5 as run, from cached features: F0.5 0.9682, recall 0.948, oracle 0.980 |
| `holdout_india_baseline_grid.json` | the same probabilities under 14 global decision rules (float64 decisions): expF α 1.0 best, 0.96835 |
| `holdout_india_b12.json` | B1 glued-word splits + B2 mined spellings as test-time normalisation: recall +0.18pp, F0.5 flat (0.96813) |
| `holdout_india_orphan20.json` | India rebuilt with 20% of S1 removed (orphan records, like test): −0.0018; expF α 1.5 best (0.96688) |
| `holdout_india_prune0.01.json` / `0.02` / `0.05` | stricter pruner cuts (6.2 / 5.5 / 4.8 candidates per S1): F0.5 −0.0014 / −0.0029 / −0.0062 |
