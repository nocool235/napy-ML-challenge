"""
Cascade pruner: a small LightGBM on retrieval-time features only, applied between retrieval and the
(expensive) pair features. It keeps a target share of the retrieved true pairs (default 99.9%) and drops
the rest, so the matcher scores several times fewer pairs.

Features need nothing but the retrieval output (i, j, kbits, tf_cos, tf_rank): which hash keys fired,
the TF-IDF cosine and rank, and how the pair compares with the other pairs of the same S1 / candidate.
The threshold is picked on S1 records the pruner never trained on.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from retrieval import HASH_KEY_TYPES

PRUNER_VERSION = "p3"   # bump when the pruner's features, model or threshold rule change
BASE_PRUNER_FEATURES = ["tf_cos", "tf_rank", *[f"k_{n}" for n in HASH_KEY_TYPES], "n_keys",
                        "tf_max_j", "tf_gap_j", "tf_rank_i", "tf_gap_i", "n_i", "n_j"]
# extra-view suffixes match match.py's EXTRA_VIEWS keys ("char", "addr", "ns") - kept as plain strings here
# (not imported) to avoid a match.py <-> cascade_pruner circular import; only the naming needs to agree.
EXTRA_VIEW_SUFFIXES = ("char", "addr", "ns")
PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_child_samples=200,
              feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=7)


def pruner_features(pairs: pl.DataFrame) -> pl.DataFrame:
    """pairs (i, j, kbits, tf_cos, tf_rank[, tf_cos_<suffix>, tf_rank_<suffix>, ...]) -> the same rows plus
    pruner features. Extra-view columns are optional (present only when that retrieval view ran)."""
    flags = [((pl.col("kbits") // (1 << b)) % 2).cast(pl.Int8).alias(f"k_{n}") for b, n in enumerate(HASH_KEY_TYPES)]
    out = pairs.with_columns(flags).with_columns(
        n_keys=pl.sum_horizontal([f"k_{n}" for n in HASH_KEY_TYPES]).cast(pl.Int8),
        tf_max_j=pl.col("tf_cos").max().over("j"),
        tf_gap_j=pl.col("tf_cos").max().over("j") - pl.col("tf_cos"),
        tf_rank_i=pl.col("tf_cos").rank("min", descending=True).over("i").cast(pl.Float32),
        tf_gap_i=pl.col("tf_cos").max().over("i") - pl.col("tf_cos"),
        n_i=pl.len().over("i").cast(pl.Float32),
        n_j=pl.len().over("j").cast(pl.Float32),
    )
    for suf in EXTRA_VIEW_SUFFIXES:
        c = f"tf_cos_{suf}"
        if c in pairs.columns:
            out = out.with_columns(**{
                f"tf_max_j_{suf}": pl.col(c).max().over("j"),
                f"tf_gap_j_{suf}": pl.col(c).max().over("j") - pl.col(c),
                f"tf_rank_i_{suf}": pl.col(c).rank("min", descending=True).over("i").cast(pl.Float32),
                f"tf_gap_i_{suf}": pl.col(c).max().over("i") - pl.col(c),
            })
    return out


def extra_features_present(F: pl.DataFrame) -> list[str]:
    """Which extra-view pruner feature names are actually present in F (for _feature_list)."""
    feats = []
    for suf in EXTRA_VIEW_SUFFIXES:
        if f"tf_cos_{suf}" in F.columns:
            feats += [f"tf_cos_{suf}", f"tf_rank_{suf}", f"tf_max_j_{suf}", f"tf_gap_j_{suf}",
                     f"tf_rank_i_{suf}", f"tf_gap_i_{suf}"]
    return feats


def _feature_list(F: pl.DataFrame) -> list[str]:
    return BASE_PRUNER_FEATURES + extra_features_present(F)


def _X(F: pl.DataFrame, features: list[str]) -> np.ndarray:
    return F.select(pl.col(features).cast(pl.Float32)).to_numpy()


class Pruner:
    def __init__(self, model: lgb.Booster | None = None, threshold: float = 0.0, features: list[str] | None = None):
        self.model, self.threshold = model, threshold
        self.features = features or BASE_PRUNER_FEATURES

    def fit(self, F: pl.DataFrame, y: np.ndarray, n_s1: int, target_recall: float = 0.997,
            frac: float = 0.2, log=print) -> dict:
        """Train on a `frac` sample of S1 records, pick the threshold on a disjoint 10% sample so that
        `target_recall` of its retrieved true pairs survive."""
        self.features = _feature_list(F)
        rng = np.random.default_rng(11)
        u = rng.random(n_s1)[F["i"].to_numpy()]
        tr, ho = u < frac, (u >= frac) & (u < frac + 0.1)
        X = _X(F, self.features)
        dtr = lgb.Dataset(X[tr], y[tr], feature_name=self.features)
        self.model = lgb.train(PARAMS, dtr, 300)
        p = self.model.predict(X[ho])
        pos = np.sort(p[y[ho] == 1])
        self.threshold = float(pos[int(np.floor((1 - target_recall) * len(pos)))]) if len(pos) else 0.0
        keep = p >= self.threshold
        stats = {"threshold": self.threshold, "holdout_pairs": int(ho.sum()),
                 "holdout_kept_share": float(keep.mean()),
                 "holdout_recall_of_retrieved": float(keep[y[ho] == 1].mean()),
                 "reduction_x": float(1 / max(keep.mean(), 1e-9))}
        for r in (0.999, 0.998, 0.995, 0.99):   # the tradeoff curve, for choosing target_recall
            if len(pos):
                stats[f"kept_share_at_{r}"] = float(np.mean(p >= pos[int(np.floor((1 - r) * len(pos)))]))
        log(f"pruner: {json.dumps({k: round(v, 5) for k, v in stats.items()})}")
        return stats

    def score(self, F: pl.DataFrame, chunk: int = 10_000_000) -> np.ndarray:
        out = np.empty(len(F), np.float32)
        for s in range(0, len(F), chunk):
            out[s:s + chunk] = self.model.predict(_X(F.slice(s, chunk), self.features))
        return out

    def save(self, d: Path):
        d.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(d / "pruner.txt"))
        (d / "pruner.json").write_text(json.dumps({"threshold": self.threshold, "features": self.features}))

    @classmethod
    def load(cls, d: Path) -> "Pruner":
        meta = json.loads((d / "pruner.json").read_text())
        return cls(lgb.Booster(model_file=str(d / "pruner.txt")), meta["threshold"],
                   meta.get("features", BASE_PRUNER_FEATURES))
