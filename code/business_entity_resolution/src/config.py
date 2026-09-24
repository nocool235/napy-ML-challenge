"""All tunable knobs in one place. Edit here, not inside the modules."""
from pathlib import Path

SEED = 42

# ---------------- blocking views ----------------
# col   : normalised column the view is built on (see normalize.py)
# tfidf : kwargs for sklearn TfidfVectorizer
VIEWS = {
    "name":      {"col": "name_core",  "tfidf": {"analyzer": "char_wb", "ngram_range": (2, 4)}},
    "addr":      {"col": "addr_clean", "tfidf": {"analyzer": "char_wb", "ngram_range": (3, 4)}},
    "name_addr": {"col": "name_addr",  "tfidf": {"analyzer": "char_wb", "ngram_range": (3, 4)}},
    "name_word": {"col": "name_core",  "tfidf": {"analyzer": "word", "token_pattern": r"\S+"}},
    "nums_name": {"col": "blk_nums_name", "tfidf": {"analyzer": "word", "token_pattern": r"\S+"}},
}
# neighbours kept per view: forward = per S1 record per source (S2, S3); backward = per S2/S3 record
K_FWD = {"name": 10, "addr": 5, "name_addr": 15, "name_word": 5, "nums_name": 5}
K_BWD = {"name": 3, "addr": 2, "name_addr": 5, "name_word": 2, "nums_name": 2}

# ---------------- model ----------------
N_FOLDS = 5
LGB_PARAMS = dict(
    objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=20,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    verbose=-1, seed=SEED, num_threads=0,
)
NUM_BOOST_ROUND = 2000
EARLY_STOP = 100

# ---------------- decision rule ----------------
THRESH_GRID = [round(x, 3) for x in [i / 100 for i in range(5, 96, 1)]]

CACHE_DIR = Path("cache")
