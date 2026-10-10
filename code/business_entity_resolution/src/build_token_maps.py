"""
Token maps for normalize.TOKEN_MAPS (applied right after basic_clean, before the canonical dictionaries).

B1, glued words (unsupervised): a token of >= 6 letters that never occurs in any Source 1 name or address
(train + test S1, no labels) is split into Source 1 words when a clean split exists:
"nantesclub" -> "nantes club", "internationaladvancedexports" -> "international advanced exports".
Only tokens absent from S1 are touched, and the same rule applies to every record.

B2, spellings mined from train labels: on true train pairs, a token the S2/S3 side uses where the S1 side
has a different token, consistently (>= 5 pairs, >= 70% to the same S1 token): "fyuchar" -> "future".
--mine-s1 final-train mines only the S1 records the final model of a run trained on (same seed, folds and
train_frac as match.train_oof), so the rest of train stays an honest holdout for that model; use
--mine-s1 all for a submission.

  python build_token_maps.py --norm-dir ../../cache/real_prep/norm-0c9c1740ce --data-dir ../../dataset \
      --mine-s1 all --out ../../cache/real_prep/token_maps_all.json
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import normalize as N  # noqa: E402
import prep  # noqa: E402
from utils import log, read_tsv  # noqa: E402

MP = mp.get_context("spawn")


def _toks(args):
    names, addrs = args
    return [N.name_map_tokens(x) for x in names], [N.addr_map_tokens(x) for x in addrs]


def raw_tokens(df: pl.DataFrame, pool, chunk: int = 100_000):
    """(name token lists, address token lists) in df's row order, exactly as TOKEN_MAPS will see them."""
    jobs = [(df["business_name"][k:k + chunk].to_list(), df["business_address"][k:k + chunk].to_list())
            for k in range(0, len(df), chunk)]
    names, addrs = [], []
    for n, a in pool.map(_toks, jobs):
        names += n
        addrs += a
    return names, addrs


def segment(tok: str, logp: dict, maxlen: int, min_piece: int = 2, piece_penalty: float = 2.0):
    """Most probable split of `tok` into vocabulary words (unigram log-probs); None if none exists."""
    n, neg = len(tok), float("-inf")
    score, back = [neg] * (n + 1), [-1] * (n + 1)
    score[0] = 0.0
    for i in range(min_piece, n + 1):
        for j in range(max(0, i - maxlen), i - min_piece + 1):
            if score[j] == neg:
                continue
            lp = logp.get(tok[j:i])
            if lp is None:
                continue
            s = score[j] + lp - piece_penalty
            if s > score[i]:
                score[i], back[i] = s, j
    if score[n] == neg:
        return None
    out, i = [], n
    while i > 0:
        out.append(tok[back[i]:i])
        i = back[i]
    return out[::-1] if len(out) >= 2 else None


def mine_subst(pairs, min_count=5, min_prec=0.7, min_unmatched_share=0.5, max_diff=3, skip_c=None) -> dict:
    """pairs: (s1_tokens, cand_tokens) of TRUE pairs -> {noisy cand token: s1 token}."""
    co, unm, tot = Counter(), Counter(), Counter()
    for a, b in pairs:
        a, b = set(a), set(b)
        tot.update(b)
        da, db = a - b, b - a
        if not db or len(da) > max_diff or len(db) > max_diff:
            continue
        for c in db:
            if skip_c is not None and c in skip_c:
                continue
            unm[c] += 1
            for s in da:
                co[(c, s)] += 1
    best = {}
    for (c, s), n in co.items():
        if n > best.get(c, (None, 0))[1]:
            best[c] = (s, n)
    return {c: s for c, (s, n) in best.items()
            if n >= min_count and n / unm[c] >= min_prec and unm[c] / tot[c] >= min_unmatched_share
            and not c.isdigit() and not any(ch.isdigit() for ch in s)}


def plausible(sub: dict, field: str) -> dict:
    """Keep mined spellings that are safe to apply everywhere: alphabetic on both sides (hyphenated legal forms
    like private-limited -> limited would break the legal-form feature), and for addresses string-similar
    (typos / transliterations such as raod -> road, karnatak -> karnataka), not semantic swaps such as a
    neighbourhood -> its city (dorchester -> boston) or a state spelling -> one of its words (tamizhnatu -> nadu)."""
    from rapidfuzz import fuzz
    return {c: s for c, s in sub.items() if c.isalpha() and s.isalpha()
            and (field == "name" or fuzz.ratio(c, s) >= 75)}


def final_train_s1(n_s1: int, seed: int = 42, folds: int = 5, frac: float = 0.3) -> np.ndarray:
    """The S1 records match.train_oof's final model trained on (same RNG draws, same order)."""
    rng = np.random.default_rng(seed)
    rng.permutation(n_s1)
    return rng.random(n_s1) < frac


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--norm-dir", required=True, help="an existing normalisation cache (raw text + row order)")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--mine-s1", choices=["final-train", "all", "none"], default="all")
    ap.add_argument("--no-seg", action="store_true", help="skip B1 (glued-word splits)")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    norm = Path(a.norm_dir)
    raw = ["entity_id", "business_name", "business_address", "country_norm"]

    with ProcessPoolExecutor(a.jobs, mp_context=MP) as pool:
        # ---- S1 vocabulary (train + test S1, raw tokens): the words a glued token may be split into
        s1n, s1a = [], []
        for sp in ("train", "test"):
            n, ad = raw_tokens(prep.load(norm, sp, ("source1",), raw), pool)
            s1n += n
            s1a += ad
        cnt = Counter(t for ts in s1n for t in ts)
        cnt.update(t for ts in s1a for t in ts)
        s1_name_vocab = set(t for ts in s1n for t in ts)
        vocab = set(cnt)
        log(f"S1 vocabulary: {len(vocab):,} tokens ({len(s1_name_vocab):,} in names)")
        del s1n, s1a

        seg = {}
        if not a.no_seg:
            words = {w: c for w, c in cnt.items() if c >= 3 and len(w) >= 2 and w.isalpha()}
            total = sum(words.values()) + len(words)
            logp = {w: math.log((c + 1) / total) for w, c in words.items()}
            maxlen = min(24, max(map(len, logp)))
            cands = set()
            for sp in ("train", "test"):
                for srcs in (("source1",), ("source2", "source3")):
                    d = prep.load(norm, sp, srcs, ["name_clean", "addr_clean"])
                    for col in ("name_clean", "addr_clean"):
                        u = (d.select(pl.col(col).str.split(" ").alias("t")).explode("t").unique()
                               .filter((pl.col("t").str.len_chars() >= 6) & pl.col("t").str.contains(r"^[a-z]+$")))
                        cands.update(u["t"].to_list())
            cands -= vocab
            for t in cands:
                parts = segment(t, logp, maxlen)
                if parts and len(parts) <= 4 and sum(map(len, parts)) / len(parts) >= 4:
                    seg[t] = " ".join(parts)
            log(f"B1: {len(cands):,} tokens of >= 6 letters absent from S1; {len(seg):,} split, e.g. "
                + ", ".join(f"{k}->{seg[k]}" for k in sorted(seg)[:: max(1, len(seg) // 25)][:25]))

        subst_n, subst_a = {}, {}
        if a.mine_s1 != "none":
            s1 = prep.load(norm, "train", ("source1",), raw).with_row_index("i")
            cand = prep.load(norm, "train", ("source2", "source3"), raw).with_row_index("j")
            gt = (read_tsv(Path(a.data_dir) / "train" / "train_ground_truth.tsv")
                  .with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
                  .filter(pl.col("matched_entity_ids").str.len_chars() > 0)
                  .join(s1.select("i", "entity_id"), left_on="source1_entity_id", right_on="entity_id")
                  .join(cand.select("j", "entity_id"), left_on="matched_entity_ids", right_on="entity_id")
                  .select("i", "j"))
            if a.mine_s1 == "final-train":
                keep = final_train_s1(len(s1))
                gt = gt.filter(pl.Series(keep[gt["i"].to_numpy()]))
            ui, uj = gt["i"].unique().sort(), gt["j"].unique().sort()
            n1, a1 = raw_tokens(s1[ui.to_numpy()], pool)
            n2, a2 = raw_tokens(cand[uj.to_numpy()], pool)
            pos_i = dict(zip(ui.to_list(), range(len(ui))))
            pos_j = dict(zip(uj.to_list(), range(len(uj))))

            def mp_(t):
                return " ".join(seg.get(x, x) for x in t).split() if seg else t

            name_pairs, addr_pairs = [], []
            for i, j in zip(gt["i"].to_list(), gt["j"].to_list()):
                pi_, pj_ = pos_i[i], pos_j[j]
                name_pairs.append((mp_(n1[pi_]), mp_(n2[pj_])))
                addr_pairs.append(([t for t in mp_(a1[pi_]) if t.isalpha() and len(t) >= 4],
                                   [t for t in mp_(a2[pj_]) if t.isalpha() and len(t) >= 4]))
            subst_n = plausible(mine_subst(name_pairs, skip_c=s1_name_vocab), "name")
            subst_a = plausible(mine_subst(addr_pairs), "addr")
            for label, sub, pairs in (("names", subst_n, name_pairs), ("addresses", subst_a, addr_pairs)):
                use = Counter(c for _, b in pairs for c in set(b) if c in sub)
                log(f"B2 {label}: {len(sub):,} spellings mined from {len(pairs):,} true pairs "
                    f"({a.mine_s1}); top: " + ", ".join(f"{c}->{sub[c]} ({n})" for c, n in use.most_common(50)))

    def compose(sub):   # split glued tokens first, then spell-correct each piece
        out = {t: " ".join(sub.get(x, x) for x in v.split()) for t, v in seg.items()}
        out.update({t: v for t, v in sub.items() if t not in seg})
        return {t: v for t, v in out.items() if v != t}

    maps = {"name": compose(subst_n), "addr": compose(subst_a),
            "meta": {"norm_dir": str(norm), "mine_s1": a.mine_s1, "seg": len(seg), "subst_name": len(subst_n),
                     "subst_addr": len(subst_a)}}
    Path(a.out).write_text(json.dumps(maps, sort_keys=True))
    log(f"wrote {a.out}: name map {len(maps['name']):,}, address map {len(maps['addr']):,}")


if __name__ == "__main__":
    main()
