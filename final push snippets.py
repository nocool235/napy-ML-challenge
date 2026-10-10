"""final_push_snippets.py — tested helper code referenced by FINAL_PUSH.md.
Adapt to the repo's module layout; every function is self-contained.
Run `python final_push_snippets.py` to self-test on toy data (needs numpy, polars)."""

import math
import re
import unicodedata
from collections import Counter

import numpy as np
import polars as pl


# ---------------------------------------------------------------- A2: fold
_SPECIAL = str.maketrans({"œ": "oe", "Œ": "oe", "æ": "ae", "Æ": "ae", "ß": "ss",
                          "ø": "o", "Ø": "o", "ł": "l", "Ł": "l", "đ": "d", "Đ": "d",
                          "’": "'", "‘": "'", "´": "'", "`": "'"})
_ELISION = re.compile(r"\b(?:l|d|j|m|n|s|t|c|qu)'\s*")
_ORDINAL = re.compile(r"\b(\d+)(?:er|ere|eme|e)\b")


def fold(s: str) -> str:
    """Run FIRST, before any regex that drops non [a-z0-9] characters."""
    s = (s or "").translate(_SPECIAL)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = _ELISION.sub(" ", s)          # l'eglise -> eglise, d'or -> or
    s = _ORDINAL.sub(r"\1", s)        # 2eme / 1er -> 2 / 1
    return s


# ---------------------------------------------------------------- B1: glued-word split
def build_logp(s1_token_lists, min_len=2, max_piece=24):
    """Vocab log-probs from S1 tokens. Pieces capped at max_piece chars keeps segment() fast:
    ~1M unique tokens segment in 1-2 min single-process (no multiprocessing needed on Windows)."""
    c = Counter(t for toks in s1_token_lists for t in toks
                if len(t) >= min_len and not t.isdigit())
    total = sum(c.values()) + len(c)
    logp = {w: math.log((n + 1) / total) for w, n in c.items()}
    return logp, min(max_piece, max(map(len, logp)))


def segment(tok, logp, maxlen, min_piece=2, piece_penalty=2.0):
    """Split a glued token into S1-vocabulary words; None if no clean split."""
    n, NEG = len(tok), float("-inf")
    score, back = [NEG] * (n + 1), [-1] * (n + 1)
    score[0] = 0.0
    for i in range(min_piece, n + 1):
        for j in range(max(0, i - maxlen), i - min_piece + 1):
            if score[j] == NEG:
                continue
            lp = logp.get(tok[j:i])
            if lp is None:
                continue
            s = score[j] + lp - piece_penalty
            if s > score[i]:
                score[i], back[i] = s, j
    if score[n] == NEG:
        return None
    out, i = [], n
    while i > 0:
        out.append(tok[back[i]:i])
        i = back[i]
    return out[::-1] if len(out) >= 2 else None


# ---------------------------------------------------------------- B2: label-mined dictionary
def mine_subst(pairs, min_count=5, min_prec=0.7, min_unmatched_share=0.5, max_diff=3,
               skip_c=None):
    """pairs: (s1_tokens, cand_tokens) for TRUE train pairs, already normalized + segmented.
    Returns {noisy_cand_token: clean_s1_token}.
    RAM (32 GB box): for NAMES pass skip_c=set(S1 name vocab) so only noisy tokens are counted;
    for ADDRESSES pass None but feed only alphabetic tokens of len>=4 (keeps city aliases)."""
    co, unm, tot = Counter(), Counter(), Counter()
    for a, b in pairs:
        a, b = set(a), set(b)
        for t in b:
            tot[t] += 1
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
            if n >= min_count and n / unm[c] >= min_prec
            and unm[c] / tot[c] >= min_unmatched_share and not c.isdigit()}


# ---------------------------------------------------------------- B4: numeric composites
def num_composites(tokens):
    """['g','5','6','sec','10','dwarka'] -> ['g_5_6'] ; keeps house-number blocks as one rare token."""
    out, run = [], []
    for t in list(tokens) + ["<end>"]:
        if any(ch.isdigit() for ch in t) or (len(t) == 1 and t.isalpha()):
            run.append(t)
        else:
            if len(run) >= 2:
                out.append("_".join(run))
            run = []
    return out


# ---------------------------------------------------------------- C1: EM prior shift
def em_prior(p, pi_tr, iters=500, tol=1e-8):
    """Saerens-Latinne-Decaestecker label-shift EM. p: model probs on one country's test pairs.
    pi_tr: positive rate of the pairs that model was trained on. Returns (pi_test, adjusted probs)."""
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    pi = pi_tr
    for _ in range(iters):
        a = (pi / pi_tr) * p
        b = ((1 - pi) / (1 - pi_tr)) * (1 - p)
        q = a / (a + b)
        new = q.mean()
        if abs(new - pi) < tol:
            pi = new
            break
        pi = new
    a = (pi / pi_tr) * p
    b = ((1 - pi) / (1 - pi_tr)) * (1 - p)
    return pi, a / (a + b)



# ---------------------------------------------------------------- C2: sibling links
def sibling_edges(recs: pl.DataFrame, s1_vocab: set, min_df=2, max_df=6, min_len=4) -> pl.DataFrame:
    """recs: one country's S2∪S3 with columns id (str), fp (list[str]) = fingerprint tokens:
    normalized name+address tokens BEFORE dictionary mapping, plus numeric composites and
    phone numbers (>=7-digit runs). A fingerprint = token absent from S1 and shared by 2..6 records
    (generator artifacts: typos, made-up names, unit/PMB numbers, phones).
    Returns edges (a, b, shared)."""
    ex = (recs.select("id", "fp").explode("fp").drop_nulls().unique()
              .filter(pl.col("fp").str.len_chars() >= min_len)
              .join(pl.DataFrame({"fp": list(s1_vocab)}), on="fp", how="anti"))
    ok = (ex.group_by("fp").agg(pl.len().alias("n"))
            .filter((pl.col("n") >= min_df) & (pl.col("n") <= max_df)).select("fp"))
    g = ex.join(ok, on="fp")
    return (g.join(g, on="fp", suffix="_b")
             .filter(pl.col("id") < pl.col("id_b"))
             .group_by(["id", "id_b"]).agg(pl.len().alias("shared"))
             .rename({"id": "a", "id_b": "b"}))


# ---------------------------------------------------------------- C3: propagation
def propagate(assign: pl.DataFrame, best_p: pl.DataFrame, edges: pl.DataFrame,
              t_src=0.95, t_dst=0.3, k=1) -> pl.DataFrame:
    """assign: final decided pairs (s1, rec, p). best_p: (rec, best_p) = each record's max prob over
    ALL its candidates (0 if none). Adds (s1, rec) when a sibling is confidently assigned to s1 and rec
    itself is weak/unassigned. Conflicts (two different S1 proposed for one rec) are dropped."""
    e = pl.concat([edges.rename({"a": "src", "b": "dst"}),
                   edges.rename({"b": "src", "a": "dst"}).select("src", "dst", "shared")])
    e = e.filter(pl.col("shared") >= k)
    src = assign.filter(pl.col("p") >= t_src).select(pl.col("rec").alias("src"), "s1", "p")
    prop = e.join(src, on="src")
    assigned = assign.select(pl.col("rec").alias("dst")).unique()
    prop = (prop.join(assigned, on="dst", how="anti")
                .join(best_p.rename({"rec": "dst"}), on="dst", how="left")
                .filter(pl.col("best_p").fill_null(0) < t_dst))
    ok = (prop.group_by("dst").agg(pl.col("s1").n_unique().alias("nx"), pl.col("s1").first())
              .filter(pl.col("nx") == 1).drop("nx"))
    new = ok.select("s1", pl.col("dst").alias("rec"), pl.lit(-1.0).alias("p"))
    return pl.concat([assign, new])


# ---------------------------------------------------------------- B4: name|locality keys
def name_loc_keys(df: pl.DataFrame, loc_ok: set) -> pl.DataFrame:
    """df: id, country, core_sorted (str: legal/generic words removed, tokens sorted),
    rare2 (list[str]: the 2 rarest core tokens by S1 df), addr_toks (list[str]).
    loc_ok: alphabetic address tokens (len>=3) with S1 df >= 20 in that country (cities, districts,
    localities). Emits keys core|loc and rare|loc."""
    loc = (df.select("id", "country", "core_sorted", "rare2", pl.col("addr_toks").alias("loc"))
             .explode("loc").join(pl.DataFrame({"loc": list(loc_ok)}), on="loc", how="semi"))
    k1 = loc.select("id", "country", (pl.lit("c:") + pl.col("core_sorted") + "|" + pl.col("loc")).alias("k"))
    k2 = (loc.explode("rare2").drop_nulls("rare2")
             .select("id", "country", (pl.lit("r:") + pl.col("rare2") + "|" + pl.col("loc")).alias("k")))
    return pl.concat([k1, k2]).unique()


def key_pairs(s1_keys: pl.DataFrame, c_keys: pl.DataFrame, cap_s1=5) -> pl.DataFrame:
    grp = (s1_keys.group_by(["country", "k"]).agg(pl.col("id").alias("s1"))
                  .filter(pl.col("s1").list.len() <= cap_s1))
    return (c_keys.join(grp, on=["country", "k"]).explode("s1")
                  .select("s1", pl.col("id").alias("rec")).unique())


if __name__ == "__main__":
    print(fold("Société Générale SARL, 12 Rue de l'Église, 75002 Paris 2ème Cedex"))
    s1 = [["intl", "advanced", "exports"], ["international"], ["md", "commerce"], ["bros", "private", "curative"]]
    logp, mx = build_logp(s1)
    for t in ["internationaladvancedexports", "mdcommerce", "brosprivatecurative", "pmasrtdfers"]:
        print(t, "->", segment(t, logp, mx))
    pairs = [(["future", "marketing"], ["fyuchar", "marketing"])] * 20 + [(["tx", "lab"], ["tx", "svc"])] * 20 \
          + [(["blue", "svc"], ["blue", "svc"])] * 40
    print(mine_subst(pairs))
    print(num_composites("g 5 6 sec 10 dwarka delhi".split()))
    rng = np.random.default_rng(0)
    def post(x, pi):
        l1 = np.exp(-(x - 1.5) ** 2 / 2); l0 = np.exp(-(x + 1.5) ** 2 / 2)
        return pi * l1 / (pi * l1 + (1 - pi) * l0)
    y = rng.random(200000) < 0.34
    x = np.where(y, rng.normal(1.5, 1, y.size), rng.normal(-1.5, 1, y.size))
    print("EM test prior (true 0.34):", round(em_prior(post(x, 0.43), 0.43)[0], 4))
    recs = pl.DataFrame({"id": ["S2-1", "S3-9", "S2-5", "S3-7"],
                         "fp": [["pyrafluxdelta", "pmb_7481"], ["continental", "pmb_7481"],
                                ["sri", "7895948886"], ["nagpal", "7895948886"]]})
    ed = sibling_edges(recs, {"continental", "nagpal", "sri"})
    assign = pl.DataFrame({"s1": ["S1-A", "S1-B"], "rec": ["S3-9", "S3-7"], "p": [0.99, 0.97]})
    best = pl.DataFrame({"rec": ["S2-1", "S2-5"], "best_p": [0.01, 0.2]})
    print(propagate(assign, best, ed))
