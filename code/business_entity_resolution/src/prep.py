"""
Stage 1 - normalise every record once, in parallel, into parquet.

    <work>/norm-<hash>/skeleton_maps.json
    <work>/norm-<hash>/<split>_<source>/part-00000.parquet, ...   (+ _SUCCESS when complete)

<hash> covers normalize.py, translit.py and NORM_VERSION, so any change to the normalisation writes a
new directory instead of silently reusing stale parts. Parts that already exist are skipped, so a
killed run resumes where it stopped.

    python src/prep.py --data-dir ../../dataset --work-dir ../../work
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import normalize as N  # noqa: E402
from utils import Progress, code_hash, log, n_jobs, read_tsv  # noqa: E402

NORM_VERSION = "v2"
SRC = Path(__file__).resolve().parent
SPLITS = ("train", "test")
SOURCES = ("source1", "source2", "source3")
STR_COLS = ["name_clean", "name_core", "name_nospace", "name_initials", "name_legal", "name_alts", "name_nums",
            "name_skel", "addr_clean", "addr_nolm", "addr_words", "addr_nums", "addr_postcode", "addr_house",
            "addr_street"]
INT_COLS = ["name_ind", "addr_has_landmark", "addr_ntok", "addr_ind"]


def norm_dir(work_dir: Path) -> Path:
    return Path(work_dir) / f"norm-{code_hash(SRC / 'normalize.py', SRC / 'translit.py', extra=NORM_VERSION)}"


# ----------------------------------------------------------------------------- workers
def _count_tokens(args):
    names, addrs = args
    cn, ca = Counter(), Counter()
    for x in names:
        cn.update(N.basic_clean(x).split())
    for x in addrs:
        ca.update(N.basic_clean(x).split())
    return cn, ca


def _init(maps):
    N.set_skeleton_maps(maps)


def _normalise_part(args):
    df, out = args
    rows = {c: [] for c in STR_COLS + INT_COLS}
    for name, addr in zip(df["business_name"].to_list(), df["business_address"].to_list()):
        f = N.name_fields(name)
        f.update(N.addr_fields(addr))
        for c in rows:
            rows[c].append(f[c])
    res = df.with_columns(
        source=pl.col("entity_id").str.slice(0, 2),
        country_norm=pl.col("country").str.to_lowercase().str.replace_all(r"[^\p{L}\p{N}]", ""),
        **{c: pl.Series(rows[c], dtype=pl.String) for c in STR_COLS},
        **{c: pl.Series(rows[c], dtype=pl.Int16) for c in INT_COLS},
    ).with_columns(
        name_addr=pl.concat_str([pl.col("name_core"), pl.lit(" | "), pl.col("addr_clean")]).str.strip_chars(" |"),
    )
    tmp = out.with_suffix(".tmp")
    res.write_parquet(tmp, compression="zstd")
    tmp.rename(out)                       # atomic: a half-written part never looks complete
    return len(res), int(res["name_ind"].sum()), int((res["name_core"] == "").sum())


# ----------------------------------------------------------------------------- driver
def build_skeleton_maps(data_dir: Path, pool, chunk: int) -> dict:
    s1 = read_tsv(data_dir / "train" / "train_source1.tsv")
    jobs = [(s1["business_name"][i:i + chunk].to_list(), s1["business_address"][i:i + chunk].to_list())
            for i in range(0, len(s1), chunk)]
    cn, ca = Counter(), Counter()
    for a, b in pool.map(_count_tokens, jobs):
        cn.update(a)
        ca.update(b)
    return {"name": N.build_skeleton_map(cn, N.NAME_CANON, min_count=3, min_len=4, lead=1.0),
            "addr": N.build_skeleton_map(ca, {**N.ADDR_CANON, **N.PHRASES}, min_count=3, min_len=4, lead=2.0)}


def run(data_dir: Path, work_dir: Path, jobs: int = 0, chunk: int = 100_000) -> Path:
    out_dir = norm_dir(work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = n_jobs(jobs)
    log(f"prep: normalising into {out_dir} with {jobs} workers")
    maps_p = out_dir / "skeleton_maps.json"
    if maps_p.exists():
        maps = json.loads(maps_p.read_text())
    else:
        with ProcessPoolExecutor(jobs) as pool:
            maps = build_skeleton_maps(data_dir, pool, chunk)
        maps_p.write_text(json.dumps(maps))
        log(f"prep: skeleton maps built from train S1: name={len(maps['name']):,} addr={len(maps['addr']):,}")
    todo = [(sp, so) for sp in SPLITS for so in SOURCES
            if (data_dir / sp / f"{sp}_{so}.tsv").exists() and not (out_dir / f"{sp}_{so}" / "_SUCCESS").exists()]
    if not todo:
        log("prep: all parts present, nothing to do")
        return out_dir
    with ProcessPoolExecutor(jobs, initializer=_init, initargs=(maps,)) as pool:
        for sp, so in todo:
            d = out_dir / f"{sp}_{so}"
            d.mkdir(exist_ok=True)
            df = read_tsv(data_dir / sp / f"{sp}_{so}.tsv")
            tasks = [(df.slice(i, chunk), d / f"part-{i // chunk:05d}.parquet") for i in range(0, len(df), chunk)]
            tasks = [t for t in tasks if not t[1].exists()]
            prog = Progress(len(df), f"prep {sp}_{so}", every=20)
            prog.update(len(df) - sum(len(t[0]) for t in tasks))
            n_ind = n_empty = 0
            for n, ind, empty in pool.map(_normalise_part, tasks):
                n_ind += ind
                n_empty += empty
                prog.update(n)
            (d / "_SUCCESS").write_text(str(len(df)))
            log(f"prep {sp}_{so}: {len(df):,} rows; Indic names in this run {n_ind:,}; empty name_core {n_empty:,}")
    return out_dir


def load(norm: Path, split: str, sources=SOURCES, columns: list[str] | None = None) -> pl.DataFrame:
    """Read normalised parts back (one or several sources of a split), in file order. Refuses to read a
    source whose normalisation has not completed (no _SUCCESS marker)."""
    for so in sources:
        if not (norm / f"{split}_{so}" / "_SUCCESS").exists():
            raise FileNotFoundError(f"{norm / f'{split}_{so}'} is incomplete - run prep.py first")
    frames = [pl.read_parquet(sorted((norm / f"{split}_{so}").glob("part-*.parquet")), columns=columns)
              for so in sources]
    return pl.concat(frames)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../../dataset")
    ap.add_argument("--work-dir", default="../../work")
    ap.add_argument("--n-jobs", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=100_000)
    a = ap.parse_args()
    run(Path(a.data_dir), Path(a.work_dir), a.n_jobs, a.chunk)
