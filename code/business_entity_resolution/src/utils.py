"""Small shared helpers: flushed logging with ETA, config hashing, machine resources, TSV reading."""
from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path

import polars as pl

T0 = time.time()


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:7.0f}s] {msg}", flush=True)


class Progress:
    """Logs every `every` seconds: done/total, rate, ETA and projected finish time."""

    def __init__(self, total: float, what: str, every: float = 30.0):
        self.total, self.what, self.every = total, what, every
        self.t0 = self.last = time.time()
        self.done = 0.0

    def update(self, n: float, extra: str = ""):
        self.done += n
        now = time.time()
        if now - self.last >= self.every or self.done >= self.total:
            self.last = now
            rate = self.done / max(now - self.t0, 1e-9)
            eta = (self.total - self.done) / max(rate, 1e-9)
            log(f"{self.what}: {self.done:,.0f}/{self.total:,.0f} ({self.done / max(self.total, 1):.0%}) "
                f"{rate:,.0f}/s, ETA {eta / 60:.1f} min (~{time.strftime('%H:%M', time.localtime(now + eta))}) {extra}")


def code_hash(*files: Path, extra: str = "") -> str:
    h = hashlib.sha1(extra.encode())
    for f in files:
        h.update(Path(f).read_bytes())
    return h.hexdigest()[:10]


def n_jobs(requested: int = 0) -> int:
    return requested if requested > 0 else max(1, os.cpu_count() or 1)


def available_memory_gb() -> float:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024 ** 2
    except OSError:
        pass
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3 * 0.5


def read_tsv(p: Path) -> pl.DataFrame:
    """All columns as strings, empty fields as "" (not null), no quoting (addresses contain quotes)."""
    return pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False,
                       missing_utf8_is_empty_string=True)


def die(msg: str):
    log("FATAL: " + msg)
    sys.exit(1)
