"""
Build <team>_submission.zip in the layout the organisers require:

<team>_submission.zip
├── output/matching_results.tsv, candidate_pairs.tsv
├── code/business_entity_resolution/{src/, README.md, requirements.txt}
└── Documentation_template.md

    python tools/package_submission.py --team TEAMNAME --doc Documentation_template.md
"""
import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {"__pycache__", "cache", ".ipynb_checkpoints"}

ap = argparse.ArgumentParser()
ap.add_argument("--team", required=True)
ap.add_argument("--doc", default=str(ROOT / "Documentation_template.md"))
ap.add_argument("--output-dir", default=str(ROOT / "output"))
ap.add_argument("--keep-synthetic", action="store_true", help="include src/synthetic.py (dev-only)")
a = ap.parse_args()

out = Path(a.output_dir)
need = [out / "matching_results.tsv", out / "candidate_pairs.tsv", Path(a.doc)]
missing = [str(p) for p in need if not p.exists()]
if missing:
    raise SystemExit(f"missing: {missing}")

zp = ROOT / f"{a.team}_submission.zip"
code = ROOT / "code" / "business_entity_resolution"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(out / "matching_results.tsv", "output/matching_results.tsv")
    z.write(out / "candidate_pairs.tsv", "output/candidate_pairs.tsv")
    z.write(a.doc, Path(a.doc).name)
    for f in code.rglob("*"):
        if f.is_dir() or SKIP_DIRS & set(f.parts):
            continue
        if f.name == "synthetic.py" and not a.keep_synthetic:
            continue
        z.write(f, f"code/business_entity_resolution/{f.relative_to(code)}")
print("wrote", zp)
with zipfile.ZipFile(zp) as z:
    for n in z.namelist():
        print("  ", n)
