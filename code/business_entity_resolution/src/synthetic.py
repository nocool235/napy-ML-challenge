"""
DEV ONLY - generates a fake dataset in the exact competition layout so the pipeline can be
smoke-tested without the real data:  <out>/train/train_source{1,2,3}.tsv, train_ground_truth.tsv,
<out>/test/test_source{1,2,3}.tsv  (test includes a France slice, like the real test set).
Not used by the real pipeline; delete it from the final zip if you like.

    python src/synthetic.py --out ../../synthetic_dataset --n-train 3000 --n-test 1500
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

R = random.Random(7)
W = {
    "US": dict(
        words="Summit Eagle Pioneer Liberty Golden Harbor Pacific Atlantic Cedar Maple Granite River Blue Star Apex Prime Vertex Keystone Heritage Frontier Silver Oak Pine Metro Coastal Northern United American Rocky Valley Sunrise".split(),
        kinds="Bakery Plumbing Consulting Logistics Dental Motors Realty Pharmacy Construction Foods Electric Roofing Insurance Software Printing Auto Repair Landscaping Fitness Cafe".split(),
        legal=["Inc", "Inc.", "LLC", "Corp", "Corporation", "Co", "Company", "Ltd", ""],
        streets="Main Oak Maple Washington Lake Hill Park Elm Pine Cedar Sunset Madison Franklin Jackson Lincoln".split(),
        stype=["Street", "St", "Avenue", "Ave", "Road", "Rd", "Boulevard", "Blvd", "Drive", "Dr", "Lane"],
        cities=[("Austin", "TX", "Texas"), ("Denver", "CO", "Colorado"), ("Seattle", "WA", "Washington"),
                ("Chicago", "IL", "Illinois"), ("Boston", "MA", "Massachusetts"), ("Miami", "FL", "Florida"),
                ("New York", "NY", "New York"), ("Phoenix", "AZ", "Arizona")],
    ),
    "India": dict(
        words="Shree Sri Ganesh Lakshmi Balaji Om Sai Krishna Bharat Hindustan Maruti Jai Ambika Vinayak Shiv Durga Mahalaxmi Royal Sunrise Star New Modern Supreme Galaxy Annapurna".split(),
        kinds="Traders Enterprises Textiles Jewellers Electronics Sweets Medicals Hardware Motors Steels Builders Pharma Agencies Industries Tours Stores".split(),
        legal=["Pvt Ltd", "Private Limited", "Pvt. Ltd.", "Ltd", "Limited", "LLP", ""],
        streets="MG Station Nehru Gandhi Ring Link Mall Temple Market Civil Lines Sector 18 Sector 62".split(),
        stype=["Road", "Rd", "Marg", "Nagar", "Street"],
        cities=[("Mumbai", "400001", "Maharashtra"), ("Delhi", "110001", "Delhi"), ("Bengaluru", "560001", "Karnataka"),
                ("Noida", "201301", "Uttar Pradesh"), ("Pune", "411001", "Maharashtra"), ("Jaipur", "302001", "Rajasthan"),
                ("Chennai", "600001", "Tamil Nadu"), ("Kolkata", "700001", "West Bengal")],
        landmarks=["SBI ATM", "Hanuman Mandir", "Metro Station", "City Mall", "Bus Stand", "HDFC Bank", "Railway Station"],
    ),
    "France": dict(
        words="Boulangerie Maison Atelier Garage Pharmacie Cabinet Societe Hotel Brasserie Librairie Fromagerie Transports Menuiserie Boucherie Galerie Clinique".split(),
        kinds="Dupont Martin Lefevre Moreau Laurent Girard Bonnet Mercier Rousseau Lambert Fontaine Chevalier Durand Blanc Garnier".split(),
        legal=["SARL", "SAS", "S.A.", "EURL", "SA", ""],
        streets=["de la République", "Victor Hugo", "Jean Jaurès", "de Paris", "Pasteur", "du Général de Gaulle", "des Lilas", "Saint-Michel", "Gambetta"],
        stype=["Rue", "Avenue", "Av.", "Boulevard", "Bd", "Place", "Chemin", "Allée"],
        cities=[("Paris", "75008", "Île-de-France"), ("Lyon", "69002", "Auvergne-Rhône-Alpes"), ("Marseille", "13001", "Provence"),
                ("Toulouse", "31000", "Occitanie"), ("Bordeaux", "33000", "Nouvelle-Aquitaine"), ("Lille", "59000", "Hauts-de-France")],
    ),
}
ABBR = {"Street": "St", "Road": "Rd", "Avenue": "Ave", "Boulevard": "Blvd", "Drive": "Dr", "Private": "Pvt",
        "Limited": "Ltd", "Corporation": "Corp", "Company": "Co", "Incorporated": "Inc", "and": "&", "Rue": "R.",
        "Avenue ": "Av. ", "Saint": "St", "Enterprises": "Ent", "Industries": "Inds", "Shree": "Shri", "Sri": "Shree"}


def make_entity(country):
    w = W[country]
    if country == "France":
        name = f"{R.choice(w['words'])} {R.choice(w['kinds'])}"
    else:
        name = " ".join(R.sample(w["words"], R.choice([1, 2]))) + " " + R.choice(w["kinds"])
    legal = R.choice(w["legal"])
    city = R.choice(w["cities"])
    num = str(R.randint(1, 9999 if country == "US" else 300))
    street = f"{R.choice(w['streets'])} {R.choice(w['stype'])}" if country != "France" else f"{R.choice(w['stype'])} {R.choice(w['streets'])}"
    if country == "US":
        zipc = f"{R.randint(10000, 99999)}"
        addr = f"{num} {street}, {city[0]}, {city[1]} {zipc}"
    elif country == "India":
        pin = str(int(city[1]) + R.randint(0, 99))
        addr = f"{num}, {street}, {city[0]}, {city[2]} {pin}"
    else:
        cp = str(int(city[1]) + R.randint(0, 20)).zfill(5)
        addr = f"{num} {street}, {cp} {city[0]}"
    return {"business_name": f"{name} {legal}".strip(), "business_address": addr, "country": country}


def typo(s):
    if len(s) < 4:
        return s
    k = R.randrange(1, len(s) - 1)
    op = R.random()
    if op < 0.33:
        return s[:k] + s[k + 1:]
    if op < 0.66:
        return s[:k] + s[k + 1] + s[k] + s[k + 2:]
    return s[:k] + R.choice("aeiourstn") + s[k:]


def noisy(e):
    n, a = e["business_name"], e["business_address"]
    for full, short in ABBR.items():
        if R.random() < 0.5:
            n = n.replace(full, short); a = a.replace(full, short)
    if R.random() < 0.3:
        n = " ".join(n.split()[:-1]) if len(n.split()) > 2 else n
    if R.random() < 0.3:
        n = typo(n)
    if R.random() < 0.15:
        t = n.split(); R.shuffle(t); n = " ".join(t)
    if R.random() < 0.1:
        n = n.upper()
    if R.random() < 0.08:
        n = f"{n} DBA {R.choice(['Quick', 'Best', 'City'])} {R.choice(['Mart', 'Shop', 'Point'])}"
    parts = [p.strip() for p in a.split(",")]
    if R.random() < 0.3 and len(parts) > 2:
        parts.pop(R.randrange(1, len(parts)))
    if e["country"] == "India" and R.random() < 0.35:
        parts.insert(R.randrange(1, len(parts) + 1), f"Near {R.choice(W['India']['landmarks'])}")
    if R.random() < 0.2:
        a = typo(", ".join(parts))
    else:
        a = ", ".join(parts)
    if R.random() < 0.05:
        a = ""
    return {"business_name": n, "business_address": a, "country": e["country"]}


def build(n, countries, split, out: Path, with_truth: bool):
    ents = []
    chain_pool = []
    for _ in range(n):
        c = R.choices(countries, weights=[1] * len(countries))[0]
        e = make_entity(c)
        if chain_pool and R.random() < 0.12:        # chain store: same name, different address
            src = R.choice([x for x in chain_pool if x["country"] == c] or [e])
            e["business_name"] = src["business_name"]
        chain_pool.append(e)
        ents.append(e)
    s1, s2, s3, gt = [], [], [], []
    for k, e in enumerate(ents):
        sid = f"S1-{k + 1:05d}"
        s1.append({"entity_id": sid, **e})
        matches = []
        if R.random() > 0.35:  # 35% singletons
            for src, lst, lam in (("S2", s2, 1.2), ("S3", s3, 0.8)):
                for _ in range(min(3, int(R.expovariate(1 / lam)))):
                    rid = f"{src}-{len(lst) + 1:05d}"
                    lst.append({"entity_id": rid, **noisy(e)}); matches.append(rid)
        gt.append({"source1_entity_id": sid, "matched_entity_ids": ",".join(matches)})
    for lst, src in ((s2, "S2"), (s3, "S3")):          # distractors: businesses absent from S1
        for _ in range(n // 3):
            e = make_entity(R.choice(countries))
            lst.append({"entity_id": f"{src}-{len(lst) + 1:05d}", **noisy(e)})
    for lst in (s2, s3):
        R.shuffle(lst)
    d = out / split; d.mkdir(parents=True, exist_ok=True)
    for name, rows in (("source1", s1), ("source2", s2), ("source3", s3)):
        pd.DataFrame(rows).to_csv(d / f"{split}_{name}.tsv", sep="\t", index=False)
    if with_truth:
        pd.DataFrame(gt).to_csv(d / f"{split}_ground_truth.tsv", sep="\t", index=False)
    else:  # keep hidden truth next to it for local scoring of the smoke test
        pd.DataFrame(gt).to_csv(out / f"_hidden_{split}_ground_truth.tsv", sep="\t", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="synthetic_dataset")
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-test", type=int, default=1500)
    a = ap.parse_args()
    out = Path(a.out)
    build(a.n_train, ["US", "India"], "train", out, True)
    build(a.n_test, ["US", "India", "France"], "test", out, False)
    print("synthetic data written to", out)
