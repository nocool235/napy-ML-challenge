"""
Text normalisation for business names and addresses.

Design rules (important for the unseen-country test set, i.e. France):
  * Country-agnostic: the same pipeline runs on every record; nothing branches on country.
  * Symmetric canonicalisation: every variant of a token maps to ONE short canonical form
    (street/str/st -> st, saint -> st, avenue/av/ave -> ave ...). Both sides of a pair go
    through the same map, so collisions are harmless and variants always meet.
  * No external lookups (forbidden by the rules) - all dictionaries are hand-written here.

Add new abbreviations to the dicts below as error analysis finds them.
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

# --------------------------------------------------------------------------------------
# Dictionaries
# --------------------------------------------------------------------------------------
NAME_CANON = {
    # legal forms (EN / IN)
    "corporation": "corp", "corpn": "corp", "corp": "corp",
    "incorporated": "inc", "inc": "inc",
    "company": "co", "comp": "co", "co": "co", "cie": "co", "compagnie": "co",
    "limited": "ltd", "ltd": "ltd", "ltda": "ltd",
    "private": "pvt", "pvt": "pvt", "pvte": "pvt", "pte": "pvt", "prv": "pvt",
    "llc": "llc", "llp": "llp", "lp": "lp", "plc": "plc", "pllc": "pllc", "opc": "opc",
    # legal forms (FR / EU)
    "societe": "soc", "society": "soc", "soc": "soc",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "eurl": "eurl", "snc": "snc",
    "sci": "sci", "scp": "scp", "selarl": "selarl", "gmbh": "gmbh", "ag": "ag", "bv": "bv", "nv": "nv",
    "etablissements": "ets", "ets": "ets",
    # common business words
    "international": "intl", "intl": "intl", "internationale": "intl",
    "industries": "ind", "industry": "ind", "industrial": "ind", "inds": "ind", "indus": "ind",
    "services": "svc", "service": "svc", "svcs": "svc", "svc": "svc",
    "technologies": "tech", "technology": "tech", "technologie": "tech", "technologies": "tech", "tech": "tech",
    "manufacturing": "mfg", "mfg": "mfg",
    "enterprises": "ent", "enterprise": "ent", "entreprise": "ent", "ent": "ent",
    "associates": "assoc", "association": "assoc", "assoc": "assoc", "assocs": "assoc",
    "brothers": "bros", "bros": "bros", "bro": "bros", "freres": "bros",
    "department": "dept", "dept": "dept",
    "national": "natl", "natl": "natl", "nationale": "natl",
    "center": "ctr", "centre": "ctr", "ctr": "ctr",
    "solutions": "soln", "solution": "soln", "soln": "soln", "solns": "soln",
    "systems": "sys", "system": "sys", "sys": "sys", "systemes": "sys",
    "management": "mgmt", "mgmt": "mgmt",
    "group": "grp", "grp": "grp", "groupe": "grp",
    "holdings": "hldg", "holding": "hldg", "hldgs": "hldg", "hldg": "hldg",
    "hospital": "hosp", "hosp": "hosp", "hopital": "hosp",
    "pharmaceuticals": "pharma", "pharmaceutical": "pharma", "pharma": "pharma", "pharmacie": "pharmacy",
    "restaurant": "rest", "restaurants": "rest", "rest": "rest",
    "trading": "trdg", "traders": "trdr", "trader": "trdr",
    "shree": "sri", "shri": "sri", "sree": "sri", "sri": "sri", "sh": "sri",
    "saint": "st", "st": "st", "sainte": "ste", "ste": "ste",
    "and": "and", "et": "and", "n": "and",
    "mountain": "mtn", "mtn": "mtn", "mount": "mt", "mt": "mt",
}

LEGAL_TOKENS = {
    "inc", "corp", "co", "ltd", "pvt", "llc", "llp", "lp", "plc", "pllc", "opc", "pc",
    "sarl", "sas", "sasu", "sa", "eurl", "snc", "sci", "scp", "selarl", "gmbh", "ag",
    "bv", "nv", "ets",
}
NAME_STOP = {"the", "of", "a", "an", "and", "le", "la", "les", "de", "des", "du", "l", "d"}

ADDR_CANON = {
    "street": "st", "str": "st", "st": "st", "saint": "st", "strt": "st",
    "road": "rd", "rd": "rd",
    "avenue": "ave", "ave": "ave", "av": "ave", "aven": "ave", "avn": "ave",
    "boulevard": "blvd", "blvd": "blvd", "bd": "blvd", "boul": "blvd", "bvd": "blvd",
    "drive": "dr", "dr": "dr", "drv": "dr",
    "lane": "ln", "ln": "ln",
    "court": "ct", "ct": "ct", "crt": "ct",
    "place": "pl", "pl": "pl", "plaza": "plz", "plz": "plz",
    "square": "sq", "sq": "sq",
    "highway": "hwy", "hwy": "hwy", "hiway": "hwy",
    "parkway": "pkwy", "pkwy": "pkwy",
    "circle": "cir", "cir": "cir",
    "terrace": "ter", "ter": "ter",
    "suite": "ste", "ste": "ste", "sainte": "ste", "unit": "ste",
    "apartment": "apt", "apt": "apt", "flat": "apt",
    "building": "bldg", "bldg": "bldg", "bld": "bldg", "batiment": "bldg", "bat": "bldg",
    "floor": "flr", "flr": "flr", "fl": "flr", "etage": "flr",
    "near": "near", "nr": "near", "nar": "near",
    "opposite": "opp", "opp": "opp", "opps": "opp",
    "behind": "behind", "bhd": "behind", "beside": "beside", "besides": "beside",
    "north": "n", "south": "s", "east": "e", "west": "w", "nord": "n", "sud": "s", "est": "e", "ouest": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    "mount": "mt", "mt": "mt", "fort": "ft", "ft": "ft",
    "sector": "sec", "sec": "sec", "sect": "sec",
    "nagar": "nagar", "ngr": "nagar",
    "district": "dist", "dist": "dist", "distt": "dist",
    "colony": "colony", "col": "colony",
    "extension": "ext", "extn": "ext", "ext": "ext",
    "phase": "ph", "ph": "ph",
    "market": "mkt", "mkt": "mkt", "marche": "mkt",
    "industrial": "ind", "indl": "ind", "ind": "ind",
    "estate": "est", "estt": "est",
    "cross": "cross", "crs": "cross",
    "main": "main", "mn": "main",
    "post": "po", "po": "po",
    "chemin": "ch", "ch": "ch", "chem": "ch",
    "impasse": "imp", "imp": "imp",
    "allee": "allee", "all": "allee",
    "route": "rte", "rte": "rte",
    "faubourg": "fbg", "fbg": "fbg",
    "rue": "rue", "r": "rue",
    "quai": "quai", "qu": "quai",
    "cours": "crs",
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
    "sixth": "6", "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10",
    "premier": "1", "premiere": "1", "deuxieme": "2", "troisieme": "3",
}
ADDR_DROP = {"no", "number", "num", "nos", "the", "of", "de", "du", "des", "la", "le", "les", "d", "l", "and", "et"}
LANDMARK_WORDS = {"near", "opp", "behind", "beside", "next", "adjacent", "adj", "opposite", "pres", "face"}

# Multi-word phrases replaced before tokenisation (state names etc.). Symmetric: all
# variants go to the same short form.
PHRASES = {
    # US states
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
    "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "district of columbia": "dc",
    # Indian states / cities with common variants
    "uttar pradesh": "up", "madhya pradesh": "mp", "andhra pradesh": "ap", "himachal pradesh": "hp",
    "arunachal pradesh": "arp", "maharashtra": "mh", "karnataka": "ka", "tamil nadu": "tn",
    "tamilnadu": "tn", "west bengal": "wb", "gujarat": "gj", "gujrat": "gj", "rajasthan": "rj",
    "haryana": "hr", "punjab": "pb", "kerala": "kl", "telangana": "ts", "bihar": "br",
    "odisha": "od", "orissa": "od", "jharkhand": "jh", "chhattisgarh": "cg", "uttarakhand": "uk",
    "jammu and kashmir": "jk", "jammu kashmir": "jk", "new delhi": "delhi", "nct of delhi": "delhi",
    "bombay": "mumbai", "calcutta": "kolkata", "madras": "chennai", "bengaluru": "bangalore",
    "gurugram": "gurgaon", "poona": "pune", "mysuru": "mysore",
    "mahatma gandhi": "mg", "m g": "mg",
    # FR
    "cedex": "",
}
_PHRASE_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, PHRASES), key=len, reverse=True)) + r")\b")

_ORDINAL_RE = re.compile(r"\b(\d+)\s*(st|nd|rd|th|er|ere|eme|e|ème)\b")
_PIN_SPLIT_RE = re.compile(r"\b(\d{3})\s(\d{3})\b")          # "110 001" -> "110001"
_ZIP4_RE = re.compile(r"\b(\d{5})-(\d{4})\b")                  # "12345-6789" -> "12345"
_DBA_RE = re.compile(r"\b(?:d\s*/?\s*b\s*/?\s*a|doing business as|trading as|t\s*/\s*a|a\s*/?\s*k\s*/?\s*a|aka|formerly)\b")
_NUM_RE = re.compile(r"\d+")


# --------------------------------------------------------------------------------------
# Primitive cleaning
# --------------------------------------------------------------------------------------
def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def basic_clean(s) -> str:
    """lowercase, accents stripped, & -> and, punctuation -> space, whitespace collapsed."""
    if not isinstance(s, str):
        return ""
    s = strip_accents(s).lower()
    s = s.replace("&", " and ").replace("@", " at ")
    s = re.sub(r"(?<=\w)['’`](?=\w)", "", s)       # o'brien -> obrien
    s = re.sub(r"(?<=\b\w)\.(?=\w\b)", "", s)         # p.v.t / l.l.c -> pvt / llc
    s = s.replace(".", " ")
    s = re.sub(r"[^a-z0-9/#\-\s]", " ", s)
    s = s.replace("/", " / ")
    return re.sub(r"\s+", " ", s).strip()


def _canon_tokens(text: str, canon: dict, drop: set) -> list[str]:
    toks = []
    for t in text.replace("-", " ").replace("#", " ").replace("/", " ").split():
        t = canon.get(t, t)
        if t and t not in drop:
            toks.append(t)
    return toks


# --------------------------------------------------------------------------------------
# Names
# --------------------------------------------------------------------------------------
def name_fields(raw) -> dict:
    s = basic_clean(raw)
    # alternative names: DBA / trade names / parenthesised parts
    parts = [p.strip() for p in _DBA_RE.split(s) if p.strip()]
    paren = re.findall(r"\(([^)]*)\)", str(raw).lower()) if isinstance(raw, str) else []
    parts += [basic_clean(p) for p in paren if basic_clean(p)]
    s_noparen = re.sub(r"\([^)]*\)", " ", s)
    s_noparen = _DBA_RE.sub(" ", s_noparen)

    toks = _canon_tokens(s_noparen, NAME_CANON, set())
    core = [t for t in toks if t not in LEGAL_TOKENS and t not in NAME_STOP]
    legal = sorted({t for t in toks if t in LEGAL_TOKENS})
    alt = []
    for p in parts:
        pt = [t for t in _canon_tokens(p, NAME_CANON, set()) if t not in LEGAL_TOKENS and t not in NAME_STOP]
        if pt:
            alt.append(" ".join(pt))
    core_str = " ".join(core) if core else " ".join(toks)
    return {
        "name_clean": " ".join(toks),
        "name_core": core_str,
        "name_nospace": core_str.replace(" ", ""),
        "name_sorted": " ".join(sorted(core_str.split())),
        "name_initials": "".join(t[0] for t in core if t and not t.isdigit()),
        "name_legal": " ".join(legal),
        "name_alts": "|".join(dict.fromkeys(alt + [core_str])),
        "name_nums": " ".join(_NUM_RE.findall(core_str)),
    }


# --------------------------------------------------------------------------------------
# Addresses
# --------------------------------------------------------------------------------------
def _addr_prep(s: str) -> str:
    s = _ZIP4_RE.sub(r"\1", s)
    s = _PIN_SPLIT_RE.sub(r"\1\2", s)
    s = _ORDINAL_RE.sub(r"\1", s)
    return _PHRASE_RE.sub(lambda m: PHRASES[m.group(1)], s)


def addr_fields(raw) -> dict:
    s = _addr_prep(basic_clean(raw))

    # landmark removal works on comma-separated segments of the raw string
    raw_l = strip_accents(raw).lower() if isinstance(raw, str) else ""
    segs = [basic_clean(x) for x in re.split(r"[,;]", raw_l)]
    keep, landmarks = [], []
    for seg in segs:
        first = seg.split()[:2]
        first = [ADDR_CANON.get(t, t) for t in first]
        (landmarks if any(t in LANDMARK_WORDS for t in first) else keep).append(seg)

    toks = _canon_tokens(s, ADDR_CANON, ADDR_DROP)
    nolm = _addr_prep(" ".join(keep))
    nolm_toks = _canon_tokens(nolm, ADDR_CANON, ADDR_DROP)

    nums = _NUM_RE.findall(" ".join(toks))
    postcodes = [n for n in nums if len(n) in (5, 6)]
    house = next((n for n in nums if 1 <= len(n) <= 4), "")
    words = [t for t in toks if not t.isdigit()]
    return {
        "addr_clean": " ".join(toks),
        "addr_nolm": " ".join(nolm_toks) if nolm_toks else " ".join(toks),
        "addr_sorted": " ".join(sorted(toks)),
        "addr_words": " ".join(words),
        "addr_nums": " ".join(nums),
        "addr_postcode": " ".join(dict.fromkeys(postcodes)),
        "addr_house": house,
        "addr_has_landmark": int(bool(landmarks)),
        "addr_ntok": len(toks),
    }


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Adds all normalised columns to a source frame (entity_id, business_name, business_address, country)."""
    out = df.copy()
    out["business_name"] = out["business_name"].fillna("").astype(str)
    out["business_address"] = out["business_address"].fillna("").astype(str)
    out["country"] = out["country"].fillna("").astype(str)
    nf = pd.DataFrame([name_fields(x) for x in out["business_name"]], index=out.index)
    af = pd.DataFrame([addr_fields(x) for x in out["business_address"]], index=out.index)
    out = pd.concat([out, nf, af], axis=1)
    out["country_norm"] = out["country"].map(lambda c: basic_clean(c).replace(" ", ""))
    out["source"] = out["entity_id"].str.slice(0, 2)
    out["name_addr"] = (out["name_core"] + " | " + out["addr_clean"]).str.strip()
    # word view for blocking: postcode / house-number tokens (prefixed) + name words
    out["blk_nums_name"] = (
        out["addr_postcode"].str.split().map(lambda xs: " ".join("pc" + x for x in xs))
        + " " + out["addr_house"].map(lambda x: ("hn" + x) if x else "")
        + " " + out["name_core"]
    ).str.strip()
    return out


if __name__ == "__main__":  # quick manual check
    for n in ["Shree Ganesh Traders Pvt. Ltd.", "ABC Holdings LLC d/b/a Joe's Pizza", "Société Générale S.A.", "Wal-Mart Stores, Inc."]:
        print(n, "->", name_fields(n))
    for a in ["12, MG Road, Near SBI ATM, Bengaluru, Karnataka 560 001", "1st Street, Suite 200, New York, NY 10001-1234", "12 bis Av. des Champs-Élysées, 75008 Paris Cedex"]:
        print(a, "->", addr_fields(a))
