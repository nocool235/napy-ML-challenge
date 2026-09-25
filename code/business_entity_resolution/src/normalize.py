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

from translit import INDIC_RE, skeleton, translit

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
NAME_STOP = {"the", "of", "a", "an", "and", "le", "la", "les", "de", "des", "du", "l", "d",
             # web handles, honorifics and country words that the sources add or drop at random
             "www", "com", "net", "org", "http", "https", "mr", "mrs", "ms", "messrs", "smt",
             "india", "france", "usa"}

# Transliterated (Indic script) spellings of legal forms / connectors that the skeleton map can't
# resolve because they are abbreviations ("प्रा. लि." = Pvt. Ltd.).
TRANSLIT_CANON = {"pra": "pvt", "li": "ltd", "praivet": "pvt", "prayvet": "pvt", "privet": "pvt",
                  "limitad": "ltd", "limited": "ltd", "elelpi": "llp", "elalpi": "llp", "kampani": "co",
                  "kampni": "co", "end": "and", "aind": "and"}

# skeleton -> most frequent Latin word with that skeleton, per field; built from train S1 by prep.py
# (data-driven, no external data). Transliterated words are replaced by that Latin word.
SKEL_MAPS: dict[str, dict[str, str]] = {"name": {}, "addr": {}}


def set_skeleton_maps(maps: dict[str, dict[str, str]]):
    SKEL_MAPS["name"], SKEL_MAPS["addr"] = maps.get("name", {}), maps.get("addr", {})


def build_skeleton_map(token_counts: dict[str, int], canon: dict[str, str], min_count: int = 3,
                       min_len: int = 4, lead: float = 1.0) -> dict[str, str]:
    """skeleton -> dominant Latin token. Candidates are pooled by canonical form (sri/shree/shri count
    together). Skeletons of >= min_len chars need a `lead`x lead over the runner-up, shorter ones 5x -
    short skeletons are ambiguous ("brt" = bharat / bright), and a wrong snap is worse than none."""
    by_sk: dict[str, dict[str, list]] = {}
    for tok, c in token_counts.items():
        if c >= min_count and tok.isalpha() and len(tok) >= 2:
            g = by_sk.setdefault(skeleton(tok), {}).setdefault(canon.get(tok, tok), [0, tok, 0])
            g[0] += c
            if c > g[2]:
                g[1], g[2] = tok, c
    out = {}
    for sk, groups in by_sk.items():
        if len(sk) < 2:
            continue
        lst = sorted(groups.values(), reverse=True)
        need = lead if len(sk) >= min_len else 5.0
        if len(lst) == 1 or lst[0][0] >= need * lst[1][0]:
            out[sk] = lst[0][1]
    return out


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
    "eleventh": "11", "twelfth": "12", "thirteenth": "13", "fourteenth": "14", "fifteenth": "15",
    "sixteenth": "16", "seventeenth": "17", "eighteenth": "18", "nineteenth": "19", "twentieth": "20",
    "thirtieth": "30", "fortieth": "40", "fiftieth": "50",
}
ADDR_DROP = {"no", "number", "num", "nos", "the", "of", "de", "du", "des", "la", "le", "les", "d", "l", "and", "et",
             # empty placeholders and unit words the sources swap freely ("Flat No 7" / "H.NO 7" / "DOOR NO 7")
             "null", "nil", "none", "house", "door", "plot", "shop", "office"}
STREET_WORDS = {"st", "rd", "ave", "blvd", "dr", "ln", "ct", "pl", "plz", "sq", "hwy", "pkwy", "cir", "ter", "way",
                "rue", "allee", "ch", "imp", "rte", "quai", "crs", "fbg", "n", "s", "e", "w", "ne", "nw", "se", "sw",
                "ste", "apt", "flr", "bldg", "main", "cross", "sec"}

# French regions and departements -> one region code. Applied only when a whole comma-separated
# component is the name, so "Rue du Nord" or the city of Paris are untouched.
_FR_REGIONS = {
    "frara": "auvergne rhone alpes|ain|allier|ardeche|cantal|drome|isere|loire|haute loire|puy de dome|rhone|savoie|haute savoie",
    "frbfc": "bourgogne franche comte|cote d or|doubs|jura|nievre|haute saone|saone et loire|yonne|territoire de belfort",
    "frbre": "bretagne|cotes d armor|finistere|ille et vilaine|morbihan",
    "frcvl": "centre val de loire|cher|eure et loir|indre|indre et loire|loir et cher|loiret",
    "frcor": "corse|corse du sud|haute corse",
    "frges": "grand est|ardennes|aube|marne|haute marne|meurthe et moselle|meuse|moselle|bas rhin|haut rhin|vosges",
    "frhdf": "hauts de france|aisne|nord|oise|pas de calais|somme",
    "fridf": "ile de france|seine et marne|yvelines|essonne|hauts de seine|seine saint denis|val de marne|val d oise",
    "frnor": "normandie|calvados|eure|manche|orne|seine maritime",
    "frnaq": "nouvelle aquitaine|charente|charente maritime|correze|creuse|dordogne|gironde|landes|lot et garonne|"
             "pyrenees atlantiques|deux sevres|vienne|haute vienne",
    "frocc": "occitanie|ariege|aude|aveyron|gard|haute garonne|gers|herault|lot|lozere|hautes pyrenees|"
             "pyrenees orientales|tarn|tarn et garonne",
    "frpdl": "pays de la loire|loire atlantique|maine et loire|mayenne|sarthe|vendee",
    "frpac": "provence alpes cote d azur|alpes de haute provence|hautes alpes|alpes maritimes|bouches du rhone|var|vaucluse",
}
FR_ADMIN = {name: code for code, names in _FR_REGIONS.items() for name in names.split("|")}
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
_MS_RE = re.compile(r"(?i)\b(m\s*/\s*s|messrs)\b\.?")
_HANDLE_RE = re.compile(r"(^|\s)@(?=\w)")
_CO_RE = re.compile(r"(?i)\bc\s*/\s*o\b")
_HNO_RE = re.compile(r"\b(h|d|hno|dno)\s*no\b")
_BIS_RE = re.compile(r"\b(\d+)\s*(bis|ter|quater)\b")
_LEET = str.maketrans("013458", "oleasb")


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
def _translit_words(s: str, field: str) -> str:
    """Transliterate Indic words one by one and snap each to the Latin word with the same skeleton."""
    out = []
    for w in s.split():
        if INDIC_RE.search(w):
            t = re.sub(r"[^a-z0-9]", "", strip_accents(translit(w)).lower())
            if t and not t.isdigit():
                t = TRANSLIT_CANON.get(t) or SKEL_MAPS[field].get(skeleton(t)) or t
            out.append(t)
        else:
            out.append(w)
    return " ".join(out)


def _leet(t: str) -> str:
    """'chir0practic' -> 'chiropractic', '8rotheas' -> 'brotheas'; leaves codes like 'b2b' or '24x7' alone."""
    if t.isalpha() or t.isdigit():
        return t
    digits = [c for c in t if c.isdigit()]
    if len(t) - len(digits) >= 3 and len(digits) <= 2 and all(c in "013458" for c in digits):
        return t.translate(_LEET)
    return t


def _name_tokens(text: str) -> list[str]:
    toks = []
    for t in text.replace("-", " ").replace("#", " ").replace("/", " ").split():
        t = NAME_CANON.get(_leet(t), _leet(t))
        if t and (not toks or toks[-1] != t):          # "OWL OWL" -> "owl"
            toks.append(t)
    return toks


def name_fields(raw) -> dict:
    raw = raw if isinstance(raw, str) else ""
    ind = bool(INDIC_RE.search(raw))
    if ind:
        raw = _translit_words(raw, "name")
    raw = _HANDLE_RE.sub(r"\1", _MS_RE.sub(" ", raw))
    s = basic_clean(raw)
    # alternative names: DBA / trade names / parenthesised parts
    parts = [p.strip() for p in _DBA_RE.split(s) if p.strip()]
    paren = re.findall(r"\(([^)]*)\)", raw.lower())
    parts += [basic_clean(p) for p in paren if basic_clean(p)]
    s_noparen = re.sub(r"\([^)]*\)", " ", s)
    s_noparen = _DBA_RE.sub(" ", s_noparen)

    toks = _name_tokens(s_noparen)
    core = [t for t in toks if t not in LEGAL_TOKENS and t not in NAME_STOP]
    legal = sorted({t for t in toks if t in LEGAL_TOKENS})
    alt = []
    for p in parts:
        pt = [t for t in _name_tokens(p) if t not in LEGAL_TOKENS and t not in NAME_STOP]
        if pt:
            alt.append(" ".join(pt))
    core_str = " ".join(core) if core else " ".join(t for t in toks if t not in NAME_STOP)
    return {
        "name_clean": " ".join(toks),
        "name_core": core_str,
        "name_nospace": core_str.replace(" ", ""),
        "name_initials": "".join(t[0] for t in core if t and not t.isdigit()),
        "name_legal": " ".join(legal),
        "name_alts": "|".join(dict.fromkeys(alt + [core_str])),
        "name_nums": " ".join(_NUM_RE.findall(core_str)),
        "name_skel": " ".join(k for k in (skeleton(t) for t in core_str.split()) if k),
        "name_ind": int(ind),
    }


# --------------------------------------------------------------------------------------
# Addresses
# --------------------------------------------------------------------------------------
def _addr_prep(s: str) -> str:
    s = _ZIP4_RE.sub(r"\1", s)
    s = _PIN_SPLIT_RE.sub(r"\1\2", s)
    s = _ORDINAL_RE.sub(r"\1", s)
    s = _BIS_RE.sub(r"\1", s)
    s = _HNO_RE.sub(" ", s)
    return _PHRASE_RE.sub(lambda m: PHRASES[m.group(1)], s)


def _admin_key(c: str) -> str:
    return re.sub(r"\s+", " ", basic_clean(c).replace("-", " ")).strip()


def addr_fields(raw) -> dict:
    raw = raw if isinstance(raw, str) else ""
    ind = bool(INDIC_RE.search(raw))
    if ind:
        raw = _translit_words(raw, "addr")
    raw = _CO_RE.sub(" ", raw)
    comps = re.split(r"[,;]", raw)
    comps = [FR_ADMIN.get(_admin_key(c), c) for c in comps]
    raw = ", ".join(comps)
    s = _addr_prep(basic_clean(raw))

    # landmark removal and positional number logic work on comma-separated segments
    raw_l = strip_accents(raw).lower()
    segs = [basic_clean(x) for x in re.split(r"[,;]", raw_l)]
    keep, landmarks = [], []
    seg_toks = []
    for seg in segs:
        first = [ADDR_CANON.get(t, t) for t in seg.split()[:2]]
        (landmarks if any(t in LANDMARK_WORDS for t in first) else keep).append(seg)
        seg_toks.append(_canon_tokens(_addr_prep(seg), ADDR_CANON, ADDR_DROP))

    toks = [t.lstrip("0") or "0" if t.isdigit() else t for t in _canon_tokens(s, ADDR_CANON, ADDR_DROP)]
    nolm = _addr_prep(" ".join(keep))
    nolm_toks = _canon_tokens(nolm, ADDR_CANON, ADDR_DROP)

    # postcode = 5-6 digit number that does not start a "number street" run; house = first other number
    postcodes, house, street = [], "", ""
    for ct in seg_toks:
        for k, t in enumerate(ct):
            if not t.isdigit():
                continue
            n = t.lstrip("0") or "0"
            starts_street = k + 1 < len(ct) and not ct[k + 1].isdigit()
            if len(t) in (5, 6) and not (k == 0 and starts_street):
                postcodes.append(n)
            elif not house:
                house = n
                nxt = next((x for x in ct[k + 1:] if x not in STREET_WORDS and (x.isdigit() or len(x) > 1)), "")
                street = f"{n} {nxt}" if nxt else ""
    nums = [n.lstrip("0") or "0" for n in _NUM_RE.findall(" ".join(toks))]
    words = [t for t in toks if not t.isdigit()]
    return {
        "addr_clean": " ".join(toks),
        "addr_nolm": " ".join(nolm_toks) if nolm_toks else " ".join(toks),
        "addr_words": " ".join(words),
        "addr_nums": " ".join(nums),
        "addr_postcode": " ".join(dict.fromkeys(postcodes)),
        "addr_house": house,
        "addr_street": street,
        "addr_has_landmark": int(bool(landmarks)),
        "addr_ntok": len(toks),
        "addr_ind": int(ind),
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
