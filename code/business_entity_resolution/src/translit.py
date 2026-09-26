"""
Rule-based transliteration of Indic (Brahmic) scripts to Latin, plus a phonetic skeleton.
Hand-written tables only - no external data.

Unicode lays out Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada and
Malayalam in parallel 128-code-point blocks (the ISCII layout), so one table indexed by the offset
inside the block covers all nine scripts; a few script-specific letters are overridden.
The output is a rough phonetic spelling ("मार्केटिंग" -> "marketing") meant for fuzzy matching
against Latin names, not a scholarly romanisation.

skeleton() maps a Latin word to a consonant class string so that English spellings and
transliterated spellings of the same word collide ("consultancy" / "kansaltensi" -> "knsltns").
"""
from __future__ import annotations

import re

INDIC_RE = re.compile(r"[ऀ-ൿ]")

# offset inside a block -> (kind, latin). kinds: V vowel, C consonant (inherent a), C0 consonant
# without inherent vowel, M vowel sign, H virama, N nasal/visarga, D digit, X drop/space
_T: dict[int, tuple[str, str]] = {}


def _add(kind: str, pairs: dict[int, str]):
    for off, lat in pairs.items():
        _T[off] = (kind, lat)


_add("N", {0x01: "\x01", 0x02: "\x01", 0x03: "h"})   # \x01 = anusvara, resolved below
_add("V", {0x04: "a", 0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u", 0x0B: "ri",
           0x0C: "li", 0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o", 0x13: "o",
           0x14: "au", 0x50: "om", 0x60: "ri", 0x61: "li"})
_add("C", {0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n", 0x1A: "ch", 0x1B: "chh", 0x1C: "j",
           0x1D: "jh", 0x1E: "n", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n", 0x24: "t",
           0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n", 0x2A: "p", 0x2B: "f", 0x2C: "b",
           0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "zh",
           0x35: "v", 0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h",
           0x58: "q", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "r", 0x5D: "rh", 0x5E: "f", 0x5F: "y"})
_add("M", {0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri", 0x44: "ri", 0x45: "e",
           0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o", 0x4C: "au", 0x4E: "e",
           0x4F: "au", 0x55: "", 0x56: "ai", 0x57: "au", 0x62: "li", 0x63: "li"})
_add("H", {0x4D: ""})
_add("X", {0x00: "", 0x3A: "", 0x3B: "", 0x3C: "", 0x3D: "", 0x51: "", 0x52: "", 0x53: "", 0x54: "",
           0x64: " ", 0x65: " ", 0x70: " "})
for _d in range(10):
    _T[0x66 + _d] = ("D", str(_d))

_BLOCK = {0x0900: "deva", 0x0980: "beng", 0x0A00: "guru", 0x0A80: "gujr", 0x0B00: "orya",
          0x0B80: "taml", 0x0C00: "telu", 0x0C80: "knda", 0x0D00: "mlym"}
_OVR = {
    ("beng", 0x4E): ("C0", "t"),                      # khanda ta
    ("beng", 0x70): ("C", "r"), ("beng", 0x71): ("C", "v"),
    ("guru", 0x70): ("N", "n"), ("guru", 0x71): ("X", ""),   # tippi, addak
    ("guru", 0x72): ("V", ""), ("guru", 0x73): ("V", ""),    # vowel bearers: the matra supplies the vowel
    ("guru", 0x74): ("X", ""), ("guru", 0x75): ("X", ""),
    ("orya", 0x71): ("C", "v"),
    ("mlym", 0x4E): ("X", ""),                        # dot reph
    ("mlym", 0x7A): ("C0", "n"), ("mlym", 0x7B): ("C0", "n"), ("mlym", 0x7C): ("C0", "r"),
    ("mlym", 0x7D): ("C0", "l"), ("mlym", 0x7E): ("C0", "l"), ("mlym", 0x7F): ("C0", "k"),
    ("mlym", 0x54): ("C0", "m"), ("mlym", 0x55): ("C0", "y"), ("mlym", 0x56): ("C0", "zh"),
}
_ZW = {0x200C, 0x200D, 0x00AD}


def translit(s: str) -> str:
    """Indic script -> rough Latin; everything else passes through unchanged."""
    if not s or not INDIC_RE.search(s):
        return s
    out: list[str] = []
    pending = False  # last consonant still carries its inherent vowel
    for ch in s:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x0D7F:
            base = cp & ~0x7F
            off = cp - base
            kind, lat = _OVR.get((_BLOCK[base], off)) or _T.get(off, ("X", ""))
        elif cp in _ZW:
            continue
        else:
            pending = False          # word end: final schwa is not pronounced
            out.append(ch)
            continue
        if kind == "C" or kind == "C0":
            if pending:
                out.append("a")
            out.append(lat)
            pending = kind == "C"
        elif kind == "M":
            out.append(lat)
            pending = False
        elif kind == "H":
            pending = False
        elif kind == "V":
            if pending:
                out.append("a")
            out.append(lat)
            pending = False
        elif kind == "N":
            if pending:
                out.append("a")
            pending = False
            out.append(lat)
        elif kind == "D":
            pending = False
            out.append(lat)
        elif lat == " ":
            pending = False
            out.append(" ")
    res = "".join(out)
    res = re.sub("\x01(?=[pbm]|\\W|$)", "m", res).replace("\x01", "n")   # anusvara: m before labials / word end
    if re.search(r"[\u0D00-\u0D7F]", s):
        res = res.replace("rr", "tt")                  # Malayalam retroflex double: "limirrad" -> "limittad"
    return res


_SKEL_SUBS = [("chh", "k"), ("ch", "k"), ("ck", "k"), ("ph", "f"), ("gh", "g"), ("kh", "k"), ("bh", "b"),
              ("dh", "d"), ("th", "t"), ("jh", "j"), ("sh", "s"), ("zh", "l")]


def skeleton(word: str) -> str:
    """Consonant-class skeleton of a Latin word, symmetric for English and transliterated spellings."""
    t = re.sub(r"[^a-z]", "", word.lower())
    if not t:
        return ""
    t = t.replace("ght", "t")                   # silent gh: "bright" / "brait"
    for a, b in _SKEL_SUBS:
        t = t.replace(a, b)
    t = re.sub(r"ti(?=[aou])", "s", t)          # -tion / -tial sound "sh" ("intaraneshanal")
    t = re.sub(r"c(?=[eiy])", "s", t).replace("c", "k")
    t = re.sub(r"g(?=[eiy])", "j", t)
    t = re.sub(r"w(?=[^aeiou]|$)", "", t)        # silent w: "new" / "nyu"
    t = t.replace("q", "k").replace("x", "ks").replace("z", "s").replace("w", "b").replace("v", "b")  # v/w/b: Bengali has no v
    t = re.sub(r"j$", "s", t)                   # transliterated English plural: "-ij" = "-ies"
    t = re.sub(r"[aeiouyh]", "", t)
    return re.sub(r"(.)\1+", r"\1", t)


if __name__ == "__main__":
    for x in ["राम मार्केटिंग प्राइवेट लिमिटेड", "यूनाइटेड श्री इंफ्रा प्राइवेट लिमिटेड", "जैन क्लासिक वेंचर्स प्राइवेट लिमिटेड",
              "डायनामिक कंसल्टेंसी", "ईस्ट टेक लिमिटेड", "ബോംബെ ഗ്ലോബൽ ലിമിറ്റഡ്", "આદિત્ય સર્વિસીસ પ્રા. લિ.",
              "ஷ்யாம் இண்டஸ்ட்ரீஸ் எல்எல்பி", "ಬಿಗ್ ಕನ್‌ಸ್ಟ್ರಕ್ಷನ್ಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್", "స్మార్ట్ ఆల్ఫా ఫుడ్ ఎల్‌ఎల్‌పీ",
              "महाराष्ट्र", "उत्तर प्रदेश", "తెలంగాణ", "தமிழ்நாடு", "ગુજરાત", "दिल्ली", "कंपनी", "टेक्नोलॉजीज"]:
        t = translit(x)
        print(f"{x} -> {t} -> {' '.join(skeleton(w) for w in t.split())}")
    for w in ["consultancy", "technologies", "services", "enterprises", "private", "united", "maharashtra", "company"]:
        print(w, skeleton(w))
