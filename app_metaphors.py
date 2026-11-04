"""
WAIC Metaphor Tracer — research-specific variant of Sinocode Discourse Analyzer.

Purpose: trace the metaphors of Xi Jinping's WAIC 2026 speech (long forms) across
a corpus of earlier Chinese-language Xi speeches: when each metaphor first appears,
how often, in which documents, and in what context.

Kept from app.py (same logic): CSV dictionary format, file reading, year parsing,
counting modes (substring = longest non-overlapping matches; jieba precise/search,
THULAC, HanLP token / n-gram matching).
Removed: Set 2 / compare, Economist & Tianming watchlists, share-based tables,
events explanations, relative importance, discourse shift detection.
Added: first-appearance table, KWIC with annotation, corrected Per 10k,
mode comparison, CSV export.
"""

import html as html_module
import io
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

try:
    from docx import Document  # python-docx
    DOCX_AVAILABLE = True
except Exception:
    DOCX_AVAILABLE = False

try:
    import jieba
    JIEBA_AVAILABLE = True
except Exception:
    JIEBA_AVAILABLE = False

try:
    import thulac
    THULAC_AVAILABLE = True
except Exception:
    THULAC_AVAILABLE = False

try:
    import hanlp
    HANLP_AVAILABLE = True
except Exception:
    HANLP_AVAILABLE = False

APP_VERSION = "M2.1"
DEFAULT_TERMS_PATH = "WAIC2026_metaphor_terms_long.csv"

# Numatytasis korpusas: visi xi_speeches_YYYY aplankai iš discourse-analyzer-docs (main)
CORPUS_REPO_ZIP = "https://codeload.github.com/iconally/discourse-analyzer-docs/zip/refs/heads/main"
CORPUS_FILE_RE = re.compile(r"(?:^|/)xi_speeches_(\d{4})/([^/]+\.txt)$")
# Pati WAIC 2026 kalba (žodyno šaltinis): jhsjk article_id
WAIC_SOURCE_ID = "40763179"
# WAIC 2026 kalbos data: „nauja WAIC“ skaičiuojama tik pagal kalbas IKI šios datos
WAIC_DATE = "2026-07-17"
# Several versions of one speech in the database (news report, full text, People's Daily print, Qiushi):
# list of duplicate versions in the docs repository root
DEDUP_FILE = "duplicate_versions.csv"

MODE_LABELS = {
    "hybrid": "hybrid (for metaphors): normalized substring + split forms",
    "substring": "substring (longest non-overlapping matches)",
    "jieba_precise": "jieba (precise)",
    "jieba_search": "jieba (search)",
    "jieba_ngram": "jieba (precise) + n-grams (1–8 tokens)",
    "thulac": "THULAC",
    "hanlp": "HanLP",
}
MODE_AVAILABLE = {
    "hybrid": True,
    "substring": True,
    "jieba_precise": JIEBA_AVAILABLE,
    "jieba_search": JIEBA_AVAILABLE,
    "jieba_ngram": JIEBA_AVAILABLE,
    "thulac": THULAC_AVAILABLE,
    "hanlp": HANLP_AVAILABLE,
}

MODE_SHORT = {
    "hybrid": "Hybrid",
    "substring": "Substring",
    "jieba_precise": "jieba precise",
    "jieba_search": "jieba search",
    "jieba_ngram": "jieba n-gram",
    "thulac": "THULAC",
    "hanlp": "HanLP",
}

# Trumpi algoritmų aprašymai (rodomi skirtuke „Segmentavimas“)
ALGO_INFO = {
    "hybrid": (
        "**Hybrid (for metaphors).** No segmentation. (1) Text and terms are normalized: quotation marks, brackets "
        "and spaces are removed, “、，,；;” are unified. (2) Longest non-overlapping matches are searched. "
        "(3) If a term consists of several clauses (e.g. 各美其美、美美与共), a sentence containing all clauses "
        "(e.g. 各美其美…美人之美，美美与共) is counted as a **split form**. "
        "Single-character terms are matched as jieba precise tokens."
    ),
    "substring": (
        "**Substring.** No segmentation. A term is searched as an exact character sequence; longer terms take priority "
        "and matches do not overlap (AB is not counted inside ABCD)."
    ),
    "jieba_precise": (
        "**jieba precise.** The text is split into words (dictionary + HMM model). A term is counted only when "
        "it **equals a single token**, so long phrases that jieba splits are not found."
    ),
    "jieba_search": (
        "**jieba search.** Like precise, but long words are additionally split into shorter sub-words (search-engine mode). "
        "A term must equal one of the tokens."
    ),
    "jieba_ngram": (
        "**jieba n-gram.** Control mode: the same jieba precise tokens, but matched with the THULAC/HanLP rule – "
        "a term is found if it equals a **sequence of 1–8 adjacent tokens**. Shows whether differences come from the "
        "segmenter or from the matching rule."
    ),
    "thulac": (
        "**THULAC** (Tsinghua). Statistical segmenter. A term is found if it equals a **sequence of 1–8 adjacent tokens**, "
        "so phrases are found when their boundaries coincide with token boundaries."
    ),
    "hanlp": (
        "**HanLP** (neural, ELECTRA). A term is found if it equals a **sequence of 1–8 adjacent tokens**; "
        "segmentation depends on context."
    ),
}

HIGHLIGHT_PALETTE = [
    "#ffcccc", "#ccffcc", "#ccccff", "#ffffcc", "#ffccff", "#ccffff",
    "#ffddbb", "#ddffbb", "#bbddff", "#ffbbdd", "#ddbbff", "#bbffdd",
    "#e6ccff", "#cce6ff", "#ffe6cc",
]

st.set_page_config(page_title="WAIC Metaphor Tracer", layout="wide")


# -----------------------------
# Helpers (same logic as app.py)
# -----------------------------
def safe_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    return str(x)


def normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def read_txt(file_bytes: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "gb18030", "big5", "latin-1"):
        try:
            return file_bytes.decode(enc)
        except Exception:
            continue
    return file_bytes.decode("utf-8", errors="ignore")


def read_docx(file_bytes: bytes) -> str:
    if not DOCX_AVAILABLE:
        raise RuntimeError("python-docx is not available in this environment.")
    doc = Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs if p.text)


def file_bytes_to_text(name: str, data: bytes) -> Optional[str]:
    lower = name.lower()
    if lower.endswith(".txt"):
        return normalize_text(read_txt(data))
    if lower.endswith(".docx"):
        try:
            return normalize_text(read_docx(data))
        except Exception:
            return None
    return None


@st.cache_data(show_spinner="Downloading the Xi speech corpus from GitHub…", ttl=24 * 3600)
def load_github_corpus(url: str = CORPUS_REPO_ZIP) -> List[Tuple[str, bytes]]:
    """Parsiunčia discourse-analyzer-docs ZIP ir grąžina visus xi_speeches_YYYY/*.txt (vardas, baitai)."""
    import urllib.request
    with urllib.request.urlopen(url, timeout=180) as r:
        data = r.read()
    out: List[Tuple[str, bytes]] = []
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            name = info.filename
            if not (info.flag_bits & 0x800):  # ZIP be UTF-8 žymos: zipfile iškoduoja cp437
                try:
                    name = name.encode("cp437").decode("utf-8")
                except Exception:
                    pass
            m = CORPUS_FILE_RE.search(name)
            if m:
                out.append((m.group(2), z.read(info)))
            elif name.endswith("/" + DEDUP_FILE) or name == DEDUP_FILE:
                out.append(("__dedup__", z.read(info)))
    return sorted(out)


_BULLETS = re.compile(r"^[■●◆▲★☆□○◇\s]+")


def drop_repeated_lines(text: str) -> Tuple[str, int]:
    """Remove paragraphs repeated inside one document (pull quotes, repeated headings, photo captions).
    A paragraph of 15+ characters is dropped if it also occurs in an earlier paragraph or inside a longer one."""
    lines = text.split("\n")
    cores = [_BULLETS.sub("", l.strip()) for l in lines]
    keep, removed = [], 0
    for i, (l, c) in enumerate(zip(lines, cores)):
        if len(c) >= 15 and any(
            j != i and c in m and (len(m) > len(c) or j < i) for j, m in enumerate(cores)
        ):
            removed += 1
            continue
        keep.append(l)
    return "\n".join(keep), removed


def parse_dedup(raw: bytes) -> Tuple[set, Dict[str, str], set]:
    """duplicate_versions.csv -> (IDs to drop, {kept ID: date of the group's first version}, compilation IDs)."""
    df = pd.read_csv(io.BytesIO(raw), dtype=str, encoding="utf-8-sig").fillna("")
    drop = set(df.loc[df["action"] == "drop", "article_id"])
    keep_date = dict(zip(df.loc[df["action"] == "keep", "article_id"], df.loc[df["action"] == "keep", "group_first_date"]))
    comp = set(df.loc[df["compilation"] == "yes", "article_id"]) if "compilation" in df.columns else set()
    return drop, keep_date, comp


_SENT = re.compile(r"[^。！？；!?;\n]+[。！？；!?;]?")


def _sent_key(s: str) -> str:
    return re.sub(r"[\s“”‘’\"'《》〈〉（）()■●◆▲★☆□○◇—…]", "", s)


def trim_compilations(docs: list, comp_ids: set, id_of) -> int:
    """In compilation documents (e.g. Qiushi articles assembled from earlier speeches), remove sentences
    that already appeared in an earlier document or earlier in the same compilation. Other documents are not changed. Returns removed sentence count."""
    seen, removed = set(), 0
    for d in sorted(docs, key=lambda x: x["date"]):
        sents = _SENT.findall(d["text"])
        keys = [_sent_key(x) for x in sents]
        if id_of(d) in comp_ids:
            kept = []
            for x, k in zip(sents, keys):
                if len(k) >= 8 and k in seen:
                    removed += 1
                    continue
                kept.append(x)
                if len(k) >= 8:
                    seen.add(k)  # also removes sentences repeated inside the same compilation
            d["text"] = "".join(kept)
        else:
            seen.update(k for k in keys if len(k) >= 8)
    return removed


def load_terms(raw: bytes) -> pd.DataFrame:
    """Expected columns (semicolon separated): concept;term;pinyin;translation;category"""
    df = pd.read_csv(io.StringIO(read_txt(raw)), sep=";", dtype=str, keep_default_na=False)
    df.columns = [c.strip().lower().rstrip(",") for c in df.columns]
    required = ["concept", "term", "pinyin", "translation", "category"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dictionary is missing columns: {missing}. Required: concept;term;pinyin;translation;category")
    for c in required:
        df[c] = df[c].astype(str).map(lambda s: s.strip().rstrip(","))
    df = df[df["term"].map(len) > 0]
    df = df.drop_duplicates(subset=["term"]).reset_index(drop=True)
    return df[required]


def parse_date_from_filename(filename: str) -> Tuple[str, str, str]:
    """(year, month, day) from names like 2017-01-03__....txt, 2017_01_03, 20170103, 2017-01, 2017."""
    stem = re.sub(r"\.[^.]+$", "", safe_str(filename)).strip()
    m = re.search(r"(?<!\d)((?:19|20)\d{2})[-_\.](0[1-9]|1[0-2])[-_\.](0[1-9]|[12]\d|3[01])(?!\d)", stem)
    if m:
        return m.group(1), m.group(2), m.group(3)
    m = re.search(r"(?<!\d)((?:19|20)\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?!\d)", stem)
    if m:
        return m.group(1), m.group(2), m.group(3)
    m = re.search(r"(?<!\d)((?:19|20)\d{2})[-_\.](0[1-9]|1[0-2])(?!\d)", stem)
    if m:
        return m.group(1), m.group(2), ""
    m = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", stem)
    if m:
        return m.group(1), "", ""
    return "", "", ""


def longest_match_positions(text: str, terms: List[str]) -> List[Tuple[int, int, str]]:
    """Longest-first, non-overlapping matches (same strategy as app.py substring mode)."""
    n = len(text)
    occupied = bytearray(n)
    out: List[Tuple[int, int, str]] = []
    for term in sorted(set(terms), key=len, reverse=True):
        if not term:
            continue
        for m in re.finditer(re.escape(term), text):
            s, e = m.start(), m.end()
            if any(occupied[s:e]):
                continue
            out.append((s, e, term))
            occupied[s:e] = b"\x01" * (e - s)
    out.sort()
    return out


@st.cache_resource(show_spinner=False)
def get_hanlp_tokenizer():
    if not HANLP_AVAILABLE:
        return None
    try:
        return hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def get_thulac_tokenizer():
    if not THULAC_AVAILABLE:
        return None
    try:
        return thulac.thulac(seg_only=True)
    except Exception:
        return None


def tokenize(text: str, mode: str) -> Optional[List[str]]:
    if mode in ("jieba_precise", "jieba_ngram") and JIEBA_AVAILABLE:
        return [t for t in (safe_str(x).strip() for x in jieba.cut(text, cut_all=False)) if t]
    if mode == "jieba_search" and JIEBA_AVAILABLE:
        return [t for t in (safe_str(x).strip() for x in jieba.cut_for_search(text)) if t]
    if mode == "thulac":
        tok = get_thulac_tokenizer()
        if tok is None:
            return None
        out = []
        for it in tok.cut(text, text=False):
            s = safe_str(it[0] if isinstance(it, (list, tuple)) and it else it).strip()
            if s:
                out.append(s)
        return out
    if mode == "hanlp":
        tok = get_hanlp_tokenizer()
        if tok is None:
            return None
        try:
            res = tok([text])
            if isinstance(res, dict):
                res = next(iter(res.values()), [])
            flat: List[str] = []
            for item in res:
                if isinstance(item, (list, tuple)):
                    flat.extend(safe_str(t).strip() for t in item if safe_str(t).strip())
                elif safe_str(item).strip():
                    flat.append(safe_str(item).strip())
            return flat
        except Exception:
            return None
    return None


def ngram_counter(tokens: List[str], max_n: int = 8) -> Dict[str, int]:
    """Same as app.py: concatenated 1..max_n-grams, so terms spanning several tokens are counted."""
    c: Dict[str, int] = defaultdict(int)
    n = len(tokens)
    for k in range(1, min(max_n + 1, n + 1)):
        for i in range(n - k + 1):
            c["".join(tokens[i:i + k])] += 1
    return c


# -----------------------------
# Hybrid (metaforoms)
# -----------------------------
NORM_DROP = set(" \t　 “”\"‘’'《》〈〉「」『』（）()【】[]")
NORM_SEP = set("、，,；;")
SENT_SPLIT_RE = re.compile(r"[^。！？!?\n]+[。！？!?]*")


def normalize_with_map(text: str) -> Tuple[str, List[int]]:
    """Normalized text + map normalized index -> original index."""
    out, idx = [], []
    for i, ch in enumerate(text):
        if ch in NORM_DROP:
            continue
        out.append("，" if ch in NORM_SEP else ch)
        idx.append(i)
    return "".join(out), idx


def normalize_term(term: str) -> str:
    return normalize_with_map(term)[0]


def term_clauses(term: str) -> List[str]:
    parts = [p for p in normalize_term(term).split("，") if len(p) >= 2]
    return parts if len(parts) >= 2 else []


def jieba_token_spans(text: str) -> List[Tuple[int, int, str]]:
    spans, pos = [], 0
    if not JIEBA_AVAILABLE:
        return spans
    for tok in jieba.cut(text, cut_all=False):
        spans.append((pos, pos + len(tok), tok))
        pos += len(tok)
    return spans


def hybrid_matches(text: str, terms: List[str]) -> List[dict]:
    """
    Records: {s, e, term, kind, parts}
      kind = 'exact'   – normalized longest non-overlapping match (2+ chars)
             'variant' – all clauses of a multi-clause term in one sentence (no exact match there)
             'token'   – 1-char term matched as a jieba precise token
    """
    out: List[dict] = []
    norm, nmap = normalize_with_map(text)
    occupied = bytearray(len(norm))
    multi = [t for t in set(terms) if len(normalize_term(t)) >= 2]
    for term in sorted(multi, key=lambda t: len(normalize_term(t)), reverse=True):
        nt = normalize_term(term)
        start = 0
        while True:
            i = norm.find(nt, start)
            if i < 0:
                break
            j = i + len(nt)
            if not any(occupied[i:j]):
                occupied[i:j] = b"\x01" * (j - i)
                s0, e0 = nmap[i], nmap[j - 1] + 1
                out.append({"s": s0, "e": e0, "term": term, "kind": "exact", "parts": [(s0, e0)]})
                start = j
            else:
                start = i + 1
    exact_terms_by_sent = defaultdict(set)
    sents = [(m.start(), m.end()) for m in SENT_SPLIT_RE.finditer(text)]
    for r in out:
        for k, (a, b) in enumerate(sents):
            if a <= r["s"] < b:
                exact_terms_by_sent[k].add(r["term"])
                break
    for term in set(terms):
        cl = term_clauses(term)
        if not cl:
            continue
        for k, (a, b) in enumerate(sents):
            if term in exact_terms_by_sent[k]:
                continue
            ns, smap = normalize_with_map(text[a:b])
            pos_list = []
            for c in cl:
                i = ns.find(c)
                if i < 0:
                    break
                pos_list.append((a + smap[i], a + smap[i + len(c) - 1] + 1))
            if len(pos_list) == len(cl):
                out.append({"s": min(p[0] for p in pos_list), "e": max(p[1] for p in pos_list),
                            "term": term, "kind": "variant", "parts": sorted(pos_list)})
    singles = {t for t in terms if len(normalize_term(t)) == 1}
    if singles:
        if JIEBA_AVAILABLE:
            for s0, e0, tok in jieba_token_spans(text):
                if tok in singles and not any(r["s"] <= s0 < r["e"] for r in out if r["kind"] == "exact"):
                    out.append({"s": s0, "e": e0, "term": tok, "kind": "token", "parts": [(s0, e0)]})
        else:
            for s0, e0, t in longest_match_positions(text, list(singles)):
                out.append({"s": s0, "e": e0, "term": t, "kind": "token", "parts": [(s0, e0)]})
    out.sort(key=lambda r: (r["s"], r["e"]))
    return out


@st.cache_data(show_spinner=False)
def hybrid_counts(text: str, terms: Tuple[str, ...]) -> Tuple[Dict[str, int], Dict[str, int]]:
    total: Dict[str, int] = defaultdict(int)
    variant: Dict[str, int] = defaultdict(int)
    for r in hybrid_matches(text, list(terms)):
        total[r["term"]] += 1
        if r["kind"] == "variant":
            variant[r["term"]] += 1
    return dict(total), dict(variant)


# -----------------------------
# Segmentation view (HTML)
# -----------------------------
SEP_HTML = '<span style="color:#b0b0b0;"> | </span>'
MISS_STYLE = "border-bottom:2px dashed #d62728;"
VARIANT_STYLE = "box-shadow:inset 0 0 0 1px #555;"


def token_spans(line: str, mode: str) -> Optional[List[Tuple[int, int, str]]]:
    """Žetonai su TIKSLIOMIS pozicijomis. jieba: jieba.tokenize (search režime žetonai persidengia)."""
    if mode in ("jieba_precise", "jieba_search", "jieba_ngram"):
        if not JIEBA_AVAILABLE:
            return None
        jm = "search" if mode == "jieba_search" else "default"
        return [(a, b, w) for w, a, b in jieba.tokenize(line, mode=jm) if w.strip()]
    toks = tokenize(line, mode)
    if toks is None:
        return None
    out, pos = [], 0
    for t in toks:
        i = line.find(t, pos)
        if i < 0:
            continue
        out.append((i, i + len(t), t))
        pos = i + len(t)
    return out


def analyze_line(line: str, mode: str, term_set: set, ref_terms: List[str]) -> dict:
    """
    Vienos eilutės analizė vaizdui. Grąžina:
      main_starts – pagrindinių žetonų pradžios; sub_starts – jieba search pažodžių ribos;
      char_term / char_kind – kiekvieno ženklo terminas ir atitikmens tipas;
      miss – ženklai, kur terminas yra (hibridinio režimo atskaita), bet šis režimas jo nerado;
      found – rasti terminai; main_tokens – (a, b) pagrindiniams žetonams.
    """
    L = len(line)
    char_term: List[Optional[str]] = [None] * L
    char_kind: List[str] = [""] * L
    ref_recs = hybrid_matches(line, ref_terms)
    ref_cover = bytearray(L)
    for r in ref_recs:
        for (x, y) in r["parts"]:
            ref_cover[x:y] = b"\x01" * (y - x)
    res = {"main_starts": set(), "sub_starts": set(), "char_term": char_term, "char_kind": char_kind,
           "miss": bytearray(L), "found": set(), "main_tokens": [(0, L)], "tokenized": False}

    def mark(x, y, term, kind):
        if any(char_term[x:y]):
            return
        res["found"].add(term)
        for k in range(x, y):
            char_term[k], char_kind[k] = term, kind

    if mode in ("substring", "hybrid"):
        recs = ref_recs if mode == "hybrid" else [
            {"s": x, "e": y, "term": t, "kind": "exact", "parts": [(x, y)]}
            for x, y, t in longest_match_positions(line, ref_terms)]
        for r in recs:
            for (x, y) in r["parts"]:
                mark(x, y, r["term"], r["kind"])
        return res

    spans = token_spans(line, mode)
    if spans is None:
        return res
    res["tokenized"] = True
    if mode == "jieba_search":
        main = [t for t in spans if not any(o[0] <= t[0] and t[1] <= o[1] and (o[1] - o[0]) > (t[1] - t[0]) for o in spans)]
        main_set = {(t[0], t[1]) for t in main}
        for x, y, _ in spans:
            if (x, y) not in main_set:
                res["sub_starts"].update({x, y})
        for x, y, w in sorted(spans, key=lambda t: t[1] - t[0], reverse=True):
            if w in term_set:
                mark(x, y, w, "exact")
    else:
        main = spans
        max_n = 8 if mode in ("thulac", "hanlp", "jieba_ngram") else 1
        k = 0
        while k < len(main):
            hit = 0
            for n in range(min(max_n, len(main) - k), 0, -1):
                if "".join(t[2] for t in main[k:k + n]) in term_set:
                    hit = n
                    break
            if hit:
                mark(main[k][0], main[k + hit - 1][1], "".join(t[2] for t in main[k:k + hit]), "exact")
                k += hit
            else:
                k += 1
    res["main_starts"] = {t[0] for t in main}
    res["sub_starts"] -= res["main_starts"]
    res["main_tokens"] = [(t[0], t[1]) for t in main]
    for k in range(L):
        if ref_cover[k] and char_term[k] is None:
            res["miss"][k] = 1
    return res


def _char_style(an: dict, k: int, color_of: Dict[str, str]) -> Tuple[str, str]:
    t = an["char_term"][k]
    st_, title = "", ""
    if t is not None:
        st_ += f"background-color:{color_of.get(t, HIGHLIGHT_PALETTE[0])};"
        if an["char_kind"][k] == "variant":
            st_ += VARIANT_STYLE
    elif an["miss"][k]:
        st_ += MISS_STYLE
    return st_, t or ""


def render_line(line: str, mode: str, term_set: set, tip: Dict[str, str], color_of: Dict[str, str],
                ref_terms: List[str]) -> Tuple[str, set]:
    """Laisvo srauto vaizdas (žetonai atskirti „|“). Grąžina (html, rasti terminai)."""
    if not line.strip():
        return "", set()
    an = analyze_line(line, mode, term_set, ref_terms)
    pieces = []
    for (x, y) in an["main_tokens"]:
        runs, k = [], x
        while k < y:
            st_, t = _char_style(an, k, color_of)
            j = k + 1
            while j < y and _char_style(an, j, color_of) == (st_, t) and j not in an["sub_starts"]:
                j += 1
            sub = "border-left:1px dotted #999;" if k in an["sub_starts"] and k > x else ""
            frag = html_module.escape(line[k:j])
            if st_ or sub:
                ttl = tip.get(t, "") if t else ("A dictionary term is here, but this mode did not find it" if an["miss"][k] else "")
                if t and an["char_kind"][k] == "variant":
                    ttl += " · split form"
                runs.append(f'<span style="{st_}{sub}padding:0 1px;" title="{html_module.escape(ttl)}">{frag}</span>')
            else:
                runs.append(frag)
            k = j
        pieces.append("".join(runs))
    joiner = SEP_HTML if an["tokenized"] else ""
    return joiner.join(pieces), an["found"]


def render_text_html(text: str, mode: str, terms: List[str], tip: Dict[str, str], color_of: Dict[str, str]) -> str:
    term_set = set(terms)
    return "<br>".join(render_line(ln, mode, term_set, tip, color_of, terms)[0] for ln in text.split("\n"))


CELL_BASE = ("display:inline-block;width:1.75em;text-align:center;box-sizing:border-box;"
             "line-height:1.9;border-left:2px solid transparent;")


def render_grid_row(line: str, an: dict, a: int, b: int, tip: Dict[str, str], color_of: Dict[str, str]) -> str:
    """Ženklai [a, b) – po vieną vienodo pločio langelį; ribos – linijos tarp langelių."""
    cells = []
    for k in range(a, b):
        st_ = CELL_BASE
        if an["tokenized"] and k in an["main_starts"] and k > 0:
            st_ += "border-left:2px solid #888;"
        elif k in an["sub_starts"]:
            st_ += "border-left:1px dotted #888;"
        cs, t = _char_style(an, k, color_of)
        st_ += cs
        ttl = tip.get(t, "") if t else ("not found by this mode" if an["miss"][k] else "")
        cells.append(f'<span style="{st_}" title="{html_module.escape(ttl)}">{html_module.escape(line[k])}</span>')
    return "".join(cells)


def grid_blocks_html(line: str, analyses: Dict[str, dict], modes: List[str], blocks: List[Tuple[int, int]],
                     per_row: int, tip: Dict[str, str], color_of: Dict[str, str]) -> str:
    label_css = "display:inline-block;width:8.5em;font-size:0.85em;color:#444;vertical-align:middle;"
    html_parts = []
    for (sa, sb) in blocks:
        group = []
        for ca in range(sa, sb, int(per_row)):
            cb = min(sb, ca + int(per_row))
            rows = "".join(
                f'<div style="white-space:nowrap;"><span style="{label_css}">{MODE_SHORT[m_]}</span>'
                f'{render_grid_row(line, analyses[m_], ca, cb, tip, color_of)}</div>'
                for m_ in modes)
            group.append(f'<div style="margin-bottom:0.7rem;">{rows}</div>')
        html_parts.append('<div style="padding:0.5rem 0;border-bottom:1px solid #e6e6e6;overflow-x:auto;">'
                          + "".join(group) + "</div>")
    return "".join(html_parts)


def sentence_spans(line: str) -> List[Tuple[int, int]]:
    return [(m_.start(), m_.end()) for m_ in SENT_SPLIT_RE.finditer(line) if line[m_.start():m_.end()].strip()]


@st.cache_data(show_spinner=False)
def count_doc(text: str, terms: Tuple[str, ...], mode: str) -> Dict[str, int]:
    """term -> count for one document, following app.py's analyze_text logic per mode."""
    if mode == "hybrid":
        return hybrid_counts(text, terms)[0]
    if mode == "substring":
        c: Dict[str, int] = defaultdict(int)
        for _, _, t in longest_match_positions(text, list(terms)):
            c[t] += 1
        return dict(c)
    tokens = tokenize(text, mode)
    if tokens is None:  # segmenter unavailable -> fallback to substring (as in app.py)
        return count_doc(text, terms, "substring")
    if mode in ("thulac", "hanlp", "jieba_ngram"):
        counter = ngram_counter(tokens)
    else:  # jieba: exact token match (as in app.py)
        counter = defaultdict(int)
        for t in tokens:
            counter[t] += 1
    return {t: int(counter.get(t, 0)) for t in terms if counter.get(t, 0)}


def per10k(count: float, chars: float) -> float:
    return (count / chars) * 10000.0 if chars > 0 else 0.0


# -----------------------------
# Sidebar: inputs
# -----------------------------
st.title("WAIC Metaphor Tracer")
st.caption(f"Sinocode Discourse Analyzer · research version {APP_VERSION} · tracing WAIC 2026 metaphors in a corpus of Xi Jinping's speeches")

with st.sidebar:
    st.header("1. Dictionary")
    terms_upload = st.file_uploader("Metaphor CSV (concept;term;pinyin;translation;category)", type=["csv"])

    st.header("2. Corpus")
    use_github = st.checkbox(
        "Xi speeches from GitHub (xi_speeches_2013–2026)",
        value=True,
        help="All .txt files from the xi_speeches_YYYY folders of iconally/discourse-analyzer-docs. Downloaded once and cached for 24 h.",
    )
    merge_dupes = st.checkbox(
        "Merge duplicate versions of a speech",
        value=True,
        help="The database often stores one speech several times (news report, full text, People's Daily print version, Qiushi reprint). "
             "Keeps one version per speech (full text preferred) and dates it by the group's first version. List: duplicate_versions.csv in discourse-analyzer-docs. "
             "Also removes paragraphs repeated inside one document (pull quotes, captions), and in compilations "
             "(articles assembled from earlier speeches) removes the sentences already published earlier.",
    )
    exclude_source = st.checkbox(
        "Exclude the WAIC 2026 speech itself",
        value=True,
        help=f"The dictionary was compiled from this speech (article_id {WAIC_SOURCE_ID}), so including it would distort first-appearance results.",
    )
    if use_github and st.button("Refresh corpus from GitHub"):
        load_github_corpus.clear()
    files = st.file_uploader("Additional documents (TXT / DOCX)", type=["txt", "docx"], accept_multiple_files=True)
    folder = st.text_input(
        "Or a folder path (when running locally)",
        placeholder="/Users/.../xi_speeches",
        help="All .txt and .docx files in this folder and its subfolders.",
    )

    st.header("3. Matching")
    avail_modes = [m for m in MODE_LABELS if MODE_AVAILABLE[m]]
    mode = st.radio("Main mode", avail_modes, format_func=lambda m: MODE_LABELS[m])
    compare_modes = st.checkbox(
        "Compute and compare all modes",
        value=False,
        help="Slower (especially HanLP). Results appear in the “Mode comparison” tab.",
    )
    unavailable = [MODE_LABELS[m] for m in MODE_LABELS if not MODE_AVAILABLE[m]]
    if unavailable:
        st.caption("Not installed in this environment: " + ", ".join(unavailable))

    st.header("4. Context")
    kwic_window = st.slider("Characters before and after a match (±N)", 5, 60, 15)

# Load dictionary
try:
    if terms_upload is not None:
        terms_df = load_terms(terms_upload.getvalue())
    elif Path(DEFAULT_TERMS_PATH).exists():
        terms_df = load_terms(Path(DEFAULT_TERMS_PATH).read_bytes())
        st.sidebar.caption(f"Using the default dictionary: {DEFAULT_TERMS_PATH}")
    else:
        terms_df = None
except Exception as e:
    st.error(f"Could not read the dictionary: {e}")
    st.stop()

# Load documents
raw_docs: List[Tuple[str, bytes]] = []
dedup_raw = None
if use_github:
    try:
        gh_docs = load_github_corpus()
        dedup_raw = next((b for n, b in gh_docs if n == "__dedup__"), None)
        gh_docs = [(n, b) for n, b in gh_docs if n != "__dedup__"]
        raw_docs.extend(gh_docs)
        st.sidebar.caption(f"GitHub corpus: {len(gh_docs)} files")
    except Exception as e:
        st.sidebar.error(f"Could not download the corpus from GitHub: {e}")
if files:
    raw_docs.extend((f.name, f.getvalue()) for f in files)
if folder.strip():
    p = Path(folder.strip()).expanduser()
    if p.is_dir():
        for fp in sorted(p.rglob("*")):
            if fp.suffix.lower() in (".txt", ".docx") and fp.is_file():
                raw_docs.append((fp.name, fp.read_bytes()))
    else:
        st.sidebar.error("Folder not found.")

if terms_df is None or terms_df.empty or not raw_docs:
    st.info("Upload a metaphor dictionary (CSV) and corpus documents (or give a folder path) to start the analysis.")
    st.stop()

docs = []
skipped = []
seen_names = set()
for name, data in raw_docs:
    if name in seen_names:
        continue
    seen_names.add(name)
    text = file_bytes_to_text(name, data)
    y, mo, d = parse_date_from_filename(name)
    if not text or not y:
        skipped.append(name)
        continue
    docs.append({"filename": name, "year": int(y), "date": "-".join(x for x in (y, mo, d) if x), "text": text, "chars": len(text)})

if merge_dupes:
    if dedup_raw is None:
        st.sidebar.warning(f"{DEDUP_FILE} not found in the corpus repository: duplicate versions are not merged.")
    else:
        drop_ids, keep_date, comp_ids = parse_dedup(dedup_raw)
        before = len(docs)
        merged = []
        for d in docs:
            parts = d["filename"].split("__")
            aid = parts[1] if len(parts) > 2 else ""
            if aid in drop_ids:
                continue
            if aid in keep_date and keep_date[aid]:
                d["date"] = keep_date[aid]
                d["year"] = int(keep_date[aid][:4])
            merged.append(d)
        rep = 0
        for d in merged:
            d["text"], k = drop_repeated_lines(d["text"])
            d["chars"] = len(d["text"])
            rep += k
        _aid = lambda d: (d["filename"].split("__") + ["", ""])[1]
        trimmed = trim_compilations(merged, comp_ids, _aid)
        for d in merged:
            d["chars"] = len(d["text"])
        docs = merged
        st.sidebar.caption(f"Duplicate versions merged: {before - len(docs)} files removed, {len(docs)} speeches left; "
                           f"{rep} repeated paragraphs removed inside documents; {trimmed} reprinted sentences removed "
                           f"from {len(comp_ids)} compilations")

view_docs = list(docs)  # skirtukui „Segmentavimas“: visi dokumentai, įskaitant WAIC kalbą
if exclude_source:
    docs = [d for d in docs if f"__{WAIC_SOURCE_ID}__" not in d["filename"]]

if skipped:
    st.warning(f"Skipped {len(skipped)} files (no year in the file name or unreadable): " + ", ".join(skipped[:10]) + (" …" if len(skipped) > 10 else ""))
if not docs:
    st.stop()

all_years = sorted({d["year"] for d in docs})
with st.sidebar:
    st.header("5. Period")
    if len(all_years) > 1:
        yr = st.slider("Years", min(all_years), max(all_years), (min(all_years), max(all_years)))
    else:
        yr = (all_years[0], all_years[0])
docs = [d for d in docs if yr[0] <= d["year"] <= yr[1]]
years = sorted({d["year"] for d in docs})
terms = tuple(terms_df["term"].tolist())
meta = terms_df.set_index("term")


# -----------------------------
# Counting
# -----------------------------
def run_mode(m: str) -> pd.DataFrame:
    rows = []
    prog = st.progress(0.0, text=f"Counting: {MODE_LABELS[m]}")
    for i, d in enumerate(docs):
        variant = hybrid_counts(d["text"], terms)[1] if m == "hybrid" else {}
        for t, c in count_doc(d["text"], terms, m).items():
            rows.append({"filename": d["filename"], "year": d["year"], "date": d["date"], "term": t, "count": c,
                         "variant": int(variant.get(t, 0))})
        prog.progress((i + 1) / len(docs), text=f"Counting: {MODE_LABELS[m]} ({i + 1}/{len(docs)})")
    prog.empty()
    return pd.DataFrame(rows, columns=["filename", "year", "date", "term", "count", "variant"])


hits = run_mode(mode)
docs_df = pd.DataFrame([{k: d[k] for k in ("filename", "year", "date", "chars")} for d in docs])
year_docs = docs_df.groupby("year").size()
year_chars = docs_df.groupby("year")["chars"].sum()

# Term x year raw counts (all dictionary terms, zeros included)
term_year = (
    hits.pivot_table(index="term", columns="year", values="count", aggfunc="sum", fill_value=0)
    .reindex(index=list(terms), columns=years, fill_value=0)
    .astype(int)
)
term_year_docs = (
    hits.pivot_table(index="term", columns="year", values="filename", aggfunc="nunique", fill_value=0)
    .reindex(index=list(terms), columns=years, fill_value=0)
    .astype(int)
)

# First appearance
first_rows = []
for t in terms:
    h_all = hits[hits["term"] == t].sort_values(["date", "filename"])
    h = h_all[h_all["date"] < WAIC_DATE]
    h_post = h_all[h_all["date"] >= WAIC_DATE]
    r = meta.loc[t]
    row = {
        "CH term": t,
        "Pinyin": r["pinyin"],
        "ENG translation": r["translation"],
        "Concept": r["concept"],
        "Category": r["category"],
        "First year": int(h["year"].iloc[0]) if len(h) else None,
        "First date": h["date"].iloc[0] if len(h) else "",
        "First document": h["filename"].iloc[0] if len(h) else "",
        "Before WAIC": int(h["count"].sum()),
        "After WAIC": int(h_post["count"].sum()),
        "Total": int(h_all["count"].sum()),
        "Of which split form": int(h_all["variant"].sum()),
        "Documents": int(h["filename"].nunique()),
        "Active years": int((term_year.loc[t] > 0).sum()),
    }
    for y in years:
        row[str(y)] = int(term_year.loc[t, y])
    row["Status"] = "new in WAIC" if row["Before WAIC"] == 0 else "found before WAIC"
    row["After WAIC dates"] = ", ".join(sorted(h_post["date"].unique()))
    if mode != "hybrid":
        del row["Of which split form"]
    first_rows.append(row)
first_df = pd.DataFrame(first_rows)


def kwic_rows(selected_terms: List[str], window: int) -> pd.DataFrame:
    out = []
    sel = set(selected_terms)
    for d in docs:
        text = d["text"]
        if mode == "hybrid":
            matches = [(r["s"], r["e"], r["term"]) for r in hybrid_matches(text, list(terms))]
        else:
            matches = longest_match_positions(text, list(terms))
        for s, e, t in matches:
            if t not in sel:
                continue
            left = text[max(0, s - window):s].replace("\n", " ")
            right = text[e:e + window].replace("\n", " ")
            out.append({
                "Year": d["year"], "Date": d["date"], "Document": d["filename"],
                "Left": left, "Term": t, "Right": right,
                "Metaphor?": "", "Note": "",
            })
    return pd.DataFrame(out, columns=["Year", "Date", "Document", "Left", "Term", "Right", "Metaphor?", "Note"]).sort_values(["Date", "Document"]) if out else pd.DataFrame(columns=["Year", "Date", "Document", "Left", "Term", "Right", "Metaphor?", "Note"])


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, sep=";").encode("utf-8-sig")


# -----------------------------
# Tabs
# -----------------------------
seg_tab, *tabs = st.tabs(["Segmentation", "Summary", "First appearance", "Dynamics", "Year-pair changes", "Context (KWIC)", "Mode comparison", "Export"])


# 0. Segmentation (document view, styled as in app.py)
def _title_cn(filename: str) -> str:
    stem = re.sub(r"\.[^.]+$", "", filename)
    parts = stem.split("__", 2)
    return parts[2] if len(parts) == 3 else stem


with seg_tab:
    st.subheader("Document info")
    tip = {t: f"{meta.loc[t, 'pinyin']} — {meta.loc[t, 'translation']}" for t in terms}
    color_of = {t: HIGHLIGHT_PALETTE[i % len(HIGHLIGHT_PALETTE)] for i, t in enumerate(terms)}
    vdocs = sorted(view_docs, key=lambda d: (d["date"], d["filename"]))
    src = next((d for d in vdocs if f"__{WAIC_SOURCE_ID}__" in d["filename"]), vdocs[-1])
    v_years = sorted({d["year"] for d in vdocs})
    cy, cd = st.columns([1, 3])
    sel_year = cy.selectbox("Year", v_years, index=v_years.index(src["year"]), key="seg_year")
    ydocs = [d for d in vdocs if d["year"] == sel_year]
    day_count = defaultdict(int)
    for d in ydocs:
        day_count[d["date"]] += 1
    seen_day, labels = defaultdict(int), []
    for d in ydocs:
        seen_day[d["date"]] += 1
        suf = chr(ord("A") + seen_day[d["date"]] - 1) if day_count[d["date"]] > 1 else ""
        labels.append(f"{d['date']}{suf}")
    default_i = next((i for i, d in enumerate(ydocs) if d is src), 0)
    li = cd.selectbox("Choose a document", range(len(ydocs)), index=default_i,
                      format_func=lambda i: labels[i], key="seg_doc")
    doc = ydocs[li]
    text = doc["text"]

    col1, col2 = st.columns([1, 4])
    col1.write(f"**Year:** {doc['year']}")
    col2.write(f"**Title (CN):** {_title_cn(doc['filename'])}")
    if f"__{WAIC_SOURCE_ID}__" in doc["filename"] and exclude_source:
        st.caption("This is the WAIC 2026 speech (dictionary source): shown here but excluded from the corpus analysis.")

    with st.expander("Text statistics", expanded=False):
        st.write({"Characters": len(text), "Lines": text.count("\n") + 1 if text else 0})

    with st.expander("How the text is segmented", expanded=True):
        view_mode = st.radio("Mode", avail_modes, index=avail_modes.index(mode), horizontal=True,
                             format_func=lambda m: MODE_SHORT[m], key="seg_view_mode")
        st.markdown(ALGO_INFO[view_mode])
        st.caption("Token boundaries “|”; dictionary terms are highlighted (hover for pinyin and translation). "
                   "Red dashed underline – the term is in the text but this mode did not find it. "
                   "Thin frame – split form (hybrid mode).")
        body = render_text_html(text, view_mode, list(terms), tip, color_of)
        st.markdown(f'<div style="max-height:420px;overflow-y:auto;padding:0.5rem 0;line-height:1.9;">{body}</div>',
                    unsafe_allow_html=True)

    # --- Režimų palyginimo lentelė (rodoma skyriuje 4) ir šuolis į pastraipą pagal pasirinktą metaforą ---
    cmp = pd.DataFrame({MODE_SHORT[m_]: pd.Series(count_doc(text, terms, m_), dtype="int64")
                        for m_ in avail_modes}).reindex(list(terms)).fillna(0).astype(int)
    cmp = cmp[cmp.sum(axis=1) > 0]
    doc_lines = text.split("\n")

    def term_occurrences(term: str) -> List[Tuple[int, int, int]]:
        occ = []
        for li_, ln in enumerate(doc_lines):
            if not ln.strip():
                continue
            hits_ = [(r["s"], r["e"]) for r in hybrid_matches(ln, list(terms)) if r["term"] == term]
            if not hits_:
                i_ = ln.find(term)
                while i_ >= 0:
                    hits_.append((i_, i_ + len(term)))
                    i_ = ln.find(term, i_ + 1)
            occ.extend((li_, a_, b_) for a_, b_ in hits_)
        return occ

    cmp_key = f"seg_cmp_{doc['filename']}"
    sel_term = None
    _sel = st.session_state.get(cmp_key)
    try:
        _rows = _sel["selection"]["rows"] if _sel else []
    except Exception:
        _rows = []
    if _rows and _rows[0] < len(cmp):
        sel_term = cmp.index[_rows[0]]

    with st.expander("Same paragraph in all modes", expanded=False):
        doc_found = {m_: sum(1 for c in count_doc(text, terms, m_).values() if c > 0) for m_ in avail_modes}
        doc_ref = doc_found.get("hybrid", max(doc_found.values()) if doc_found else 0)
        st.markdown("**Whole document:** " + " · ".join(
            f"**{MODE_SHORT[m_]}** {doc_found[m_]}/{doc_ref}" for m_ in avail_modes))
        paras = [(i, ln) for i, ln in enumerate(text.split("\n")) if ln.strip()]
        ref_hits = {i: len(hybrid_matches(ln, list(terms))) for i, ln in paras}
        cand = [(i, ln) for i, ln in paras if ref_hits[i] > 0] or paras
        # Šuolis: kai lentelėje 4 pasirenkama nauja metafora, perjungiam pastraipą ir sakinį
        if sel_term is not None and st.session_state.get("seg_jump_applied") != (cmp_key, sel_term):
            occ0 = term_occurrences(sel_term)
            if occ0:
                li0, a0, _ = occ0[0]
                pos0 = next((k for k, (i, _) in enumerate(cand) if i == li0), None)
                if pos0 is not None:
                    st.session_state["seg_para"] = pos0
                    ss0 = sentence_spans(doc_lines[li0])
                    st.session_state["seg_sent"] = next((k + 1 for k, (x, y) in enumerate(ss0) if x <= a0 < y), 0)
            st.session_state["seg_jump_applied"] = (cmp_key, sel_term)
        pc1, pc2, pc3 = st.columns([3, 2, 1])
        pi = pc1.selectbox("Paragraph", range(len(cand)),
                           format_func=lambda k: f"¶{cand[k][0] + 1} · {ref_hits[cand[k][0]]} terms · {cand[k][1][:40]}…",
                           key="seg_para")
        line = cand[pi][1]
        sents = sentence_spans(line)
        if st.session_state.get("seg_sent", 0) > len(sents):
            st.session_state["seg_sent"] = 0
        si = pc2.selectbox("Sentence", range(len(sents) + 1),
                           format_func=lambda k: "whole paragraph" if k == 0 else f"{k}. {line[sents[k - 1][0]:sents[k - 1][1]][:18]}…",
                           key="seg_sent")
        per_row = pc3.number_input("Characters per row", 15, 60, 30, step=5, key="seg_per_row")
        analyses = {m_: analyze_line(line, m_, set(terms), list(terms)) for m_ in avail_modes}
        ref_n = len(analyses["hybrid"]["found"]) if "hybrid" in analyses else 0
        st.markdown("**This paragraph:** " + " · ".join(f"**{MODE_SHORT[m_]}** {len(analyses[m_]['found'])}/{ref_n}" for m_ in avail_modes))
        st.caption("Each character occupies a cell of equal width, so the same character sits in the same position in every mode. "
                   "Grey line – token boundary; dotted line – jieba search sub-word boundary; red dashed underline – term present but not found; "
                   "frame – split form.")
        blocks = sents if si == 0 else [sents[si - 1]]
        st.markdown(grid_blocks_html(line, analyses, avail_modes, blocks, int(per_row), tip, color_of),
                    unsafe_allow_html=True)

    st.divider()
    st.subheader("Analysis")
    st.caption(f"Mode: {MODE_LABELS[view_mode]}")
    dcounts = count_doc(text, terms, view_mode)
    dvariant = hybrid_counts(text, terms)[1] if view_mode == "hybrid" else {}
    td = pd.DataFrame([{
        "CH term": t, "Pinyin": meta.loc[t, "pinyin"], "ENG translation": meta.loc[t, "translation"],
        "Concept": meta.loc[t, "concept"], "Category": meta.loc[t, "category"], "Count": c,
        **({"Of which split form": int(dvariant.get(t, 0))} if view_mode == "hybrid" else {}),
    } for t, c in dcounts.items() if c > 0])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total matches", int(td["Count"].sum()) if not td.empty else 0)
    m2.metric("Unique terms", int(td["CH term"].nunique()) if not td.empty else 0)
    m3.metric("Unique concepts", int(td["Concept"].nunique()) if not td.empty else 0)
    m4.metric("Categories hit", int(td["Category"].nunique()) if not td.empty else 0)

    st.markdown("### 1) Term detail")
    if td.empty:
        st.info("No dictionary terms were found in this document.")
    else:
        tdv = td.sort_values("Count", ascending=False).reset_index(drop=True)
        tdv.index = range(1, len(tdv) + 1)
        st.dataframe(tdv, width="stretch")
        for lvl, key in (("2) Concept summary", "Concept"), ("3) Category summary", "Category")):
            st.markdown(f"### {lvl}")
            agg = (td.groupby(key).agg(**{"Total count": ("Count", "sum"), "Unique terms": ("CH term", "nunique")})
                   .sort_values("Total count", ascending=False).reset_index())
            agg.index = range(1, len(agg) + 1)
            st.dataframe(agg, width="stretch")

    st.markdown("### 4) Mode comparison in this document")
    if cmp.empty:
        st.info("No mode found any terms in this document.")
    else:
        st.caption("Tick the box on the left of a row: the sentence containing that metaphor appears below in all modes, "
                   "and its paragraph is selected in “Same paragraph in all modes”.")
        mode_cols = list(cmp.columns)
        tbl = cmp.copy()
        tbl.insert(0, "ENG translation", [meta.loc[t, "translation"] for t in tbl.index])
        tot = pd.DataFrame([["", *(cmp > 0).sum(axis=0).tolist()], ["", *cmp.sum(axis=0).tolist()]],
                           index=["Terms found", "Total matches"], columns=tbl.columns)
        tbl = pd.concat([tbl, tot])
        tbl.index.name = "CH term"

        def _cell(v):
            return "background-color:#d9f2d9;" if v > 0 else "background-color:#f8d7d7;"

        st.dataframe(tbl.style.map(_cell, subset=pd.IndexSlice[list(cmp.index), mode_cols]), width="stretch",
                     on_select="rerun", selection_mode="single-row", key=cmp_key)
        st.download_button("Download comparison (CSV)", csv_bytes(tbl.reset_index()),
                           f"{doc['date']}_mode_comparison.csv", "text/csv")

        if sel_term is not None:
            occ = term_occurrences(sel_term)
            st.markdown(f"#### {sel_term} — {meta.loc[sel_term, 'translation']}")
            if not occ:
                st.info("Could not locate this term in the text.")
            else:
                oi = 0
                if len(occ) > 1:
                    oi = st.selectbox("Occurrence", range(len(occ)),
                                      format_func=lambda k: f"{k + 1} of {len(occ)} · ¶{occ[k][0] + 1}",
                                      key=f"seg_occ_{cmp_key}_{sel_term}")
                li_, a_, b_ = occ[oi]
                ln = doc_lines[li_]
                ss = sentence_spans(ln)
                blk = [next(((x, y) for (x, y) in ss if x <= a_ < y), (0, len(ln)))]
                # jei terminas išeina už sakinio ribų (išskaidyta forma), rodom iki termino pabaigos
                blk = [(min(blk[0][0], a_), max(blk[0][1], b_))]
                an_ = {m_: analyze_line(ln, m_, set(terms), list(terms)) for m_ in avail_modes}
                st.caption(f"¶{li_ + 1} · sentence containing this metaphor")
                st.markdown(grid_blocks_html(ln, an_, avail_modes, blk,
                                             int(st.session_state.get("seg_per_row", 30)), tip, color_of),
                            unsafe_allow_html=True)

# 1. Summary
with tabs[0]:
    st.subheader("Corpus and match summary")
    st.caption(f"Mode: {MODE_LABELS[mode]} · {len(terms)} dictionary terms · {len(docs)} documents in corpus")
    summ = pd.DataFrame({
        "Documents": year_docs,
        "Characters": year_chars,
        "Matches": term_year.sum(axis=0),
        "Terms found": (term_year > 0).sum(axis=0),
    })
    summ["Matches per 10k"] = [round(per10k(summ.loc[y, "Matches"], summ.loc[y, "Characters"]), 3) for y in summ.index]
    summ.index = summ.index.astype(str)
    st.dataframe(summ, width="stretch", height=(len(summ) + 1) * 35 + 3)  # visos eilutės be slinkimo
    gaps = [y for y in range(min(years), max(years) + 1) if y not in years]
    if gaps:
        st.warning("Years missing from the corpus: " + ", ".join(map(str, gaps)))
    c1, c2, c3 = st.columns(3)
    c1.metric("Found before WAIC", int((first_df["Before WAIC"] > 0).sum()))
    c2.metric("New in WAIC", int((first_df["Before WAIC"] == 0).sum()))
    c3.metric("Dictionary terms", len(terms))

# 2. First appearance
with tabs[1]:
    st.subheader("When does a metaphor first appear in the corpus?")
    st.caption(f"Status and first year are computed only from speeches before WAIC ({WAIC_DATE}). "
               "“New in WAIC” – the metaphor was not found in any earlier speech in the selected period.")
    f1, f2, f3 = st.columns(3)
    status_f = f1.multiselect("Status", ["found before WAIC", "new in WAIC"], default=["found before WAIC", "new in WAIC"])
    concept_f = f2.multiselect("Concept", sorted(first_df["Concept"].unique()))
    cat_f = f3.multiselect("Category", sorted(first_df["Category"].unique()))
    v = first_df[first_df["Status"].isin(status_f)]
    if concept_f:
        v = v[v["Concept"].isin(concept_f)]
    if cat_f:
        v = v[v["Category"].isin(cat_f)]
    v = v.sort_values(["First year", "First date", "CH term"], na_position="last")
    st.dataframe(v, width="stretch", hide_index=True)
    st.download_button("Download CSV", csv_bytes(v), "first_appearance.csv", "text/csv")

    st.markdown("#### Are WAIC metaphors used after WAIC?")
    post_docs = docs_df[docs_df["date"] >= WAIC_DATE]
    st.caption(f"Speeches after WAIC in the corpus: {len(post_docs)}"
               + (f" ({post_docs['date'].min()} – {post_docs['date'].max()})" if len(post_docs) else "")
               + ". Rows with status “new in WAIC” matter most: first used in the WAIC speech and repeated later.")
    pw = first_df[first_df["After WAIC"] > 0][["CH term", "ENG translation", "Status", "Before WAIC", "After WAIC", "After WAIC dates"]]
    new_after = pw[pw["Status"] == "new in WAIC"]
    if len(post_docs) == 0:
        st.info("No speeches after WAIC in the selected period.")
    elif pw.empty:
        st.info("None of the WAIC speech metaphors was used after WAIC.")
    else:
        if new_after.empty:
            st.info(f"None of the “new in WAIC” metaphors was used after WAIC; only {len(pw)} earlier metaphors recur after WAIC.")
        else:
            st.success(f"{len(new_after)} “new in WAIC” metaphor(s) also used after WAIC.")
        st.dataframe(pw.sort_values(["Status", "After WAIC"], ascending=[False, False]), width="stretch", hide_index=True)

# 3. Dynamics
with tabs[2]:
    st.subheader("Dynamics by year")
    a, b = st.columns(2)
    level = a.radio("Level", ["Term", "Concept", "Category"], horizontal=True)
    metric = b.radio(
        "Metric",
        ["Per 10k characters", "Raw count", "Share of documents (%)"],
        horizontal=True,
        help="Per 10k = all matches in a year / all characters in that year × 10,000. "
             "Share of documents = % of that year's documents with at least one match.",
    )
    key_col = {"Term": "term", "Concept": "concept", "Category": "category"}[level]
    h = hits.merge(terms_df[["term", "concept", "category"]], on="term", how="left")
    if metric == "Share of documents (%)":
        piv = h.pivot_table(index=key_col, columns="year", values="filename", aggfunc="nunique", fill_value=0)
        piv = piv.reindex(columns=years, fill_value=0)
        piv = piv.div(year_docs.reindex(years).values, axis=1) * 100
    else:
        piv = h.pivot_table(index=key_col, columns="year", values="count", aggfunc="sum", fill_value=0).reindex(columns=years, fill_value=0)
        if metric == "Per 10k characters":
            piv = piv.div(year_chars.reindex(years).values, axis=1) * 10000
    piv = piv.round(3)
    if piv.empty:
        st.info("No matches in the selected period.")
    else:
        order = piv.sum(axis=1).sort_values(ascending=False).index.tolist()
        chosen = st.multiselect(f"Choose {level} (8–12 recommended)", order, default=order[:8])
        if chosen:
            chart = piv.loc[chosen].T
            chart.index = chart.index.astype(str)
            st.line_chart(chart)
        tbl = piv.loc[order].copy()
        tbl.columns = tbl.columns.astype(str)
        st.dataframe(tbl, width="stretch")
        st.download_button("Download table", csv_bytes(tbl.reset_index()), f"dynamics_{level}_{metric}.csv", "text/csv")

# 4. Year pairs
with tabs[3]:
    st.subheader("Term changes between consecutive years")
    if len(years) < 2:
        st.info("At least two years are needed.")
    else:
        pair_rows = []
        for y1, y2 in zip(years[:-1], years[1:]):
            for t in terms:
                c1, c2 = int(term_year.loc[t, y1]), int(term_year.loc[t, y2])
                if c1 == 0 and c2 == 0:
                    continue
                p1, p2 = per10k(c1, year_chars[y1]), per10k(c2, year_chars[y2])
                if c1 == 0:
                    status = "appeared"
                elif c2 == 0:
                    status = "disappeared"
                elif p2 > p1 * 1.1:
                    status = "↑"
                elif p2 < p1 * 0.9:
                    status = "↓"
                else:
                    status = "→"
                pair_rows.append({
                    "Pair": f"{y1}–{y2}", "CH term": t, "Concept": meta.loc[t, "concept"],
                    f"Count y1": c1, f"Count y2": c2,
                    "Per10k y1": round(p1, 3), "Per10k y2": round(p2, 3), "Change": status,
                })
        pairs_df = pd.DataFrame(pair_rows)
        if pairs_df.empty:
            st.info("No matches.")
        else:
            sel_pair = st.selectbox("Year pair", pairs_df["Pair"].unique().tolist())
            st.dataframe(pairs_df[pairs_df["Pair"] == sel_pair], width="stretch", hide_index=True)
            st.download_button("Download all pairs", csv_bytes(pairs_df), "year_pairs.csv", "text/csv")

# 5. KWIC
with tabs[4]:
    st.subheader("Context (KWIC)")
    st.caption(
        f"Each match with ±{kwic_window} characters of context (set in the sidebar). "
        "Positions follow the main mode (hybrid or substring longest match). "
        "You can annotate the “Metaphor?” and “Note” columns manually and download them."
    )
    found = first_df[first_df["Total"] > 0]["CH term"].tolist()
    if not found:
        st.info("No matches.")
    else:
        sel_terms = st.multiselect("Terms", found, default=found[:1])
        if sel_terms:
            kw = kwic_rows(sel_terms, kwic_window)
            st.caption(f"Matches: {len(kw)}")
            edited = st.data_editor(
                kw,
                width="stretch",
                hide_index=True,
                disabled=["Year", "Date", "Document", "Left", "Term", "Right"],
                column_config={"Metaphor?": st.column_config.SelectboxColumn(options=["", "yes", "no", "unclear"])},
                key="kwic_editor",
            )
            st.download_button("Download KWIC with annotations", csv_bytes(edited), "kwic.csv", "text/csv")

# 6. Mode comparison
STATUS_ORDER = {"new in WAIC": 0, "mode-dependent": 1, "earlier (all modes)": 2}


def _cjk_font_name() -> Optional[str]:
    try:
        from matplotlib import font_manager
        for f in font_manager.fontManager.ttflist:
            if any(k in f.name for k in ("Noto Sans CJK", "Noto Serif CJK", "WenQuanYi", "Droid Sans Fallback", "SimHei", "PingFang", "Heiti")):
                return f.name
    except Exception:
        pass
    return None


def heatmap_png(firsts_df: pd.DataFrame, labels: List[str], title: str) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    font = _cjk_font_name()
    if font:
        plt.rcParams["font.family"] = [font, "DejaVu Sans"]
    vals = firsts_df.astype(float).values
    n_rows, n_cols = vals.shape
    fig, ax = plt.subplots(figsize=(1.3 * n_cols + 4.2, 0.27 * n_rows + 1.2))
    cmap = plt.get_cmap("viridis_r").copy()
    cmap.set_bad("#e6e6e6")
    finite = vals[np.isfinite(vals)]
    vmin, vmax = (finite.min(), finite.max()) if finite.size else (0, 1)
    ax.imshow(np.ma.masked_invalid(vals), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    for i in range(n_rows):
        for j in range(n_cols):
            v = vals[i, j]
            txt = "—" if not np.isfinite(v) else str(int(v))
            col = "#777777" if not np.isfinite(v) else ("white" if (v - vmin) > 0.55 * (vmax - vmin + 1e-9) else "black")
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.5, color=col)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(list(firsts_df.columns), fontsize=8)
    ax.xaxis.tick_top()
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_title(title, loc="left", fontsize=9, pad=24)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200)
    plt.close(fig)
    return buf.getvalue()


with tabs[5]:
    st.subheader("Mode comparison")
    st.caption(
        f"The same terms are counted in every mode. First year and status use only speeches before WAIC ({WAIC_DATE}). "
        "jieba precise/search match a single token; jieba n-gram, THULAC and HanLP join up to 8 adjacent tokens; hybrid is normalized substring + split forms."
    )
    if not compare_modes:
        st.info("Turn on “Compute and compare all modes” in the sidebar (the first HanLP run may take a few minutes).")
    else:
        mode_hits = {m: (hits if m == mode else run_mode(m)) for m in avail_modes}
        cols = [MODE_SHORT[m] for m in avail_modes]
        firsts = pd.DataFrame(index=list(terms))
        counts_pre = pd.DataFrame(index=list(terms))
        for m in avail_modes:
            hm = mode_hits[m]
            pre = hm[(hm["date"] < WAIC_DATE) & (hm["count"] > 0)]
            firsts[MODE_SHORT[m]] = pre.groupby("term")["year"].min().reindex(list(terms))
            counts_pre[MODE_SHORT[m]] = pre.groupby("term")["count"].sum().reindex(list(terms), fill_value=0).astype(int)

        found = firsts.notna()
        n_found = found.sum(axis=1)
        status = pd.Series("mode-dependent", index=firsts.index)
        status[n_found == 0] = "new in WAIC"
        status[n_found == len(cols)] = "earlier (all modes)"

        def _year_diff(row):
            ys = sorted({int(v) for v in row.dropna()})
            return f"{ys[0]}–{ys[-1]}" if len(ys) > 1 else ""

        post_main = first_df.set_index("CH term")["After WAIC"].reindex(list(terms)).fillna(0).astype(int)

        # 1) Suvestinė pagal režimą
        st.markdown("**1) How many metaphors would look new in WAIC if only this mode were used**")
        summ_modes = pd.DataFrame({
            "Found before WAIC": found.sum(axis=0).astype(int),
            "Looks new in WAIC": (~found).sum(axis=0).astype(int),
            "Matches before WAIC": counts_pre.sum(axis=0).astype(int),
        })
        c_a, c_b = st.columns([1, 1])
        c_a.dataframe(summ_modes, width="stretch")
        c_b.bar_chart(summ_modes["Looks new in WAIC"], horizontal=True, height=260)

        # 2) Lentelė metafora × režimas
        st.markdown("**2) Metaphor × mode: first year before WAIC**")
        tbl = firsts.copy()
        tbl.insert(0, "ENG translation", [meta.loc[t, "translation"] for t in tbl.index])
        tbl["Status"] = status
        tbl["First-year range"] = firsts.apply(_year_diff, axis=1)
        tbl[f"After WAIC ({MODE_SHORT[mode]})"] = post_main
        tbl.index.name = "CH term"
        tbl["_ord"] = tbl["Status"].map(STATUS_ORDER)
        tbl["_min"] = firsts.min(axis=1)
        tbl = tbl.sort_values(["_ord", "_min"]).drop(columns=["_ord", "_min"])
        k1, k2, k3 = st.columns(3)
        k1.metric("New in WAIC (no mode)", int((status == "new in WAIC").sum()))
        k2.metric("Mode-dependent", int((status == "mode-dependent").sum()))
        k3.metric("Earlier (all modes)", int((status == "earlier (all modes)").sum()))
        only_diff = st.checkbox("Show only rows where modes disagree (status or first year)", value=False)
        show = tbl[(tbl["Status"] == "mode-dependent") | (tbl["First-year range"] != "")] if only_diff else tbl

        def _cell(v):
            return "background-color:#f8d7d7;" if v == "—" else "background-color:#d9f2d9;"

        disp = show.copy()
        for c in cols:
            disp[c] = [("—" if pd.isna(v) else str(int(v))) for v in show[c]]
        st.dataframe(disp.style.map(_cell, subset=cols), width="stretch",
                     height=min(35 * (len(disp) + 1) + 3, 900))
        st.download_button("Download table (CSV)", csv_bytes(tbl.reset_index()), "mode_comparison_first_years.csv", "text/csv")

        with st.expander("Number of matches before WAIC by mode"):
            st.dataframe(counts_pre.loc[show.index], width="stretch")

        # 3) Karščio žemėlapis
        st.markdown("**3) Heatmap (for figures)**")
        st.caption("Colour – first year before WAIC; grey “—” – not found. Rows are ordered as in the table.")
        incl_new = st.checkbox("Include “new in WAIC” metaphors (all grey)", value=False)
        hm_rows = [t for t in show.index if incl_new or show.loc[t, "Status"] != "new in WAIC"]
        if not hm_rows:
            st.info("No rows for the figure.")
        else:
            use_cjk = _cjk_font_name() is not None
            labels = [f"{t}  {meta.loc[t, 'translation'][:38]}" if use_cjk else f"{meta.loc[t, 'pinyin']} — {meta.loc[t, 'translation'][:30]}"
                      for t in hm_rows]
            if not use_cjk:
                st.caption("No Chinese font in this environment, so the figure shows pinyin (add packages.txt with fonts-noto-cjk).")
            png = heatmap_png(firsts.loc[hm_rows, cols], labels,
                              f"First appearance before WAIC 2026 ({years[0]}–{WAIC_DATE[:4]}) by segmentation mode")
            st.image(png, width="stretch")
            st.download_button("Download PNG", png, "mode_heatmap.png", "image/png")

# 7. Export
with tabs[6]:
    st.subheader("Export")
    st.caption("All main tables in one ZIP archive (CSV, “;” separator, UTF-8).")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("summary_by_year.csv", csv_bytes(summ.reset_index(names="year")))
        z.writestr("first_appearance.csv", csv_bytes(first_df))
        ty = term_year.copy()
        ty.columns = ty.columns.astype(str)
        z.writestr("term_year_counts.csv", csv_bytes(ty.reset_index(names="term")))
        tp = term_year.div(year_chars.reindex(years).values, axis=1).mul(10000).round(3)
        tp.columns = tp.columns.astype(str)
        z.writestr("term_year_per10k.csv", csv_bytes(tp.reset_index(names="term")))
        td = term_year_docs.copy()
        td.columns = td.columns.astype(str)
        z.writestr("term_year_docs.csv", csv_bytes(td.reset_index(names="term")))
        z.writestr("hits_by_document.csv", csv_bytes(hits))
        z.writestr("documents.csv", csv_bytes(docs_df))
    st.download_button("Download ZIP", buf.getvalue(), f"waic_metaphors_{mode}_{years[0]}-{years[-1]}.zip", "application/zip")
