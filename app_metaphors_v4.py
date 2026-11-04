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

APP_VERSION = "M1.5"
DEFAULT_TERMS_PATH = "WAIC2026_metaphor_terms_long.csv"

# Numatytasis korpusas: visi xi_speeches_YYYY aplankai iš discourse-analyzer-docs (main)
CORPUS_REPO_ZIP = "https://codeload.github.com/iconally/discourse-analyzer-docs/zip/refs/heads/main"
CORPUS_FILE_RE = re.compile(r"(?:^|/)xi_speeches_(\d{4})/([^/]+\.txt)$")
# Pati WAIC 2026 kalba (žodyno šaltinis): jhsjk article_id
WAIC_SOURCE_ID = "40763179"

MODE_LABELS = {
    "hybrid": "hibridinis (metaforoms): normalizuotas substring + išskaidytos formos",
    "substring": "substring (ilgiausi nepersidengiantys atitikmenys)",
    "jieba_precise": "jieba (precise)",
    "jieba_search": "jieba (search)",
    "thulac": "THULAC",
    "hanlp": "HanLP",
}
MODE_AVAILABLE = {
    "hybrid": True,
    "substring": True,
    "jieba_precise": JIEBA_AVAILABLE,
    "jieba_search": JIEBA_AVAILABLE,
    "thulac": THULAC_AVAILABLE,
    "hanlp": HANLP_AVAILABLE,
}

MODE_SHORT = {
    "hybrid": "Hibridinis",
    "substring": "Substring",
    "jieba_precise": "jieba precise",
    "jieba_search": "jieba search",
    "thulac": "THULAC",
    "hanlp": "HanLP",
}

# Trumpi algoritmų aprašymai (rodomi skirtuke „Segmentavimas“)
ALGO_INFO = {
    "hybrid": (
        "**Hibridinis (metaforoms).** Be segmentavimo. (1) Tekstas ir terminai normalizuojami: kabutės, skliaustai "
        "ir tarpai pašalinami, „、，,；;“ suvienodinami. (2) Ieškoma ilgiausių nepersidengiančių atitikmenų. "
        "(3) Jei terminas sudarytas iš kelių dalių (pvz. 各美其美、美美与共), kaip **išskaidyta forma** "
        "skaičiuojamas sakinys, kuriame yra visos dalys (pvz. 各美其美…美人之美，美美与共). "
        "1 hieroglifo terminai – per jieba precise žetonus."
    ),
    "substring": (
        "**Substring.** Be segmentavimo. Terminas ieškomas kaip tiksli ženklų seka; ilgesni terminai turi pirmenybę, "
        "atitikmenys nepersidengia (AB nėra skaičiuojamas ABCD viduje)."
    ),
    "jieba_precise": (
        "**jieba precise.** Tekstas skaidomas į žodžius (žodyno + HMM modelis). Terminas skaičiuojamas tik tada, "
        "kai jis **sutampa su vienu žetonu** – ilgos frazės, kurias jieba suskaido, nerandamos."
    ),
    "jieba_search": (
        "**jieba search.** Kaip precise, bet ilgi žodžiai papildomai skaidomi į trumpesnius (paieškos variklio režimas). "
        "Terminas turi sutapti su vienu iš žetonų."
    ),
    "thulac": (
        "**THULAC** (Tsinghua). Statistinis segmentatorius. Terminas randamas, jei jis sutampa su **1–8 gretimų žetonų** "
        "junginiu, todėl frazės randamos, jei jų ribos sutampa su žetonų ribomis."
    ),
    "hanlp": (
        "**HanLP** (neuroninis, ELECTRA). Terminas randamas, jei jis sutampa su **1–8 gretimų žetonų** junginiu; "
        "skaidymas priklauso nuo konteksto."
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


@st.cache_data(show_spinner="Atsisiunčiamas Xi kalbų korpusas iš GitHub…", ttl=24 * 3600)
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
    return sorted(out)


def load_terms(raw: bytes) -> pd.DataFrame:
    """Expected columns (semicolon separated): concept;term;pinyin;translation;category"""
    df = pd.read_csv(io.StringIO(read_txt(raw)), sep=";", dtype=str, keep_default_na=False)
    df.columns = [c.strip().lower().rstrip(",") for c in df.columns]
    required = ["concept", "term", "pinyin", "translation", "category"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Žodyne trūksta stulpelių: {missing}. Reikia: concept;term;pinyin;translation;category")
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
    if mode == "jieba_precise" and JIEBA_AVAILABLE:
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
    if mode in ("jieba_precise", "jieba_search"):
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
        max_n = 8 if mode in ("thulac", "hanlp") else 1
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
                ttl = tip.get(t, "") if t else ("Žodyno terminas čia yra, bet šiuo režimu nerastas" if an["miss"][k] else "")
                if t and an["char_kind"][k] == "variant":
                    ttl += " · išskaidyta forma"
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
        ttl = tip.get(t, "") if t else ("nerasta šiuo režimu" if an["miss"][k] else "")
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
    if mode in ("thulac", "hanlp"):
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
st.caption(f"Sinocode Discourse Analyzer · tyrimo versija {APP_VERSION} · WAIC 2026 metaforų paieška Xi kalbų korpuse")

with st.sidebar:
    st.header("1. Žodynas")
    terms_upload = st.file_uploader("Metaforų CSV (concept;term;pinyin;translation;category)", type=["csv"])

    st.header("2. Korpusas")
    use_github = st.checkbox(
        "Xi kalbos iš GitHub (xi_speeches_2013–2026)",
        value=True,
        help="Visi .txt iš iconally/discourse-analyzer-docs aplankų xi_speeches_YYYY. Atsisiunčiama vieną kartą ir laikoma talpykloje 24 val.",
    )
    exclude_source = st.checkbox(
        "Neįtraukti pačios WAIC 2026 kalbos",
        value=True,
        help=f"Žodynas sudarytas iš šios kalbos (article_id {WAIC_SOURCE_ID}), todėl ji iškreiptų pirmo pasirodymo rezultatus.",
    )
    if use_github and st.button("Atnaujinti korpusą iš GitHub"):
        load_github_corpus.clear()
    files = st.file_uploader("Papildomi dokumentai (TXT / DOCX)", type=["txt", "docx"], accept_multiple_files=True)
    folder = st.text_input(
        "Arba aplanko kelias (kai programa paleista lokaliai)",
        placeholder="/Users/.../xi_speeches",
        help="Visi .txt ir .docx failai šiame aplanke ir jo poaplankiuose.",
    )

    st.header("3. Atpažinimas")
    avail_modes = [m for m in MODE_LABELS if MODE_AVAILABLE[m]]
    mode = st.radio("Pagrindinis režimas", avail_modes, format_func=lambda m: MODE_LABELS[m])
    compare_modes = st.checkbox(
        "Skaičiuoti ir palyginti visus režimus",
        value=False,
        help="Lėčiau (ypač HanLP). Rezultatas – skirtukas „Režimų palyginimas“.",
    )
    unavailable = [MODE_LABELS[m] for m in MODE_LABELS if not MODE_AVAILABLE[m]]
    if unavailable:
        st.caption("Neįdiegta šioje aplinkoje: " + ", ".join(unavailable))

    st.header("4. Kontekstas")
    kwic_window = st.slider("Hieroglifų prieš ir po atitikmens (±N)", 5, 60, 15)

# Load dictionary
try:
    if terms_upload is not None:
        terms_df = load_terms(terms_upload.getvalue())
    elif Path(DEFAULT_TERMS_PATH).exists():
        terms_df = load_terms(Path(DEFAULT_TERMS_PATH).read_bytes())
        st.sidebar.caption(f"Naudojamas numatytasis žodynas: {DEFAULT_TERMS_PATH}")
    else:
        terms_df = None
except Exception as e:
    st.error(f"Nepavyko nuskaityti žodyno: {e}")
    st.stop()

# Load documents
raw_docs: List[Tuple[str, bytes]] = []
if use_github:
    try:
        gh_docs = load_github_corpus()
        raw_docs.extend(gh_docs)
        st.sidebar.caption(f"GitHub korpusas: {len(gh_docs)} failų")
    except Exception as e:
        st.sidebar.error(f"Nepavyko parsiųsti korpuso iš GitHub: {e}")
if files:
    raw_docs.extend((f.name, f.getvalue()) for f in files)
if folder.strip():
    p = Path(folder.strip()).expanduser()
    if p.is_dir():
        for fp in sorted(p.rglob("*")):
            if fp.suffix.lower() in (".txt", ".docx") and fp.is_file():
                raw_docs.append((fp.name, fp.read_bytes()))
    else:
        st.sidebar.error("Aplankas nerastas.")

if terms_df is None or terms_df.empty or not raw_docs:
    st.info("Įkelk metaforų žodyną (CSV) ir korpuso dokumentus (arba nurodyk aplanką), kad prasidėtų analizė.")
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

view_docs = list(docs)  # skirtukui „Segmentavimas“: visi dokumentai, įskaitant WAIC kalbą
if exclude_source:
    docs = [d for d in docs if f"__{WAIC_SOURCE_ID}__" not in d["filename"]]

if skipped:
    st.warning(f"Praleista {len(skipped)} failų (nėra metų pavadinime arba nepavyko perskaityti): " + ", ".join(skipped[:10]) + (" …" if len(skipped) > 10 else ""))
if not docs:
    st.stop()

all_years = sorted({d["year"] for d in docs})
with st.sidebar:
    st.header("5. Laikotarpis")
    if len(all_years) > 1:
        yr = st.slider("Metai", min(all_years), max(all_years), (min(all_years), max(all_years)))
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
    prog = st.progress(0.0, text=f"Skaičiuojama: {MODE_LABELS[m]}")
    for i, d in enumerate(docs):
        variant = hybrid_counts(d["text"], terms)[1] if m == "hybrid" else {}
        for t, c in count_doc(d["text"], terms, m).items():
            rows.append({"filename": d["filename"], "year": d["year"], "date": d["date"], "term": t, "count": c,
                         "variant": int(variant.get(t, 0))})
        prog.progress((i + 1) / len(docs), text=f"Skaičiuojama: {MODE_LABELS[m]} ({i + 1}/{len(docs)})")
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
    h = hits[hits["term"] == t].sort_values(["date", "filename"])
    r = meta.loc[t]
    row = {
        "CH term": t,
        "Pinyin": r["pinyin"],
        "ENG translation": r["translation"],
        "Concept": r["concept"],
        "Category": r["category"],
        "Pirmi metai": int(h["year"].iloc[0]) if len(h) else None,
        "Pirma data": h["date"].iloc[0] if len(h) else "",
        "Pirmas dokumentas": h["filename"].iloc[0] if len(h) else "",
        "Iš viso": int(h["count"].sum()),
        "Iš jų išskaidyta forma": int(h["variant"].sum()),
        "Dokumentų": int(h["filename"].nunique()),
        "Aktyvių metų": int((term_year.loc[t] > 0).sum()),
    }
    for y in years:
        row[str(y)] = int(term_year.loc[t, y])
    row["Statusas"] = "nerasta korpuse" if row["Iš viso"] == 0 else "rasta korpuse"
    if mode != "hybrid":
        del row["Iš jų išskaidyta forma"]
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
                "Metai": d["year"], "Data": d["date"], "Dokumentas": d["filename"],
                "Kairė": left, "Terminas": t, "Dešinė": right,
                "Metafora?": "", "Pastaba": "",
            })
    return pd.DataFrame(out, columns=["Metai", "Data", "Dokumentas", "Kairė", "Terminas", "Dešinė", "Metafora?", "Pastaba"]).sort_values(["Data", "Dokumentas"]) if out else pd.DataFrame(columns=["Metai", "Data", "Dokumentas", "Kairė", "Terminas", "Dešinė", "Metafora?", "Pastaba"])


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, sep=";").encode("utf-8-sig")


# -----------------------------
# Tabs
# -----------------------------
seg_tab, *tabs = st.tabs(["Segmentavimas", "Suvestinė", "Pirmas pasirodymas", "Dinamika", "Pokyčiai per metų poras", "Kontekstas (KWIC)", "Režimų palyginimas", "Eksportas"])


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
    sel_year = cy.selectbox("Metai", v_years, index=v_years.index(src["year"]), key="seg_year")
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
    li = cd.selectbox("Pasirink dokumentą", range(len(ydocs)), index=default_i,
                      format_func=lambda i: labels[i], key="seg_doc")
    doc = ydocs[li]
    text = doc["text"]

    col1, col2 = st.columns([1, 4])
    col1.write(f"**Metai (Year):** {doc['year']}")
    col2.write(f"**Pavadinimas (CN):** {_title_cn(doc['filename'])}")
    if f"__{WAIC_SOURCE_ID}__" in doc["filename"] and exclude_source:
        st.caption("Tai WAIC 2026 kalba (žodyno šaltinis): rodoma čia, bet neįtraukta į korpuso analizę.")

    with st.expander("Teksto statistika", expanded=False):
        st.write({"Characters": len(text), "Lines": text.count("\n") + 1 if text else 0})

    with st.expander("Kaip suskaidytas tekstas", expanded=True):
        view_mode = st.radio("Režimas", avail_modes, index=avail_modes.index(mode), horizontal=True,
                             format_func=lambda m: MODE_SHORT[m], key="seg_view_mode")
        st.markdown(ALGO_INFO[view_mode])
        st.caption("Žetonų ribos „|“, žodyno terminai nuspalvinti (užvesk pelę – pinyin ir vertimas). "
                   "Raudonas punktyras – terminas tekste yra, bet šiuo režimu nerastas. "
                   "Punktyrinis rėmelis – hibridinio režimo išskaidyta forma.")
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

    with st.expander("Ta pati pastraipa visais režimais", expanded=False):
        doc_found = {m_: sum(1 for c in count_doc(text, terms, m_).values() if c > 0) for m_ in avail_modes}
        doc_ref = doc_found.get("hybrid", max(doc_found.values()) if doc_found else 0)
        st.markdown("**Visas dokumentas:** " + " · ".join(
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
        pi = pc1.selectbox("Pastraipa", range(len(cand)),
                           format_func=lambda k: f"¶{cand[k][0] + 1} · {ref_hits[cand[k][0]]} term. · {cand[k][1][:40]}…",
                           key="seg_para")
        line = cand[pi][1]
        sents = sentence_spans(line)
        if st.session_state.get("seg_sent", 0) > len(sents):
            st.session_state["seg_sent"] = 0
        si = pc2.selectbox("Sakinys", range(len(sents) + 1),
                           format_func=lambda k: "visa pastraipa" if k == 0 else f"{k}. {line[sents[k - 1][0]:sents[k - 1][1]][:18]}…",
                           key="seg_sent")
        per_row = pc3.number_input("Ženklų eilutėje", 15, 60, 30, step=5, key="seg_per_row")
        analyses = {m_: analyze_line(line, m_, set(terms), list(terms)) for m_ in avail_modes}
        ref_n = len(analyses["hybrid"]["found"]) if "hybrid" in analyses else 0
        st.markdown("**Ši pastraipa:** " + " · ".join(f"**{MODE_SHORT[m_]}** {len(analyses[m_]['found'])}/{ref_n}" for m_ in avail_modes))
        st.caption("Kiekvienas hieroglifas – vienodo pločio langelis, todėl visuose režimuose tas pats ženklas yra toje pačioje vietoje. "
                   "Pilka linija – žetono riba; taškinė – jieba search pažodžio riba; raudonas punktyras – terminas yra, bet nerastas; "
                   "rėmelis – išskaidyta forma.")
        blocks = sents if si == 0 else [sents[si - 1]]
        st.markdown(grid_blocks_html(line, analyses, avail_modes, blocks, int(per_row), tip, color_of),
                    unsafe_allow_html=True)

    st.divider()
    st.subheader("Analizė")
    st.caption(f"Režimas: {MODE_LABELS[view_mode]}")
    dcounts = count_doc(text, terms, view_mode)
    dvariant = hybrid_counts(text, terms)[1] if view_mode == "hybrid" else {}
    td = pd.DataFrame([{
        "CH term": t, "Pinyin": meta.loc[t, "pinyin"], "ENG translation": meta.loc[t, "translation"],
        "Concept": meta.loc[t, "concept"], "Category": meta.loc[t, "category"], "Count": c,
        **({"Iš jų išskaidyta forma": int(dvariant.get(t, 0))} if view_mode == "hybrid" else {}),
    } for t, c in dcounts.items() if c > 0])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total matches", int(td["Count"].sum()) if not td.empty else 0)
    m2.metric("Unique terms", int(td["CH term"].nunique()) if not td.empty else 0)
    m3.metric("Unique concepts", int(td["Concept"].nunique()) if not td.empty else 0)
    m4.metric("Categories hit", int(td["Category"].nunique()) if not td.empty else 0)

    st.markdown("### 1) Term detail")
    if td.empty:
        st.info("Šiame dokumente nerasta nė vieno žodyno termino.")
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

    st.markdown("### 4) Režimų palyginimas šiame dokumente")
    if cmp.empty:
        st.info("Nė vienas režimas nerado terminų šiame dokumente.")
    else:
        st.caption("Paspausk eilutės kairėje esantį langelį – žemiau atsiras tos metaforos sakinys visais režimais, "
                   "o bloke „Ta pati pastraipa visais režimais“ bus atidaryta jos pastraipa.")
        mode_cols = list(cmp.columns)
        tbl = cmp.copy()
        tbl.insert(0, "ENG translation", [meta.loc[t, "translation"] for t in tbl.index])
        tot = pd.DataFrame([["", *(cmp > 0).sum(axis=0).tolist()], ["", *cmp.sum(axis=0).tolist()]],
                           index=["Rasta terminų", "Atitikmenų iš viso"], columns=tbl.columns)
        tbl = pd.concat([tbl, tot])
        tbl.index.name = "CH term"

        def _cell(v):
            return "background-color:#d9f2d9;" if v > 0 else "background-color:#f8d7d7;"

        st.dataframe(tbl.style.map(_cell, subset=pd.IndexSlice[list(cmp.index), mode_cols]), width="stretch",
                     on_select="rerun", selection_mode="single-row", key=cmp_key)
        st.download_button("Atsisiųsti palyginimą (CSV)", csv_bytes(tbl.reset_index()),
                           f"{doc['date']}_mode_comparison.csv", "text/csv")

        if sel_term is not None:
            occ = term_occurrences(sel_term)
            st.markdown(f"#### {sel_term} — {meta.loc[sel_term, 'translation']}")
            if not occ:
                st.info("Šio termino vietos tekste nustatyti nepavyko.")
            else:
                oi = 0
                if len(occ) > 1:
                    oi = st.selectbox("Pasikartojimas", range(len(occ)),
                                      format_func=lambda k: f"{k + 1} iš {len(occ)} · ¶{occ[k][0] + 1}",
                                      key=f"seg_occ_{cmp_key}_{sel_term}")
                li_, a_, b_ = occ[oi]
                ln = doc_lines[li_]
                ss = sentence_spans(ln)
                blk = [next(((x, y) for (x, y) in ss if x <= a_ < y), (0, len(ln)))]
                # jei terminas išeina už sakinio ribų (išskaidyta forma), rodom iki termino pabaigos
                blk = [(min(blk[0][0], a_), max(blk[0][1], b_))]
                an_ = {m_: analyze_line(ln, m_, set(terms), list(terms)) for m_ in avail_modes}
                st.caption(f"¶{li_ + 1} · sakinys su šia metafora")
                st.markdown(grid_blocks_html(ln, an_, avail_modes, blk,
                                             int(st.session_state.get("seg_per_row", 30)), tip, color_of),
                            unsafe_allow_html=True)

# 1. Summary
with tabs[0]:
    st.subheader("Korpuso ir atitikmenų suvestinė")
    st.caption(f"Režimas: {MODE_LABELS[mode]} · žodyne {len(terms)} terminų · korpuse {len(docs)} dokumentų")
    summ = pd.DataFrame({
        "Dokumentų": year_docs,
        "Simbolių": year_chars,
        "Atitikmenų": term_year.sum(axis=0),
        "Rastų terminų": (term_year > 0).sum(axis=0),
    })
    summ["Atitikmenų per 10k"] = [round(per10k(summ.loc[y, "Atitikmenų"], summ.loc[y, "Simbolių"]), 3) for y in summ.index]
    summ.index = summ.index.astype(str)
    st.dataframe(summ, width="stretch")
    gaps = [y for y in range(min(years), max(years) + 1) if y not in years]
    if gaps:
        st.warning("Korpuse trūksta metų: " + ", ".join(map(str, gaps)))
    c1, c2, c3 = st.columns(3)
    c1.metric("Rasta korpuse", int((first_df["Iš viso"] > 0).sum()))
    c2.metric("Nerasta korpuse", int((first_df["Iš viso"] == 0).sum()))
    c3.metric("Žodyno terminų", len(terms))

# 2. First appearance
with tabs[1]:
    st.subheader("Kada metafora pirmą kartą pasirodo korpuse")
    st.caption("„Nerasta korpuse“ – kandidatai į naujas WAIC 2026 metaforas pasirinktame laikotarpyje.")
    f1, f2, f3 = st.columns(3)
    status_f = f1.multiselect("Statusas", ["rasta korpuse", "nerasta korpuse"], default=["rasta korpuse", "nerasta korpuse"])
    concept_f = f2.multiselect("Concept", sorted(first_df["Concept"].unique()))
    cat_f = f3.multiselect("Category", sorted(first_df["Category"].unique()))
    v = first_df[first_df["Statusas"].isin(status_f)]
    if concept_f:
        v = v[v["Concept"].isin(concept_f)]
    if cat_f:
        v = v[v["Category"].isin(cat_f)]
    v = v.sort_values(["Pirmi metai", "Pirma data", "CH term"], na_position="last")
    st.dataframe(v, width="stretch", hide_index=True)
    st.download_button("Atsisiųsti CSV", csv_bytes(v), "first_appearance.csv", "text/csv")

# 3. Dynamics
with tabs[2]:
    st.subheader("Dinamika pagal metus")
    a, b = st.columns(2)
    level = a.radio("Lygis", ["Term", "Concept", "Category"], horizontal=True)
    metric = b.radio(
        "Metrika",
        ["Per 10k characters", "Raw count", "Dokumentų dalis (%)"],
        horizontal=True,
        help="Per 10k = visi metų atitikmenys / visi metų simboliai × 10 000. "
             "Dokumentų dalis = kiek % tų metų dokumentų turi bent vieną atitikmenį.",
    )
    key_col = {"Term": "term", "Concept": "concept", "Category": "category"}[level]
    h = hits.merge(terms_df[["term", "concept", "category"]], on="term", how="left")
    if metric == "Dokumentų dalis (%)":
        piv = h.pivot_table(index=key_col, columns="year", values="filename", aggfunc="nunique", fill_value=0)
        piv = piv.reindex(columns=years, fill_value=0)
        piv = piv.div(year_docs.reindex(years).values, axis=1) * 100
    else:
        piv = h.pivot_table(index=key_col, columns="year", values="count", aggfunc="sum", fill_value=0).reindex(columns=years, fill_value=0)
        if metric == "Per 10k characters":
            piv = piv.div(year_chars.reindex(years).values, axis=1) * 10000
    piv = piv.round(3)
    if piv.empty:
        st.info("Pasirinktame laikotarpyje atitikmenų nėra.")
    else:
        order = piv.sum(axis=1).sort_values(ascending=False).index.tolist()
        chosen = st.multiselect(f"Pasirink {level} (rekomenduojama iki 8–12)", order, default=order[:8])
        if chosen:
            chart = piv.loc[chosen].T
            chart.index = chart.index.astype(str)
            st.line_chart(chart)
        tbl = piv.loc[order].copy()
        tbl.columns = tbl.columns.astype(str)
        st.dataframe(tbl, width="stretch")
        st.download_button("Atsisiųsti lentelę", csv_bytes(tbl.reset_index()), f"dynamics_{level}_{metric}.csv", "text/csv")

# 4. Year pairs
with tabs[3]:
    st.subheader("Terminų pokyčiai per metų poras")
    if len(years) < 2:
        st.info("Reikia bent dviejų metų.")
    else:
        pair_rows = []
        for y1, y2 in zip(years[:-1], years[1:]):
            for t in terms:
                c1, c2 = int(term_year.loc[t, y1]), int(term_year.loc[t, y2])
                if c1 == 0 and c2 == 0:
                    continue
                p1, p2 = per10k(c1, year_chars[y1]), per10k(c2, year_chars[y2])
                if c1 == 0:
                    status = "atsirado"
                elif c2 == 0:
                    status = "išnyko"
                elif p2 > p1 * 1.1:
                    status = "↑"
                elif p2 < p1 * 0.9:
                    status = "↓"
                else:
                    status = "→"
                pair_rows.append({
                    "Pora": f"{y1}–{y2}", "CH term": t, "Concept": meta.loc[t, "concept"],
                    f"Count y1": c1, f"Count y2": c2,
                    "Per10k y1": round(p1, 3), "Per10k y2": round(p2, 3), "Pokytis": status,
                })
        pairs_df = pd.DataFrame(pair_rows)
        if pairs_df.empty:
            st.info("Atitikmenų nėra.")
        else:
            sel_pair = st.selectbox("Metų pora", pairs_df["Pora"].unique().tolist())
            st.dataframe(pairs_df[pairs_df["Pora"] == sel_pair], width="stretch", hide_index=True)
            st.download_button("Atsisiųsti visas poras", csv_bytes(pairs_df), "year_pairs.csv", "text/csv")

# 5. KWIC
with tabs[4]:
    st.subheader("Kontekstas (KWIC)")
    st.caption(
        f"Kiekvienas atitikmuo su ±{kwic_window} hieroglifų kontekstu (keičiama šoninėje juostoje). "
        "Pozicijos nustatomos substring (ilgiausių atitikmenų) būdu. "
        "Stulpeliuose „Metafora?“ ir „Pastaba“ gali žymėti rankiniu būdu ir atsisiųsti."
    )
    found = first_df[first_df["Iš viso"] > 0]["CH term"].tolist()
    if not found:
        st.info("Atitikmenų nėra.")
    else:
        sel_terms = st.multiselect("Terminai", found, default=found[:1])
        if sel_terms:
            kw = kwic_rows(sel_terms, kwic_window)
            st.caption(f"Atitikmenų: {len(kw)}")
            edited = st.data_editor(
                kw,
                width="stretch",
                hide_index=True,
                disabled=["Metai", "Data", "Dokumentas", "Kairė", "Terminas", "Dešinė"],
                column_config={"Metafora?": st.column_config.SelectboxColumn(options=["", "taip", "ne", "neaišku"])},
                key="kwic_editor",
            )
            st.download_button("Atsisiųsti KWIC su žymėjimais", csv_bytes(edited), "kwic.csv", "text/csv")

# 6. Mode comparison
with tabs[5]:
    st.subheader("Režimų palyginimas")
    st.caption(
        "Tie patys terminai skaičiuojami skirtingais segmentatoriais. "
        "jieba (kaip ir app.py) lygina su vienu žetonu, todėl ilgos frazės dažnai negaunamos; "
        "THULAC ir HanLP jungia iki 8 gretimų žetonų."
    )
    if not compare_modes:
        st.info("Įjunk šoninėje juostoje „Skaičiuoti ir palyginti visus režimus“.")
    else:
        comp = pd.DataFrame(index=list(terms))
        firsts = pd.DataFrame(index=list(terms))
        for m in avail_modes:
            hm = hits if m == mode else run_mode(m)
            comp[MODE_LABELS[m]] = hm.groupby("term")["count"].sum().reindex(list(terms), fill_value=0).astype(int)
            firsts[MODE_LABELS[m]] = hm.groupby("term")["year"].min().reindex(list(terms))
        comp["Skiriasi"] = comp.nunique(axis=1) > 1
        only_diff = st.checkbox("Rodyti tik terminus, kurių rezultatai skiriasi", value=True)
        show = comp[comp["Skiriasi"]] if only_diff else comp
        st.markdown("**Atitikmenų skaičius**")
        st.dataframe(show, width="stretch")
        st.markdown("**Pirmi metai**")
        st.dataframe(firsts.loc[show.index], width="stretch")
        st.download_button("Atsisiųsti palyginimą", csv_bytes(comp.reset_index(names="CH term")), "mode_comparison.csv", "text/csv")

# 7. Export
with tabs[6]:
    st.subheader("Eksportas")
    st.caption("Visos pagrindinės lentelės vienu ZIP archyvu (CSV, skirtukas „;“, UTF-8).")
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
    st.download_button("Atsisiųsti ZIP", buf.getvalue(), f"waic_metaphors_{mode}_{years[0]}-{years[-1]}.zip", "application/zip")
