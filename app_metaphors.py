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

APP_VERSION = "M1.1"
DEFAULT_TERMS_PATH = "WAIC2026_metaphor_terms_long.csv"

# Numatytasis korpusas: visi xi_speeches_YYYY aplankai iš discourse-analyzer-docs (main)
CORPUS_REPO_ZIP = "https://codeload.github.com/iconally/discourse-analyzer-docs/zip/refs/heads/main"
CORPUS_FILE_RE = re.compile(r"(?:^|/)xi_speeches_(\d{4})/([^/]+\.txt)$")
# Pati WAIC 2026 kalba (žodyno šaltinis): jhsjk article_id
WAIC_SOURCE_ID = "40763179"

MODE_LABELS = {
    "substring": "substring (ilgiausi nepersidengiantys atitikmenys)",
    "jieba_precise": "jieba (precise)",
    "jieba_search": "jieba (search)",
    "thulac": "THULAC",
    "hanlp": "HanLP",
}
MODE_AVAILABLE = {
    "substring": True,
    "jieba_precise": JIEBA_AVAILABLE,
    "jieba_search": JIEBA_AVAILABLE,
    "thulac": THULAC_AVAILABLE,
    "hanlp": HANLP_AVAILABLE,
}

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
            m = CORPUS_FILE_RE.search(info.filename)
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


@st.cache_data(show_spinner=False)
def count_doc(text: str, terms: Tuple[str, ...], mode: str) -> Dict[str, int]:
    """term -> count for one document, following app.py's analyze_text logic per mode."""
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

if exclude_source:
    raw_docs = [(n, b) for n, b in raw_docs if f"__{WAIC_SOURCE_ID}__" not in n]

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
        for t, c in count_doc(d["text"], terms, m).items():
            rows.append({"filename": d["filename"], "year": d["year"], "date": d["date"], "term": t, "count": c})
        prog.progress((i + 1) / len(docs), text=f"Skaičiuojama: {MODE_LABELS[m]} ({i + 1}/{len(docs)})")
    prog.empty()
    return pd.DataFrame(rows, columns=["filename", "year", "date", "term", "count"])


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
        "Dokumentų": int(h["filename"].nunique()),
        "Aktyvių metų": int((term_year.loc[t] > 0).sum()),
    }
    for y in years:
        row[str(y)] = int(term_year.loc[t, y])
    row["Statusas"] = "nerasta korpuse" if row["Iš viso"] == 0 else "rasta korpuse"
    first_rows.append(row)
first_df = pd.DataFrame(first_rows)


def kwic_rows(selected_terms: List[str], window: int) -> pd.DataFrame:
    out = []
    sel = set(selected_terms)
    for d in docs:
        text = d["text"]
        for s, e, t in longest_match_positions(text, list(terms)):
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
tabs = st.tabs(["Suvestinė", "Pirmas pasirodymas", "Dinamika", "Pokyčiai per metų poras", "Kontekstas (KWIC)", "Režimų palyginimas", "Eksportas"])

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
