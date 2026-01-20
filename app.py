import re
import io
import csv
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict

import streamlit as st

try:
    import docx
except Exception:
    docx = None

st.set_page_config(
    page_title="Discourse Analyzer (CN ↔ EN) – per-doc, dominant CN term",
    layout="wide"
)

# =========================
# Regex
# =========================
YEAR_RE = re.compile(r"(20[0-2]\d)")
DOCID_RE = re.compile(r"(?:^|_)([A-Za-z0-9\-]+)(?:_CN)(?:_|\.)")
ORDER_RE = re.compile(r"(20[0-2]\d)[_\-\.](\d{1,2})(?=[_\-\.])")

# =========================
# File reading
# =========================
def read_txt(uploaded_file) -> str:
    raw = uploaded_file.read()
    for enc in ("utf-8", "utf-8-sig", "gb18030", "big5", "latin-1"):
        try:
            return raw.decode(enc)
        except Exception:
            continue
    return raw.decode("utf-8", errors="replace")


def read_docx(uploaded_file) -> str:
    if docx is None:
        raise RuntimeError("python-docx neįdiegtas.")
    bio = io.BytesIO(uploaded_file.read())
    document = docx.Document(bio)
    return "\n".join(p.text for p in document.paragraphs)


def detect_year(filename: str) -> Optional[int]:
    m = YEAR_RE.search(filename)
    return int(m.group(1)) if m else None


def detect_order_in_year(filename: str) -> Optional[int]:
    m = ORDER_RE.search(filename)
    return int(m.group(2)) if m else None


def detect_docid(filename: str) -> Optional[str]:
    m = DOCID_RE.search(filename)
    return m.group(1) if m else None


# =========================
# CSV helpers
# =========================
def load_csv(uploaded_file) -> List[dict]:
    content = uploaded_file.getvalue().decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(content))
    return [{k.strip().lower(): (v or "").strip() for k, v in r.items()} for r in reader]


def to_csv_bytes(rows: List[dict], fields: List[str]) -> bytes:
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return out.getvalue().encode("utf-8")


# =========================
# Data models
# =========================
@dataclass
class Term:
    concept: str
    term: str
    category: str


@dataclass
class Doc:
    filename: str
    year: Optional[int]
    order_in_year: Optional[int]
    docid: str
    text: str


# =========================
# Load terms
# =========================
def load_terms(rows: List[dict]) -> List[Term]:
    return [
        Term(r["concept"], r["term"], r["category"])
        for r in rows if r.get("concept") and r.get("term")
    ]


def build_cn_patterns(terms: List[Term]) -> Dict[Tuple[str, str], re.Pattern]:
    return {(t.concept, t.term): re.compile(re.escape(t.term)) for t in terms}


# =========================
# Ingest docs
# =========================
def ingest_docs(files) -> List[Doc]:
    docs = []
    for f in files:
        name = f.name
        year = detect_year(name)
        order = detect_order_in_year(name)
        docid = detect_docid(name)

        if not docid:
            raise ValueError(f"Nerastas DocID faile: {name}")

        ext = name.lower().split(".")[-1]
        if ext == "txt":
            text = read_txt(f)
        elif ext == "docx":
            text = read_docx(f)
        else:
            raise ValueError("Naudok tik .txt arba .docx")

        docs.append(Doc(name, year, order, docid, text))

    return sorted(docs, key=lambda d: (d.year or 9999, d.order_in_year or 9999, d.filename))


# =========================
# Analysis
# =========================
def analyze_terms_per_doc(docs: List[Doc], terms: List[Term]):
    patterns = build_cn_patterns(terms)

    # docid → concept → Counter(term → count)
    per_doc = defaultdict(lambda: defaultdict(Counter))

    for d in docs:
        for t in terms:
            cnt = len(patterns[(t.concept, t.term)].findall(d.text))
            if cnt > 0:
                per_doc[d.docid][t.concept][t.term] += cnt

    return per_doc


# =========================
# UI
# =========================
st.title("Discourse Analyzer – dokumentai po vieną (CN)")

with st.sidebar:
    cn_files = st.file_uploader(
        "CN dokumentai (.txt, .docx)", type=["txt", "docx"], accept_multiple_files=True
    )
    cn_terms_file = st.file_uploader(
        "terms_cn.csv (concept, term, category)", type=["csv"]
    )
    en_terms_file = st.file_uploader(
        "terms_en.csv (concept, term, category)", type=["csv"]
    )

run = st.button("▶️ Analizuoti")

if not run:
    st.stop()

try:
    cn_terms = load_terms(load_csv(cn_terms_file))
    en_terms = load_terms(load_csv(en_terms_file)) if en_terms_file else []

    en_label = {}
    for t in en_terms:
        en_label.setdefault(t.concept, t.term)

    cn_variants = defaultdict(list)
    category = {}
    for t in cn_terms:
        cn_variants[t.concept].append(t.term)
        category.setdefault(t.concept, t.category)

    docs = ingest_docs(cn_files)
    per_doc = analyze_terms_per_doc(docs, cn_terms)

    tabs = st.tabs([f"{d.year} | {d.docid}" for d in docs])

    for d, tab in zip(docs, tabs):
        with tab:
            st.markdown("### Dokumento informacija")
            st.write({
                "Metai": d.year,
                "DocID": d.docid,
                "Failas": d.filename
            })

            rows = []
            for concept, counter in per_doc[d.docid].items():
                dominant_cn = counter.most_common(1)[0][0]
                total = sum(counter.values())
                rows.append({
                    "CH term": dominant_cn,
                    "vertimas ENG": en_label.get(concept, ""),
                    "concept": concept,
                    "category": category.get(concept, ""),
                    "count": total,
                    "CH term variants": " / ".join(cn_variants[concept])
                })

            rows.sort(key=lambda x: x["count"], reverse=True)
            st.dataframe(rows, use_container_width=True)

            st.download_button(
                "⬇️ Atsisiųsti CSV",
                data=to_csv_bytes(
                    rows,
                    ["CH term", "vertimas ENG", "concept", "category", "count", "CH term variants"]
                ),
                file_name=f"{d.docid}_analysis.csv"
            )

except Exception as e:
    st.error(str(e))
