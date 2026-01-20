import re
import io
import csv
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict

import streamlit as st

try:
    import docx  # python-docx
except Exception:
    docx = None

st.set_page_config(page_title="Discourse Analyzer (CN ↔ EN) - per-doc tabs", layout="wide")

# ----------------------------
# Regex
# ----------------------------
YEAR_RE = re.compile(r"(20[0-2]\d)")
DOCID_RE = re.compile(r"(?:^|_)([A-Za-z0-9\-]+)(?:_CN|_EN)(?:_|\.)")
ORDER_RE = re.compile(r"(20[0-2]\d)[_\-\.](\d{1,2})(?=[_\-\.])")
CN_KEEP_RE = re.compile(r"[\u4e00-\u9fff]")

# ----------------------------
# File reading
# ----------------------------
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
        raise RuntimeError("python-docx neįdiegtas arba nepavyko importuoti.")
    data = uploaded_file.read()
    bio = io.BytesIO(data)
    document = docx.Document(bio)
    return "\n".join(p.text for p in document.paragraphs)


def detect_year(filename: str) -> Optional[int]:
    m = YEAR_RE.search(filename)
    return int(m.group(1)) if m else None


def detect_order_in_year(filename: str) -> Optional[int]:
    m = ORDER_RE.search(filename)
    if not m:
        return None
    try:
        return int(m.group(2))
    except Exception:
        return None


def detect_docid(filename: str) -> Optional[str]:
    m = DOCID_RE.search(filename)
    return m.group(1) if m else None


# ----------------------------
# CSV helpers (no pandas)
# ----------------------------
def load_csv(uploaded_file) -> List[dict]:
    content = uploaded_file.getvalue().decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(content))
    rows = []
    for r in reader:
        rows.append({(k or "").strip().lower(): (v or "").strip() for k, v in r.items()})
    return rows


def to_csv_bytes(rows: List[dict], fieldnames: List[str]) -> bytes:
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=fieldnames, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return out.getvalue().encode("utf-8")


# ----------------------------
# Term matching
# ----------------------------
@dataclass
class Term:
    concept: str
    term: str
    category: str


def load_terms(rows: List[dict], lang: str) -> List[Term]:
    required = {"concept", "term", "category"}
    if not rows:
        raise ValueError(f"{lang} terms CSV tuščias.")
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"{lang} terms CSV trūksta stulpelių: {', '.join(sorted(missing))}")
    out = []
    for r in rows:
        c = r.get("concept", "").strip()
        t = r.get("term", "").strip()
        cat = r.get("category", "").strip()
        if c and t:
            out.append(Term(concept=c, term=t, category=cat))
    return out


def build_patterns(terms: List[Term], lang: str) -> Dict[Tuple[str, str], re.Pattern]:
    patterns = {}
    for t in terms:
        escaped = re.escape(t.term)
        if lang.upper() == "CN":
            patterns[(t.concept, t.term)] = re.compile(escaped)
        else:
            if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 \-]*[A-Za-z0-9]", t.term):
                patterns[(t.concept, t.term)] = re.compile(rf"\b{escaped}\b", re.IGNORECASE)
            else:
                patterns[(t.concept, t.term)] = re.compile(escaped, re.IGNORECASE)
    return patterns


# ----------------------------
# Docs
# ----------------------------
@dataclass
class Doc:
    lang: str
    filename: str
    year: Optional[int]
    order_in_year: Optional[int]
    docid: Optional[str]
    text: str


def make_doc_key(d: Doc) -> str:
    y = d.year if d.year is not None else 9999
    o = d.order_in_year if d.order_in_year is not None else 99
    did = d.docid if d.docid else d.filename
    return f"{y:04d}.{o:02d}|{did}|{d.lang}"


def ingest_files(files, lang: str) -> List[Doc]:
    docs: List[Doc] = []
    for f in files:
        name = f.name
        year = detect_year(name)
        order_in_year = detect_order_in_year(name)
        docid = detect_docid(name)

        ext = name.lower().split(".")[-1]
        if ext == "txt":
            txt = read_txt(f)
        elif ext == "docx":
            txt = read_docx(f)
        elif ext == "doc":
            raise ValueError(f"Failas {name} yra .doc. Konvertuok į .docx.")
        else:
            raise ValueError(f"Nepalaikomas formatas: {name}. Naudok .txt arba .docx.")

        docs.append(Doc(lang=lang, filename=name, year=year, order_in_year=order_in_year, docid=docid, text=txt))
    return docs


def sort_docs(docs: List[Doc]) -> List[Doc]:
    def key(d: Doc):
        y = d.year if d.year is not None else 9999
        o = d.order_in_year if d.order_in_year is not None else 9999
        return (y, o, d.filename)
    return sorted(docs, key=key)


# ----------------------------
# Analysis
# ----------------------------
def analyze_docs(docs: List[Doc], terms: List[Term], lang: str) -> List[dict]:
    patterns = build_patterns(terms, lang=lang)
    rows: List[dict] = []
    for d in docs:
        doc_key = make_doc_key(d)
        for t in terms:
            pat = patterns[(t.concept, t.term)]
            cnt = len(pat.findall(d.text))
            rows.append({
                "lang": d.lang,
                "year": d.year,
                "order_in_year": d.order_in_year,
                "docid": d.docid,
                "filename": d.filename,
                "doc_key": doc_key,
                "concept": t.concept,
                "term": t.term,
                "category": t.category,
                "count": cnt,
            })
    return rows


def doc_profile_concept(detail_rows: List[dict]) -> List[dict]:
    """
    Aggregates counts per doc_key + concept (summing term variants).
    """
    agg = defaultdict(int)
    cat_by_concept = {}

    for r in detail_rows:
        key = (r["doc_key"], r["lang"], r["year"], r["order_in_year"], r["docid"], r["concept"])
        agg[key] += int(r["count"] or 0)
        # category from CN terms (first seen)
        if r["concept"] not in cat_by_concept:
            cat_by_concept[r["concept"]] = r.get("category", "")

    out = []
    for (doc_key, lang, year, order_in_year, docid, concept), cnt in agg.items():
        out.append({
            "doc_key": doc_key,
            "lang": lang,
            "year": year,
            "order_in_year": order_in_year,
            "docid": docid,
            "concept": concept,
            "category": cat_by_concept.get(concept, ""),
            "count": cnt
        })

    out.sort(key=lambda x: (x["lang"], x["year"] or 9999, x["order_in_year"] or 9999, x["docid"] or "", x["category"], x["concept"]))
    return out


# ----------------------------
# Concept label helpers
# ----------------------------
def build_concept_to_en_label(en_terms: List[Term]) -> Dict[str, str]:
    """
    concept -> one English label (first seen EN term for that concept)
    """
    m = {}
    for t in en_terms:
        if t.concept not in m and t.term:
            m[t.concept] = t.term
    return m


def build_concept_to_cn_label(cn_terms: List[Term]) -> Dict[str, str]:
    """
    concept -> one CN label (first seen CN term for that concept)
    """
    m = {}
    for t in cn_terms:
        if t.concept not in m and t.term:
            m[t.concept] = t.term
    return m


def build_concept_to_category(cn_terms: List[Term]) -> Dict[str, str]:
    """
    concept -> category (first seen)
    """
    m = {}
    for t in cn_terms:
        if t.concept not in m:
            m[t.concept] = t.category
    return m


def build_concept_to_cn_variants(cn_terms: List[Term]) -> Dict[str, str]:
    """
    concept -> joined CN term variants (unique)
    """
    tmp = defaultdict(list)
    for t in cn_terms:
        if t.term:
            tmp[t.concept].append(t.term)
    out = {}
    for c, lst in tmp.items():
        seen = []
        for x in lst:
            if x not in seen:
                seen.append(x)
        out[c] = " / ".join(seen)
    return out


# ----------------------------
# UI
# ----------------------------
st.title("Discourse Analyzer: analizė po vieną dokumentą (CN 简体, tab’ai)")
st.caption(
    "Įkeli dokumentus visus kartu, bet analizė ir peržiūra yra **po vieną dokumentą**: "
    "kiekvienas dokumentas turi savo tab’ą su metaduomenimis ir raktažodžių lentele."
)

with st.sidebar:
    st.header("1) Dokumentai")
    cn_files = st.file_uploader("CN originalai (.txt, .docx)", type=["txt", "docx", "doc"], accept_multiple_files=True)
    st.caption("Failo pavadinime turi būti metai ir DocID, pvz.: 2020_01_DOC123_CN.txt")

    st.header("2) Žodynai (CSV)")
    cn_terms_file = st.file_uploader("terms_cn.csv (concept, term, category)", type=["csv"])
    en_terms_file = st.file_uploader("terms_en.csv (concept, term, category) — ENG vertimui", type=["csv"])
    st.caption("ENG vertimas stulpelyje bus pirmas term iš terms_en.csv tam pačiam concept.")

    st.header("3) Nustatymai")
    show_zero = st.checkbox("Rodyti ir 0 (lentelėse)", value=False)

run = st.button("▶️ Analizuoti", type="primary")

if not run:
    st.info("Įkelk CN dokumentus + terms_cn.csv ir paspausk **Analizuoti**.")
    st.stop()

try:
    if not cn_files:
        st.warning("Įkelk bent 1 CN dokumentą.")
        st.stop()
    if not cn_terms_file:
        st.warning("Įkelk terms_cn.csv.")
        st.stop()

    # Load terms
    cn_terms_rows = load_csv(cn_terms_file)
    cn_terms = load_terms(cn_terms_rows, "CN")

    have_en_terms = bool(en_terms_file)
    en_terms = []
    if have_en_terms:
        en_terms_rows = load_csv(en_terms_file)
        en_terms = load_terms(en_terms_rows, "EN")

    # Ingest docs
    cn_docs = sort_docs(ingest_files(cn_files, "CN"))

    # Require docid for each file
    missing_cn = [d.filename for d in cn_docs if not d.docid]
    if missing_cn:
        st.error(
            "CN failams nepavyko nustatyti DocID iš pavadinimo. Reikia formato su ..._DOCID_CN...\n\n"
            "Pvz: 2020_01_DOC123_CN.txt\n\n"
            "Probleminiai: " + ", ".join(missing_cn)
        )
        st.stop()

    # Analyze CN docs with CN terms
    detail_cn = analyze_docs(cn_docs, cn_terms, "CN")
    profile_cn = doc_profile_concept(detail_cn)

    # Filter zeros if needed
    if not show_zero:
        profile_cn_view = [r for r in profile_cn if int(r["count"]) > 0]
    else:
        profile_cn_view = profile_cn

    # Build mappings for labels/variants/categories
    concept_to_en = build_concept_to_en_label(en_terms) if have_en_terms else {}
    concept_to_cn_label = build_concept_to_cn_label(cn_terms)
    concept_to_category = build_concept_to_category(cn_terms)
    concept_to_cn_variants = build_concept_to_cn_variants(cn_terms)

    # Index profile rows by doc_key
    rows_by_doc = defaultdict(list)
    for r in profile_cn_view:
        rows_by_doc[r["doc_key"]].append(r)

    st.subheader("Dokumentų analizė po vieną (CN)")
    st.write("Pasirink dokumento tab’ą — matysi dokumento metaduomenis ir raktažodžių (concept) lentelę.")

    # Create tabs for each doc
    tab_labels = []
    for d in cn_docs:
        y = d.year if d.year is not None else "????"
        did = d.docid or "NOID"
        tab_labels.append(f"{y} | {did}")

    doc_tabs = st.tabs(tab_labels)

    # Render each document
    for i, d in enumerate(cn_docs):
        dk = make_doc_key(d)
        with doc_tabs[i]:
            st.markdown("### Dokumento informacija")
            st.write({
                "Metai": d.year,
                "Eilė metuose (jei yra)": d.order_in_year,
                "DocID": d.docid,
                "Kalba": "CN (简体)",
                "Pilnas failo vardas": d.filename,
            })

            st.markdown("### Raktažodžių (concept) lentelė")
            per = rows_by_doc.get(dk, [])

            enriched = []
            for r in per:
                concept = r["concept"]
                enriched.append({
                    "CH term": concept_to_cn_label.get(concept, ""),
                    "vertimas ENG": concept_to_en.get(concept, ""),
                    "concept": concept,
                    "category": concept_to_category.get(concept, r.get("category", "")),
                    "count": int(r.get("count", 0)),
                    "CH term variants": concept_to_cn_variants.get(concept, ""),
                })

            enriched.sort(key=lambda x: x["count"], reverse=True)

            if not enriched:
                st.info("Šiame dokumente nerasta nė vieno raktažodžio (pagal tavo terms_cn.csv).")
            else:
                st.dataframe(enriched, use_container_width=True)

                st.download_button(
                    "⬇️ Atsisiųsti šio dokumento lentelę (CSV)",
                    data=to_csv_bytes(
                        enriched,
                        ["CH term", "vertimas ENG", "concept", "category", "count", "CH term variants"]
                    ),
                    file_name=f"{d.docid}_concept_table.csv",
                    mime="text/csv"
                )

    # Global summary
    st.divider()
    st.subheader("Bendra suvestinė (visi CN dokumentai)")
    st.write("Jei reikia, atsisiųsk visų dokumentų rezultatus viename faile.")

    all_enriched = []
    for r in profile_cn_view:
        c = r["concept"]
        all_enriched.append({
            "doc_key": r["doc_key"],
            "year": r["year"],
            "order_in_year": r["order_in_year"],
            "docid": r["docid"],
            "CH term": concept_to_cn_label.get(c, ""),
            "vertimas ENG": concept_to_en.get(c, ""),
            "concept": c,
            "category": concept_to_category.get(c, r.get("category", "")),
            "count": int(r.get("count", 0)),
            "CH term variants": concept_to_cn_variants.get(c, ""),
        })

    # sort for readability
    all_enriched.sort(key=lambda x: (x["year"] or 9999, x["order_in_year"] or 9999, x["docid"] or "", -x["count"], x["category"], x["concept"]))

    st.dataframe(all_enriched, use_container_width=True)

    st.download_button(
        "⬇️ Atsisiųsti visų dokumentų suvestinę (CSV)",
        data=to_csv_bytes(
            all_enriched,
            ["doc_key", "year", "order_in_year", "docid", "CH term", "vertimas ENG", "concept", "category", "count", "CH term variants"]
        ),
        file_name="all_documents_concept_profile.csv",
        mime="text/csv"
    )

    st.success("Baigta ✅")

except Exception as e:
    st.error(f"Klaida: {e}")
    st.stop()
