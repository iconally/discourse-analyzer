import re
import io
import csv
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict

import streamlit as st

try:
    import docx  # python-docx
except Exception:
    docx = None


st.set_page_config(
    page_title="Discourse Analyzer (CN 简体 ↔ EN) — v1.0",
    layout="wide"
)

# =========================
# Regex / naming rules
# =========================
YEAR_RE = re.compile(r"(20[0-2]\d)")
# expects ..._DOCID_CN.(txt|docx)
DOCID_CN_RE = re.compile(r"(?:^|_)([A-Za-z0-9\-]+)(?:_CN)(?:_|\.)")
ORDER_RE = re.compile(r"(20[0-2]\d)[_\-\.](\d{1,2})(?=[_\-\.])")


# =========================
# Models
# =========================
@dataclass
class Term:
    concept: str
    term: str
    category: str


@dataclass
class Doc:
    lang: str
    filename: str
    year: Optional[int]
    order_in_year: Optional[int]
    docid: str
    text: str


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


def detect_docid_cn(filename: str) -> Optional[str]:
    m = DOCID_CN_RE.search(filename)
    return m.group(1) if m else None


def infer_ch_title_from_filename(filename: str) -> str:
    """
    Tries to create a human-friendly CN doc title from filename by removing:
    - year
    - order
    - docid
    - _CN
    - extension
    Then converts underscores/dashes to spaces.
    """
    base = filename
    base = re.sub(r"\.(txt|docx)$", "", base, flags=re.IGNORECASE)
    base = re.sub(r"_CN$", "", base, flags=re.IGNORECASE)

    # remove leading year/order if present
    base = re.sub(r"^(20[0-2]\d)[_\-\.](\d{1,2})[_\-\.]", "", base)
    base = re.sub(r"^(20[0-2]\d)[_\-\.]", "", base)

    # remove docid token if it appears as a standalone underscore-separated chunk
    # Example: 2020_01_DOC123_CN -> DOC123 removed
    base = re.sub(r"(?:^|_)([A-Za-z0-9\-]+)$", r"\1", base)  # no-op safety
    base = re.sub(r"(?:^|_)([A-Za-z0-9\-]+)(?=_|$)", lambda m: m.group(0), base)

    # If we can detect docid, remove it
    docid = detect_docid_cn(filename)
    if docid:
        base = re.sub(rf"(?:^|_){re.escape(docid)}(?:_|$)", "_", base)

    # cleanup underscores/dashes
    base = base.replace("__", "_").strip("_- .")
    base = re.sub(r"[_\-\.]+", " ", base).strip()
    return base


# =========================
# CSV helpers (no pandas)
# =========================
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


# =========================
# Terms loading
# =========================
def load_terms(rows: List[dict], lang: str) -> List[Term]:
    required = {"concept", "term", "category"}
    if not rows:
        raise ValueError(f"{lang} terms CSV tuščias.")
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"{lang} terms CSV trūksta stulpelių: {', '.join(sorted(missing))}")
    out: List[Term] = []
    for r in rows:
        c = r.get("concept", "").strip()
        t = r.get("term", "").strip()
        cat = r.get("category", "").strip()
        if c and t:
            out.append(Term(concept=c, term=t, category=cat))
    return out


def build_cn_patterns(terms_cn: List[Term]) -> Dict[Tuple[str, str], re.Pattern]:
    """
    Exact substring match for CN (no segmentation).
    """
    return {(t.concept, t.term): re.compile(re.escape(t.term)) for t in terms_cn}


def build_concept_maps(terms_cn: List[Term], terms_en: List[Term]) -> Tuple[
    Dict[str, List[str]], Dict[str, str], Dict[str, str], Dict[str, str]
]:
    """
    Returns:
    - concept_to_cn_variants: concept -> [CN term variants...]
    - concept_to_cn_primary: concept -> first CN term (label fallback)
    - concept_to_category: concept -> first category
    - concept_to_en_label: concept -> first EN term (label)
    """
    concept_to_cn_variants: Dict[str, List[str]] = defaultdict(list)
    concept_to_cn_primary: Dict[str, str] = {}
    concept_to_category: Dict[str, str] = {}
    concept_to_en_label: Dict[str, str] = {}

    for t in terms_cn:
        concept_to_cn_variants[t.concept].append(t.term)
        if t.concept not in concept_to_cn_primary:
            concept_to_cn_primary[t.concept] = t.term
        if t.concept not in concept_to_category:
            concept_to_category[t.concept] = t.category

    for t in terms_en:
        if t.concept not in concept_to_en_label and t.term:
            concept_to_en_label[t.concept] = t.term

    # de-duplicate variants, preserve order
    for c, lst in list(concept_to_cn_variants.items()):
        seen = []
        for x in lst:
            if x not in seen:
                seen.append(x)
        concept_to_cn_variants[c] = seen

    return concept_to_cn_variants, concept_to_cn_primary, concept_to_category, concept_to_en_label


# =========================
# Docs ingest
# =========================
def ingest_cn_docs(files) -> List[Doc]:
    docs: List[Doc] = []
    for f in files:
        name = f.name
        year = detect_year(name)
        order = detect_order_in_year(name)
        docid = detect_docid_cn(name)

        if not docid:
            raise ValueError(
                f"Nepavyko nustatyti DocID iš CN failo pavadinimo: {name}\n"
                f"Reikia formato su ..._DOCID_CN..., pvz: 2020_01_DOC123_CN.txt"
            )

        ext = name.lower().split(".")[-1]
        if ext == "txt":
            text = read_txt(f)
        elif ext == "docx":
            text = read_docx(f)
        elif ext == "doc":
            raise ValueError(f"Failas {name} yra .doc. Konvertuok į .docx.")
        else:
            raise ValueError(f"Nepalaikomas formatas: {name}. Naudok .txt arba .docx.")

        docs.append(Doc(lang="CN", filename=name, year=year, order_in_year=order, docid=docid, text=text))

    docs.sort(key=lambda d: (d.year or 9999, d.order_in_year or 9999, d.filename))
    return docs


# =========================
# Analysis core
# =========================
def analyze_cn_docs_termlevel(
    docs: List[Doc],
    terms_cn: List[Term],
) -> Tuple[
    Dict[str, Dict[str, Counter]],   # per_doc_concept_term_counts: docid -> concept -> Counter(term -> count)
    Dict[str, Dict[str, int]]        # per_doc_concept_total: docid -> concept -> total_count
]:
    patterns = build_cn_patterns(terms_cn)

    per_doc_concept_term_counts: Dict[str, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    per_doc_concept_total: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for d in docs:
        for t in terms_cn:
            pat = patterns[(t.concept, t.term)]
            cnt = len(pat.findall(d.text))
            if cnt > 0:
                per_doc_concept_term_counts[d.docid][t.concept][t.term] += cnt
                per_doc_concept_total[d.docid][t.concept] += cnt

    return per_doc_concept_term_counts, per_doc_concept_total


def build_doc_table_rows(
    doc: Doc,
    all_concepts: List[str],
    per_doc_concept_term_counts: Dict[str, Dict[str, Counter]],
    per_doc_concept_total: Dict[str, Dict[str, int]],
    concept_to_cn_variants: Dict[str, List[str]],
    concept_to_cn_primary: Dict[str, str],
    concept_to_category: Dict[str, str],
    concept_to_en_label: Dict[str, str],
    show_zero: bool
) -> List[dict]:
    """
    Creates the per-document table with required columns:
    CH term | vertimas ENG | concept | category | count | CH term variants
    """
    rows: List[dict] = []
    docid = doc.docid
    concept_totals = per_doc_concept_total.get(docid, {})
    concept_terms = per_doc_concept_term_counts.get(docid, {})

    for concept in all_concepts:
        total = int(concept_totals.get(concept, 0))
        if (not show_zero) and total == 0:
            continue

        # Dominant CH term in THIS document:
        # - if doc has counts for variants, pick most_common
        # - else fallback to primary CN label
        dominant = concept_to_cn_primary.get(concept, "")
        ctr = concept_terms.get(concept)
        if ctr:
            dominant = ctr.most_common(1)[0][0]

        rows.append({
            "CH term": dominant,
            "vertimas ENG": concept_to_en_label.get(concept, ""),
            "concept": concept,
            "category": concept_to_category.get(concept, ""),
            "count": total,
            "CH term variants": " / ".join(concept_to_cn_variants.get(concept, [])),
        })

    rows.sort(key=lambda r: r["count"], reverse=True)
    return rows


# =========================
# UI
# =========================
st.title("Discourse Analyzer (CN 简体 ↔ EN) — v1.0")
st.caption(
    "V1.0: dokumentai analizuojami po vieną (tab’ai). "
    "CH term = dominuojantis CN term variantas dokumente; count = concept suma per variants."
)

with st.sidebar:
    st.header("1) Dokumentai")
    cn_files = st.file_uploader(
        "CN originalai (.txt, .docx)",
        type=["txt", "docx", "doc"],
        accept_multiple_files=True
    )
    st.caption("Rekomenduojamas pavadinimas: 2020_01_DOC123_CN.txt (arba .docx)")

    st.header("2) Žodynai (CSV)")
    cn_terms_file = st.file_uploader("terms_cn.csv (concept, term, category)", type=["csv"])
    en_terms_file = st.file_uploader("terms_en.csv (concept, term, category) — ENG vertimams", type=["csv"])
    st.caption("ENG vertimas lentelėje: pirmas term iš terms_en.csv tam pačiam concept.")

    st.header("3) Nustatymai")
    show_zero = st.checkbox("Rodyti ir concept su count=0", value=False)

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
    terms_cn = load_terms(load_csv(cn_terms_file), "CN")
    terms_en = load_terms(load_csv(en_terms_file), "EN") if en_terms_file else []

    # Maps
    concept_to_cn_variants, concept_to_cn_primary, concept_to_category, concept_to_en_label = build_concept_maps(terms_cn, terms_en)
    all_concepts = sorted(concept_to_category.keys())

    # Ingest docs
    docs_cn = ingest_cn_docs(cn_files)

    # Analyze
    per_doc_concept_term_counts, per_doc_concept_total = analyze_cn_docs_termlevel(docs_cn, terms_cn)

    # Tabs per document
    st.subheader("Dokumentų analizė po vieną (CN)")

    tab_labels = []
    for d in docs_cn:
        y = d.year if d.year is not None else "????"
        tab_labels.append(f"{y} | {d.docid}")
    tabs = st.tabs(tab_labels)

    # Collect global summary (all docs)
    all_docs_rows: List[dict] = []

    for d, tab in zip(docs_cn, tabs):
        with tab:
            st.markdown("### Dokumento informacija")
            st.write({
                "Metai": d.year,
                "Eilė metuose (jei yra)": d.order_in_year,
                "DocID": d.docid,
                "Kalba": "CN (简体)",
                "CH pavadinimas (iš failo vardo)": infer_ch_title_from_filename(d.filename),
                "Pilnas failo vardas": d.filename,
            })

            st.markdown("### Raktažodžių (concept) lentelė")
            rows = build_doc_table_rows(
                doc=d,
                all_concepts=all_concepts,
                per_doc_concept_term_counts=per_doc_concept_term_counts,
                per_doc_concept_total=per_doc_concept_total,
                concept_to_cn_variants=concept_to_cn_variants,
                concept_to_cn_primary=concept_to_cn_primary,
                concept_to_category=concept_to_category,
                concept_to_en_label=concept_to_en_label,
                show_zero=show_zero
            )

            if not rows:
                st.info("Šiame dokumente nerasta raktažodžių pagal tavo terms_cn.csv (arba visi count=0).")
            else:
                st.dataframe(rows, use_container_width=True)

                st.download_button(
                    "⬇️ Atsisiųsti šio dokumento lentelę (CSV)",
                    data=to_csv_bytes(
                        rows,
                        ["CH term", "vertimas ENG", "concept", "category", "count", "CH term variants"]
                    ),
                    file_name=f"{d.docid}_concept_table.csv",
                    mime="text/csv"
                )

            # Add to global summary
            for r in rows:
                all_docs_rows.append({
                    "year": d.year,
                    "order_in_year": d.order_in_year,
                    "docid": d.docid,
                    "filename": d.filename,
                    "CH term": r["CH term"],
                    "vertimas ENG": r["vertimas ENG"],
                    "concept": r["concept"],
                    "category": r["category"],
                    "count": r["count"],
                    "CH term variants": r["CH term variants"],
                })

    # Global summary
    st.divider()
    st.subheader("Bendra suvestinė (visi CN dokumentai)")

    if all_docs_rows:
        all_docs_rows.sort(key=lambda x: (
            x["year"] if x["year"] is not None else 9999,
            x["order_in_year"] if x["order_in_year"] is not None else 9999,
            x["docid"],
            -int(x["count"]),
            x["category"],
            x["concept"],
        ))
        st.dataframe(all_docs_rows, use_container_width=True)

        st.download_button(
            "⬇️ Atsisiųsti visų dokumentų suvestinę (CSV)",
            data=to_csv_bytes(
                all_docs_rows,
                ["year", "order_in_year", "docid", "filename", "CH term", "vertimas ENG", "concept", "category", "count", "CH term variants"]
            ),
            file_name="all_documents_concept_profile.csv",
            mime="text/csv"
        )
    else:
        st.info("Nėra duomenų suvestinei.")

    st.success("V1.0 baigta ✅")

except Exception as e:
    st.error(f"Klaida: {e}")
    st.stop()
