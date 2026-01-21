# app.py
import io
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

try:
    from docx import Document  # python-docx
    DOCX_AVAILABLE = True
except Exception:
    DOCX_AVAILABLE = False


# -----------------------------
# Config
# -----------------------------
st.set_page_config(
    page_title="Discourse Analyzer",
    layout="wide",
)

DEFAULT_TERMS_PATH = "terms_cn.csv"  # keep in repo root


# -----------------------------
# Helpers
# -----------------------------
def safe_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    return str(x)


def normalize_text(text: str) -> str:
    # Minimal normalization: unify newlines and strip
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def read_txt(file_bytes: bytes) -> str:
    # Try utf-8 first, fallback to gb18030, then latin-1
    for enc in ("utf-8-sig", "utf-8", "gb18030", "big5", "latin-1"):
        try:
            return file_bytes.decode(enc)
        except Exception:
            continue
    # last resort
    return file_bytes.decode("utf-8", errors="ignore")


def read_docx(file_bytes: bytes) -> str:
    if not DOCX_AVAILABLE:
        raise RuntimeError("python-docx is not available in this environment.")
    bio = io.BytesIO(file_bytes)
    doc = Document(bio)
    parts = []
    for p in doc.paragraphs:
        if p.text:
            parts.append(p.text)
    return "\n".join(parts)


def load_terms_csv(uploaded_file) -> pd.DataFrame:
    """
    Expected columns (semicolon separated):
    concept;term;pinyin;translation;category
    """
    if uploaded_file is None:
        with open(DEFAULT_TERMS_PATH, "rb") as f:
            raw = f.read()
    else:
        raw = uploaded_file.getvalue()

    # Read with ; separator (as you decided)
    text = read_txt(raw)
    df = pd.read_csv(io.StringIO(text), sep=";", dtype=str, keep_default_na=False)

    # Normalize headers and required columns
    df.columns = [c.strip().lower() for c in df.columns]

    required = ["concept", "term", "pinyin", "translation", "category"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"terms_cn.csv is missing required columns: {missing}. "
            f"Expected: concept;term;pinyin;translation;category"
        )

    # Clean whitespace
    for c in required:
        df[c] = df[c].astype(str).map(lambda s: s.strip())

    # Drop empty terms
    df = df[df["term"].map(lambda x: len(x) > 0)].copy()

    # Deduplicate exact rows
    df = df.drop_duplicates(subset=["concept", "term", "pinyin", "translation", "category"]).reset_index(drop=True)

    return df


def count_substring_occurrences(text: str, term: str) -> int:
    """
    Count non-overlapping occurrences of `term` in `text`.
    For Chinese terms (multi-char), this is usually fine.
    """
    if not term:
        return 0
    # Escape regex special chars
    pattern = re.escape(term)
    return len(re.findall(pattern, text))


def analyze_text(text: str, terms_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a term-level results dataframe with columns:
    term, pinyin, translation, concept, category, count
    """
    text = normalize_text(text)

    rows = []
    # Iterate terms; for large dictionaries you can optimize later, but this is stable.
    for _, r in terms_df.iterrows():
        term = safe_str(r["term"])
        cnt = count_substring_occurrences(text, term)
        if cnt > 0:
            rows.append(
                {
                    "CH term": term,
                    "Pinyin": safe_str(r["pinyin"]),
                    "ENG translation": safe_str(r["translation"]),
                    "Concept": safe_str(r["concept"]),
                    "Category": safe_str(r["category"]),
                    "Count": int(cnt),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["CH term", "Pinyin", "ENG translation", "Concept", "Category", "Count"])

    df = pd.DataFrame(rows)

    # Combine duplicates just in case (same term may appear multiple times in csv)
    df = (
        df.groupby(["CH term", "Pinyin", "ENG translation", "Concept", "Category"], as_index=False)["Count"]
        .sum()
        .sort_values(["Category", "Concept", "Count"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    return df


def category_summary(term_hits: pd.DataFrame, terms_df: pd.DataFrame) -> pd.DataFrame:
    """
    Category summary per document:
    - unique_terms_detected
    - total_count
    - coverage (detected unique terms / total unique terms in dictionary for category)
    - share (category_total_count / total_count_all_categories)
    """
    if term_hits.empty:
        return pd.DataFrame(columns=["Category", "Unique terms", "Total count", "Coverage", "Share"])

    # Totals in dictionary
    dict_totals = (
        terms_df.groupby("category")["term"]
        .nunique()
        .rename("Dict terms")
        .reset_index()
        .rename(columns={"category": "Category"})
    )

    # Detected per category
    detected = (
        term_hits.groupby("Category")
        .agg(**{"Unique terms": ("CH term", "nunique"), "Total count": ("Count", "sum")})
        .reset_index()
    )

    out = detected.merge(dict_totals, on="Category", how="left")
    out["Dict terms"] = out["Dict terms"].fillna(0).astype(int)

    out["Coverage"] = out.apply(
        lambda r: (r["Unique terms"] / r["Dict terms"]) if r["Dict terms"] > 0 else 0.0,
        axis=1,
    )

    total_all = out["Total count"].sum()
    out["Share"] = out["Total count"].apply(lambda x: (x / total_all) if total_all > 0 else 0.0)

    # Formatting helpers (keep raw numeric too)
    out = out.sort_values(["Total count", "Unique terms"], ascending=[False, False]).reset_index(drop=True)

    # Human-readable
    out["Coverage"] = out["Coverage"].map(lambda x: f"{x:.1%}")
    out["Share"] = out["Share"].map(lambda x: f"{x:.1%}")

    # Keep Dict terms visible for methodological clarity
    out = out[["Category", "Unique terms", "Total count", "Coverage", "Share", "Dict terms"]]
    return out


def concept_summary(term_hits: pd.DataFrame) -> pd.DataFrame:
    """
    Concept-level summary (grouped by Concept).
    Shows:
    - Total count (sum across terms)
    - Unique terms detected
    - Category (if multiple, show comma-joined unique list)
    """
    if term_hits.empty:
        return pd.DataFrame(columns=["Concept", "Category", "Unique terms", "Total count"])

    tmp = term_hits.copy()

    grouped = (
        tmp.groupby("Concept")
        .agg(
            **{
                "Total count": ("Count", "sum"),
                "Unique terms": ("CH term", "nunique"),
                "Category": ("Category", lambda s: ", ".join(sorted(set([safe_str(x) for x in s if safe_str(x)])))),
            }
        )
        .reset_index()
        .sort_values(["Total count", "Unique terms"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return grouped[["Concept", "Category", "Unique terms", "Total count"]]


def get_file_text(uploaded) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (text, error_message)
    """
    name = uploaded.name
    data = uploaded.getvalue()
    lower = name.lower()

    if lower.endswith(".txt"):
        return normalize_text(read_txt(data)), None

    if lower.endswith(".docx"):
        try:
            return normalize_text(read_docx(data)), None
        except Exception as e:
            return None, f"Nepavyko perskaityti DOCX: {e}"

    if lower.endswith(".doc"):
        return None, (
            "DOC formatas dažnai nėra patikimai skaitomas be papildomų serverio įrankių. "
            "Rekomendacija: išsaugok kaip DOCX arba TXT ir įkelk iš naujo."
        )

    return None, "Palaikomi formatai: .txt ir .docx (DOC – konvertuoti į DOCX)."


@dataclass
class DocMeta:
    year: str = ""
    title_cn: str = ""


def meta_key(filename: str) -> str:
    return f"doc_meta::{filename}"


def ensure_doc_meta(filename: str):
    k = meta_key(filename)
    if k not in st.session_state:
        st.session_state[k] = {"year": "", "title_cn": ""}


# -----------------------------
# UI
# -----------------------------
st.title("Discourse Analyzer (CN religious-coded terms)")

with st.sidebar:
    st.header("Žodynas (terms_cn.csv)")
    st.caption("CSV formatas: concept;term;pinyin;translation;category")
    terms_upload = st.file_uploader("Įkelk terms_cn.csv (nebūtina, jei yra repo)", type=["csv"])
    st.divider()

    st.header("Dokumentai")
    files = st.file_uploader(
        "Įkelk dokumentus (TXT / DOCX)",
        type=["txt", "docx", "doc"],
        accept_multiple_files=True,
    )
    st.caption("DOC rekomenduojama konvertuoti į DOCX arba TXT.")

    st.divider()
    st.header("Rodymas / filtrai")
    show_zero_rows = st.checkbox("Rodyti termus su 0 (nerekomenduojama)", value=False, disabled=True)
    # Paliekam vietą ateičiai (v2): substring vs token
    st.caption("V2 idėja: substring vs token (jieba/pkuseg) — kol kas substring.")


# Load dictionary
try:
    terms_df_raw = load_terms_csv(terms_upload)
except Exception as e:
    st.error(f"Klaida skaitant terms_cn.csv: {e}")
    st.stop()

# Normalize dictionary column casing for internal use
terms_df = terms_df_raw.copy()
terms_df["concept"] = terms_df["concept"].astype(str)
terms_df["term"] = terms_df["term"].astype(str)
terms_df["pinyin"] = terms_df["pinyin"].astype(str)
terms_df["translation"] = terms_df["translation"].astype(str)
terms_df["category"] = terms_df["category"].astype(str)

st.success(f"Žodynas užkrautas: {len(terms_df):,} eilučių, {terms_df['term'].nunique():,} unikalių termų.")

if not files:
    st.info("Įkelk bent vieną dokumentą (TXT/DOCX), kad pamatytum analizę.")
    st.stop()

# Read all docs first and keep only valid
docs: List[Tuple[str, str]] = []  # (filename, text)
read_errors: Dict[str, str] = {}

for f in files:
    text, err = get_file_text(f)
    if err:
        read_errors[f.name] = err
    else:
        docs.append((f.name, text))

if read_errors:
    with st.expander("Dokumentų skaitymo problemos", expanded=True):
        for fn, msg in read_errors.items():
            st.warning(f"**{fn}**: {msg}")

if not docs:
    st.error("Nė vieno dokumento nepavyko nuskaityti. Įkelk TXT arba DOCX.")
    st.stop()

# Tabs per document
tab_names = [fn for fn, _ in docs]
tabs = st.tabs(tab_names)

for (filename, text), tab in zip(docs, tabs):
    ensure_doc_meta(filename)

    with tab:
        # -----------------------------
        # Document info block
        # -----------------------------
        st.subheader("Document info")

        col1, col2, col3 = st.columns([1, 2, 2])
        with col1:
            year_val = st.text_input(
                "Metai (Year)",
                value=st.session_state[meta_key(filename)]["year"],
                key=f"year::{filename}",
                placeholder="pvz. 2017",
            )
        with col2:
            title_cn_val = st.text_input(
                "Pavadinimas (CN)",
                value=st.session_state[meta_key(filename)]["title_cn"],
                key=f"title_cn::{filename}",
                placeholder="pvz. 新一代人工智能发展规划",
            )
        with col3:
            st.text_input("Pilnas failo pavadinimas", value=filename, disabled=True)

        # Persist
        st.session_state[meta_key(filename)]["year"] = year_val
        st.session_state[meta_key(filename)]["title_cn"] = title_cn_val

        # Optional small stats
        with st.expander("Teksto statistika", expanded=False):
            st.write(
                {
                    "Characters": len(text),
                    "Lines": text.count("\n") + 1 if text else 0,
                }
            )

        st.divider()

        # -----------------------------
        # Analysis
        # -----------------------------
        st.subheader("Analizė")

        term_hits = analyze_text(text, terms_df)

        total_hits = int(term_hits["Count"].sum()) if not term_hits.empty else 0
        unique_terms = int(term_hits["CH term"].nunique()) if not term_hits.empty else 0
        unique_concepts = int(term_hits["Concept"].nunique()) if not term_hits.empty else 0
        unique_categories = int(term_hits["Category"].nunique()) if not term_hits.empty else 0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total matches", total_hits)
        m2.metric("Unique terms", unique_terms)
        m3.metric("Unique concepts", unique_concepts)
        m4.metric("Categories hit", unique_categories)

        # Prepare summaries (needed also for downloads below)
        cat_sum = category_summary(term_hits, terms_df)
        conc_sum = concept_summary(term_hits)

        # -----------------------------
        # Term detail (FIRST) + sorting by Count desc + index from 1
        # -----------------------------
        st.markdown("### 1) Term detail")
        if term_hits.empty:
            st.info("Nėra termų detalių (nes nėra hitų).")
        else:
            term_hits_view = term_hits.sort_values(["Count"], ascending=[False]).reset_index(drop=True)
            term_hits_view.index = range(1, len(term_hits_view) + 1)
            st.dataframe(term_hits_view, width="stretch")

            # Downloads
            cdl1, cdl2 = st.columns(2)
            with cdl1:
                st.download_button(
                    "Download term detail (CSV)",
                    data=term_hits.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"{filename}_term_detail.csv",
                    mime="text/csv",
                )
            with cdl2:
                # Build a single CSV with category+concept summary for convenience
                summary_pack = {
                    "category_summary": cat_sum,
                    "concept_summary": conc_sum,
                }
                # Write a simple multi-section CSV
                out = io.StringIO()
                out.write("=== CATEGORY SUMMARY ===\n")
                cat_sum.to_csv(out, index=False)
                out.write("\n=== CONCEPT SUMMARY ===\n")
                conc_sum.to_csv(out, index=False)

                st.download_button(
                    "Download summaries (CSV)",
                    data=out.getvalue().encode("utf-8-sig"),
                    file_name=f"{filename}_summaries.csv",
                    mime="text/csv",
                )

        # -----------------------------
        # Concept summary (SECOND) + sorting by Total count desc + index from 1
        # -----------------------------
        st.markdown("### 2) Concept summary")
        if conc_sum.empty:
            st.info("Nėra concept rezultatų (nes nėra termų).")
        else:
            conc_sum_view = conc_sum.sort_values(["Total count"], ascending=[False]).reset_index(drop=True)
            conc_sum_view.index = range(1, len(conc_sum_view) + 1)
            st.dataframe(conc_sum_view, width="stretch")

        # -----------------------------
        # Category summary (THIRD) + sorting by Total count desc + index from 1
        # -----------------------------
        st.markdown("### 3) Category summary")
        if cat_sum.empty:
            st.info("Šiame dokumente nerasta nė vieno termino iš žodyno.")
        else:
            cat_sum_view = cat_sum.sort_values(["Total count"], ascending=[False]).reset_index(drop=True)
            cat_sum_view.index = range(1, len(cat_sum_view) + 1)
            st.dataframe(cat_sum_view, width="stretch")
