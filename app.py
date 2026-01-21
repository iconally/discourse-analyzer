# app.py
# Version: V1.5

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

# ---- jieba (OPTIONAL) ----
try:
    import jieba  # type: ignore
    JIEBA_AVAILABLE = True
except Exception:
    JIEBA_AVAILABLE = False


# -----------------------------
# Config
# -----------------------------
APP_VERSION = "V1.5"

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
        raise RuntimeError("python-docx is not available.")
    bio = io.BytesIO(file_bytes)
    doc = Document(bio)
    return "\n".join(p.text for p in doc.paragraphs if p.text)


def load_terms_csv(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        with open(DEFAULT_TERMS_PATH, "rb") as f:
            raw = f.read()
    else:
        raw = uploaded_file.getvalue()

    text = read_txt(raw)
    df = pd.read_csv(io.StringIO(text), sep=";", dtype=str, keep_default_na=False)
    df.columns = [c.strip().lower() for c in df.columns]

    required = ["concept", "term", "pinyin", "translation", "category"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"terms_cn.csv missing columns: {missing}")

    for c in required:
        df[c] = df[c].astype(str).str.strip()

    df = df[df["term"].str.len() > 0]
    df = df.drop_duplicates(subset=required).reset_index(drop=True)
    return df


def count_substring_occurrences(text: str, term: str) -> int:
    if not term:
        return 0
    return len(re.findall(re.escape(term), text))


# ---------- jieba helpers ----------
def init_jieba_with_terms(terms: pd.Series) -> None:
    if not JIEBA_AVAILABLE:
        return
    for t in terms.dropna().astype(str):
        if t.strip():
            jieba.add_word(t.strip())


def build_token_counter(text: str) -> Dict[str, int]:
    if not JIEBA_AVAILABLE:
        return {}
    tokens = jieba.cut(text, HMM=False)
    counter: Dict[str, int] = {}
    for t in tokens:
        counter[t] = counter.get(t, 0) + 1
    return counter


# ---------- analysis ----------
def analyze_text(text: str, terms_df: pd.DataFrame, mode: str = "substring") -> pd.DataFrame:
    text = normalize_text(text)

    token_counter: Dict[str, int] = {}
    if mode == "jieba" and JIEBA_AVAILABLE:
        token_counter = build_token_counter(text)

    rows = []
    for _, r in terms_df.iterrows():
        term = safe_str(r["term"])

        if mode == "jieba" and JIEBA_AVAILABLE:
            cnt = int(token_counter.get(term, 0))
        else:
            cnt = count_substring_occurrences(text, term)

        if cnt > 0:
            rows.append(
                {
                    "CH term": term,
                    "Pinyin": safe_str(r["pinyin"]),
                    "ENG translation": safe_str(r["translation"]),
                    "Concept": safe_str(r["concept"]),
                    "Category": safe_str(r["category"]),
                    "Count": cnt,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["CH term", "Pinyin", "ENG translation", "Concept", "Category", "Count"])

    df = pd.DataFrame(rows)
    return (
        df.groupby(["CH term", "Pinyin", "ENG translation", "Concept", "Category"], as_index=False)["Count"]
        .sum()
        .sort_values(["Category", "Concept", "Count"], ascending=[True, True, False])
        .reset_index(drop=True)
    )


def category_summary(term_hits: pd.DataFrame, terms_df: pd.DataFrame) -> pd.DataFrame:
    if term_hits.empty:
        return pd.DataFrame(columns=["Category", "Unique terms", "Total count", "Coverage", "Share", "Dict terms"])

    dict_totals = terms_df.groupby("category")["term"].nunique().reset_index(name="Dict terms")
    dict_totals = dict_totals.rename(columns={"category": "Category"})

    detected = (
        term_hits.groupby("Category")
        .agg(**{"Unique terms": ("CH term", "nunique"), "Total count": ("Count", "sum")})
        .reset_index()
    )

    out = detected.merge(dict_totals, on="Category", how="left").fillna(0)
    out["Coverage"] = out["Unique terms"] / out["Dict terms"].replace(0, pd.NA)
    total_all = out["Total count"].sum()
    out["Share"] = out["Total count"] / total_all if total_all else 0

    out["Coverage"] = out["Coverage"].fillna(0).map(lambda x: f"{x:.1%}")
    out["Share"] = out["Share"].map(lambda x: f"{x:.1%}")

    return out.sort_values("Total count", ascending=False).reset_index(drop=True)


def concept_summary(term_hits: pd.DataFrame) -> pd.DataFrame:
    if term_hits.empty:
        return pd.DataFrame(columns=["Concept", "Category", "Unique terms", "Total count"])

    return (
        term_hits.groupby("Concept")
        .agg(
            **{
                "Total count": ("Count", "sum"),
                "Unique terms": ("CH term", "nunique"),
                "Category": ("Category", lambda s: ", ".join(sorted(set(s)))),
            }
        )
        .reset_index()
        .sort_values("Total count", ascending=False)
    )


def get_file_text(uploaded) -> Tuple[Optional[str], Optional[str]]:
    name = uploaded.name.lower()
    data = uploaded.getvalue()

    if name.endswith(".txt"):
        return normalize_text(read_txt(data)), None
    if name.endswith(".docx"):
        try:
            return normalize_text(read_docx(data)), None
        except Exception as e:
            return None, str(e)
    if name.endswith(".doc"):
        return None, "DOC formatas nepalaikomas. Konvertuok į DOCX."
    return None, "Palaikomi formatai: TXT, DOCX."


def parse_meta_from_filename(filename: str) -> Dict[str, str]:
    stem = re.sub(r"\.[^.]+$", "", filename)

    m = re.search(r"(?<!\d)(19|20)\d{2}(?!\d)", stem)
    year = m.group(0) if m else ""

    title = stem
    if year:
        title = re.sub(year, "", title)

    title = re.sub(r"^[\s\-_–—:：]+", "", title)
    title = re.sub(r"^(0?[1-9]|1[0-2])[\s\-_–—:：]+", "", title)
    title = re.sub(r"[\s\-_–—:：]+CN$", "", title, flags=re.IGNORECASE)
    title = re.sub(r"[\s\-_–—:：]{2,}", " ", title).strip()

    return {"year": year, "title_cn": title}


def meta_key(filename: str) -> str:
    return f"doc_meta::{filename}"


def ensure_doc_meta(filename: str):
    if meta_key(filename) not in st.session_state:
        st.session_state[meta_key(filename)] = {"year": "", "title_cn": ""}


# -----------------------------
# UI
# -----------------------------
st.title(f"Discourse Analyzer (CN religious-coded terms) — {APP_VERSION}")

with st.sidebar:
    st.header("Žodynas (terms_cn.csv)")
    terms_upload = st.file_uploader("Įkelk terms_cn.csv", type=["csv"])
    st.divider()

    st.header("Dokumentai")
    files = st.file_uploader("Įkelk dokumentus (TXT / DOCX)", type=["txt", "docx", "doc"], accept_multiple_files=True)

    st.divider()
    st.header("Counting mode")
    count_mode = st.selectbox("Skaičiavimo režimas", ["substring", "jieba (token)"], index=0)

    if count_mode.startswith("jieba") and not JIEBA_AVAILABLE:
        st.warning("jieba nepasiekiama — naudojamas substring režimas.")


# -----------------------------
# Load dictionary
# -----------------------------
terms_df = load_terms_csv(terms_upload)
st.success(f"Žodynas užkrautas: {len(terms_df)} eilučių, {terms_df['term'].nunique()} unikalių termų.")

if count_mode.startswith("jieba") and JIEBA_AVAILABLE:
    init_jieba_with_terms(terms_df["term"])

if not files:
    st.stop()

docs = []
for f in files:
    text, err = get_file_text(f)
    if not err:
        docs.append((f.name, text))

tabs = st.tabs([fn for fn, _ in docs])

for (filename, text), tab in zip(docs, tabs):
    ensure_doc_meta(filename)

    with tab:
        meta = parse_meta_from_filename(filename)
        st.session_state[meta_key(filename)].update(meta)

        st.subheader("Document info")
        st.write(f"**Metai:** {meta['year']}")
        st.write(f"**Pavadinimas (CN):** {meta['title_cn']}")

        st.divider()
        st.subheader("Analizė")

        mode_internal = "jieba" if count_mode.startswith("jieba") and JIEBA_AVAILABLE else "substring"
        term_hits = analyze_text(text, terms_df, mode=mode_internal)

        cat_sum = category_summary(term_hits, terms_df)
        conc_sum = concept_summary(term_hits)

        st.markdown("### 1) Term detail")
        st.dataframe(term_hits)

        st.markdown("### 2) Concept summary")
        st.dataframe(conc_sum)

        st.markdown("### 3) Category summary")
        st.dataframe(cat_sum)

        if not term_hits.empty:
            st.divider()
            st.subheader("Downloads")
            st.download_button(
                "Download term detail (CSV)",
                term_hits.to_csv(index=False).encode("utf-8-sig"),
                f"{filename}_term_detail.csv",
            )
