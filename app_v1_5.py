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

try:
    import jieba
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
    pattern = re.escape(term)
    return len(re.findall(pattern, text))


def build_token_counter(text: str, mode: str) -> Optional[Dict[str, int]]:
    """
    mode:
      - jieba_precise: jieba.cut (precise)
      - jieba_search: jieba.cut_for_search
    Returns dict token -> count, or None if jieba unavailable.
    """
    if not JIEBA_AVAILABLE:
        return None

    if mode == "jieba_search":
        tokens = jieba.cut_for_search(text)
    else:
        tokens = jieba.cut(text, cut_all=False)

    counter: Dict[str, int] = {}
    for t in tokens:
        t = safe_str(t).strip()
        if not t:
            continue
        counter[t] = counter.get(t, 0) + 1
    return counter


def analyze_text(text: str, terms_df: pd.DataFrame, match_mode: str) -> pd.DataFrame:
    """
    Returns a term-level results dataframe with columns:
    term, pinyin, translation, concept, category, count

    match_mode:
      - substring
      - jieba_precise (token matching)
      - jieba_search   (token matching)
      - hybrid (default): substring for len>=2, jieba for single-char terms
    """
    text = normalize_text(text)

    token_counter_precise = None
    token_counter_search = None

    if match_mode in ("jieba_precise", "hybrid") and JIEBA_AVAILABLE:
        token_counter_precise = build_token_counter(text, "jieba_precise")
    if match_mode == "jieba_search" and JIEBA_AVAILABLE:
        token_counter_search = build_token_counter(text, "jieba_search")

    rows = []
    for _, r in terms_df.iterrows():
        term = safe_str(r["term"])
        if not term:
            continue

        cnt = 0
        if match_mode == "substring":
            cnt = count_substring_occurrences(text, term)

        elif match_mode == "jieba_precise":
            if token_counter_precise is None:
                cnt = count_substring_occurrences(text, term)  # fallback
            else:
                cnt = int(token_counter_precise.get(term, 0))

        elif match_mode == "jieba_search":
            if token_counter_search is None:
                cnt = count_substring_occurrences(text, term)  # fallback
            else:
                cnt = int(token_counter_search.get(term, 0))

        elif match_mode == "hybrid":
            # Default for Chinese dictionaries:
            # - phrases (2+ chars) -> substring (stable for fixed expressions)
            # - single characters -> token counting (reduces overcounting in dense texts)
            if len(term) >= 2:
                cnt = count_substring_occurrences(text, term)
            else:
                if token_counter_precise is None:
                    cnt = count_substring_occurrences(text, term)
                else:
                    cnt = int(token_counter_precise.get(term, 0))
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

    dict_totals = (
        terms_df.groupby("category")["term"]
        .nunique()
        .rename("Dict terms")
        .reset_index()
        .rename(columns={"category": "Category"})
    )

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

    out = out.sort_values(["Total count", "Unique terms"], ascending=[False, False]).reset_index(drop=True)

    out["Coverage"] = out["Coverage"].map(lambda x: f"{x:.1%}")
    out["Share"] = out["Share"].map(lambda x: f"{x:.1%}")

    out = out[["Category", "Unique terms", "Total count", "Coverage", "Share", "Dict terms"]]
    return out


def concept_summary(term_hits: pd.DataFrame) -> pd.DataFrame:
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


def parse_meta_from_filename(filename: str) -> Dict[str, str]:
    """
    - year: first 4-digit year like 2017/2020 (works with underscores)
    - title_cn: stem after removing year + leading numeric prefixes (01..12) + CN suffix at end
    """
    stem = re.sub(r"\.[^.]+$", "", filename).strip()

    m = re.search(r"(?<!\d)(19|20)\d{2}(?!\d)", stem)
    year = m.group(0) if m else ""

    title = stem
    if year:
        title = re.sub(rf"{re.escape(year)}", "", title)

    title = re.sub(r"^[\s\-_–—:：]+", "", title)
    title = re.sub(r"[\s\-_–—:：]+$", "", title)

    title = re.sub(r"^(0?[1-9]|1[0-2])[\s\-_–—:：]+", "", title)
    title = re.sub(r"[\s\-_–—:：]+CN$", "", title, flags=re.IGNORECASE)
    title = re.sub(r"[\s\-_–—:：]{2,}", " ", title).strip()

    return {"year": year, "title_cn": title}


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


def safe_year(y: str) -> Optional[int]:
    y = safe_str(y).strip()
    if not y:
        return None
    if re.fullmatch(r"(19|20)\d{2}", y):
        return int(y)
    return None


def normalize_per_10k_chars(count: int, char_count: int) -> float:
    if char_count <= 0:
        return 0.0
    return (count / char_count) * 10000.0


# -----------------------------
# UI
# -----------------------------
st.title(f"Discourse Analyzer (CN religious-coded terms) — {APP_VERSION}")

with st.sidebar:
    st.header("Žodynas (terms_cn.csv)")
    st.caption("CSV formatas: concept;term;pinyin;translation;category")
    terms_upload = st.file_uploader("Įkelk terms_cn.csv (nebūtina, jei yra repo)", type=["csv"])

    st.divider()
    st.header("Atpažinimo režimas")
    match_mode = st.selectbox(
        "Kaip skaičiuoti terminų pasikartojimus?",
        options=[
            "hybrid",
            "substring",
            "jieba_precise",
            "jieba_search",
        ],
        index=0,
        help=(
            "hybrid = frazės (2+ ženklai) skaičiuojamos substring metodu, "
            "o 1-ženkliai terminai — per jieba tokenizaciją. "
            "Jei jieba neįdiegta / neveikia, automatiškai bus fallback į substring."
        ),
    )
    if match_mode.startswith("jieba") or match_mode == "hybrid":
        if not JIEBA_AVAILABLE:
            st.warning("jieba biblioteka nepasiekiama šiame env. Naudosiu substring fallback.")
        else:
            st.caption("jieba aktyvi ✅")

    st.divider()
    st.header("Dokumentai")
    files = st.file_uploader(
        "Įkelk dokumentus (TXT / DOCX)",
        type=["txt", "docx", "doc"],
        accept_multiple_files=True,
    )
    st.caption("DOC rekomenduojama konvertuoti į DOCX arba TXT.")


# Load dictionary
try:
    terms_df_raw = load_terms_csv(terms_upload)
except Exception as e:
    st.error(f"Klaida skaitant terms_cn.csv: {e}")
    st.stop()

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

# Precompute per-doc analysis (needed for Timeline)
doc_rows = []
doc_term_hits: Dict[str, pd.DataFrame] = {}
doc_cat_hits: Dict[str, pd.DataFrame] = {}
doc_conc_hits: Dict[str, pd.DataFrame] = {}

for filename, text in docs:
    ensure_doc_meta(filename)
    inferred = parse_meta_from_filename(filename)

    if not st.session_state[meta_key(filename)]["year"]:
        st.session_state[meta_key(filename)]["year"] = inferred["year"]
    if not st.session_state[meta_key(filename)]["title_cn"]:
        st.session_state[meta_key(filename)]["title_cn"] = inferred["title_cn"]

    term_hits = analyze_text(text, terms_df, match_mode=match_mode)
    cat_sum = category_summary(term_hits, terms_df)
    conc_sum = concept_summary(term_hits)

    doc_term_hits[filename] = term_hits
    doc_cat_hits[filename] = cat_sum
    doc_conc_hits[filename] = conc_sum

    total_hits = int(term_hits["Count"].sum()) if not term_hits.empty else 0

    doc_rows.append(
        {
            "filename": filename,
            "year": st.session_state[meta_key(filename)]["year"],
            "title_cn": st.session_state[meta_key(filename)]["title_cn"],
            "chars": len(text),
            "total_hits": total_hits,
            "total_hits_per_10k_chars": normalize_per_10k_chars(total_hits, len(text)),
            "unique_terms": int(term_hits["CH term"].nunique()) if not term_hits.empty else 0,
            "unique_concepts": int(term_hits["Concept"].nunique()) if not term_hits.empty else 0,
            "unique_categories": int(term_hits["Category"].nunique()) if not term_hits.empty else 0,
        }
    )

docs_overview = pd.DataFrame(doc_rows)

# Tabs: Timeline + per-document
tab_names = ["📈 Timeline"] + [fn for fn, _ in docs]
tabs = st.tabs(tab_names)

# -----------------------------
# Timeline tab
# -----------------------------
with tabs[0]:
    st.subheader("Semantiniai pokyčiai laiko perspektyvoje")

    if len(docs) < 2:
        st.info("Įkelk bent 2 dokumentus (su metais failo pavadinime), kad matytum laiko analizę.")
    else:
        # Ensure years are usable
        docs_overview["year_int"] = docs_overview["year"].map(safe_year)
        usable = docs_overview.dropna(subset=["year_int"]).copy()

        if usable.empty:
            st.warning("Nepavyko ištraukti metų iš failų pavadinimų. Įdėk metus į failo pavadinimą, pvz. 2017_....txt")
        else:
            st.caption("Patarimas: kad būtų palyginama tarp dokumentų, rinkis normalizaciją per 10 000 simbolių.")

            view_level = st.radio(
                "Ką sekti laike?",
                options=["Category", "Concept", "Term"],
                horizontal=True,
                index=0,
            )
            metric = st.selectbox(
                "Metrika",
                options=["Raw count", "Per 10k characters"],
                index=1,
            )

            # Build long format by year
            rows_long = []
            for filename, _text in docs:
                year = st.session_state[meta_key(filename)]["year"]
                y_int = safe_year(year)
                if y_int is None:
                    continue
                char_count = int(docs_overview.loc[docs_overview["filename"] == filename, "chars"].iloc[0])

                th = doc_term_hits[filename]
                if th.empty:
                    continue

                if view_level == "Category":
                    g = th.groupby("Category")["Count"].sum().reset_index().rename(columns={"Category": "label"})
                elif view_level == "Concept":
                    g = th.groupby("Concept")["Count"].sum().reset_index().rename(columns={"Concept": "label"})
                else:
                    g = th.groupby("CH term")["Count"].sum().reset_index().rename(columns={"CH term": "label"})

                for _, r in g.iterrows():
                    cnt = int(r["Count"])
                    rows_long.append(
                        {
                            "year": y_int,
                            "label": safe_str(r["label"]),
                            "count": cnt,
                            "per_10k": normalize_per_10k_chars(cnt, char_count),
                        }
                    )

            long_df = pd.DataFrame(rows_long)
            if long_df.empty:
                st.info("Nėra pakankamai hitų, kad sudaryčiau laiko grafiką.")
            else:
                # Choose labels to plot
                # Default: top 8 by total count across all years
                totals = (
                    long_df.groupby("label")["count"].sum().sort_values(ascending=False).reset_index()
                )
                default_labels = totals["label"].head(8).tolist()

                selected = st.multiselect(
                    f"Pasirink {view_level} (maks. rekomenduojama 8–12 grafike)",
                    options=totals["label"].tolist(),
                    default=default_labels,
                )

                plot_df = long_df[long_df["label"].isin(selected)].copy()
                value_col = "count" if metric == "Raw count" else "per_10k"

                pivot = (
                    plot_df.pivot_table(index="year", columns="label", values=value_col, aggfunc="sum")
                    .fillna(0)
                    .sort_index()
                )

                st.line_chart(pivot, height=320)
                st.dataframe(pivot.reset_index(), width="stretch")

            st.divider()
            st.markdown("### Dokumentų suvestinė")
            st.dataframe(
                usable.sort_values("year_int")[["year", "title_cn", "filename", "chars", "total_hits", "total_hits_per_10k_chars"]],
                width="stretch",
            )

# -----------------------------
# Per-document tabs
# -----------------------------
for (filename, text), tab in zip(docs, tabs[1:]):
    with tab:
        st.subheader("Document info")

        col1, col2 = st.columns([1, 4])
        with col1:
            st.write(f"**Metai (Year):** {st.session_state[meta_key(filename)]['year']}")
        with col2:
            st.write(f"**Pavadinimas (CN):** {st.session_state[meta_key(filename)]['title_cn']}")

        with st.expander("Teksto statistika", expanded=False):
            st.write(
                {
                    "Characters": len(text),
                    "Lines": text.count("\n") + 1 if text else 0,
                }
            )

        st.divider()

        st.subheader("Analizė")

        term_hits = doc_term_hits[filename]
        cat_sum = doc_cat_hits[filename]
        conc_sum = doc_conc_hits[filename]

        total_hits = int(term_hits["Count"].sum()) if not term_hits.empty else 0
        unique_terms = int(term_hits["CH term"].nunique()) if not term_hits.empty else 0
        unique_concepts = int(term_hits["Concept"].nunique()) if not term_hits.empty else 0
        unique_categories = int(term_hits["Category"].nunique()) if not term_hits.empty else 0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total matches", total_hits)
        m2.metric("Unique terms", unique_terms)
        m3.metric("Unique concepts", unique_concepts)
        m4.metric("Categories hit", unique_categories)

        st.markdown("### 1) Term detail")
        if term_hits.empty:
            st.info("Nėra termų detalių (nes nėra hitų).")
        else:
            term_hits_view = term_hits.sort_values(["Count"], ascending=[False]).reset_index(drop=True)
            term_hits_view.index = range(1, len(term_hits_view) + 1)
            st.dataframe(term_hits_view, width="stretch")

        st.markdown("### 2) Concept summary")
        if conc_sum.empty:
            st.info("Nėra concept rezultatų (nes nėra termų).")
        else:
            conc_sum_view = conc_sum.sort_values(["Total count"], ascending=[False]).reset_index(drop=True)
            conc_sum_view.index = range(1, len(conc_sum_view) + 1)
            st.dataframe(conc_sum_view, width="stretch")

        st.markdown("### 3) Category summary")
        if cat_sum.empty:
            st.info("Šiame dokumente nerasta nė vieno termino iš žodyno.")
        else:
            cat_sum_view = cat_sum.sort_values(["Total count"], ascending=[False]).reset_index(drop=True)
            cat_sum_view.index = range(1, len(cat_sum_view) + 1)
            st.dataframe(cat_sum_view, width="stretch")

        if not term_hits.empty:
            st.divider()
            st.subheader("Downloads")

            cdl1, cdl2 = st.columns(2)
            with cdl1:
                st.download_button(
                    "Download term detail (CSV)",
                    data=term_hits.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"{filename}_term_detail.csv",
                    mime="text/csv",
                )
            with cdl2:
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
