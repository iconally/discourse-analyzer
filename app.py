# app.py
from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st

# Optional: docx support
try:
    import docx  # python-docx
except Exception:
    docx = None


# ----------------------------
# Config
# ----------------------------
APP_DIR = Path(__file__).resolve().parent
DEFAULT_TERMS_PATH = APP_DIR / "terms_cn.csv"

st.set_page_config(page_title="Discourse Analyzer", layout="wide")

st.title("Discourse Analyzer (CN term matching)")
st.caption("Dokumentai analizuojami po vieną (tabs). EN tikrinimo kol kas nedarome.")


# ----------------------------
# Data structures
# ----------------------------
@dataclass
class DocInfo:
    filename: str
    year: Optional[int]
    title_cn: str
    full_name: str


# ----------------------------
# Helpers
# ----------------------------
def safe_strip(x) -> str:
    if isinstance(x, list):
        x = " ".join(str(i) for i in x)
    if x is None:
        return ""
    return str(x).strip()


def normalize_text(s: str) -> str:
    if not s:
        return ""
    return s.replace("\r\n", "\n").replace("\r", "\n").replace("\u3000", " ").strip()


def parse_year_from_filename(name: str) -> Optional[int]:
    m = re.search(r"(19|20)\d{2}", name)
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


def infer_cn_title_from_filename(filename: str) -> str:
    """
    Iš failo vardo pašalina metus ir techninius gabalus, palieka „pavadinimo“ užuominą.
    Jei nepavyksta – grąžina filename be plėtinio.
    """
    base = re.sub(r"\.(txt|docx|doc)$", "", filename, flags=re.IGNORECASE)

    # pašalinam pradžios datą/seką pvz 2017-07-Doc4_CN -> Doc4_CN
    base = re.sub(r"^(19|20)\d{2}[-_.]?\d{0,2}[-_.]?", "", base).strip("-_. ")

    # nuimam _CN, -CN, .CN
    base = re.sub(r"[-_.]?CN$", "", base, flags=re.IGNORECASE)

    # kosmetika
    base = base.replace("_", " ").replace("-", " ").strip()
    return base if base else re.sub(r"\.(txt|docx|doc)$", "", filename, flags=re.IGNORECASE)


def read_txt(uploaded_file) -> str:
    raw = uploaded_file.read()
    for enc in ("utf-8", "utf-8-sig", "gb18030", "gbk", "big5"):
        try:
            return raw.decode(enc)
        except Exception:
            continue
    return raw.decode("utf-8", errors="ignore")


def read_docx(uploaded_file) -> str:
    if docx is None:
        raise RuntimeError("python-docx neįdiegtas. Naudok .txt arba įdiek python-docx.")
    d = docx.Document(uploaded_file)
    return "\n".join(p.text for p in d.paragraphs if p.text)


def load_terms_from_repo(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No such file or directory: {path.name}. Įkelk terms_cn.csv per sidebar arba pridėk į repo.")
    df = pd.read_csv(path, sep=";", dtype=str, keep_default_na=False, encoding="utf-8")

    expected = ["concept", "term", "pinyin", "translation", "category"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"terms_cn.csv trūksta stulpelių: {missing}. Turi būti: {expected}")

    for c in expected:
        df[c] = df[c].apply(safe_strip)

    df = df[df["term"].str.len() > 0].copy()
    df = df.drop_duplicates(subset=expected).reset_index(drop=True)
    return df


def load_terms_from_upload(uploaded) -> pd.DataFrame:
    df = pd.read_csv(uploaded, sep=";", dtype=str, keep_default_na=False, encoding="utf-8")
    expected = ["concept", "term", "pinyin", "translation", "category"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"CSV trūksta stulpelių: {missing}. Turi būti: {expected}")

    for c in expected:
        df[c] = df[c].apply(safe_strip)

    df = df[df["term"].str.len() > 0].copy()
    df = df.drop_duplicates(subset=expected).reset_index(drop=True)
    return df


def count_term_occurrences(text: str, term: str) -> int:
    if not term:
        return 0
    return len(re.findall(re.escape(term), text))


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
def analyze_document(text: str, terms_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in terms_df.iterrows():
        term = r["term"]
        cnt = count_term_occurrences(text, term)
        if cnt > 0:
            rows.append(
                {
                    "CH term": r["term"],
                    "Pinyin": r["pinyin"],
                    "ENG translation": r["translation"],
                    "Concept": r["concept"],
                    "Category": r["category"],
                    "Count": int(cnt),
                }
            )

    out = pd.DataFrame(rows, columns=["CH term", "Pinyin", "ENG translation", "Concept", "Category", "Count"])
    if out.empty:
        return out

    # sum duplicates just in case
    out = (
        out.groupby(["CH term", "Pinyin", "ENG translation", "Concept", "Category"], as_index=False)["Count"]
        .sum()
        .sort_values("Count", ascending=False)
        .reset_index(drop=True)
    )
    return out


def add_index_from_1(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2.insert(0, "№", range(1, len(df2) + 1))
    return df2


def concept_summary(term_hits: pd.DataFrame) -> pd.DataFrame:
    """
    Grupuoja pagal Concept (+ Category), rodo:
    - Unique terms (doc)
    - Total count (doc)
    """
    cols = ["Concept", "Category", "Unique terms (doc)", "Total count (doc)"]
    if term_hits.empty:
        return pd.DataFrame(columns=cols)

    g = (
        term_hits.groupby(["Concept", "Category"])
        .agg(**{
            "Unique terms (doc)": ("CH term", "nunique"),
            "Total count (doc)": ("Count", "sum"),
        })
        .reset_index()
        .sort_values(["Total count (doc)", "Unique terms (doc)"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return g[cols]


def category_summary(term_hits: pd.DataFrame) -> pd.DataFrame:
    """
    Grupuoja pagal Category:
    - Unique terms (doc)
    - Total count (doc)
    - Category share (percent)
    """
    cols = ["Category", "Unique terms (doc)", "Total count (doc)", "Category share"]
    if term_hits.empty:
        return pd.DataFrame(columns=cols)

    g = (
        term_hits.groupby("Category")
        .agg(**{
            "Unique terms (doc)": ("CH term", "nunique"),
            "Total count (doc)": ("Count", "sum"),
        })
        .reset_index()
        .sort_values(["Total count (doc)", "Unique terms (doc)"], ascending=[False, False])
        .reset_index(drop=True)
    )

    total = g["Total count (doc)"].sum()
    if total > 0:
        g["Category share"] = g["Total count (doc)"] / total
    else:
        g["Category share"] = 0.0

    # percent display
    g["Category share"] = g["Category share"].map(lambda x: f"{x:.1%}")
    return g[cols]


# ----------------------------
# Sidebar: load dictionary + docs upload
# ----------------------------
with st.sidebar:
    st.header("Žodynas (terms_cn.csv)")
    terms_source = st.radio(
        "Kaip įkelti terms_cn.csv?",
        ["Naudoti repo terms_cn.csv", "Įkelti savo CSV"],
        index=0,
    )

    terms_df = None
    try:
        if terms_source == "Naudoti repo terms_cn.csv":
            terms_df = load_terms_from_repo(DEFAULT_TERMS_PATH)
            st.success(f"Įkeltas: {DEFAULT_TERMS_PATH.name}")
        else:
            up = st.file_uploader("Įkelk terms_cn.csv", type=["csv"])
            if up is not None:
                terms_df = load_terms_from_upload(up)
                st.success(f"Įkeltas: {up.name}")
            else:
                st.info("Įkelk CSV, kad pradėtume analizę.")
    except Exception as e:
        st.error(f"Klaida skaitant terms_cn.csv: {e}")

    if terms_df is not None:
        st.caption(f"Žodyne: {terms_df['term'].nunique()} unikalūs terminai")
        with st.expander("Žodyno pavyzdys"):
            st.dataframe(terms_df.head(20), width="stretch", hide_index=True)

    st.divider()
    st.header("Įkelk dokumentus (CN)")
    uploaded_docs = st.file_uploader(
        "TXT / DOCX (galima kelis iškart)",
        type=["txt", "docx", "doc"],
        accept_multiple_files=True,
    )
    st.caption("DOC rekomenduojama konvertuoti į DOCX arba TXT.")


# ----------------------------
# Main flow
# ----------------------------
if terms_df is None:
    st.warning("Pirma sutvarkyk / įkelk terms_cn.csv (sidebar), tada galėsime analizuoti dokumentus.")
    st.stop()

if not uploaded_docs:
    st.info("Įkelk bent vieną dokumentą (.txt / .docx) sidebar’e.")
    st.stop()

docs_data: List[Tuple[DocInfo, str]] = []
read_errors = []

for f in uploaded_docs:
    name = f.name
    suffix = Path(name).suffix.lower().lstrip(".")
    try:
        if suffix == "txt":
            text = read_txt(f)
        elif suffix == "docx":
            text = read_docx(f)
        elif suffix == "doc":
            raise RuntimeError("DOC formatas nepalaikomas patikimai. Konvertuok į DOCX arba TXT.")
        else:
            raise RuntimeError(f"Nežinomas formatas: {suffix}")

        text = normalize_text(text)
        year = parse_year_from_filename(name)
        title_cn = infer_cn_title_from_filename(name)

        info = DocInfo(
            filename=name,
            year=year,
            title_cn=title_cn,
            full_name=name,
        )
        docs_data.append((info, text))
    except Exception as e:
        read_errors.append((name, str(e)))

if read_errors:
    st.error("Kai kurių failų nepavyko perskaityti:")
    for n, err in read_errors:
        st.write(f"- **{n}**: {err}")

if not docs_data:
    st.stop()

docs_data.sort(key=lambda t: (t[0].year if t[0].year is not None else 9999, t[0].filename))

st.header("Rezultatai (po vieną dokumentą)")
tabs = st.tabs([info.filename for info, _ in docs_data])

for tab, (info, text) in zip(tabs, docs_data):
    with tab:
        # Document info (read-only)
        st.subheader("Document info")
        c1, c2, c3 = st.columns([1, 2, 2])
        with c1:
            st.metric("Metai", info.year if info.year is not None else "—")
        with c2:
            st.markdown(f"**Pavadinimas (CN):**  \n{info.title_cn}")
        with c3:
            st.markdown(f"**Pilnas dokumento vardas:**  \n{info.full_name}")

        # Analyze
        hits = analyze_document(text, terms_df)

        st.divider()

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

        # 1) Term detail
        st.subheader("1) Term detail")
        if hits.empty:
            st.info("Šiame dokumente nerasta nei vieno termino iš žodyno.")
        else:
            hits_show = add_index_from_1(hits)  # already sorted by Count desc
            st.dataframe(hits_show, width="stretch", hide_index=True)

        st.divider()

        # 2) Concept summary
        st.subheader("2) Concept summary")
        cs = concept_summary(hits)
        cs_show = add_index_from_1(cs)
        st.dataframe(cs_show, width="stretch", hide_index=True)

        st.divider()

        # 3) Category summary
        st.subheader("3) Category summary")
        cat = category_summary(hits)
        cat_show = add_index_from_1(cat)
        st.dataframe(cat_show, width="stretch", hide_index=True)

st.success("Paruošta ✅")
