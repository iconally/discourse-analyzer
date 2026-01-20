# app.py
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

st.set_page_config(
    page_title="Discourse Analyzer",
    layout="wide",
)

st.title("Discourse Analyzer (CN term matching)")

st.caption(
    "V1: term matching pagal žodyną (terms_cn.csv), dokumentai analizuojami po vieną (tabs). "
    "EN tikrinimo dar nedarome."
)


# ----------------------------
# Data structures
# ----------------------------
@dataclass
class TermRow:
    concept: str
    term: str
    pinyin: str
    translation: str
    category: str


@dataclass
class DocInfo:
    filename: str
    year: Optional[int]
    title_cn: str
    full_name: str


# ----------------------------
# Helpers
# ----------------------------
def parse_year_from_filename(name: str) -> Optional[int]:
    """
    Bando ištraukti metus iš failo pavadinimo, pvz:
    2017-07-Doc4_CN.txt -> 2017
    """
    m = re.search(r"(19|20)\d{2}", name)
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


def read_txt(uploaded_file) -> str:
    raw = uploaded_file.read()
    # bandome utf-8, jei nepavyksta - gb18030 (dažna CN)
    for enc in ("utf-8", "utf-8-sig", "gb18030", "gbk"):
        try:
            return raw.decode(enc)
        except Exception:
            continue
    # fallback
    return raw.decode("utf-8", errors="ignore")


def read_docx(uploaded_file) -> str:
    if docx is None:
        raise RuntimeError("python-docx neįdiegtas. Įdiek python-docx arba naudok .txt.")
    d = docx.Document(uploaded_file)
    return "\n".join(p.text for p in d.paragraphs)


def normalize_text(s: str) -> str:
    # minimalus normalizavimas
    if not s:
        return ""
    return s.replace("\u3000", " ").strip()


def safe_strip(x):
    # apsauga nuo situacijos kai kažkur pateko list vietoje string
    if isinstance(x, list):
        x = " ".join(str(i) for i in x)
    if x is None:
        return ""
    return str(x).strip()


def load_terms_csv(path: Path) -> pd.DataFrame:
    """
    Tikimasi: concept;term;pinyin;translation;category
    Separatorius: ';'
    """
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")

    df = pd.read_csv(
        path,
        sep=";",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
    )

    # suvienodinam stulpelių pavadinimus
    expected = ["concept", "term", "pinyin", "translation", "category"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(
            f"terms_cn.csv trūksta stulpelių: {missing}. Turi būti: {expected}"
        )

    # sutvarkom tarpus ir keistus tipus
    for c in expected:
        df[c] = df[c].apply(safe_strip)

    # išmetam tuščius term
    df = df[df["term"].str.len() > 0].copy()

    return df


def count_term_occurrences(text: str, term: str) -> int:
    """
    Paprastas substring match (ne tokenizacija).
    Skaičiuoja neperdengiančius atitikmenis.
    """
    if not term:
        return 0
    return len(re.findall(re.escape(term), text))


def analyze_document(text: str, terms_df: pd.DataFrame) -> pd.DataFrame:
    """
    Grąžina term-level rezultatų DF:
    concept, term, pinyin, translation, category, count
    """
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

    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(
            columns=["CH term", "Pinyin", "ENG translation", "Concept", "Category", "Count"]
        )
        return out

    out = out.sort_values("Count", ascending=False).reset_index(drop=True)
    return out


def add_index_from_1(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2.insert(0, "№", range(1, len(df2) + 1))
    return df2


def concept_summary(term_hits: pd.DataFrame, terms_df: pd.DataFrame) -> pd.DataFrame:
    """
    Grupuoja pagal Concept (+ Category), nes tavo žodyne Concept dažniausiai priklauso vienai Category.
    Rodom:
    - unique_terms_in_doc
    - total_count_in_doc
    - coverage_in_doc = unique_terms_in_doc / total_terms_in_dictionary_for_that_concept
    """
    if term_hits.empty:
        return pd.DataFrame(
            columns=[
                "Concept",
                "Category",
                "Unique terms (doc)",
                "Total count (doc)",
                "Coverage (doc)",
            ]
        )

    dict_counts = (
        terms_df.groupby(["concept", "category"])["term"]
        .nunique()
        .reset_index()
        .rename(columns={"concept": "Concept", "category": "Category", "term": "Terms in dictionary"})
    )

    g = (
        term_hits.groupby(["Concept", "Category"])
        .agg(**{
            "Unique terms (doc)": ("CH term", "nunique"),
            "Total count (doc)": ("Count", "sum"),
        })
        .reset_index()
    )

    merged = g.merge(dict_counts, on=["Concept", "Category"], how="left")
    merged["Terms in dictionary"] = merged["Terms in dictionary"].fillna(0).astype(int)

    def cov(row):
        denom = row["Terms in dictionary"]
        if denom <= 0:
            return 0.0
        return row["Unique terms (doc)"] / denom

    merged["Coverage (doc)"] = merged.apply(cov, axis=1)
    merged = merged.drop(columns=["Terms in dictionary"])

    merged = merged.sort_values("Total count (doc)", ascending=False).reset_index(drop=True)
    return merged


def category_summary(term_hits: pd.DataFrame) -> pd.DataFrame:
    """
    Grupuoja pagal Category:
    - unique_terms
    - total_count
    - category_share = total_count / sum(total_count)
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
    )

    total = g["Total count (doc)"].sum()
    if total > 0:
        g["Category share"] = g["Total count (doc)"] / total
    else:
        g["Category share"] = 0.0

    g = g.sort_values("Total count (doc)", ascending=False).reset_index(drop=True)
    return g


def overall_coverage(term_hits: pd.DataFrame, terms_df: pd.DataFrame) -> Tuple[float, int, int]:
    """
    coverage = unique_terms_found / total_terms_in_dictionary
    """
    total_dict_terms = int(terms_df["term"].nunique())
    unique_found = int(term_hits["CH term"].nunique()) if not term_hits.empty else 0
    cov = (unique_found / total_dict_terms) if total_dict_terms > 0 else 0.0
    return cov, unique_found, total_dict_terms


# ----------------------------
# Sidebar: dictionary
# ----------------------------
st.sidebar.header("Žodynas (terms_cn.csv)")

terms_source = st.sidebar.radio(
    "Kaip įkelti terms_cn.csv?",
    ["Naudoti repo terms_cn.csv", "Įkelti savo CSV"],
    index=0,
)

terms_df: Optional[pd.DataFrame] = None
terms_path_used: Optional[str] = None

try:
    if terms_source == "Naudoti repo terms_cn.csv":
        terms_df = load_terms_csv(DEFAULT_TERMS_PATH)
        terms_path_used = str(DEFAULT_TERMS_PATH)
        st.sidebar.success(f"Įkeltas: {DEFAULT_TERMS_PATH.name}")
    else:
        up = st.sidebar.file_uploader("Įkelk terms_cn.csv", type=["csv"])
        if up is not None:
            # Streamlit UploadedFile neturi patogaus Path, skaitom į pandas iš bytes
            df_tmp = pd.read_csv(up, sep=";", dtype=str, keep_default_na=False, encoding="utf-8")
            # sutvarkom
            expected = ["concept", "term", "pinyin", "translation", "category"]
            missing = [c for c in expected if c not in df_tmp.columns]
            if missing:
                raise ValueError(f"CSV trūksta stulpelių: {missing}. Turi būti: {expected}")
            for c in expected:
                df_tmp[c] = df_tmp[c].apply(safe_strip)
            df_tmp = df_tmp[df_tmp["term"].str.len() > 0].copy()
            terms_df = df_tmp
            terms_path_used = up.name
            st.sidebar.success(f"Įkeltas: {up.name}")
        else:
            st.sidebar.info("Įkelk CSV, kad pradėtume analizę.")
except Exception as e:
    st.sidebar.error(f"Klaida skaitant terms_cn.csv: {e}")

if terms_df is not None:
    st.sidebar.caption(f"Žodyne: {terms_df['term'].nunique()} unikalūs terminai")
    with st.sidebar.expander("Peržiūrėti žodyno pavyzdį"):
        st.dataframe(terms_df.head(30), width="stretch")


# ----------------------------
# Main: upload documents
# ----------------------------
st.header("1) Įkelk dokumentus (CN)")

uploaded_docs = st.file_uploader(
    "Galima įkelti kelis iškart. Vėliau rezultatai bus atskiruose TAB’uose.",
    type=["txt", "docx", "doc"],
    accept_multiple_files=True,
)

if terms_df is None:
    st.warning("Pirma sutvarkyk / įkelk terms_cn.csv (sidebar), tada galėsime analizuoti dokumentus.")
    st.stop()

if not uploaded_docs:
    st.info("Įkelk bent vieną dokumentą (.txt / .docx).")
    st.stop()

# perskaitom dokumentus ir surūšiuojam chronologiškai pagal metus (jei ištraukiami)
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
            # be special libs dažnai nepavyks
            raise RuntimeError("DOC formatas nepalaikomas patikimai. Konvertuok į DOCX arba TXT.")
        else:
            raise RuntimeError(f"Nežinomas formatas: {suffix}")

        text = normalize_text(text)

        year = parse_year_from_filename(name)
        # kol kas title_cn ir full_name imam iš filename
        info = DocInfo(
            filename=name,
            year=year,
            title_cn=name,      # vėliau galėsi įkelti tikrą CN pavadinimą
            full_name=name,     # vėliau galėsi įkelti pilną
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

# sort: year asc, then filename
docs_data.sort(key=lambda t: (t[0].year if t[0].year is not None else 9999, t[0].filename))

# ----------------------------
# Analyze each doc, show tabs
# ----------------------------
tab_names = [d[0].filename for d in docs_data]
tabs = st.tabs(tab_names)

for tab, (info, text) in zip(tabs, docs_data):
    with tab:
        # ---- Document info (READ-ONLY)
        st.subheader("Document info")

        c1, c2, c3 = st.columns([1, 2, 2])
        with c1:
            st.metric("Metai", info.year if info.year is not None else "—")
        with c2:
            st.markdown(f"**Pavadinimas (CN / filename):**  \n{info.title_cn}")
        with c3:
            st.markdown(f"**Pilnas dokumento vardas:**  \n{info.full_name}")

        # ---- Run analysis
        hits = analyze_document(text, terms_df)

        # Overall metrics
        cov, uniq_found, total_dict = overall_coverage(hits, terms_df)
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Coverage (unique terms / dictionary)", f"{cov:.1%}")
        with m2:
            st.metric("Unikalūs terminai rasti", uniq_found)
        with m3:
            st.metric("Terminai žodyne", total_dict)

        st.divider()

        # ---- 1) Term details
        st.subheader("Term details")
        if hits.empty:
            st.info("Šiame dokumente nerasta nei vieno termino iš žodyno.")
        else:
            hits_show = add_index_from_1(hits)
            # jau surūšiuota pagal Count desc
            st.dataframe(hits_show, width="stretch", hide_index=True)

        st.divider()

        # ---- 2) Concept summary
        st.subheader("Concept summary")
        cs = concept_summary(hits, terms_df)
        cs_show = add_index_from_1(cs)
        st.dataframe(cs_show, width="stretch", hide_index=True)

        st.divider()

        # ---- 3) Category summary
        st.subheader("Category summary")
        cat = category_summary(hits)
        cat_show = add_index_from_1(cat)
        st.dataframe(cat_show, width="stretch", hide_index=True)

st.success("Paruošta. Jei nori – kitame žingsnyje galim pridėti: dokumentų palyginimą (timeline), eksportą į CSV/XLSX, ir t.t.")
