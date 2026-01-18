app.py
import re
import io
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict

import pandas as pd
import streamlit as st

try:
    import docx  # python-docx
except Exception:
    docx = None

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Bilingual Discourse Analyzer (CN ↔ EN)", layout="wide")

# ----------------------------
# Regex helpers
# ----------------------------
YEAR_RE = re.compile(r"(20[0-2]\d)")  # 2000-2029
# DocID: anything alnum/hyphen between underscores before _CN/_EN
DOCID_RE = re.compile(r"(?:^|_)([A-Za-z0-9\-]+)(?:_CN|_EN)(?:_|\.)")
# Optional per-year order: e.g., 2020_01_..., 2020-01-..., 2020.01....
ORDER_RE = re.compile(r"(20[0-2]\d)[_\-\.](\d{1,2})(?=[_\-\.])")

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
    """
    Detects order if filename contains pattern like 2020_01_ or 2020-2- or 2020.03.
    Returns int 1..12.. etc.
    """
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


def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


# ----------------------------
# Term structures
# ----------------------------
@dataclass
class Term:
    concept: str
    term: str
    category: str


def load_terms_csv(df: pd.DataFrame, lang: str) -> List[Term]:
    """
    Expected columns: concept, term, category
    """
    df2 = df.copy()
    df2.columns = [c.strip().lower() for c in df2.columns]
    required = {"concept", "term", "category"}
    missing = required - set(df2.columns)
    if missing:
        raise ValueError(f"{lang} terms CSV trūksta stulpelių: {', '.join(sorted(missing))}")

    terms: List[Term] = []
    for _, row in df2.iterrows():
        concept = str(row["concept"]).strip()
        term = str(row["term"]).strip()
        category = str(row["category"]).strip()
        if concept and term:
            terms.append(Term(concept=concept, term=term, category=category))
    return terms


def build_patterns(terms: List[Term], lang: str) -> Dict[Tuple[str, str], re.Pattern]:
    """
    CN: literal substring regex (escaped)
    EN: tries word boundaries for plain alpha-num phrases; case-insensitive
    """
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


def count_terms(text: str, terms: List[Term], patterns: Dict[Tuple[str, str], re.Pattern]) -> pd.DataFrame:
    rows = []
    for t in terms:
        pat = patterns[(t.concept, t.term)]
        c = len(pat.findall(text))
        rows.append({"concept": t.concept, "term": t.term, "category": t.category, "count": c})
    return pd.DataFrame(rows)


# ----------------------------
# Corpus ingestion
# ----------------------------
@dataclass
class Doc:
    lang: str
    filename: str
    year: Optional[int]
    order_in_year: Optional[int]
    docid: Optional[str]
    text: str


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
            raise ValueError(f"Failas {name} yra .doc. Prašau konvertuoti į .docx.")
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
# Analytics: keyword counts
# ----------------------------
def analyze_corpus(docs: List[Doc], terms: List[Term], lang: str) -> pd.DataFrame:
    patterns = build_patterns(terms, lang=lang)
    all_rows = []
    for d in docs:
        df_counts = count_terms(d.text, terms, patterns)
        df_counts["lang"] = d.lang
        df_counts["filename"] = d.filename
        df_counts["year"] = d.year
        df_counts["order_in_year"] = d.order_in_year
        df_counts["docid"] = d.docid
        df_counts["doc_key"] = make_doc_key(d)
        all_rows.append(df_counts)

    if not all_rows:
        return pd.DataFrame(columns=["concept", "term", "category", "count", "lang", "filename", "year", "order_in_year", "docid", "doc_key"])
    return pd.concat(all_rows, ignore_index=True)


def make_doc_key(d: Doc) -> str:
    """
    Stable chronological key for comparison when multiple docs per year:
      2020.01|DOCID|CN
    If order missing -> 2020.99|DOCID|CN (still sorts after ordered ones)
    """
    y = d.year if d.year is not None else 9999
    o = d.order_in_year if d.order_in_year is not None else 99
    did = d.docid if d.docid else d.filename
    return f"{y:04d}.{o:02d}|{did}|{d.lang}"


def build_doc_level_profile(df_detail: pd.DataFrame) -> pd.DataFrame:
    """
    Returns: doc_key, lang, year, order_in_year, docid, concept, category, count (summed across term variants)
    """
    if df_detail.empty:
        return df_detail
    g = (
        df_detail.groupby(["doc_key", "lang", "year", "order_in_year", "docid", "concept", "category"], dropna=False)["count"]
        .sum()
        .reset_index()
    )
    return g.sort_values(["lang", "year", "order_in_year", "docid", "category", "concept"])


def build_matrix(df_doc_profile: pd.DataFrame, lang: str) -> pd.DataFrame:
    """
    Matrix Document x Concept (counts). Index = doc_key
    """
    part = df_doc_profile[df_doc_profile["lang"] == lang].copy()
    if part.empty:
        return pd.DataFrame()
    mat = part.pivot_table(index="doc_key", columns="concept", values="count", aggfunc="sum", fill_value=0)
    # helpful metadata columns
    meta = (
        part[["doc_key", "year", "order_in_year", "docid"]]
        .drop_duplicates()
        .set_index("doc_key")
        .sort_values(["year", "order_in_year", "docid"])
    )
    mat = meta.join(mat, how="left")
    return mat.reset_index()


def compute_changes(df_doc_profile: pd.DataFrame, lang: str) -> pd.DataFrame:
    """
    For chronological sequence (within a lang):
      - NEW: count_now>0 and count_prev==0
      - LOST: count_now==0 and count_prev>0
      - DELTA: count_now - count_prev
    Output rows per (doc_key_now, concept)
    """
    part = df_doc_profile[df_doc_profile["lang"] == lang].copy()
    if part.empty:
        return pd.DataFrame()

    # aggregate per doc_key & concept
    agg = part.groupby(["doc_key", "year", "order_in_year", "docid", "concept", "category"], dropna=False)["count"].sum().reset_index()

    # get ordered docs
    docs_meta = agg[["doc_key", "year", "order_in_year", "docid"]].drop_duplicates()
    docs_meta = docs_meta.sort_values(["year", "order_in_year", "docid"]).reset_index(drop=True)

    # build dict: doc_key -> concept -> count
    per_doc = defaultdict(dict)
    per_cat = {}
    for _, r in agg.iterrows():
        per_doc[r["doc_key"]][r["concept"]] = int(r["count"])
        per_cat[r["concept"]] = r["category"]

    rows = []
    prev_counts = {}
    prev_key = None
    for i, dm in docs_meta.iterrows():
        cur_key = dm["doc_key"]
        cur_counts = per_doc.get(cur_key, {})
        concepts = set(prev_counts.keys()) | set(cur_counts.keys())

        for c in sorted(concepts):
            now = int(cur_counts.get(c, 0))
            prev = int(prev_counts.get(c, 0))
            delta = now - prev
            status = ""
            if now > 0 and prev == 0:
                status = "NEW"
            elif now == 0 and prev > 0:
                status = "LOST"
            elif now > 0 and prev > 0 and delta != 0:
                status = "CHANGED"
            elif now > 0 and prev > 0 and delta == 0:
                status = "SAME"
            else:
                status = "ZERO"

            rows.append({
                "lang": lang,
                "doc_key": cur_key,
                "year": dm["year"],
                "order_in_year": dm["order_in_year"],
                "docid": dm["docid"],
                "prev_doc_key": prev_key,
                "concept": c,
                "category": per_cat.get(c, ""),
                "count_prev": prev,
                "count_now": now,
                "delta": delta,
                "status": status
            })

        prev_counts = cur_counts
        prev_key = cur_key

    out = pd.DataFrame(rows)
    return out.sort_values(["year", "order_in_year", "docid", "category", "concept"])


def summarize_by_year(df_detail: pd.DataFrame) -> pd.DataFrame:
    if df_detail.empty:
        return df_detail
    g = df_detail.groupby(["lang", "year", "concept", "category"], dropna=False)["count"].sum().reset_index()
    return g.sort_values(["lang", "year", "category", "concept"])


def first_last_peak(df_year: pd.DataFrame) -> pd.DataFrame:
    if df_year.empty:
        return df_year
    out = []
    for (lang, concept), part in df_year.groupby(["lang", "concept"]):
        part2 = part.dropna(subset=["year"]).sort_values("year")
        if part2.empty:
            continue
        nonzero = part2[part2["count"] > 0]
        first_y = int(nonzero["year"].min()) if not nonzero.empty else None
        last_y = int(nonzero["year"].max()) if not nonzero.empty else None
        peak_row = part2.loc[part2["count"].idxmax()]
        out.append({
            "lang": lang,
            "concept": concept,
            "category": str(peak_row["category"]),
            "first_year": first_y,
            "last_year": last_y,
            "peak_year": int(peak_row["year"]) if pd.notna(peak_row["year"]) else None,
            "peak_count": int(peak_row["count"]),
        })
    return pd.DataFrame(out).sort_values(["lang", "category", "concept"])


# ----------------------------
# CN/EN comparison by DocID + concept_map
# ----------------------------
def compare_doc_pairs(df_detail_all: pd.DataFrame, concept_map: pd.DataFrame) -> pd.DataFrame:
    if df_detail_all.empty:
        return pd.DataFrame()

    cm = concept_map.copy()
    cm.columns = [c.strip().lower() for c in cm.columns]
    if "concept" not in cm.columns:
        raise ValueError("concept_map CSV turi turėti stulpelį 'concept'.")

    comparable = set(cm["concept"].astype(str).str.strip())

    agg = df_detail_all.groupby(["docid", "lang", "concept"], dropna=False)["count"].sum().reset_index()
    agg = agg[agg["concept"].isin(comparable)]

    cn = agg[agg["lang"] == "CN"].rename(columns={"count": "count_cn"}).drop(columns=["lang"])
    en = agg[agg["lang"] == "EN"].rename(columns={"count": "count_en"}).drop(columns=["lang"])

    merged = pd.merge(cn, en, on=["docid", "concept"], how="outer")
    merged["count_cn"] = merged["count_cn"].fillna(0).astype(int)
    merged["count_en"] = merged["count_en"].fillna(0).astype(int)
    merged["diff_en_minus_cn"] = merged["count_en"] - merged["count_cn"]

    merged["flag"] = ""
    merged.loc[(merged["count_cn"] > 0) & (merged["count_en"] == 0), "flag"] = "CN>0, EN=0 (gal prarasta vertime?)"
    merged.loc[(merged["count_cn"] == 0) & (merged["count_en"] > 0), "flag"] = "CN=0, EN>0 (gal pridėta vertime?)"
    merged.loc[(merged["count_cn"] > 0) & (merged["count_en"] > 0) & (merged["diff_en_minus_cn"].abs() >= 3), "flag"] = "Skirtumas >= 3 (patikrinti)"
    return merged.sort_values(["docid", "concept"])


# ----------------------------
# Candidate extraction (CN n-grams + simple TF-IDF)
# ----------------------------
CN_KEEP_RE = re.compile(r"[\u4e00-\u9fff]")  # basic CJK range

def extract_cn_char_ngrams(text: str, n_min: int = 2, n_max: int = 4, top_k: int = 80) -> pd.DataFrame:
    """
    Character n-grams (2-4) from CN text, without segmentation.
    Filters out n-grams containing non-CJK chars.
    """
    s = re.sub(r"\s+", "", text)
    chars = [ch for ch in s if CN_KEEP_RE.match(ch)]
    if len(chars) < n_min:
        return pd.DataFrame(columns=["ngram", "count"])

    counts = Counter()
    L = len(chars)
    for n in range(n_min, n_max + 1):
        for i in range(L - n + 1):
            ng = "".join(chars[i:i+n])
            counts[ng] += 1

    items = counts.most_common(top_k)
    return pd.DataFrame(items, columns=["ngram", "count"])


def tfidf_candidates_cn(docs: List[Doc], n_min: int = 2, n_max: int = 4, top_k_per_doc: int = 40, min_df: int = 2) -> pd.DataFrame:
    """
    Simple TF-IDF over CN char n-grams (no sklearn).
    - Builds n-gram vocab across docs (filtered by min_df)
    - Computes tf-idf score per doc and returns top_k_per_doc candidates
    """
    # build per-doc ngram counts
    per_doc_counts: Dict[str, Counter] = {}
    df_counts = Counter()

    for d in docs:
        s = re.sub(r"\s+", "", d.text)
        chars = [ch for ch in s if CN_KEEP_RE.match(ch)]
        c = Counter()
        L = len(chars)
        for n in range(n_min, n_max + 1):
            for i in range(L - n + 1):
                ng = "".join(chars[i:i+n])
                c[ng] += 1
        per_doc_counts[make_doc_key(d)] = c

        # document frequency
        for ng in c.keys():
            df_counts[ng] += 1

    N = max(1, len(per_doc_counts))
    vocab = {ng for ng, df in df_counts.items() if df >= min_df}

    rows = []
    for doc_key, c in per_doc_counts.items():
        # total ngrams for tf normalization
        total = sum(c[ng] for ng in c if ng in vocab)
        if total == 0:
            continue
        scored = []
        for ng, tf_raw in c.items():
            if ng not in vocab:
                continue
            tf = tf_raw / total
            df = df_counts[ng]
            idf = math.log((N + 1) / (df + 1)) + 1.0
            scored.append((ng, tf_raw, tf * idf))
        scored.sort(key=lambda x: x[2], reverse=True)
        for ng, tf_raw, score in scored[:top_k_per_doc]:
            rows.append({"doc_key": doc_key, "ngram": ng, "count": int(tf_raw), "tfidf": float(score)})

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["doc_key", "tfidf"], ascending=[True, False])


# ----------------------------
# UI
# ----------------------------
st.title("Discourse Analyzer: CN originalai ↔ EN vertimai (chronologija + keli docs per metus)")
st.caption(
    "Įkeli CN (简体) ir EN dokumentus, įkeli savo raktažodžių žodynus (CSV), gauni: "
    "kiekvieno dokumento profilį, dokumentų matricą, NEW/LOST dinamiką, kandidatų (n-grams/TF-IDF) sąrašus, ir CN↔EN patikrą."
)

with st.sidebar:
    st.header("1) Dokumentai")
    cn_files = st.file_uploader("CN originalai (.txt, .docx)", type=["txt", "docx", "doc"], accept_multiple_files=True)
    en_files = st.file_uploader("EN vertimai (.txt, .docx)", type=["txt", "docx", "doc"], accept_multiple_files=True)

    st.header("2) Žodynai (CSV)")
    cn_terms_file = st.file_uploader("terms_cn.csv (concept, term, category)", type=["csv"], accept_multiple_files=False)
    en_terms_file = st.file_uploader("terms_en.csv (concept, term, category)", type=["csv"], accept_multiple_files=False)
    map_file = st.file_uploader("concept_map.csv (bent stulpelis: concept)", type=["csv"], accept_multiple_files=False)

    st.header("3) Nustatymai")
    show_zero = st.checkbox("Rodyti ir 0 dažnius (detail lentelėje)", value=False)
    candidates_top_ngrams = st.slider("Candidates: Top n-grams per doc (2–4)", min_value=20, max_value=200, value=80, step=10)
    tfidf_top = st.slider("Candidates: Top TF-IDF per doc", min_value=10, max_value=100, value=40, step=5)
    tfidf_min_df = st.slider("Candidates: min_df (kiek doc turi turėti n-gram)", min_value=1, max_value=10, value=2, step=1)

run = st.button("▶️ Analizuoti", type="primary")

if not run:
    st.info("Įkelk failus ir paspausk **Analizuoti**.")
    st.stop()

# ----------------------------
# Run analysis
# ----------------------------
try:
    if not cn_files:
        st.warning("Įkelk bent 1 CN dokumentą.")
        st.stop()

    if not cn_terms_file:
        st.warning("Įkelk terms_cn.csv.")
        st.stop()

    # Load CSVs
    cn_terms_df = pd.read_csv(cn_terms_file)
    cn_terms = load_terms_csv(cn_terms_df, lang="CN")

    have_en = bool(en_files) and bool(en_terms_file)
    if en_files and not en_terms_file:
        st.warning("Įkėlei EN dokumentus, bet neįkėlei terms_en.csv. EN analizė bus praleista.")
        have_en = False

    if have_en:
        en_terms_df = pd.read_csv(en_terms_file)
        en_terms = load_terms_csv(en_terms_df, lang="EN")
    else:
        en_terms = []

    # Ingest docs
    cn_docs = sort_docs(ingest_files(cn_files, lang="CN"))
    en_docs = sort_docs(ingest_files(en_files, lang="EN")) if have_en else []

    # Validate docid (for pairing CN/EN)
    missing_docid_cn = [d.filename for d in cn_docs if not d.docid]
    if missing_docid_cn:
        st.error(
            "Kai kuriems CN failams nepavyko nustatyti DocID iš pavadinimo.\n\n"
            "Rekomenduojamas formatas: 2020_01_DOC123_CN.txt (DocID=DOC123)\n\n"
            f"Probleminiai: {', '.join(missing_docid_cn)}"
        )
        st.stop()

    if have_en:
        missing_docid_en = [d.filename for d in en_docs if not d.docid]
        if missing_docid_en:
            st.error(
                "Kai kuriems EN failams nepavyko nustatyti DocID iš pavadinimo.\n\n"
                "Rekomenduojamas formatas: 2020_01_DOC123_EN.txt (DocID=DOC123)\n\n"
                f"Probleminiai: {', '.join(missing_docid_en)}"
            )
            st.stop()

    # Analyze keyword counts
    df_cn_detail = analyze_corpus(cn_docs, cn_terms, lang="CN")
    df_en_detail = analyze_corpus(en_docs, en_terms, lang="EN") if have_en else pd.DataFrame(columns=df_cn_detail.columns)
    df_detail_all = pd.concat([df_cn_detail, df_en_detail], ignore_index=True)

    if not show_zero:
        df_detail_view = df_detail_all[df_detail_all["count"] > 0].copy()
    else:
        df_detail_view = df_detail_all.copy()

    # Doc-level profile (concept aggregated)
    df_doc_profile = build_doc_level_profile(df_detail_all)

    # Matrices
    cn_matrix = build_matrix(df_doc_profile, lang="CN")
    en_matrix = build_matrix(df_doc_profile, lang="EN") if have_en else pd.DataFrame()

    # Changes
    cn_changes = compute_changes(df_doc_profile, lang="CN")
    en_changes = compute_changes(df_doc_profile, lang="EN") if have_en else pd.DataFrame()

    # Year summaries + first/last/peak
    df_year = summarize_by_year(df_detail_all)
    df_flp = first_last_peak(df_year)

    # Candidates (CN)
    # Top n-grams per CN doc + TF-IDF candidates across CN corpus
    cand_ng_rows = []
    for d in cn_docs:
        topn = extract_cn_char_ngrams(d.text, n_min=2, n_max=4, top_k=candidates_top_ngrams)
        topn["doc_key"] = make_doc_key(d)
        topn["year"] = d.year
        topn["order_in_year"] = d.order_in_year
        topn["docid"] = d.docid
        cand_ng_rows.append(topn)
    cn_ngrams = pd.concat(cand_ng_rows, ignore_index=True) if cand_ng_rows else pd.DataFrame(columns=["ngram", "count", "doc_key", "year", "order_in_year", "docid"])

    cn_tfidf = tfidf_candidates_cn(cn_docs, n_min=2, n_max=4, top_k_per_doc=tfidf_top, min_df=tfidf_min_df)

    # CN↔EN compare (optional)
    df_cmp = pd.DataFrame()
    if have_en and map_file:
        map_df = pd.read_csv(map_file)
        df_cmp = compare_doc_pairs(df_detail_all, map_df)

    # ----------------------------
    # Tabs
    # ----------------------------
    tabs = [
        "Dokumentų tvarka (CN/EN)",
        "Detail: termai per dokumentą",
        "Profilis: concept per dokumentą",
        "Matrica: Document × Concept",
        "NEW/LOST pokyčiai (chronologiškai)",
        "Candidates (CN n-grams / TF-IDF)",
        "Suvestinė per metus",
        "First/Last/Peak",
    ]
    if have_en and not df_cmp.empty:
        tabs.append("CN vs EN patikra (DocID)")

    tab_objs = st.tabs(tabs)

    # 1) Order
    with tab_objs[0]:
        st.subheader("Chronologinė tvarka (su keliais dokumentais tais pačiais metais)")
        st.write("Rikiuojama pagal **(year, order_in_year, filename)**. Jei `order_in_year` nerastas, dokumentas eina vėliau (kaip 99).")

        cn_order = pd.DataFrame([{
            "lang": d.lang,
            "year": d.year,
            "order_in_year": d.order_in_year,
            "docid": d.docid,
            "filename": d.filename,
            "doc_key": make_doc_key(d),
        } for d in cn_docs])
        st.markdown("### CN")
        st.dataframe(cn_order)

        if have_en:
            en_order = pd.DataFrame([{
                "lang": d.lang,
                "year": d.year,
                "order_in_year": d.order_in_year,
                "docid": d.docid,
                "filename": d.filename,
                "doc_key": make_doc_key(d),
            } for d in en_docs])
            st.markdown("### EN")
            st.dataframe(en_order)

    # 2) Detail per document per term
    with tab_objs[1]:
        st.subheader("Detail: kiekviename dokumente – kokie termai ir kiek kartų")
        st.dataframe(df_detail_view.sort_values(["lang", "year", "order_in_year", "docid", "category", "concept", "term"]))

        st.download_button(
            "⬇️ Atsisiųsti detail_counts.csv",
            data=df_detail_view.to_csv(index=False).encode("utf-8"),
            file_name="detail_counts.csv",
            mime="text/csv"
        )

    # 3) Profile per document aggregated by concept
    with tab_objs[2]:
        st.subheader("Dokumento profilis: concept (sumuojant term variantus)")
        st.dataframe(df_doc_profile.sort_values(["lang", "year", "order_in_year", "docid", "category", "concept"]))

        st.download_button(
            "⬇️ Atsisiųsti doc_profile_concept.csv",
            data=df_doc_profile.to_csv(index=False).encode("utf-8"),
            file_name="doc_profile_concept.csv",
            mime="text/csv"
        )

    # 4) Matrix
    with tab_objs[3]:
        st.subheader("Matrica: Document × Concept")
        st.markdown("### CN")
        st.dataframe(cn_matrix)

        st.download_button(
            "⬇️ Atsisiųsti cn_matrix.csv",
            data=cn_matrix.to_csv(index=False).encode("utf-8"),
            file_name="cn_matrix.csv",
            mime="text/csv"
        )

        if have_en:
            st.markdown("### EN")
            st.dataframe(en_matrix)
            st.download_button(
                "⬇️ Atsisiųsti en_matrix.csv",
                data=en_matrix.to_csv(index=False).encode("utf-8"),
                file_name="en_matrix.csv",
                mime="text/csv"
            )

    # 5) Changes NEW/LOST
    with tab_objs[4]:
        st.subheader("Pokyčiai tarp gretimų dokumentų (chronologiškai): NEW / LOST / DELTA")
        st.markdown("### CN")
        st.dataframe(cn_changes)

        st.download_button(
            "⬇️ Atsisiųsti cn_changes.csv",
            data=cn_changes.to_csv(index=False).encode("utf-8"),
            file_name="cn_changes.csv",
            mime="text/csv"
        )

        if have_en:
            st.markdown("### EN")
            st.dataframe(en_changes)
            st.download_button(
                "⬇️ Atsisiųsti en_changes.csv",
                data=en_changes.to_csv(index=False).encode("utf-8"),
                file_name="en_changes.csv",
                mime="text/csv"
            )

    # 6) Candidates
    with tab_objs[5]:
        st.subheader("Candidates (CN): top n-grams ir TF-IDF (be segmentavimo)")
        st.write(
            "Tai skirta tavo workflow: prieš susidarant galutinį religinių raktažodžių žodyną, "
            "pažiūri pasikartojančias frazes (n-grams) ir dokumentui būdingas frazes (TF-IDF)."
        )

        st.markdown("### Top n-grams (2–4) kiekvienam CN dokumentui")
        st.dataframe(cn_ngrams.sort_values(["year", "order_in_year", "docid", "count"], ascending=[True, True, True, False]))

        st.download_button(
            "⬇️ Atsisiųsti cn_ngrams.csv",
            data=cn_ngrams.to_csv(index=False).encode("utf-8"),
            file_name="cn_ngrams.csv",
            mime="text/csv"
        )

        st.markdown("### TF-IDF kandidatai (CN corpus)")
        st.dataframe(cn_tfidf)

        st.download_button(
            "⬇️ Atsisiųsti cn_tfidf_candidates.csv",
            data=cn_tfidf.to_csv(index=False).encode("utf-8"),
            file_name="cn_tfidf_candidates.csv",
            mime="text/csv"
        )

    # 7) Year summary
    with tab_objs[6]:
        st.subheader("Suvestinė per metus: lang × year × concept")
        st.dataframe(df_year)

        st.download_button(
            "⬇️ Atsisiųsti year_summary.csv",
            data=df_year.to_csv(index=False).encode("utf-8"),
            file_name="year_summary.csv",
            mime="text/csv"
        )

    # 8) First/Last/Peak
    with tab_objs[7]:
        st.subheader("Pirmas pasirodymas / paskutinis / pikas (pagal metus)")
        st.dataframe(df_flp)

        st.download_button(
            "⬇️ Atsisiųsti first_last_peak.csv",
            data=df_flp.to_csv(index=False).encode("utf-8"),
            file_name="first_last_peak.csv",
            mime="text/csv"
        )

    # 9) CN vs EN compare
    if have_en and not df_cmp.empty:
        with tab_objs[8]:
            st.subheader("CN ↔ EN patikra (DocID + concept_map): ar vertimas „nepametė“ sąvokų")
            st.dataframe(df_cmp)

            st.download_button(
                "⬇️ Atsisiųsti cn_en_compare.csv",
                data=df_cmp.to_csv(index=False).encode("utf-8"),
                file_name="cn_en_compare.csv",
                mime="text/csv"
            )

    st.success("Baigta ✅")

except Exception as e:
    st.error(f"Klaida: {e}")
    st.stop()
