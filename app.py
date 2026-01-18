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

st.set_page_config(page_title="Discourse Analyzer (CN ↔ EN) - no pandas", layout="wide")

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
# CSV loaders (no pandas)
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


def doc_profile(rows: List[dict]) -> List[dict]:
    agg = defaultdict(int)
    meta = {}
    for r in rows:
        key = (r["doc_key"], r["lang"], r["year"], r["order_in_year"], r["docid"], r["concept"], r["category"])
        agg[key] += int(r["count"] or 0)
        meta[(r["doc_key"], r["lang"], r["year"], r["order_in_year"], r["docid"])] = True

    out = []
    for (doc_key, lang, year, order_in_year, docid, concept, category), cnt in agg.items():
        out.append({
            "doc_key": doc_key,
            "lang": lang,
            "year": year,
            "order_in_year": order_in_year,
            "docid": docid,
            "concept": concept,
            "category": category,
            "count": cnt
        })
    out.sort(key=lambda x: (x["lang"], x["year"] or 9999, x["order_in_year"] or 9999, x["docid"] or "", x["category"], x["concept"]))
    return out


def build_matrix(profile: List[dict], lang: str) -> List[dict]:
    # doc_key -> row dict
    docs_meta = {}
    concepts = set()
    for r in profile:
        if r["lang"] != lang:
            continue
        docs_meta[r["doc_key"]] = {"doc_key": r["doc_key"], "year": r["year"], "order_in_year": r["order_in_year"], "docid": r["docid"]}
        concepts.add(r["concept"])

    # initialize matrix
    mat = {dk: {**meta} for dk, meta in docs_meta.items()}
    for dk in mat:
        for c in concepts:
            mat[dk][c] = 0

    for r in profile:
        if r["lang"] != lang:
            continue
        mat[r["doc_key"]][r["concept"]] += int(r["count"])

    rows = list(mat.values())
    rows.sort(key=lambda x: (x["year"] or 9999, x["order_in_year"] or 9999, x["docid"] or ""))
    return rows


def compute_changes(profile: List[dict], lang: str) -> List[dict]:
    # order docs
    docs = []
    for r in profile:
        if r["lang"] == lang:
            docs.append((r["doc_key"], r["year"], r["order_in_year"], r["docid"]))
    docs = sorted(set(docs), key=lambda x: (x[1] or 9999, x[2] or 9999, x[3] or ""))

    per_doc = defaultdict(dict)
    cat = {}
    for r in profile:
        if r["lang"] != lang:
            continue
        per_doc[r["doc_key"]][r["concept"]] = int(r["count"])
        cat[r["concept"]] = r["category"]

    rows = []
    prev = {}
    prev_key = None
    for doc_key, year, order_in_year, docid in docs:
        cur = per_doc.get(doc_key, {})
        concepts = set(prev.keys()) | set(cur.keys())
        for c in sorted(concepts):
            now = int(cur.get(c, 0))
            before = int(prev.get(c, 0))
            delta = now - before
            if now > 0 and before == 0:
                status = "NEW"
            elif now == 0 and before > 0:
                status = "LOST"
            elif now > 0 and before > 0 and delta != 0:
                status = "CHANGED"
            elif now > 0 and before > 0 and delta == 0:
                status = "SAME"
            else:
                status = "ZERO"
            rows.append({
                "lang": lang,
                "doc_key": doc_key,
                "year": year,
                "order_in_year": order_in_year,
                "docid": docid,
                "prev_doc_key": prev_key,
                "concept": c,
                "category": cat.get(c, ""),
                "count_prev": before,
                "count_now": now,
                "delta": delta,
                "status": status
            })
        prev = cur
        prev_key = doc_key

    rows.sort(key=lambda x: (x["year"] or 9999, x["order_in_year"] or 9999, x["docid"] or "", x["category"], x["concept"]))
    return rows


def extract_cn_char_ngrams(text: str, top_k: int = 80, n_min: int = 2, n_max: int = 4) -> List[dict]:
    s = re.sub(r"\s+", "", text)
    chars = [ch for ch in s if CN_KEEP_RE.match(ch)]
    if len(chars) < n_min:
        return []
    counts = Counter()
    L = len(chars)
    for n in range(n_min, n_max + 1):
        for i in range(L - n + 1):
            ng = "".join(chars[i:i+n])
            counts[ng] += 1
    return [{"ngram": ng, "count": c} for ng, c in counts.most_common(top_k)]


def tfidf_candidates_cn(docs: List[Doc], top_k_per_doc: int = 40, min_df: int = 2, n_min: int = 2, n_max: int = 4) -> List[dict]:
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
        dk = make_doc_key(d)
        per_doc_counts[dk] = c
        for ng in c.keys():
            df_counts[ng] += 1

    N = max(1, len(per_doc_counts))
    vocab = {ng for ng, df in df_counts.items() if df >= min_df}

    out = []
    for dk, c in per_doc_counts.items():
        total = sum(v for ng, v in c.items() if ng in vocab)
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
            out.append({"doc_key": dk, "ngram": ng, "count": int(tf_raw), "tfidf": float(score)})
    return out


def compare_cn_en(detail_rows: List[dict], concept_map_rows: List[dict]) -> List[dict]:
    comparable = set(r.get("concept", "").strip() for r in concept_map_rows if r.get("concept", "").strip())
    # docid, lang, concept -> sum
    agg = defaultdict(int)
    for r in detail_rows:
        concept = r.get("concept", "")
        if concept not in comparable:
            continue
        docid = r.get("docid", "")
        lang = r.get("lang", "")
        agg[(docid, lang, concept)] += int(r.get("count", 0))

    # merge
    all_keys = set()
    for (docid, lang, concept) in agg.keys():
        if docid:
            all_keys.add((docid, concept))

    out = []
    for docid, concept in sorted(all_keys):
        cn = agg.get((docid, "CN", concept), 0)
        en = agg.get((docid, "EN", concept), 0)
        diff = en - cn
        flag = ""
        if cn > 0 and en == 0:
            flag = "CN>0, EN=0 (gal prarasta vertime?)"
        elif cn == 0 and en > 0:
            flag = "CN=0, EN>0 (gal pridėta vertime?)"
        elif cn > 0 and en > 0 and abs(diff) >= 3:
            flag = "Skirtumas >= 3 (patikrinti)"
        out.append({"docid": docid, "concept": concept, "count_cn": cn, "count_en": en, "diff_en_minus_cn": diff, "flag": flag})
    return out


# ----------------------------
# UI
# ----------------------------
st.title("Discourse Analyzer (CN 简体 ↔ EN) — stabilus deploy (be pandas)")
st.caption("Jei Streamlit Cloud naudoja Python 3.13, pandas kartais kompiliuojasi labai ilgai. Ši versija veikia be pandas.")

with st.sidebar:
    st.header("1) Dokumentai")
    cn_files = st.file_uploader("CN originalai (.txt, .docx)", type=["txt", "docx", "doc"], accept_multiple_files=True)
    en_files = st.file_uploader("EN vertimai (.txt, .docx)", type=["txt", "docx", "doc"], accept_multiple_files=True)

    st.header("2) Žodynai (CSV)")
    cn_terms_file = st.file_uploader("terms_cn.csv (concept, term, category)", type=["csv"])
    en_terms_file = st.file_uploader("terms_en.csv (concept, term, category)", type=["csv"])
    map_file = st.file_uploader("concept_map.csv (bent stulpelis: concept)", type=["csv"])

    st.header("3) Nustatymai")
    show_zero = st.checkbox("Rodyti 0 dažnius detail lentelėje", value=False)
    top_ngrams = st.slider("Candidates: Top n-grams per CN doc", 20, 200, 80, 10)
    tfidf_top = st.slider("Candidates: Top TF-IDF per CN doc", 10, 100, 40, 5)
    tfidf_min_df = st.slider("Candidates: min_df", 1, 10, 2, 1)

run = st.button("▶️ Analizuoti", type="primary")

if not run:
    st.info("Įkelk failus ir paspausk **Analizuoti**.")
    st.stop()

try:
    if not cn_files:
        st.warning("Įkelk bent 1 CN dokumentą.")
        st.stop()
    if not cn_terms_file:
        st.warning("Įkelk terms_cn.csv.")
        st.stop()

    cn_terms_rows = load_csv(cn_terms_file)
    cn_terms = load_terms(cn_terms_rows, "CN")

    have_en = bool(en_files) and bool(en_terms_file)
    en_terms = []
    if have_en:
        en_terms_rows = load_csv(en_terms_file)
        en_terms = load_terms(en_terms_rows, "EN")

    cn_docs = sort_docs(ingest_files(cn_files, "CN"))
    en_docs = sort_docs(ingest_files(en_files, "EN")) if have_en else []

    missing_cn = [d.filename for d in cn_docs if not d.docid]
    if missing_cn:
        st.error("CN failams nepavyko nustatyti DocID iš pavadinimo (reikia ..._DOCID_CN...). Probleminiai: " + ", ".join(missing_cn))
        st.stop()

    if have_en:
        missing_en = [d.filename for d in en_docs if not d.docid]
        if missing_en:
            st.error("EN failams nepavyko nustatyti DocID iš pavadinimo (reikia ..._DOCID_EN...). Probleminiai: " + ", ".join(missing_en))
            st.stop()

    detail_cn = analyze_docs(cn_docs, cn_terms, "CN")
    detail_en = analyze_docs(en_docs, en_terms, "EN") if have_en else []
    detail_all = detail_cn + detail_en

    detail_view = detail_all if show_zero else [r for r in detail_all if int(r["count"]) > 0]

    profile = doc_profile(detail_all)
    cn_matrix = build_matrix(profile, "CN")
    en_matrix = build_matrix(profile, "EN") if have_en else []
    cn_changes = compute_changes(profile, "CN")
    en_changes = compute_changes(profile, "EN") if have_en else []

    # Candidates CN
    cn_ngrams = []
    for d in cn_docs:
        for r in extract_cn_char_ngrams(d.text, top_k=top_ngrams):
            cn_ngrams.append({
                "doc_key": make_doc_key(d),
                "year": d.year,
                "order_in_year": d.order_in_year,
                "docid": d.docid,
                "ngram": r["ngram"],
                "count": r["count"]
            })
    cn_tfidf = tfidf_candidates_cn(cn_docs, top_k_per_doc=tfidf_top, min_df=tfidf_min_df)

    cmp_rows = []
    if have_en and map_file:
        concept_map_rows = load_csv(map_file)
        cmp_rows = compare_cn_en(detail_all, concept_map_rows)

    tabs = ["Dokumentų tvarka", "Detail", "Profilis (concept)", "Matrica", "NEW/LOST", "Candidates (CN)", "CN↔EN patikra"]
    tab_objs = st.tabs(tabs)

    with tab_objs[0]:
        st.subheader("Chronologinė tvarka")
        cn_order = [{
            "lang": d.lang, "year": d.year, "order_in_year": d.order_in_year,
            "docid": d.docid, "filename": d.filename, "doc_key": make_doc_key(d)
        } for d in cn_docs]
        st.markdown("### CN")
        st.dataframe(cn_order)
        if have_en:
            en_order = [{
                "lang": d.lang, "year": d.year, "order_in_year": d.order_in_year,
                "docid": d.docid, "filename": d.filename, "doc_key": make_doc_key(d)
            } for d in en_docs]
            st.markdown("### EN")
            st.dataframe(en_order)

    with tab_objs[1]:
        st.subheader("Detail: term dažniai per dokumentą")
        st.dataframe(detail_view)
        fields = ["lang","year","order_in_year","docid","filename","doc_key","concept","term","category","count"]
        st.download_button("⬇️ detail_counts.csv", data=to_csv_bytes(detail_view, fields), file_name="detail_counts.csv")

    with tab_objs[2]:
        st.subheader("Profilis: concept (sumuojant term variantus)")
        st.dataframe(profile)
        fields = ["doc_key","lang","year","order_in_year","docid","concept","category","count"]
        st.download_button("⬇️ doc_profile_concept.csv", data=to_csv_bytes(profile, fields), file_name="doc_profile_concept.csv")

    with tab_objs[3]:
        st.subheader("Matrica: Document × Concept")
        st.markdown("### CN")
        st.dataframe(cn_matrix)
        if cn_matrix:
            st.download_button("⬇️ cn_matrix.csv", data=to_csv_bytes(cn_matrix, list(cn_matrix[0].keys())), file_name="cn_matrix.csv")
        if have_en:
            st.markdown("### EN")
            st.dataframe(en_matrix)
            if en_matrix:
                st.download_button("⬇️ en_matrix.csv", data=to_csv_bytes(en_matrix, list(en_matrix[0].keys())), file_name="en_matrix.csv")

    with tab_objs[4]:
        st.subheader("Pokyčiai tarp gretimų dokumentų: NEW / LOST / DELTA")
        st.markdown("### CN")
        st.dataframe(cn_changes)
        fields = ["lang","doc_key","year","order_in_year","docid","prev_doc_key","concept","category","count_prev","count_now","delta","status"]
        st.download_button("⬇️ cn_changes.csv", data=to_csv_bytes(cn_changes, fields), file_name="cn_changes.csv")
        if have_en:
            st.markdown("### EN")
            st.dataframe(en_changes)
            st.download_button("⬇️ en_changes.csv", data=to_csv_bytes(en_changes, fields), file_name="en_changes.csv")

    with tab_objs[5]:
        st.subheader("Candidates (CN): top n-grams + TF-IDF")
        st.markdown("### Top n-grams (2–4)")
        st.dataframe(cn_ngrams)
        if cn_ngrams:
            st.download_button("⬇️ cn_ngrams.csv", data=to_csv_bytes(cn_ngrams, list(cn_ngrams[0].keys())), file_name="cn_ngrams.csv")
        st.markdown("### TF-IDF kandidatai")
        st.dataframe(cn_tfidf)
        if cn_tfidf:
            st.download_button("⬇️ cn_tfidf_candidates.csv", data=to_csv_bytes(cn_tfidf, list(cn_tfidf[0].keys())), file_name="cn_tfidf_candidates.csv")

    with tab_objs[6]:
        st.subheader("CN ↔ EN patikra (DocID + concept_map)")
        if not (have_en and map_file):
            st.info("Įkelk EN failus + terms_en.csv + concept_map.csv, kad matytum palyginimą.")
        else:
            st.dataframe(cmp_rows)
            if cmp_rows:
                st.download_button("⬇️ cn_en_compare.csv", data=to_csv_bytes(cmp_rows, list(cmp_rows[0].keys())), file_name="cn_en_compare.csv")

    st.success("Baigta ✅")

except Exception as e:
    st.error(f"Klaida: {e}")
