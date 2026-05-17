# app.py
# Version: V1.5

import html as html_module
import io
import re
import time
from pathlib import Path
from collections import defaultdict
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

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False
    plt = None  # type: ignore

THULAC_IMPORT_ERROR: Optional[str] = None
try:
    import thulac
    THULAC_AVAILABLE = True
except Exception as e:
    THULAC_AVAILABLE = False
    THULAC_IMPORT_ERROR = str(e)

HANLP_IMPORT_ERROR: Optional[str] = None
try:
    import hanlp
    HANLP_AVAILABLE = True
except Exception as e:
    HANLP_AVAILABLE = False
    HANLP_IMPORT_ERROR = str(e)


# -----------------------------
# Config
# -----------------------------
APP_VERSION = "V1.5"

_hanlp_tokenizer = None  # lazy-loaded
_hanlp_load_error: Optional[str] = None  # model load failure message
_thulac_tokenizer = None  # lazy-loaded
_thulac_load_error: Optional[str] = None  # model load failure message

st.set_page_config(
    page_title="Sinocode Discourse Analyzer (CN)",
    layout="wide",
)

DEFAULT_TERMS_PATH = "terms_cn.csv"  # keep in repo root


def _inject_sinocode_title_css() -> None:
    """Google Fonts + SINOCODE title sizing only (default Streamlit chrome/background)."""
    css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Ma+Shan+Zheng&family=Noto+Serif+SC:wght@400;600&display=swap');
.sinocode-header {
  line-height: 1.2;
}
/* Explicit rem clamps — Ma Shan Zheng x-height is small vs Latin, so em-based 2× looked equal */
.sinocode-title {
  font-family: "Ma Shan Zheng", "Noto Serif SC", serif;
  font-size: clamp(2.44rem, 8.25vw, 4.31rem) !important;
  letter-spacing: 0.12em;
  color: #1a1a1a;
  margin: 0 0 0.15rem 0;
  line-height: 1.05;
}
.sinocode-sub {
  font-family: "Noto Serif SC", "Georgia", serif;
  font-size: clamp(1rem, 2.6vw, 1.35rem) !important;
  color: #333;
  margin: 0 0 0.75rem 0;
}
</style>
"""
    st.markdown(css, unsafe_allow_html=True)


_inject_sinocode_title_css()


def _get_hanlp_tokenizer():
    """Lazy-load HanLP tokenizer (COARSE_ELECTRA_SMALL_ZH). Returns None if unavailable."""
    global _hanlp_tokenizer, _hanlp_load_error
    if not HANLP_AVAILABLE:
        return None
    if _hanlp_tokenizer is not None:
        return _hanlp_tokenizer
    try:
        _hanlp_tokenizer = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
        _hanlp_load_error = None  # clear previous load error on success
        return _hanlp_tokenizer
    except Exception as e:
        _hanlp_load_error = str(e)
        return None


def _hanlp_tokenize(text: str) -> List[str]:
    """Segment text with HanLP; returns flat list of token strings for the whole text."""
    tok = _get_hanlp_tokenizer()
    if tok is None:
        return []
    if not text or not text.strip():
        return []
    try:
        result = tok([text])
        if not result:
            return []
        # HanLP may return a dict/Document (e.g. tok/coarse, tok/fine) or list of token lists
        if isinstance(result, dict):
            for key in ("tok/coarse", "tok_fine", "tok", "tok/fine"):
                if key in result and result[key]:
                    result = result[key]
                    break
            else:
                result = list(result.values())[0] if result else []
        flat: List[str] = []
        for item in result:
            if isinstance(item, (list, tuple)):
                flat.extend(safe_str(t) for t in item if safe_str(t).strip())
            else:
                s = safe_str(item).strip()
                if s:
                    flat.append(s)
        return flat
    except Exception:
        return []


def _hanlp_ngram_counter(tokens: List[str], max_n: int = 8) -> Dict[str, int]:
    """
    Build count of 1-gram, 2-gram, ... max_n-gram (concatenated) so that
    dictionary terms matching multiple consecutive HanLP tokens are counted.
    """
    counter: Dict[str, int] = {}
    n = len(tokens)
    for k in range(1, min(max_n + 1, n + 1)):
        for i in range(n - k + 1):
            ngram = "".join(tokens[i : i + k])
            if ngram:
                counter[ngram] = counter.get(ngram, 0) + 1
    return counter


def _get_thulac_tokenizer():
    """Lazy-load THULAC tokenizer. Returns None if unavailable."""
    global _thulac_tokenizer, _thulac_load_error
    if not THULAC_AVAILABLE:
        return None
    if _thulac_tokenizer is not None:
        return _thulac_tokenizer
    try:
        _thulac_tokenizer = thulac.thulac(seg_only=True)
        _thulac_load_error = None
        return _thulac_tokenizer
    except Exception as e:
        _thulac_load_error = str(e)
        return None


def _thulac_tokenize(text: str) -> List[str]:
    """Segment text with THULAC and return flat token list."""
    tok = _get_thulac_tokenizer()
    if tok is None:
        return []
    if not text or not text.strip():
        return []
    try:
        items = tok.cut(text, text=False)
        out: List[str] = []
        for it in items:
            if isinstance(it, (tuple, list)) and it:
                s = safe_str(it[0]).strip()
            else:
                s = safe_str(it).strip()
            if s:
                out.append(s)
        return out
    except Exception:
        return []


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
    # Tolerate accidental trailing commas in header cells, e.g. "category,"
    df.columns = [c.strip().lower().rstrip(",") for c in df.columns]

    required = ["concept", "term", "pinyin", "translation", "category"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"terms_cn.csv is missing required columns: {missing}. "
            f"Expected: concept;term;pinyin;translation;category"
        )

    # Clean whitespace and tolerate accidental trailing commas in values
    for c in required:
        df[c] = df[c].astype(str).map(lambda s: s.strip().rstrip(","))

    # Drop empty terms
    df = df[df["term"].map(lambda x: len(x) > 0)].copy()

    # Deduplicate exact rows
    df = df.drop_duplicates(subset=["concept", "term", "pinyin", "translation", "category"]).reset_index(drop=True)

    return df


def is_civilization_lexicon(terms_df: pd.DataFrame) -> bool:
    """
    Heuristic: the bundled civilization dictionary marks metaphor lemmas with concept == "Metaphor"
    and uses a category string mentioning "conceptual metaphor".
    """
    if terms_df is None or terms_df.empty:
        return False
    if "concept" not in terms_df.columns or "category" not in terms_df.columns:
        return False
    conc = terms_df["concept"].astype(str).str.strip()
    cat = terms_df["category"].astype(str).str.lower()
    if (conc == "Metaphor").any():
        return True
    return cat.str.contains("conceptual metaphor", na=False).any()


def count_substring_occurrences(text: str, term: str) -> int:
    """
    Count non-overlapping occurrences of `term` in `text`.
    For Chinese terms (multi-char), this is usually fine.
    """
    if not term:
        return 0
    pattern = re.escape(term)
    return len(re.findall(pattern, text))


def count_longest_non_overlapping_matches(text: str, terms: List[str]) -> Dict[str, int]:
    """
    Grąžina term -> count, taikant strategiją „pirmiausia ilgiausi atitikmenys“.
    Trumpesni terminai, persidengiantys su jau pažymėtais intervalais, neskaičiuojami.
    """
    text_len = len(text)
    if text_len == 0 or not terms:
        return {t: 0 for t in terms}

    unique_terms = sorted(set(terms), key=len, reverse=True)
    occupied = [False] * text_len
    counts: Dict[str, int] = {t: 0 for t in unique_terms}

    for term in unique_terms:
        if not term:
            continue
        pattern = re.escape(term)
        for m in re.finditer(pattern, text):
            start, end = m.start(), m.end()
            if any(occupied[i] for i in range(start, min(end, text_len))):
                continue
            counts[term] += 1
            for i in range(start, min(end, text_len)):
                occupied[i] = True

    return counts


def get_longest_match_positions(
    text: str,
    terms: List[str],
    term_to_meta: Dict[str, Tuple[str, str]],
) -> List[Tuple[int, int, str, str, str]]:
    """
    Same logic as count_longest_non_overlapping_matches but returns
    (start, end, term, pinyin, translation) for each match.
    """
    text_len = len(text)
    if text_len == 0 or not terms:
        return []
    unique_terms = sorted(set(terms), key=len, reverse=True)
    occupied = [False] * text_len
    out: List[Tuple[int, int, str, str, str]] = []
    for term in unique_terms:
        if not term:
            continue
        pinyin = term_to_meta.get(term, ("", ""))[0]
        translation = term_to_meta.get(term, ("", ""))[1]
        pattern = re.escape(term)
        for m in re.finditer(pattern, text):
            start, end = m.start(), m.end()
            if any(occupied[i] for i in range(start, min(end, text_len))):
                continue
            out.append((start, end, term, pinyin, translation))
            for i in range(start, min(end, text_len)):
                occupied[i] = True
    return sorted(out, key=lambda x: x[0])


def get_jieba_match_positions(text: str, term_hits: pd.DataFrame) -> List[Tuple[int, int, str, str, str]]:
    """
    For jieba modes: find all substring matches for each term in term_hits (Count > 0),
    then sort by start and drop overlapping spans (keep first).
    """
    if term_hits.empty:
        return []
    raw: List[Tuple[int, int, str, str, str]] = []
    for _, r in term_hits.iterrows():
        term = safe_str(r["CH term"])
        if not term or int(r.get("Count", 0)) <= 0:
            continue
        pinyin = safe_str(r.get("Pinyin", ""))
        translation = safe_str(r.get("ENG translation", ""))
        for m in re.finditer(re.escape(term), text):
            raw.append((m.start(), m.end(), term, pinyin, translation))
    raw.sort(key=lambda x: x[0])
    out: List[Tuple[int, int, str, str, str]] = []
    last_end = 0
    for start, end, term, pinyin, translation in raw:
        if start < last_end:
            continue
        out.append((start, end, term, pinyin, translation))
        last_end = end
    return out


# Palette for term highlighting (distinct colors)
HIGHLIGHT_PALETTE = [
    "#ffcccc", "#ccffcc", "#ccccff", "#ffffcc", "#ffccff", "#ccffff",
    "#ffddbb", "#ddffbb", "#bbddff", "#ffbbdd", "#ddbbff", "#bbffdd",
    "#e6ccff", "#cce6ff", "#ffe6cc",
]


def load_economist_watchlist(txt_path: Optional[str] = None) -> List[Tuple[str, str]]:
    """
    Load semicolon watchlist (e.g. economist_terms.txt): concept;CH_term;pinyin;english;category.
    Returns (CH term, English label) in file order; first occurrence wins if duplicates.
    Relative txt_path is resolved against the app directory (same folder as app.py).
    """
    base = Path(__file__).resolve().parent
    if txt_path:
        p = Path(txt_path)
        path = p if p.is_absolute() else (base / p)
    else:
        path = base / "economist_terms.txt"
    if not path.is_file():
        return []
    out: List[Tuple[str, str]] = []
    seen_ch: set = set()
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(";")
        if len(parts) < 4:
            continue
        ch = safe_str(parts[1]).strip()
        eng = safe_str(parts[3]).strip()
        if not ch:
            continue
        if ch in seen_ch:
            continue
        seen_ch.add(ch)
        out.append((ch, eng if eng else ch))
    return out


def build_economist_year_share_table(
    term_hits_all_docs: pd.DataFrame,
    watchlist: List[Tuple[str, str]],
    year_from: int = 2017,
    year_to: int = 2025,
) -> pd.DataFrame:
    """
    For each watchlist CH term: share (%) of that term's hits vs all dictionary hits
    in the same calendar year (term_hits_all_docs).
    Rows: English label only; columns: Term (EN), 2017..year_to.
    """
    year_cols = [str(y) for y in range(year_from, year_to + 1)]
    if not watchlist:
        return pd.DataFrame(columns=["Term (EN)"] + year_cols)
    if term_hits_all_docs is None or term_hits_all_docs.empty:
        return pd.DataFrame(
            [{"Term (EN)": eng, **{c: 0.0 for c in year_cols}} for _ch, eng in watchlist]
        )

    th = term_hits_all_docs.copy()
    th["year_int"] = th["year"].map(safe_year)
    th = th.dropna(subset=["year_int"])
    if th.empty:
        return pd.DataFrame(
            [{"Term (EN)": eng, **{c: 0.0 for c in year_cols}} for _ch, eng in watchlist]
        )
    th["year_int"] = th["year_int"].astype(int)

    rows_out: List[Dict[str, object]] = []
    for ch, eng in watchlist:
        row: Dict[str, object] = {"Term (EN)": eng}
        for y in range(year_from, year_to + 1):
            sub_y = th[th["year_int"] == y]
            total = int(sub_y["Count"].sum()) if not sub_y.empty else 0
            sub_t = sub_y[sub_y["CH term"] == ch]
            tcount = int(sub_t["Count"].sum()) if not sub_t.empty else 0
            row[str(y)] = round((tcount / total * 100.0), 2) if total > 0 else 0.0
        rows_out.append(row)
    return pd.DataFrame(rows_out)


def build_economist_share_mini_chart_figure(
    title_en: str,
    years: List[int],
    pct_values: List[float],
    bar_color: str = "#e64c3c",
    border_color: Optional[str] = None,
    border_width: float = 0.0,
):
    """
    Small vertical bar chart: cream background, colored bars (default Economist orange-red),
    Y-axis ticks on the right, light horizontal grid.
    """
    if not MATPLOTLIB_AVAILABLE or plt is None:
        return None
    if len(years) != len(pct_values):
        raise ValueError("years and pct_values length mismatch")

    bg = "#f4f1ea"
    n_years = len(years)
    # Default was too narrow for 9× four-digit years — labels overlapped when st.pyplot scales the figure.
    fig_w = max(3.0, min(9.0, 0.48 * n_years))
    fig_h = 2.05 if n_years >= 7 else 1.85
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=120, facecolor=bg)
    if border_color and border_width > 0:
        fig.patch.set_edgecolor(border_color)
        fig.patch.set_linewidth(border_width)
    ax.set_facecolor(bg)

    x = list(range(len(years)))
    bar_w = min(0.72, max(0.35, 0.82 / max(n_years, 1)))
    ax.bar(
        x,
        pct_values,
        color=bar_color,
        width=bar_w,
        align="center",
        edgecolor="none",
        zorder=2,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years], fontsize=6.5, color="#333333")
    plt.setp(ax.get_xticklabels(), rotation=42, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", length=3, width=0.6, color="#888888", pad=1)

    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.tick_params(axis="y", labelsize=7, left=False, right=True, colors="#333333")
    ax.tick_params(axis="y", length=3, width=0.6, color="#888888")

    ax.set_title(title_en, loc="left", fontsize=9.5, fontweight="normal", color="#222222", pad=6)
    ax.grid(True, axis="y", color="#cfc8bc", linestyle="-", linewidth=0.6, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)

    m = max(pct_values) if pct_values else 0.0
    if m <= 0:
        y_hi = 5.0
    else:
        y_hi = max(m * 1.18, m + 0.5)
    ax.set_ylim(0, y_hi)

    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#aaaaaa")
    ax.spines["right"].set_color("#aaaaaa")

    bottom_pad = 0.30 if n_years >= 7 else 0.26
    fig.subplots_adjust(left=0.10, right=0.92, top=0.78, bottom=bottom_pad)
    return fig


def render_watchlist_year_share_mini_charts(
    term_hits_all_docs: pd.DataFrame,
    watchlist: List[Tuple[str, str]],
    bar_color: str,
    year_from: int = 2017,
    year_to: int = 2025,
) -> None:
    """
    Streamlit: grid of mini bar charts (or dataframe if matplotlib missing).
    Caller must ensure watchlist is non-empty.
    """
    years = list(range(year_from, year_to + 1))
    share_df = build_economist_year_share_table(
        term_hits_all_docs, watchlist, year_from=year_from, year_to=year_to
    )
    if MATPLOTLIB_AVAILABLE and plt is not None:
        records = share_df.to_dict("records")
        ncol = 2
        for i in range(0, len(records), ncol):
            chunk = records[i : i + ncol]
            st_cols = st.columns(len(chunk))
            for col, rec in zip(st_cols, chunk):
                with col:
                    title = safe_str(rec.get("Term (EN)", ""))
                    vals = [float(rec.get(str(y), 0.0)) for y in years]
                    fig = build_economist_share_mini_chart_figure(
                        title, years, vals, bar_color=bar_color
                    )
                    if fig is not None:
                        st.pyplot(fig, width="stretch")
                        plt.close(fig)
    else:
        st.warning(
            "Matplotlib nepasiekiamas — rodoma lentelė. "
            "Įdiekite: pip install matplotlib"
        )
        st.dataframe(share_df, width="stretch", hide_index=True)


def load_events(csv_path: str = "events_CN_2017_2026.csv") -> pd.DataFrame:
    """
    Load external events (Party congresses, BRICS, G20, etc.) used to contextualize discourse shifts.
    """
    try:
        df = pd.read_csv(csv_path, sep=";")
    except FileNotFoundError:
        return pd.DataFrame(columns=["id", "year", "date_start", "date_end", "title", "category"])
    for col in ("date_start", "date_end"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    else:
        df["year"] = pd.NA
    return df


def suggest_events_for_period(events_df: pd.DataFrame, period: str, max_events: int = 3) -> List[str]:
    """
    Given a period label like '2019-2020', return a short list of high-priority events that fall in that year span.
    """
    if events_df.empty:
        return []
    try:
        y1_str, y2_str = period.split("-")
        y1, y2 = int(y1_str), int(y2_str)
    except Exception:
        return []
    mask = (events_df["year"] >= y1) & (events_df["year"] <= y2)
    sub = events_df[mask].copy()
    if sub.empty:
        return []
    priority_keywords = [
        "Congress",
        "Plenum",
        "Five-Year Plan",
        "NPC",
        "Central Economic Work Conference",
        "BRICS",
        "Belt and Road",
        "G20",
        "SCO",
        "FOCAC",
        "AI",
        "COVID",
        "Hong Kong",
    ]

    def _score(row: pd.Series) -> int:
        title = safe_str(row.get("title", ""))
        t_lower = title.lower()
        return sum(1 for kw in priority_keywords if kw.lower() in t_lower)

    sub["__score"] = sub.apply(_score, axis=1)
    sub = sub.sort_values(["__score", "year"], ascending=[False, True])
    rows = sub.head(max_events)
    return [f"{int(r['year'])}: {safe_str(r['title'])} ({safe_str(r['category'])})" for _, r in rows.iterrows()]


def _mean_share_per_item(
    term_hits_all_docs: pd.DataFrame,
    item_col: str,
) -> pd.Series:
    """
    Compute mean Share (%) per item across all years present in term_hits_all_docs.
    Share(year) = item_count(year) / total_count(year) * 100
    mean_share = mean over years of Share(year)
    """
    if term_hits_all_docs is None or term_hits_all_docs.empty:
        return pd.Series(dtype=float)

    df = term_hits_all_docs.copy()
    if "year" not in df.columns or item_col not in df.columns or "Count" not in df.columns:
        return pd.Series(dtype=float)

    df["year_int"] = df["year"].map(lambda y: safe_year(safe_str(y)))
    df = df.dropna(subset=["year_int"]).copy()
    if df.empty:
        return pd.Series(dtype=float)
    df["year_int"] = df["year_int"].astype(int)

    item_year = (
        df.groupby(["year_int", item_col], as_index=False)["Count"]
        .sum()
        .rename(columns={"Count": "item_count"})
    )
    total_year = (
        df.groupby("year_int", as_index=False)["Count"]
        .sum()
        .rename(columns={"Count": "total_count"})
    )
    item_year = item_year.merge(total_year, on="year_int", how="left")
    if item_year.empty:
        return pd.Series(dtype=float)

    item_year["share"] = (item_year["item_count"] / item_year["total_count"]) * 100.0
    mean_share = item_year.groupby(item_col)["share"].mean()
    return mean_share


def _yellow_styler(df: pd.DataFrame, color: str = "#fff2a8") -> "pd.io.formats.style.Styler":
    """Styler with yellow background for all data cells (used for Compare tables)."""
    def _cell(_v):
        return f"background-color: {color};"

    return df.style.applymap(_cell)


def _trend_slope(years: List[int], vals: List[float]) -> float:
    """Simple linear trend slope for yearly values."""
    if not years or not vals or len(years) != len(vals):
        return 0.0
    n = len(years)
    x_mean = sum(years) / n
    y_mean = sum(vals) / n
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(years, vals))
    den = sum((x - x_mean) ** 2 for x in years)
    if den == 0:
        return 0.0
    return num / den


def _share_series_for_term(
    term_hits_all_docs: pd.DataFrame,
    ch_term: str,
    year_from: int = 2017,
    year_to: int = 2025,
) -> Tuple[List[int], List[float]]:
    years = list(range(year_from, year_to + 1))
    if term_hits_all_docs is None or term_hits_all_docs.empty:
        return years, [0.0 for _ in years]
    th = term_hits_all_docs.copy()
    th["year_int"] = th["year"].map(safe_year)
    th = th.dropna(subset=["year_int"]).copy()
    if th.empty:
        return years, [0.0 for _ in years]
    th["year_int"] = th["year_int"].astype(int)
    vals: List[float] = []
    for y in years:
        sub_y = th[th["year_int"] == y]
        total = int(sub_y["Count"].sum()) if not sub_y.empty else 0
        sub_t = sub_y[sub_y["CH term"] == ch_term]
        tcount = int(sub_t["Count"].sum()) if not sub_t.empty else 0
        vals.append(round((tcount / total * 100.0), 4) if total > 0 else 0.0)
    return years, vals


def compute_watchlist_set_compare(
    data1: Dict,
    data2: Dict,
    watchlist: List[Tuple[str, str]],
    year_from: int = 2017,
    year_to: int = 2025,
) -> pd.DataFrame:
    """Compare watchlist terms between Set 1 and Set 2 using mean share and trend slope."""
    df1 = (data1 or {}).get("term_hits_all_docs")
    df2 = (data2 or {}).get("term_hits_all_docs")
    rows: List[Dict[str, object]] = []
    for ch, en in watchlist:
        years1, s1_vals = _share_series_for_term(df1, ch, year_from=year_from, year_to=year_to)
        years2, s2_vals = _share_series_for_term(df2, ch, year_from=year_from, year_to=year_to)
        s1_mean = sum(s1_vals) / len(s1_vals) if s1_vals else 0.0
        s2_mean = sum(s2_vals) / len(s2_vals) if s2_vals else 0.0
        s1_slope = _trend_slope(years1, s1_vals)
        s2_slope = _trend_slope(years2, s2_vals)
        d_mean = s1_mean - s2_mean
        d_slope = s1_slope - s2_slope
        eps = 0.03
        if abs(d_mean) < eps:
            winner = "Balanced"
        elif d_mean > 0:
            winner = "Set 1"
        else:
            winner = "Set 2"
        rows.append(
            {
                "CH term": ch,
                "Term (EN)": en,
                "Set1 mean Share (%)": round(s1_mean, 3),
                "Set2 mean Share (%)": round(s2_mean, 3),
                "Δ mean (S1-S2)": round(d_mean, 3),
                "Set1 trend slope": round(s1_slope, 4),
                "Set2 trend slope": round(s2_slope, 4),
                "Δ slope (S1-S2)": round(d_slope, 4),
                "Winner": winner,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["Winner", "Δ mean (S1-S2)"], ascending=[True, False]).reset_index(drop=True)
    return out


def _watchlist_compare_styler(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    """Highlight mean-share winner cells in compare table."""
    def _row_styles(row: pd.Series) -> List[str]:
        s1 = ""
        s2 = ""
        if row.get("Winner") == "Set 1":
            s1 = "background-color: #d9f2d9;"
        elif row.get("Winner") == "Set 2":
            s2 = "background-color: #d9f2d9;"
        elif row.get("Winner") == "Balanced":
            s1 = "background-color: #f9f3cf;"
            s2 = "background-color: #f9f3cf;"
        return [
            "",
            "",
            s1,
            s2,
            "",
            "",
            "",
            "",
            "",
        ]

    return df.style.apply(_row_styles, axis=1)


def compute_common_top15_concepts_terms(
    data1: Dict,
    data2: Dict,
    top_n: int = 15,
):
    """
    Common Top 15 Concepts and Terms between Set 1 and Set 2.
    - Candidates: mean_share > 0 in both sets
    - Score: min(mean_share_set1, mean_share_set2)
    - mean_share computed as mean( Share(Year) ) across years
    """
    df1 = (data1 or {}).get("term_hits_all_docs")
    df2 = (data2 or {}).get("term_hits_all_docs")
    if df1 is None or df2 is None or df1.empty or df2.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Concepts
    s1_conc = _mean_share_per_item(df1, "Concept").rename("set1_mean_share")
    s2_conc = _mean_share_per_item(df2, "Concept").rename("set2_mean_share")
    conc = (
        pd.concat([s1_conc, s2_conc], axis=1)
        .dropna(subset=["set1_mean_share", "set2_mean_share"])
        .copy()
    )
    conc = conc[(conc["set1_mean_share"] > 0) & (conc["set2_mean_share"] > 0)]
    if conc.empty:
        conc_out = pd.DataFrame()
    else:
        conc["score"] = conc[["set1_mean_share", "set2_mean_share"]].min(axis=1)
        conc_out = (
            conc.sort_values("score", ascending=False)
            .head(top_n)
            .reset_index()
            .rename(columns={"Concept": "Concept"})
        )
        conc_out.insert(0, "Rank", range(1, len(conc_out) + 1))
        conc_out = conc_out.rename(
            columns={
                "set1_mean_share": "Set1 mean Share (%)",
                "set2_mean_share": "Set2 mean Share (%)",
            }
        )
        conc_out["Score (min)"] = conc_out["score"]
        conc_out = conc_out.drop(columns=["score"])

    # Terms
    s1_term = _mean_share_per_item(df1, "CH term").rename("set1_mean_share")
    s2_term = _mean_share_per_item(df2, "CH term").rename("set2_mean_share")
    term = (
        pd.concat([s1_term, s2_term], axis=1)
        .dropna(subset=["set1_mean_share", "set2_mean_share"])
        .copy()
    )
    term = term[(term["set1_mean_share"] > 0) & (term["set2_mean_share"] > 0)]

    if term.empty:
        term_out = pd.DataFrame()
    else:
        term["score"] = term[["set1_mean_share", "set2_mean_share"]].min(axis=1)
        term_top = (
            term.sort_values("score", ascending=False)
            .head(top_n)
            .reset_index()
        )

        # Metadata for terms (translation/concept/category)
        meta_cols = ["CH term", "ENG translation", "Concept", "Category"]
        meta_df = pd.concat([df1[meta_cols], df2[meta_cols]], axis=0).dropna(subset=["CH term"]).copy()
        meta_df["CH term"] = meta_df["CH term"].astype(str)

        meta = (
            meta_df.groupby("CH term", as_index=False)
            .agg(
                **{
                    "ENG translation": ("ENG translation", "first"),
                    "Concept": ("Concept", lambda s: ", ".join(sorted(set(s.astype(str))))),
                    "Category": ("Category", lambda s: ", ".join(sorted(set(s.astype(str))))),
                }
            )
        )

        term_top = term_top.merge(meta, on="CH term", how="left")
        term_out = term_top.rename(
            columns={
                "CH term": "CH term",
                "set1_mean_share": "Set1 mean Share (%)",
                "set2_mean_share": "Set2 mean Share (%)",
            }
        )
        term_out.insert(0, "Rank", range(1, len(term_out) + 1))
        term_out["Score (min)"] = term_out["score"]
        term_out = term_out.drop(columns=["score"])

    # Round for readability
    if not conc_out.empty:
        for c in ["Set1 mean Share (%)", "Set2 mean Share (%)", "Score (min)"]:
            conc_out[c] = conc_out[c].map(lambda x: round(float(x), 3))
    if not term_out.empty:
        for c in ["Set1 mean Share (%)", "Set2 mean Share (%)", "Score (min)"]:
            term_out[c] = term_out[c].map(lambda x: round(float(x), 3))

    return conc_out, term_out


def _yoy_arrow_from_cell(cell) -> str:
    """Iš YOY lentelės langelio (pvz. '↑ (+1.23)') ištraukia pirmą rodyklę."""
    s = safe_str(cell).strip()
    if not s:
        return ""
    if s[0] in "↑↓→":
        return s[0]
    return ""


def _yoy_cell_display(sym: str, delta: float) -> str:
    """Rodomasis tekstas: simbolis + pokytis procentiniais punktais (share pp)."""
    return f"{sym} ({delta:+.2f})"


def build_highlight_html(text: str, spans: List[Tuple[int, int, str, str, str]]) -> str:
    """
    Build HTML with highlighted spans. Each unique term gets one color; tooltip = pinyin — translation.
    All text is HTML-escaped.
    """
    if not spans:
        return html_module.escape(text)
    seen_terms: List[str] = []
    term_to_color: Dict[str, str] = {}
    for _, _, term, _, _ in spans:
        if term not in term_to_color:
            term_to_color[term] = HIGHLIGHT_PALETTE[len(seen_terms) % len(HIGHLIGHT_PALETTE)]
            seen_terms.append(term)
    parts: List[str] = []
    pos = 0
    for start, end, term, pinyin, translation in spans:
        if start > pos:
            parts.append(html_module.escape(text[pos:start]))
        frag = text[start:end]
        title = f"{html_module.escape(pinyin)} — {html_module.escape(translation)}" if (pinyin or translation) else html_module.escape(term)
        color = term_to_color.get(term, HIGHLIGHT_PALETTE[0])
        parts.append(f'<span style="background-color: {color}; padding: 0 1px;" title="{title}">{html_module.escape(frag)}</span>')
        pos = end
    if pos < len(text):
        parts.append(html_module.escape(text[pos:]))
    return "".join(parts)


def get_jieba_segmented_display(text: str, mode: str) -> Optional[str]:
    """
    Return full text segmented by jieba with " | " between tokens.
    mode: "jieba_precise" -> jieba.cut, "jieba_search" -> jieba.cut_for_search.
    Returns None if jieba unavailable.
    """
    if not JIEBA_AVAILABLE:
        return None
    if mode == "jieba_search":
        tokens = jieba.cut_for_search(text)
    else:
        tokens = jieba.cut(text, cut_all=False)
    return " | ".join(safe_str(t) for t in tokens if safe_str(t).strip())


def build_segmented_highlighted_html(text: str, mode: str, term_hits: pd.DataFrame) -> str:
    """
    One combined view: original layout (line by line), token boundaries ( | ),
    and dictionary terms highlighted with color + tooltip (pinyin — translation).
    mode: "jieba_precise", "jieba_search", "thulac", or "hanlp".
    HanLP: one tokenization for full document, then tokens assigned to lines by character offsets.
    """
    use_jieba = mode in ("jieba_precise", "jieba_search") and JIEBA_AVAILABLE
    use_thulac = mode == "thulac" and THULAC_AVAILABLE
    use_hanlp = mode == "hanlp" and HANLP_AVAILABLE
    if not use_jieba and not use_thulac and not use_hanlp:
        return html_module.escape(text)
    term_to_meta: Dict[str, Tuple[str, str]] = {}
    term_to_color: Dict[str, str] = {}
    if not term_hits.empty:
        seen: List[str] = []
        for _, r in term_hits.iterrows():
            t = safe_str(r["CH term"])
            if t and t not in term_to_meta:
                term_to_meta[t] = (safe_str(r.get("Pinyin", "")), safe_str(r.get("ENG translation", "")))
                term_to_color[t] = HIGHLIGHT_PALETTE[len(seen) % len(HIGHLIGHT_PALETTE)]
                seen.append(t)
    lines = text.split("\n")
    # Line start offsets in full text (character index)
    line_starts: List[int] = []
    pos = 0
    for line in lines:
        line_starts.append(pos)
        pos += len(line) + 1  # +1 for \n

    if use_hanlp:
        tokens_full = _hanlp_tokenize(text)
        # Greedy token offsets: match each token in text left-to-right
        token_offsets: List[Tuple[int, int]] = []
        offset = 0
        for token in tokens_full:
            idx = text.find(token, offset)
            if idx == -1:
                break
            token_offsets.append((idx, idx + len(token)))
            offset = idx + len(token)
        # Which tokens belong to which line (by token start position)
        result_lines = []
        for i, line in enumerate(lines):
            line_start = line_starts[i]
            line_end = line_starts[i] + len(line)
            indices = [j for j in range(len(tokens_full)) if j < len(token_offsets) and line_start <= token_offsets[j][0] < line_end]
            line_tokens = [tokens_full[j] for j in indices]
            if not line_tokens:
                if line.strip():
                    result_lines.append(html_module.escape(line))
                else:
                    result_lines.append("")
                continue
            # Longest-match over token sequence: try 1..max_ngram tokens, match against term_to_meta
            token_parts = []
            max_ngram = 8
            idx = 0
            while idx < len(line_tokens):
                t_str = safe_str(line_tokens[idx]).strip()
                if not t_str:
                    idx += 1
                    continue
                matched = False
                for k in range(min(max_ngram, len(line_tokens) - idx), 0, -1):
                    phrase = "".join(safe_str(line_tokens[idx + j]).strip() for j in range(k))
                    if not phrase:
                        continue
                    if phrase in term_to_meta:
                        pinyin, trans = term_to_meta[phrase]
                        color = term_to_color.get(phrase, HIGHLIGHT_PALETTE[0])
                        title = f"{html_module.escape(pinyin)} — {html_module.escape(trans)}"
                        token_parts.append(
                            f'<span style="background-color: {color}; padding: 0 1px;" title="{title}">{html_module.escape(phrase)}</span>'
                        )
                        idx += k
                        matched = True
                        break
                if not matched:
                    token_parts.append(html_module.escape(t_str))
                    idx += 1
            result_lines.append(" | ".join(token_parts))
        return "\n".join(result_lines)

    if use_thulac:
        result_lines = []
        for line in lines:
            line_tokens = _thulac_tokenize(line)
            if not line_tokens:
                result_lines.append(html_module.escape(line) if line.strip() else "")
                continue
            token_parts = []
            max_ngram = 8
            idx = 0
            while idx < len(line_tokens):
                t_str = safe_str(line_tokens[idx]).strip()
                if not t_str:
                    idx += 1
                    continue
                matched = False
                for k in range(min(max_ngram, len(line_tokens) - idx), 0, -1):
                    phrase = "".join(safe_str(line_tokens[idx + j]).strip() for j in range(k))
                    if not phrase:
                        continue
                    if phrase in term_to_meta:
                        pinyin, trans = term_to_meta[phrase]
                        color = term_to_color.get(phrase, HIGHLIGHT_PALETTE[0])
                        title = f"{html_module.escape(pinyin)} — {html_module.escape(trans)}"
                        token_parts.append(
                            f'<span style="background-color: {color}; padding: 0 1px;" title="{title}">{html_module.escape(phrase)}</span>'
                        )
                        idx += k
                        matched = True
                        break
                if not matched:
                    token_parts.append(html_module.escape(t_str))
                    idx += 1
            result_lines.append(" | ".join(token_parts))
        return "\n".join(result_lines)

    result_lines = []
    for line in lines:
        if mode == "jieba_search":
            tokens = list(jieba.cut_for_search(line))
        else:
            tokens = list(jieba.cut(line, cut_all=False))
        token_parts: List[str] = []
        for t in tokens:
            t_str = safe_str(t).strip()
            if not t_str:
                continue
            if t_str in term_to_meta:
                pinyin, trans = term_to_meta[t_str]
                color = term_to_color.get(t_str, HIGHLIGHT_PALETTE[0])
                title = f"{html_module.escape(pinyin)} — {html_module.escape(trans)}"
                token_parts.append(
                    f'<span style="background-color: {color}; padding: 0 1px;" title="{title}">{html_module.escape(t_str)}</span>'
                )
            else:
                token_parts.append(html_module.escape(t_str))
        result_lines.append(" | ".join(token_parts))
    return "\n".join(result_lines)


def build_token_counter(text: str, mode: str) -> Optional[Dict[str, int]]:
    """
    mode:
      - jieba_precise: jieba.cut (precise)
      - jieba_search: jieba.cut_for_search
      - thulac: THULAC segmentatorius
      - hanlp: HanLP segmentatorius
    Returns dict token -> count, or None if segmentator unavailable.
    """
    if mode == "hanlp":
        if not HANLP_AVAILABLE:
            return None
        tokens = _hanlp_tokenize(text)
        return _hanlp_ngram_counter(tokens)
    if mode == "thulac":
        if not THULAC_AVAILABLE:
            return None
        tokens = _thulac_tokenize(text)
        return _hanlp_ngram_counter(tokens)
    elif not JIEBA_AVAILABLE:
        return None
    elif mode == "jieba_search":
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
      - hanlp (HanLP segmentatorius)
      - hybrid (default): substring for len>=2, jieba for single-char terms
    """
    text = normalize_text(text)

    token_counter_precise = None
    token_counter_search = None
    token_counter_thulac = None
    token_counter_hanlp = None

    if match_mode in ("jieba_precise", "hybrid") and JIEBA_AVAILABLE:
        token_counter_precise = build_token_counter(text, "jieba_precise")
    if match_mode == "jieba_search" and JIEBA_AVAILABLE:
        token_counter_search = build_token_counter(text, "jieba_search")
    if match_mode == "thulac" and THULAC_AVAILABLE:
        token_counter_thulac = build_token_counter(text, "thulac")
    if match_mode == "hanlp" and HANLP_AVAILABLE:
        token_counter_hanlp = build_token_counter(text, "hanlp")

    # Substring-based modes: use longest-match logic to avoid counting
    # shorter terms inside longer ones (e.g. AB inside ABCD)
    substring_counts: Optional[Dict[str, int]] = None
    if match_mode == "substring":
        terms_for_substring = [safe_str(r["term"]) for _, r in terms_df.iterrows() if safe_str(r["term"])]
        substring_counts = count_longest_non_overlapping_matches(text, terms_for_substring)
    elif match_mode == "hybrid":
        terms_for_substring = [safe_str(r["term"]) for _, r in terms_df.iterrows() if len(safe_str(r["term"])) >= 2]
        substring_counts = count_longest_non_overlapping_matches(text, terms_for_substring)

    rows = []
    for _, r in terms_df.iterrows():
        term = safe_str(r["term"])
        if not term:
            continue

        cnt = 0
        if match_mode == "substring":
            cnt = int(substring_counts.get(term, 0))

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

        elif match_mode == "hanlp":
            if token_counter_hanlp is None or len(token_counter_hanlp) == 0:
                cnt = count_substring_occurrences(text, term)  # fallback when HanLP unavailable or returned no tokens
            else:
                cnt = int(token_counter_hanlp.get(term, 0))

        elif match_mode == "thulac":
            if token_counter_thulac is None or len(token_counter_thulac) == 0:
                cnt = count_substring_occurrences(text, term)  # fallback when THULAC unavailable or returned no tokens
            else:
                cnt = int(token_counter_thulac.get(term, 0))

        elif match_mode == "hybrid":
            # Default for Chinese dictionaries:
            # - phrases (2+ chars) -> substring (stable for fixed expressions)
            # - single characters -> token counting (reduces overcounting in dense texts)
            if len(term) >= 2:
                cnt = int(substring_counts.get(term, 0))
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


def meta_key(filename: str, set_id: Optional[int] = None) -> str:
    if set_id is not None:
        return f"doc_meta::set{set_id}::{filename}"
    return f"doc_meta::{filename}"


def ensure_doc_meta(filename: str, set_id: Optional[int] = None):
    k = meta_key(filename, set_id)
    if k not in st.session_state:
        st.session_state[k] = {"year": "", "title_cn": ""}


def run_pipeline_for_set(
    terms_upload,
    files_upload_list,
    match_mode: str,
    run_label: str,
    set_id: int,
) -> Optional[Dict]:
    """
    Run full analysis pipeline for one set. Returns dict with terms_df, docs, doc_term_hits,
    doc_cat_hits, doc_conc_hits, docs_overview, docs_overview_export, term_hits_all_docs,
    doc_infos, years; or None if no documents (or for set 2 no terms).
    """
    if not files_upload_list:
        return None
    if set_id == 2 and terms_upload is None:
        return None
    try:
        terms_df_raw = load_terms_csv(terms_upload)
    except Exception:
        raise
    terms_df = terms_df_raw.copy()
    for c in ("concept", "term", "pinyin", "translation", "category"):
        terms_df[c] = terms_df[c].astype(str)

    docs: List[Tuple[str, str]] = []
    read_errors: Dict[str, str] = {}
    for f in files_upload_list:
        text, err = get_file_text(f)
        if err:
            read_errors[f.name] = err
        else:
            docs.append((f.name, text))
    if not docs:
        return None

    doc_term_hits: Dict[str, pd.DataFrame] = {}
    doc_cat_hits: Dict[str, pd.DataFrame] = {}
    doc_conc_hits: Dict[str, pd.DataFrame] = {}
    doc_rows = []
    for filename, text in docs:
        ensure_doc_meta(filename, set_id)
        inferred = parse_meta_from_filename(filename)
        mk = meta_key(filename, set_id)
        if not st.session_state[mk]["year"]:
            st.session_state[mk]["year"] = inferred["year"]
        if not st.session_state[mk]["title_cn"]:
            st.session_state[mk]["title_cn"] = inferred["title_cn"]
        term_hits = analyze_text(text, terms_df, match_mode=match_mode)
        cat_sum = category_summary(term_hits, terms_df)
        conc_sum = concept_summary(term_hits)
        doc_term_hits[filename] = term_hits
        doc_cat_hits[filename] = cat_sum
        doc_conc_hits[filename] = conc_sum
        total_hits = int(term_hits["Count"].sum()) if not term_hits.empty else 0
        doc_rows.append({
            "filename": filename,
            "year": st.session_state[mk]["year"],
            "title_cn": st.session_state[mk]["title_cn"],
            "chars": len(text),
            "total_hits": total_hits,
            "total_hits_per_10k_chars": normalize_per_10k_chars(total_hits, len(text)),
            "unique_terms": int(term_hits["CH term"].nunique()) if not term_hits.empty else 0,
            "unique_concepts": int(term_hits["Concept"].nunique()) if not term_hits.empty else 0,
            "unique_categories": int(term_hits["Category"].nunique()) if not term_hits.empty else 0,
        })

    docs_overview = pd.DataFrame(doc_rows)
    docs_overview["run_label"] = run_label
    docs_overview_export = docs_overview[[
        "run_label", "filename", "year", "title_cn", "chars",
        "total_hits", "total_hits_per_10k_chars", "unique_terms", "unique_concepts", "unique_categories",
    ]].copy()

    rows_all_terms = []
    for filename, _text in docs:
        mk = meta_key(filename, set_id)
        year = st.session_state[mk]["year"]
        th = doc_term_hits[filename]
        if th.empty:
            continue
        for _, r in th.iterrows():
            rows_all_terms.append({
                "run_label": run_label,
                "filename": filename,
                "year": year,
                "CH term": safe_str(r["CH term"]),
                "Pinyin": safe_str(r["Pinyin"]),
                "ENG translation": safe_str(r["ENG translation"]),
                "Concept": safe_str(r["Concept"]),
                "Category": safe_str(r["Category"]),
                "Count": int(r["Count"]),
            })
    term_hits_all_docs = pd.DataFrame(rows_all_terms)

    doc_infos = []
    for idx, (fn, _text) in enumerate(docs):
        y, m, d = parse_year_month_day_from_filename(fn)
        doc_infos.append({"idx": idx, "filename": fn, "year": y, "month": m, "day": d})
    years = sorted({di["year"] for di in doc_infos if di["year"]}, key=lambda x: int(x))
    if any(not di["year"] for di in doc_infos):
        years.append("Unknown")

    return {
        "terms_df": terms_df,
        "docs": docs,
        "doc_term_hits": doc_term_hits,
        "doc_cat_hits": doc_cat_hits,
        "doc_conc_hits": doc_conc_hits,
        "docs_overview": docs_overview,
        "docs_overview_export": docs_overview_export,
        "term_hits_all_docs": term_hits_all_docs,
        "doc_infos": doc_infos,
        "years": years,
        "read_errors": read_errors,
    }


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


def describe_mode_runtime(mode: str) -> str:
    """Human-readable runtime status for a match mode (active engine vs fallback)."""
    if mode == "hanlp":
        if not HANLP_AVAILABLE:
            return "Engine: HanLP nepasiekiamas -> substring fallback."
        if _hanlp_load_error:
            err_preview = (_hanlp_load_error[:120] + "…") if len(_hanlp_load_error) > 120 else _hanlp_load_error
            return f"Engine: HanLP modelis nepakrautas -> substring fallback. Klaida: {err_preview}"
        return "Engine: HanLP aktyvus."

    if mode in ("jieba_precise", "jieba_search"):
        if not JIEBA_AVAILABLE:
            return "Engine: jieba nepasiekiama -> substring fallback."
        return "Engine: jieba aktyvi."

    if mode == "thulac":
        if not THULAC_AVAILABLE:
            return "Engine: THULAC nepasiekiamas -> substring fallback."
        if _thulac_load_error:
            err_preview = (_thulac_load_error[:120] + "…") if len(_thulac_load_error) > 120 else _thulac_load_error
            return f"Engine: THULAC modelis nepakrautas -> substring fallback. Klaida: {err_preview}"
        return "Engine: THULAC aktyvus."

    if mode == "hybrid":
        if not JIEBA_AVAILABLE:
            return "Engine: hybrid (2+ simbolių = substring, 1 simbolio = substring fallback; jieba nepasiekiama)."
        return "Engine: hybrid (2+ simbolių = substring, 1 simbolio = jieba precise)."

    return "Engine: substring (tiesioginis atitikmuo)."


def has_single_char_terms(terms_df: pd.DataFrame) -> bool:
    """True if dictionary has at least one non-empty single-character term."""
    if terms_df is None or terms_df.empty or "term" not in terms_df.columns:
        return False
    for t in terms_df["term"].tolist():
        if len(safe_str(t)) == 1:
            return True
    return False


def get_effective_aggregate_modes(all_modes: List[str], terms_df: pd.DataFrame, preferred_mode: Optional[str] = None) -> List[str]:
    """
    Aggregate voting modes with de-duplication rule:
    if no 1-char terms, hybrid and substring are effectively the same -> keep only one.
    Prefer keeping currently selected mode (preferred_mode) when applicable.
    """
    modes = list(all_modes)
    if has_single_char_terms(terms_df):
        return modes

    if "hybrid" in modes and "substring" in modes:
        keep = preferred_mode if preferred_mode in ("hybrid", "substring") else "substring"
        drop = "hybrid" if keep == "substring" else "substring"
        modes = [m for m in modes if m != drop]
    return modes


def parse_year_month_from_filename(filename: str) -> Tuple[str, str]:
    """
    Try to extract (year, month) from filename.
    Supported examples:
      - 2017-01-03_foo.txt  -> ("2017", "01")
      - 2017_01_foo.txt     -> ("2017", "01")
      - 201701_foo.txt      -> ("2017", "01")
    Fallback: year via 4-digit year regex, month "" if unknown.
    """
    stem = re.sub(r"\.[^.]+$", "", safe_str(filename)).strip()

    m = re.search(r"(?<!\d)((?:19|20)\d{2})[-_\.](0[1-9]|1[0-2])(?:[-_\.](0[1-9]|[12]\d|3[01]))?(?!\d)", stem)
    if m:
        return m.group(1), m.group(2)

    m2 = re.search(r"(?<!\d)((?:19|20)\d{2})(0[1-9]|1[0-2])(?!\d)", stem)
    if m2:
        return m2.group(1), m2.group(2)

    m3 = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", stem)
    if m3:
        return m3.group(1), ""

    return "", ""


def parse_year_month_day_from_filename(filename: str) -> Tuple[str, str, str]:
    """
    Try to extract (year, month, day) from filename.
    Supported examples:
      - 2017-01-03_foo.txt  -> ("2017", "01", "03")
      - 2017_01_03_foo.txt  -> ("2017", "01", "03")
      - 20170103_foo.txt    -> ("2017", "01", "03")
      - 2017-01_foo.txt     -> ("2017", "01", "")
      - 201701_foo.txt      -> ("2017", "01", "")
    """
    stem = re.sub(r"\.[^.]+$", "", safe_str(filename)).strip()

    m = re.search(
        r"(?<!\d)((?:19|20)\d{2})[-_\.](0[1-9]|1[0-2])[-_\.](0[1-9]|[12]\d|3[01])(?!\d)",
        stem,
    )
    if m:
        return m.group(1), m.group(2), m.group(3)

    m2 = re.search(r"(?<!\d)((?:19|20)\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?!\d)", stem)
    if m2:
        return m2.group(1), m2.group(2), m2.group(3)

    y, mo = parse_year_month_from_filename(filename)
    return y, mo, ""


def month_tab_suffix(n_1based: int) -> str:
    # A, B, C ... (up to 26)
    idx = n_1based - 1
    if 0 <= idx < 26:
        return chr(ord("A") + idx)
    return f"({n_1based})"


# -----------------------------
# UI
# -----------------------------
st.markdown(
    f'<div class="sinocode-header">'
    f'<p class="sinocode-title">SINOCODE</p>'
    f'<p class="sinocode-sub">Discourse Analyzer (CN) — {html_module.escape(APP_VERSION)}</p>'
    f"</div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    # Export/metadata pavadinimas (run_label) paliekamas konstantinis, nes naujam palyginimui jo nenaudojame.
    # ZIP eksportas bendras nereikalingas, bet per-set eksportui run_label vis tiek perduodamas.
    run_label = "compare_run"
    st.header("Atpažinimo režimas")
    mode_labels = {
        "hybrid": "Hibridinis (rekomenduojama)",
        "substring": "Tik substring (tiesioginis atitikmuo)",
        "jieba_precise": "Jieba – precise (tikslus skaidymas)",
        "jieba_search": "Jieba – search (daugiau galimų hitų)",
        "thulac": "THULAC (kinų segmentacija)",
        "hanlp": "HanLP (kinų segmentacija)",
    }
    match_mode = st.radio(
        "Kaip skaičiuoti terminų pasikartojimus?",
        options=list(mode_labels.keys()),
        index=0,
        help=(
            "hybrid = frazės (2+ ženklai) skaičiuojamos substring metodu, "
            "o 1-ženkliai terminai — per jieba tokenizaciją. "
            "Jei jieba neįdiegta / neveikia, automatiškai bus fallback į substring. "
            "THULAC ir HanLP – alternatyvūs segmentatoriai (jei nepasiekiami, fallback į substring). "
            "HanLP – kinų segmentacija (pirmą paleidimą gali atsisiųsti modelius)."
        ),
        format_func=lambda k: mode_labels.get(k, k),
    )
    st.caption("Pastaba: „Hybrid“ nuo „substring“ skirsis tik jei žodyne yra bent vienas 1 hieroglifo termas.")

    # -----------------------------
    # Compare Set 1 vs Set 2
    # -----------------------------
    compare_ready = bool(st.session_state.get("_data_set1")) and bool(st.session_state.get("_data_set2"))
    compare_clicked = st.button(
        "Compare Set 1 vs Set 2",
        disabled=not compare_ready,
        key="compare_set_btn",
        help="Palygina bendrus dažniausiai vartojamus Concept ir Term elementus tarp Set 1 ir Set 2.",
    )
    if compare_clicked:
        st.session_state["_compare_requested"] = True
    elif not compare_ready:
        st.session_state["_compare_requested"] = False

    # Engine status (always visible)
    st.caption("jieba aktyvi ✅" if JIEBA_AVAILABLE else "jieba neaktyvi ❌")
    if HANLP_AVAILABLE and not _hanlp_load_error:
        st.caption("HanLP aktyvi ✅")
    else:
        st.caption("HanLP neaktyvi ❌")
    if THULAC_AVAILABLE and not _thulac_load_error:
        st.caption("THULAC aktyvi ✅")
    else:
        st.caption("THULAC neaktyvi ❌")

    # Mode-specific warnings/fallback details
    if (match_mode.startswith("jieba") or match_mode == "hybrid") and not JIEBA_AVAILABLE:
        st.warning("jieba biblioteka nepasiekiama šiame env. Naudosiu substring fallback.")
    if match_mode == "hanlp":
        if not HANLP_AVAILABLE:
            msg = (
                "HanLP nepasiekiama. Naudosiu substring fallback. "
                "Patikrink: `pip install hanlp`. Jei reikia, įdiegk: `pip install tensorflow` arba `pip install torch` (pagal HanLP dokumentaciją)."
            )
            if HANLP_IMPORT_ERROR:
                err_preview = (HANLP_IMPORT_ERROR[:150] + "…") if len(HANLP_IMPORT_ERROR) > 150 else HANLP_IMPORT_ERROR
                msg += f" Klaida: {err_preview}"
            st.warning(msg)
        elif _hanlp_load_error:
            err_preview = (_hanlp_load_error[:150] + "…") if len(_hanlp_load_error) > 150 else _hanlp_load_error
            st.warning(f"HanLP modelis nepakrovė. Naudosiu substring fallback. Klaida: {err_preview}")
    if match_mode == "thulac":
        if not THULAC_AVAILABLE:
            msg = "THULAC nepasiekiama. Naudosiu substring fallback. Patikrink: `pip install thulac`."
            if THULAC_IMPORT_ERROR:
                err_preview = (THULAC_IMPORT_ERROR[:150] + "…") if len(THULAC_IMPORT_ERROR) > 150 else THULAC_IMPORT_ERROR
                msg += f" Klaida: {err_preview}"
            st.warning(msg)
        elif _thulac_load_error:
            err_preview = (_thulac_load_error[:150] + "…") if len(_thulac_load_error) > 150 else _thulac_load_error
            st.warning(f"THULAC modelis nepakrovė. Naudosiu substring fallback. Klaida: {err_preview}")


# Two-column layout: Set 1 | Set 2
col_set1, col_set2 = st.columns(2)


def _render_set_ui(data: Dict, set_id: int, match_mode: str, run_label: str):
    """Render Timeline tab + per-year tabs + export for one set. Uses meta_key(., set_id)."""
    def _mk(fn):
        return meta_key(fn, set_id)

    terms_df = data["terms_df"]
    docs = data["docs"]
    doc_term_hits = data["doc_term_hits"]
    doc_cat_hits = data["doc_cat_hits"]
    doc_conc_hits = data["doc_conc_hits"]
    docs_overview = data["docs_overview"]
    docs_overview_export = data["docs_overview_export"]
    term_hits_all_docs = data["term_hits_all_docs"]
    doc_infos = data["doc_infos"]
    years = data["years"]
    read_errors = data.get("read_errors") or {}
    aggregate_view_enabled = bool(st.session_state.get(f"_aggregate_view_set{set_id}", False))

    if read_errors:
        with st.expander("Dokumentų skaitymo problemos", expanded=True):
            for fn, msg in read_errors.items():
                st.warning(f"**{fn}**: {msg}")

    tab_names = ["📈 Timeline"] + years
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
                st.caption("Patarimas: palyginimui tarp dokumentų – „Per 10k characters“ arba „Share (%)“.")
    
                view_level = st.radio(
                    "Ką sekti laike?",
                    options=["Category", "Concept", "Term"],
                    horizontal=True,
                    index=1,  # default: Concept
                    key=f"view_level_set{set_id}",
                )
                metric = st.selectbox(
                    "Metrika",
                    options=["Raw count", "Per 10k characters", "Share (%)"],
                    index=2,  # default: Share (%)
                    key=f"metric_set{set_id}",
                )
    
                # Build long format by year
                rows_long = []
                for filename, _text in docs:
                    year = st.session_state[_mk(filename)]["year"]
                    y_int = safe_year(year)
                    if y_int is None:
                        continue
                    char_count = int(docs_overview.loc[docs_overview["filename"] == filename, "chars"].iloc[0])
    
                    th = doc_term_hits[filename]
                    if th.empty:
                        continue
    
                    doc_total_hits = int(th["Count"].sum())
                    share_div = doc_total_hits if doc_total_hits > 0 else 1
    
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
                                "share": (cnt / share_div) * 100.0,
                            }
                        )
    
                long_df = pd.DataFrame(rows_long)
                if long_df.empty:
                    st.info("Nėra pakankamai hitų, kad sudaryčiau laiko grafiką.")
                else:
                    label_col = "label"
                    if view_level == "Term":
                        term_to_translation = dict(
                            zip(terms_df["term"].astype(str), terms_df["translation"].astype(str))
                        )
                        long_df["label_display"] = long_df["label"].map(
                            lambda t: safe_str(term_to_translation.get(safe_str(t), "")).strip()
                            if safe_str(term_to_translation.get(safe_str(t), "")).strip()
                            else safe_str(t)
                        )
                        label_col = "label_display"

                    # Choose labels to plot
                    # Default: all when Concept, top 12 when Term, else top 8
                    totals = (
                        long_df.groupby(label_col)["count"].sum().sort_values(ascending=False).reset_index()
                    )
                    default_labels = (
                        totals[label_col].tolist()
                        if view_level == "Concept"
                        else totals[label_col].head(12).tolist() if view_level == "Term" else totals[label_col].head(8).tolist()
                    )

                    selected_key = f"selected_labels_set{set_id}_{view_level}"
                    if view_level == "Term":
                        prev_vals = st.session_state.get(selected_key)
                        if not prev_vals:
                            st.session_state[selected_key] = default_labels

                    ms_kwargs = {
                        "label": f"Pasirink {view_level} (maks. rekomenduojama 8–12 grafike)",
                        "options": totals[label_col].tolist(),
                        "key": selected_key,
                    }
                    if selected_key not in st.session_state:
                        ms_kwargs["default"] = default_labels
                    selected = st.multiselect(**ms_kwargs)

                    if not selected:
                        st.info(f"Pasirink bent vieną {view_level}, kad būtų galima sudaryti grafiką.")
                    else:
                        if metric == "Raw count":
                            value_col = "count"
                            agg = "sum"
                        elif metric == "Share (%)":
                            value_col = "share"
                            agg = "mean"  # average share across docs when multiple per year
                        else:
                            value_col = "per_10k"
                            agg = "sum"

                        # If aggregate mode is enabled, build per-mode pivots and then aggregate them.
                        if aggregate_view_enabled:
                            modes_data = st.session_state.get(f"_data_set{set_id}_modes") or {}
                            agg_modes = get_effective_aggregate_modes(
                                list(mode_labels.keys()),
                                terms_df,
                                preferred_mode=match_mode,
                            )
                            if not modes_data:
                                st.warning(
                                    "Aggregate mode įjungtas, bet trūksta per-mode rezultatų Timeline grafikui. "
                                    "Pateiksiu tik pasirinkto režimo grafiką."
                                )
                                plot_df = long_df[long_df[label_col].isin(selected)].copy()
                                pivot = (
                                    plot_df.pivot_table(
                                        index="year",
                                        columns=label_col,
                                        values=value_col,
                                        aggfunc=agg,
                                    )
                                    .fillna(0)
                                    .sort_index()
                                )
                                pivot = pivot.reindex(columns=selected).fillna(0)
                                st.line_chart(pivot, height=320)
                                display_df = pivot.reset_index()
                                if metric == "Share (%)":
                                    for col in display_df.columns:
                                        if col != "year":
                                            display_df[col] = display_df[col].map(lambda x: f"{x:.1f}%")
                                st.dataframe(display_df, width="stretch")
                            else:
                                year_index = sorted(usable["year_int"].dropna().unique())

                                per_mode_pivots: Dict[str, pd.DataFrame] = {}

                                for m in agg_modes:
                                    data_m = modes_data.get(m)
                                    if not data_m:
                                        continue

                                    docs_m = data_m["docs"]
                                    doc_term_hits_m = data_m["doc_term_hits"]

                                    rows_long_m = []
                                    for filename, _text in docs_m:
                                        year = st.session_state[_mk(filename)]["year"]
                                        y_int = safe_year(year)
                                        if y_int is None:
                                            continue

                                        char_count = int(
                                            docs_overview.loc[
                                                docs_overview["filename"] == filename, "chars"
                                            ].iloc[0]
                                        )

                                        th = doc_term_hits_m[filename]
                                        if th.empty:
                                            continue

                                        doc_total_hits = int(th["Count"].sum())
                                        share_div = doc_total_hits if doc_total_hits > 0 else 1

                                        if view_level == "Category":
                                            g = (
                                                th.groupby("Category")["Count"]
                                                .sum()
                                                .reset_index()
                                                .rename(columns={"Category": "label"})
                                            )
                                        elif view_level == "Concept":
                                            g = (
                                                th.groupby("Concept")["Count"]
                                                .sum()
                                                .reset_index()
                                                .rename(columns={"Concept": "label"})
                                            )
                                        else:
                                            g = (
                                                th.groupby("CH term")["Count"]
                                                .sum()
                                                .reset_index()
                                                .rename(columns={"CH term": "label"})
                                            )

                                        for _, r in g.iterrows():
                                            cnt = int(r["Count"])
                                            rows_long_m.append(
                                                {
                                                    "year": y_int,
                                                    "label": safe_str(r["label"]),
                                                    "count": cnt,
                                                    "per_10k": normalize_per_10k_chars(cnt, char_count),
                                                    "share": (cnt / share_div) * 100.0,
                                                }
                                            )

                                    long_df_m = pd.DataFrame(rows_long_m)
                                    if long_df_m.empty:
                                        continue

                                    if view_level == "Term":
                                        long_df_m["label_display"] = long_df_m["label"].map(
                                            lambda t: safe_str(term_to_translation.get(safe_str(t), "")).strip()
                                            if safe_str(term_to_translation.get(safe_str(t), "")).strip()
                                            else safe_str(t)
                                        )

                                    plot_df_m = long_df_m[long_df_m[label_col].isin(selected)].copy()
                                    pivot_m = (
                                        plot_df_m.pivot_table(
                                            index="year",
                                            columns=label_col,
                                            values=value_col,
                                            aggfunc=agg,
                                        )
                                        .fillna(0)
                                        .sort_index()
                                    )
                                    pivot_m = pivot_m.reindex(index=year_index, columns=selected).fillna(0)
                                    per_mode_pivots[m] = pivot_m

                                if not per_mode_pivots:
                                    st.warning("Nepavyko paruošti per-mode pivots agreguotam grafiko vaizdui.")
                                else:
                                    # Aggregate: Share -> mean per modes (kaip tu prašei), kitiems metricams taip pat darom mean.
                                    pivot_agg = pd.DataFrame(index=year_index, columns=selected, dtype=float)
                                    for col in selected:
                                        stack = pd.concat([pm[col] for pm in per_mode_pivots.values()], axis=1)
                                        pivot_agg[col] = stack.mean(axis=1)

                                    st.line_chart(pivot_agg, height=320)
                                    display_df = pivot_agg.reset_index()
                                    if metric == "Share (%)":
                                        for col in display_df.columns:
                                            if col != "year":
                                                display_df[col] = display_df[col].map(lambda x: f"{x:.1f}%")
                                    st.dataframe(display_df, width="stretch")

                                    # Per-mode expanders (būtina)
                                    for m in agg_modes:
                                        if m not in per_mode_pivots:
                                            continue
                                        mode_label = mode_labels.get(m, m) if "mode_labels" in globals() else m
                                        with st.expander(f"{mode_label} ({m})", expanded=False):
                                            st.caption(describe_mode_runtime(m))
                                            st.line_chart(per_mode_pivots[m], height=280)
                                            df_m_show = per_mode_pivots[m].reset_index()
                                            if metric == "Share (%)":
                                                for col in df_m_show.columns:
                                                    if col != "year":
                                                        df_m_show[col] = df_m_show[col].map(lambda x: f"{x:.1f}%")
                                            st.dataframe(df_m_show, width="stretch")

                        # Normal (single-mode) timeline
                        else:
                            plot_df = long_df[long_df[label_col].isin(selected)].copy()
                            pivot = (
                                plot_df.pivot_table(index="year", columns=label_col, values=value_col, aggfunc=agg)
                                .fillna(0)
                                .sort_index()
                            )
                            pivot = pivot.reindex(columns=selected).fillna(0)

                            st.line_chart(pivot, height=320)
                            display_df = pivot.reset_index()
                            if metric == "Share (%)":
                                for col in display_df.columns:
                                    if col != "year":
                                        display_df[col] = display_df[col].map(lambda x: f"{x:.1f}%")
                            st.dataframe(display_df, width="stretch")

            # Top 12 raktažodžių (Term): bendras count ir bendras share visame korpuse
            if view_level == "Term":
                term_totals: Dict[str, int] = defaultdict(int)
                corpus_total_hits = 0
                for filename, _text in docs:
                    th = doc_term_hits.get(filename)
                    if th is None or th.empty:
                        continue
                    corpus_total_hits += int(th["Count"].sum())
                    for _, r in th.iterrows():
                        ch = safe_str(r.get("CH term", ""))
                        if not ch:
                            continue
                        term_totals[ch] += int(r.get("Count", 0))
                if term_totals and corpus_total_hits > 0:
                    term_to_en_top = dict(zip(terms_df["term"].astype(str), terms_df["translation"].astype(str)))
                    rows_top: List[Dict[str, object]] = []
                    for ch, c in term_totals.items():
                        eng = safe_str(term_to_en_top.get(ch, "")).strip()
                        rows_top.append(
                            {
                                "CH term": ch,
                                "Translation (EN)": eng if eng else "",
                                "Total count": int(c),
                                "Corpus share (%)": round((c / corpus_total_hits) * 100.0, 3),
                            }
                        )
                    df_top_all = pd.DataFrame(rows_top)
                    df_top_count = (
                        df_top_all.sort_values("Total count", ascending=False).head(12).reset_index(drop=True)
                    )
                    df_top_count.insert(0, "Rank", range(1, len(df_top_count) + 1))

                    # Mean yearly share: for each calendar year, term_share_y = term_hits_y / total_hits_y * 100,
                    # then average across years (differs from corpus-wide share ranking).
                    term_year_totals: Dict[Tuple[str, int], int] = defaultdict(int)
                    year_total_hits: Dict[int, int] = defaultdict(int)
                    for filename, _text in docs:
                        y_int = safe_year(st.session_state[_mk(filename)]["year"])
                        if y_int is None:
                            continue
                        th = doc_term_hits.get(filename)
                        if th is None or th.empty:
                            continue
                        for _, r in th.iterrows():
                            ch = safe_str(r.get("CH term", ""))
                            if not ch:
                                continue
                            c = int(r.get("Count", 0))
                            term_year_totals[(ch, y_int)] += c
                            year_total_hits[y_int] += c

                    mean_yearly_rows: List[Dict[str, object]] = []
                    years_with_hits = sorted(y for y, tot in year_total_hits.items() if tot > 0)
                    for ch in term_totals:
                        if not years_with_hits:
                            mean_y = 0.0
                        else:
                            shares_y: List[float] = []
                            for y in years_with_hits:
                                tot = year_total_hits[y]
                                if tot <= 0:
                                    continue
                                tc = term_year_totals.get((ch, y), 0)
                                shares_y.append((tc / tot) * 100.0)
                            mean_y = sum(shares_y) / len(shares_y) if shares_y else 0.0
                        eng = safe_str(term_to_en_top.get(ch, "")).strip()
                        mean_yearly_rows.append(
                            {
                                "CH term": ch,
                                "Translation (EN)": eng if eng else "",
                                "Total count": int(term_totals[ch]),
                                "Mean yearly Share (%)": round(mean_y, 3),
                                "Corpus share (%)": round((term_totals[ch] / corpus_total_hits) * 100.0, 3),
                            }
                        )
                    df_mean_share = pd.DataFrame(mean_yearly_rows)
                    df_top_share = (
                        df_mean_share.sort_values("Mean yearly Share (%)", ascending=False).head(12).reset_index(drop=True)
                    )
                    df_top_share.insert(0, "Rank", range(1, len(df_top_share) + 1))
                    st.markdown("#### Top 12 raktažodžių (bendras *count* visame korpuse)")
                    st.caption("Rikiuota pagal bendrą aptiktų hitų skaičių visuose dokumentuose.")
                    st.dataframe(df_top_count, width="stretch", hide_index=True)
                    st.markdown("#### Top 12 raktažodžių (vidutinis metinis *Share (%)* )")
                    st.caption(
                        "Kiekvienais metais: termino hitai ÷ visų žodyno hitų toje metų imtyje × 100; "
                        "tada imamas aritmetinis vidurkis per metus su bent vienu hitu. "
                        f"Metai imtyje: {', '.join(str(y) for y in years_with_hits)}."
                    )
                    st.dataframe(df_top_share, width="stretch", hide_index=True)

                    corpus_chars = 0
                    for filename, _text in docs:
                        try:
                            corpus_chars += int(
                                docs_overview.loc[docs_overview["filename"] == filename, "chars"].iloc[0]
                            )
                        except Exception:
                            continue
                    if corpus_chars > 0:
                        per10k_rows: List[Dict[str, object]] = []
                        for ch, c in term_totals.items():
                            eng = safe_str(term_to_en_top.get(ch, "")).strip()
                            per10k_rows.append(
                                {
                                    "CH term": ch,
                                    "Translation (EN)": eng if eng else "",
                                    "Total count": int(c),
                                    "Per 10k chars": round(
                                        float(normalize_per_10k_chars(int(c), corpus_chars)), 4
                                    ),
                                    "Corpus share (%)": round((int(c) / corpus_total_hits) * 100.0, 3),
                                }
                            )
                        df_per10k = pd.DataFrame(per10k_rows)
                        df_top_per10k = (
                            df_per10k.sort_values("Per 10k chars", ascending=False).head(12).reset_index(drop=True)
                        )
                        df_top_per10k.insert(0, "Rank", range(1, len(df_top_per10k) + 1))
                        st.markdown("#### Top 12 raktažodžių (*Per 10k characters* visame korpuse)")
                        st.caption(
                            "Normalizacija pagal visų įkeltų dokumentų simbolių sumą "
                            f"({corpus_chars:,} simb.). Rodiklis = (termino hitai ÷ simboliai) × 10 000."
                        )
                        st.dataframe(df_top_per10k, width="stretch", hide_index=True)

                    # Civilizacinio žodyno metaforos: tik aptiktos „Metaphor“ eilutės
                    if is_civilization_lexicon(terms_df) and not term_hits_all_docs.empty:
                        mh = term_hits_all_docs[
                            term_hits_all_docs["Concept"].astype(str).str.strip() == "Metaphor"
                        ].copy()
                        if not mh.empty:
                            agg = (
                                mh.groupby(
                                    ["CH term", "Pinyin", "ENG translation", "Concept", "Category"],
                                    as_index=False,
                                )["Count"]
                                .sum()
                                .sort_values("Count", ascending=False)
                                .reset_index(drop=True)
                            )
                            metaphor_total = int(agg["Count"].sum())
                            corpus_chars_m = 0
                            for filename, _text in docs:
                                try:
                                    corpus_chars_m += int(
                                        docs_overview.loc[docs_overview["filename"] == filename, "chars"].iloc[0]
                                    )
                                except Exception:
                                    continue
                            if corpus_chars_m > 0 and metaphor_total > 0:
                                agg["Per 10k chars"] = agg["Count"].map(
                                    lambda c: round(float(normalize_per_10k_chars(int(c), corpus_chars_m)), 4)
                                )
                                agg["Share of metaphor hits (%)"] = agg["Count"].map(
                                    lambda c: round((int(c) / metaphor_total) * 100.0, 3)
                                )
                            elif metaphor_total > 0:
                                agg["Per 10k chars"] = 0.0
                                agg["Share of metaphor hits (%)"] = agg["Count"].map(
                                    lambda c: round((int(c) / metaphor_total) * 100.0, 3)
                                )
                            agg.insert(0, "Rank", range(1, len(agg) + 1))
                            st.markdown("#### Aptiktos metaforos (civilizacinis žodynas, *Concept = Metaphor*)")
                            st.caption(
                                "Rodomi tik žodyno terminai, pažymėti kaip „Metaphor“, su bent vienu hitu korpuse."
                            )
                            st.dataframe(agg, width="stretch", hide_index=True)
                            st.download_button(
                                "Atsisiųsti metaforų suvestinę (CSV)",
                                agg.to_csv(index=False, encoding="utf-8-sig"),
                                file_name="metaphor_hits_civilization_lexicon.csv",
                                mime="text/csv",
                                key=f"dl_metaphor_hits_{set_id}",
                            )

            # Terminų augimo / mažėjimo lentelė (Trend Table)
            year_ints = sorted(usable["year_int"].dropna().unique())
            events_df = load_events()
            if year_ints:
                    term_to_translation = dict(zip(terms_df["term"].astype(str), terms_df["translation"].astype(str)))
                    first_label: str
                    last_label: str
                    term_count_first: Dict[str, int] = {}
                    term_count_last: Dict[str, int] = {}
    
                    if len(year_ints) >= 2:
                        # Kelios metų: agreguojame pagal (term, year)
                        term_year_count: Dict[Tuple[str, int], int] = defaultdict(int)
                        for filename, _text in docs:
                            y_int = safe_year(st.session_state[_mk(filename)]["year"])
                            if y_int is None:
                                continue
                            th = doc_term_hits[filename]
                            for _, r in th.iterrows():
                                term = safe_str(r["CH term"])
                                if not term:
                                    continue
                                term_year_count[(term, y_int)] += int(r["Count"])
                        first_year = int(year_ints[0])
                        last_year = int(year_ints[-1])
                        first_label = str(first_year)
                        last_label = str(last_year)
                        seen_terms = {term for (term, _y) in term_year_count}
                        for term in seen_terms:
                            term_count_first[term] = term_year_count.get((term, first_year), 0)
                            term_count_last[term] = term_year_count.get((term, last_year), 0)
                    else:
                        # Vieni metai: periodai nuo anksčiausio iki vėliausio (YYYY-MM)
                        term_period_count: Dict[Tuple[str, str], int] = defaultdict(int)
                        unique_periods: set = set()
                        for filename, _text in docs:
                            y_int = safe_year(st.session_state[_mk(filename)]["year"])
                            if y_int is None:
                                continue
                            y_str, mo, _ = parse_year_month_day_from_filename(filename)
                            period = f"{y_str}-{mo}" if (mo and mo != "??") else str(y_int)
                            unique_periods.add(period)
                            th = doc_term_hits[filename]
                            for _, r in th.iterrows():
                                term = safe_str(r["CH term"])
                                if not term:
                                    continue
                                term_period_count[(term, period)] += int(r["Count"])
                        periods_sorted = sorted(unique_periods)
                        first_period = periods_sorted[0] if periods_sorted else str(year_ints[0])
                        last_period = periods_sorted[-1] if periods_sorted else str(year_ints[0])
                        first_label = first_period
                        last_label = last_period
                        seen_terms = {term for (term, _p) in term_period_count}
                        for term in seen_terms:
                            term_count_first[term] = term_period_count.get((term, first_period), 0)
                            term_count_last[term] = term_period_count.get((term, last_period), 0)
    
                    trend_rows = []
                    for term in sorted(seen_terms):
                        first_count = term_count_first.get(term, 0)
                        last_count = term_count_last.get(term, 0)
                        if first_count == 0 and last_count == 0:
                            continue  # nerodom terminų, kurių nebuvo aptikta
                        change = last_count - first_count
                        if first_count > 0:
                            pct = change / first_count
                        else:
                            pct = 1.0 if last_count > 0 else 0.0
                        if pct > 0.2:
                            trend = "↑ increasing"
                        elif pct < -0.2:
                            trend = "↓ decreasing"
                        else:
                            trend = "→ stable"
                        change_str = f"+{change}" if change > 0 else str(change)
                        total_count = first_count + last_count
                        trend_rows.append({
                            "Term": term,
                            "Translation": term_to_translation.get(term, ""),
                            first_label: first_count,
                            last_label: last_count,
                            "Change": change_str,
                            "Trend": trend,
                            "_pct": pct,
                            "_total": total_count,
                        })
                    if trend_rows:
                        # Rūšiavimas: increasing (didžiausias teigiamas Change viršuje), declining (didžiausias neigiamas viršuje), stable (didžiausias total viršuje)
                        def _sort_key(r):
                            change = r[last_label] - r[first_label]
                            if r["Trend"] == "↑ increasing":
                                return (0, -change)
                            if r["Trend"] == "↓ decreasing":
                                return (1, change)  # didžiausias neigiamas viršuje
                            return (2, -r["_total"])
                        trend_rows.sort(key=_sort_key)
                        display_rows = [
                            {k: v for k, v in r.items() if not k.startswith("_")}
                            for r in trend_rows
                        ]
                        trend_df = pd.DataFrame(display_rows).reset_index(drop=True)
    
                        # Gradient spalvos: increasing = žalia, decreasing = raudona, stable = oranžinė (pagal total)
                        def _lerp(hex1: str, hex2: str, t: float) -> str:
                            t = max(0.0, min(1.0, t))
                            r1, g1, b1 = int(hex1[1:3], 16), int(hex1[3:5], 16), int(hex1[5:7], 16)
                            r2, g2, b2 = int(hex2[1:3], 16), int(hex2[3:5], 16), int(hex2[5:7], 16)
                            r = int(r1 + (r2 - r1) * t)
                            g = int(g1 + (g2 - g1) * t)
                            b = int(b1 + (b2 - b1) * t)
                            return f"#{r:02x}{g:02x}{b:02x}"
                        green_dark, green_light = "#0d5c0d", "#e8f5e9"
                        red_dark, red_light = "#b71c1c", "#ffcdd2"
                        orange_dark, orange_light = "#e65100", "#ffe0b2"
                        row_colors: List[str] = []
                        inc_rows = [r for r in trend_rows if r["Trend"] == "↑ increasing"]
                        dec_rows = [r for r in trend_rows if r["Trend"] == "↓ decreasing"]
                        stable_rows = [r for r in trend_rows if r["Trend"] == "→ stable"]
                        n_inc = len(inc_rows)
                        n_dec = len(dec_rows)
                        n_stable = len(stable_rows)
                        for i, r in enumerate(trend_rows):
                            if r["Trend"] == "↑ increasing":
                                idx = inc_rows.index(r)
                                t = idx / (n_inc - 1) if n_inc > 1 else 0
                                row_colors.append(_lerp(green_dark, green_light, t))
                            elif r["Trend"] == "↓ decreasing":
                                idx = dec_rows.index(r)
                                t = idx / (n_dec - 1) if n_dec > 1 else 0
                                row_colors.append(_lerp(red_dark, red_light, t))
                            else:
                                idx = stable_rows.index(r)
                                t = idx / (n_stable - 1) if n_stable > 1 else 0
                                row_colors.append(_lerp(orange_dark, orange_light, t))
                        row_color_map = dict(zip(range(len(trend_df)), row_colors))
    
                        def _style_row(row, color_map):
                            idx = row.name
                            bg = color_map.get(idx, "#ffffff")
                            return [f"background-color: {bg}"] * len(row)
    
                        styled = trend_df.style.apply(lambda row: _style_row(row, row_color_map), axis=1)
                        st.markdown("### Terminų augimo / mažėjimo lentelė")
                        st.dataframe(styled, width="stretch")

                        # Papildomai: per-year porų lentelės (2017-2018, 2018-2019, ...)
                        if len(year_ints) >= 2:
                            st.markdown("#### Terminų augimo / mažėjimo lentelė per metų poras")
                            year_pairs = [(int(year_ints[i]), int(year_ints[i + 1])) for i in range(len(year_ints) - 1)]
                            for y1, y2 in year_pairs:
                                with st.expander(f"{y1}-{y2}", expanded=False):
                                    pair_rows = []
                                    for term in sorted(seen_terms):
                                        c1 = int(term_year_count.get((term, y1), 0))
                                        c2 = int(term_year_count.get((term, y2), 0))
                                        if c1 == 0 and c2 == 0:
                                            continue
                                        ch = c2 - c1
                                        if c1 > 0:
                                            pct_pair = ch / c1
                                        else:
                                            pct_pair = 1.0 if c2 > 0 else 0.0
                                        if pct_pair > 0.2:
                                            tr = "↑ increasing"
                                        elif pct_pair < -0.2:
                                            tr = "↓ decreasing"
                                        else:
                                            tr = "→ stable"
                                        pair_rows.append(
                                            {
                                                "Term": term,
                                                "Translation": term_to_translation.get(term, ""),
                                                str(y1): c1,
                                                str(y2): c2,
                                                "Change": (f"+{ch}" if ch > 0 else str(ch)),
                                                "Trend": tr,
                                                "_pct": pct_pair,
                                                "_total": c1 + c2,
                                            }
                                        )
                                    if not pair_rows:
                                        st.info("Nėra pakankamai termų hitų šiai metų porai.")
                                    else:
                                        def _pair_sort_key(r):
                                            d = r[str(y2)] - r[str(y1)]
                                            if r["Trend"] == "↑ increasing":
                                                return (0, -d)
                                            if r["Trend"] == "↓ decreasing":
                                                return (1, d)
                                            return (2, -r["_total"])

                                        pair_rows.sort(key=_pair_sort_key)
                                        pair_df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in pair_rows])
                                        st.dataframe(pair_df, width="stretch")
    
                        # Didžiausi pokyčiai (Top Changes Table) – surūšiuota pagal Change % (didžiausi teigiami viršuje)
                        top_sorted = sorted(trend_rows, key=lambda r: -r["_pct"])
                        top_rows = []
                        for rank, r in enumerate(top_sorted, start=1):
                            pct = r["_pct"]
                            if pct >= 0:
                                change_pct_str = f"+{pct * 100:.0f}%"
                            else:
                                change_pct_str = f"−{abs(pct) * 100:.0f}%"
                            top_rows.append({
                                "Rank": rank,
                                "Term": r["Term"],
                                "Translation": r["Translation"],
                                "Change %": change_pct_str,
                            })
                        if top_rows:
                            top_df = pd.DataFrame(top_rows)
                            st.markdown("### Didžiausi pokyčiai (Top Changes Table)")
                            st.dataframe(top_df, width="stretch")

                            # Papildomai: Top changes per metų poras
                            if len(year_ints) >= 2:
                                st.markdown("#### Top Changes Table per metų poras")
                                year_pairs = [(int(year_ints[i]), int(year_ints[i + 1])) for i in range(len(year_ints) - 1)]
                                for y1, y2 in year_pairs:
                                    with st.expander(f"{y1}-{y2}", expanded=False):
                                        pair_rows = []
                                        for term in sorted(seen_terms):
                                            c1 = int(term_year_count.get((term, y1), 0))
                                            c2 = int(term_year_count.get((term, y2), 0))
                                            if c1 == 0 and c2 == 0:
                                                continue
                                            ch = c2 - c1
                                            if c1 > 0:
                                                pct_pair = ch / c1
                                            else:
                                                pct_pair = 1.0 if c2 > 0 else 0.0
                                            pair_rows.append(
                                                {
                                                    "Term": term,
                                                    "Translation": term_to_translation.get(term, ""),
                                                    "_pct": pct_pair,
                                                }
                                            )
                                        if not pair_rows:
                                            st.info("Nėra pakankamai termų hitų šiai metų porai.")
                                        else:
                                            pair_sorted = sorted(pair_rows, key=lambda r: -r["_pct"])
                                            out_rows = []
                                            for rk, r in enumerate(pair_sorted, start=1):
                                                pct_pair = r["_pct"]
                                                if pct_pair >= 0:
                                                    pct_s = f"+{pct_pair * 100:.0f}%"
                                                else:
                                                    pct_s = f"−{abs(pct_pair) * 100:.0f}%"
                                                out_rows.append(
                                                    {
                                                        "Rank": rk,
                                                        "Term": r["Term"],
                                                        "Translation": r["Translation"],
                                                        "Change %": pct_s,
                                                    }
                                                )
                                            st.dataframe(pd.DataFrame(out_rows), width="stretch")
    
                        # Peak Year lentelė: metai/periodas, kai terminas turi didžiausią dalį (share) nuo bendro terminų skaičiaus tame periode
                        peak_rows = []
                        if len(year_ints) >= 2:
                            # bendras visų terminų skaičius per metus (normalizavimui)
                            year_total_hits: Dict[int, int] = defaultdict(int)
                            for (term_key, y_key), cnt in term_year_count.items():
                                year_total_hits[y_key] += cnt

                            for term in seen_terms:
                                # share per metus
                                shares = []
                                for y in year_ints:
                                    cnt = term_year_count.get((term, y), 0)
                                    total = year_total_hits.get(y, 0)
                                    share = cnt / total if total > 0 else 0.0
                                    shares.append((y, share))
                                max_share = max(s for _y, s in shares)
                                peak_years = sorted([str(y) for y, s in shares if s == max_share and s > 0])
                                if not peak_years:
                                    continue
                                peak_rows.append({
                                    "Term": term,
                                    "Translation": term_to_translation.get(term, ""),
                                    "Peak year": ", ".join(peak_years),
                                    "Peak share %" : max_share * 100.0,
                                })
                        else:
                            # bendras visų terminų skaičius per periodą (normalizavimui)
                            period_total_hits: Dict[str, int] = defaultdict(int)
                            for (term_key, p_key), cnt in term_period_count.items():
                                period_total_hits[p_key] += cnt

                            for term in seen_terms:
                                shares = []
                                for p in periods_sorted:
                                    cnt = term_period_count.get((term, p), 0)
                                    total = period_total_hits.get(p, 0)
                                    share = cnt / total if total > 0 else 0.0
                                    shares.append((p, share))
                                max_share = max(s for _p, s in shares)
                                peak_periods = sorted([p for p, s in shares if s == max_share and s > 0])
                                if not peak_periods:
                                    continue
                                peak_rows.append({
                                    "Term": term,
                                    "Translation": term_to_translation.get(term, ""),
                                    "Peak year": ", ".join(peak_periods),
                                    "Peak share %": max_share * 100.0,
                                })
                        if peak_rows:
                            peak_df = pd.DataFrame(peak_rows).sort_values("Peak share %", ascending=False).reset_index(drop=True)
                            st.markdown("### Peak Year lentelė (pagal share)")
                            st.dataframe(peak_df, width="stretch")
    
                        # Koncepcijų agreguota dinamika: dalis (Share %) po konceptą pirmais ir paskutiniais metais/periodais, Change ↑/↓/→
                        if len(year_ints) >= 2:
                            concept_year_count: Dict[Tuple[str, int], int] = defaultdict(int)
                            year_total_hits: Dict[int, int] = defaultdict(int)
                            for filename, _text in docs:
                                y_int = safe_year(st.session_state[_mk(filename)]["year"])
                                if y_int is None:
                                    continue
                                th = doc_term_hits[filename]
                                for _, r in th.iterrows():
                                    concept = safe_str(r["Concept"])
                                    if not concept:
                                        continue
                                    cnt = int(r["Count"])
                                    concept_year_count[(concept, y_int)] += cnt
                                    year_total_hits[y_int] += cnt
                            first_key = int(year_ints[0])
                            last_key = int(year_ints[-1])
                            seen_concepts = {c for (c, _y) in concept_year_count}
                        else:
                            concept_period_count: Dict[Tuple[str, str], int] = defaultdict(int)
                            period_total_hits: Dict[str, int] = defaultdict(int)
                            for filename, _text in docs:
                                y_int = safe_year(st.session_state[_mk(filename)]["year"])
                                if y_int is None:
                                    continue
                                y_str, mo, _ = parse_year_month_day_from_filename(filename)
                                period = f"{y_str}-{mo}" if (mo and mo != "??") else str(y_int)
                                th = doc_term_hits[filename]
                                for _, r in th.iterrows():
                                    concept = safe_str(r["Concept"])
                                    if not concept:
                                        continue
                                    cnt = int(r["Count"])
                                    concept_period_count[(concept, period)] += cnt
                                    period_total_hits[period] += cnt
                            first_key = periods_sorted[0] if periods_sorted else str(year_ints[0])
                            last_key = periods_sorted[-1] if periods_sorted else str(year_ints[0])
                            seen_concepts = {c for (c, _p) in concept_period_count}
                        concept_rows = []
                        share_epsilon = 0.1  # minimalus skirtumas procentiniais punktais, kad laikytume pokytį reikšmingu
                        concept_first_label = f"{first_label} Share (%)"
                        concept_last_label = f"{last_label} Share (%)"
                        for concept in sorted(seen_concepts):
                            if len(year_ints) >= 2:
                                first_count = concept_year_count.get((concept, first_key), 0)
                                last_count = concept_year_count.get((concept, last_key), 0)
                                first_total = year_total_hits.get(first_key, 0)
                                last_total = year_total_hits.get(last_key, 0)
                            else:
                                first_count = concept_period_count.get((concept, first_key), 0)
                                last_count = concept_period_count.get((concept, last_key), 0)
                                first_total = period_total_hits.get(first_key, 0)
                                last_total = period_total_hits.get(last_key, 0)
    
                            first_share = (first_count / first_total * 100.0) if first_total > 0 else 0.0
                            last_share = (last_count / last_total * 100.0) if last_total > 0 else 0.0
    
                            if last_share > first_share + share_epsilon:
                                change_sym = "↑"
                            elif last_share < first_share - share_epsilon:
                                change_sym = "↓"
                            else:
                                change_sym = "→"
                            delta_share = last_share - first_share
                            concept_rows.append({
                                "Concept": concept,
                                concept_first_label: round(first_share, 2),
                                concept_last_label: round(last_share, 2),
                                "Change": change_sym,
                                "_first_share": first_share,
                                "_last_share": last_share,
                                "_delta_share": delta_share,
                            })
                        if concept_rows:
                            # Rūšiavimas: ↑ increasing, po to → stable, galiausiai ↓ decreasing; viduje pagal dydį
                            def _concept_sort_key(r):
                                if r["Change"] == "↑":
                                    return (0, -r["_delta_share"])
                                if r["Change"] == "→":
                                    return (1, 0)
                                return (2, r["_delta_share"])  # didžiausias mažėjimas viršuje
                            concept_rows.sort(key=_concept_sort_key)
                            concept_display = [
                                {k: v for k, v in r.items() if not k.startswith("_")}
                                for r in concept_rows
                            ]
                            concept_df = pd.DataFrame(concept_display).reset_index(drop=True)
    
                            # Gradient spalvos: increasing = žalia, stable = oranžinė, decreasing = raudona
                            def _lerp_hex(hex1: str, hex2: str, t: float) -> str:
                                t = max(0.0, min(1.0, t))
                                r1, g1, b1 = int(hex1[1:3], 16), int(hex1[3:5], 16), int(hex1[5:7], 16)
                                r2, g2, b2 = int(hex2[1:3], 16), int(hex2[3:5], 16), int(hex2[5:7], 16)
                                return f"#{int(r1 + (r2 - r1) * t):02x}{int(g1 + (g2 - g1) * t):02x}{int(b1 + (b2 - b1) * t):02x}"
                            green_dark, green_light = "#0d5c0d", "#e8f5e9"
                            orange = "#ffcc80"
                            red_dark, red_light = "#b71c1c", "#ffcdd2"
                            inc_list = [r for r in concept_rows if r["Change"] == "↑"]
                            stable_list = [r for r in concept_rows if r["Change"] == "→"]
                            dec_list = [r for r in concept_rows if r["Change"] == "↓"]
                            n_inc = len(inc_list)
                            n_dec = len(dec_list)
                            concept_row_colors: List[str] = []
                            for r in concept_rows:
                                if r["Change"] == "↑":
                                    idx = inc_list.index(r)
                                    t = idx / (n_inc - 1) if n_inc > 1 else 0
                                    concept_row_colors.append(_lerp_hex(green_dark, green_light, t))
                                elif r["Change"] == "→":
                                    concept_row_colors.append(orange)
                                else:
                                    idx = dec_list.index(r)
                                    t = idx / (n_dec - 1) if n_dec > 1 else 0
                                    concept_row_colors.append(_lerp_hex(red_dark, red_light, t))
                            concept_color_map = dict(zip(range(len(concept_df)), concept_row_colors))
    
                            def _concept_style_row(row, color_map):
                                bg = color_map.get(row.name, "#ffffff")
                                return [f"background-color: {bg}"] * len(row)
                            concept_styled = concept_df.style.apply(lambda row: _concept_style_row(row, concept_color_map), axis=1)
                            st.markdown("### Koncepcijų agreguota dinamika")
                            st.dataframe(concept_styled, width="stretch")

                            # Papildomai: Koncepcijų agreguota dinamika per metų poras
                            if len(year_ints) >= 2:
                                st.markdown("#### Koncepcijų agreguota dinamika per metų poras")
                                year_pairs = [(int(year_ints[i]), int(year_ints[i + 1])) for i in range(len(year_ints) - 1)]
                                for y1, y2 in year_pairs:
                                    with st.expander(f"{y1}-{y2}", expanded=False):
                                        pair_rows = []
                                        col1 = f"{y1} Share (%)"
                                        col2 = f"{y2} Share (%)"
                                        for concept in sorted(seen_concepts):
                                            c1 = int(concept_year_count.get((concept, y1), 0))
                                            c2 = int(concept_year_count.get((concept, y2), 0))
                                            t1 = int(year_total_hits.get(y1, 0))
                                            t2 = int(year_total_hits.get(y2, 0))
                                            s1 = (c1 / t1 * 100.0) if t1 > 0 else 0.0
                                            s2 = (c2 / t2 * 100.0) if t2 > 0 else 0.0
                                            if s1 == 0.0 and s2 == 0.0:
                                                continue
                                            if s2 > s1 + share_epsilon:
                                                ch_sym = "↑"
                                            elif s2 < s1 - share_epsilon:
                                                ch_sym = "↓"
                                            else:
                                                ch_sym = "→"
                                            pair_rows.append(
                                                {
                                                    "Concept": concept,
                                                    col1: round(s1, 2),
                                                    col2: round(s2, 2),
                                                    "Change": ch_sym,
                                                    "_delta": (s2 - s1),
                                                }
                                            )
                                        if not pair_rows:
                                            st.info("Nėra pakankamai konceptų hitų šiai metų porai.")
                                        else:
                                            def _pair_concept_sort_key(r):
                                                if r["Change"] == "↑":
                                                    return (0, -r["_delta"])
                                                if r["Change"] == "→":
                                                    return (1, 0)
                                                return (2, r["_delta"])

                                            pair_rows.sort(key=_pair_concept_sort_key)
                                            pair_df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in pair_rows])
                                            st.dataframe(pair_df, width="stretch")
    
                            # Koncepcijų metiniai pokyčiai pagal Share (%): 2017–2018, 2018–2019 ir t.t., rodyklės ↑/↓/→ + spalvų gradientas pagal pokyčio dydį
                            if len(year_ints) >= 2:
                                years_sorted = [int(y) for y in year_ints]
                                years_sorted.sort()
                                year_pairs: List[Tuple[int, int]] = [
                                    (years_sorted[i], years_sorted[i + 1]) for i in range(len(years_sorted) - 1)
                                ]

                                yoy_df = pd.DataFrame()
                                yoy_delta_df = pd.DataFrame()
                                numeric_cols: List[str] = [f"{y1}-{y2}" for y1, y2 in year_pairs]

                                green_light, green_dark = "#e8f5e9", "#0d5c0d"
                                red_light, red_dark = "#ffcdd2", "#b71c1c"
                                orange_light, orange_dark = "#ffe0b2", "#e65100"

                                def _style_yoy_with_delta(data: pd.DataFrame, delta_df: pd.DataFrame) -> pd.DataFrame:
                                    styles = pd.DataFrame("", index=data.index, columns=data.columns)
                                    cols = [col for col in data.columns if col != "Concept"]

                                    # Separate scales:
                                    # - '↑' gradient: based on the maximum positive delta in the whole table
                                    # - '↓' gradient: based on the maximum absolute negative delta in the whole table
                                    max_up = 0.0
                                    max_down = 0.0
                                    for c in cols:
                                        ser = pd.to_numeric(delta_df.get(c), errors="coerce")
                                        pos = ser[ser > share_epsilon]
                                        if not pos.empty:
                                            max_up = max(max_up, float(pos.max()))
                                        neg = ser[ser < -share_epsilon]
                                        if not neg.empty:
                                            # neg values are negative; compare by absolute magnitude
                                            max_down = max(max_down, float((-neg).max()))

                                    # Avoid division by zero; if no such deltas exist, all gradients will become light.
                                    if max_up <= 0:
                                        max_up = 1.0
                                    if max_down <= 0:
                                        max_down = 1.0

                                    for i in data.index:
                                        for c in data.columns:
                                            if c == "Concept":
                                                continue
                                            sym = _yoy_arrow_from_cell(data.loc[i, c])
                                            if sym not in ("↑", "↓", "→"):
                                                continue
                                            delta_val = delta_df.loc[i, c]
                                            if pd.isna(delta_val):
                                                continue
                                            if sym == "↑":
                                                t = min(1.0, float(abs(delta_val)) / max_up) if max_up > 0 else 0.0
                                                bg = _lerp_hex(green_light, green_dark, t)
                                            elif sym == "↓":
                                                t = min(1.0, float(abs(delta_val)) / max_down) if max_down > 0 else 0.0
                                                bg = _lerp_hex(red_light, red_dark, t)
                                            else:  # "→"
                                                # Stable: fixed orange (no gradient).
                                                bg = orange_light
                                            styles.loc[i, c] = f"background-color: {bg}"
                                    return styles

                                if not aggregate_view_enabled:
                                    yoy_rows: List[Dict[str, str]] = []
                                    yoy_delta_rows: List[Dict[str, float]] = []
                                    for concept in sorted(seen_concepts):
                                        row: Dict[str, str] = {"Concept": concept}
                                        delta_row: Dict[str, float] = {"Concept": 0.0}
                                        for y1, y2 in year_pairs:
                                            c1 = concept_year_count.get((concept, y1), 0)
                                            c2 = concept_year_count.get((concept, y2), 0)
                                            total1 = year_total_hits.get(y1, 0)
                                            total2 = year_total_hits.get(y2, 0)
                                            s1 = (c1 / total1 * 100.0) if total1 > 0 else 0.0
                                            s2 = (c2 / total2 * 100.0) if total2 > 0 else 0.0
                                            delta = s2 - s1
                                            if s2 > s1 + share_epsilon:
                                                sym = "↑"
                                            elif s2 < s1 - share_epsilon:
                                                sym = "↓"
                                            else:
                                                sym = "→"
                                            col_name = f"{y1}-{y2}"
                                            row[col_name] = _yoy_cell_display(sym, delta)
                                            delta_row[col_name] = delta
                                        yoy_rows.append(row)
                                        yoy_delta_rows.append(delta_row)

                                    if yoy_rows:
                                        yoy_df = pd.DataFrame(yoy_rows).reset_index(drop=True)
                                        yoy_delta_df = pd.DataFrame(yoy_delta_rows).reindex_like(yoy_df)
                                        yoy_styled = yoy_df.style.apply(
                                            lambda _data: _style_yoy_with_delta(_data, yoy_delta_df), axis=None
                                        )
                                        st.markdown("### Koncepcijų metiniai pokyčiai (Share, ↑/↓/→)")
                                        st.dataframe(yoy_styled, width="stretch")
                                else:
                                    modes_data_yoy = st.session_state.get(f"_data_set{set_id}_modes")
                                    if not modes_data_yoy:
                                        st.warning(
                                            "Aggregate mode įjungtas, bet trūksta per-mode rezultatų Koncepcijų metinių pokyčių lentelei. "
                                            "Paleisk „Run Set (Aggregate all modes)“ dar kartą."
                                        )
                                    else:
                                        agg_modes_yoy = get_effective_aggregate_modes(
                                            list(mode_labels.keys()),
                                            terms_df,
                                            preferred_mode=match_mode,
                                        )
                                        year_set_yoy = {int(y) for y in year_ints}
                                        mode_to_yoy_df: Dict[str, pd.DataFrame] = {}
                                        mode_to_delta_df: Dict[str, pd.DataFrame] = {}
                                        all_concepts_yoy: set = set()

                                        for m in agg_modes_yoy:
                                            data_m = modes_data_yoy.get(m)
                                            if not data_m:
                                                continue
                                            docs_m = data_m["docs"]
                                            doc_term_hits_m = data_m["doc_term_hits"]
                                            concept_year_count_m: Dict[Tuple[str, int], int] = defaultdict(int)
                                            year_total_hits_m: Dict[int, int] = defaultdict(int)

                                            for filename, _text in docs_m:
                                                y_int = safe_year(st.session_state[_mk(filename)]["year"])
                                                if y_int is None or y_int not in year_set_yoy:
                                                    continue
                                                th = doc_term_hits_m.get(filename)
                                                if th is None or th.empty:
                                                    continue
                                                for _, r in th.iterrows():
                                                    concept = safe_str(r["Concept"])
                                                    if not concept:
                                                        continue
                                                    cnt = int(r["Count"])
                                                    concept_year_count_m[(concept, y_int)] += cnt
                                                    year_total_hits_m[y_int] += cnt

                                            seen_concepts_m = sorted({c for (c, _y) in concept_year_count_m.keys()})
                                            all_concepts_yoy.update(seen_concepts_m)
                                            rows_m: List[Dict[str, str]] = []
                                            delta_rows_m: List[Dict[str, float]] = []
                                            for concept in seen_concepts_m:
                                                row_m: Dict[str, str] = {"Concept": concept}
                                                delta_row_m: Dict[str, float] = {"Concept": 0.0}
                                                for y1, y2 in year_pairs:
                                                    c1 = concept_year_count_m.get((concept, y1), 0)
                                                    c2 = concept_year_count_m.get((concept, y2), 0)
                                                    total1 = year_total_hits_m.get(y1, 0)
                                                    total2 = year_total_hits_m.get(y2, 0)
                                                    s1 = (c1 / total1 * 100.0) if total1 > 0 else 0.0
                                                    s2 = (c2 / total2 * 100.0) if total2 > 0 else 0.0
                                                    delta = s2 - s1
                                                    if s2 > s1 + share_epsilon:
                                                        sym = "↑"
                                                    elif s2 < s1 - share_epsilon:
                                                        sym = "↓"
                                                    else:
                                                        sym = "→"
                                                    col_name = f"{y1}-{y2}"
                                                    row_m[col_name] = _yoy_cell_display(sym, delta)
                                                    delta_row_m[col_name] = delta
                                                rows_m.append(row_m)
                                                delta_rows_m.append(delta_row_m)
                                            if rows_m:
                                                df_m = pd.DataFrame(rows_m).reset_index(drop=True)
                                                delta_df_m = pd.DataFrame(delta_rows_m).reindex_like(df_m)
                                                mode_to_yoy_df[m] = df_m
                                                mode_to_delta_df[m] = delta_df_m

                                        # Aggregate symbols+delta across modes
                                        yoy_rows_agg: List[Dict[str, str]] = []
                                        yoy_delta_rows_agg: List[Dict[str, float]] = []
                                        for concept in sorted(all_concepts_yoy):
                                            row_agg: Dict[str, str] = {"Concept": concept}
                                            delta_row_agg: Dict[str, float] = {"Concept": 0.0}
                                            for col_name in numeric_cols:
                                                votes: List[str] = []
                                                deltas: List[float] = []
                                                sym_to_deltas: Dict[str, List[float]] = {"↑": [], "↓": [], "→": []}
                                                for m in agg_modes_yoy:
                                                    df_m = mode_to_yoy_df.get(m)
                                                    delta_df_m = mode_to_delta_df.get(m)
                                                    if df_m is None or delta_df_m is None:
                                                        continue
                                                    row_match = df_m[df_m["Concept"] == concept]
                                                    if row_match.empty:
                                                        continue
                                                    sym = _yoy_arrow_from_cell(row_match.iloc[0][col_name])
                                                    delta_match = delta_df_m[delta_df_m["Concept"] == concept]
                                                    if sym in ("↑", "↓", "→"):
                                                        votes.append(sym)
                                                    if not delta_match.empty:
                                                        d_val = delta_match.iloc[0][col_name]
                                                        if pd.notna(d_val):
                                                            d_float = float(d_val)
                                                            deltas.append(d_float)
                                                            if sym in sym_to_deltas:
                                                                sym_to_deltas[sym].append(d_float)

                                                if not votes:
                                                    delta_row_agg[col_name] = 0.0
                                                    row_agg[col_name] = _yoy_cell_display("→", 0.0)
                                                    continue

                                                vote_counts: Dict[str, int] = defaultdict(int)
                                                for v in votes:
                                                    vote_counts[v] += 1
                                                max_count = max(vote_counts.values()) if vote_counts else 0
                                                leaders = [v for v, c in vote_counts.items() if c == max_count]
                                                chosen_sym = "→"
                                                if len(leaders) == 1:
                                                    chosen_sym = leaders[0]
                                                else:
                                                    # tie-break: if HanLP and Jieba-search agree on direction
                                                    hanlp_sym = ""
                                                    search_sym = ""
                                                    df_h = mode_to_yoy_df.get("hanlp")
                                                    df_s = mode_to_yoy_df.get("jieba_search")
                                                    if df_h is not None:
                                                        r_h = df_h[df_h["Concept"] == concept]
                                                        if not r_h.empty:
                                                            hanlp_sym = _yoy_arrow_from_cell(r_h.iloc[0][col_name])
                                                    if df_s is not None:
                                                        r_s = df_s[df_s["Concept"] == concept]
                                                        if not r_s.empty:
                                                            search_sym = _yoy_arrow_from_cell(r_s.iloc[0][col_name])
                                                    chosen_sym = hanlp_sym if (hanlp_sym and search_sym and hanlp_sym == search_sym) else "→"

                                                # Keep gradient strength aligned with chosen direction.
                                                # Use the strongest same-direction delta to avoid washed-out aggregate colors.
                                                if chosen_sym in ("↑", "↓") and sym_to_deltas.get(chosen_sym):
                                                    candidates = [float(v) for v in sym_to_deltas[chosen_sym] if pd.notna(v)]
                                                    if candidates:
                                                        chosen_delta = max(candidates, key=lambda v: abs(v))
                                                    else:
                                                        chosen_delta = (share_epsilon + 0.01) * (1.0 if chosen_sym == "↑" else -1.0)
                                                    if abs(chosen_delta) < (share_epsilon + 0.01):
                                                        chosen_delta = (share_epsilon + 0.01) * (1.0 if chosen_sym == "↑" else -1.0)
                                                    delta_row_agg[col_name] = chosen_delta
                                                elif chosen_sym == "→" and sym_to_deltas.get("→"):
                                                    delta_row_agg[col_name] = sum(sym_to_deltas["→"]) / len(sym_to_deltas["→"])
                                                else:
                                                    delta_row_agg[col_name] = (sum(deltas) / len(deltas)) if deltas else 0.0

                                                row_agg[col_name] = _yoy_cell_display(chosen_sym, float(delta_row_agg[col_name]))

                                            yoy_rows_agg.append(row_agg)
                                            yoy_delta_rows_agg.append(delta_row_agg)

                                        if yoy_rows_agg:
                                            yoy_df = pd.DataFrame(yoy_rows_agg).reset_index(drop=True)
                                            yoy_delta_df = pd.DataFrame(yoy_delta_rows_agg).reindex_like(yoy_df)
                                            with st.container(border=True):
                                                st.caption("Aggregate + per-mode blokas")
                                                yoy_styled = yoy_df.style.apply(
                                                    lambda _data: _style_yoy_with_delta(_data, yoy_delta_df),
                                                    axis=None,
                                                )
                                                st.markdown("### Koncepcijų metiniai pokyčiai (Share, ↑/↓/→) — aggregated")
                                                st.dataframe(yoy_styled, width="stretch")

                                                # Per-mode Koncepcijų metiniai pokyčiai pirmiausia (po agreguotos lentelės)
                                                st.markdown("#### Per-mode Koncepcijų metiniai pokyčiai")
                                                for m in agg_modes_yoy:
                                                    df_m = mode_to_yoy_df.get(m)
                                                    delta_df_m = mode_to_delta_df.get(m)
                                                    if df_m is None or delta_df_m is None:
                                                        continue
                                                    label = mode_labels.get(m, m)
                                                    with st.expander(f"{label} ({m})", expanded=False):
                                                        st.caption(describe_mode_runtime(m))
                                                        df_m_styled = df_m.style.apply(
                                                            lambda _data: _style_yoy_with_delta(_data, delta_df_m), axis=None
                                                        )
                                                        st.dataframe(df_m_styled, width="stretch")

                                    # Tekstinės išvados tik kai turime pilnai suformuotą YOY lentelę su periodų stulpeliais.
                                    yoy_ready = (
                                        not yoy_df.empty
                                        and not yoy_delta_df.empty
                                        and len(numeric_cols) > 0
                                    )
                                    if yoy_ready:
                                        analysis_cols = [c for c in numeric_cols if c in yoy_df.columns and c in yoy_delta_df.columns]
                                        if not analysis_cols:
                                            analysis_cols = [c for c in yoy_df.columns if c != "Concept" and c in yoy_delta_df.columns]
                                        if not analysis_cols:
                                            analysis_cols = []

                                        # Išvados: labiausiai augo/krito periodai, „krito nuo…“, atsistabilizavosi
                                        conclusion_lines: List[str] = []
                                        for i in yoy_df.index:
                                            concept = str(yoy_df.loc[i, "Concept"])
                                            parts: List[str] = []
                                            # Labiausiai augo (periodas su didžiausiu teigiamu delta)
                                            best_up_col = None
                                            best_up_delta = -1e9
                                            for c in analysis_cols:
                                                d = yoy_delta_df.loc[i, c]
                                                if pd.notna(d) and d > best_up_delta and d > share_epsilon:
                                                    best_up_delta = d
                                                    best_up_col = c
                                            if best_up_col:
                                                parts.append(f"labiausiai augo {best_up_col}")
                                            # Labiausiai krito (periodas su mažiausiu delta)
                                            best_down_col = None
                                            best_down_delta = 1e9
                                            for c in analysis_cols:
                                                d = yoy_delta_df.loc[i, c]
                                                if pd.notna(d) and d < best_down_delta and d < -share_epsilon:
                                                    best_down_delta = d
                                                    best_down_col = c
                                            if best_down_col:
                                                parts.append(f"labiausiai krito {best_down_col}")
                                            # Krito nuo X–Y; paskutiniais metais atsistabilizavosi
                                            first_down = None
                                            for c in analysis_cols:
                                                if _yoy_arrow_from_cell(yoy_df.loc[i, c]) == "↓":
                                                    first_down = c
                                                    break
                                            n_last = min(2, len(analysis_cols))
                                            last_periods = analysis_cols[-n_last:] if analysis_cols else []
                                            last_stable_or_up = (
                                                all(_yoy_arrow_from_cell(yoy_df.loc[i, c]) in ("→", "↑") for c in last_periods)
                                                if last_periods else False
                                            )
                                            if first_down and last_stable_or_up:
                                                parts.append(f"krito nuo {first_down}, paskutiniais metais atsistabilizavosi")
                                            elif first_down and not (best_down_col and best_down_col == first_down and len(parts) >= 1):
                                                parts.append(f"krito nuo {first_down}")
                                            # Tendencija dabar (naudojam paskutinį periodą ir kelis paskutinius)
                                            trend_now = None
                                            if analysis_cols:
                                                last_col = analysis_cols[-1]
                                                last_sym = _yoy_arrow_from_cell(yoy_df.loc[i, last_col])
                                                delta_last_raw = yoy_delta_df.loc[i, last_col]
                                                delta_last = float(delta_last_raw) if pd.notna(delta_last_raw) else 0.0
                                                recent_cols = analysis_cols[-3:]
                                                recent_syms = [_yoy_arrow_from_cell(yoy_df.loc[i, c]) for c in recent_cols]
                                                if last_sym == "↑" and abs(delta_last) > share_epsilon and recent_syms.count("↓") == 0:
                                                    if abs(delta_last) > 0.5:
                                                        trend_now = "tendencija dabar: ryškiai kyla"
                                                    else:
                                                        trend_now = "tendencija dabar: švelniai kyla"
                                                elif last_sym == "↓" and abs(delta_last) > share_epsilon and recent_syms.count("↑") == 0:
                                                    if abs(delta_last) > 0.5:
                                                        trend_now = "tendencija dabar: ryškiai mažėja"
                                                    else:
                                                        trend_now = "tendencija dabar: švelniai mažėja"
                                                elif all(s == "→" for s in recent_syms):
                                                    trend_now = "tendencija dabar: stabili"
                                                elif last_sym == "↑":
                                                    trend_now = "tendencija dabar: kyla"
                                                elif last_sym == "↓":
                                                    trend_now = "tendencija dabar: mažėja"
                                                else:
                                                    trend_now = "tendencija dabar: stabili"
                                            if trend_now:
                                                parts.append(trend_now)
                                            if parts:
                                                conclusion_lines.append(f"- **{concept}**: " + "; ".join(parts) + ".")
                                        if conclusion_lines:
                                            st.markdown("#### Išvados")
                                            st.markdown("\n".join(conclusion_lines))
                                        
                                        # Papildomas, "žmogiškas" įvykių paaiškinimas: kas galėjo lemti kryptį (↑/↓) perioduose
                                        if analysis_cols:
                                            best_up_period_pair = None
                                            best_up_score = -1e9
                                            best_down_period_pair = None
                                            best_down_score = -1e9
                                            for col in analysis_cols:
                                                col_arrows = yoy_df[col].map(_yoy_arrow_from_cell)
                                                up_cnt = int((col_arrows == "↑").sum())
                                                down_cnt = int((col_arrows == "↓").sum())
                                                if up_cnt > down_cnt and up_cnt > 0:
                                                    score = (up_cnt - down_cnt)
                                                    if score > best_up_score:
                                                        best_up_score = score
                                                        best_up_period_pair = col
                                                if down_cnt > up_cnt and down_cnt > 0:
                                                    score = (down_cnt - up_cnt)
                                                    if score > best_down_score:
                                                        best_down_score = score
                                                        best_down_period_pair = col

                                            best_up_concept = None
                                            best_up_delta = None
                                            if best_up_period_pair:
                                                best_up_delta = -1e9
                                                for ii in yoy_df.index:
                                                    if _yoy_arrow_from_cell(yoy_df.loc[ii, best_up_period_pair]) != "↑":
                                                        continue
                                                    d_loc = yoy_delta_df.loc[ii, best_up_period_pair]
                                                    if pd.notna(d_loc):
                                                        d_float = float(d_loc)
                                                        if d_float > best_up_delta:
                                                            best_up_delta = d_float
                                                            best_up_concept = str(yoy_df.loc[ii, "Concept"])

                                            best_down_concept = None
                                            best_down_delta = None
                                            if best_down_period_pair:
                                                best_down_delta = 1e9
                                                for ii in yoy_df.index:
                                                    if _yoy_arrow_from_cell(yoy_df.loc[ii, best_down_period_pair]) != "↓":
                                                        continue
                                                    d_loc = yoy_delta_df.loc[ii, best_down_period_pair]
                                                    if pd.notna(d_loc):
                                                        d_float = float(d_loc)
                                                        if d_float < best_down_delta:
                                                            best_down_delta = d_float
                                                            best_down_concept = str(yoy_df.loc[ii, "Concept"])

                                            # Render
                                            st.markdown("#### Galimi įvykių paaiškinimai (kodėl matome kryptį ↑/↓/→)")
                                            any_rendered = False
                                            if best_up_period_pair and best_up_concept and best_up_delta is not None:
                                                col_arrows_u = yoy_df[best_up_period_pair].map(_yoy_arrow_from_cell)
                                                up_cnt = int((col_arrows_u == "↑").sum())
                                                down_cnt = int((col_arrows_u == "↓").sum())
                                                evs_up = suggest_events_for_period(events_df, best_up_period_pair, max_events=3)
                                                st.markdown(
                                                    f"Periodu **{best_up_period_pair}** agreguota lentelė rodo daugiau augimo (`↑`: {up_cnt}, `↓`: {down_cnt}). "
                                                    f"Didžiausias augimas: **{best_up_concept}** (Δ≈{best_up_delta:.2f} pp)."
                                                )
                                                if evs_up:
                                                    st.markdown("Galimi kontekstiniai įvykiai:")
                                                    st.markdown("\n".join([f"- {e}" for e in evs_up]))
                                                any_rendered = True

                                            if best_down_period_pair and best_down_concept and best_down_delta is not None:
                                                col_arrows_d = yoy_df[best_down_period_pair].map(_yoy_arrow_from_cell)
                                                up_cnt = int((col_arrows_d == "↑").sum())
                                                down_cnt = int((col_arrows_d == "↓").sum())
                                                evs_down = suggest_events_for_period(events_df, best_down_period_pair, max_events=3)
                                                st.markdown(
                                                    f"Periodu **{best_down_period_pair}** agreguota lentelė rodo daugiau mažėjimo (`↓`: {down_cnt}, `↑`: {up_cnt}). "
                                                    f"Didžiausias kritimas: **{best_down_concept}** (Δ≈{best_down_delta:.2f} pp)."
                                                )
                                                if evs_down:
                                                    st.markdown("Galimi kontekstiniai įvykiai:")
                                                    st.markdown("\n".join([f"- {e}" for e in evs_down]))
                                                any_rendered = True

                                            if not any_rendered:
                                                st.info("Nepavyko identifikuoti ryškių dominuojančių ↑/↓ periodų iš pateiktos YOY lentelės.")
                                        # Stipriausias augimas/kritimas visoje lentelėje
                                        best_concept, best_period, best_d = None, None, -1e9
                                        worst_concept, worst_period, worst_d = None, None, 1e9
                                        for i in yoy_df.index:
                                            for c in analysis_cols:
                                                d = yoy_delta_df.loc[i, c]
                                                if pd.notna(d):
                                                    if d > best_d and d > share_epsilon:
                                                        best_d, best_concept, best_period = d, str(yoy_df.loc[i, "Concept"]), c
                                                    if d < worst_d and d < -share_epsilon:
                                                        worst_d, worst_concept, worst_period = d, str(yoy_df.loc[i, "Concept"]), c
                                        summary_parts: List[str] = []
                                        if best_concept and best_period:
                                            summary_parts.append(f"Stipriausias augimas: **{best_concept}** ({best_period}).")
                                        if worst_concept and worst_period:
                                            summary_parts.append(f"Stipriausias kritimas: **{worst_concept}** ({worst_period}).")
                                        if summary_parts:
                                            st.markdown("\n\n".join(summary_parts))

                                        # Papildoma prognozė remiantis paskutiniu periodu
                                        will_grow: List[str] = []
                                        will_decline: List[str] = []
                                        will_stay: List[str] = []
                                        if analysis_cols:
                                            last_col = analysis_cols[-1]
                                            for i in yoy_df.index:
                                                concept = str(yoy_df.loc[i, "Concept"])
                                                sym = _yoy_arrow_from_cell(yoy_df.loc[i, last_col])
                                                delta_last_raw = yoy_delta_df.loc[i, last_col]
                                                delta_last = float(delta_last_raw) if pd.notna(delta_last_raw) else 0.0
                                                if sym == "↑" and delta_last > share_epsilon:
                                                    will_grow.append(concept)
                                                elif sym == "↓" and delta_last < -share_epsilon:
                                                    will_decline.append(concept)
                                                elif sym == "→" and abs(delta_last) <= share_epsilon:
                                                    will_stay.append(concept)
                                        forecast_lines: List[str] = []
                                        if will_grow:
                                            forecast_lines.append(
                                                "Tikėtina, kad artimiausiu metu augs: " + ", ".join(sorted(set(will_grow))) + "."
                                            )
                                        if will_decline:
                                            forecast_lines.append(
                                                "Tikėtina, kad artimiausiu metu mažės: " + ", ".join(sorted(set(will_decline))) + "."
                                            )
                                        if will_stay:
                                            forecast_lines.append(
                                                "Stabiliai laikosi: " + ", ".join(sorted(set(will_stay))) + "."
                                            )
                                        if forecast_lines:
                                            st.markdown("\n\n".join(forecast_lines))

                                        # Įvykių kontekstas stipriausiam augimui/kritimui (atskirti blokai, be sulipimo)
                                        evs_up: List[str] = []
                                        evs_down: List[str] = []
                                        if best_concept and best_period:
                                            evs_up = suggest_events_for_period(events_df, best_period)
                                        if worst_concept and worst_period:
                                            evs_down = suggest_events_for_period(events_df, worst_period)
                                        if evs_up or evs_down:
                                            st.markdown("#### Galimi paaiškinimai (įvykių kontekstas)")
                                        if evs_up:
                                            st.markdown(
                                                f"Galimi kontekstiniai įvykiai, susiję su stipriausiu augimu ({best_period}):"
                                            )
                                            st.markdown("\n".join([f"- {e}" for e in evs_up]))
                                        if evs_down:
                                            st.markdown(
                                                f"Galimi kontekstiniai įvykiai, susiję su stipriausiu kritimu ({worst_period}):"
                                            )
                                            st.markdown("\n".join([f"- {e}" for e in evs_down]))

                                        # (Per-mode YOY lentelės dabar rodomos prieš Išvadas ir įvykių kontekstą.)
    
                        # Relative importance chart: Top 10 terms per year/period (vertimai + pastovi spalva per terminą)
                        top10_rows = []
                        top10_terms_matrix: List[List[str]] = []

                        if not aggregate_view_enabled:
                            # Original chart (single match_mode)
                            if len(year_ints) >= 2:
                                for y in year_ints:
                                    y_int = int(y)
                                    term_counts = [(term, term_year_count.get((term, y_int), 0)) for term in seen_terms]
                                    term_counts = [(t, c) for t, c in term_counts if c > 0]
                                    term_counts.sort(key=lambda x: -x[1])
                                    top10_terms = [t[0] for t in term_counts[:10]]
                                    top10_display = [term_to_translation.get(t, t) for t in top10_terms]
                                    row = {"Year": str(y_int)}
                                    for rk in range(1, 11):
                                        row[rk] = top10_display[rk - 1] if rk <= len(top10_display) else ""
                                    top10_rows.append(row)
                                    top10_terms_matrix.append(top10_terms + [""] * (10 - len(top10_terms)))
                            else:
                                for p in periods_sorted:
                                    term_counts = [(term, term_period_count.get((term, p), 0)) for term in seen_terms]
                                    term_counts = [(t, c) for t, c in term_counts if c > 0]
                                    term_counts.sort(key=lambda x: -x[1])
                                    top10_terms = [t[0] for t in term_counts[:10]]
                                    top10_display = [term_to_translation.get(t, t) for t in top10_terms]
                                    row = {"Period": p}
                                    for rk in range(1, 11):
                                        row[rk] = top10_display[rk - 1] if rk <= len(top10_display) else ""
                                    top10_rows.append(row)
                                    top10_terms_matrix.append(top10_terms + [""] * (10 - len(top10_terms)))

                            if top10_rows:
                                all_terms_in_chart = sorted(set(t for row_terms in top10_terms_matrix for t in row_terms if t))
                                term_to_color_chart = {
                                    t: HIGHLIGHT_PALETTE[i % len(HIGHLIGHT_PALETTE)]
                                    for i, t in enumerate(all_terms_in_chart)
                                }
                                top10_df = pd.DataFrame(top10_rows)

                                def _top10_cell_style(row_series):
                                    idx = row_series.name
                                    row_terms = top10_terms_matrix[idx] if idx < len(top10_terms_matrix) else []
                                    styles = []
                                    for c in row_series.index:
                                        if c in ("Year", "Period"):
                                            styles.append("background-color: #ffffff")
                                        else:
                                            j = int(c) - 1 if isinstance(c, (int, float)) else 0
                                            term = row_terms[j] if j < len(row_terms) else ""
                                            bg = term_to_color_chart.get(term, "#ffffff") if term else "#ffffff"
                                            styles.append(f"background-color: {bg}")
                                    return styles

                                top10_styled = top10_df.style.apply(_top10_cell_style, axis=1)
                                st.markdown("### Relative importance chart (Top 10 terms per year/period)")
                                st.dataframe(top10_styled, width="stretch")

                        else:
                            # Aggregated chart across all match modes (only this table)
                            modes_data = st.session_state.get(f"_data_set{set_id}_modes")
                            if not modes_data:
                                st.warning(
                                    "Aggregate mode įjungtas, bet trūksta per-mode rezultatų. "
                                    "Paleisk „Run Set (Aggregate all modes)“ dar kartą."
                                )
                            else:
                                agg_modes = get_effective_aggregate_modes(
                                    list(mode_labels.keys()),
                                    terms_df,
                                    preferred_mode=match_mode,
                                )
                                if not has_single_char_terms(terms_df) and ("hybrid" in mode_labels and "substring" in mode_labels):
                                    st.caption(
                                        "Žodyne nėra 1 hieroglifo termų: agregate „hybrid“ ir „substring“ skaičiuojami kaip vienas režimas."
                                    )
                                n_rows = len(year_ints) if len(year_ints) >= 2 else len(periods_sorted)
                                n_ranks = 10
                                year_set = {int(y) for y in year_ints} if len(year_ints) >= 2 else set()
                                period_set = set(periods_sorted) if len(year_ints) < 2 else set()

                                mode_to_matrix: Dict[str, List[List[str]]] = {}
                                all_terms_union: set = set()

                                # Build top10 matrices per mode
                                for m in agg_modes:
                                    data_m = modes_data.get(m)
                                    if not data_m:
                                        continue

                                    docs_m = data_m["docs"]
                                    doc_term_hits_m = data_m["doc_term_hits"]

                                    if len(year_ints) >= 2:
                                        year_to_term_counts: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
                                        for filename, _text in docs_m:
                                            y_int = safe_year(st.session_state[_mk(filename)]["year"])
                                            if y_int is None or y_int not in year_set:
                                                continue
                                            th = doc_term_hits_m.get(filename)
                                            if th is None or th.empty:
                                                continue
                                            for _, r in th.iterrows():
                                                term = safe_str(r["CH term"])
                                                if not term:
                                                    continue
                                                year_to_term_counts[y_int][term] += int(r["Count"])

                                        matrix = []
                                        for y in year_ints:
                                            y_int = int(y)
                                            term_counts = list(year_to_term_counts.get(y_int, {}).items())
                                            term_counts = [(t, c) for t, c in term_counts if c > 0]
                                            term_counts.sort(key=lambda x: -x[1])
                                            top10_terms = [t for t, _c in term_counts[:10]]
                                            all_terms_union.update(top10_terms)
                                            matrix.append(top10_terms + [""] * (10 - len(top10_terms)))
                                        mode_to_matrix[m] = matrix
                                    else:
                                        period_to_term_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
                                        for filename, _text in docs_m:
                                            y_int = safe_year(st.session_state[_mk(filename)]["year"])
                                            if y_int is None:
                                                continue
                                            y_str, mo, _ = parse_year_month_day_from_filename(filename)
                                            period = f"{y_str}-{mo}" if (mo and mo != "??") else str(y_int)
                                            if period not in period_set:
                                                continue
                                            th = doc_term_hits_m.get(filename)
                                            if th is None or th.empty:
                                                continue
                                            for _, r in th.iterrows():
                                                term = safe_str(r["CH term"])
                                                if not term:
                                                    continue
                                                period_to_term_counts[period][term] += int(r["Count"])

                                        matrix = []
                                        for p in periods_sorted:
                                            term_counts = list(period_to_term_counts.get(p, {}).items())
                                            term_counts = [(t, c) for t, c in term_counts if c > 0]
                                            term_counts.sort(key=lambda x: -x[1])
                                            top10_terms = [t for t, _c in term_counts[:10]]
                                            all_terms_union.update(top10_terms)
                                            matrix.append(top10_terms + [""] * (10 - len(top10_terms)))
                                        mode_to_matrix[m] = matrix

                                if not mode_to_matrix:
                                    st.warning("Nepavyko paruošti top10 matricų agregavimui.")
                                else:
                                    # Color mapping based on union of all terms
                                    all_terms_in_chart = sorted(all_terms_union)
                                    term_to_color_chart = {
                                        t: HIGHLIGHT_PALETTE[i % len(HIGHLIGHT_PALETTE)]
                                        for i, t in enumerate(all_terms_in_chart)
                                    }

                                    # Aggregate cell-by-cell: majority vote; tie => grey/blank
                                    aggregated_matrix: List[List[str]] = [[""] * n_ranks for _ in range(n_rows)]
                                    grey_mask: List[List[bool]] = [[False] * n_ranks for _ in range(n_rows)]

                                    for i in range(n_rows):
                                        for j in range(n_ranks):
                                            votes: List[str] = []
                                            for m in agg_modes:
                                                mat = mode_to_matrix.get(m)
                                                if not mat or i >= len(mat) or j >= len(mat[i]):
                                                    continue
                                                t = mat[i][j]
                                                if t:
                                                    votes.append(t)

                                            if not votes:
                                                grey_mask[i][j] = True
                                                aggregated_matrix[i][j] = ""
                                                continue

                                            vote_counts: Dict[str, int] = defaultdict(int)
                                            for t in votes:
                                                vote_counts[t] += 1
                                            max_count = max(vote_counts.values()) if vote_counts else 0
                                            leaders = [t for t, c in vote_counts.items() if c == max_count]
                                            if len(leaders) > 1:
                                                # Tie-break: if HanLP and Jieba-search agree, prefer that value.
                                                hanlp_vote = ""
                                                search_vote = ""
                                                mat_hanlp = mode_to_matrix.get("hanlp")
                                                mat_search = mode_to_matrix.get("jieba_search")
                                                if mat_hanlp and i < len(mat_hanlp) and j < len(mat_hanlp[i]):
                                                    hanlp_vote = safe_str(mat_hanlp[i][j]).strip()
                                                if mat_search and i < len(mat_search) and j < len(mat_search[i]):
                                                    search_vote = safe_str(mat_search[i][j]).strip()
                                                if hanlp_vote and search_vote and hanlp_vote == search_vote:
                                                    aggregated_matrix[i][j] = hanlp_vote
                                                else:
                                                    grey_mask[i][j] = True
                                                    aggregated_matrix[i][j] = ""
                                            else:
                                                aggregated_matrix[i][j] = leaders[0]

                                    # Build aggregated dataframe
                                    for i in range(n_rows):
                                        if len(year_ints) >= 2:
                                            y_int = int(year_ints[i])
                                            row = {"Year": str(y_int)}
                                        else:
                                            row = {"Period": periods_sorted[i]}

                                        for rk in range(1, 11):
                                            term = aggregated_matrix[i][rk - 1]
                                            row[rk] = term_to_translation.get(term, term) if term else ""
                                        top10_rows.append(row)

                                    top10_terms_matrix = aggregated_matrix
                                    if top10_rows:
                                        top10_df = pd.DataFrame(top10_rows)

                                        def _top10_cell_style(row_series):
                                            idx = row_series.name
                                            row_terms = top10_terms_matrix[idx] if idx < len(top10_terms_matrix) else []
                                            styles = []
                                            for c in row_series.index:
                                                if c in ("Year", "Period"):
                                                    styles.append("background-color: #ffffff")
                                                else:
                                                    j = int(c) - 1 if isinstance(c, (int, float)) else 0
                                                    term = row_terms[j] if j < len(row_terms) else ""
                                                    if idx < len(grey_mask) and j < len(grey_mask[idx]) and grey_mask[idx][j]:
                                                        styles.append("background-color: #cfcfcf")
                                                    else:
                                                        bg = term_to_color_chart.get(term, "#ffffff") if term else "#ffffff"
                                                        styles.append(f"background-color: {bg}")
                                            return styles

                                        top10_styled = top10_df.style.apply(_top10_cell_style, axis=1)
                                        with st.container(border=True):
                                            st.caption("Aggregate + per-mode blokas")
                                            st.markdown("### Relative importance chart (Top 10 terms per year/period) — aggregated")
                                            st.dataframe(top10_styled, width="stretch")

                                            # Per-mode charts
                                            st.markdown("#### Per-mode Relative importance chart")
                                            for m in agg_modes:
                                                mat_m = mode_to_matrix.get(m)
                                                if not mat_m:
                                                    continue
                                                label = mode_labels.get(m, m)
                                                with st.expander(f"{label} ({m})", expanded=False):
                                                    st.caption(describe_mode_runtime(m))

                                                    rows_m = []
                                                    for i in range(n_rows):
                                                        if len(year_ints) >= 2:
                                                            y_int = int(year_ints[i])
                                                            row = {"Year": str(y_int)}
                                                        else:
                                                            row = {"Period": periods_sorted[i]}

                                                        for rk in range(1, 11):
                                                            term = mat_m[i][rk - 1] if i < len(mat_m) and rk - 1 < len(mat_m[i]) else ""
                                                            row[rk] = term_to_translation.get(term, term) if term else ""
                                                        rows_m.append(row)

                                                    df_m = pd.DataFrame(rows_m)

                                                    def _mode_cell_style(row_series):
                                                        idx = row_series.name
                                                        row_terms = mat_m[idx] if idx < len(mat_m) else []
                                                        styles = []
                                                        for c in row_series.index:
                                                            if c in ("Year", "Period"):
                                                                styles.append("background-color: #ffffff")
                                                            else:
                                                                j = int(c) - 1 if isinstance(c, (int, float)) else 0
                                                                term = row_terms[j] if j < len(row_terms) else ""
                                                                bg = term_to_color_chart.get(term, "#ffffff") if term else "#ffffff"
                                                                styles.append(f"background-color: {bg}")
                                                        return styles

                                                    df_m_styled = df_m.style.apply(_mode_cell_style, axis=1)
                                                    st.dataframe(df_m_styled, width="stretch")
    
                        # Discourse shift detection: periodai, kai dominuoja tas pats konceptas (tik keli metai)
                        if len(year_ints) >= 2:
                            def _build_shift_rows(dominant_by_year: List[Tuple[int, str]]) -> List[Dict[str, str]]:
                                rows: List[Dict[str, str]] = []
                                if not dominant_by_year:
                                    return rows
                                i_local = 0
                                while i_local < len(dominant_by_year):
                                    start_year = dominant_by_year[i_local][0]
                                    concept = dominant_by_year[i_local][1]
                                    j_local = i_local + 1
                                    while j_local < len(dominant_by_year) and dominant_by_year[j_local][1] == concept:
                                        j_local += 1
                                    end_year = dominant_by_year[j_local - 1][0]
                                    period_str = f"{start_year}–{end_year}" if start_year != end_year else str(start_year)
                                    rows.append({"Period": period_str, "Dominant discourse": concept})
                                    i_local = j_local
                                return rows

                            if not aggregate_view_enabled:
                                dominant_per_year: List[Tuple[int, str]] = []
                                for y in year_ints:
                                    y_int = int(y)
                                    best_concept = ""
                                    best_count = -1
                                    for c in seen_concepts:
                                        cnt = concept_year_count.get((c, y_int), 0)
                                        if cnt > best_count:
                                            best_count = cnt
                                            best_concept = c
                                    dominant_per_year.append((y_int, best_concept))
                                shift_rows = _build_shift_rows(dominant_per_year)
                                if shift_rows:
                                    shift_df = pd.DataFrame(shift_rows)
                                    st.markdown("### Discourse shift detection")
                                    st.dataframe(shift_df, width="stretch")
                            else:
                                modes_data_shift = st.session_state.get(f"_data_set{set_id}_modes")
                                if not modes_data_shift:
                                    st.warning(
                                        "Aggregate mode įjungtas, bet trūksta per-mode rezultatų Discourse shift lentelei. "
                                        "Paleisk „Run Set (Aggregate all modes)“ dar kartą."
                                    )
                                else:
                                    with st.container(border=True):
                                        st.caption("Aggregate + per-mode blokas")
                                        agg_modes_shift = get_effective_aggregate_modes(
                                            list(mode_labels.keys()),
                                            terms_df,
                                            preferred_mode=match_mode,
                                        )
                                        year_set_shift = {int(y) for y in year_ints}
                                        mode_to_dominant: Dict[str, List[Tuple[int, str]]] = {}

                                        for m in agg_modes_shift:
                                            data_m = modes_data_shift.get(m)
                                            if not data_m:
                                                continue
                                            docs_m = data_m["docs"]
                                            doc_term_hits_m = data_m["doc_term_hits"]
                                            concept_year_count_m: Dict[Tuple[str, int], int] = defaultdict(int)

                                            for filename, _text in docs_m:
                                                y_int = safe_year(st.session_state[_mk(filename)]["year"])
                                                if y_int is None or y_int not in year_set_shift:
                                                    continue
                                                th = doc_term_hits_m.get(filename)
                                                if th is None or th.empty:
                                                    continue
                                                for _, r in th.iterrows():
                                                    concept = safe_str(r["Concept"])
                                                    if not concept:
                                                        continue
                                                    concept_year_count_m[(concept, y_int)] += int(r["Count"])

                                            dominant_per_year_m: List[Tuple[int, str]] = []
                                            seen_concepts_m = {c for (c, _y) in concept_year_count_m.keys()}
                                            for y in year_ints:
                                                y_int = int(y)
                                                best_concept = ""
                                                best_count = -1
                                                for c in seen_concepts_m:
                                                    cnt = concept_year_count_m.get((c, y_int), 0)
                                                    if cnt > best_count:
                                                        best_count = cnt
                                                        best_concept = c
                                                dominant_per_year_m.append((y_int, best_concept))
                                            mode_to_dominant[m] = dominant_per_year_m

                                        # Aggregate dominant discourse per year: majority vote; tie-break HanLP+Jieba-search agreement.
                                        aggregated_dominant: List[Tuple[int, str]] = []
                                        for y in year_ints:
                                            y_int = int(y)
                                            votes: List[str] = []
                                            for m in agg_modes_shift:
                                                dom = mode_to_dominant.get(m)
                                                if not dom:
                                                    continue
                                                vote = next((v for yy, v in dom if yy == y_int), "")
                                                vote = safe_str(vote).strip()
                                                if vote:
                                                    votes.append(vote)

                                            if not votes:
                                                aggregated_dominant.append((y_int, ""))
                                                continue

                                            vote_counts: Dict[str, int] = defaultdict(int)
                                            for v in votes:
                                                vote_counts[v] += 1
                                            max_count = max(vote_counts.values()) if vote_counts else 0
                                            leaders = [v for v, c in vote_counts.items() if c == max_count]

                                            if len(leaders) == 1:
                                                aggregated_dominant.append((y_int, leaders[0]))
                                            else:
                                                hanlp_vote = ""
                                                search_vote = ""
                                                dom_hanlp = mode_to_dominant.get("hanlp")
                                                dom_search = mode_to_dominant.get("jieba_search")
                                                if dom_hanlp:
                                                    hanlp_vote = safe_str(next((v for yy, v in dom_hanlp if yy == y_int), "")).strip()
                                                if dom_search:
                                                    search_vote = safe_str(next((v for yy, v in dom_search if yy == y_int), "")).strip()
                                                if hanlp_vote and search_vote and hanlp_vote == search_vote:
                                                    aggregated_dominant.append((y_int, hanlp_vote))
                                                else:
                                                    aggregated_dominant.append((y_int, ""))

                                        shift_rows = _build_shift_rows(aggregated_dominant)
                                        if shift_rows:
                                            shift_df = pd.DataFrame(shift_rows)
                                            st.markdown("### Discourse shift detection — aggregated")
                                            st.dataframe(shift_df, width="stretch")

                                        st.markdown("#### Per-mode Discourse shift detection")
                                        for m in agg_modes_shift:
                                            dom_m = mode_to_dominant.get(m)
                                            if not dom_m:
                                                continue
                                            label = mode_labels.get(m, m)
                                            with st.expander(f"{label} ({m})", expanded=False):
                                                st.caption(describe_mode_runtime(m))
                                                rows_m = _build_shift_rows(dom_m)
                                                if rows_m:
                                                    st.dataframe(pd.DataFrame(rows_m), width="stretch")
                                                else:
                                                    st.info("Nėra pakankamai duomenų Discourse shift lentelei.")

                        # -----------------------------
                        # Naujos santraukos lentelės (prieš Dokumentų suvestinė)
                        # -----------------------------
                        TOP_K_FOR_STABILITY = 3
                        N_TOP_CONCEPTS_PER_ROW = 5

                        if len(year_ints) >= 2:
                            # 1: Coverage pagal metus
                            _cov_rows = []
                            for _y in year_ints:
                                _yi = int(_y)
                                _sub = usable[usable["year_int"] == _yi]
                                _n_docs = len(_sub)
                                _sum_chars = int(_sub["chars"].sum()) if _n_docs else 0
                                _tot_h = year_total_hits.get(_yi, 0)
                                _per10k = normalize_per_10k_chars(_tot_h, _sum_chars)
                                _share_chk = 100.0 if _tot_h > 0 else 0.0
                                _cov_rows.append(
                                    {
                                        "year": _yi,
                                        "n_docs": _n_docs,
                                        "sum_chars": _sum_chars,
                                        "total_hits_dict": _tot_h,
                                        "total_hits_per_10k": round(_per10k, 4),
                                        "share_sum_check_%": round(_share_chk, 2),
                                    }
                                )
                            st.markdown("### Coverage pagal metus (spike sanity-check)")
                            st.caption(
                                "total_hits_dict = visų žodyno hitų suma tame year; share_sum_check_% ≈ 100, jei share skaičiuojamas kaip concept_hits/year_total_hits."
                            )
                            st.dataframe(pd.DataFrame(_cov_rows), width="stretch")

                            # 2: Concept ranking (full window) + stabilumas (tik active-years, kur hits > 0)
                            _rank_rows: List[Dict] = []
                            for _c in sorted(seen_concepts):
                                _sum_h = sum(concept_year_count.get((_c, int(_yy)), 0) for _yy in year_ints)
                                if _sum_h <= 0:
                                    continue
                                _active_shares: List[float] = []
                                for _yy in year_ints:
                                    _yii = int(_yy)
                                    _cnt = concept_year_count.get((_c, _yii), 0)
                                    _toty = year_total_hits.get(_yii, 0)
                                    if _cnt > 0 and _toty > 0:
                                        _active_shares.append(_cnt / _toty * 100.0)
                                _ay = len(_active_shares)
                                if _ay == 0:
                                    continue
                                _mean_sa = sum(_active_shares) / _ay
                                if _ay >= 2:
                                    _std_sa = float(pd.Series(_active_shares).std(ddof=0))
                                else:
                                    _std_sa = 0.0
                                _cv_sa = (_std_sa / _mean_sa) if _mean_sa > 1e-12 else 0.0
                                _topk_years = 0
                                for _yy in year_ints:
                                    _yii = int(_yy)
                                    _toty = year_total_hits.get(_yii, 0)
                                    if _toty <= 0:
                                        continue
                                    _pairs = [
                                        (_cc, concept_year_count.get((_cc, _yii), 0) / _toty * 100.0)
                                        for _cc in seen_concepts
                                    ]
                                    _pairs.sort(key=lambda x: -x[1])
                                    _leads = [_cc for _cc, _sh in _pairs[:TOP_K_FOR_STABILITY] if _sh > 0]
                                    if _c in _leads:
                                        _topk_years += 1
                                _active_year_ints = sorted(
                                    int(_yy) for _yy in year_ints if concept_year_count.get((_c, int(_yy)), 0) > 0
                                )
                                _max_streak = 0
                                if _active_year_ints:
                                    _cur = 1
                                    _max_streak = 1
                                    for _ii in range(1, len(_active_year_ints)):
                                        if _active_year_ints[_ii] == _active_year_ints[_ii - 1] + 1:
                                            _cur += 1
                                            _max_streak = max(_max_streak, _cur)
                                        else:
                                            _cur = 1
                                _rank_rows.append(
                                    {
                                        "Concept": _c,
                                        "sum_hits": _sum_h,
                                        "active_years": _ay,
                                        "mean_share_active_%": round(_mean_sa, 4),
                                        "stability_std_active_pp": round(_std_sa, 4),
                                        "stability_cv_active": round(_cv_sa, 4),
                                        f"top{TOP_K_FOR_STABILITY}_years": _topk_years,
                                        "max_consecutive_active_years": _max_streak,
                                    }
                                )
                            _rank_df = pd.DataFrame(_rank_rows)
                            if not _rank_df.empty:
                                _rank_df = _rank_df.sort_values(
                                    ["mean_share_active_%", "stability_cv_active", "active_years"],
                                    ascending=[False, True, False],
                                ).reset_index(drop=True)
                            st.markdown("### Concept ranking (visas intervalas) + stabilumas (active-years only)")
                            st.caption(
                                "mean_share_active_% ir stabilumas skaičiuojami tik metais, kur concept hits > 0; share = concept_hits / year_total_hits."
                            )
                            if _rank_df.empty:
                                st.info("Nėra konceptų su hitais.")
                            else:
                                st.dataframe(_rank_df, width="stretch")

                            # 3: Top concepts per year
                            _top_long: List[Dict] = []
                            for _yy in year_ints:
                                _yii = int(_yy)
                                _toty = year_total_hits.get(_yii, 0)
                                if _toty <= 0:
                                    continue
                                _clist = [
                                    (_cc, concept_year_count.get((_cc, _yii), 0))
                                    for _cc in seen_concepts
                                ]
                                _clist.sort(key=lambda x: -x[1])
                                _clist = [(_cc, _cnt) for _cc, _cnt in _clist if _cnt > 0][:N_TOP_CONCEPTS_PER_ROW]
                                for _rk, (_cc, _cnt) in enumerate(_clist, start=1):
                                    _top_long.append(
                                        {
                                            "Year": _yii,
                                            "Rank": _rk,
                                            "Concept": _cc,
                                            "Share (%)": round(_cnt / _toty * 100.0, 4),
                                            "Hits": _cnt,
                                        }
                                    )
                            st.markdown("### Top concepts per year (pagal share)")
                            if not _top_long:
                                st.info("Nėra duomenų top konceptams.")
                            else:
                                st.dataframe(pd.DataFrame(_top_long), width="stretch")

                            # 4: Dominant per year (pilna lentelė, ne periodų sujungimas)
                            _dom_rows: List[Dict] = []
                            for _yy in year_ints:
                                _yii = int(_yy)
                                _toty = year_total_hits.get(_yii, 0)
                                if _toty <= 0:
                                    continue
                                _best_sh = -1.0
                                _leaders: List[Tuple[str, int, float]] = []
                                for _cc in seen_concepts:
                                    _cnt = concept_year_count.get((_cc, _yii), 0)
                                    if _cnt <= 0:
                                        continue
                                    _sh = _cnt / _toty * 100.0
                                    if _sh > _best_sh + 1e-12:
                                        _best_sh = _sh
                                        _leaders = [(_cc, _cnt, _sh)]
                                    elif abs(_sh - _best_sh) <= 1e-12:
                                        _leaders.append((_cc, _cnt, _sh))
                                _dom_rows.append(
                                    {
                                        "Year": _yii,
                                        "Dominant_concepts": ", ".join(x[0] for x in _leaders) if _leaders else "",
                                        "Dominant_share_%": round(_best_sh, 4) if _leaders else 0.0,
                                        "Dominant_hits": _leaders[0][1] if _leaders else 0,
                                        "Tie_count": len(_leaders),
                                    }
                                )
                            st.markdown("### Dominant concept per year (pilna lentelė)")
                            st.dataframe(pd.DataFrame(_dom_rows), width="stretch")

                        else:
                            # Vienas kalendorinis metas: periodai YYYY-MM
                            st.markdown("### Coverage pagal periodą (vieni metai kalendoriuje)")
                            _period_chars: Dict[str, int] = defaultdict(int)
                            _period_docs: Dict[str, int] = defaultdict(int)
                            for _, _row in usable.iterrows():
                                _fn = _row["filename"]
                                _yi = int(_row["year_int"])
                                _ys, _mo, _ = parse_year_month_day_from_filename(_fn)
                                _per = f"{_ys}-{_mo}" if (_mo and _mo != "??") else str(_yi)
                                _period_chars[_per] += int(_row["chars"])
                                _period_docs[_per] += 1
                            _cov_p = []
                            for _p in periods_sorted:
                                _sum_c = _period_chars.get(_p, 0)
                                _nd = _period_docs.get(_p, 0)
                                _th = period_total_hits.get(_p, 0)
                                _cov_p.append(
                                    {
                                        "period": _p,
                                        "n_docs": _nd,
                                        "sum_chars": _sum_c,
                                        "total_hits_dict": _th,
                                        "total_hits_per_10k": round(normalize_per_10k_chars(_th, _sum_c), 4),
                                        "share_sum_check_%": round(100.0 if _th > 0 else 0.0, 2),
                                    }
                                )
                            if _cov_p:
                                st.dataframe(pd.DataFrame(_cov_p), width="stretch")
                            else:
                                st.info("Nėra periodų suvestinės.")

                            _rank_p: List[Dict] = []
                            for _c in sorted(seen_concepts):
                                _sum_h = sum(concept_period_count.get((_c, _pp), 0) for _pp in periods_sorted)
                                if _sum_h <= 0:
                                    continue
                                _active_sh: List[float] = []
                                for _pp in periods_sorted:
                                    _cnt = concept_period_count.get((_c, _pp), 0)
                                    _totp = period_total_hits.get(_pp, 0)
                                    if _cnt > 0 and _totp > 0:
                                        _active_sh.append(_cnt / _totp * 100.0)
                                _ay = len(_active_sh)
                                if _ay == 0:
                                    continue
                                _mean_sa = sum(_active_sh) / _ay
                                _std_sa = float(pd.Series(_active_sh).std(ddof=0)) if _ay >= 2 else 0.0
                                _cv_sa = (_std_sa / _mean_sa) if _mean_sa > 1e-12 else 0.0
                                _topk_p = 0
                                for _pp in periods_sorted:
                                    _totp = period_total_hits.get(_pp, 0)
                                    if _totp <= 0:
                                        continue
                                    _pairs = [
                                        (_cc, concept_period_count.get((_cc, _pp), 0) / _totp * 100.0)
                                        for _cc in seen_concepts
                                    ]
                                    _pairs.sort(key=lambda x: -x[1])
                                    _leads = [_cc for _cc, _sh in _pairs[:TOP_K_FOR_STABILITY] if _sh > 0]
                                    if _c in _leads:
                                        _topk_p += 1
                                _rank_p.append(
                                    {
                                        "Concept": _c,
                                        "sum_hits": _sum_h,
                                        "active_periods": _ay,
                                        "mean_share_active_%": round(_mean_sa, 4),
                                        "stability_std_active_pp": round(_std_sa, 4),
                                        "stability_cv_active": round(_cv_sa, 4),
                                        f"top{TOP_K_FOR_STABILITY}_periods": _topk_p,
                                    }
                                )
                            _rank_pdf = pd.DataFrame(_rank_p)
                            if not _rank_pdf.empty:
                                _rank_pdf = _rank_pdf.sort_values(
                                    ["mean_share_active_%", "stability_cv_active", "active_periods"],
                                    ascending=[False, True, False],
                                ).reset_index(drop=True)
                            st.markdown("### Concept ranking (visas intervalas) + stabilumas — pagal periodą")
                            if _rank_pdf.empty:
                                st.info("Nėra konceptų su hitais.")
                            else:
                                st.dataframe(_rank_pdf, width="stretch")

                            _top_pl: List[Dict] = []
                            for _pp in periods_sorted:
                                _totp = period_total_hits.get(_pp, 0)
                                if _totp <= 0:
                                    continue
                                _clist = [
                                    (_cc, concept_period_count.get((_cc, _pp), 0)) for _cc in seen_concepts
                                ]
                                _clist.sort(key=lambda x: -x[1])
                                _clist = [(_cc, _cnt) for _cc, _cnt in _clist if _cnt > 0][:N_TOP_CONCEPTS_PER_ROW]
                                for _rk, (_cc, _cnt) in enumerate(_clist, start=1):
                                    _top_pl.append(
                                        {
                                            "Period": _pp,
                                            "Rank": _rk,
                                            "Concept": _cc,
                                            "Share (%)": round(_cnt / _totp * 100.0, 4),
                                            "Hits": _cnt,
                                        }
                                    )
                            st.markdown("### Top concepts per periodą (pagal share)")
                            if not _top_pl:
                                st.info("Nėra duomenų.")
                            else:
                                st.dataframe(pd.DataFrame(_top_pl), width="stretch")

                            _dom_p: List[Dict] = []
                            for _pp in periods_sorted:
                                _totp = period_total_hits.get(_pp, 0)
                                if _totp <= 0:
                                    continue
                                _best_sh = -1.0
                                _leaders = []
                                for _cc in seen_concepts:
                                    _cnt = concept_period_count.get((_cc, _pp), 0)
                                    if _cnt <= 0:
                                        continue
                                    _sh = _cnt / _totp * 100.0
                                    if _sh > _best_sh + 1e-12:
                                        _best_sh = _sh
                                        _leaders = [(_cc, _cnt, _sh)]
                                    elif abs(_sh - _best_sh) <= 1e-12:
                                        _leaders.append((_cc, _cnt, _sh))
                                _dom_p.append(
                                    {
                                        "Period": _pp,
                                        "Dominant_concepts": ", ".join(x[0] for x in _leaders) if _leaders else "",
                                        "Dominant_share_%": round(_best_sh, 4) if _leaders else 0.0,
                                        "Dominant_hits": _leaders[0][1] if _leaders else 0,
                                        "Tie_count": len(_leaders),
                                    }
                                )
                            st.markdown("### Dominant concept per periodą (pilna lentelė)")
                            st.dataframe(pd.DataFrame(_dom_p), width="stretch")

                        # 5–6: Diskurse nevartotini (absolute 0) — po coverage/ranking/top/dominant
                        if not term_hits_all_docs.empty:
                            _tsum = term_hits_all_docs.groupby("CH term", as_index=False)["Count"].sum()
                            _detected_terms = set(_tsum.loc[_tsum["Count"] > 0, "CH term"].astype(str))
                            _csum = term_hits_all_docs.groupby("Concept", as_index=False)["Count"].sum()
                            _detected_concepts = set(_csum.loc[_csum["Count"] > 0, "Concept"].astype(str))
                        else:
                            _detected_terms = set()
                            _detected_concepts = set()
                        _all_terms = set(terms_df["term"].astype(str).unique())
                        _all_concepts = set(terms_df["concept"].astype(str).unique())
                        _miss_terms = sorted(_all_terms - _detected_terms)
                        _miss_concepts = sorted(_all_concepts - _detected_concepts)

                        st.markdown("### Diskurse nevartotini terminai (absolute 0, visas intervalas)")
                        if _miss_terms:
                            _zt = terms_df[terms_df["term"].astype(str).isin(_miss_terms)].copy()
                            _zt = _zt.rename(
                                columns={
                                    "term": "CH term",
                                    "concept": "Concept",
                                    "category": "Category",
                                    "pinyin": "Pinyin",
                                    "translation": "ENG translation",
                                }
                            )
                            st.dataframe(
                                _zt[["CH term", "Concept", "Category", "Pinyin", "ENG translation"]].sort_values(
                                    ["Category", "Concept", "CH term"]
                                ),
                                width="stretch",
                            )
                            st.caption(f"Termų skaičius: {len(_miss_terms)} (iš žodyno: {len(_all_terms)}).")
                        else:
                            st.info("Visi žodyno terminai turi bent vieną hitą intervale.")

                        st.markdown("### Diskurse nevartotini konceptai (absolute 0, visas intervalas)")
                        if _miss_concepts:
                            _zc = (
                                terms_df[terms_df["concept"].astype(str).isin(_miss_concepts)]
                                .groupby("concept", as_index=False)
                                .agg(
                                    Categories=("category", lambda s: ", ".join(sorted(set(s.astype(str))))),
                                    terms_in_dict=("term", "count"),
                                )
                                .rename(columns={"concept": "Concept"})
                                .sort_values("Concept")
                            )
                            st.dataframe(_zc, width="stretch")
                            st.caption(f"Konceptų skaičius: {len(_miss_concepts)} (iš žodyno: {len(_all_concepts)}).")
                        else:
                            st.info("Visi žodyno konceptai turi bent vieną hitą intervale.")

                        st.divider()
                        st.markdown("### Dokumentų suvestinė")
                        st.dataframe(
                            usable.sort_values("year_int")[["year", "title_cn", "filename", "chars", "total_hits", "total_hits_per_10k_chars"]],
                            width="stretch",
                        )

                        with st.expander(
                            "Economist watchlist (EN) — yearly dictionary hit share, %",
                            expanded=False,
                        ):
                            st.caption(
                                "One mini chart per term (English title). "
                                "Bars = that term's summed hits ÷ all dictionary hits in the corpus for that year × 100."
                            )
                            _watch = load_economist_watchlist()
                            if _watch:
                                render_watchlist_year_share_mini_charts(
                                    term_hits_all_docs, _watch, bar_color="#e64c3c"
                                )
                            else:
                                st.info("economist_terms.txt nerastas arba tuščias.")

                        with st.expander(
                            "Tanming terms (EN) — yearly dictionary hit share, %",
                            expanded=False,
                        ):
                            st.caption(
                                "Same metric as Economist block; source: tanming_terms.txt. "
                                "Gold bars (imperial yellow–gold palette)."
                            )
                            _tan_watch = load_economist_watchlist("tanming_terms.txt")
                            if _tan_watch:
                                render_watchlist_year_share_mini_charts(
                                    term_hits_all_docs,
                                    _tan_watch,
                                    bar_color="#c9a227",
                                )
                            else:
                                st.info("tanming_terms.txt nerastas arba tuščias.")

                        # -----------------------------
                        # Full run export (ZIP)
                        # -----------------------------
                        st.subheader("Export full run")

                        if not term_hits_all_docs.empty:
                            import io
                            import zipfile
                            from datetime import datetime

                            buf = io.BytesIO()
                            with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                                # 1) metadata
                                meta = {
                                    "run_label": run_label,
                                    "created_at_utc": datetime.utcnow().isoformat() + "Z",
                                    "n_documents": int(len(docs_overview)),
                                    "dictionary_rows": int(len(terms_df)),
                                    "dictionary_unique_terms": int(terms_df["term"].nunique()),
                                    "match_mode": safe_str(match_mode),
                                }
                                meta_df = pd.DataFrame([meta])
                                zf.writestr("metadata.csv", meta_df.to_csv(index=False, encoding="utf-8-sig"))

                                # 2) documents overview
                                zf.writestr("docs_overview.csv", docs_overview_export.to_csv(index=False, encoding="utf-8-sig"))

                                # 3) all term hits (long form)
                                zf.writestr(
                                    "term_hits_all_docs.csv",
                                    term_hits_all_docs.to_csv(index=False, encoding="utf-8-sig"),
                                )

                                # 4) example aggregated timeline by category (sum of counts per year & category)
                                if not term_hits_all_docs.empty:
                                    agg_cat = (
                                        term_hits_all_docs.groupby(["year", "Category"], as_index=False)["Count"]
                                        .sum()
                                        .rename(columns={"Count": "Total_count"})
                                    )
                                    zf.writestr(
                                        "timeline_by_category.csv",
                                        agg_cat.to_csv(index=False, encoding="utf-8-sig"),
                                    )

                            buf.seek(0)
                            zip_filename = f"{run_label or 'analysis_run'}_set{set_id}.zip".replace(" ", "_")

                            st.download_button(
                                f"Download full run Set {set_id} (ZIP)",
                                data=buf,
                                file_name=zip_filename,
                                mime="application/zip",
                                key=f"download_run_set{set_id}",
                            )
                        else:
                            st.info("Nėra termų hitų šiame run'e, todėl pilnas eksportas neaktyvus.")

    # -----------------------------
    # Per-document view (single tab + selector)
    # -----------------------------
    for year, tab in zip(years, tabs[1:]):
        with tab:
            st.subheader("Document info")
    
            year_docs = [di for di in doc_infos if (di["year"] or "Unknown") == year]
    
            # Build labels: YYYY-MM-DD + A/B/C if multiple on same exact day
            day_key_counts: Dict[str, int] = {}
            for di in year_docs:
                y = di["year"] or "Unknown"
                mo = di["month"] or "??"
                da = di["day"] or "??"
                key = f"{y}-{mo}-{da}"
                day_key_counts[key] = day_key_counts.get(key, 0) + 1
    
            day_key_seen: Dict[str, int] = {}
            options = []
            for di in year_docs:
                y = di["year"] or "Unknown"
                mo = di["month"] or "??"
                da = di["day"] or "??"
                key = f"{y}-{mo}-{da}"
    
                day_key_seen[key] = day_key_seen.get(key, 0) + 1
                suffix = ""
                if day_key_counts.get(key, 0) > 1:
                    suffix = month_tab_suffix(day_key_seen[key])
                label = f"{key}{suffix}"
                options.append((label, di["idx"]))
    
            # Sort options by date-like label; Unknown year stays as-is
            options_sorted = sorted(options, key=lambda t: t[0])
            labels = [t[0] for t in options_sorted]
            idx_by_label = {t[0]: t[1] for t in options_sorted}
    
            selected_label = st.selectbox(
                "Pasirink dokumentą", options=labels, index=0,
                key=f"doc_selector_set{set_id}_{year}",
            )
            selected_idx = idx_by_label[selected_label]
            filename, text = docs[selected_idx]
    
            col1, col2 = st.columns([1, 4])
            with col1:
                st.write(f"**Metai (Year):** {st.session_state[_mk(filename)]['year']}")
            with col2:
                st.write(f"**Pavadinimas (CN):** {st.session_state[_mk(filename)]['title_cn']}")
    
            with st.expander("Teksto statistika", expanded=False):
                st.write(
                    {
                        "Characters": len(text),
                        "Lines": text.count("\n") + 1 if text else 0,
                    }
                )

            with st.expander("Kaip suskaidytas tekstas", expanded=False):
                _term_hits = doc_term_hits[filename]
                segment_view = (
                    (match_mode in ("jieba_precise", "jieba_search") and JIEBA_AVAILABLE)
                    or (match_mode == "thulac" and THULAC_AVAILABLE)
                    or (match_mode == "hanlp" and HANLP_AVAILABLE)
                )
                if segment_view:
                    body_html = build_segmented_highlighted_html(text, match_mode, _term_hits)
                    st.caption("Originalus išdėstymas, tokenų ribos | , žodyno terminai nuspalvinti. Užvesk pelę ant termino – tooltip.")
                    wrapper = f'<div style="max-height: 400px; overflow-y: auto; padding: 0.5rem 0; font-family: inherit; white-space: pre-wrap;">{body_html}</div>'
                    st.markdown(wrapper, unsafe_allow_html=True)
                else:
                    if _term_hits.empty:
                        st.info("Nėra rastų terminų šiame dokumente.")
                    else:
                        term_to_meta: Dict[str, Tuple[str, str]] = {}
                        for _, r in terms_df.iterrows():
                            t = safe_str(r["term"])
                            if t and t not in term_to_meta:
                                term_to_meta[t] = (safe_str(r["pinyin"]), safe_str(r["translation"]))
                        if match_mode in ("substring", "hanlp", "thulac"):
                            terms_list = [safe_str(r["term"]) for _, r in terms_df.iterrows() if safe_str(r["term"])]
                        else:
                            terms_list = [safe_str(r["term"]) for _, r in terms_df.iterrows() if len(safe_str(r["term"])) >= 2]
                        spans = get_longest_match_positions(text, terms_list, term_to_meta)
                        if not spans:
                            st.info("Nėra rastų terminų šiame dokumente.")
                        else:
                            body_html = build_highlight_html(text, spans)
                            wrapper = f'<div style="max-height: 400px; overflow-y: auto; padding: 0.5rem 0; font-family: inherit; white-space: pre-wrap;">{body_html}</div>'
                            st.markdown(wrapper, unsafe_allow_html=True)
    
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
                        key=f"download_term_set{set_id}_{year}_{selected_idx}",
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
                        key=f"download_summaries_set{set_id}_{year}_{selected_idx}",
                    )


# -----------------------------
# Two-column driver: Set 1 | Set 2
# -----------------------------
def _upload_signature(terms_upload, files) -> Tuple:
    t_sig = None
    if terms_upload is not None:
        t_sig = (safe_str(getattr(terms_upload, "name", "")), int(getattr(terms_upload, "size", 0) or 0))
    f_sig = []
    for f in (files or []):
        f_sig.append((safe_str(getattr(f, "name", "")), int(getattr(f, "size", 0) or 0)))
    return (t_sig, tuple(sorted(f_sig)))


with col_set1:
    st.subheader("Set 1")
    st.caption("CSV: concept;term;pinyin;translation;category")
    terms_upload_1 = st.file_uploader("Žodynas Set 1 (nebūtina – naudos default terms_cn.csv)", type=["csv"], key="terms_set1")
    files_1 = st.file_uploader("Dokumentai Set 1 (TXT / DOCX)", type=["txt", "docx", "doc"], accept_multiple_files=True, key="files_set1")
    sig1 = _upload_signature(terms_upload_1, files_1)
    if st.session_state.get("_sig_set1") != sig1:
        st.session_state["_sig_set1"] = sig1
        st.session_state.pop("_data_set1", None)
        st.session_state.pop("_data_set1_modes", None)
        st.session_state["_aggregate_view_set1"] = False
        st.session_state.pop("_aggregate_cache_info_set1", None)

    run1 = st.button("Run Set 1", key="run_set1", disabled=not bool(files_1))
    run1_agg = st.button(
        "Run Set 1 (Aggregate all modes)",
        key="run_set1_agg",
        disabled=not bool(files_1),
        help="Paleidžia pagrindinį režimą + visus papildomus režimus su progreso indikatoriumi.",
    )
    clear1 = st.button("Clear Set 1 results", key="clear_set1", disabled=not bool(st.session_state.get("_data_set1")))
    if clear1:
        st.session_state.pop("_data_set1", None)
        st.session_state.pop("_data_set1_modes", None)
        st.session_state["_aggregate_view_set1"] = False
        st.session_state.pop("_aggregate_cache_info_set1", None)

    if run1 or run1_agg:
        try:
            aggregate_run_active = bool(run1_agg)
            data1_main = run_pipeline_for_set(terms_upload_1, files_1, match_mode, run_label, 1)
            if data1_main is None:
                raise RuntimeError(
                    "Nepavyko apdoroti Set 1 dokumentų (visi failai neįskaitomi arba neaptikta tekstinių dokumentų)."
                )
            st.session_state["_data_set1"] = data1_main

            if aggregate_run_active:
                agg_modes = get_effective_aggregate_modes(
                    list(mode_labels.keys()),
                    data1_main["terms_df"],
                    preferred_mode=match_mode,
                )
                data1_modes: Dict[str, Dict] = {}
                cached_modes_prev = st.session_state.get("_data_set1_modes") or {}
                cache_hits = 0
                total_modes = max(1, len(agg_modes))
                progress = st.progress(0, text=f"Aggregate Set 1: 0/{total_modes} režimų.")
                start_ts = time.time()
                done = 0
                for m in agg_modes:
                    if m == match_mode:
                        data1_modes[m] = data1_main
                    elif m in cached_modes_prev and cached_modes_prev.get(m):
                        data1_modes[m] = cached_modes_prev[m]
                        cache_hits += 1
                    else:
                        with st.spinner(f"Set 1 aggregate: vykdomas režimas `{m}`..."):
                            data1_modes[m] = run_pipeline_for_set(terms_upload_1, files_1, m, run_label, 1)
                    done += 1
                    elapsed = max(0.0, time.time() - start_ts)
                    avg = elapsed / done if done > 0 else 0.0
                    eta = max(0.0, avg * (total_modes - done))
                    progress.progress(
                        int(done * 100 / total_modes),
                        text=f"Aggregate Set 1: {done}/{total_modes} režimų | elapsed {elapsed:.1f}s | ETA ~{eta:.1f}s",
                    )
                st.session_state["_data_set1_modes"] = data1_modes
                st.session_state["_aggregate_view_set1"] = True
                st.session_state["_aggregate_cache_info_set1"] = {
                    "used": bool(cache_hits > 0),
                    "hits": int(cache_hits),
                    "total": int(total_modes),
                }
            else:
                st.session_state.pop("_data_set1_modes", None)
                st.session_state["_aggregate_view_set1"] = False
                st.session_state.pop("_aggregate_cache_info_set1", None)
        except Exception as e:
            st.session_state.pop("_data_set1", None)
            st.session_state.pop("_data_set1_modes", None)
            st.error(f"Set 1 klaida: {e}")

    data1 = st.session_state.get("_data_set1")
    if data1:
        st.success(f"Žodynas: {len(data1['terms_df']):,} raktažodžių. Dokumentų: {len(data1['docs']):,}.")
        if st.session_state.get("_aggregate_view_set1") and st.session_state.get("_data_set1_modes"):
            cache_info_1 = st.session_state.get("_aggregate_cache_info_set1") or {}
            if cache_info_1.get("used"):
                st.caption(
                    f"Aggregate cache (Set 1): panaudota {cache_info_1.get('hits', 0)}/{cache_info_1.get('total', 0)} režimų iš ankstesnio paleidimo."
                )
            else:
                st.caption("Aggregate cache (Set 1): cache nepanaudotas (visi režimai perskaičiuoti).")
        _render_set_ui(data1, 1, match_mode, run_label)
    else:
        st.info("Set 1: įkelk dokumentus ir spausk „Run Set 1“.")

with col_set2:
    st.subheader("Set 2")
    st.caption("CSV: concept;term;pinyin;translation;category")
    terms_upload_2 = st.file_uploader("Žodynas Set 2 (būtina)", type=["csv"], key="terms_set2")
    files_2 = st.file_uploader("Dokumentai Set 2 (TXT / DOCX)", type=["txt", "docx", "doc"], accept_multiple_files=True, key="files_set2")
    sig2 = _upload_signature(terms_upload_2, files_2)
    if st.session_state.get("_sig_set2") != sig2:
        st.session_state["_sig_set2"] = sig2
        st.session_state.pop("_data_set2", None)
        st.session_state.pop("_data_set2_modes", None)
        st.session_state["_aggregate_view_set2"] = False
        st.session_state.pop("_aggregate_cache_info_set2", None)

    run2 = st.button("Run Set 2", key="run_set2", disabled=not (bool(files_2) and bool(terms_upload_2)))
    run2_agg = st.button(
        "Run Set 2 (Aggregate all modes)",
        key="run_set2_agg",
        disabled=not (bool(files_2) and bool(terms_upload_2)),
        help="Paleidžia pagrindinį režimą + visus papildomus režimus su progreso indikatoriumi.",
    )
    clear2 = st.button("Clear Set 2 results", key="clear_set2", disabled=not bool(st.session_state.get("_data_set2")))
    if clear2:
        st.session_state.pop("_data_set2", None)
        st.session_state.pop("_data_set2_modes", None)
        st.session_state["_aggregate_view_set2"] = False
        st.session_state.pop("_aggregate_cache_info_set2", None)

    if run2 or run2_agg:
        try:
            aggregate_run_active = bool(run2_agg)
            data2_main = run_pipeline_for_set(terms_upload_2, files_2, match_mode, run_label, 2)
            if data2_main is None:
                raise RuntimeError(
                    "Nepavyko apdoroti Set 2 dokumentų (visi failai neįskaitomi arba neaptikta tekstinių dokumentų)."
                )
            st.session_state["_data_set2"] = data2_main

            if aggregate_run_active:
                agg_modes = get_effective_aggregate_modes(
                    list(mode_labels.keys()),
                    data2_main["terms_df"],
                    preferred_mode=match_mode,
                )
                data2_modes: Dict[str, Dict] = {}
                cached_modes_prev = st.session_state.get("_data_set2_modes") or {}
                cache_hits = 0
                total_modes = max(1, len(agg_modes))
                progress = st.progress(0, text=f"Aggregate Set 2: 0/{total_modes} režimų.")
                start_ts = time.time()
                done = 0
                for m in agg_modes:
                    if m == match_mode:
                        data2_modes[m] = data2_main
                    elif m in cached_modes_prev and cached_modes_prev.get(m):
                        data2_modes[m] = cached_modes_prev[m]
                        cache_hits += 1
                    else:
                        with st.spinner(f"Set 2 aggregate: vykdomas režimas `{m}`..."):
                            data2_modes[m] = run_pipeline_for_set(terms_upload_2, files_2, m, run_label, 2)
                    done += 1
                    elapsed = max(0.0, time.time() - start_ts)
                    avg = elapsed / done if done > 0 else 0.0
                    eta = max(0.0, avg * (total_modes - done))
                    progress.progress(
                        int(done * 100 / total_modes),
                        text=f"Aggregate Set 2: {done}/{total_modes} režimų | elapsed {elapsed:.1f}s | ETA ~{eta:.1f}s",
                    )
                st.session_state["_data_set2_modes"] = data2_modes
                st.session_state["_aggregate_view_set2"] = True
                st.session_state["_aggregate_cache_info_set2"] = {
                    "used": bool(cache_hits > 0),
                    "hits": int(cache_hits),
                    "total": int(total_modes),
                }
                st.rerun()
            else:
                st.session_state.pop("_data_set2_modes", None)
                st.session_state["_aggregate_view_set2"] = False
                st.session_state.pop("_aggregate_cache_info_set2", None)
        except Exception as e:
            st.session_state.pop("_data_set2", None)
            st.session_state.pop("_data_set2_modes", None)
            st.error(f"Set 2 klaida: {e}")

    data2 = st.session_state.get("_data_set2")
    if data2:
        st.success(f"Žodynas: {len(data2['terms_df']):,} raktažodžių. Dokumentų: {len(data2['docs']):,}.")
        if st.session_state.get("_aggregate_view_set2") and st.session_state.get("_data_set2_modes"):
            cache_info_2 = st.session_state.get("_aggregate_cache_info_set2") or {}
            if cache_info_2.get("used"):
                st.caption(
                    f"Aggregate cache (Set 2): panaudota {cache_info_2.get('hits', 0)}/{cache_info_2.get('total', 0)} režimų iš ankstesnio paleidimo."
                )
            else:
                st.caption("Aggregate cache (Set 2): cache nepanaudotas (visi režimai perskaičiuoti).")
        _render_set_ui(data2, 2, match_mode, run_label)
    else:
        st.info("Set 2: įkelk žodyną + dokumentus ir spausk „Run Set 2“.")

# -----------------------------
# Compare common tables (after both sets render/exporst)
# -----------------------------
if st.session_state.get("_compare_requested") and st.session_state.get("_data_set1") and st.session_state.get("_data_set2"):
    data1_cmp = st.session_state.get("_data_set1")
    data2_cmp = st.session_state.get("_data_set2")

    st.divider()
    st.subheader("Compare: Common Top 15 (Concepts + Terms)")

    conc_df, term_df = compute_common_top15_concepts_terms(data1_cmp, data2_cmp, top_n=15)

    with st.container(border=True):
        st.caption("Common Top 15 Concepts (yellow background)")
        if conc_df is None or conc_df.empty:
            st.info("Nerasta pakankamai bendrų Concept elementų (Share > 0 abiejuose setuose).")
        else:
            st.dataframe(_yellow_styler(conc_df, color="#fff2a8"), width="stretch")

        st.caption("Common Top 15 Terms (yellow background)")
        if term_df is None or term_df.empty:
            st.info("Nerasta pakankamai bendrų Terms elementų (Share > 0 abiejuose setuose).")
        else:
            st.dataframe(_yellow_styler(term_df, color="#fff2a8"), width="stretch")

    st.subheader("Compare: Tianming terms (Set 1 vs Set 2)")
    tan_watch_cmp = load_economist_watchlist("tanming_terms.txt")
    if not tan_watch_cmp:
        st.info("tanming_terms.txt nerastas arba tuščias.")
    else:
        tan_cmp_df = compute_watchlist_set_compare(data1_cmp, data2_cmp, tan_watch_cmp, year_from=2017, year_to=2025)
        st.caption("Žalia spalva rodo setą, kuriame terminas vidutiniškai dažniau naudojamas dinamikoje.")
        st.dataframe(_watchlist_compare_styler(tan_cmp_df), width="stretch", hide_index=True)

        rows_to_show = tan_cmp_df[
            (tan_cmp_df["Set1 mean Share (%)"] > 0) | (tan_cmp_df["Set2 mean Share (%)"] > 0)
        ].copy()
        if rows_to_show.empty:
            st.info("Nėra Tianming terminų su nenuline reikšme nei Set 1, nei Set 2.")
        else:
            st.caption("Rodomi visi terminai su nenuline reikšme; žalias rėmelis pažymi laimintį setą.")
            rows_records = rows_to_show.to_dict("records")
            for i in range(0, len(rows_records), 2):
                pair = rows_records[i : i + 2]
                cols = st.columns(4)
                for j, selected_row in enumerate(pair):
                    sel_ch = safe_str(selected_row["CH term"])
                    sel_en = safe_str(selected_row["Term (EN)"])
                    winner = safe_str(selected_row["Winner"])
                    years_s1, vals_s1 = _share_series_for_term(
                        data1_cmp.get("term_hits_all_docs"), sel_ch, year_from=2017, year_to=2025
                    )
                    years_s2, vals_s2 = _share_series_for_term(
                        data2_cmp.get("term_hits_all_docs"), sel_ch, year_from=2017, year_to=2025
                    )
                    left_idx = 0 if j == 0 else 2
                    right_idx = left_idx + 1
                    with cols[left_idx]:
                        st.caption(f"{sel_en} — Set 1")
                        border_s1 = "#22c55e" if winner == "Set 1" else "#bfc5cf"
                        fig_s1 = build_economist_share_mini_chart_figure(
                            f"{sel_en} — Set 1",
                            years_s1,
                            vals_s1,
                            bar_color="#c9a227",
                            border_color=border_s1,
                            border_width=2.2 if winner == "Set 1" else 1.0,
                        )
                        if fig_s1 is not None:
                            st.pyplot(fig_s1, width="stretch")
                            plt.close(fig_s1)
                    with cols[right_idx]:
                        st.caption(f"{sel_en} — Set 2")
                        border_s2 = "#22c55e" if winner == "Set 2" else "#bfc5cf"
                        fig_s2 = build_economist_share_mini_chart_figure(
                            f"{sel_en} — Set 2",
                            years_s2,
                            vals_s2,
                            bar_color="#c9a227",
                            border_color=border_s2,
                            border_width=2.2 if winner == "Set 2" else 1.0,
                        )
                        if fig_s2 is not None:
                            st.pyplot(fig_s2, width="stretch")
                            plt.close(fig_s2)
                st.divider()
