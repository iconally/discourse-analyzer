# -*- coding: utf-8 -*-
"""
Offline Economist-style mini charts (same metric as app.py expander).

Builds term_hits_all_docs from a folder of .txt (optional .docx), full civilization CSV,
economist_terms.txt watchlist, yearly share (%) same as Streamlit Economist block.

Default rules (2017+):
  - Skip files if filename year is missing or < 2017.
  - Skip files if any **folder** name on the path contains a 4-digit year < 2017
    (e.g. ``jhsjk_FULL_2013`` under ``D:\\Xi`` is ignored).
  - Chart columns: ``chart_first_year`` (2017) through ``max(chart_last_year, latest data year)``.

Use ``--fixed-years --year-from 2017 --year-to 2025`` for a fixed axis only.
  - economist_year_share.csv
  - economist_charts/*.png  (one mini chart per watchlist row)

Does not modify app.py.

Example (2017+ .txt only, x-axis 2017..max(2025, latest year in corpus)):
  python batch_economist_charts.py --docs-dir "D:\\Xi" --terms civilization_terms_UPDATED_full.csv --out-dir economist_batch_out --recursive

Whole D: drive (slow; many non-text folders):
  python batch_economist_charts.py --docs-dir "D:\\" --recursive ...
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False
    plt = None  # type: ignore

from batch_analyze_hanlp import analyze_text, iter_doc_paths, load_terms_csv_path, read_local_file, rel_under_root


def safe_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    return str(x)


def parse_meta_from_filename(filename: str) -> Dict[str, str]:
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


def safe_year(y: str) -> Optional[int]:
    y = safe_str(y).strip()
    if not y:
        return None
    if re.fullmatch(r"(19|20)\d{2}", y):
        return int(y)
    return None


_YEAR_TOKEN = re.compile(r"(?<!\d)(19|20)\d{2}(?!\d)")


def folder_path_has_year_before(rel: str, min_year: int) -> bool:
    """
    True if any directory segment (not the filename) contains a 4-digit year
    strictly before min_year. Used to skip e.g. .../jhsjk_FULL_2013/... when min_year=2017.
    """
    p = Path(rel)
    parts = p.parts
    if len(parts) < 2:
        return False
    for seg in parts[:-1]:
        for m in _YEAR_TOKEN.finditer(seg):
            if int(m.group(0)) < int(min_year):
                return True
    return False


def load_economist_watchlist(path: Path) -> List[Tuple[str, str]]:
    if not path.is_file():
        return []
    out: List[Tuple[str, str]] = []
    seen_ch: set = set()
    raw = path.read_text(encoding="utf-8")
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(";")
        if len(parts) < 4:
            continue
        ch = safe_str(parts[1]).strip()
        eng = safe_str(parts[3]).strip()
        if not ch or ch in seen_ch:
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
    year_cols = [str(y) for y in range(year_from, year_to + 1)]
    if not watchlist:
        return pd.DataFrame(columns=["Term (EN)"] + year_cols)
    if term_hits_all_docs is None or term_hits_all_docs.empty:
        return pd.DataFrame([{"Term (EN)": eng, **{c: 0.0 for c in year_cols}} for _ch, eng in watchlist])

    th = term_hits_all_docs.copy()
    th["year_int"] = th["year"].map(safe_year)
    th = th.dropna(subset=["year_int"])
    if th.empty:
        return pd.DataFrame([{"Term (EN)": eng, **{c: 0.0 for c in year_cols}} for _ch, eng in watchlist])
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
):
    if not MATPLOTLIB_AVAILABLE or plt is None:
        return None
    if len(years) != len(pct_values):
        raise ValueError("years and pct_values length mismatch")
    bg = "#f4f1ea"
    n_years = len(years)
    fig_w = max(3.0, min(9.0, 0.48 * n_years))
    fig_h = 2.05 if n_years >= 7 else 1.85
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=120, facecolor=bg)
    ax.set_facecolor(bg)
    x = list(range(len(years)))
    bar_w = min(0.72, max(0.35, 0.82 / max(n_years, 1)))
    ax.bar(x, pct_values, color=bar_color, width=bar_w, align="center", edgecolor="none", zorder=2)
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
    y_hi = 5.0 if m <= 0 else max(m * 1.18, m + 0.5)
    ax.set_ylim(0, y_hi)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#aaaaaa")
    ax.spines["right"].set_color("#aaaaaa")
    bottom_pad = 0.30 if n_years >= 7 else 0.26
    fig.subplots_adjust(left=0.10, right=0.92, top=0.78, bottom=bottom_pad)
    return fig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs-dir", type=str, required=True)
    ap.add_argument("--terms", type=str, default="civilization_terms_UPDATED_full.csv")
    ap.add_argument(
        "--watchlist",
        type=str,
        default="economist_terms.txt",
        help="Semicolon file: concept;CH;pinyin;english;category (same as app load_economist_watchlist)",
    )
    ap.add_argument("--out-dir", type=str, default="economist_batch_out")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument(
        "--match-mode",
        default="jieba_precise",
        choices=("jieba_precise", "jieba_search", "substring", "hybrid", "hanlp"),
    )
    ap.add_argument(
        "--fixed-years",
        action="store_true",
        help="Use only --year-from/--year-to for chart columns (ignore corpus-based rules below).",
    )
    ap.add_argument("--year-from", type=int, default=2017, help="With --fixed-years: first chart column year.")
    ap.add_argument("--year-to", type=int, default=2025, help="With --fixed-years: last chart column year.")
    ap.add_argument(
        "--min-filename-year",
        type=int,
        default=2017,
        help="Skip documents whose filename has no year or year < this (default 2017).",
    )
    ap.add_argument(
        "--chart-first-year",
        type=int,
        default=2017,
        help="First year on x-axis / CSV columns (default 2017).",
    )
    ap.add_argument(
        "--chart-last-year",
        type=int,
        default=2025,
        help="Chart always includes at least through this year; extended if corpus has later hits.",
    )
    ap.add_argument(
        "--include-docx",
        action="store_true",
        help="Also process .docx (default: .txt only).",
    )
    ap.add_argument(
        "--no-folder-year-filter",
        action="store_true",
        help="Do not skip paths where a folder name contains a year < --min-filename-year.",
    )
    args = ap.parse_args()

    base = Path(__file__).resolve().parent
    root = Path(args.docs_dir)
    if not root.is_dir():
        print(f"ERROR: not a directory: {root}", file=sys.stderr)
        return 2
    terms_path = Path(args.terms)
    if not terms_path.is_absolute():
        terms_path = base / terms_path
    if not terms_path.is_file():
        print(
            f"ERROR: terms not found: {terms_path}\n"
            f"  (resolved from --terms {args.terms!r} relative to script dir {base})",
            file=sys.stderr,
        )
        return 2
    watch_path = Path(args.watchlist)
    if not watch_path.is_absolute():
        watch_path = base / watch_path

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chart_dir = out_dir / "economist_charts"
    chart_dir.mkdir(exist_ok=True)

    terms_df = load_terms_csv_path(terms_path)
    for c in ("concept", "term", "pinyin", "translation", "category"):
        terms_df[c] = terms_df[c].astype(str)

    watchlist = load_economist_watchlist(watch_path)
    if not watchlist:
        print(f"ERROR: empty or missing watchlist: {watch_path}", file=sys.stderr)
        return 3

    exts: Tuple[str, ...] = (".txt", ".docx") if args.include_docx else (".txt",)
    paths = iter_doc_paths(root, args.recursive, exts)
    if args.max_files and args.max_files > 0:
        paths = paths[: int(args.max_files)]

    print(
        f"Corpus: {root} | extensions={exts} recursive={args.recursive} | "
        f"matched_files={len(paths)}",
        flush=True,
    )
    if not paths:
        print(
            "WARN: no matching documents. If your .txt files are in subfolders, "
            "add --recursive. Only *.txt in the root of --docs-dir are scanned when "
            "recursive is off.",
            file=sys.stderr,
            flush=True,
        )

    rows_all: List[Dict] = []
    errors: List[Tuple[str, str]] = []
    skipped_year: List[str] = []
    skipped_folder_year: List[str] = []
    t0 = time.time()

    for i, p in enumerate(paths, start=1):
        rel = rel_under_root(root, p)
        if not args.no_folder_year_filter and folder_path_has_year_before(rel, int(args.min_filename_year)):
            skipped_folder_year.append(rel)
            continue
        meta = parse_meta_from_filename(p.name)
        year = meta["year"]
        yi = safe_year(year)
        if yi is None or yi < int(args.min_filename_year):
            skipped_year.append(rel)
            continue
        text, err = read_local_file(p)
        if err or text is None:
            errors.append((rel, err or "empty"))
            continue
        try:
            th = analyze_text(text, terms_df, args.match_mode)
        except Exception as e:
            errors.append((rel, str(e)))
            continue
        if th.empty:
            continue
        for _, r in th.iterrows():
            rows_all.append(
                {
                    "filename": p.name,
                    "source_path": rel,
                    "year": year,
                    "CH term": safe_str(r["CH term"]),
                    "Pinyin": safe_str(r["Pinyin"]),
                    "ENG translation": safe_str(r["ENG translation"]),
                    "Concept": safe_str(r["Concept"]),
                    "Category": safe_str(r["Category"]),
                    "Count": int(r["Count"]),
                }
            )
        if i % 50 == 0 or i == len(paths):
            print(f"  progress {i}/{len(paths)} elapsed={time.time() - t0:.1f}s", flush=True)

    term_hits_all_docs = pd.DataFrame(rows_all)
    term_hits_all_docs.to_csv(out_dir / "term_hits_all_docs.csv", sep=";", index=False, encoding="utf-8-sig")

    print(
        f"Scan summary: files_matched={len(paths)} skipped_folder_year={len(skipped_folder_year)} "
        f"skipped_filename_year={len(skipped_year)} read_errors={len(errors)} "
        f"term_hit_rows={len(term_hits_all_docs)}",
        flush=True,
    )

    if args.fixed_years:
        year_from, year_to = int(args.year_from), int(args.year_to)
    else:
        year_from = int(args.chart_first_year)
        cap = int(args.chart_last_year)
        if term_hits_all_docs.empty:
            year_to = cap
        else:
            y_int = term_hits_all_docs["year"].map(safe_year).dropna().astype(int)
            if y_int.empty:
                year_to = cap
            else:
                data_max = int(y_int.max())
                year_to = max(cap, data_max)

    if year_from > year_to:
        year_from, year_to = year_to, year_from

    print(
        f"Year columns: {year_from}..{year_to} (fixed-years={args.fixed_years}, min_filename_year={args.min_filename_year})",
        flush=True,
    )

    share_df = build_economist_year_share_table(
        term_hits_all_docs, watchlist, year_from=year_from, year_to=year_to
    )
    share_df.to_csv(out_dir / "economist_year_share.csv", sep=";", index=False, encoding="utf-8-sig")

    years = list(range(year_from, year_to + 1))
    if MATPLOTLIB_AVAILABLE and plt is not None:
        for idx, rec in enumerate(share_df.to_dict("records")):
            title = safe_str(rec.get("Term (EN)", ""))
            vals = [float(rec.get(str(y), 0.0)) for y in years]
            fig = build_economist_share_mini_chart_figure(title, years, vals, bar_color="#e64c3c")
            if fig is not None:
                safe_t = re.sub(r"[^\w\-]+", "_", title)[:80] or f"term_{idx}"
                fig.savefig(chart_dir / f"{idx:02d}_{safe_t}.png", facecolor=fig.get_facecolor())
                plt.close(fig)
    else:
        print("WARN: matplotlib not available — only CSV written.", flush=True)
        (chart_dir / "README_no_matplotlib.txt").write_text(
            "PNG charts were skipped because matplotlib is not installed or failed to import.\n"
            "Install: pip install matplotlib\n",
            encoding="utf-8",
        )

    if errors:
        pd.DataFrame(errors, columns=["file", "error"]).to_csv(
            out_dir / "read_errors.csv", sep=";", index=False, encoding="utf-8-sig"
        )
    if skipped_year:
        pd.DataFrame({"file": skipped_year}).to_csv(
            out_dir / "skipped_before_min_year.csv", sep=";", index=False, encoding="utf-8-sig"
        )
    if skipped_folder_year:
        pd.DataFrame({"file": skipped_folder_year}).to_csv(
            out_dir / "skipped_folder_year_before_cutoff.csv", sep=";", index=False, encoding="utf-8-sig"
        )

    meta = (
        f"match_mode={args.match_mode}\n"
        f"year_from={year_from}\n"
        f"year_to={year_to}\n"
        f"fixed_years={args.fixed_years}\n"
        f"min_filename_year={args.min_filename_year}\n"
        f"folder_year_filter={not args.no_folder_year_filter}\n"
        f"chart_first_year={args.chart_first_year}\n"
        f"chart_last_year={args.chart_last_year}\n"
        f"include_docx={args.include_docx}\n"
        f"paths_seen={len(paths)}\n"
        f"skipped_folder_year_before_cutoff={len(skipped_folder_year)}\n"
        f"skipped_before_min_year={len(skipped_year)}\n"
        f"watchlist_terms={len(watchlist)}\n"
        f"term_hit_rows={len(term_hits_all_docs)}\n"
        f"seconds={time.time() - t0:.1f}\n"
    )
    (out_dir / "run_meta.txt").write_text(meta, encoding="utf-8")
    print(f"Done -> {out_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
