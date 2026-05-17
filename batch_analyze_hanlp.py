# -*- coding: utf-8 -*-
"""
Batch discourse analysis (no Streamlit). Does not modify app.py.

Loads semicolon CSV (concept;term;pinyin;translation;category), scans .txt/.docx
under a folder, and mirrors app.py analyze_text() for the chosen match_mode.

Default match_mode is **jieba_precise** — usually fastest for many long .txt
files vs HanLP. Use --match-mode hanlp for the neural segmenter path.

Examples:
  python batch_analyze_hanlp.py --docs-dir "D:\\Xi" --terms civilization_terms_UPDATED_full.csv --out-dir batch_out_fast --recursive
  python batch_analyze_hanlp.py --docs-dir "D:\\Xi" --terms civilization_terms_UPDATED_full.csv --out-dir batch_out_hanlp --recursive --match-mode hanlp
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    from docx import Document

    DOCX_AVAILABLE = True
except Exception:
    DOCX_AVAILABLE = False

try:
    import jieba

    JIEBA_AVAILABLE = True
except Exception:
    JIEBA_AVAILABLE = False

HANLP_IMPORT_ERROR: Optional[str] = None
try:
    import hanlp

    HANLP_AVAILABLE = True
except Exception as e:
    HANLP_AVAILABLE = False
    HANLP_IMPORT_ERROR = str(e)

_hanlp_tokenizer = None
_hanlp_load_error: Optional[str] = None


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
    parts = []
    for p in doc.paragraphs:
        if p.text:
            parts.append(p.text)
    return "\n".join(parts)


def load_terms_csv_path(path: Path) -> pd.DataFrame:
    raw = path.read_bytes()
    text = read_txt(raw)
    df = pd.read_csv(io.StringIO(text), sep=";", dtype=str, keep_default_na=False)
    df.columns = [c.strip().lower().rstrip(",") for c in df.columns]
    required = ["concept", "term", "pinyin", "translation", "category"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")
    for c in required:
        df[c] = df[c].astype(str).map(lambda s: s.strip().rstrip(","))
    df = df[df["term"].map(lambda x: len(x) > 0)].copy()
    df = df.drop_duplicates(subset=["concept", "term", "pinyin", "translation", "category"]).reset_index(drop=True)
    return df


def count_substring_occurrences(text: str, term: str) -> int:
    if not term:
        return 0
    pattern = re.escape(term)
    return len(re.findall(pattern, text))


def count_longest_non_overlapping_matches(text: str, terms: List[str]) -> Dict[str, int]:
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


def _get_hanlp_tokenizer():
    global _hanlp_tokenizer, _hanlp_load_error
    if not HANLP_AVAILABLE:
        return None
    if _hanlp_tokenizer is not None:
        return _hanlp_tokenizer
    try:
        _hanlp_tokenizer = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
        _hanlp_load_error = None
        return _hanlp_tokenizer
    except Exception as e:
        _hanlp_load_error = str(e)
        return None


def _hanlp_tokenize(text: str) -> List[str]:
    tok = _get_hanlp_tokenizer()
    if tok is None:
        return []
    if not text or not text.strip():
        return []
    try:
        result = tok([text])
        if not result:
            return []
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
    counter: Dict[str, int] = {}
    n = len(tokens)
    for k in range(1, min(max_n + 1, n + 1)):
        for i in range(n - k + 1):
            ngram = "".join(tokens[i : i + k])
            if ngram:
                counter[ngram] = counter.get(ngram, 0) + 1
    return counter


def build_token_counter(text: str, mode: str) -> Optional[Dict[str, int]]:
    """Same modes as app.build_token_counter (subset used here)."""
    if mode == "hanlp":
        if not HANLP_AVAILABLE:
            return None
        tokens = _hanlp_tokenize(text)
        return _hanlp_ngram_counter(tokens)
    if not JIEBA_AVAILABLE:
        return None
    if mode == "jieba_search":
        tokens = jieba.cut_for_search(text)
    elif mode == "jieba_precise":
        tokens = jieba.cut(text, cut_all=False)
    else:
        return None
    counter: Dict[str, int] = {}
    for t in tokens:
        t = safe_str(t).strip()
        if not t:
            continue
        counter[t] = counter.get(t, 0) + 1
    return counter


def analyze_text(text: str, terms_df: pd.DataFrame, match_mode: str) -> pd.DataFrame:
    """Mirrors app.analyze_text for supported match_mode values."""
    text = normalize_text(text)

    token_counter_precise = None
    token_counter_search = None
    token_counter_hanlp = None

    if match_mode in ("jieba_precise", "hybrid") and JIEBA_AVAILABLE:
        token_counter_precise = build_token_counter(text, "jieba_precise")
    if match_mode == "jieba_search" and JIEBA_AVAILABLE:
        token_counter_search = build_token_counter(text, "jieba_search")
    if match_mode == "hanlp" and HANLP_AVAILABLE:
        token_counter_hanlp = build_token_counter(text, "hanlp")

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
                cnt = count_substring_occurrences(text, term)
            else:
                cnt = int(token_counter_precise.get(term, 0))
        elif match_mode == "jieba_search":
            if token_counter_search is None:
                cnt = count_substring_occurrences(text, term)
            else:
                cnt = int(token_counter_search.get(term, 0))
        elif match_mode == "hanlp":
            if token_counter_hanlp is None or len(token_counter_hanlp) == 0:
                cnt = count_substring_occurrences(text, term)
            else:
                cnt = int(token_counter_hanlp.get(term, 0))
        elif match_mode == "hybrid":
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
    df = (
        df.groupby(["CH term", "Pinyin", "ENG translation", "Concept", "Category"], as_index=False)["Count"]
        .sum()
        .sort_values(["Category", "Concept", "Count"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    return df


def iter_doc_paths(root: Path, recursive: bool, exts: Tuple[str, ...]) -> List[Path]:
    if recursive:
        out: List[Path] = []
        for ext in exts:
            out.extend(root.rglob(f"*{ext}"))
        return sorted(set(out))
    out = []
    for ext in exts:
        out.extend(root.glob(f"*{ext}"))
    return sorted(set(out))


def rel_under_root(root: Path, p: Path) -> str:
    try:
        return str(p.relative_to(root))
    except ValueError:
        return str(p)


def read_local_file(path: Path) -> Tuple[Optional[str], Optional[str]]:
    data = path.read_bytes()
    lower = path.name.lower()
    if lower.endswith(".txt"):
        return normalize_text(read_txt(data)), None
    if lower.endswith(".docx"):
        try:
            return normalize_text(read_docx(data)), None
        except Exception as e:
            return None, str(e)
    return None, "unsupported"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs-dir", type=str, required=True)
    ap.add_argument("--terms", type=str, default="civilization_terms_UPDATED_full.csv")
    ap.add_argument("--out-dir", type=str, default="batch_out")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--max-files", type=int, default=0, help="0 = all")
    ap.add_argument(
        "--match-mode",
        type=str,
        default="jieba_precise",
        choices=("jieba_precise", "jieba_search", "substring", "hybrid", "hanlp"),
        help="Default jieba_precise (fast on many .txt). Use hanlp for neural segmentation.",
    )
    args = ap.parse_args()

    root = Path(args.docs_dir)
    if not root.is_dir():
        print(f"ERROR: not a directory: {root}", file=sys.stderr)
        return 2

    terms_path = Path(args.terms)
    if not terms_path.is_file():
        print(f"ERROR: terms file not found: {terms_path}", file=sys.stderr)
        return 2

    match_mode = args.match_mode
    if match_mode == "hanlp" and not HANLP_AVAILABLE:
        print(f"ERROR: HanLP not available: {HANLP_IMPORT_ERROR}", file=sys.stderr)
        return 3
    if match_mode in ("jieba_precise", "jieba_search", "hybrid") and not JIEBA_AVAILABLE:
        print("WARN: jieba not available — falling back to match_mode=substring", flush=True)
        match_mode = "substring"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    terms_df = load_terms_csv_path(terms_path)
    print(f"Terms: {terms_path} rows={len(terms_df)}", flush=True)
    print(f"match_mode={match_mode} (requested={args.match_mode})", flush=True)

    paths = iter_doc_paths(root, args.recursive, (".txt", ".docx"))
    if args.max_files and args.max_files > 0:
        paths = paths[: int(args.max_files)]
    print(f"Documents: {len(paths)} under {root} recursive={args.recursive}", flush=True)

    all_rows: List[Dict] = []
    errors: List[Tuple[str, str]] = []
    t0 = time.time()
    per_dir = out_dir / "per_doc"
    per_dir.mkdir(exist_ok=True)

    for i, p in enumerate(paths, start=1):
        rel = rel_under_root(root, p)
        text, err = read_local_file(p)
        if err or text is None:
            errors.append((rel, err or "empty"))
            continue
        try:
            hits = analyze_text(text, terms_df, match_mode)
        except Exception as e:
            errors.append((rel, str(e)))
            continue
        if not hits.empty:
            h = hits.copy()
            h.insert(0, "source_file", rel)
            safe_name = rel.replace("\\", "_").replace("/", "_").replace(":", "_")
            if len(safe_name) > 180:
                safe_name = safe_name[:180] + "__trunc"
            h.to_csv(per_dir / f"{safe_name}.hits.csv", sep=";", index=False, encoding="utf-8-sig")
            for _, row in h.iterrows():
                all_rows.append(row.to_dict())
        if i % 25 == 0 or i == len(paths):
            print(f"  progress {i}/{len(paths)} elapsed={time.time() - t0:.1f}s", flush=True)

    if all_rows:
        pd.DataFrame(all_rows).to_csv(out_dir / "all_docs_combined.hits.csv", sep=";", index=False, encoding="utf-8-sig")

    if errors:
        pd.DataFrame(errors, columns=["file", "error"]).to_csv(
            out_dir / "read_errors.csv", sep=";", index=False, encoding="utf-8-sig"
        )

    meta = (
        f"match_mode_effective={match_mode}\n"
        f"match_mode_requested={args.match_mode}\n"
        f"docs={len(paths)}\n"
        f"terms_rows={len(terms_df)}\n"
        f"seconds={time.time() - t0:.1f}\n"
    )
    (out_dir / "run_meta.txt").write_text(meta, encoding="utf-8")

    print(f"Done in {time.time() - t0:.1f}s -> {out_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
