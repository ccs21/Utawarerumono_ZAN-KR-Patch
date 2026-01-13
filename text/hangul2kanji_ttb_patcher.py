#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hangul2kanji_ttb_patcher.py

Purpose
- Replace Hangul syllables in a TTB-exported CSV (e.g., text.csv) with mapped Kanji glyphs
  (per hangul_to_kanji_mapping_2350.csv), then import back into a .ttb using ttb_tool.py.

Typical use (Windows)
  python hangul2kanji_ttb_patcher.py ^
    --ttb text_ori.ttb ^
    --csv text.csv ^
    --mapping hangul_to_kanji_mapping_2350.csv ^
    --out-ttb text_new.ttb

Notes
- Only Hangul syllables (U+AC00..U+D7A3) are checked for "missing mapping".
- Non-Hangul characters (ASCII, tags, placeholders, punctuation, Kanji, etc.) are preserved.
- Output CSV is written as UTF-8 with BOM (utf-8-sig) for Excel friendliness.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd


HANGUL_SYLLABLE_START = 0xAC00
HANGUL_SYLLABLE_END = 0xD7A3


def is_hangul_syllable(ch: str) -> bool:
    if not ch:
        return False
    o = ord(ch)
    return HANGUL_SYLLABLE_START <= o <= HANGUL_SYLLABLE_END


def load_mapping(mapping_csv: Path) -> Dict[str, str]:
    m = pd.read_csv(mapping_csv, encoding="utf-8-sig")
    if "hangul" not in m.columns or "kanji" not in m.columns:
        raise ValueError("Mapping CSV must contain columns: 'hangul', 'kanji'")
    # Convert to strings just in case
    h = m["hangul"].astype(str).tolist()
    k = m["kanji"].astype(str).tolist()
    d = dict(zip(h, k))
    # Remove accidental "nan" keys if any
    d.pop("nan", None)
    return d


def convert_text(s: str, h2k: Dict[str, str], missing: Set[str]) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s)
    # Normalize to NFC so matching keys like "슌" are stable
    s = unicodedata.normalize("NFC", s)

    out_chars: List[str] = []
    for ch in s:
        if ch in h2k:
            out_chars.append(h2k[ch])
        else:
            if is_hangul_syllable(ch):
                missing.add(ch)
            out_chars.append(ch)
    return "".join(out_chars)


def find_text_column(df: pd.DataFrame, lang: str, explicit: str | None) -> str:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"Specified --text-col '{explicit}' not found in CSV columns.")
        return explicit

    # default convention from ttb_tool export: "{lang}_text"
    cand = f"{lang}_text"
    if cand in df.columns:
        return cand

    # fallback: try case-insensitive match
    lower_map = {c.lower(): c for c in df.columns}
    if cand.lower() in lower_map:
        return lower_map[cand.lower()]

    raise ValueError(
        f"Could not find text column for lang='{lang}'. "
        f"Expected '{cand}'. Use --text-col to specify."
    )


def write_missing_report(
    missing: Set[str],
    df: pd.DataFrame,
    text_col: str,
    report_path: Path,
    max_examples: int = 50,
) -> None:
    missing_sorted = sorted(missing)
    lines: List[str] = []
    lines.append("Missing Hangul mappings report")
    lines.append(f"- Unique missing chars: {len(missing_sorted)}")
    if missing_sorted:
        lines.append("- Characters:")
        lines.append("  " + " ".join(missing_sorted))

        # Examples: show rows that contain any missing char
        import re
        pattern = "[" + re.escape("".join(missing_sorted)) + "]"
        mask = df[text_col].astype(str).str.contains(pattern, na=False)
        ex = df.loc[mask, :].copy()
        lines.append("")
        lines.append(f"- Example rows containing missing chars (up to {max_examples}):")
        cols_priority = [c for c in ["record_index", "key_hash_hex", "row", text_col] if c in ex.columns]
        if cols_priority:
            ex2 = ex[cols_priority].head(max_examples)
            # Plain text dump
            for _, r in ex2.iterrows():
                rec = []
                for c in cols_priority:
                    rec.append(f"{c}={r[c]}")
                lines.append("  - " + " | ".join(rec))
        else:
            lines.append("  (No suitable id columns found to display examples.)")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_ttb_import(
    ttb_tool: Path,
    in_ttb: Path,
    mapped_csv: Path,
    out_ttb: Path,
    lang: str,
) -> None:
    cmd = [sys.executable, str(ttb_tool), "import", str(in_ttb), str(mapped_csv), str(out_ttb), "--lang", lang]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ttb_tool import failed (code {proc.returncode}). Output:\n{proc.stdout}")
    print(proc.stdout.strip())


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Replace Hangul in CSV with mapped Kanji glyphs, then import into .ttb using ttb_tool.py."
    )
    ap.add_argument("--ttb", type=Path, required=True, help="Input original .ttb (e.g., text_ori.ttb)")
    ap.add_argument("--csv", type=Path, required=True, help="Input CSV exported from ttb_tool.py (e.g., text.csv)")
    ap.add_argument("--mapping", type=Path, required=True, help="Hangul→Kanji mapping CSV")
    ap.add_argument("--out-ttb", type=Path, required=True, help="Output patched .ttb")
    ap.add_argument("--out-csv", type=Path, default=None, help="Output mapped CSV (default: <csv>_mapped.csv)")
    ap.add_argument("--lang", default="jaJP", help="Target language code to replace in TTB (default: jaJP)")
    ap.add_argument("--text-col", default=None, help="Text column name to transform (default: <lang>_text)")
    ap.add_argument("--ttb-tool", type=Path, default=None, help="Path to ttb_tool.py (default: next to this script)")
    ap.add_argument("--strict", action="store_true", help="Fail if any Hangul syllable has no mapping")
    ap.add_argument("--only-csv", action="store_true", help="Only write mapped CSV and report; do NOT rebuild .ttb")
    ap.add_argument("--report", type=Path, default=None, help="Write missing-mapping report to this path")

    args = ap.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"[ERR] CSV not found: {args.csv}")
    if not args.mapping.exists():
        raise SystemExit(f"[ERR] Mapping CSV not found: {args.mapping}")
    if not args.ttb.exists():
        raise SystemExit(f"[ERR] TTB not found: {args.ttb}")

    ttb_tool = args.ttb_tool
    if ttb_tool is None:
        # Assume ttb_tool.py sits next to this script
        ttb_tool = Path(__file__).with_name("ttb_tool.py")
    if not args.only_csv and not ttb_tool.exists():
        raise SystemExit(f"[ERR] ttb_tool.py not found: {ttb_tool}")

    out_csv = args.out_csv
    if out_csv is None:
        out_csv = args.csv.with_name(args.csv.stem + "_mapped.csv")

    report_path = args.report
    if report_path is None:
        report_path = args.csv.with_name(args.csv.stem + "_missing_report.txt")

    # Load
    df = pd.read_csv(args.csv, encoding="utf-8-sig")
    text_col = find_text_column(df, args.lang, args.text_col)

    h2k = load_mapping(args.mapping)

    missing: Set[str] = set()
    df[text_col] = df[text_col].apply(lambda s: convert_text(s, h2k, missing))

    # Save mapped CSV
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] Wrote mapped CSV: {out_csv}")

    # Report missing mappings (if any)
    if missing:
        write_missing_report(missing, df, text_col, report_path)
        print(f"[WARN] Missing mapping chars: {''.join(sorted(missing))}")
        print(f"[WARN] Wrote report: {report_path}")
        if args.strict:
            raise SystemExit("[ERR] --strict enabled: missing mappings found. Add mappings then rerun.")
    else:
        # still write a small report for completeness
        write_missing_report(missing, df, text_col, report_path)
        print(f"[OK] No missing Hangul mappings. Report: {report_path}")

    if args.only_csv:
        print("[OK] --only-csv specified. Skipping .ttb rebuild.")
        return

    # Import into TTB
    run_ttb_import(ttb_tool, args.ttb, out_csv, args.out_ttb, args.lang)
    print(f"[OK] Wrote patched TTB: {args.out_ttb}")


if __name__ == "__main__":
    main()
