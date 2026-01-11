#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utawarerumono ZAN - advtext 일본어 번역 파이프라인 (단일 파일 툴) v2

변경점(v2):
- "일본어가 하나도 없는 레코드"에서 영어가 뽑히는 문제를 해결:
  -> 이제는 기본적으로 **가나(ひらがな/カタカナ)가 포함된 문자열만** 일본어로 인정하고 추출합니다.
  -> 옵션 --include-kanji-only 를 주면 한자-only도 포함합니다(인명/용어 등).

사용:

[1] 일본어 추출(번역 작업용 CSV 생성)
  python advtext_ja_pipeline.py extract advtext.csv advtext_ja.csv

[2] advtext_ja.csv에서 kr 컬럼 채우기(번역)

[3] 병합(임포트용 CSV 생성)
  python advtext_ja_pipeline.py merge advtext.csv advtext_ja.csv advtext_ready.csv

그 다음 임포트:
  python advtext_tool_final.py import advtext.cat advtext_ready.csv advtext_patched.cat

advtext_ja.csv 컬럼:
  section,index,id_hex,slot,ja,kr
"""
from __future__ import annotations
import argparse
import csv
import re
from pathlib import Path

KANA_RE = re.compile(r'[\u3040-\u30FF]')   # hiragana+katakana
CJK_RE  = re.compile(r'[\u4E00-\u9FFF]')   # han (kanji/hanzi)
HEX_ESC_RE = re.compile(r'\\x[0-9A-Fa-f]{2}')

def strip_hex_escapes(s: str) -> str:
    return HEX_ESC_RE.sub('', s)

def is_japanese_like(s: str, include_kanji_only: bool=False) -> bool:
    s2 = strip_hex_escapes(s)
    if not s2:
        return False
    if KANA_RE.search(s2):
        return True
    if include_kanji_only and CJK_RE.search(s2):
        return True
    return False

def score_japanese(s: str) -> int:
    """Higher = more likely to be Japanese text."""
    s2 = strip_hex_escapes(s)
    score = 0
    if KANA_RE.search(s2):
        score += 100
    # kanji presence helps a bit
    if CJK_RE.search(s2):
        score += 10
    # longish text preferred over very short tokens
    if len(s2) >= 8:
        score += 5
    return score

def pick_best_ja_slot(row: dict, include_kanji_only: bool=False) -> tuple[str,str]:
    """
    Pick best Japanese candidate among text1..text4.
    IMPORTANT: We only consider slots that pass is_japanese_like() so English-only lines won't be picked.
    """
    best_slot = ""
    best_text = ""
    best_score = -1
    for slot in ("text1","text2","text3","text4"):
        s = row.get(slot, "")
        if s is None:
            s = ""
        s = str(s)
        if not is_japanese_like(s, include_kanji_only=include_kanji_only):
            continue
        sc = score_japanese(s)
        if sc > best_score:
            best_score = sc
            best_slot, best_text = slot, s
    return best_slot, best_text

def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        return list(r)

def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

def cmd_extract(args):
    src = read_csv(Path(args.advtext_csv))
    out_rows = []
    seen = set()
    for row in src:
        section = int(row["section"])
        index = int(row["index"])
        id_hex = row.get("id_hex", "")
        slot, ja = pick_best_ja_slot(row, include_kanji_only=args.include_kanji_only)
        if not slot or not ja:
            continue  # no JP-like candidate -> skip (prevents English-only leakage)
        key = (section, index, slot)
        if key in seen:
            continue
        seen.add(key)
        out_rows.append({
            "section": section,
            "index": index,
            "id_hex": id_hex,
            "slot": slot,
            "ja": ja,
            "kr": "",
        })
    out_rows.sort(key=lambda r: (r["section"], r["index"]))
    write_csv(Path(args.out_csv), ["section","index","id_hex","slot","ja","kr"], out_rows)
    print(f"[OK] extracted {len(out_rows)} JP-like rows -> {args.out_csv}")

def cmd_merge(args):
    base_rows = read_csv(Path(args.advtext_csv))
    ja_rows = read_csv(Path(args.ja_csv))

    patch = {}
    for r in ja_rows:
        sec = int(r["section"]); idx = int(r["index"])
        slot = str(r.get("slot","")).strip()
        kr = r.get("kr","")
        if kr is None:
            continue
        kr = str(kr)
        if not kr.strip():
            continue
        if slot not in ("text1","text2","text3","text4"):
            continue
        patch[(sec, idx, slot)] = kr

    for r in base_rows:
        sec = int(r["section"]); idx = int(r["index"])
        for slot in ("text1","text2","text3","text4"):
            key = (sec, idx, slot)
            if key in patch:
                r[slot] = patch[key]

    if not base_rows:
        raise SystemExit("base csv has no rows")
    fieldnames = list(base_rows[0].keys())
    write_csv(Path(args.out_csv), fieldnames, base_rows)
    print(f"[OK] merged -> {args.out_csv}")

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="advtext.csv에서 일본어만 추출(기본: 가나 포함만)")
    e.add_argument("advtext_csv")
    e.add_argument("out_csv")
    e.add_argument("--include-kanji-only", action="store_true",
                   help="가나 없는 한자-only 문자열도 일본어 후보로 포함")

    m = sub.add_parser("merge", help="번역된 ja csv를 advtext.csv에 병합")
    m.add_argument("advtext_csv")
    m.add_argument("ja_csv")
    m.add_argument("out_csv")

    args = ap.parse_args()
    if args.cmd == "extract":
        cmd_extract(args)
    else:
        cmd_merge(args)

if __name__ == "__main__":
    main()
