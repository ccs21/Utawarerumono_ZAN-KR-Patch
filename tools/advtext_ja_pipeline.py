#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utawarerumono ZAN - advtext 일본어 번역 파이프라인 (단일 파일 툴)

이 툴 하나로:
1) advtext.csv(원본 export 결과)에서 "일본어로 보이는 문자열"만 뽑아 정렬/추출
2) 번역 후(kr 컬럼 채움) 다시 원본 advtext.csv의 "정확한 슬롯(text1~text4)"에 병합하여 advtext_ready.csv 생성

중국어/영어는 무시(참고 컬럼도 생성하지 않음).

사용 예)

[1] 일본어 추출(번역 작업용 CSV 생성)
  python advtext_ja_pipeline.py extract advtext.csv advtext_ja.csv

[2] advtext_ja.csv에서 kr 컬럼 채우기(번역)

[3] 병합(임포트용 CSV 생성)
  python advtext_ja_pipeline.py merge advtext.csv advtext_ja.csv advtext_ready.csv

그 다음은 기존 임포트 툴로:
  python advtext_tool_final.py import advtext.cat advtext_ready.csv advtext_patched.cat

CSV 형식 (advtext_ja.csv):
  section,index,id_hex,slot,ja,kr

주의:
- ja 문자열 안에 \\xNN(제어코드)나 특수 태그가 있으면 그대로 유지하는 것을 강력 권장.
- slot은 text1~text4 중 하나이며, merge 시 이 slot만 교체됩니다.

판정 로직:
- kana(ひらがな/カタカナ)가 포함되면 일본어로 강하게 판정
- 그 외 CJK만 있으면(한자만) 일본어/중국어 구분이 어려워서 기본적으로 제외
  (필요하면 --include-kanji-only 옵션으로 포함 가능)
"""
from __future__ import annotations
import argparse
import csv
import re
from pathlib import Path

KANA_RE = re.compile(r'[\u3040-\u30FF]')  # hiragana+katakana
CJK_RE  = re.compile(r'[\u4E00-\u9FFF]')  # han
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
        # kanji-only lines (names, terms) - optional
        return True
    return False

def pick_best_ja_slot(row: dict, include_kanji_only: bool=False) -> tuple[str,str]:
    """
    Return (slot, text) where slot in text1..text4, best Japanese-like candidate.
    Preference:
      - contains kana (strong)
      - if include_kanji_only, allow kanji-only but lower priority
    """
    best = ("", "")
    best_score = -10
    for slot in ("text1","text2","text3","text4"):
        s = row.get(slot, "")
        if s is None:
            s = ""
        s = str(s)
        s2 = strip_hex_escapes(s)
        if not s2:
            continue
        score = 0
        if KANA_RE.search(s2):
            score += 10
        if CJK_RE.search(s2):
            score += 2
        # penalize pure ASCII (english-ish)
        if all(ord(ch) < 128 for ch in s2):
            score -= 5
        # optional allow kanji-only if include_kanji_only
        if not KANA_RE.search(s2) and CJK_RE.search(s2) and not include_kanji_only:
            # skip kanji-only by default
            continue
        if score > best_score:
            best_score = score
            best = (slot, s)
    return best

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
            continue
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
    # sort for stable editing
    out_rows.sort(key=lambda r: (r["section"], r["index"]))
    write_csv(Path(args.out_csv), ["section","index","id_hex","slot","ja","kr"], out_rows)
    print(f"[OK] extracted {len(out_rows)} JP-like rows -> {args.out_csv}")

def cmd_merge(args):
    base_rows = read_csv(Path(args.advtext_csv))
    ja_rows = read_csv(Path(args.ja_csv))

    # Build patch map: (section,index) -> (slot, kr)
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

    # Apply
    for r in base_rows:
        sec = int(r["section"]); idx = int(r["index"])
        for slot in ("text1","text2","text3","text4"):
            key = (sec, idx, slot)
            if key in patch:
                r[slot] = patch[key]

    # Preserve original columns order
    if not base_rows:
        raise SystemExit("base csv has no rows")
    fieldnames = list(base_rows[0].keys())
    write_csv(Path(args.out_csv), fieldnames, base_rows)
    print(f"[OK] merged -> {args.out_csv}")

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="advtext.csv에서 일본어만 추출")
    e.add_argument("advtext_csv")
    e.add_argument("out_csv")
    e.add_argument("--include-kanji-only", action="store_true",
                   help="가나 없는 한자-only 문자열도 일본어 후보로 포함(기본은 제외)")

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
