# -*- coding: utf-8 -*-
"""
extract_all_unpatched_advtext.py

Extract ALL Japanese-looking strings from advtext.cat that are STILL UNPATCHED
(by comparing raw slot bytes with advtext_patched.cat).

Output CSV is compatible with:
python advtext_used_inplace_workflow_v3_fixed.py apply ...

Usage:
python extract_all_unpatched_advtext.py ^
  --orig Binary\Adv\advtext.cat ^
  --patched Binary\Adv\advtext_patched.cat ^
  --out all_unpatched_advtext_jp.csv ^
  --min-chars 1
"""

from __future__ import annotations
import argparse
from pathlib import Path
import advtext_used_inplace_workflow_v3_fixed as wf
import re

# 한자(기본 CJK Unified Ideographs) 범위
_HAN_RE = re.compile(r"[\u4E00-\u9FFF]")

def contains_kana(s: str) -> bool:
    """일본어 히라가나/가타카나가 포함돼 있으면 True"""
    for ch in s:
        o = ord(ch)
        if (0x3040 <= o <= 0x309F) or (0x30A0 <= o <= 0x30FF):
            return True
    return False

def has_han_run_ge(s: str, n: int = 8) -> bool:
    """
    쉼표/공백(반각/전각) 제거 후,
    한자가 n개 이상 '연속'으로 등장하면 True
    """
    t = s.replace(" ", "").replace("　", "").replace(",", "").replace("，", "")
    run = 0
    for ch in t:
        if _HAN_RE.match(ch):
            run += 1
            if run >= n:
                return True
        else:
            run = 0
    return False


def iter_cstring_slots(buf: bytes):
    """
    Yield (start, end_excl) for each NUL-terminated slot.
    end_excl points to position AFTER the 0x00 terminator.
    """
    n = len(buf)
    i = 0
    while i < n:
        start = i
        # find next null
        j = buf.find(b"\x00", i)
        if j == -1:
            # no terminator -> stop (broken file)
            break
        end_excl = j + 1
        yield start, end_excl
        i = end_excl


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract all unpatched JP strings from advtext by full slot scan.")
    ap.add_argument("--orig", required=True, help="Original advtext.cat (JP baseline)")
    ap.add_argument("--patched", required=True, help="Patched advtext.cat to compare (current)")
    ap.add_argument("--out", required=True, help="Output CSV path")

    ap.add_argument("--lang", choices=["jp", "zh", "en", "all"], default="jp",
                    help="Language filter heuristic (default: jp). Use 'all' to keep all decoded strings.")
    ap.add_argument("--min-chars", type=int, default=1,
                    help="Minimum stripped char length to keep (default: 1 for short lines like 今 / 走).")
    ap.add_argument("--min-budget-bytes", type=int, default=2,
                    help="Minimum slot byte budget (incl NUL). default 2 keeps 1-byte text + NUL.")
    ap.add_argument("--ruby", choices=["keep", "strip"], default="strip",
                    help="Strip ruby tags like <帝:ミカド> -> 帝 (default: strip)")
    ap.add_argument("--no-protect-special", action="store_true",
                    help="Do not tokenise #@－－# marker in CSV (default: protected token)")

    args = ap.parse_args()

    orig_path = Path(args.orig)
    patched_path = Path(args.patched)
    out_csv = Path(args.out)

    orig = orig_path.read_bytes()
    patched = patched_path.read_bytes()

    if len(orig) != len(patched):
        print(f"[WARN] size differs: orig={len(orig)} patched={len(patched)}")
        print("       비교는 공통 범위 내에서만 진행합니다. (범위 밖 슬롯은 제외)")

    size = min(len(orig), len(patched))

    slots = {}
    ref_count = 0  # unknown (full scan), apply에는 영향 없음

    protect_special = not args.no_protect_special

    for start, end_excl in iter_cstring_slots(orig[:size]):
        budget = end_excl - start
        if budget < args.min_budget_bytes:
            continue

        # 아직 패치 안 된 슬롯만
        if orig[start:end_excl] != patched[start:end_excl]:
            continue

        raw = orig[start:end_excl - 1]  # without NUL
        s = wf.try_decode_utf8(raw)
        if s is None:
            continue

        if args.ruby == "strip":
            s = wf.strip_ruby_tags(s)

        if protect_special:
            s = wf.tokenise_marker(s)

        if wf.looks_like_text(s, min_chars=args.min_chars) is False:
            # min_chars=1이면 대부분 통과, 그래도 바이너리 찌꺼기 방지용
            continue

        # 중국어로 판단되는 "한자 8연속(쉼표/공백 제외) + 가나 없음"은 제외
        if has_han_run_ge(s, 8) and not contains_kana(s):
            continue

        if args.lang != "all" and wf.classify_lang(s) != args.lang:
            continue

        locks = wf.extract_locks(s)

        slots[start] = wf.Slot(
            start=start,
            end=end_excl,
            budget=budget,
            ref_count=ref_count,
            ja=s,
            locks=locks
        )

    wf.write_slots_csv(out_csv, slots)
    print(f"[OK] unpatched slots extracted: {len(slots)} -> {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
