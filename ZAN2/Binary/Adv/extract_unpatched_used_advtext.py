#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
extract_unpatched_used_advtext.py

- Extract "used" string slots from original advtext.cat by scanning script .cat files
  (same logic as advtext_used_inplace_workflow_v3_fixed.py extract)
- Compare each slot region (start ~ start+budget) between original and patched advtext
- Output ONLY slots that are identical (i.e., still unpatched)

Usage example:
python extract_unpatched_used_advtext.py ^
  --orig advtext.cat ^
  --patched advtext_patched.cat ^
  --scripts advspt_story.cat advspt_proto.cat ^
  --out used_advtext_unpatched_jp.csv ^
  --lang jp --min-chars 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Reuse the proven extractor/writer from your workflow script
import advtext_used_inplace_workflow_v3_fixed as wf


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract ONLY unpatched (still-original) used advtext strings by comparing orig vs patched."
    )
    p.add_argument("--orig", required=True, help="Original advtext.cat (JP baseline)")
    p.add_argument("--patched", required=True, help="Already-patched advtext.cat to compare against")
    p.add_argument("--scripts", nargs="+", required=True, help="Script .cat files to scan (e.g., advspt_story.cat advspt_proto.cat)")
    p.add_argument("--out", required=True, help="Output CSV path (only unpatched slots)")

    p.add_argument("--step", type=int, default=1, help="Byte step when scanning scripts for u32 values (1=most thorough; 4=faster).")
    p.add_argument("--include-binary-like", action="store_true",
                   help="Do not filter out strings that don't look like text (may add noise).")
    p.add_argument("--lang", choices=["jp", "zh", "en", "all"], default="jp",
                   help="Language filter heuristic (default: jp). Use 'all' to keep all.")
    p.add_argument("--min-chars", type=int, default=1,
                   help="Minimum stripped character length to keep (default: 1 for short lines).")
    p.add_argument("--min-budget-bytes", type=int, default=8,
                   help="Minimum byte budget (including NUL) to keep (default: 8).")
    p.add_argument("--min-ref", type=int, default=1,
                   help="Minimum reference count across scripts to keep (default: 1).")
    p.add_argument("--ruby", choices=["keep", "strip"], default="strip",
                   help="How to handle ruby/furigana tags like <帝:ミカド>. 'strip' keeps only base text (default).")
    p.add_argument("--no-protect-special", action="store_true",
                   help="Do not replace special marker #@－－# with CSV-safe token (default: protected).")
    return p


def main() -> int:
    args = build_parser().parse_args()

    orig_path = Path(args.orig)
    patched_path = Path(args.patched)
    script_paths = [Path(p) for p in args.scripts]
    out_csv = Path(args.out)

    if not orig_path.exists():
        raise SystemExit(f"[ERR] --orig not found: {orig_path}")
    if not patched_path.exists():
        raise SystemExit(f"[ERR] --patched not found: {patched_path}")
    for sp in script_paths:
        if not sp.exists():
            raise SystemExit(f"[ERR] script not found: {sp}")

    # 1) Collect used slots from ORIGINAL (so 'ja' column is clean JP baseline)
    slots = wf.collect_used_slots(
        advtext_path=orig_path,
        script_paths=script_paths,
        step=args.step,
        require_text=not args.include_binary_like,
        lang=args.lang,
        min_chars=args.min_chars,
        min_budget_bytes=args.min_budget_bytes,
        min_ref_count=args.min_ref,
        ruby=args.ruby,
        protect_special=not args.no_protect_special,
    )

    orig_bytes = orig_path.read_bytes()
    patched_bytes = patched_path.read_bytes()

    if len(orig_bytes) != len(patched_bytes):
        print(f"[WARN] size differs: orig={len(orig_bytes)} bytes, patched={len(patched_bytes)} bytes")
        print("       비교는 가능한 범위에서만 진행합니다(오프셋 초과 슬롯은 제외).")

    # 2) Filter: keep only slots whose region is identical in patched (=> not touched)
    unpatched = {}
    skipped_oob = 0
    for start, slot in slots.items():
        end = start + slot.budget
        if end > len(orig_bytes) or end > len(patched_bytes):
            skipped_oob += 1
            continue

        if orig_bytes[start:end] == patched_bytes[start:end]:
            unpatched[start] = slot

    # 3) Write CSV in the exact same format your apply pipeline expects
    wf.write_slots_csv(out_csv, unpatched)

    print(f"[OK] used slots (orig): {len(slots)}")
    print(f"[OK] unpatched slots  : {len(unpatched)} -> {out_csv}")
    if skipped_oob:
        print(f"[WARN] skipped out-of-bounds slots: {skipped_oob}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
