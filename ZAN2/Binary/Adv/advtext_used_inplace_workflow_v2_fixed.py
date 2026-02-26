#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
advtext_used_inplace_workflow.py

Goal:
- Extract only actually-referenced text strings from advtext.cat by scanning script .cat files
  (advspt_story.cat / advspt_proto.cat, etc.) for 32-bit little-endian offsets into advtext.cat.
- Normalize mid-string pointers to the true string start (backtrack to previous 0x00).
- Produce a translation CSV with strict byte budgets (in-place patching: offsets must not move).
- Apply translations:
    translate -> (optional) Hangul->Kanji mapping (font workaround) -> strict locks check -> in-place overwrite.

Design principles:
- NEVER change advtext.cat size or any offsets. Only overwrite bytes within each original null-terminated slot.
- Fail fast if:
    * budget overflow
    * missing required locks/tokens (tags, placeholders, #@－－#)
    * unmapped Hangul remains after mapping
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# -----------------------------
# Helpers / detection
# -----------------------------

RE_TAG = re.compile(r"<[^>]+>")                 # e.g. <onpu>, <帝:...>
RE_PLACEHOLDER = re.compile(r"\{\{L\d{3}\}\}")  # e.g. {{L000}}
SPECIAL_LOCKS = ["#@－－#"]                      # must be preserved verbatim

# Hangul unicode blocks (syllables + jamo)
def is_hangul(ch: str) -> bool:
    o = ord(ch)
    return (
        0xAC00 <= o <= 0xD7A3 or  # Hangul syllables
        0x1100 <= o <= 0x11FF or  # Hangul Jamo
        0x3130 <= o <= 0x318F or  # Hangul Compatibility Jamo
        0xA960 <= o <= 0xA97F or  # Hangul Jamo Extended-A
        0xD7B0 <= o <= 0xD7FF     # Hangul Jamo Extended-B
    )

def classify_lang(s: str) -> str:
    """Rough language tag based on Unicode ranges."""
    if not s:
        return "other"
    if re.search(r'[\u3040-\u30FF\uFF66-\uFF9D「」『』。！？…]', s):
        return "jp"
    if re.search(r'[A-Za-z]', s) and not re.search(r'[\u4E00-\u9FFF\u3040-\u30FF\uFF66-\uFF9D]', s):
        return "en"
    if re.search(r'[\u4E00-\u9FFF]', s) and not re.search(r'[\u3040-\u30FF\uFF66-\uFF9D]', s):
        return "zh"
    return "other"

def looks_like_text(s: str, min_chars: int = 4) -> bool:
    """Stricter heuristic to drop binary garbage decoded as UTF-8."""
    if not s:
        return False
    t = s.strip()
    if len(t) < min_chars:
        return False

    # Reject if it contains control characters (except \n/\r/\t).
    if any((ord(c) < 0x20 and c not in "\n\r\t") for c in t):
        return False

    # Printable ratio: allow spaces/newlines, punctuation, CJK, letters, digits.
    printable = sum(1 for c in t if (c.isprintable() or c in "\n\r\t"))
    if printable / max(1, len(t)) < 0.90:
        return False

    # Must contain at least one meaningful char (CJK/letters/digits).
    if not re.search(r'[\u4E00-\u9FFF\u3040-\u30FF\uFF66-\uFF9DA-Za-z0-9]', t):
        return False

    return True

def extract_locks(src: str) -> List[str]:
    locks: List[str] = []
    locks.extend(RE_TAG.findall(src))
    locks.extend(RE_PLACEHOLDER.findall(src))
    for sp in SPECIAL_LOCKS:
        if sp in src:
            locks.append(sp)
    # De-dup preserving order
    seen = set()
    out = []
    for x in locks:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

# -----------------------------
# Data structures
# -----------------------------

@dataclass
class Slot:
    start: int           # start offset of null-terminated string
    end: int             # end offset (exclusive) - points to the 0x00 terminator +1
    budget: int          # end-start bytes; includes 0x00 terminator inside that region (budget>=1)
    ref_count: int
    ja: str
    locks: List[str]

# -----------------------------
# Scanning logic
# -----------------------------

def read_bytes(p: Path) -> bytes:
    return p.read_bytes()

def iter_u32_le_candidates(buf: bytes, step: int = 1) -> Iterable[Tuple[int, int]]:
    """Yield (pos, value) for each u32 at pos. step is bytes to move each iteration."""
    if step < 1:
        step = 1
    n = len(buf)
    last = n - 4
    for i in range(0, last + 1, step):
        yield i, struct.unpack_from("<I", buf, i)[0]

def normalize_to_string_start(adv: bytes, off: int) -> Optional[int]:
    """Backtrack to previous 0x00 and return start+1; if off is invalid return None."""
    if off < 0 or off >= len(adv):
        return None
    # If off points to 0x00 itself, it's probably boundary; treat the next byte as potential start
    j = off
    # Backtrack until 0x00 or beginning
    while j > 0 and adv[j-1] != 0:
        j -= 1
    # j is start
    return j

def read_c_string(adv: bytes, start: int, max_len: int = 1_000_000) -> Optional[Tuple[bytes, int]]:
    """Return (raw_bytes_without_null, end_exclusive) where end_exclusive is position after 0x00.
    If no null found within max_len or bounds, return None.
    """
    n = len(adv)
    if start < 0 or start >= n:
        return None
    end = start
    limit = min(n, start + max_len)
    while end < limit and adv[end] != 0:
        end += 1
    if end >= limit:
        return None
    # end points to 0x00
    raw = adv[start:end]
    return raw, end + 1

def try_decode_utf8(raw: bytes) -> Optional[str]:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None

def collect_used_slots(advtext_path: Path, script_paths: List[Path], step: int = 1,
                       require_text: bool = True, lang: str = 'jp', min_chars: int = 4,
                       min_budget_bytes: int = 8, min_ref_count: int = 1) -> Dict[int, Slot]:
    adv = read_bytes(advtext_path)
    size = len(adv)

    # count references per normalized start
    counts: Dict[int, int] = {}

    for sp in script_paths:
        buf = read_bytes(sp)
        for _, v in iter_u32_le_candidates(buf, step=step):
            if v < size:
                start = normalize_to_string_start(adv, v)
                if start is None:
                    continue
                counts[start] = counts.get(start, 0) + 1

    slots: Dict[int, Slot] = {}
    for start, ref_count in counts.items():
        cs = read_c_string(adv, start)
        if cs is None:
            continue
        raw, end_excl = cs
        s = try_decode_utf8(raw)
        if s is None:
            continue
        if (len(raw) + 1) < min_budget_bytes:
            continue
        if ref_count < min_ref_count:
            continue
        if require_text and not looks_like_text(s, min_chars=min_chars):
            continue
        if lang != 'all' and classify_lang(s) != lang:
            continue
        locks = extract_locks(s)
        slots[start] = Slot(
            start=start,
            end=end_excl,
            budget=end_excl - start,
            ref_count=ref_count,
            ja=s,
            locks=locks,
        )
    return slots

# -----------------------------
# CSV I/O
# -----------------------------

CSV_FIELDS = [
    "start_offset_hex",
    "start_offset_dec",
    "byte_budget",
    "ref_count",
    "ja",
    "locks",
    "kr",
]

def write_slots_csv(out_path: Path, slots: Dict[int, Slot]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for start in sorted(slots.keys()):
            s = slots[start]
            w.writerow({
                "start_offset_hex": f"0x{s.start:X}",
                "start_offset_dec": str(s.start),
                "byte_budget": str(s.budget),
                "ref_count": str(s.ref_count),
                "ja": s.ja,
                "locks": json.dumps(s.locks, ensure_ascii=False),
                "kr": "",
            })

def read_translation_csv(p: Path) -> List[dict]:
    # Accept utf-8-sig / utf-8
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        rows = [row for row in r]
    return rows

# -----------------------------
# Hangul->Kanji mapping
# -----------------------------

def load_hangul_mapping(mapping_csv: Path) -> Dict[str, str]:
    mp: Dict[str, str] = {}
    # mapping file can be messy; read as csv with two columns (hangul, kanji) or (kr, jp)
    with mapping_csv.open("r", encoding="utf-8-sig", newline="") as f:
        rdr = csv.reader(f)
        for row in rdr:
            if not row:
                continue
            if len(row) < 2:
                continue
            a = row[0].strip()
            b = row[1].strip()
            if not a or not b:
                continue
            # Usually 1 char -> 1 char. If multiple chars, we still allow but only for exact match char mapping.
            if len(a) == 1 and len(b) >= 1:
                mp[a] = b[0]
    return mp

def apply_hangul_mapping(text: str, mp: Dict[str, str]) -> Tuple[str, List[str]]:
    missing: List[str] = []
    out_chars: List[str] = []
    for ch in text:
        if is_hangul(ch):
            rep = mp.get(ch)
            if rep is None:
                missing.append(ch)
                out_chars.append(ch)  # keep for now to report
            else:
                out_chars.append(rep)
        else:
            out_chars.append(ch)
    return "".join(out_chars), missing

# -----------------------------
# Patch apply
# -----------------------------

class PatchError(Exception):
    pass

def strict_lock_check(src_ja: str, dst_text: str) -> List[str]:
    """Return missing locks that exist in src but not in dst."""
    required = extract_locks(src_ja)
    missing = [lk for lk in required if lk not in dst_text]
    return missing

def apply_inplace_patch(
    advtext_path: Path,
    used_csv_path: Path,
    out_advtext_path: Path,
    mapping_csv: Optional[Path] = None,
    fail_on_unmapped_hangul: bool = True,
    fail_on_lock_missing: bool = True,
    dry_run: bool = False,
) -> None:
    adv = bytearray(read_bytes(advtext_path))
    adv_ro = bytes(adv)

    mp: Dict[str, str] = {}
    if mapping_csv is not None:
        mp = load_hangul_mapping(mapping_csv)

    rows = read_translation_csv(used_csv_path)

    # We'll collect detailed errors and stop at first by default.
    for idx, row in enumerate(rows, start=2):  # +1 for header; start=2 matches Excel-ish row number
        # Parse offsets / budget
        off_s = (row.get("start_offset_dec") or "").strip()
        if not off_s:
            continue
        try:
            start = int(off_s)
        except ValueError:
            raise PatchError(f"[CSV] invalid start_offset_dec at row {idx}: {off_s!r}")

        budget_s = (row.get("byte_budget") or "").strip()
        if not budget_s:
            raise PatchError(f"[CSV] missing byte_budget at row {idx} (offset={start})")
        try:
            budget = int(budget_s)
        except ValueError:
            raise PatchError(f"[CSV] invalid byte_budget at row {idx}: {budget_s!r}")

        ja = row.get("ja", "")
        kr = row.get("kr", "")
        if kr is None:
            kr = ""
        kr = kr.replace("\r\n", "\n").replace("\r", "\n")

        if not kr.strip():
            # Not translated; skip
            continue

        # Apply hangul mapping if provided
        mapped = kr
        missing_hangul: List[str] = []
        if mp:
            mapped, missing_hangul = apply_hangul_mapping(kr, mp)
            if missing_hangul and fail_on_unmapped_hangul:
                uniq = "".join(sorted(set(missing_hangul)))
                raise PatchError(
                    f"[MAPPING FAIL] row={idx} offset=0x{start:X} unmapped_hangul={uniq!r}\n"
                    f"KR={kr}"
                )

        # Locks check
        if fail_on_lock_missing:
            missing_locks = strict_lock_check(ja, mapped)
            if missing_locks:
                raise PatchError(
                    f"[LOCK FAIL] row={idx} offset=0x{start:X} missing={missing_locks}\n"
                    f"JA={ja}\n"
                    f"KR={kr}\n"
                    f"MAPPED={mapped}"
                )

        # Encode and size check
        new_bytes = mapped.encode("utf-8")
        # Must fit within budget-1 (leave 0x00 terminator)
        if len(new_bytes) > budget - 1:
            over = len(new_bytes) - (budget - 1)
            raise PatchError(
                f"[OVERFLOW] row={idx} offset=0x{start:X} budget={budget-1} new_len={len(new_bytes)} over_by={over}\n"
                f"JA={ja}\n"
                f"KR={kr}\n"
                f"MAPPED={mapped}"
            )

        # Sanity: ensure slot end really has null in original
        slot_end = start + budget - 1
        if slot_end < 0 or slot_end >= len(adv_ro):
            raise PatchError(f"[RANGE] row={idx} offset=0x{start:X} budget={budget} out of range")
        if adv_ro[slot_end] != 0:
            # budget likely wrong; refuse to patch to avoid shifting into next string
            raise PatchError(
                f"[SLOT] row={idx} offset=0x{start:X} expected_null_at=0x{slot_end:X} but found=0x{adv_ro[slot_end]:02X}\n"
                f"JA={ja}"
            )

        if dry_run:
            continue

        # Write: bytes + null + padding zeros
        adv[start:start + len(new_bytes)] = new_bytes
        adv[start + len(new_bytes)] = 0
        # pad rest with zeros
        pad_from = start + len(new_bytes) + 1
        pad_to = start + budget
        if pad_from < pad_to:
            adv[pad_from:pad_to] = b"\x00" * (pad_to - pad_from)

    if dry_run:
        return

    out_advtext_path.parent.mkdir(parents=True, exist_ok=True)
    out_advtext_path.write_bytes(bytes(adv))

# -----------------------------
# CLI
# -----------------------------

def cmd_extract(args: argparse.Namespace) -> int:
    advtext = Path(args.advtext)
    scripts = [Path(p) for p in args.scripts]
    out_csv = Path(args.out)

    slots = collect_used_slots(
        advtext_path=advtext,
        script_paths=scripts,
        step=args.step,
        require_text=not args.include_binary_like,
        lang=args.lang,
        min_chars=args.min_chars,
        min_budget_bytes=args.min_budget_bytes,
        min_ref_count=args.min_ref,
    )
    write_slots_csv(out_csv, slots)
    print(f"[OK] extracted used slots: {len(slots)} -> {out_csv}")
    return 0

def cmd_apply(args: argparse.Namespace) -> int:
    advtext = Path(args.advtext)
    used_csv = Path(args.csv)
    out_advtext = Path(args.out)

    mapping = Path(args.mapping) if args.mapping else None
    try:
        apply_inplace_patch(
            advtext_path=advtext,
            used_csv_path=used_csv,
            out_advtext_path=out_advtext,
            mapping_csv=mapping,
            fail_on_unmapped_hangul=not args.allow_unmapped_hangul,
            fail_on_lock_missing=not args.no_lock_check,
            dry_run=args.dry_run,
        )
    except PatchError as e:
        print(str(e))
        return 2

    if args.dry_run:
        print("[OK] dry-run passed (no file written).")
    else:
        print(f"[OK] wrote patched advtext: {out_advtext}")
    return 0

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract/apply in-place translations for advtext.cat using advspt_story/proto references."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_ex = sub.add_parser("extract", help="Extract actually-used advtext strings referenced by scripts.")
    p_ex.add_argument("advtext", help="Path to advtext.cat")
    p_ex.add_argument("--scripts", nargs="+", required=True, help="Script .cat files to scan (e.g., advspt_story.cat advspt_proto.cat)")
    p_ex.add_argument("--out", required=True, help="Output CSV path (translation template)")
    p_ex.add_argument("--step", type=int, default=1, help="Byte step when scanning scripts for u32 values (1=most thorough; 4=faster).")
    p_ex.add_argument("--include-binary-like", action="store_true",
                      help="Do not filter out strings that don't look like text (may add noise).")
    p_ex.add_argument("--lang", choices=["jp","zh","en","all"], default="jp",
                  help="Language filter heuristic for extracted strings (default: jp). Use 'all' to keep all.")
    p_ex.add_argument("--min-chars", type=int, default=4,
                  help="Minimum stripped character length to keep (default: 4).")
    p_ex.add_argument("--min-budget-bytes", type=int, default=8,
                  help="Minimum byte budget (including NUL) to keep (default: 8).")
    p_ex.add_argument("--min-ref", type=int, default=1,
                  help="Minimum reference count across scripts to keep (default: 1).")
    p_ex.set_defaults(func=cmd_extract)

    p_ap = sub.add_parser("apply", help="Apply translations from CSV into advtext.cat in-place (no offset change).")
    p_ap.add_argument("advtext", help="Path to original advtext.cat")
    p_ap.add_argument("csv", help="CSV produced by 'extract' with translated 'kr' column filled")
    p_ap.add_argument("--out", required=True, help="Output patched advtext.cat path")
    p_ap.add_argument("--mapping", default="", help="Hangul->Kanji mapping csv (e.g., hangul_to_kanji_mapping_2350.csv)")
    p_ap.add_argument("--allow-unmapped-hangul", action="store_true", help="Do not fail if some Hangul characters are not in mapping.")
    p_ap.add_argument("--no-lock-check", action="store_true", help="Disable strict lock/token check (NOT recommended).")
    p_ap.add_argument("--dry-run", action="store_true", help="Validate only; do not write output file.")
    p_ap.set_defaults(func=cmd_apply)

    return p

def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))

if __name__ == "__main__":
    raise SystemExit(main())
