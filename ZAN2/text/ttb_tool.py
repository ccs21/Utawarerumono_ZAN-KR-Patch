#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utawarerumono ZAN - text.ttb importer/exporter

✅ Discovered format (from real file inspection):
- Header: "TTB0" + uint32 fields
- 14 * 16B section table at offset header_size (0xB0)
- 3874 * 8B entry table at offset entry_table_off (0x190)
  - (key_hash_u32, record_index_u32)
- 3 language blocks, each:
  - 3874 * 16B records: (marker_or_langcode_u32, str_off_u32, str_len_u32, flags_u32)
    - record 0 marker == 4CC ("jaJP","enUS","zhTW" etc)
    - record 1.. marker == 642 (0x282) in the observed file
  - followed by uint32 terminator == 642
- String pool starts at pool_data_off (header field)
  - Strings are UTF-8 and include trailing NUL (0x00)
  - Offsets are relative to pool_data_off

Export:
  python ttb_tool.py export text.ttb out.csv

Import (updates one language, rebuilds pool, writes new TTB):
  python ttb_tool.py import text.ttb in.csv text_new.ttb --lang jaJP

CSV columns:
  row,key_hash_hex,record_index,
  ja_flags,ja_text,
  en_flags,en_text,
  zh_flags,zh_text

Notes:
- Text is stored without the final NUL in CSV; on import we re-add NUL.
- If a translation cell is empty, original text is kept.
"""
from __future__ import annotations

import argparse
import csv
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


MARKER_DEFAULT = 0x282  # 642 observed
MAGIC = b"TTB0"


def u32(x: int) -> int:
    return x & 0xFFFFFFFF


def fourcc_to_str(v: int) -> str:
    return struct.pack("<I", v).decode("ascii", errors="replace")


def str_to_fourcc(s: str) -> int:
    b = s.encode("ascii")
    if len(b) != 4:
        raise ValueError("Language code must be 4 ASCII chars (e.g. jaJP, enUS, zhTW).")
    return struct.unpack("<I", b)[0]


@dataclass
class TTBHeader:
    header_size: int
    section_count: int
    lang_count: int
    entry_count: int
    bucket_count: int
    section_table_off: int
    entry_table_off: int
    lang_blocks_off: int
    pool_data_off: int
    lang_block_size: int

    @staticmethod
    def parse(buf: bytes) -> "TTBHeader":
        if buf[:4] != MAGIC:
            raise ValueError("Not a TTB0 file.")
        # We only rely on the first 14 uint32s that were meaningful in the sampled file.
        # Layout (little-endian):
        # 0: 'TTB0'
        # 1: header_size
        # 2: section_count
        # 3: lang_count
        # 4: entry_count
        # 5: bucket_count
        # 6: section_table_off
        # 7: entry_table_off
        # 8: lang_blocks_off
        # 9: pool_data_off
        # 13: lang_block_size
        ints = struct.unpack("<4s44I", buf[:4 + 44 * 4])
        magic, *u = ints
        # u has 44 uint32 values: u[0] corresponds to header index 1 in the notes above.
        header_size = u[0]
        section_count = u[1]
        lang_count = u[2]
        entry_count = u[3]
        bucket_count = u[4]
        section_table_off = u[5]
        entry_table_off = u[6]
        lang_blocks_off = u[7]
        pool_data_off = u[8]
        # lang_block_size is at original uint32 index 13 (0-based including magic int),
        # i.e. u[12] because u starts at index 1.
        lang_block_size = u[12]
        return TTBHeader(
            header_size=header_size,
            section_count=section_count,
            lang_count=lang_count,
            entry_count=entry_count,
            bucket_count=bucket_count,
            section_table_off=section_table_off,
            entry_table_off=entry_table_off,
            lang_blocks_off=lang_blocks_off,
            pool_data_off=pool_data_off,
            lang_block_size=lang_block_size,
        )

    def patch_pool_data_off(self, buf: bytearray, new_pool_data_off: int) -> None:
        # pool_data_off is uint32 at header index 9 (after magic)
        struct.pack_into("<I", buf, 4 + 9 * 4, u32(new_pool_data_off))


@dataclass
class LangRecord:
    marker: int
    off: int
    length: int
    flags: int


@dataclass
class TTBFile:
    raw: bytes
    header: TTBHeader
    section_table: bytes  # keep as-is
    entry_table: List[Tuple[int, int]]  # (key_hash, rec_index)
    lang_codes: List[str]
    lang_records: List[List[LangRecord]]  # [lang][rec_index]
    strings: List[List[bytes]]  # [lang][rec_index] raw bytes including NUL

    @staticmethod
    def load(path: Path) -> "TTBFile":
        raw = path.read_bytes()
        if len(raw) < 4 + 44 * 4:
            raise ValueError("File too small.")
        header = TTBHeader.parse(raw)

        # Basic sanity checks
        if header.section_table_off != header.header_size:
            # Not necessarily fatal, but it helps catch wrong assumptions early
            raise ValueError(f"Unexpected section_table_off: {header.section_table_off} (expected header_size {header.header_size})")
        if header.entry_table_off <= header.section_table_off:
            raise ValueError("entry_table_off looks invalid.")
        if header.lang_blocks_off <= header.entry_table_off:
            raise ValueError("lang_blocks_off looks invalid.")
        if header.pool_data_off <= header.lang_blocks_off:
            raise ValueError("pool_data_off looks invalid.")

        # Section table (keep as-is)
        sec_size = header.section_count * 16
        section_table = raw[header.section_table_off: header.section_table_off + sec_size]
        if len(section_table) != sec_size:
            raise ValueError("Section table truncated.")

        # Entry table: entry_count * 8
        ent_size = header.entry_count * 8
        ent_buf = raw[header.entry_table_off: header.entry_table_off + ent_size]
        if len(ent_buf) != ent_size:
            raise ValueError("Entry table truncated.")
        entry_table = [struct.unpack_from("<2I", ent_buf, i * 8) for i in range(header.entry_count)]

        # Language blocks
        lang_codes: List[str] = []
        lang_records: List[List[LangRecord]] = []
        strings: List[List[bytes]] = []

        for li in range(header.lang_count):
            block_off = header.lang_blocks_off + li * header.lang_block_size
            # records
            recs: List[LangRecord] = []
            for i in range(header.entry_count):
                marker, off, length, flags = struct.unpack_from("<4I", raw, block_off + i * 16)
                recs.append(LangRecord(marker=marker, off=off, length=length, flags=flags))
            term = struct.unpack_from("<I", raw, block_off + header.entry_count * 16)[0]
            if term != MARKER_DEFAULT:
                # Some games might use a different terminator; keep reading but warn
                raise ValueError(f"Unexpected language-block terminator: {term} (expected {MARKER_DEFAULT}) at lang#{li}")

            lang_code = fourcc_to_str(recs[0].marker)
            lang_codes.append(lang_code)
            lang_records.append(recs)

            # Extract strings
            lang_strings: List[bytes] = []
            for r in recs:
                start = header.pool_data_off + r.off
                end = start + r.length
                if end > len(raw):
                    raise ValueError(f"String points outside file (lang={lang_code}, off={r.off}, len={r.length})")
                lang_strings.append(raw[start:end])
            strings.append(lang_strings)

        return TTBFile(
            raw=raw,
            header=header,
            section_table=section_table,
            entry_table=entry_table,
            lang_codes=lang_codes,
            lang_records=lang_records,
            strings=strings,
        )

    def decode_text(self, b: bytes) -> str:
        # Remove final NUL if present; keep internal NULs (rare)
        if b.endswith(b"\x00"):
            b = b[:-1]
        return b.decode("utf-8", errors="replace")

    def encode_text(self, s: str) -> bytes:
        return s.encode("utf-8") + b"\x00"


def export_csv(ttb: TTBFile, out_csv: Path) -> None:
    # Identify languages (best-effort by code)
    # We'll always export up to first 3 languages in file order.
    langs = ttb.lang_codes
    # Ensure exactly 3 columns are output for compatibility
    while len(langs) < 3:
        langs.append(f"lang{len(langs)}")

    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "row", "key_hash_hex", "record_index",
            f"{ttb.lang_codes[0]}_flags", f"{ttb.lang_codes[0]}_text",
            f"{ttb.lang_codes[1]}_flags", f"{ttb.lang_codes[1]}_text",
            f"{ttb.lang_codes[2]}_flags", f"{ttb.lang_codes[2]}_text",
        ])
        for row_i, (key_hash, rec_idx) in enumerate(ttb.entry_table):
            cells = [row_i, f"{key_hash:08X}", rec_idx]
            for li in range(min(3, ttb.header.lang_count)):
                r = ttb.lang_records[li][rec_idx]
                text = ttb.decode_text(ttb.strings[li][rec_idx])
                cells.extend([r.flags, text])
            w.writerow(cells)


def import_csv(ttb: TTBFile, in_csv: Path, out_ttb: Path, target_lang: str) -> None:
    if target_lang not in ttb.lang_codes:
        raise ValueError(f"Language '{target_lang}' not in file. Available: {ttb.lang_codes}")
    li = ttb.lang_codes.index(target_lang)

    # Load translations keyed by record_index
    translations: Dict[int, str] = {}
    with in_csv.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        # Find the right column
        text_col = f"{target_lang}_text"
        if text_col not in r.fieldnames:
            raise ValueError(f"CSV missing column: {text_col}")
        idx_col = "record_index"
        for row in r:
            try:
                rec_idx = int(row[idx_col])
            except Exception:
                continue
            new_text = (row.get(text_col) or "").strip("\ufeff")
            if new_text == "":
                continue
            translations[rec_idx] = new_text

    # Build new strings for each language (only replace target language where translation exists)
    new_strings: List[List[bytes]] = []
    for lang_i in range(ttb.header.lang_count):
        lang_list: List[bytes] = []
        for rec_idx in range(ttb.header.entry_count):
            old_b = ttb.strings[lang_i][rec_idx]
            if lang_i == li and rec_idx in translations:
                lang_list.append(ttb.encode_text(translations[rec_idx]))
            else:
                lang_list.append(old_b)
        new_strings.append(lang_list)

    # Rebuild pool in language order: jaJP block strings, then enUS, then zhTW ...
    pool = bytearray()
    new_offsets: List[List[int]] = [[0] * ttb.header.entry_count for _ in range(ttb.header.lang_count)]
    new_lengths: List[List[int]] = [[0] * ttb.header.entry_count for _ in range(ttb.header.lang_count)]

    for lang_i in range(ttb.header.lang_count):
        for rec_idx in range(ttb.header.entry_count):
            new_offsets[lang_i][rec_idx] = len(pool)
            b = new_strings[lang_i][rec_idx]
            new_lengths[lang_i][rec_idx] = len(b)
            pool.extend(b)

    # Construct new file bytes:
    # [header (same 0xB0)] [section_table (same)] [entry_table (same)] [lang_blocks (patched)] [pool]
    header_size = ttb.header.header_size
    sec_size = ttb.header.section_count * 16
    ent_size = ttb.header.entry_count * 8
    lang_blocks_size = ttb.header.lang_count * ttb.header.lang_block_size

    # Keep everything up to the end of the original language blocks identical
    # (we'll patch the language blocks in-place).
    prefix_end = ttb.header.pool_data_off
    prefix = bytearray(ttb.raw[:prefix_end])

    # Patch language blocks in-place (same size)
    for lang_i in range(ttb.header.lang_count):
        block_off = ttb.header.lang_blocks_off + lang_i * ttb.header.lang_block_size
        for rec_idx in range(ttb.header.entry_count):
            # marker stays as-is (lang code in rec0, 0x282 for others)
            marker = ttb.lang_records[lang_i][rec_idx].marker
            flags = ttb.lang_records[lang_i][rec_idx].flags  # keep flags by default
            struct.pack_into("<4I", prefix, block_off + rec_idx * 16,
                             u32(marker),
                             u32(new_offsets[lang_i][rec_idx]),
                             u32(new_lengths[lang_i][rec_idx]),
                             u32(flags))
        # terminator remains as-is (should be 0x282)
        struct.pack_into("<I", prefix, block_off + ttb.header.entry_count * 16, MARKER_DEFAULT)

    # New pool_data_off is right after the language blocks (same as original in this format)
    new_pool_data_off = ttb.header.lang_blocks_off + lang_blocks_size
    # Ensure prefix is at least that long (it should be, since it includes lang blocks)
    if len(prefix) < new_pool_data_off:
        raise RuntimeError("Internal error: prefix shorter than expected.")
    # Patch header pool_data_off
    ttb.header.patch_pool_data_off(prefix, new_pool_data_off)

    new_raw = bytes(prefix[:new_pool_data_off]) + bytes(pool)
    out_ttb.write_bytes(new_raw)


def main() -> None:
    ap = argparse.ArgumentParser(description="Utawarerumono ZAN .ttb import/export tool")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_exp = sub.add_parser("export", help="Export .ttb to CSV")
    ap_exp.add_argument("ttb", type=Path)
    ap_exp.add_argument("csv", type=Path)

    ap_imp = sub.add_parser("import", help="Import CSV and rebuild .ttb")
    ap_imp.add_argument("ttb", type=Path)
    ap_imp.add_argument("csv", type=Path)
    ap_imp.add_argument("out_ttb", type=Path)
    ap_imp.add_argument("--lang", default="jaJP", help="Target language code to replace (default: jaJP)")

    args = ap.parse_args()

    ttb = TTBFile.load(args.ttb)

    if args.cmd == "export":
        export_csv(ttb, args.csv)
        print(f"[OK] Exported: {args.csv}")
        print(f"Languages: {ttb.lang_codes}")
        print(f"Entries: {ttb.header.entry_count}")
    elif args.cmd == "import":
        import_csv(ttb, args.csv, args.out_ttb, args.lang)
        print(f"[OK] Wrote: {args.out_ttb}")
    else:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
