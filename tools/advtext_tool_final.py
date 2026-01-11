#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utawarerumono ZAN (Tamsoft) - advtext.cat exporter/importer (safe rebuild)

왜 "safe"인가?
- advtext.cat 안의 레코드 오프셋이 '문자열 시작'이 아니라 "문자열 중간(서브스트링)"을 가리키는 경우가 있습니다.
- 그래서 "풀 전체를 그대로 유지하면서 특정 문자열만 교체" 방식은 오프셋 보정이 매우 까다롭습니다.
- 이 툴은 **레코드에서 실제로 참조하는 모든 오프셋(중간 오프셋 포함)을 각각 독립 문자열로 '실체화(materialize)'** 해서
  새 풀을 다시 만드는 방식이라, 중간 오프셋도 문제없이 유지됩니다.
- 그 결과, 파일 크기는 바뀔 수 있지만(대개 줄어듦/늘어날 수 있음) **헤더의 섹션 오프셋 테이블을 같이 갱신**하므로
  게임은 새 파일을 정상적으로 따라갈 수 있습니다.

Export:
  python advtext_tool_final.py export advtext.cat advtext.csv

Import:
  python advtext_tool_final.py import advtext.cat advtext.csv advtext_patched.cat

CSV columns:
  section, index, id_hex, off1, off2, off3, off4, text1, text2, text3, text4

텍스트 인코딩:
- 기본은 UTF-8.
- UTF-8로 해석 불가한 바이트(0x80/0x81 등 제어코드)는 \\xNN 형태로 이스케이프해서 CSV로 내보냅니다.
- 임포트 시 \\xNN을 다시 원래 바이트로 되돌립니다.
"""
from __future__ import annotations
import argparse
import csv
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

BIAS = 112  # header stored offset + 112 = actual section start
HEX_RE = re.compile(r'\\x([0-9a-fA-F]{2})')

def bytes_to_escaped_text(b: bytes) -> str:
    s = b.decode('utf-8', errors='surrogateescape')
    out = []
    for ch in s:
        o = ord(ch)
        if 0xDC80 <= o <= 0xDCFF:
            out.append(f"\\x{o-0xDC00:02X}")
        else:
            out.append(ch)
    return ''.join(out)

def escaped_text_to_bytes(s: str) -> bytes:
    # Preserve literal backslashes
    s = s.replace('\\\\', '\u0000')
    s = s.replace('\\n', '\n').replace('\\r', '\r').replace('\\t', '\t')
    s = s.replace('\u0000', '\\\\')

    parts: List[bytes] = []
    i = 0
    while i < len(s):
        m = HEX_RE.search(s, i)
        if not m:
            parts.append(s[i:].encode('utf-8'))
            break
        if m.start() > i:
            parts.append(s[i:m.start()].encode('utf-8'))
        parts.append(bytes([int(m.group(1), 16)]))
        i = m.end()
    return b''.join(parts)

def read_cstr(buf: bytes, abs_off: int) -> bytes:
    end = buf.find(b'\x00', abs_off)
    if end < 0:
        raise ValueError(f"NUL terminator not found from {abs_off}")
    return buf[abs_off:end]

@dataclass
class Section:
    start: int
    end: int
    count: int
    table_off: int
    pool_off: int
    records: List[Tuple[int,int,int,int,int]]  # (id, off1..off4)

def parse_header_section_offsets(buf: bytes) -> List[int]:
    if len(buf) < 288:
        raise ValueError("File too small to be advtext.cat")
    hdr = struct.unpack_from("<72I", buf, 0)
    section_count = hdr[26]
    first_index = 33
    offs = []
    for i in range(section_count):
        stored = hdr[first_index + i]
        offs.append(stored + BIAS)
    # Deduplicate while keeping header order
    seen = set()
    ordered = []
    for o in offs:
        if 0 <= o < len(buf) and o not in seen:
            seen.add(o)
            ordered.append(o)
    return ordered

def parse_sections(buf: bytes) -> List[Section]:
    starts = parse_header_section_offsets(buf)
    starts_sorted = sorted(starts)
    spans = []
    for i, st in enumerate(starts_sorted):
        en = starts_sorted[i+1] if i+1 < len(starts_sorted) else len(buf)
        spans.append((st, en))

    out: List[Section] = []
    for st, en in spans:
        if en - st < 4:
            continue
        count = struct.unpack_from("<I", buf, st)[0]
        table_off = st + 4
        table_size = count * 20
        pool_off = table_off + table_size
        if pool_off > en:
            raise ValueError(f"Section at {st} invalid: pool_off({pool_off}) > end({en})")
        records = []
        for i in range(count):
            idv, o1, o2, o3, o4 = struct.unpack_from("<5I", buf, table_off + i*20)
            records.append((idv, o1, o2, o3, o4))
        out.append(Section(st, en, count, table_off, pool_off, records))
    return out

def export_csv(inp: Path, out_csv: Path) -> None:
    buf = inp.read_bytes()
    sections = parse_sections(buf)
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section","index","id_hex","off1","off2","off3","off4","text1","text2","text3","text4"])
        for s_idx, sec in enumerate(sections):
            for i, (idv,o1,o2,o3,o4) in enumerate(sec.records):
                texts = []
                for off in (o1,o2,o3,o4):
                    b = read_cstr(buf, sec.pool_off + off)
                    texts.append(bytes_to_escaped_text(b))
                w.writerow([s_idx, i, f"{idv:08X}", o1, o2, o3, o4, *texts])
    print(f"[OK] Exported {sum(s.count for s in sections)} records ({len(sections)} sections) -> {out_csv}")

def import_csv(inp: Path, csv_path: Path, out_path: Path) -> None:
    orig = inp.read_bytes()
    sections = parse_sections(orig)

    # Load translations
    trans: Dict[Tuple[int,int], List[str]] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            s_idx = int(row["section"])
            idx = int(row["index"])
            trans[(s_idx, idx)] = [row["text1"], row["text2"], row["text3"], row["text4"]]

    starts_sorted = sorted(sec.start for sec in sections)
    start_to_sec = {sec.start: sec for sec in sections}

    header_end = min(starts_sorted) if starts_sorted else 288
    header = bytearray(orig[:header_end])

    rebuilt_sections: List[bytes] = []
    new_starts_sorted: List[int] = []
    cursor = header_end

    for s_idx, st in enumerate(starts_sorted):
        sec = start_to_sec[st]
        new_starts_sorted.append(cursor)

        # Collect ALL offsets actually referenced by records (includes "mid-string" offsets).
        used_offsets = sorted({off for _,o1,o2,o3,o4 in sec.records for off in (o1,o2,o3,o4)})

        # Read original strings at each used offset
        bytes_by_old_off: Dict[int, bytes] = {}
        for off in used_offsets:
            bytes_by_old_off[off] = read_cstr(orig, sec.pool_off + off)

        # Apply translations per record-field (shared offset => shared string)
        for i, (idv,o1,o2,o3,o4) in enumerate(sec.records):
            key = (s_idx, i)
            if key not in trans:
                continue
            t1,t2,t3,t4 = trans[key]
            for off_key, t in zip((o1,o2,o3,o4), (t1,t2,t3,t4)):
                bytes_by_old_off[off_key] = escaped_text_to_bytes(t)

        # Build new pool by "materializing" every used offset as its own string.
        new_pool = bytearray()
        new_off_map: Dict[int,int] = {}
        for old_off in used_offsets:
            new_off_map[old_off] = len(new_pool)
            new_pool += bytes_by_old_off[old_off] + b"\x00"

        # Rebuild section
        out_sec = bytearray()
        out_sec += struct.pack("<I", sec.count)
        for (idv,o1,o2,o3,o4) in sec.records:
            out_sec += struct.pack("<5I", idv, new_off_map[o1], new_off_map[o2], new_off_map[o3], new_off_map[o4])
        out_sec += new_pool

        rebuilt_sections.append(bytes(out_sec))
        cursor += len(out_sec)

    # Update header offsets table (stored = section_start - BIAS) in first 288 bytes.
    if len(header) < 288:
        header += b"\x00" * (288 - len(header))
    hdr = list(struct.unpack_from("<72I", header[:288], 0))
    section_count = hdr[26]
    first_index = 33

    orig_hdr_starts = parse_header_section_offsets(orig)
    orig_to_new = {orig_start: new_start for orig_start, new_start in zip(starts_sorted, new_starts_sorted)}
    for i in range(min(section_count, len(orig_hdr_starts))):
        os_ = orig_hdr_starts[i]
        if os_ in orig_to_new:
            hdr[first_index + i] = orig_to_new[os_] - BIAS
    header[:288] = struct.pack("<72I", *hdr)[:288]

    out = bytes(header[:header_end]) + b"".join(rebuilt_sections)
    out_path.write_bytes(out)
    print(f"[OK] Rebuilt {len(rebuilt_sections)} sections -> {out_path}")
    print("     NOTE: Section offset table updated; IDs/order preserved; pool re-materialized by used offsets.")

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="export to CSV")
    e.add_argument("inp", type=Path)
    e.add_argument("csv", type=Path)
    im = sub.add_parser("import", help="import from CSV and rebuild")
    im.add_argument("inp", type=Path)
    im.add_argument("csv", type=Path)
    im.add_argument("out", type=Path)
    args = ap.parse_args()
    if args.cmd == "export":
        export_csv(args.inp, args.csv)
    else:
        import_csv(args.inp, args.csv, args.out)

if __name__ == "__main__":
    main()
