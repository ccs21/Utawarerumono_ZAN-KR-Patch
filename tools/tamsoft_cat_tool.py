#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tamsoft_cat_tool.py
- Tamsoft 계열 .cat 아카이브 언팩/리팩(안전 모드)
- 포맷 문서화가 없으므로, 파일 내부에서 "오프셋/사이즈 테이블" 후보를 자동 탐색해서 블록을 추출합니다.
- 리팩은 원본 CAT에 "동일 블록 개수 + 원본보다 크지 않게" 덮어쓰는 안전 모드입니다.

사용 예)
  python tamsoft_cat_tool.py unpack  texture_extend.cat out_font
  python tamsoft_cat_tool.py repack texture_extend.cat out_font texture_extend_patched.cat
"""

from __future__ import annotations

import argparse
import json
import os
import struct
from pathlib import Path
from typing import List, Tuple, Optional, Dict

U32 = struct.Struct("<I")

MAGIC_EXTS = [
    (b"DDS ", ".dds"),
    (b"\x89PNG\r\n\x1a\n", ".png"),
    (b"OggS", ".ogg"),
    (b"RIFF", ".wav"),
    (b"TTB", ".ttb"),  # (정확한 시그니처는 게임별 상이할 수 있음)
]

def read_u32(data: bytes, off: int) -> int:
    return U32.unpack_from(data, off)[0]

def is_reasonable_offset(x: int, fsize: int) -> bool:
    return 0 <= x < fsize

def guess_ext(blob: bytes) -> str:
    for sig, ext in MAGIC_EXTS:
        if blob.startswith(sig):
            return ext
    # 텍스트/바이너리 구분 힌트
    if blob[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return ".txt"
    return ".bin"

def parse_global_header(data: bytes) -> Dict[str, int]:
    """
    너가 올린 샘플들에서 공통으로 보이는 32바이트 헤더:
      u32[0]=1, u32[1]=1, u32[2]=0
      u32[3]=global_header_size (보통 32 또는 256)
      u32[4]=tail_offset (대개 파일 끝 - global_header_size)
      나머지는 게임별/파일별로 의미가 다름
    """
    if len(data) < 32:
        raise ValueError("File too small")

    a0 = read_u32(data, 0)
    a1 = read_u32(data, 4)
    hdr_size = read_u32(data, 12)
    tail_off = read_u32(data, 16)

    return {
        "a0": a0,
        "a1": a1,
        "global_header_size": hdr_size,
        "tail_offset": tail_off,
        "file_size": len(data),
    }

def find_table_candidate(data: bytes, count: int, fsize: int, start: int, end: int) -> Optional[Tuple[int, List[int], List[int]]]:
    """
    오프셋/사이즈 테이블 후보 찾기:
      - 어떤 위치 pos에서 u32[count]를 offsets, 이어서 u32[count]를 sizes라고 가정
      - offsets는 증가(또는 비감소)하고, 각 offset+size가 파일 범위 내여야 함
      - offsets[0]는 대체로 start보다 크고, end보다 작아야 함
    반환: (pos, offsets, sizes)
    """
    need = count * 8  # offsets(count*4) + sizes(count*4)
    if end - start < need:
        return None

    best = None
    best_score = None

    # 4바이트 정렬 기준으로 스캔
    for pos in range(start, end - need + 1, 4):
        offsets = [read_u32(data, pos + i * 4) for i in range(count)]
        sizes   = [read_u32(data, pos + count * 4 + i * 4) for i in range(count)]

        # 기본 검증
        if not all(is_reasonable_offset(o, fsize) for o in offsets):
            continue
        if any(s <= 0 or s > fsize for s in sizes):
            continue

        # 오프셋 증가 성질(완전 엄격 X, 일부는 동일 시작도 가능)
        ok_inc = True
        for i in range(1, count):
            if offsets[i] < offsets[i - 1]:
                ok_inc = False
                break
        if not ok_inc:
            continue

        # 범위 체크
        ok_range = True
        for o, s in zip(offsets, sizes):
            if o + s > fsize:
                ok_range = False
                break
        if not ok_range:
            continue

        # 너무 앞쪽(헤더 안) / 너무 뒤(테일)쪽 배제
        if offsets[0] < start or offsets[0] >= end:
            continue

        # 스코어: 테이블이 “헤더 근처에 있고”, 데이터가 “겹치지 않는” 후보 선호
        overlap_pen = 0
        for i in range(1, count):
            prev_end = offsets[i - 1] + sizes[i - 1]
            if offsets[i] < prev_end:
                overlap_pen += (prev_end - offsets[i])

        # 테이블 위치가 앞일수록 가산점
        score = overlap_pen + (pos - start) // 4

        if best_score is None or score < best_score:
            best = (pos, offsets, sizes)
            best_score = score

    return best

def locate_blocks(data: bytes, meta: Dict[str, int]) -> Tuple[int, List[int], List[int]]:
    """
    내부 블록 찾기:
    - 일부 파일은 global_header_size가 32/256
    - tail_offset은 보통 파일끝-헤더사이즈. 테일 구간은 건드리지 않는 편이 안전.
    - count는 내부 헤더(대개 global_header_size 위치)에 들어있음:
        internal_base = global_header_size
        count = u32[internal_base + 4]
    - 그 후 오프셋/사이즈 테이블을 자동 탐색
    """
    ghs = meta["global_header_size"]
    tail_off = meta["tail_offset"]
    fsize = meta["file_size"]

    if ghs >= fsize:
        raise ValueError("global_header_size looks wrong")

    internal_base = ghs
    if internal_base + 8 > fsize:
        raise ValueError("No space for internal header")

    count = read_u32(data, internal_base + 4)
    if count <= 0 or count > 100000:
        raise ValueError(f"Unreasonable block count: {count}")

    # 테이블 탐색 범위: internal_base ~ tail_off
    # 너무 넓으면 오래걸리니, 우선 internal_base+0x10 ~ internal_base+0x4000 정도를 먼저 보고,
    # 실패하면 tail_off 근처까지 확대.
    scan_starts = [
        (internal_base + 0x10, min(tail_off, internal_base + 0x4000)),
        (internal_base + 0x10, tail_off),
    ]

    cand = None
    for s, e in scan_starts:
        cand = find_table_candidate(data, count, fsize, s, e)
        if cand:
            break

    if not cand:
        raise ValueError("Failed to locate offset/size table automatically.")

    table_pos, offsets, sizes = cand
    return table_pos, offsets, sizes

def unpack(cat_path: Path, out_dir: Path) -> None:
    data = cat_path.read_bytes()
    meta = parse_global_header(data)

    table_pos, offsets, sizes = locate_blocks(data, meta)

    out_dir.mkdir(parents=True, exist_ok=True)
    blobs_dir = out_dir / "blobs"
    blobs_dir.mkdir(parents=True, exist_ok=True)

    items = []
    for i, (o, s) in enumerate(zip(offsets, sizes)):
        blob = data[o:o+s]
        ext = guess_ext(blob)
        name = f"{i:04d}{ext}"
        (blobs_dir / name).write_bytes(blob)
        items.append({
            "index": i,
            "offset": o,
            "size": s,
            "file_hint": name,
        })

    manifest = {
        "source_cat": str(cat_path),
        "global": meta,
        "detected": {
            "internal_base": meta["global_header_size"],
            "count": len(items),
            "table_pos": table_pos,
            "table_layout": "offsets[u32]*count + sizes[u32]*count",
        },
        "items": items,
        "notes": [
            "repack은 안전 모드입니다: 각 blob 크기는 원본 size를 넘을 수 없습니다.",
            "원본보다 작으면 나머지는 0x00으로 패딩됩니다.",
        ],
    }

    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Unpacked {len(items)} blocks -> {blobs_dir}")
    print(f"[OK] Manifest -> {out_dir/'manifest.json'}")

def repack(original_cat: Path, unpack_dir: Path, out_cat: Path) -> None:
    data = bytearray(original_cat.read_bytes())
    manifest_path = unpack_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    blobs_dir = unpack_dir / "blobs"

    meta = manifest["global"]
    count = manifest["detected"]["count"]
    table_pos = manifest["detected"]["table_pos"]
    fsize = len(data)

    # 원본 offsets/sizes를 다시 읽어서 “정확히 동일” 기준으로 리팩
    offsets = [read_u32(data, table_pos + i*4) for i in range(count)]
    sizes   = [read_u32(data, table_pos + count*4 + i*4) for i in range(count)]

    for i in range(count):
        # blobs/0000.* 중 하나를 찾는다
        # manifest의 file_hint 확장자를 그대로 따라감
        hint = manifest["items"][i]["file_hint"]
        src_file = blobs_dir / hint
        if not src_file.exists():
            # 혹시 확장자가 바뀐 경우 대비: 0000.* 매칭
            candidates = list(blobs_dir.glob(f"{i:04d}.*"))
            if not candidates:
                raise FileNotFoundError(f"Missing blob for index {i}: expected {hint} or {i:04d}.*")
            src_file = candidates[0]

        new_blob = src_file.read_bytes()
        orig_size = sizes[i]
        orig_off = offsets[i]

        if len(new_blob) > orig_size:
            raise ValueError(
                f"Blob {i} too large: {len(new_blob)} > original {orig_size}. "
                f"(safe repack only)"
            )
        if orig_off + orig_size > fsize:
            raise ValueError(f"Original range out of file: idx={i} off={orig_off} size={orig_size}")

        # 덮어쓰기 + 패딩
        data[orig_off:orig_off+orig_size] = new_blob + (b"\x00" * (orig_size - len(new_blob)))

        # sizes 테이블은 “원본 유지”가 가장 안전하지만,
        # 엔진이 실제 size를 참조하는 케이스도 있어서, "실제 길이"로 줄이는 건 허용.
        # (늘리기는 금지)
        new_size = len(new_blob)
        U32.pack_into(data, table_pos + count*4 + i*4, new_size)

    out_cat.write_bytes(data)
    print(f"[OK] Repacked -> {out_cat}")

def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)

    ap_u = sp.add_parser("unpack", help="unpack .cat to folder")
    ap_u.add_argument("cat", type=Path)
    ap_u.add_argument("out_dir", type=Path)

    ap_r = sp.add_parser("repack", help="repack folder back into .cat (safe overwrite mode)")
    ap_r.add_argument("original_cat", type=Path)
    ap_r.add_argument("unpack_dir", type=Path)
    ap_r.add_argument("out_cat", type=Path)

    args = ap.parse_args()

    if args.cmd == "unpack":
        unpack(args.cat, args.out_dir)
        return 0
    if args.cmd == "repack":
        repack(args.original_cat, args.unpack_dir, args.out_cat)
        return 0

    return 2

if __name__ == "__main__":
    raise SystemExit(main())
