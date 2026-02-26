# inject_dds_into_cat.py
# Usage:
#   python inject_dds_into_cat.py "D:\path\to\logo.cat" --index 2 --dds "D:\edited\tex_002.dds" -o "logo_patched.cat"
#
# Default: STRICT (same byte length only) => safest
# If you REALLY want to allow size change (risk!), add: --allow-resize

import argparse
import os
import struct
from typing import List, Tuple

DDS_MAGIC = b"DDS "

def find_all_dds_offsets(fp) -> List[int]:
    offsets = []
    chunk_size = 8 * 1024 * 1024
    overlap = 3
    pos = 0

    fp.seek(0, os.SEEK_END)
    file_size = fp.tell()
    fp.seek(0)

    prev_tail = b""
    while True:
        data = fp.read(chunk_size)
        if not data:
            break

        buf = prev_tail + data
        start = 0
        while True:
            idx = buf.find(DDS_MAGIC, start)
            if idx == -1:
                break
            off = pos - len(prev_tail) + idx
            offsets.append(off)
            start = idx + 1

        prev_tail = buf[-overlap:] if len(buf) >= overlap else buf
        pos += len(data)
        if pos >= file_size:
            break

    return sorted(set(offsets))

def read_dds_brief(dds_bytes: bytes):
    # returns (w,h,fourcc,dxgi_format_or_None) or None if invalid
    if len(dds_bytes) < 4 + 124:
        return None
    if dds_bytes[:4] != DDS_MAGIC:
        return None
    header = dds_bytes[4:4+124]
    dwSize = struct.unpack_from("<I", header, 0)[0]
    if dwSize != 124:
        return None
    h = struct.unpack_from("<I", header, 8)[0]
    w = struct.unpack_from("<I", header, 12)[0]
    fourcc = header[84:88]
    dxgi = None
    if fourcc == b"DX10" and len(dds_bytes) >= 4 + 124 + 20:
        dx10 = dds_bytes[4+124:4+124+20]
        dxgi = struct.unpack_from("<I", dx10, 0)[0]
    return (w, h, fourcc, dxgi)

def main():
    ap = argparse.ArgumentParser(description="Replace a DDS block inside .cat by scanning for 'DDS ' signatures.")
    ap.add_argument("cat", help="Path to .cat file")
    ap.add_argument("--index", type=int, required=True, help="Which DDS block index to replace (0-based)")
    ap.add_argument("--dds", required=True, help="Edited .dds file path to insert")
    ap.add_argument("-o", "--out", required=True, help="Output patched .cat path")
    ap.add_argument("--allow-resize", action="store_true",
                    help="Allow different byte length (RISK: may break if container uses internal offsets)")
    args = ap.parse_args()

    cat_path = os.path.abspath(args.cat)
    dds_path = os.path.abspath(args.dds)
    out_path = os.path.abspath(args.out)

    if not os.path.isfile(cat_path):
        raise SystemExit(f"[!] cat not found: {cat_path}")
    if not os.path.isfile(dds_path):
        raise SystemExit(f"[!] dds not found: {dds_path}")

    with open(cat_path, "rb") as fp:
        offsets = find_all_dds_offsets(fp)
        if not offsets:
            raise SystemExit("[!] No DDS signatures found in cat.")
        if args.index < 0 or args.index >= len(offsets):
            raise SystemExit(f"[!] index out of range. found={len(offsets)}")

        fp.seek(0, os.SEEK_END)
        file_size = fp.tell()

        i = args.index
        start = offsets[i]
        end = offsets[i+1] if i+1 < len(offsets) else file_size
        old_size = end - start

        fp.seek(0)
        cat_bytes = fp.read()

    new_dds = open(dds_path, "rb").read()
    old_dds = cat_bytes[start:end]

    # basic sanity: both should look like DDS
    old_info = read_dds_brief(old_dds)
    new_info = read_dds_brief(new_dds)
    if not old_info:
        print("[!] Warning: old block header not parsed (still proceeding).")
    if not new_info:
        raise SystemExit("[!] New file doesn't look like a valid DDS (missing DDS header?)")

    if old_info:
        ow, oh, ofourcc, odxgi = old_info
        nw, nh, nfourcc, ndxgi = new_info
        if (ow, oh) != (nw, nh) or ofourcc != nfourcc or odxgi != ndxgi:
            print("[!] WARNING: DDS format mismatch!")
            print(f"    OLD: {ow}x{oh} fourcc={ofourcc} dxgi={odxgi}")
            print(f"    NEW: {nw}x{nh} fourcc={nfourcc} dxgi={ndxgi}")
            print("    -> This may crash or render wrong if the engine expects exact format.")

    if len(new_dds) != old_size:
        if not args.allow_resize:
            raise SystemExit(
                f"[!] Size mismatch: old_block={old_size} bytes, new_dds={len(new_dds)} bytes.\n"
                f"    Re-export DDS to EXACT same byte size, or rerun with --allow-resize (RISK)."
            )
        else:
            print("[!] allow-resize enabled: offsets after this block will shift (RISK).")

    # build patched bytes
    patched = cat_bytes[:start] + new_dds + cat_bytes[end:]

    # write output
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as out:
        out.write(patched)

    print("[OK] Patched written:", out_path)
    print(f"     Replaced index {i} at [{start}, {end}) old={old_size} new={len(new_dds)}")

if __name__ == "__main__":
    main()
