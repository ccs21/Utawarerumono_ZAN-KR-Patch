# extract_dds_from_cat.py
# Windows-only OK. Requires: Python 3.8+
# Usage:
#   python extract_dds_from_cat.py "D:\path\to\logo.cat" -o out
# Optional:
#   python extract_dds_from_cat.py "logo.cat" -o out --min-gap 1024

import argparse
import os
import struct
from typing import List, Tuple

DDS_MAGIC = b"DDS "


def find_all_dds_offsets(fp) -> List[int]:
    """Scan file for 'DDS ' signatures and return offsets."""
    offsets = []
    chunk_size = 8 * 1024 * 1024  # 8MB
    overlap = 3  # to catch magic across boundary
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

        # prepare next overlap
        prev_tail = buf[-overlap:] if len(buf) >= overlap else buf
        pos += len(data)

        if pos >= file_size:
            break

    # de-dup + sort
    offsets = sorted(set(offsets))
    return offsets


def try_read_dds_info(fp, dds_offset: int):
    """
    Read minimal DDS header fields if possible.
    Returns dict or None if header is invalid/unreadable.
    """
    try:
        fp.seek(dds_offset)
        magic = fp.read(4)
        if magic != DDS_MAGIC:
            return None

        header = fp.read(124)
        if len(header) != 124:
            return None

        # DDS_HEADER layout (after magic):
        # dwSize (4) should be 124
        (dwSize,) = struct.unpack_from("<I", header, 0)
        if dwSize != 124:
            # some containers might have false positives; don't hard-fail extraction, just return None
            return None

        (dwHeight,) = struct.unpack_from("<I", header, 8)
        (dwWidth,) = struct.unpack_from("<I", header, 12)

        # DDS_PIXELFORMAT at offset 76 (size 32)
        # ddspf.dwFourCC at offset 84 (76+8)
        fourcc = header[84:88]

        info = {
            "width": dwWidth,
            "height": dwHeight,
            "fourcc": fourcc.decode("latin1", errors="replace"),
        }

        # If DX10 header exists, it starts right after DDS_HEADER (magic+124), i.e. at offset + 128
        # DDS_HEADER says FourCC == 'DX10' when extended header is present.
        if fourcc == b"DX10":
            dx10 = fp.read(20)
            if len(dx10) == 20:
                (dxgi_format,) = struct.unpack_from("<I", dx10, 0)
                info["dxgi_format"] = dxgi_format

        return info
    except Exception:
        return None


def extract_blocks(input_path: str, out_dir: str, min_gap: int = 0) -> List[Tuple[int, int, str]]:
    """
    Extract DDS blocks: from each DDS offset up to next DDS offset (or EOF).
    min_gap: ignore offsets that are too close to previous one (helps against false positives).
    Returns list of (offset, size, out_path).
    """
    os.makedirs(out_dir, exist_ok=True)

    with open(input_path, "rb") as fp:
        offsets = find_all_dds_offsets(fp)
        if not offsets:
            print("[!] No DDS signatures found.")
            return []

        # Optionally filter out suspiciously-close hits
        filtered = []
        last = None
        for off in offsets:
            if last is None or (off - last) >= min_gap:
                filtered.append(off)
                last = off
        offsets = filtered

        fp.seek(0, os.SEEK_END)
        file_size = fp.tell()

        results = []
        for i, off in enumerate(offsets):
            end = offsets[i + 1] if i + 1 < len(offsets) else file_size
            size = end - off
            out_path = os.path.join(out_dir, f"tex_{i:03d}.dds")

            fp.seek(off)
            with open(out_path, "wb") as out:
                # stream copy
                remaining = size
                bufsize = 4 * 1024 * 1024
                while remaining > 0:
                    chunk = fp.read(min(bufsize, remaining))
                    if not chunk:
                        break
                    out.write(chunk)
                    remaining -= len(chunk)

            info = try_read_dds_info(open(out_path, "rb"), 0)
            if info:
                extra = ""
                if "dxgi_format" in info:
                    extra = f", DXGI={info['dxgi_format']}"
                print(f"[OK] #{i:03d} offset={off} size={size} -> {out_path} "
                      f"({info['width']}x{info['height']}, FourCC={info['fourcc']}{extra})")
            else:
                print(f"[OK] #{i:03d} offset={off} size={size} -> {out_path} (header not parsed)")

            results.append((off, size, out_path))

        return results


def main():
    ap = argparse.ArgumentParser(description="Extract DDS textures embedded in .cat by scanning for 'DDS ' signatures.")
    ap.add_argument("input", help="Path to .cat file (e.g. logo.cat)")
    ap.add_argument("-o", "--out", default="out", help="Output folder (default: out)")
    ap.add_argument("--min-gap", type=int, default=0,
                    help="Ignore DDS signatures closer than N bytes apart (useful vs false positives). Default 0.")
    args = ap.parse_args()

    input_path = os.path.abspath(args.input)
    out_dir = os.path.abspath(args.out)

    if not os.path.isfile(input_path):
        print(f"[!] File not found: {input_path}")
        raise SystemExit(1)

    print(f"Input: {input_path}")
    print(f"Out  : {out_dir}")
    extract_blocks(input_path, out_dir, min_gap=args.min_gap)


if __name__ == "__main__":
    main()
