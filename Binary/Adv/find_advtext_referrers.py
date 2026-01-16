# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import struct

def u32le(n: int) -> bytes:
    return struct.pack("<I", n & 0xFFFFFFFF)

def iter_cat_files(root: Path):
    for p in root.rglob("*.cat"):
        # advtext 본체/패치본은 제외 (참조자만 찾고 싶으니까)
        name = p.name.lower()
        if "advtext" in name:
            continue
        yield p

def find_offsets_in_file(buf: bytes, pat: bytes):
    pos_list = []
    i = 0
    while True:
        j = buf.find(pat, i)
        if j == -1:
            break
        pos_list.append(j)
        i = j + 1
    return pos_list

def main():
    ap = argparse.ArgumentParser(
        description="Scan *.cat under a directory and find which files reference given advtext offsets (u32 LE)."
    )
    ap.add_argument("--dir", required=True, help="Directory to scan (e.g., Binary/Adv)")
    ap.add_argument("--off", nargs="+", required=True, help="Offsets to search. Accept hex(0x...) or decimal.")
    ap.add_argument("--top", type=int, default=50, help="Show top N results")
    args = ap.parse_args()

    root = Path(args.dir).resolve()
    if not root.exists():
        raise SystemExit(f"[ERR] dir not found: {root}")

    offsets = []
    for s in args.off:
        s = s.strip()
        if s.lower().startswith("0x"):
            offsets.append(int(s, 16))
        else:
            offsets.append(int(s))

    patterns = [(off, u32le(off)) for off in offsets]

    results = []  # (hit_count_total, file, detail_str)
    for f in iter_cat_files(root):
        try:
            data = f.read_bytes()
        except Exception:
            continue

        hit_total = 0
        detail = []
        for off, pat in patterns:
            hits = find_offsets_in_file(data, pat)
            if hits:
                hit_total += len(hits)
                # 파일 내에서 패턴이 나온 위치 몇 개만 표시
                sample = ", ".join([f"0x{x:X}" for x in hits[:6]])
                if len(hits) > 6:
                    sample += f" (+{len(hits)-6} more)"
                detail.append(f"off=0x{off:X} hits={len(hits)} at {sample}")

        if hit_total:
            results.append((hit_total, f, " | ".join(detail)))

    results.sort(key=lambda x: x[0], reverse=True)

    if not results:
        print("[NOT FOUND] No referrers found for given offsets.")
        return

    print(f"[OK] Found {len(results)} file(s) referencing given offsets.")
    print("---- TOP RESULTS ----")
    for i, (cnt, f, d) in enumerate(results[:args.top], 1):
        rel = f.relative_to(root)
        print(f"{i:02d}. hits_total={cnt:4d} file={rel}")
        print(f"    {d}")

if __name__ == "__main__":
    main()
