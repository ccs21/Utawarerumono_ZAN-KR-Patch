# scan_mid_refs.py
import struct, sys, pathlib

start = int(sys.argv[1], 16)          # e.g. 0x1B5D4
budget = int(sys.argv[2])            # e.g. 71
paths = [pathlib.Path(p) for p in sys.argv[3:]]

lo = start
hi = start + budget  # exclusive

hits = []
for p in paths:
    b = p.read_bytes()
    for i in range(0, len(b) - 4):
        v = struct.unpack_from("<I", b, i)[0]
        if lo < v < hi:  # strictly inside (mid-string)
            hits.append((p.as_posix(), i, v))

print(f"range: [{hex(lo)}, {hex(hi)}) mid-hits={len(hits)}")
for p, pos, v in hits[:200]:
    print(f"{p}: pos=0x{pos:X} -> off={hex(v)}")
if len(hits) > 200:
    print(f"... ({len(hits)-200} more)")
