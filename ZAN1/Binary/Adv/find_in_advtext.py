# -*- coding: utf-8 -*-
import argparse
from pathlib import Path

def normalize_to_string_start(buf: bytes, off: int) -> int:
    j = off
    while j > 0 and buf[j-1] != 0:
        j -= 1
    return j

def read_c_string(buf: bytes, start: int):
    end = start
    n = len(buf)
    while end < n and buf[end] != 0:
        end += 1
    raw = buf[start:end]
    return raw, end + 1  # end_exclusive

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("advtext", help="advtext.cat path")
    ap.add_argument("query", help="UTF-8 query text, e.g. 今だ or 走れッ")
    args = ap.parse_args()

    data = Path(args.advtext).read_bytes()
    q = args.query.encode("utf-8")

    hits = []
    i = 0
    while True:
        pos = data.find(q, i)
        if pos == -1:
            break
        start = normalize_to_string_start(data, pos)
        raw, end_excl = read_c_string(data, start)
        try:
            s = raw.decode("utf-8")
        except UnicodeDecodeError:
            s = raw.decode("utf-8", errors="replace")
        budget = end_excl - start
        hits.append((start, budget, s))
        i = pos + 1

    if not hits:
        print("[NOT FOUND] query not found in advtext.")
        return

    print(f"[FOUND] {len(hits)} hit(s)")
    for start, budget, s in hits[:50]:
        print(f"- offset=0x{start:X} ({start}) budget={budget-1}bytes text={s}")

if __name__ == "__main__":
    main()
