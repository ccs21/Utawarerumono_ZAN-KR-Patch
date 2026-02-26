from pathlib import Path

def find_all(data: bytes, sig: bytes) -> list[int]:
    offs = []
    i = 0
    while True:
        j = data.find(sig, i)
        if j == -1:
            break
        offs.append(j)
        i = j + 1
    return offs

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("lds", type=Path)
    ap.add_argument("out_dir", type=Path)
    args = ap.parse_args()

    data = args.lds.read_bytes()
    offs = find_all(data, b"DDS ")
    if not offs:
        raise SystemExit("No DDS signatures found")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    offs2 = offs + [len(data)]

    for i in range(len(offs)):
        a = offs2[i]
        b = offs2[i+1]
        chunk = data[a:b]
        if len(chunk) < 256:
            continue
        (args.out_dir / f"{i:02d}.dds").write_bytes(chunk)

    print(f"OK: extracted {len(offs)} DDS -> {args.out_dir}")

if __name__ == "__main__":
    main()
