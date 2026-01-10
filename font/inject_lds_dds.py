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
    ap.add_argument("original_lds", type=Path)
    ap.add_argument("dds_dir", type=Path, help="folder containing 00.dds, 01.dds ...")
    ap.add_argument("out_lds", type=Path)
    args = ap.parse_args()

    data = bytearray(args.original_lds.read_bytes())
    offs = find_all(data, b"DDS ")
    if not offs:
        raise SystemExit("No DDS signatures found")

    offs2 = offs + [len(data)]

    for i in range(len(offs)):
        a = offs2[i]
        b = offs2[i+1]
        orig_len = b - a

        dds_path = args.dds_dir / f"{i:02d}.dds"
        if not dds_path.exists():
            raise SystemExit(f"Missing {dds_path}")

        new_dds = dds_path.read_bytes()
        if len(new_dds) > orig_len:
            raise SystemExit(f"{dds_path} is larger than original chunk ({len(new_dds)} > {orig_len}). Keep size <= original.")

        # overwrite + pad
        data[a:b] = new_dds + b"\x00" * (orig_len - len(new_dds))

    args.out_lds.write_bytes(data)
    print(f"OK: wrote {args.out_lds}")

if __name__ == "__main__":
    main()
