import argparse, os, struct

DDS_MAGIC = b"DDS "

def find_all_dds_offsets(data: bytes):
    offs = []
    start = 0
    while True:
        i = data.find(DDS_MAGIC, start)
        if i == -1:
            break
        offs.append(i)
        start = i + 1
    return sorted(set(offs))

def parse_dds_dx10_info(dds: bytes):
    """
    Returns (w, h, mipCount, dxgiFormat)
    Requires DX10 header (FourCC=DX10).
    """
    if len(dds) < 148 or dds[:4] != DDS_MAGIC:
        raise ValueError("Not a DDS file (missing 'DDS ').")

    hdr = dds[4:4+124]
    # width/height/mipmaps
    h = struct.unpack_from("<I", hdr, 8)[0]
    w = struct.unpack_from("<I", hdr, 12)[0]
    mip = struct.unpack_from("<I", hdr, 24)[0]
    if mip == 0:
        mip = 1

    # DDS_PIXELFORMAT.dwFourCC is at offset 80 within DDS_HEADER
    fourcc = hdr[80:84]
    if fourcc != b"DX10":
        raise ValueError("DDS is not DX10 (FourCC != DX10). This patcher expects DX10+BC7 DDS.")

    dx10 = dds[4+124:4+124+20]
    dxgi = struct.unpack_from("<I", dx10, 0)[0]
    return w, h, mip, dxgi

def expected_bc7_dx10_dds_size(w, h, mip):
    """
    Compute exact byte size for a DX10 BC7 DDS containing mip levels.
    Header size: 4 + 124 + 20 = 148 bytes.
    Each BC7 block is 16 bytes, block is 4x4 pixels.
    """
    total = 148
    for level in range(mip):
        mw = max(1, w >> level)
        mh = max(1, h >> level)
        bw = (mw + 3) // 4
        bh = (mh + 3) // 4
        total += bw * bh * 16
    return total

def replace_dds_inside_block_keep_trailer(cat: bytes, offsets, index: int, new_dds: bytes):
    start = offsets[index]
    end = offsets[index+1] if index+1 < len(offsets) else len(cat)
    block = cat[start:end]

    # Old DDS spec
    ow, oh, omip, odxgi = parse_dds_dx10_info(block)
    if odxgi != 98:
        raise ValueError(f"Old DDS DXGI format is {odxgi}, not 98 (BC7_UNORM).")

    old_dds_len = expected_bc7_dx10_dds_size(ow, oh, omip)
    if len(block) < old_dds_len:
        raise ValueError(f"Block is smaller than expected DDS length. block={len(block)} expected_dds={old_dds_len}")

    # New DDS spec must match old (w/h/mip/dxgi)
    nw, nh, nmip, ndxgi = parse_dds_dx10_info(new_dds)
    if (nw, nh, nmip, ndxgi) != (ow, oh, omip, odxgi):
        raise ValueError(
            f"New DDS spec mismatch for index {index}!\n"
            f"  OLD: {ow}x{oh} mip={omip} dxgi={odxgi}\n"
            f"  NEW: {nw}x{nh} mip={nmip} dxgi={ndxgi}\n"
            f"Hint: re-export with texconv: -f BC7_UNORM -m 1"
        )

    new_expected_len = expected_bc7_dx10_dds_size(nw, nh, nmip)
    if len(new_dds) != new_expected_len:
        raise ValueError(
            f"New DDS byte size unexpected for index {index}.\n"
            f"  Expected(from header): {new_expected_len}\n"
            f"  Actual file size      : {len(new_dds)}\n"
            f"Hint: re-export with texconv: -f BC7_UNORM -m 1"
        )

    trailer = block[old_dds_len:]  # keep tail bytes
    patched_block = new_dds + trailer
    if len(patched_block) != len(block):
        raise ValueError("Patched block size changed (should not happen in keep-trailer mode).")

    return cat[:start] + patched_block + cat[end:]

def main():
    ap = argparse.ArgumentParser(description="Patch two DX10 BC7 DDS blocks inside a .cat file, keeping trailer bytes intact.")
    ap.add_argument("cat", help="Input .cat file (e.g. top_menu.cat)")
    ap.add_argument("--out", required=True, help="Output patched .cat file")
    ap.add_argument("--i29", type=int, default=29, help="DDS block index for tex_029 (default: 29)")
    ap.add_argument("--i30", type=int, default=30, help="DDS block index for tex_030 (default: 30)")
    ap.add_argument("--dds29", required=True, help="Edited tex_029 DDS path (BC7_UNORM, mip=1)")
    ap.add_argument("--dds30", required=True, help="Edited tex_030 DDS path (BC7_UNORM, mip=1)")
    args = ap.parse_args()

    cat = open(args.cat, "rb").read()
    offsets = find_all_dds_offsets(cat)
    if len(offsets) == 0:
        raise SystemExit("[!] No DDS signatures found in cat.")

    if args.i29 >= len(offsets) or args.i30 >= len(offsets):
        raise SystemExit(f"[!] Index out of range. Found DDS blocks: {len(offsets)}")

    dds29 = open(args.dds29, "rb").read()
    dds30 = open(args.dds30, "rb").read()

    # Replace 29 then 30 (sizes preserved; safe)
    cat2 = replace_dds_inside_block_keep_trailer(cat, offsets, args.i29, dds29)
    cat3 = replace_dds_inside_block_keep_trailer(cat2, offsets, args.i30, dds30)

    with open(args.out, "wb") as f:
        f.write(cat3)

    print("[OK] Patched:", args.out)
    print(f"     Replaced index {args.i29} and {args.i30} (kept trailer bytes).")

if __name__ == "__main__":
    main()
