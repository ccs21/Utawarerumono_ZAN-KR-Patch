#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
talkspt.cat 일본어 추출/대표선정/번역병합/검증/재빌드 올인원 파이프라인

advtext_jp_workflow.py(사용자 업로드본)와 같은 UX로 맞춤:
- build : talkspt.cat에서 문자열 풀을 스캔 → master.csv / text_jp.csv / text_jp_diag.csv 생성
- apply : diag의 kr_tpl을 group 기준으로 전체에 적용 → talkspt_patched.cat 생성

중요:
- talkspt.cat은 advtext.cat과 포맷이 다릅니다.
- 이 스크립트는 "널 종료 UTF-8 문자열"을 안전하게 교체하면서,
  (1) 블록 헤더(data_size) 갱신
  (2) 파일 내부에서 "문자열 시작 오프셋을 가리키는 포인터(u32)"를 위치 추적 후 함께 갱신
  을 수행합니다.

사용 예)
1) 빌드
   python talkspt_jp_workflow.py build talkspt.cat out_dir

2) out_dir/text_jp_diag.csv의 kr_tpl 채우기(번역)

3) 적용/재빌드
   python talkspt_jp_workflow.py apply talkspt.cat out_dir/text_jp.csv out_dir/text_jp_diag.csv out_dir/talkspt_patched.cat

옵션:
- --auto-fix-lock : 잠금 토큰이 누락/변형된 경우, 원문 토큰을 kr에 자동 삽입(끝에 덧붙임)
"""

from __future__ import annotations
import argparse, csv, re, struct, json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# ------------------------
# Lock / JP detection (advtext 스크립트와 동일 계열)
# ------------------------

RE_KANA = re.compile(r"[\u3040-\u309F\u30A0-\u30FF\uFF65-\uFF9F]")
RE_KANJI = re.compile(r"[\u4E00-\u9FFF]")
RE_CJK_PUNCT = re.compile(r"[。、「」、，：；]")
RE_LOCK = re.compile(r"(#@－－#|\\x[0-9A-Fa-f]{2}|<[^>]+>|%[sd]|<\d+>)")
RE_HANGUL = re.compile(r"[가-힣]")  # 한글 음절(치환 대상)

CN_HINT = set(list("的了在是不我你他她它们這这那說说會会來来對对里裡與与為为將将把被讓让並并還还"))

RE_PLACEHOLDER = re.compile(r"\{\{L(\d{3})\}\}")

def _encode_lock_list(tokens: List[str]) -> str:
    return json.dumps(tokens, ensure_ascii=False)

def _decode_lock_list(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return [str(x) for x in v]
    except Exception:
        pass
    return s.split("|")


# ------------------------
# Hangul -> Kanji mapping (폰트용 치환)
# ------------------------
def load_hangul_to_kanji_map(csv_path: Path) -> Dict[str, str]:
    """hangul_to_kanji_mapping_2350.csv 같은 매핑 파일을 로드하여 {한글:한자} dict로 반환."""
    mp: Dict[str, str] = {}
    with csv_path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            h = (row.get('hangul') or '').strip()
            k = (row.get('kanji') or '').strip()
            if not h or not k:
                continue
            # 한 셀에 여러 글자가 들어있는 경우도 방어
            for hh, kk in zip(h, k) if (len(h) == len(k) and len(h) > 1) else [(h, k)]:
                mp[hh] = kk
    if not mp:
        raise ValueError(f"매핑 로드 실패: {csv_path} (hangul/kanji 컬럼 확인)")
    return mp

def apply_hangul_to_kanji(text: str, mp: Optional[Dict[str, str]], unknown: Optional[set]=None) -> str:
    """문장 내 한글 음절을 mp의 한자로 치환. mp에 없는 한글은 그대로 두고 unknown에 기록."""
    if not mp:
        return text
    if not RE_HANGUL.search(text):
        return text
    out_chars: List[str] = []
    for ch in text:
        if '가' <= ch <= '힣':
            rep = mp.get(ch)
            if rep is None:
                if unknown is not None:
                    unknown.add(ch)
                out_chars.append(ch)
            else:
                out_chars.append(rep)
        else:
            out_chars.append(ch)
    return ''.join(out_chars)

def tokenize_locks(text: str) -> Tuple[str, List[str]]:
    tokens: List[str] = []
    def repl(m: re.Match) -> str:
        tokens.append(m.group(0))
        return "{{L%03d}}" % (len(tokens)-1)
    tpl = RE_LOCK.sub(repl, text)
    return tpl, tokens

def detokenize_locks(tpl: str, tokens: List[str]) -> str:
    out = tpl
    for i, tok in enumerate(tokens):
        out = out.replace("{{L%03d}}" % i, tok)
    return out

def validate_placeholders(tpl_src: str, tpl_dst: str) -> Tuple[bool, str]:
    src_ph = RE_PLACEHOLDER.findall(tpl_src)
    dst_ph = RE_PLACEHOLDER.findall(tpl_dst)
    if src_ph != dst_ph:
        return False, f"placeholders changed: src={src_ph} dst={dst_ph}"
    return True, ""

def lock_tokens(s: str) -> List[str]:
    return RE_LOCK.findall(s)

def validate_lock_tokens(src: str, dst: str) -> Tuple[bool, List[str]]:
    from collections import Counter
    src_t = lock_tokens(src)
    dst_t = lock_tokens(dst)
    cs, cd = Counter(src_t), Counter(dst_t)
    missing = []
    for tok, c in cs.items():
        if cd.get(tok, 0) < c:
            missing.append(tok)
    return (len(missing) == 0), missing

def auto_fix_lock(src: str, dst: str) -> str:
    ok, missing = validate_lock_tokens(src, dst)
    if ok:
        return dst
    fixed = dst
    for tok in missing:
        fixed += tok
    return fixed

def classify_text(s: str) -> Tuple[bool, str]:
    core = RE_LOCK.sub("", s)
    if RE_KANA.search(core):
        return True, "normal"
    if not RE_KANJI.search(core):
        return False, "other"

    tmp = re.sub(r"[\s0-9A-Za-z_\-—…・！？!?,.\n\r]", "", core)
    if re.fullmatch(r"[\u4E00-\u9FFF々ヶヵ〆]+", tmp or ""):
        kanji_len = len(tmp)
        cn_hits = sum(1 for ch in tmp if ch in CN_HINT)
        has_sentence_punct = bool(RE_CJK_PUNCT.search(core))
        if kanji_len >= 12 or has_sentence_punct or cn_hits >= 2:
            return True, "kanji_sentence"
        return True, "kanji_term"

    kanji_count = len(RE_KANJI.findall(core))
    if kanji_count >= 2 and len(core) <= 10:
        return True, "kanji_term"
    return False, "other"

def normalize_for_group(s: str) -> str:
    x = s.replace("\r\n", "\n").replace("\r", "\n")
    x = RE_LOCK.sub("", x)
    x = re.sub(r"\s+", " ", x).strip()
    x = x.replace("“", "\"").replace("”", "\"").replace("’", "'")
    return x

def display_len(s: str) -> int:
    x = RE_LOCK.sub("", s)
    x = x.replace("\r\n", "\n").replace("\r", "\n")
    x = re.sub(r"\s+", " ", x).strip()
    return len(x)

# ------------------------
# talkspt.cat: UTF-8 null-terminated string pool scanning
# ------------------------

@dataclass
class StrRec:
    offset: int      # start (file absolute)
    end: int         # end (exclusive, includes trailing 0x00)
    text: str

def _try_read_utf8_cstr(buf: bytes, off: int, maxlen: int = 4096) -> Optional[Tuple[str, int]]:
    end = buf.find(b"\x00", off)
    if end < 0:
        return None
    if end - off <= 0 or end - off > maxlen:
        return None
    b = buf[off:end]
    try:
        s = b.decode("utf-8")
    except Exception:
        return None
    # 너무 이상한 컨트롤 문자 제외
    for ch in s:
        if ord(ch) < 0x09:
            return None
    return s, end + 1

def scan_all_utf8_cstr(buf: bytes) -> List[StrRec]:
    """
    파일 전체에서 (prev == 0x00) 위치를 시작 후보로 보고 UTF-8 C-String을 수집.
    """
    out: List[StrRec] = []
    i = 0
    n = len(buf)
    while i < n:
        if i == 0 or buf[i-1] == 0:
            got = _try_read_utf8_cstr(buf, i)
            if got:
                s, nxt = got
                out.append(StrRec(i, nxt, s))
                i = nxt
                continue
        i += 1
    return out

# ------------------------
# 블록 헤더(32 bytes) 갱신 대상 자동 탐지
# ------------------------

BLOCK_SIG = b"\x01\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x20\x00\x00\x00"  # magic-ish
HDR_SIZE = 0x20

@dataclass
class BlockInfo:
    start: int
    header_size: int
    data_size: int

def find_blocks(buf: bytes) -> List[BlockInfo]:
    """
    talkspt.cat에서 발견되는 블록 헤더들을 찾는다.
    - 현재 샘플에서 0x00, 0x60, 0xD0, 0x1B560, 0x4C720 같은 위치가 검출됨.
    """
    positions = []
    s = 0
    while True:
        i = buf.find(BLOCK_SIG, s)
        if i < 0:
            break
        positions.append(i)
        s = i + 1

    blocks: List[BlockInfo] = []
    for p in positions:
        header_size = struct.unpack_from("<I", buf, p + 0x0C)[0]
        data_size = struct.unpack_from("<I", buf, p + 0x10)[0]
        if header_size != 0x20:
            # 예상과 다르면 일단 스킵(다른 파일 대응)
            continue
        total = header_size + data_size
        if p + total <= len(buf):
            blocks.append(BlockInfo(p, header_size, data_size))
    blocks.sort(key=lambda b: b.start)
    return blocks

# ------------------------
# 포인터(u32) 위치 추적: "문자열 시작 offset"을 가리키는 값만 안전하게 갱신
# ------------------------

def build_pointer_locations(buf: bytes, str_offsets_set: set[int]) -> List[Tuple[int, int]]:
    """
    buf 전체를 4바이트 단위로 스캔하여,
    값이 str_offsets_set에 속하면 (pointer_location, target_old_offset) 기록.
    """
    out = []
    n = len(buf)
    for loc in range(0, n - 4 + 1, 4):
        v = struct.unpack_from("<I", buf, loc)[0]
        if v in str_offsets_set:
            out.append((loc, v))
    return out

# ------------------------
# Rebuild core: apply string replacements with shifting + pointer update + block size update

# ------------------------
# In-place patch mode (NO offset shifting)
# - Keeps original [start,end) spans intact; overwrites bytes and pads with NUL.
# - This is safer when the format contains substring pointers (start+N).
# ------------------------
def normalize_width(text: str) -> str:
    """Normalize ASCII chars that change byte alignment."""
    if text is None:
        return text
    # ASCII space -> full-width space (3 bytes in UTF-8)
    text = text.replace(' ', '　')
    # Three dots -> Japanese ellipsis
    text = text.replace('...', '…')
    # Windows-style newlines to match original exporter (CRLF)
    text = text.replace('\r\n', '\n').replace('\n', '\r\n')
    return text

def apply_replacements_inplace(buf: bytes, repls: list) -> bytes:
    b = bytearray(buf)
    for r in repls:
        span = r.end - r.start
        nb = r.new_bytes
        if len(nb) > span:
            raise ValueError(f'[INPLACE] replacement too long at 0x{r.start:X}: {len(nb)} > {span}')
        # overwrite and pad with NUL
        b[r.start:r.start+len(nb)] = nb
        if len(nb) < span:
            b[r.start+len(nb):r.end] = b'\x00' * (span - len(nb))
    return bytes(b)
# ------------------------

@dataclass
class Replacement:
    start: int
    end: int
    new_bytes: bytes  # includes trailing 0x00

def compute_delta_map(repls: List[Replacement]) -> List[Tuple[int, int]]:
    """
    Returns list of (cutoff_old_end, cumulative_delta) sorted by cutoff_old_end.
    Used for mapping old offsets -> new offsets for positions AFTER each replaced region.
    """
    items = sorted(repls, key=lambda r: r.start)
    cum = 0
    out = []
    for r in items:
        delta = len(r.new_bytes) - (r.end - r.start)
        cum += delta
        out.append((r.end, cum))
    return out

def map_offset(old_off: int, delta_map: List[Tuple[int,int]], repls_by_start: Dict[int, Replacement]) -> int:
    """
    Map an old file offset to new file offset after applying replacements.
    - If old_off is exactly a replaced string start, returns new start (same as mapped start).
    - If old_off lies inside replaced region (not expected for pointers), maps to start.
    - Else add cumulative delta for replacements whose old_end <= old_off.
    """
    if old_off in repls_by_start:
        # new offset is mapped old_off itself + delta before it
        # (delta before start)
        base = old_off
    else:
        base = old_off

    # find cumulative delta for cutoffs <= base
    # linear is fine (few thousand)
    delta = 0
    for cutoff, cum in delta_map:
        if cutoff <= base:
            delta = cum
        else:
            break
    return old_off + delta

def rebuild_with_replacements(
    buf: bytes,
    repls: List[Replacement],
    pointer_locs: List[Tuple[int,int]],
    blocks: List[BlockInfo],
) -> bytes:
    """
    1) 문자열 교체로 새 바이트열 생성
    2) 포인터 위치/값을 새 오프셋으로 갱신
    3) 블록 헤더의 data_size 갱신
    """
    repls_sorted = sorted(repls, key=lambda r: r.start)
    repls_by_start = {r.start: r for r in repls_sorted}
    delta_map = compute_delta_map(repls_sorted)

    # 1) content rebuild
    out = bytearray()
    cursor = 0
    for r in repls_sorted:
        if r.start < cursor:
            raise ValueError(f"Overlapping replacement at 0x{r.start:X}")
        out += buf[cursor:r.start]
        out += r.new_bytes
        cursor = r.end
    out += buf[cursor:]

    # 2) pointer updates (safe list)
    # pointer location itself may have shifted => compute new location
    for old_loc, old_target in pointer_locs:
        new_loc = map_offset(old_loc, delta_map, repls_by_start)
        new_target = map_offset(old_target, delta_map, repls_by_start)
        if new_loc < 0 or new_loc + 4 > len(out):
            continue
        struct.pack_into("<I", out, new_loc, new_target)

    # 3) block header data_size updates
    # We only update blocks we detected in original.
    # new_start = map_offset(old_start, ...)
    for b in blocks:
        old_total = b.header_size + b.data_size
        new_start = map_offset(b.start, delta_map, repls_by_start)
        new_end = map_offset(b.start + old_total, delta_map, repls_by_start)
        new_total = new_end - new_start
        new_data_size = new_total - b.header_size
        if new_start + 0x14 <= len(out):
            struct.pack_into("<I", out, new_start + 0x10, new_data_size)

    # (선택) root block data_size가 파일크기 기반이면 자동으로 맞춰짐(위 루프에 포함됨)
    return bytes(out)

# ------------------------
# CSV I/O
# ------------------------

def load_csv_rows(p: Path) -> List[Dict[str,str]]:
    import pandas as pd
    df = pd.read_csv(p, dtype=str).fillna("")
    return df.to_dict(orient="records")

# ------------------------
# Build / Apply
# ------------------------

def export_master(cat_path: Path, master_csv: Path) -> List[Dict]:
    buf = cat_path.read_bytes()
    strs = scan_all_utf8_cstr(buf)
    rows: List[Dict] = []
    with master_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["offset","end","len_bytes","text"])
        for s in strs:
            w.writerow([f"{s.offset}", f"{s.end}", f"{s.end - s.offset}", s.text])
            rows.append({"offset":str(s.offset), "end":str(s.end), "len_bytes":str(s.end - s.offset), "text":s.text})
    return rows

def build_candidates(master_rows: List[Dict], out_jp_csv: Path, out_diag_csv: Path, include_kanji_sentence: bool=False) -> None:
    cand = []
    for r in master_rows:
        txt = r["text"]
        is_cand, jp_type = classify_text(txt)
        if not is_cand:
            continue
        norm = normalize_for_group(txt)
        if not norm:
            continue
        rr = dict(r)
        rr["jp_type"] = jp_type
        rr["group_key"] = norm
        rr["disp_len"] = display_len(txt)
        cand.append(rr)

    from collections import defaultdict
    groups = defaultdict(list)
    for r in cand:
        groups[r["group_key"]].append(r)

    # rep: longest display_len, tie by earliest offset
    reps = {}
    for k, lst in groups.items():
        lst_sorted = sorted(lst, key=lambda x: (-int(x["disp_len"]), int(x["offset"])))
        reps[k] = lst_sorted[0]

    def score(rep: Dict, group_size: int) -> int:
        s = 0
        s += min(group_size, 20) * 10
        s += { "normal":30, "kanji_term":15, "kanji_sentence":0 }.get(rep["jp_type"], 0)
        s += min(int(rep["disp_len"]), 30)
        return s

    with out_jp_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group_key","is_rep","priority","group_size","jp_type","offset","end","text","ja_tpl","locks"])
        for k, lst in groups.items():
            rep = reps[k]
            pr = score(rep, len(lst))
            for r in sorted(lst, key=lambda x:int(x["offset"])):
                tpl, toks = tokenize_locks(r["text"])
                w.writerow([k, "1" if r is rep else "0", pr, len(lst), r["jp_type"], r["offset"], r["end"], r["text"], tpl, _encode_lock_list(toks)])

    rep_rows = []
    for k, rep in reps.items():
        if rep["jp_type"] == "kanji_sentence" and not include_kanji_sentence:
            continue
        rep_rows.append((score(rep, len(groups[k])), len(groups[k]), rep))
    rep_rows.sort(key=lambda t:(-t[0], -t[1], int(t[2]["offset"])))

    with out_diag_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group_key","priority","group_size","jp_type","rep_offset","rep_end","ja","ja_tpl","ja_clean","locks","kr_tpl"])
        for pr, gsz, rep in rep_rows:
            tpl, toks = tokenize_locks(rep["text"])
            ja_clean = RE_PLACEHOLDER.sub("", tpl)
            w.writerow([rep["group_key"], pr, gsz, rep["jp_type"], rep["offset"], rep["end"], rep["text"], tpl, ja_clean, _encode_lock_list(toks), tpl])

def apply_translations(cat_in: Path, text_jp_csv: Path, diag_csv: Path, cat_out: Path, auto_fix: bool=False, hangul2kanji_csv: Optional[Path]=None, mode: str='inplace', no_width_normalize: bool=False) -> None:
    buf = cat_in.read_bytes()
    strs = scan_all_utf8_cstr(buf)
    # quick lookup by offset
    by_off: Dict[int, StrRec] = {s.offset: s for s in strs}

    # group_key -> targets(offset list)
    jp_rows = load_csv_rows(text_jp_csv)
    grp_targets: Dict[str, List[int]] = {}
    for r in jp_rows:
        gk = r["group_key"]
        off = int(r["offset"])
        grp_targets.setdefault(gk, []).append(off)

    diag_rows = load_csv_rows(diag_csv)
    trans_group: Dict[str, str] = {}
    for r in diag_rows:
        gk = r["group_key"]
        ja_rep = r["ja"]
        kr_tpl = r.get("kr_tpl","")
        locks = _decode_lock_list(r.get("locks",""))
        kr = detokenize_locks(kr_tpl, locks)
        if not kr.strip():
            continue
        ok, missing = validate_lock_tokens(ja_rep, kr)
        if not ok:
            if auto_fix:
                kr2 = auto_fix_lock(ja_rep, kr)
                ok2, missing2 = validate_lock_tokens(ja_rep, kr2)
                if not ok2:
                    raise ValueError(f"[LOCK FAIL] group_key={gk} missing={missing2} (auto-fix failed)")
                kr = kr2
            else:
                raise ValueError(f"[LOCK FAIL] group_key={gk} missing={missing} -> kr에서 잠금 토큰이 사라졌습니다.")
        trans_group[gk] = kr

    if not trans_group:
        raise ValueError("번역(kr_tpl)이 채워진 행이 없습니다. diag_csv의 kr_tpl을 채워주세요.")

    # 한글 -> 한자(폰트용) 치환 매핑 (옵션)
    h2k_map: Optional[Dict[str, str]] = None
    h2k_unknown: set = set()
    if hangul2kanji_csv is not None:
        if hangul2kanji_csv.exists():
            h2k_map = load_hangul_to_kanji_map(hangul2kanji_csv)
            print(f"[INFO] Hangul->Kanji mapping loaded: {hangul2kanji_csv} (entries={len(h2k_map)})")
        else:
            print(f"[WARN] hangul2kanji_csv not found: {hangul2kanji_csv} (치환 없이 진행)")


    # build replacements (per string offset)
    repls: List[Replacement] = []
    used_offsets = set()
    for gk, kr in trans_group.items():
        for off in grp_targets.get(gk, []):
            if off in used_offsets:
                continue
            used_offsets.add(off)
            rec = by_off.get(off)
            if not rec:
                continue
            src = rec.text
            dst = apply_hangul_to_kanji(kr, h2k_map, h2k_unknown)
            ok, missing = validate_lock_tokens(src, dst)
            if not ok:
                if auto_fix:
                    dst2 = auto_fix_lock(src, dst)
                    ok2, missing2 = validate_lock_tokens(src, dst2)
                    if not ok2:
                        raise ValueError(f"[LOCK FAIL] offset=0x{off:X} missing={missing2}")
                    dst = dst2
                else:
                    raise ValueError(f"[LOCK FAIL] offset=0x{off:X} missing={missing}")
            if not no_width_normalize:
                dst = normalize_width(dst)
            new_bytes = dst.encode('utf-8') + b'\x00'
            repls.append(Replacement(start=rec.offset, end=rec.end, new_bytes=new_bytes))

    if h2k_unknown:
        sample = ''.join(sorted(list(h2k_unknown))[:50])
        print(f"[WARN] 매핑에 없는 한글 {len(h2k_unknown)}개가 남아있습니다. (샘플: {sample})")

    # pointer locations that target any string start
    str_offsets_set = set(by_off.keys())
    pointer_locs = build_pointer_locations(buf, str_offsets_set)

    # block headers
    blocks = find_blocks(buf)

    out = rebuild_with_replacements(buf, repls, pointer_locs, blocks)
    cat_out.write_bytes(out)
    print(f"[OK] Patched talkspt.cat -> {cat_out}  (strings patched: {len(repls)})")

# ------------------------
# CLI
# ------------------------

def cmd_build(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    master_csv = out_dir / "master.csv"
    jp_csv = out_dir / "text_jp.csv"
    diag_csv = out_dir / "text_jp_diag.csv"
    rows = export_master(Path(args.talkspt_cat), master_csv)
    build_candidates(rows, jp_csv, diag_csv, include_kanji_sentence=args.include_kanji_sentence)
    print(f"[OK] build done:\n - {master_csv}\n - {jp_csv}\n - {diag_csv}")

def cmd_apply(args):
    # 기본값: CWD에 파일이 없으면 스크립트 폴더에서 다시 찾기
    h2k_csv: Optional[Path] = None
    if not getattr(args, 'no_hangul2kanji', False):
        p = Path(getattr(args, 'hangul2kanji_csv', 'hangul_to_kanji_mapping_2350.csv'))
        if not p.exists():
            p2 = Path(__file__).resolve().parent / p.name
            if p2.exists():
                p = p2
        h2k_csv = p

    apply_translations(
        Path(args.talkspt_cat),
        Path(args.text_jp_csv),
        Path(args.text_jp_diag_csv),
        Path(args.out_cat),
        auto_fix=args.auto_fix_lock,
        hangul2kanji_csv=h2k_csv,
        mode=args.mode,
        no_width_normalize=args.no_width_normalize,
    )

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="talkspt.cat -> master/jp/diag csv 생성")
    b.add_argument("talkspt_cat")
    b.add_argument("out_dir")
    b.add_argument("--include-kanji-sentence", action="store_true", help="한자-only 문장도 diag에 포함(비추)")
    b.set_defaults(func=cmd_build)

    a = sub.add_parser("apply", help="diag 번역을 group 기준으로 전체에 적용하여 talkspt.cat 재빌드")
    a.add_argument("talkspt_cat")
    a.add_argument("text_jp_csv")
    a.add_argument("text_jp_diag_csv")
    a.add_argument("out_cat")
    a.add_argument('--hangul2kanji-csv', default='hangul_to_kanji_mapping_2350.csv',
                   help='한글→한자 매핑 CSV 경로(기본: hangul_to_kanji_mapping_2350.csv). CWD에 없으면 스크립트 폴더에서 재탐색')
    a.add_argument('--no-hangul2kanji', action='store_true', help='한글→한자 치환을 끕니다(디버그/테스트용)')
    a.add_argument("--auto-fix-lock", action="store_true", help="잠금 토큰 누락 시 자동 복구 시도(끝에 덧붙임)")
    a.add_argument("--mode", choices=["shift","inplace"], default="inplace",
                   help="inplace: 기존 문자열 구간을 유지하며 덮어쓰기(안전). shift: 문자열 풀 재배치 + 포인터 업데이트(위험).")
    a.add_argument("--no-width-normalize", action="store_true",
                   help="문장 내 공백/줄바꿈 등의 정규화를 끕니다(기본은 정규화 ON).")
    a.set_defaults(func=cmd_apply)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()