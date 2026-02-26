#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
advtext.cat 일본어(한자 단어 포함) 추출/대표선정/번역병합/검증/재빌드 올인원 파이프라인

목표:
- 일본어(가나 포함) + "한자-only 단어/숙어/합성어"는 포함
- "한자-only 문장"은 보류(우선 번역 대상에서 제외)하되, 목록에는 남김
- #@－－#, \\xNN, <...> 태그, %s, <01> 같은 플레이스홀더는 '잠금' 처리:
  번역문(kr)에서 이 토큰들이 변형/삭제되면 병합을 거부(에러)하거나 자동 복구 옵션 제공
- 같은 대사(정규화 기준)끼리 그룹화하고, 가장 긴 원문을 대표(rep)로 선택
- rep만 모은 text_jp_diag.csv를 만들어 번역 대상으로 사용
- kr 번역을 rep에 입력하면 같은 그룹 전체에 동일 번역을 적용(기본)
- 반복 작업 시, 이미 번역된 그룹은 다음 diag에서 자동 제외

사용 예)
1) 원본에서 마스터/후보/대표 생성
   python advtext_jp_workflow.py build advtext.cat out_dir

   out_dir/master.csv          : 전체 레코드/텍스트(원문 기준)
   out_dir/text_jp.csv         : 일본어 후보 + 그룹/타입/rep 여부
   out_dir/text_jp_diag.csv    : 번역 대상(rep만, 보류 제외)

2) text_jp_diag.csv의 kr 채우기(번역)

3) 번역 병합 + advtext.cat 재빌드
   python advtext_jp_workflow.py apply advtext.cat out_dir/text_jp.csv out_dir/text_jp_diag.csv out_dir/advtext_patched.cat

옵션:
- --include-kanji-sentence : 한자-only 문장도 diag에 포함(비추)
- --auto-fix-lock          : 잠금 토큰이 누락/변형된 경우, 원문 토큰을 kr에 자동 삽입하여 복구 시도
"""

from __future__ import annotations
import argparse, csv, re, struct, json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Iterable

BIAS = 112  # header stored offset + 112 = actual section start

# ------------------------
# advtext.cat low-level
# ------------------------

@dataclass
class Section:
    start: int
    end: int
    count: int
    table_off: int
    pool_off: int
    records: List[Tuple[int,int,int,int,int]]  # (id, o1,o2,o3,o4)

def parse_header_section_offsets(buf: bytes) -> List[int]:
    hdr = list(struct.unpack_from("<72I", buf, 0))
    section_count = hdr[26]
    first_index = 33
    starts = []
    for i in range(section_count):
        starts.append(hdr[first_index + i] + BIAS)
    return starts

def parse_sections(buf: bytes) -> List[Section]:
    starts = parse_header_section_offsets(buf)
    starts_sorted = sorted(starts)
    sections: List[Section] = []
    for idx, st in enumerate(starts_sorted):
        en = (starts_sorted[idx+1] if idx+1 < len(starts_sorted) else len(buf))
        if st + 4 > en:
            raise ValueError(f"Section at {st} invalid (too small)")
        count = struct.unpack_from("<I", buf, st)[0]
        table_off = st + 4
        table_size = count * 20
        pool_off = table_off + table_size
        if pool_off > en:
            raise ValueError(f"Section at {st} invalid: pool_off({pool_off}) > end({en})")
        records = []
        for i in range(count):
            idv, o1, o2, o3, o4 = struct.unpack_from("<5I", buf, table_off + i*20)
            records.append((idv, o1, o2, o3, o4))
        sections.append(Section(st, en, count, table_off, pool_off, records))
    return sections

def read_cstr(buf: bytes, off: int) -> bytes:
    end = buf.find(b"\x00", off)
    if end < 0:
        end = len(buf)
    return buf[off:end]

def bytes_to_escaped_text(b: bytes) -> str:
    s = b.decode('utf-8', errors='surrogateescape')
    out = []
    for ch in s:
        o = ord(ch)
        if 0xDC80 <= o <= 0xDCFF:
            out.append(f"\\x{o-0xDC00:02X}")
        else:
            out.append(ch)
    return ''.join(out)

def escaped_text_to_bytes(s: str) -> bytes:
    # inverse of bytes_to_escaped_text, preserving literal backslashes
    out = bytearray()
    i = 0
    while i < len(s):
        if s[i] == "\\" and i+3 < len(s) and s[i+1] == "x":
            hh = s[i+2:i+4]
            try:
                out.append(int(hh, 16))
                i += 4
                continue
            except ValueError:
                pass
        ch = s[i]
        # encode as utf-8
        out.extend(ch.encode("utf-8"))
        i += 1
    return bytes(out)

# ------------------------
# JP detection / grouping
# ------------------------

RE_KANA = re.compile(r"[\u3040-\u309F\u30A0-\u30FF\uFF65-\uFF9F]")
RE_KANJI = re.compile(r"[\u4E00-\u9FFF]")
RE_CJK_PUNCT = re.compile(r"[。、「」、，：；]")
RE_LOCK = re.compile(r"(#@－－#|\\x[0-9A-Fa-f]{2}|<[^>]+>|%[sd]|<\d+>)")

CN_HINT = set(list("的了在是不我你他她它们這这那說说會会來来對对里裡與与為为將将把被讓让並并還还"))


# --- Lock token templating (CSV 안전 번역용) ---
# CSV에서 "완전 잠금"은 불가능하므로, 잠금 토큰을 눈에 띄는 플레이스홀더 {{L000}} 형태로 치환해 보여주고,
# 병합 시 플레이스홀더가 훼손/삭제되면 에러로 막은 다음 원문 토큰으로 되돌려 삽입한다.
RE_LOCK_TOKEN = RE_LOCK  # alias

def _encode_lock_list(tokens: List[str]) -> str:
    # keep UTF-8 safe CSV cell (json)
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
    # fallback: pipe-separated
    return s.split("|")

def tokenize_locks(text: str) -> Tuple[str, List[str]]:
    tokens: List[str] = []
    def repl(m: re.Match) -> str:
        tokens.append(m.group(0))
        return "{{L%03d}}" % (len(tokens)-1)
    tpl = RE_LOCK_TOKEN.sub(repl, text)
    return tpl, tokens

def detokenize_locks(tpl: str, tokens: List[str]) -> str:
    out = tpl
    for i, tok in enumerate(tokens):
        out = out.replace("{{L%03d}}" % i, tok)
    return out

RE_PLACEHOLDER = re.compile(r"\{\{L(\d{3})\}\}")

def validate_placeholders(tpl_src: str, tpl_dst: str) -> Tuple[bool, str]:
    # tpl_dst must contain all placeholders that exist in tpl_src (same multiset and order)
    src_ph = RE_PLACEHOLDER.findall(tpl_src)
    dst_ph = RE_PLACEHOLDER.findall(tpl_dst)
    if src_ph != dst_ph:
        return False, f"placeholders changed: src={src_ph} dst={dst_ph}"
    return True, ""

def classify_text(s: str) -> Tuple[bool, str]:
    """
    Returns (is_candidate, jp_type)
    jp_type:
      - normal         : kana present
      - kanji_term     : kanji-only 'term' (short / name-like)
      - kanji_sentence : kanji-only 'sentence-like' (hold/backlog)
      - other          : not JP candidate
    """
    # strip locks for detection (but keep characters)
    core = RE_LOCK.sub("", s)
    if RE_KANA.search(core):
        return True, "normal"

    # no kana
    if not RE_KANJI.search(core):
        return False, "other"

    # kanji-only-ish: allow punctuation/space/newlines/latin digits
    # Remove common non-kanji
    tmp = re.sub(r"[\s0-9A-Za-z_\-—…・！？!?,.\n\r]", "", core)
    # if what's left is only kanji + a few JP-specific marks
    if re.fullmatch(r"[\u4E00-\u9FFF々ヶヵ〆]+", tmp or ""):
        kanji_len = len(tmp)
        # chinese hint chars present?
        cn_hits = sum(1 for ch in tmp if ch in CN_HINT)
        # sentence-like heuristics
        has_sentence_punct = bool(RE_CJK_PUNCT.search(core))
        if kanji_len >= 12 or has_sentence_punct or cn_hits >= 2:
            return True, "kanji_sentence"
        # term-like
        return True, "kanji_term"

    # mixed CJK without kana: ambiguous. treat as candidate but low priority
    # if mostly kanji and short => treat as term
    kanji_count = len(RE_KANJI.findall(core))
    if kanji_count >= 2 and len(core) <= 10:
        return True, "kanji_term"
    return False, "other"

def normalize_for_group(s: str) -> str:
    # group key: remove locks and whitespace differences
    x = s
    x = x.replace("\r\n", "\n").replace("\r", "\n")
    x = RE_LOCK.sub("", x)
    x = re.sub(r"\s+", " ", x).strip()
    # normalize quotes
    x = x.replace("“", "\"").replace("”", "\"").replace("’", "'")
    return x

def display_len(s: str) -> int:
    x = RE_LOCK.sub("", s)
    x = x.replace("\r\n", "\n").replace("\r", "\n")
    x = re.sub(r"\s+", " ", x).strip()
    return len(x)

def lock_tokens(s: str) -> List[str]:
    return RE_LOCK.findall(s)

def validate_lock_tokens(src: str, dst: str) -> Tuple[bool, List[str]]:
    # all tokens in src must appear in dst (at least same multiset count)
    src_t = lock_tokens(src)
    dst_t = lock_tokens(dst)
    missing = []
    # count
    from collections import Counter
    cs, cd = Counter(src_t), Counter(dst_t)
    for tok, c in cs.items():
        if cd.get(tok, 0) < c:
            missing.append(tok)
    return (len(missing) == 0), missing

def auto_fix_lock(src: str, dst: str) -> str:
    # append missing tokens at the end (safe-ish) if translator removed them
    ok, missing = validate_lock_tokens(src, dst)
    if ok:
        return dst
    # naive but robust: ensure each missing token exists by adding it back
    fixed = dst
    for tok in missing:
        fixed += tok
    return fixed

# ------------------------
# Build / Apply
# ------------------------

def export_master(cat_path: Path, master_csv: Path) -> List[Dict]:
    buf = cat_path.read_bytes()
    sections = parse_sections(buf)
    rows: List[Dict] = []
    with master_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section","index","id_hex","slot","ja"])
        for s_idx, sec in enumerate(sections):
            for i, (idv,o1,o2,o3,o4) in enumerate(sec.records):
                offs = [o1,o2,o3,o4]
                for slot_i, off in enumerate(offs, start=1):
                    b = read_cstr(buf, sec.pool_off + off)
                    ja = bytes_to_escaped_text(b)
                    slot = f"text{slot_i}"
                    w.writerow([s_idx, i, f"{idv:08X}", slot, ja])
                    rows.append({"section":s_idx,"index":i,"id_hex":f"{idv:08X}","slot":slot,"ja":ja})
    return rows

def build_candidates(master_rows: List[Dict], out_jp_csv: Path, out_diag_csv: Path, include_kanji_sentence: bool=False) -> None:
    # filter candidates
    cand = []
    for r in master_rows:
        ja = r["ja"]
        is_cand, jp_type = classify_text(ja)
        if not is_cand:
            continue
        norm = normalize_for_group(ja)
        if not norm:
            continue
        rr = dict(r)
        rr["jp_type"] = jp_type
        rr["group_key"] = norm
        rr["disp_len"] = display_len(ja)
        cand.append(rr)

    # group
    from collections import defaultdict
    groups = defaultdict(list)
    for r in cand:
        groups[r["group_key"]].append(r)

    # pick representative (longest disp_len, tie-break by (slot preference, original order))
    slot_rank = {"text3":0, "text4":1, "text1":2, "text2":3}
    reps = {}
    for k, lst in groups.items():
        lst_sorted = sorted(lst, key=lambda x: (-x["disp_len"], slot_rank.get(x["slot"], 9), x["section"], x["index"]))
        reps[k] = lst_sorted[0]

    # scoring / priority
    def score(rep: Dict, group_size: int) -> int:
        s = 0
        # more frequent => higher
        s += min(group_size, 20) * 10
        # slot preference
        s += { "text3":40, "text4":30, "text1":10, "text2":5 }.get(rep["slot"], 0)
        # type
        s += { "normal":30, "kanji_term":15, "kanji_sentence":0 }.get(rep["jp_type"], 0)
        # length (but do not exclude short)
        s += min(rep["disp_len"], 30)
        return s

    # write text_jp.csv (all candidates)
    with out_jp_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group_key","is_rep","priority","group_size","jp_type","section","index","id_hex","slot","ja","ja_tpl","locks"])
        for k, lst in groups.items():
            rep = reps[k]
            pr = score(rep, len(lst))
            for r in sorted(lst, key=lambda x:(x["section"], x["index"], x["slot"])):
                tpl, toks = tokenize_locks(r["ja"])
                w.writerow([k, "1" if r is rep else "0", pr, len(lst), r["jp_type"], r["section"], r["index"], r["id_hex"], r["slot"], r["ja"], tpl, _encode_lock_list(toks)])

    # write diag (rep only, excluding hold unless flag)
    rep_rows = []
    for k, rep in reps.items():
        if rep["jp_type"] == "kanji_sentence" and not include_kanji_sentence:
            continue
        rep_rows.append((score(rep, len(groups[k])), len(groups[k]), rep))
    rep_rows.sort(key=lambda t:(-t[0], -t[1], t[2]["section"], t[2]["index"]))

    with out_diag_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group_key","priority","group_size","jp_type","rep_section","rep_index","rep_id_hex","rep_slot","ja","ja_tpl","ja_clean","locks","kr_tpl"])
        for pr, gsz, rep in rep_rows:
            tpl, toks = tokenize_locks(rep["ja"])
            ja_clean = RE_PLACEHOLDER.sub("", tpl)
            w.writerow([rep["group_key"], pr, gsz, rep["jp_type"], rep["section"], rep["index"], rep["id_hex"], rep["slot"], rep["ja"], tpl, ja_clean, _encode_lock_list(toks), tpl])

def load_csv_rows(p: Path) -> List[Dict[str,str]]:
    import pandas as pd
    df = pd.read_csv(p, dtype=str).fillna("")
    return df.to_dict(orient="records")


def apply_translations(cat_in: Path, text_jp_csv: Path, diag_csv: Path, cat_out: Path, auto_fix: bool=False) -> None:
    # load candidate mapping: group_key -> list of targets (section,index,slot,ja)
    jp_rows = load_csv_rows(text_jp_csv)
    grp_targets: Dict[str, List[Dict[str,str]]] = {}
    for r in jp_rows:
        grp_targets.setdefault(r["group_key"], []).append(r)

    # load translations from diag
    diag_rows = load_csv_rows(diag_csv)
    trans_group: Dict[str, Tuple[str,str]] = {}  # group_key -> (ja_rep, kr)
    for r in diag_rows:
        gk = r["group_key"]
        ja_rep = r["ja"]
        kr_tpl = r.get("kr_tpl","")
        locks = _decode_lock_list(r.get("locks",""))
        kr = detokenize_locks(kr_tpl, locks)
        if (not kr or str(kr).strip()==""):
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
                raise ValueError(f"[LOCK FAIL] group_key={gk} missing={missing}  -> kr에서 잠금 토큰이 사라졌습니다.")
        trans_group[gk] = (ja_rep, kr)

    if not trans_group:
        raise ValueError("번역(kr)이 채워진 행이 없습니다. diag_csv의 kr 컬럼을 채워주세요.")

    # build replacements per (section,index,slot)
    replace_map: Dict[Tuple[int,int,str], str] = {}
    for gk, (_, kr) in trans_group.items():
        for t in grp_targets.get(gk, []):
            sec = int(t["section"])
            idx = int(t["index"])
            slot = t["slot"]
            src = t["ja"]
            ok, missing = validate_lock_tokens(src, kr)
            if not ok:
                if auto_fix:
                    kr_fix = auto_fix_lock(src, kr)
                    ok2, missing2 = validate_lock_tokens(src, kr_fix)
                    if not ok2:
                        raise ValueError(f"[LOCK FAIL] target(sec={sec},idx={idx},slot={slot}) missing={missing2}")
                    dst = kr_fix
                else:
                    raise ValueError(f"[LOCK FAIL] target(sec={sec},idx={idx},slot={slot}) missing={missing}")
            else:
                dst = kr
            replace_map[(sec, idx, slot)] = dst

    # --- Safe rebuild (same strategy as advtext_tool_final.py) ---
    orig = cat_in.read_bytes()
    sections = parse_sections(orig)

    starts_sorted = sorted(sec.start for sec in sections)
    start_to_sec = {sec.start: sec for sec in sections}

    header_end = min(starts_sorted) if starts_sorted else 288
    header = bytearray(orig[:header_end])

    rebuilt_sections: List[bytes] = []
    new_starts_sorted: List[int] = []
    cursor = header_end

    # build per-record trans table (4 strings) applying replace_map
    # key: (section_index, record_index) -> [text1..text4]
    trans_table: Dict[Tuple[int,int], List[str]] = {}
    for s_idx, sec in enumerate(sections):
        for i, (idv,o1,o2,o3,o4) in enumerate(sec.records):
            offs = [o1,o2,o3,o4]
            texts = []
            for slot_i, off in enumerate(offs, start=1):
                raw = read_cstr(orig, sec.pool_off + off)
                ja = bytes_to_escaped_text(raw)
                slot = f"text{slot_i}"
                key = (s_idx, i, slot)
                if key in replace_map:
                    ja = replace_map[key]
                texts.append(ja)
            trans_table[(s_idx, i)] = texts

    for sec_start in starts_sorted:
        sec = start_to_sec[sec_start]
        s_idx = sections.index(sec)  # stable small list; OK

        new_starts_sorted.append(cursor)

        # gather all used offsets and materialize them as independent strings
        used_offsets = sorted({off for _,off1,off2,off3,off4 in sec.records for off in (off1,off2,off3,off4)})

        pool = bytearray()
        offset_map: Dict[int,int] = {}

        # We must ensure a single offset maps to one string. If multiple records share same offset but have different desired translations -> error.
        # Determine desired text per offset by scanning all usages.
        desired_by_off: Dict[int,str] = {}
        for rec_i, (idv,o1,o2,o3,o4) in enumerate(sec.records):
            texts = trans_table[(s_idx, rec_i)]
            for slot_i, off in enumerate((o1,o2,o3,o4), start=1):
                desired = texts[slot_i-1]
                prev = desired_by_off.get(off)
                if prev is None:
                    desired_by_off[off] = desired
                elif prev != desired:
                    raise ValueError(f"Conflicting replacements for same offset in section {s_idx}: off={off}")

        for off in used_offsets:
            offset_map[off] = len(pool)
            b = escaped_text_to_bytes(desired_by_off[off]) + b"\x00"
            pool += b

        out_sec = bytearray()
        out_sec += struct.pack("<I", sec.count)

        # record table
        for rec_i, (idv,o1,o2,o3,o4) in enumerate(sec.records):
            out_sec += struct.pack("<I", idv)
            for off in (o1,o2,o3,o4):
                out_sec += struct.pack("<I", offset_map[off])

        out_sec += pool

        rebuilt_sections.append(bytes(out_sec))
        cursor += len(out_sec)

    # Update header offsets table (stored = section_start - BIAS) in first 288 bytes.
    if len(header) < 288:
        header += b"\x00" * (288 - len(header))
    hdr = list(struct.unpack_from("<72I", header[:288], 0))
    section_count = hdr[26]
    first_index = 33

    orig_hdr_starts = parse_header_section_offsets(orig)
    orig_to_new = {orig_start: new_start for orig_start, new_start in zip(starts_sorted, new_starts_sorted)}
    for i in range(min(section_count, len(orig_hdr_starts))):
        os_ = orig_hdr_starts[i]
        if os_ in orig_to_new:
            hdr[first_index + i] = orig_to_new[os_] - BIAS
    header[:288] = struct.pack("<72I", *hdr)[:288]

    out = bytes(header[:header_end]) + b"".join(rebuilt_sections)
    cat_out.write_bytes(out)
    print(f"[OK] Patched cat -> {cat_out}  (groups translated: {len(trans_group)})")


def cmd_build(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    master_csv = out_dir / "master.csv"
    jp_csv = out_dir / "text_jp.csv"
    diag_csv = out_dir / "text_jp_diag.csv"
    rows = export_master(Path(args.advtext_cat), master_csv)
    build_candidates(rows, jp_csv, diag_csv, include_kanji_sentence=args.include_kanji_sentence)
    print(f"[OK] build done:\n - {master_csv}\n - {jp_csv}\n - {diag_csv}")

def cmd_apply(args):
    apply_translations(
        Path(args.advtext_cat),
        Path(args.text_jp_csv),
        Path(args.text_jp_diag_csv),
        Path(args.out_cat),
        auto_fix=args.auto_fix_lock
    )

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="advtext.cat -> master/jp/diag csv 생성")
    b.add_argument("advtext_cat")
    b.add_argument("out_dir")
    b.add_argument("--include-kanji-sentence", action="store_true", help="한자-only 문장도 diag에 포함(비추)")
    b.set_defaults(func=cmd_build)

    a = sub.add_parser("apply", help="diag 번역을 group 기준으로 전체에 적용하여 advtext.cat 재빌드")
    a.add_argument("advtext_cat")
    a.add_argument("text_jp_csv")
    a.add_argument("text_jp_diag_csv")
    a.add_argument("out_cat")
    a.add_argument("--auto-fix-lock", action="store_true", help="잠금 토큰 누락 시 자동 복구 시도(끝에 덧붙임)")
    a.set_defaults(func=cmd_apply)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
