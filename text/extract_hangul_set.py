# extract_hangul_set.py
# 사용법:
#   python extract_hangul_set.py text.csv
#   python extract_hangul_set.py text.ttb --out hangul_chars.txt --csv hangul_chars.csv
#
# 결과:
#  - hangul_chars.txt : 한글 유니크 글자들을 한 줄로 출력(복사/붙여넣기 용)
#  - hangul_chars.csv : 글자 + 등장횟수 (정렬: 많이 등장한 글자부터)

import argparse
from collections import Counter
from pathlib import Path

def is_hangul_syllable(ch: str) -> bool:
    o = ord(ch)
    return 0xAC00 <= o <= 0xD7A3  # 가~힣

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_file", help="번역본 파일 (csv/ttb/그 외 아무거나)")
    ap.add_argument("--out", default="hangul_chars.txt", help="유니크 한글 리스트 txt 출력")
    ap.add_argument("--csv", default="hangul_chars.csv", help="한글/빈도 csv 출력")
    ap.add_argument("--encoding", default="utf-8", help="디코딩 시도 인코딩 (기본 utf-8)")
    args = ap.parse_args()

    in_path = Path(args.input_file)

    data = in_path.read_bytes()

    # UTF-8 기준으로 최대한 문자열 복원 (깨지는 바이트는 무시)
    text = data.decode(args.encoding, errors="ignore")

    # 한글 음절만 추출
    hangul = [ch for ch in text if is_hangul_syllable(ch)]
    freq = Counter(hangul)

    uniq_sorted_by_freq = [ch for ch, _ in freq.most_common()]  # 등장 많은 순
    uniq_count = len(uniq_sorted_by_freq)
    total_count = len(hangul)

    # 1) txt: 유니크 한글만 한 줄로
    out_txt = Path(args.out)
    out_txt.write_text("".join(uniq_sorted_by_freq), encoding="utf-8")

    # 2) csv: 글자 + 등장횟수
    out_csv = Path(args.csv)
    lines = ["hangul,count"]
    for ch, c in freq.most_common():
        lines.append(f"{ch},{c}")
    out_csv.write_text("\n".join(lines), encoding="utf-8")

    print(f"[OK] input: {in_path.name}")
    print(f"  total hangul occurrences: {total_count}")
    print(f"  unique hangul syllables : {uniq_count}")
    print(f"[WRITE] {out_txt} (copy/paste용)")
    print(f"[WRITE] {out_csv} (빈도표)")

if __name__ == "__main__":
    main()
