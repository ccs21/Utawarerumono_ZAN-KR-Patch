# Utawarerumono\_ZAN-KR-Patch

칭송받는자 참! 한글화 패치 시도합니다.


## 폰트
### extract_lds_dds.py 
- python extract_lds_dds.py [폰트LDS파일] [언팩 폴더]

## UI
### inject_lds_dds.py
- python inject_lds_dds.py [원본 폰트LDS파일] [언팩 폴더] [저장될 LDS파일]

### ttb_tool.py
- python ttb_tool.py export [ttb파일] [csv파일]
- python ttb_tool.py import [원본ttb파일] [번역된csv파일] [저장 될 ttb파일] --lang [언어]

## 대사
### advtext_tool_final.py
- python advtext_tool_final.py export [cat파일] [저장 될 csv파일]
- python advtext_tool_final.py import [원본 cat파일] [번역한 csv 파일] [저장 될 cat 파일]

### advtext_ja_pipeline.py
- python advtext_ja_pipeline.py extract [추출된 csv] [일본어만 골라내서 저장 할 csv]
- python advtext_ja_pipeline.py merge [추출된 csv] [일본어만 골라내서 번역한 csv] [패치용 csv]
