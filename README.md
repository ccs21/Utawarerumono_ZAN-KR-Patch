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
### advtext_jp_workflow.py
- python advtext_jp_workflow.py build [원본 cat파일] [추출할 폴더]
- python advtext_jp_workflow.py apply [원본 cat파일] [추출 폴더]/text_jp.csv [추출 폴더]/text_jp_diag.csv [저장 될 cat 파일]

### talkspt_jp_workflow.py
- python talkspt_jp_workflow.py build [원본 cat파일] [추출할 폴더]
- python talkspt_jp_workflow.py apply [원본 cat파일] [추출 폴더]/text_jp.csv [추출 폴더]/text_jp_diag.csv [저장 될 cat 파일]
