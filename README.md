# Utawarerumono\_ZAN-KR-Patch

칭송받는자 참! 한글화 패치 시도합니다.


## 폰트
### extract_lds_dds.py 
- python extract_lds_dds.py font_00_ori.lds out_font00

### inject_lds_dds.py
- python inject_lds_dds.py font_00_ori.lds out_font00 font_00_nwew.lds

  
## UI
### ttb_tool.py
- python ttb_tool.py export text_ori.ttb text.csv

- ### hangul2kanji_ttb_patcher.py
- python hangul2kanji_ttb_patcher.py --ttb text_ori.ttb --csv text.csv --mapping hangul_to_kanji_mapping_2350.csv --out-ttb text_new.ttb



## 대사
### advtext_jp_workflow.py
- python advtext_jp_workflow.py build [원본 cat파일] [추출할 폴더]
- python advtext_jp_workflow.py apply [원본 cat파일] [추출 폴더]/text_jp.csv [추출 폴더]/text_jp_diag.csv [저장 될 cat 파일]

### talkspt_jp_workflow.py
- python talkspt_jp_workflow.py build [원본 cat파일] [추출할 폴더]
- python talkspt_jp_workflow.py apply [원본 cat파일] [추출 폴더]/text_jp.csv [추출 폴더]/text_jp_diag.csv [저장 될 cat 파일]
