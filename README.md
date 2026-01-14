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

### hangul2kanji_ttb_patcher.py
- python hangul2kanji_ttb_patcher.py --ttb text_ori.ttb --csv text.csv --mapping hangul_to_kanji_mapping_2350.csv --out-ttb text_new.ttb



## 어드벤쳐 파트 대사
### advtext_used_inplace_workflow_v3_fixed.py (추출)
- python advtext_used_inplace_workflow_v3_fixed.py extract advtext.cat --scripts advspt_story.cat advspt_proto.cat --out used_advtext_jp.csv --lang jp --step 1


## 토크 파트 대사
### talkspt_jp_workflow.py (추출)
- python talkspt_jp_workflow.py build [원본 cat파일] [추출할 폴더]

### talkspt_jp_workflow_patched.py (패치)
- python talkspt_jp_workflow_patched.py apply talkspt.cat out_talkspt/text_jp.csv out_talkspt/text_jp_diag.csv talkspt_patched.cat

