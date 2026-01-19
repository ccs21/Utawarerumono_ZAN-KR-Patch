# Utawarerumono\_ZAN-KR-Patch

칭송받는자 참! 한글화 패치 시도합니다.

<img alt="참 한패" src="https://github.com/user-attachments/assets/2419a9d8-acd4-4935-b186-a38023c6f4fe" />

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

### advtext_used_inplace_workflow_v3_fixed.py (패치)
- python advtext_used_inplace_workflow_v3_fixed.py apply advtext.cat used_advtext_jp.csv --mapping hangul_to_kanji_mapping_2350.csv --out advtext_patched.cat

### 추가 추출(미 번역 분)
- python extract_unpatched_used_advtext.py --orig advtext.cat --patched advtext_patched.cat --scripts advspt_story.cat advspt_proto.cat --out used_advtext_unpatched_jp.csv --lang jp --min-chars 1 --step 1

### 추가 추출분 패치
- python advtext_used_inplace_workflow_v3_fixed.py apply advtext_patched.cat used_advtext_unpatched_jp.csv --mapping hangul_to_kanji_mapping_2350.csv --out advtext_patched_v2.cat




## 토크 파트 대사
### talkspt_jp_workflow.py (추출)
- python talkspt_jp_workflow.py build [원본 cat파일] [추출할 폴더]

### talkspt_jp_workflow_inplace_strict_ok_v3.py (패치)
- python talkspt_jp_workflow_inplace_strict_ok_v3.py apply talkspt.cat out_talkspt\text_jp.csv out_talkspt\text_jp_diag.csv talkspt_patched.cat --only-normal


