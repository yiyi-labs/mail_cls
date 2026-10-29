"""Validate real tokenizer inputs and every row in the Train-only charts."""
from pathlib import Path
import sys
import os
import json
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'app'),str(ROOT/'src')]
os.chdir(ROOT)
from spam_analysis.inference import LocalSession,record_from_text
from spam_analysis.io import sha256
from inference_trace import classify_with_trace
from text_analysis import text_payload
from data_analysis import analysis_records,grouped_distribution,FIELDS

frozen=[Path('artifacts/evaluation_plan.json'),Path('artifacts/test_results.csv'),Path('artifacts/policy_selection.json')]
frozen+=list(Path('artifacts/predictions').glob('test_*.parquet'))
before={str(p):sha256(p) for p in frozen}
engine=LocalSession();record=record_from_text('中文课程通知','请在星期四提交课程项目与代码。')
checks=[]
for branch in ['E2','E1','S_char13']:
    expected,_=engine.scores(record,branch)
    trace=classify_with_trace(engine,record,branch)
    payload=text_payload(engine,record,branch,trace)
    assert trace['probability']==float(expected.score_spam.iloc[0])
    if branch in ('E2','E1'):
        item=payload['branches'][0]
        assert item['verified'] and item['input_ids']==trace['branches'][branch]['model_input_ids']
        checks.append({'branch':branch,'actual_input_ids_match':True,'tokens':item['total'],'algorithm':item['algorithm'],'inference_unchanged':True})
    else:checks.append({'branch':branch,'feature_blocks':len(payload['branches']),'inference_unchanged':True})
frame=analysis_records();statistics=[]
assert len(frame)==36158
for field in FIELDS:
    table=grouped_distribution(frame,field)
    assert table['邮件数'].sum()==len(frame)
    assert all(abs(v-1)<1e-12 for v in table.groupby('类别')['类内比例'].sum())
    statistics.append({'field':field,'rows_accounted':int(table['邮件数'].sum()),'class_rates_sum_to_one':True})
assert before=={str(p):sha256(p) for p in frozen}
Path('artifacts/text_analysis_validation.json').write_text(json.dumps({'checked_at':datetime.now(timezone.utc).isoformat(),'passed':True,'inference':checks,'training_statistics':statistics,'frozen_evidence_unchanged':True},ensure_ascii=False,indent=2),encoding='utf-8')
print('Actual token IDs, feature blocks, 4 Train distributions and frozen evidence verified')
