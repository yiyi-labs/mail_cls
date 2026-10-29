"""Reproduce the synthetic out-of-domain probe; never a Test accuracy estimate."""
from pathlib import Path
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'app')]
from spam_analysis.inference import LocalSession,record_from_text
from spam_analysis.io import save_json
from inference_trace import classify_with_trace

record=record_from_text('课程项目答疑时间','同学们好，本周四下午安排项目答疑，请准备好代码与问题。')
engine=LocalSession();rows=[]
for name in ['B2','S_char13','E1','E2']:
    original,_=engine.scores(record,name)
    trace=classify_with_trace(engine,record,name)
    assert abs(float(original.score_spam.iloc[0])-trace['probability'])<1e-7
    rows.append(dict(model=name,score=trace['probability'],threshold=trace['threshold'],
                     prediction=trace['prediction'],trace_matches_original=True))
save_json('artifacts/web_synthetic_probe.json',dict(
    scope='one synthetic course notice; author-intended ham; not a formal accuracy estimate',
    subject=record.subject_clean.iloc[0],body=record.body_clean.iloc[0],results=rows))
print('Four real inference routes agree with the observed UI trace; outcomes preserved.')
