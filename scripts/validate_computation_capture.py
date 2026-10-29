"""Verify measured activations vary with input and preserve real model scores."""
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'app'),str(ROOT/'src')]
from spam_analysis.inference import record_from_text
from spam_analysis.experiments.runtime import ExperimentSession,active_plan
from spam_analysis.io import save_json,sha256
from inference_trace import classify_with_trace
from mail_catalog import load_catalog,read_mail

engine=ExperimentSession();checks=[];samples=[]
mail=record_from_text('课程项目答疑时间','同学们好，本周四下午安排项目答疑，请准备好代码与问题。')
other=record_from_text('产品优惠活动','本月优惠推广，欢迎了解产品详情及价格。')
for branch in ['E2','E1']:
    expected,_=engine.scores(mail,branch)
    before=engine.loaded[branch]
    trace=classify_with_trace(engine,mail,branch);capture=trace['branches'][branch]['neural_capture']
    changed=classify_with_trace(engine,other,branch)['branches'][branch]['neural_capture']
    assert trace['probability']==float(expected.score_spam.iloc[0])
    assert engine.loaded[branch] is before
    assert capture['kind']=='actual_forward_hooks' and len(capture['frames'])>=37
    assert capture['frames'][-1]['values']!=changed['frames'][-1]['values']
    assert all(np.isfinite(frame['values']).all() for frame in capture['frames'])
    model=before[0][0] if branch=='E2' else before[0][0].model
    assert not any(module._forward_hooks for module in model.modules())
    checks.append({'branch':branch,'frames':len(capture['frames']),'matches_original':True,'input_changes_activations':True,'hooks_removed':True})
    print(branch,checks[-1],flush=True)
catalog=load_catalog()
for label in [0,1]:
    _,record=read_mail(catalog[(catalog.split=='train')&(catalog.label==label)].iloc[0])
    trace=classify_with_trace(engine,record,'E2');detail=trace['branches']['E2']
    samples.append({'split':'train','label':label,'model':'E2','raw_score':detail['raw_score'],
                    'calibrated_logit':detail['calibrated_logit'],'score':trace['probability'],
                    'stable_ham_tail':detail['stable_ham_tail'],'prediction':trace['prediction']})
save_json('artifacts/web_computation_validation.json',{'passed':True,'checks':checks,'train_samples':samples,
          'run_id':engine.run_id,'plan_sha256':sha256(active_plan()),
          'scope':'Two synthetic inputs and first Train mail of each class, read-only inference; no refitting or Test score generation.'})
print(samples,flush=True)
