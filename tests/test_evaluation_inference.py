import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from spam_analysis.evaluation import group_bootstrap
from spam_analysis.inference import record_from_text
from spam_analysis import calibration


def test_group_bootstrap_is_paired_and_reproducible():
    df=pd.DataFrame({'label':[0,0,1,1,0,1], 'template_group_id':['a','a','b','b','c','d']})
    p=np.array([.1,.2,.9,.8,.3,.7])
    specs={name:{'thresholds':{'research':.5}} for name in ['a','b']}
    result=group_bootstrap(df,{'a':p,'b':p.copy()},specs,'a',repeats=50)
    assert result==group_bootstrap(df,{'a':p,'b':p.copy()},specs,'a',repeats=50)
    assert result['models']['b']['main_minus_model_macro_f1_ci95']==[0.,0.]
    assert result['groups']==4


def test_calibrated_empty_input_uses_exact_train_prior(monkeypatch,tmp_path):
    model=LogisticRegression().fit(np.array([[-2.],[-1.],[1.],[2.]]),[0,0,1,1])
    path=tmp_path/'cal.joblib';joblib.dump(model,path)
    def raw_scores(name,entry,records,split):
        assert records.lexical_text.ne('').all()
        return np.full(len(records),2.)
    monkeypatch.setattr(calibration,'raw_scores',raw_scores)
    df=pd.DataFrame({'lexical_text':['','text']})
    p=calibration.calibrated_scores('demo',{'calibrator_path':path,'train_prior':.63},df,'demo')
    assert p[0]==.63 and p[1]!=.63


def test_user_input_does_not_require_labels_or_execute_html():
    df=record_from_text('课程安排','<p>明天开会</p><script>alert(1)</script>')
    assert 'label' not in df and df.doc_id.is_unique
    assert df.body_clean.iloc[0]=='明天开会'
    assert 'alert' not in df.lexical_text.iloc[0]


def test_empty_semantic_input_bypasses_encoder_and_uses_prior():
    from spam_analysis.inference import LocalSession
    session=LocalSession.__new__(LocalSession)
    session.entries={'E1':{'train_prior':.63,'type':'semantic'}}
    def must_not_load(name):raise AssertionError('Empty input must not enter a zero-token encoder')
    session.load=must_not_load
    assert session.branch_scores('E1',record_from_text('','')).tolist()==[.63]
