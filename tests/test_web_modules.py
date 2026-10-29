"""UI adapters must preserve data, numeric inference and content isolation."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'app'))
from model_catalog import model_name,candidate_name,NAMES
from mail_catalog import load_catalog,search_catalog,read_mail,mask_contacts
from inference_trace import classify_with_trace,ObservedCalibration,ObservedClassifier
from spam_analysis.inference import LocalSession,record_from_text


def test_all_saved_model_names_are_explicit():
    for path in ['artifacts/sparse_registry.json','artifacts/semantic_registry.json']:
        for key in json.loads(Path(path).read_text(encoding='utf8')):
            assert key in NAMES and len(model_name(key))>6
    for key in json.loads(Path('artifacts/evaluation_plan.json').read_text(encoding='utf8'))['specifications']:
        assert model_name(key)
    assert '多粒度词汇' in candidate_name('S_char13_1_plain')


def test_real_mail_labels_and_bytes_remain_separate():
    catalog=load_catalog()
    assert len(catalog)==64620 and catalog.label.sum()==42854
    chosen=search_catalog(catalog,'train','正常邮件').iloc[0]
    raw,record=read_mail(chosen)
    assert raw and len(record)==1
    assert not {'label','relative_path','template_group_id'} & set(record.columns)
    forged=chosen.copy();forged.raw_sha256='0'*64
    with pytest.raises(ValueError,match='校验'):read_mail(forged)


def test_mail_search_literal_and_contacts_display_only():
    original='联系 a@example.com 或 13800138000 [优惠]'
    rows=pd.DataFrame({'doc_id':['a','b'],'subject_clean':[original,'other'],'split':['train','test'],'label':[1,0]})
    result=search_catalog(rows,'train','垃圾邮件','[优惠]')
    assert len(result)==1
    assert '@' not in mask_contacts(original) and '13800138000' not in mask_contacts(original)
    assert rows.subject_clean.iloc[0]==original
    assert search_catalog(rows,'全部','全部','no-match').empty


def test_observation_delegates_values_exactly():
    class Calibration:
        coef_=np.array([[2.]]);intercept_=np.array([.5])
        def predict_proba(self,raw):return np.column_stack([raw[:,0],1-raw[:,0]])
    trace={};c=Calibration();raw=np.array([[.2]])
    np.testing.assert_array_equal(ObservedCalibration(c,trace).predict_proba(raw),c.predict_proba(raw))
    assert trace['raw_score']==.2
    class Classifier:
        def decision_function(self,x):return x.sum(axis=1)
    x=np.array([[.6,.8]])
    assert ObservedClassifier(Classifier(),trace).decision_function(x)[0]==1.4
    assert trace['vector_dimensions']==2 and trace['vector_norm']==1


def test_trace_matches_real_linear_prediction_and_restores_model():
    engine=LocalSession();record=record_from_text('课程安排','周四下午交作业，请准备课程项目。')
    expected,_=engine.scores(record,'B2');original=engine.loaded['B2']
    trace=classify_with_trace(engine,record,'B2')
    assert trace['probability']==float(expected.score_spam.iloc[0])
    assert engine.loaded['B2'] is original
    assert trace['branches']['B2']['raw_score']==pytest.approx(trace['branches']['B2']['reconstructed_score'],abs=1e-5)
    with pytest.raises(ValueError):classify_with_trace(engine,record.assign(label=0),'B2')


def test_observers_restore_after_inference_failure(monkeypatch):
    engine=LocalSession();engine.load('B2');original=engine.loaded['B2']
    def fail(*args,**kwargs):raise RuntimeError('inference failed')
    monkeypatch.setattr(engine,'scores',fail)
    with pytest.raises(RuntimeError):classify_with_trace(engine,record_from_text('主题','正文'),'B2')
    assert engine.loaded['B2'] is original


def test_empty_trace_uses_prior_without_loading_model():
    engine=LocalSession();trace=classify_with_trace(engine,record_from_text('',''),'E2')
    assert trace['branches']['E2']['prior_only'] and not engine.loaded
    assert trace['probability']==engine.entries['E2']['train_prior']
