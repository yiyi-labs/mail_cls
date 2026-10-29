"""Frozen inference must not fall back to a mutable development registry."""
import numpy as np
from spam_analysis import calibration,semantic_models


def test_frozen_entry_reaches_semantic_predictor(monkeypatch):
    entry={'type':'semantic','checkpoint':{'mode':'head_tail'}}
    records=object()
    def predict(name,df,split,entry=None):
        assert name=='E2' and df is records and split=='test'
        assert entry['checkpoint']['mode']=='head_tail'
        return np.array([.4])
    monkeypatch.setattr(semantic_models,'predict_semantic',predict)
    np.testing.assert_array_equal(calibration.raw_scores('E2',entry,records,'test'),[.4])


def test_explicit_semantic_entry_never_reads_registry(monkeypatch):
    entry={'checkpoint':{'mode':'head_tail'}}
    records=object()
    def forbidden(*args,**kwargs):
        raise AssertionError('Mutable registry was read')
    monkeypatch.setattr(semantic_models.Path,'read_text',forbidden)
    def score(actual,df):
        assert actual is entry and df is records
        return np.array([.7])
    monkeypatch.setattr(semantic_models,'macbert_scores',score)
    np.testing.assert_array_equal(semantic_models.predict_semantic('E2',records,'test',entry),[.7])
