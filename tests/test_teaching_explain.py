import numpy as np
import pandas as pd
from spam_analysis.teaching import tfidf_lesson,threshold_benchmark
from spam_analysis.features import Blocks,SparseArtifact
from spam_analysis.sparse_models import estimator
from spam_analysis.explain import linear_contributions

def test_teaching_math():
    assert tfidf_lesson()['sklearn_equal']
    assert threshold_benchmark(n=100)['correct']

def test_prediction_no_label_and_contribution_save_load(tmp_path):
    import joblib
    df=pd.DataFrame({'lexical_text':['课程 通知','课程 作业','免费 优惠','免费 促销']*2})
    f=Blocks(baseline=True).fit(df); clf=estimator('lr',1).fit(f.transform(df),[0,0,1,1]*2)
    model=SparseArtifact(f,clf,.5); out,detail=linear_contributions(model,df.iloc[:1])
    assert detail['actual_score']<0
    p=tmp_path/'model.joblib'; joblib.dump(model,p)
    np.testing.assert_allclose(model.decision_function(df),joblib.load(p).decision_function(df))
