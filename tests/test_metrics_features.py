import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import confusion_matrix,f1_score
from spam_analysis.metrics import metrics,threshold_table,align_scores
from spam_analysis.features import Blocks,NUMERIC_FEATURES

@pytest.mark.parametrize("y,s", [([0,1,0,1],[.1,.5,.5,.9]),([0,0],[.1,.9]),([1,1],[.1,.9]),([0,1],[.5,.5])])
def test_numpy_metrics_and_ties(y,s):
    table=threshold_table(y,s)
    for row in table.itertuples():
        m=metrics(y,s,row.threshold)
        assert [m[x] for x in ["tn","fp","fn","tp"]]==confusion_matrix(y,np.array(s)>=row.threshold,labels=[0,1]).ravel().tolist()
        assert row.macro_f1==pytest.approx(f1_score(y,np.array(s)>=row.threshold,labels=[0,1],average="macro",zero_division=0))

def test_score_alignment():
    a=pd.DataFrame({"doc_id":["a","b"],"label":[0,1]}); b=pd.DataFrame({"doc_id":["b","a"],"score_spam":[.9,.1]})
    assert align_scores(a,b).score_spam.tolist()==[.1,.9]
    with pytest.raises(ValueError): align_scores(a,b.iloc[:1])

def test_vocabulary_train_only_and_allowlist():
    train=pd.DataFrame({"body_clean":["课程 通知","课程 通知"]})
    f=Blocks().fit(train); before=dict(f.transforms_[0][2].vocabulary_)
    f.transform(pd.DataFrame({"body_clean":["保留集合专用新词"]}))
    assert f.transforms_[0][2].vocabulary_==before
    assert not set(NUMERIC_FEATURES)&{"label","index_order","split","template_group_id","parser_defects"}
