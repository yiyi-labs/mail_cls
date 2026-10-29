import numpy as np
import pandas as pd
from spam_analysis.parsing import parse_email
from spam_analysis.policy import operating_points
from spam_analysis.robustness import perturb

def test_policy_fpr_constraint_and_all_mail():
    y=np.array([0,0,0,1,1]); s=np.array([.1,.3,.8,.7,.9])
    points,table=operating_points(y,s,[.005,.01])
    for key in ['fpr_0.005','fpr_0.01']:
        assert ((s>=points[key])&(y==0)).sum()==0
    assert ((s>=points['fpr_0.005'])&(y==1)).sum()==1

def test_format_perturbations_preserve_visible_text():
    r=parse_email('Subject: =?utf-8?B?6K++56iL6YCa55+l?=\nContent-Type: text/plain; charset=utf-8\n\n这是课程通知。\n请于周五参加。'.encode())
    df=pd.DataFrame([{'doc_id':'example',**r}])
    for rule in ['fullwidth','zero_width','html_wrapper','mime_base64']:
        out,applied=perturb(df,rule)
        assert out.lexical_text.iloc[0]==df.lexical_text.iloc[0]
        assert len(out)==1


def test_safe_runner_up_is_not_hidden_by_unsafe_highest_score():
    from spam_analysis.policy import choose_main
    results=pd.DataFrame({'macro_f1':[.98,.995,.991],'ham_fpr':[0.,.01,0.]},index=['S','E2','F_0.5'])
    chosen,audit=choose_main(results,['S','E2','F_0.5'],'S')
    assert chosen=='F_0.5' and not audit.loc['E2','eligible']


def test_small_gain_keeps_sparse_and_eligible_tie_prefers_simple_model():
    from spam_analysis.policy import choose_main
    results=pd.DataFrame({'macro_f1':[.98,.9801,.9801],'ham_fpr':[0.,0.,0.]},index=['S','E2','F_0.5'])
    assert choose_main(results,['S','E2','F_0.5'],'S')[0]=='S'
    results.loc[['E2','F_0.5'],'macro_f1']=.99
    assert choose_main(results,['S','E2','F_0.5'],'S')[0]=='E2'
