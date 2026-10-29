"""Metrics use spam=1 and predict spam exactly when score >= threshold."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss

def safe_div(a,b):
    return np.divide(a,b,out=np.zeros(np.broadcast_shapes(np.shape(a),np.shape(b)),dtype=float),where=np.asarray(b)!=0)

def counts(y,pred,weights=None):
    y=np.asarray(y); pred=np.asarray(pred); w=np.ones(len(y)) if weights is None else np.asarray(weights)
    return {"tn":float(w[(y==0)&(pred==0)].sum()),"fp":float(w[(y==0)&(pred==1)].sum()),
            "fn":float(w[(y==1)&(pred==0)].sum()),"tp":float(w[(y==1)&(pred==1)].sum())}

def from_counts(tn,fp,fn,tp):
    sp=float(safe_div(tp,tp+fp)); sr=float(safe_div(tp,tp+fn)); sf=float(safe_div(2*tp,2*tp+fp+fn))
    hp=float(safe_div(tn,tn+fn)); hr=float(safe_div(tn,tn+fp)); hf=float(safe_div(2*tn,2*tn+fp+fn))
    return dict(accuracy=float(safe_div(tp+tn,tp+tn+fp+fn)),spam_precision=sp,spam_recall=sr,spam_f1=sf,
                ham_precision=hp,ham_recall=hr,ham_f1=hf,macro_f1=(sf+hf)/2,balanced_accuracy=(sr+hr)/2,
                ham_fpr=float(safe_div(fp,fp+tn)),spam_miss_rate=float(safe_div(fn,tp+fn)))

def metrics(y,scores,threshold=0.5,weights=None,probability=True):
    y=np.asarray(y); scores=np.asarray(scores,dtype=float)
    if len(y)!=len(scores) or not np.isfinite(scores).all(): raise ValueError("Invalid or nonfinite scores")
    c=counts(y,scores>=threshold,weights); out={**c,**from_counts(**c),"n":len(y),"threshold":float(threshold)}
    out["average_precision"]=float(average_precision_score(y,scores,sample_weight=weights)) if (y==1).any() else None
    out["roc_auc"]=float(roc_auc_score(y,scores,sample_weight=weights)) if len(np.unique(y))==2 else None
    out["brier"]=float(brier_score_loss(y,scores,sample_weight=weights)) if probability else None
    return out

def threshold_table(y,scores):
    y=np.asarray(y,dtype=int); scores=np.asarray(scores,dtype=float)
    if len(y)==0 or len(y)!=len(scores) or not np.isfinite(scores).all(): raise ValueError("Invalid threshold data")
    order=np.argsort(-scores,kind="stable"); s=scores[order]; yy=y[order]
    ends=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(s)-1]
    tp=np.cumsum(yy)[ends]; fp=ends+1-tp; fn=y.sum()-tp; tn=(len(y)-y.sum())-fp
    # Include the all-ham endpoint without non-standard JSON infinity.
    thresholds=np.r_[np.nextafter(s[0],np.inf),s[ends]]
    tp=np.r_[0,tp]; fp=np.r_[0,fp]; tn=np.r_[len(y)-y.sum(),tn]; fn=np.r_[y.sum(),fn]
    table=pd.DataFrame(dict(threshold=thresholds,tn=tn,fp=fp,fn=fn,tp=tp))
    table["macro_f1"]=(safe_div(2*tp,2*tp+fp+fn)+safe_div(2*tn,2*tn+fp+fn))/2
    table["ham_fpr"]=safe_div(fp,fp+tn); table["spam_recall"]=safe_div(tp,tp+fn)
    return table

def align_scores(records,predictions):
    if not records.doc_id.is_unique or not predictions.doc_id.is_unique: raise ValueError("Duplicate doc_id")
    if set(records.doc_id)!=set(predictions.doc_id): raise ValueError("Missing or extra scores")
    return records.merge(predictions,on="doc_id",validate="one_to_one",sort=False)
