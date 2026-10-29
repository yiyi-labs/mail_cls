"""Finite-budget controlled sparse experiments; no Test access."""
import gc
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from .features import Blocks,SparseArtifact,NBRatio
from .io import load_records,save_json
from .metrics import metrics

def estimator(kind,param,seed=42,balanced=False):
    if kind=="nb": return MultinomialNB(alpha=param)
    if kind=="lr": return LogisticRegression(C=param,solver="liblinear",max_iter=1000,random_state=seed,class_weight="balanced" if balanced else None)
    return LinearSVC(C=param,dual="auto",max_iter=5000,random_state=seed,class_weight="balanced" if balanced else None)

def rank(m): return (m["macro_f1"],-m["ham_fpr"])

def train_sparse(config):
    if Path("artifacts/evaluation_plan.json").exists(): raise RuntimeError("Test plan already frozen")
    train=load_records("train"); tune=load_records("tune")
    Path("artifacts/models").mkdir(parents=True,exist_ok=True); Path("artifacts/predictions").mkdir(parents=True,exist_ok=True)
    y=train.label.to_numpy(); yt=tune.label.to_numpy(); prior=float(y.mean()); records=[]; registry={}
    jobs=[
        ("B",dict(baseline=True),[("B1","nb",a,False) for a in [.1,.5,1]]+[("B2","lr",c,False) for c in [.5,1,2,4]]),
        ("body_word",{},[("body_word","svm",c,False) for c in [.5,1,2]]),
        ("char",dict(word=False,char=True),[("char","svm",c,False) for c in [.5,1,2]]),
        ("word_char",dict(char=True),[("word_char","svm",c,False) for c in [.5,1,2]]),
        ("S",dict(char=True,subject=True),[("S","svm",c,False) for c in [.5,1,2,4]]+[("S","svm",1,True)]),
        ("S_structure",dict(char=True,subject=True,numeric=True),[("S_structure","svm",1,False)]),
        ("S_char13",dict(char=True,subject=True,char_range=(1,3)),[("S_char13","svm",1,False)]),
        ("naive",dict(baseline=True,naive=True),[("naive","lr",1,False)]),
        ("N",dict(baseline=True),[("N","svm",c,False) for c in [.5,1]])]
    majority=int(prior>=.5)
    dummy=metrics(yt,np.full(len(yt),prior),.5)
    save_json("artifacts/dummy_tune.json",dummy)
    for feature_name,feature_kwargs,candidates in jobs:
        print(f"Fitting {feature_name} features on Train only ({len(train)} rows)",flush=True)
        t0=time.perf_counter(); features=Blocks(**feature_kwargs).fit(train)
        X=features.transform(train); V=features.transform(tune); feature_s=time.perf_counter()-t0
        ratio=None
        if feature_name=="N": ratio=NBRatio().fit(X,y); X=ratio.transform(X); V=ratio.transform(V)
        for family,kind,param,balanced in candidates:
            cid=f"{family}_{param}_{'balanced' if balanced else 'plain'}"
            t0=time.perf_counter()
            with threadpool_limits(limits=8): clf=estimator(kind,param,config["seed"],balanced).fit(X,y)
            fit_s=time.perf_counter()-t0; t0=time.perf_counter()
            if hasattr(clf,"decision_function"): score=clf.decision_function(V)
            else:
                lp=clf.predict_log_proba(V); score=lp[:,1]-lp[:,0]
            score[tune.lexical_text.eq("").to_numpy()]=np.log(prior/(1-prior))
            predict_s=time.perf_counter()-t0; m=metrics(yt,score,0,probability=False)
            r=dict(candidate=cid,family=family,kind=kind,param=param,balanced=balanced,feature_kwargs=feature_kwargs,
                   dimensions=X.shape[1],nnz=int(X.nnz),feature_fit_transform_s=feature_s,fit_s=fit_s,predict_s=predict_s,
                   process_rss_mb=psutil.Process().memory_info().rss/2**20,**m)
            records.append(r)
            pd.DataFrame({"doc_id":tune.doc_id.to_numpy(),"score_spam":score}).to_parquet(f"artifacts/predictions/tune_{cid}.parquet",index=False)
            if family not in registry or rank(m)>rank(registry[family]["metrics"]):
                artifact=SparseArtifact(features,clf,prior,ratio)
                modelpath=f"artifacts/models/{family}.joblib"; joblib.dump(artifact,modelpath,compress=3)
                registry[family]=dict(candidate=cid,path=modelpath,metrics=m,kind=kind,param=param,balanced=balanced,
                                      feature_kwargs=feature_kwargs,cap_audit=features.cap_audit_,fit_split="train",calibrated=False)
                loaded=joblib.load(modelpath)
                np.testing.assert_allclose(artifact.decision_function(tune.iloc[:20]),loaded.decision_function(tune.iloc[:20]),rtol=0,atol=1e-12)
            save_json("artifacts/sparse_registry.json",registry)
            pd.DataFrame([{k:v for k,v in r.items() if k!="feature_kwargs"} for r in records]).to_csv("artifacts/sparse_tune_results.csv",index=False)
            print(f"{cid}: Macro-F1={m['macro_f1']:.6f}, FP={m['fp']:.0f}, FN={m['fn']:.0f}, fit={fit_s:.2f}s",flush=True)
        del X,V,features; gc.collect()
    eligible=[k for k in registry if k not in {"naive","B1","B2"}]
    best=max(eligible,key=lambda k:rank(registry[k]["metrics"]))
    save_json("artifacts/sparse_selection.json",dict(best=best,criterion="Tune macro_f1 then lower ham_fpr",configurations=len(records),refit=False))
    # Repeat the fixed selected training configuration, with only optimizer seed
    # changed. This is a stability check, never test seed selection.
    chosen=registry[best]; model=joblib.load(chosen["path"]); X=model.features.transform(train); V=model.features.transform(tune)
    if model.ratio is not None: X=model.ratio.transform(X); V=model.ratio.transform(V)
    with threadpool_limits(limits=8): repeat=estimator(chosen["kind"],chosen["param"],43,chosen["balanced"]).fit(X,y)
    repeated=repeat.decision_function(V)
    save_json("artifacts/sparse_stability.json",dict(seed42=chosen["metrics"],seed43=metrics(yt,repeated,0,probability=False),
              max_score_delta=float(np.max(np.abs(repeated-model.decision_function(tune)))),note="Same split; optimizer repeat, not cross-validation"))
    del X,V; gc.collect()
    random_diagnostic(train,tune,config)

def random_diagnostic(train,tune,config):
    """Only Train+Tune pool; diagnostic differs in sample composition."""
    pool=pd.concat([train,tune],ignore_index=True)
    a,b=train_test_split(np.arange(len(pool)),test_size=len(tune),random_state=config["seed"],stratify=pool.label)
    tr=pool.iloc[a]; va=pool.iloc[b]
    start=time.perf_counter(); features=Blocks(baseline=True).fit(tr)
    clf=estimator("lr",1).fit(features.transform(tr),tr.label)
    scores=clf.decision_function(features.transform(va))
    overlap=set(tr.template_group_id)&set(va.template_group_id)
    save_json("artifacts/random_split_diagnostic.json",dict(**metrics(va.label,scores,0,probability=False),
              pool="train+tune only",overlap_groups=len(overlap),validation_rows_with_seen_group=int(va.template_group_id.isin(overlap).sum()),
              elapsed_s=time.perf_counter()-start,note="Fixed B2 C=1; different validation composition; not main Test and not a causal paired gain"))
