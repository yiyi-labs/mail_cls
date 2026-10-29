"""Official frozen local embeddings and a supervised Chinese encoder.

Downloads contain model files only. Email text never leaves the process.
"""
import gc
import hashlib
import json
import os
import time
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from .io import load_records,read_config,save_json,sha256
from .metrics import metrics
from .parsing import PARSER_VERSION






def train_semantic(experiment):
    if not Path("artifacts/sparse_selection.json").exists(): raise RuntimeError("P2 must complete before semantic training")
    if Path("artifacts/evaluation_plan.json").exists(): raise RuntimeError("Evaluation frozen")
    selection=json.loads(Path('artifacts/sparse_selection.json').read_text(encoding='utf-8'))
    sparse=json.loads(Path('artifacts/sparse_registry.json').read_text(encoding='utf-8'))
    if not sparse[selection['best']].get('converged',False):
        raise RuntimeError('Complete the P2 convergence and controlled-ablation audit before P3')
    config=read_config("configs/semantic.yaml"); configure_local(); train=load_records("train"); tune=load_records("tune")
    identity_path=Path("artifacts/qwen_identity.json")
    identity=json.loads(identity_path.read_text(encoding="utf-8")) if identity_path.exists() else download_official(config["qwen_repo"])
    save_json(identity_path,identity); enc=QwenEncoder(identity,config)
    print('E1 resource profile device',enc.device,'load_s',enc.load_s,flush=True)
    sample=train.sample(n=min(128,len(train)),random_state=config["seed"])
    profile=enc.profile(sample)
    lengths=[len(x) for x in enc.tokenizer(train.semantic_text.tolist(),add_special_tokens=True,truncation=False)["input_ids"]]
    profile.update(train_tokens_quantiles={str(q):float(np.quantile(lengths,q)) for q in [.5,.9,.95,.99]},train_truncated_fraction=float(np.mean(np.array(lengths)>config["max_length"])),
                   context_decision="1024 head default retained; long-text truncation reported, not hidden")
    save_json("artifacts/qwen_profile.json",profile); print(profile,flush=True)
    X=enc.cache(train,"train"); V=enc.cache(tune,"tune"); rows=[]; best=None
    for c in config["lr_candidates"]:
        started=time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',ConvergenceWarning)
            clf=LogisticRegression(C=c,solver="lbfgs",max_iter=1500,random_state=42).fit(X,train.label)
        converged=not any(issubclass(w.category,ConvergenceWarning) for w in caught)
        scores=clf.decision_function(V); m=metrics(tune.label,scores,0,probability=False)
        rows.append(dict(C=c,fit_s=time.perf_counter()-started,converged=converged,iterations=int(clf.n_iter_.max()),**m))
        pd.DataFrame({"doc_id":tune.doc_id.to_numpy(),"score_spam":scores}).to_parquet(f"artifacts/predictions/tune_E1_{c}.parquet",index=False)
        if converged and (best is None or (m["macro_f1"],-m["ham_fpr"])>(best["macro_f1"],-best["ham_fpr"])):
            best=m; joblib.dump(clf,"artifacts/models/E1.joblib"); chosen=c
            np.testing.assert_allclose(joblib.load('artifacts/models/E1.joblib').decision_function(V),scores,rtol=0,atol=1e-12)
        print('E1 LR',c,'converged',converged,m,flush=True)
    pd.DataFrame(rows).to_csv("artifacts/semantic_tune_results.csv",index=False)
    if best is None: raise RuntimeError('No E1 LR candidate converged')
    save_json("artifacts/semantic_registry.json",{"E1":{"path":"artifacts/models/E1.joblib","identity":identity,"config":config,"C":chosen,"metrics":best,"fit_split":"train"}})
    save_json("artifacts/semantic_status.json",{"E1":"complete","E2":"pending resource profile"})
    print("E1 complete",best,flush=True)
    del enc; gc.collect()
    import torch
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    train_macbert(train,tune,config)

def predict_semantic(name,df,split,entry=None):
    # Formal evaluation supplies its frozen entry. Do not consult mutable
    # development registry settings when a frozen identity is available.
    if entry is None:
        registry=json.loads(Path("artifacts/semantic_registry.json").read_text(encoding="utf-8")); entry=registry[name]
    if name=="E1":
        encoder=QwenEncoder(entry["identity"],entry["config"]); X=encoder.cache(df,split)
        clf=joblib.load(entry["path"]); scores=clf.decision_function(X)
        del encoder; gc.collect()
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return scores
    return macbert_scores(entry,df)

def train_macbert(train,tune,config):
    from .macbert import train_macbert as run
    return run(train,tune,config)

def macbert_scores(entry,df):
    from .macbert import macbert_scores as run
    return run(entry,df)

from .model_assets import configure_local, download_official

from .semantic_tokens import last_token_pool, truncate_tokens

from .embeddings import QwenEncoder
