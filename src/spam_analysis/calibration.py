import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from .io import load_records,save_json,sha256
from .metrics import metrics

def registry():
    sparse=json.loads(Path("artifacts/sparse_registry.json").read_text(encoding="utf-8"))
    chosen=json.loads(Path("artifacts/sparse_selection.json").read_text(encoding="utf-8"))["best"]
    entries={k:{**sparse[k],"type":"sparse"} for k in dict.fromkeys(["B1","B2",chosen])}
    sem_path=Path("artifacts/semantic_registry.json")
    if sem_path.exists():
        sem=json.loads(sem_path.read_text(encoding="utf-8")); entries.update({k:{**v,"type":"semantic"} for k,v in sem.items()})
    return entries

def raw_scores(name,entry,df,split):
    if entry["type"]=="sparse": return joblib.load(entry["path"]).decision_function(df)
    from .semantic_models import predict_semantic
    return predict_semantic(name,df,split,entry=entry)

def calibrated_scores(name,entry,df,split):
    active=df.lexical_text.ne('').to_numpy()
    probability=np.full(len(df),entry['train_prior'],dtype=float)
    if not active.any(): return probability
    score=raw_scores(name,entry,df.loc[active],split)
    clf=joblib.load(entry["calibrator_path"])
    probability[active]=clf.predict_proba(np.asarray(score).reshape(-1,1))[:,1]
    # Empty input stays in the denominator and uses the Train prior. Applying a
    # sigmoid to a hand-written log prior would change that explicit fallback.
    return probability

def calibrate(config):
    if Path("artifacts/evaluation_plan.json").exists(): raise RuntimeError("Already frozen")
    cal=load_records("calibrate"); policy=load_records("policy"); entries=registry(); report=[]
    prior=float(load_records('train').label.mean())
    for name,entry in entries.items():
        print("Calibrating frozen base:",name,flush=True)
        score=raw_scores(name,entry,cal,"calibrate")
        clf=LogisticRegression(C=1e6,solver="lbfgs",max_iter=1000).fit(score.reshape(-1,1),cal.label)
        path=f"artifacts/models/{name}_calibrator.joblib"; joblib.dump(clf,path)
        entry.update(calibrator_path=path,calibration_split="calibrate",model_sha256=sha256(entry["path"]),calibrator_sha256=sha256(path),train_prior=prior)
        # If E2 path is a metadata file, the metadata also binds its checkpoint.
        prob=calibrated_scores(name,entry,policy,"policy")
        pd.DataFrame({"doc_id":policy.doc_id.to_numpy(),"score_spam":prob}).to_parquet(f"artifacts/predictions/policy_{name}.parquet",index=False)
        report.append({"model":name,**metrics(policy.label,prob)})
    save_json("artifacts/calibrated_registry.json",entries)
    pd.DataFrame(report).to_csv("artifacts/policy_default_results.csv",index=False)
