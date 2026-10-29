"""Bounded post-grid feature-cap/subject-weight checks and convergence audit."""
import json
import time
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from .features import Blocks,SparseArtifact
from .sparse_models import estimator,rank
from .io import load_records,save_json
from .metrics import metrics

def run():
    train=load_records("train"); tune=load_records("tune"); y=train.label.to_numpy()
    registry=json.loads(Path("artifacts/sparse_registry.json").read_text(encoding="utf-8")); rows=[]; convergence=[]
    for name,entry in registry.items():
        model=joblib.load(entry["path"]); clf=model.estimator
        converged=not hasattr(clf,"n_iter_") or int(np.max(clf.n_iter_))<clf.max_iter
        entry["converged"]=converged; convergence.append({"family":name,"converged":converged,"n_iter":int(np.max(clf.n_iter_)) if hasattr(clf,"n_iter_") else None})
        if not converged:
            joblib.dump(model,f"artifacts/models/unconverged_{name}.joblib",compress=3)
    # Each addition has a specific single-factor counterpart in the first grid.
    jobs=[("B2_cap200k",dict(baseline=True,word_max_features=200000),"lr",1),
          ("S_cap",dict(char=True,subject=True,word_max_features=200000,char_max_features=400000),"svm",1),
          ("S_subject05",dict(char=True,subject=True,subject_weight=.5),"svm",1),
          ("S_subject2",dict(char=True,subject=True,subject_weight=2),"svm",1),
          ("S_structure_lr",dict(char=True,subject=True,numeric=True),"lr",1)]
    # This already predeclared candidate also supplies the matched LR structural
    # ablation. Preserve it even if a later parser repair changes SVM convergence.
    for name,kwargs,kind,param in jobs:
        print("Controlled supplemental candidate",name,flush=True)
        start=time.perf_counter(); f=Blocks(**kwargs).fit(train); X=f.transform(train); V=f.transform(tune); feature_s=time.perf_counter()-start
        start=time.perf_counter()
        with warnings.catch_warnings(record=True) as caught,threadpool_limits(limits=8):
            warnings.simplefilter("always",ConvergenceWarning); clf=estimator(kind,param).fit(X,y)
        fit_s=time.perf_counter()-start; converged=not any(issubclass(w.category,ConvergenceWarning) for w in caught)
        score=clf.decision_function(V); m=metrics(tune.label,score,0,probability=False)
        model=SparseArtifact(f,clf,float(y.mean())); path=f"artifacts/models/{name}.joblib"; joblib.dump(model,path,compress=3)
        registry[name]=dict(candidate=name,path=path,metrics=m,kind=kind,param=param,balanced=False,feature_kwargs=kwargs,cap_audit=f.cap_audit_,fit_split="train",calibrated=False,converged=converged)
        rows.append(dict(candidate=name,converged=converged,dimensions=X.shape[1],feature_s=feature_s,fit_s=fit_s,**m))
        pd.DataFrame({"doc_id":tune.doc_id.to_numpy(),"score_spam":score}).to_parquet(f"artifacts/predictions/tune_{name}.parquet",index=False)
        pd.DataFrame(rows).to_csv("artifacts/sparse_supplement_results.csv",index=False)
        save_json("artifacts/sparse_registry.json",registry); print(name,m,flush=True)
        del X,V,f,clf,model
        import gc; gc.collect()
    eligible=[k for k in registry if k not in {"B1","B2","naive","B2_cap200k"} and registry[k].get("converged",False)]
    best=max(eligible,key=lambda k:rank(registry[k]["metrics"]))
    save_json("artifacts/convergence_audit.json",convergence)
    save_json("artifacts/sparse_selection.json",dict(best=best,criterion="Tune macro_f1 then lower ham_fpr among converged candidates",configurations=26+len(jobs),refit=False,
              reason_supplement="Feature cap hit; two subject-weight endpoints; same-classifier LR structure comparison and convergence audit"))
    from .evaluation import error_tables
    model=joblib.load(registry[best]["path"]); error_tables(tune,model.decision_function(tune),0,"tune")
    print("Final sparse choice",best,flush=True)

if __name__=="__main__": run()
