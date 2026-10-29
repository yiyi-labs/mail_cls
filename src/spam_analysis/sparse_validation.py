"""Same-classifier structural ablation and optimizer stability of final choice."""
import json
import time
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.base import clone
from threadpoolctl import threadpool_limits
from .io import load_records, save_json
from .features import SparseArtifact
from .sparse_models import estimator, rank
from .metrics import metrics


def run():
    if Path('artifacts/evaluation_plan.json').exists():
        raise RuntimeError('Evaluation frozen')
    from .sparse_convergence import repair
    repair()
    train=load_records('train'); tune=load_records('tune')
    registry=json.loads(Path('artifacts/sparse_registry.json').read_text(encoding='utf-8'))
    # Reuse the already fitted, identical text blocks. A classifier change and
    # feature change together cannot establish the effect of structure alone.
    source=joblib.load(registry['S']['path'])
    X=source.features.transform(train); V=source.features.transform(tune)
    started=time.perf_counter()
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=8):
        warnings.simplefilter('always',ConvergenceWarning)
        clf=estimator('lr',1).fit(X,train.label)
    elapsed=time.perf_counter()-started
    converged=not any(issubclass(w.category,ConvergenceWarning) for w in caught)
    score=clf.decision_function(V); m=metrics(tune.label,score,0,probability=False)
    path='artifacts/models/S_text_lr.joblib'
    joblib.dump(SparseArtifact(source.features,clf,source.prior),path,compress=3)
    registry['S_text_lr']=dict(candidate='S_text_lr',path=path,metrics=m,kind='lr',param=1,balanced=False,
        feature_kwargs=dict(char=True,subject=True),cap_audit=source.features.cap_audit_,fit_split='train',calibrated=False,converged=converged)
    table=pd.read_csv('artifacts/sparse_supplement_results.csv')
    table=table[table.candidate.ne('S_text_lr')]
    table=pd.concat([table,pd.DataFrame([dict(candidate='S_text_lr',converged=converged,dimensions=X.shape[1],feature_s=0,
        fit_s=elapsed,**m)])],ignore_index=True)
    table.to_csv('artifacts/sparse_supplement_results.csv',index=False)
    pd.DataFrame(dict(doc_id=tune.doc_id.to_numpy(),score_spam=score)).to_parquet('artifacts/predictions/tune_S_text_lr.parquet',index=False)
    save_json('artifacts/sparse_registry.json',registry)
    del X,V,source,clf
    import gc; gc.collect()
    eligible=[k for k in registry if k not in {'B1','B2','naive','B2_cap200k'} and registry[k].get('converged',False)]
    best=max(eligible,key=lambda k:rank(registry[k]['metrics']))
    save_json('artifacts/sparse_selection.json',dict(best=best,criterion='Tune macro_f1 then lower ham_fpr among converged candidates',
        configurations=26+len(table),refit=False,reason_supplement='Feature caps, subject weights, convergence repair and matched LR structural ablation'))
    selected=registry[best]; model=joblib.load(selected['path'])
    X=model.features.transform(train); V=model.features.transform(tune)
    if model.ratio is not None: X=model.ratio.transform(X); V=model.ratio.transform(V)
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=8):
        warnings.simplefilter('always',ConvergenceWarning)
        repeated=clone(model.estimator).set_params(random_state=43).fit(X,train.label)
    repeated_scores=repeated.decision_function(V)
    original=model.decision_function(tune)
    previous=Path('artifacts/sparse_stability.json')
    if previous.exists() and not Path('artifacts/unconverged_stability_diagnostic.json').exists():
        save_json('artifacts/unconverged_stability_diagnostic.json',json.loads(previous.read_text(encoding='utf-8')))
    save_json(previous,dict(model=best,seed42=selected['metrics'],seed43=metrics(tune.label,repeated_scores,0,probability=False),
        repeated_converged=not any(issubclass(w.category,ConvergenceWarning) for w in caught),
        prediction_flips=int(np.sum((repeated_scores>=0)!=(original>=0))),max_score_delta=float(np.max(np.abs(repeated_scores-original))),
        note='Same Train/Tune split and frozen transforms; optimizer seed repeat, not cross-validation or Test selection'))
    from .evaluation import error_tables
    error_tables(tune,original,0,'tune')
    print('P2 final converged choice',best,selected['metrics'],flush=True)


if __name__=='__main__': run()
