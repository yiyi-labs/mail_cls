"""One bounded solver repair for the observed nonconverged structure SVM."""
import json
import time
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from .io import load_records,save_json
from .features import SparseArtifact
from .metrics import metrics


def repair():
    if Path('artifacts/evaluation_plan.json').exists():raise RuntimeError('Evaluation frozen')
    registry=json.loads(Path('artifacts/sparse_registry.json').read_text(encoding='utf-8'))
    if registry['S_structure'].get('converged',False):return
    train=load_records('train');tune=load_records('tune')
    original=joblib.load(registry['S_structure']['path'])
    X=original.features.transform(train);V=original.features.transform(tune)
    candidate=clone(original.estimator).set_params(dual=False)
    print('One bounded structure-SVM solver repair: identical features/C/loss, dual=False',flush=True)
    started=time.perf_counter()
    with warnings.catch_warnings(record=True) as caught,threadpool_limits(limits=8):
        warnings.simplefilter('always',ConvergenceWarning)
        candidate.fit(X,train.label)
    elapsed=time.perf_counter()-started
    converged=not any(issubclass(w.category,ConvergenceWarning) for w in caught)
    score=candidate.decision_function(V);m=metrics(tune.label,score,0,probability=False)
    name='S_structure_primal';path=f'artifacts/models/{name}.joblib'
    joblib.dump(SparseArtifact(original.features,candidate,original.prior),path,compress=3)
    restored=joblib.load(path).decision_function(tune)
    np.testing.assert_allclose(restored,score,rtol=0,atol=1e-12)
    registry[name]={**registry['S_structure'],'candidate':name,'path':path,'metrics':m,'converged':converged,
        'solver_kwargs':{'dual':False},'iterations':int(candidate.n_iter_),
        'reason':'Observed max_iter failure; same regularized squared-hinge objective, one bounded primal solver attempt'}
    save_json('artifacts/sparse_registry.json',registry)
    results=pd.read_csv('artifacts/sparse_supplement_results.csv')
    results=results[results.candidate.ne(name)]
    results=pd.concat([results,pd.DataFrame([dict(candidate=name,converged=converged,dimensions=X.shape[1],feature_s=0,fit_s=elapsed,**m)])],ignore_index=True)
    results.to_csv('artifacts/sparse_supplement_results.csv',index=False)
    pd.DataFrame({'doc_id':tune.doc_id.to_numpy(),'score_spam':score}).to_parquet(f'artifacts/predictions/tune_{name}.parquet',index=False)
    save_json('artifacts/solver_repair.json',dict(candidate=name,converged=converged,iterations=int(candidate.n_iter_),
        max_iter=candidate.max_iter,fit_s=elapsed,unchanged=['fitted Train transforms','C','penalty','loss','tolerance','class weights','Train/Tune rows'],
        changed={'dual':False},metrics=m))
    print('Primal solver repair',converged,'iterations',candidate.n_iter_,m,flush=True)


if __name__=='__main__':repair()
