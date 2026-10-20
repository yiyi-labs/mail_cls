"""Development-only parser invariance, local explanations and resource checks."""
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .io import load_records,save_json
from .robustness import RULES,perturb
from .metrics import metrics
from .explain import linear_contributions


def parser_checks():
    # Selected without model predictions; includes ham/spam and quality/length.
    tune=load_records('tune'); sampled=tune.sample(min(256,len(tune)),random_state=42)
    rows=[]
    for rule in RULES:
        changed,applied=perturb(sampled,rule)
        subject_equal=changed.subject_clean.to_numpy()==sampled.subject_clean.to_numpy()
        body_equal=changed.body_clean.to_numpy()==sampled.body_clean.to_numpy()
        rows.append(dict(rule=rule,n=len(sampled),applied=int(applied.sum()),subject_equal=int(subject_equal.sum()),body_equal=int(body_equal.sum()),
                         changed_ham=int(((~body_equal)&sampled.label.eq(0).to_numpy()).sum()),changed_spam=int(((~body_equal)&sampled.label.eq(1).to_numpy()).sum())))
    pd.DataFrame(rows).to_csv('artifacts/development_perturbations.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
    for result in rows:
        if result['rule'] in {'html_wrapper','mime_base64'}:
            assert result['subject_equal']==result['n'] and result['body_equal']==result['n'], result
    return rows


def linear_examples():
    tune=load_records('tune'); registry=json.loads(Path('artifacts/sparse_registry.json').read_text(encoding='utf-8'))
    chosen=json.loads(Path('artifacts/sparse_selection.json').read_text(encoding='utf-8'))['best']
    model=joblib.load(registry[chosen]['path'])
    selected=pd.read_csv('artifacts/private/tune_selected_errors.csv').doc_id.tolist()
    # Include two fixed development examples even if the error count is zero.
    train=load_records('train'); dev=train[train.doc_id.isin(['data/000/000','data/000/001'])]
    cases=pd.concat([tune[tune.doc_id.isin(selected)],dev]).drop_duplicates('doc_id')
    directory=Path('artifacts/private/linear_explanations'); directory.mkdir(parents=True,exist_ok=True)
    summaries=[]
    for index in range(len(cases)):
        row=cases.iloc[index:index+1]; contributions,detail=linear_contributions(model,row)
        safe=row.doc_id.iloc[0].replace('/','_')
        contributions.to_csv(directory/(safe+'.csv'),index=False,encoding='utf-8-sig')
        summaries.append(dict(doc_id=row.doc_id.iloc[0],model=chosen,label=int(row.label.iloc[0]),**detail))
    save_json('artifacts/linear_explanation_validation.json',dict(examples=summaries,all_margin_reconstructions_passed=True))


def semantic_sensitivity():
    from .inference import LocalSession
    session=LocalSession(); train=load_records('train')
    # Two fixed Train mail IDs were registered before any fitting. Never choose
    # a passage because it makes a semantic model look successful.
    selected=train[train.doc_id.isin(['data/000/000','data/000/001'])]
    rows=[]
    for name,entry in session.entries.items():
        if entry['type']!='semantic': continue
        for _,record in selected.iterrows():
            original=pd.DataFrame([record]); before=session.branch_scores(name,original)[0]
            body=record.body_clean
            for part,(start,end) in enumerate([(0,min(120,len(body))),(max(0,len(body)-120),len(body))]):
                changed=original.copy(); changed['body_clean']=body[:start]+body[end:]
                changed['semantic_text']=(record.subject_clean+'\n'+changed.body_clean.iloc[0]).strip()
                changed['lexical_text']=changed.semantic_text
                after=session.branch_scores(name,changed)[0]
                rows.append(dict(model=name,doc_id=record.doc_id,part=part,removed_chars=end-start,before=float(before),after=float(after),delta=float(after-before),
                    interpretation='Local deletion sensitivity, not causal attribution'))
        # Avoid retaining both neural networks in GPU memory.
        del session.loaded[name]
        import gc; gc.collect()
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv('artifacts/semantic_sensitivity.csv',index=False)


def local_latency():
    from email.message import EmailMessage
    from .inference import LocalSession
    train=load_records('train').sample(32,random_state=42)
    raw=[]
    for record in train.itertuples():
        mail=EmailMessage(); mail['Subject']=record.subject_clean.replace('\n',' ')
        mail.set_content(record.body_clean,charset='utf-8'); raw.append(mail.as_bytes())
    session=LocalSession(); rows=[]
    for name in session.entries:
        started=time.perf_counter(); session.load(name); load=time.perf_counter()-started
        session.predict_bytes(raw[0],name)  # one unreported warm-up
        elapsed=[]
        for blob in raw:
            started=time.perf_counter(); session.predict_bytes(blob,name); elapsed.append(1000*(time.perf_counter()-started))
        entry=session.entries[name]; paths=[Path(entry['path']),Path(entry['calibrator_path'])]
        if entry['type']=='semantic':
            folder=Path(entry['identity']['folder'] if name=='E1' else entry['checkpoint']['folder'])
            paths+=list(folder.glob('*'))
        size=sum(p.stat().st_size for p in set(paths) if p.is_file())
        rows.append(dict(model=name,n=32,load_s=load,median_e2e_ms=float(np.median(elapsed)),p95_e2e_ms=float(np.quantile(elapsed,.95)),
            mean_e2e_ms=float(np.mean(elapsed)),local_artifact_bytes=size,source='32 fixed-seed Train emails re-encapsulated as UTF-8 MIME',
            scope='Single mail: parse, transform, score, calibrate, decision; excludes initial model load; includes no network'))
        del session.loaded[name]
        import gc; gc.collect()
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv('artifacts/performance_development.csv',index=False)


def candidate_robustness():
    """Paired Tune checks for every completed branch at its saved Policy point."""
    from .inference import LocalSession
    from .robustness import paired_metrics
    session=LocalSession(); tune=load_records('tune').sample(256,random_state=42)
    rows=[]
    for name in session.entries:
        before=session.branch_scores(name,tune)
        threshold=session.specifications[name]['thresholds']['research']
        for rule in RULES:
            changed,applied=perturb(tune,rule)
            after=session.branch_scores(name,changed)
            rows.append(dict(model=name,rule=rule,split='tune',threshold=threshold,
                **paired_metrics(tune.label,before,after,threshold,applied)))
        print('Tune paired robustness complete:',name,flush=True)
        del session.loaded[name]
        import gc;gc.collect()
        import torch
        if torch.cuda.is_available():torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv('artifacts/development_model_robustness.csv',index=False)


if __name__=='__main__':
    import sys
    parser_checks(); linear_examples()
    if '--calibrated' in sys.argv:
        semantic_sensitivity(); candidate_robustness(); local_latency()
