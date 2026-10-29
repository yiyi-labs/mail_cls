"""Bounded course experiments. Never fit or select on either Test partition."""
import argparse
import gc
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from spam_analysis.io import load_records, save_json, sha256
from spam_analysis.features import SparseArtifact
from spam_analysis.macbert import batches, predict
from spam_analysis.metrics import metrics, threshold_table
from spam_analysis.model_assets import configure_local, verify_identity
from .runtime import ROOT, ExperimentSession, Budget, code_identity, ACTIVE
from .corpus import CORPUS
from .checkpoint import atomic_json
from spam_analysis.analysis.results import source_records

RUN = ROOT/'artifacts/runs/course-20261015'
CORE = ['B1','B2','S_char13','E1','E2']


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def development(split):
    if split not in {'train','tune','calibrate','policy'}:
        raise ValueError('实验开发不能读取 Test')
    frame = source_records('combined',split)
    if not frame.doc_id.is_unique or not frame.split.eq(split).all():
        raise ValueError('开发集身份不一致')
    return frame


def initialize():
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN/'models').mkdir(exist_ok=True); (RUN/'predictions').mkdir(exist_ok=True)
    config = dict(run_id=RUN.name, seed=42, gpu_limit_seconds=28800,
                  macbert_learning_rates=[2e-5,3e-5], loss_modes=['ordinary','balanced'], epochs=3,
                  calibration_C=[1.0,1e6], selection='Tune accuracy, lower ham FPR, Macro-F1; Policy promotion guard per source',
                  corpus_sha256=sha256(CORPUS),
                  corpora=read_json(CORPUS.with_suffix('.json'))['sources'])
    if (RUN/'config.json').exists():
        if read_json(RUN/'config.json') != config:
            raise ValueError('已存在不同实验配置')
    else:
        save_json(RUN/'config.json',config)
        save_json(RUN/'initial_code.json',code_identity())
    if (RUN/'evaluation_plan.json').exists():
        raise ValueError('实验已经冻结，不能继续训练')
    identities = []
    for split in ['train','tune','calibrate','policy']:
        df = development(split)
        identities.append(df[['doc_id','split','template_group_id','source_id']])
    joined = pd.concat(identities)
    if not joined.doc_id.is_unique or joined.groupby('template_group_id').split.nunique().max()!=1:
        raise ValueError('训练用途交叉')
    joined.to_parquet(RUN/'fit_membership.parquet',index=False)
    return config


def rank(m):
    return (m['accuracy'], -m['ham_fpr'], m['macro_f1'])


def train_sparse():
    train, tune = development('train'), development('tune')
    old = ExperimentSession(ROOT/'artifacts/evaluation_plan.json')
    registry = read_json(RUN/'candidates.json') if (RUN/'candidates.json').exists() else {}
    rows = []
    for name in ['B1','B2','S_char13']:
        if name in registry:
            continue
        original = joblib.load(old.entries[name]['path'])
        features = clone(original.features)
        started = time.perf_counter(); x = features.fit_transform(train)
        estimator = clone(original.estimator)
        estimator.fit(x,train.label)
        artifact = SparseArtifact(features,estimator,float(train.label.mean()))
        fit_s = time.perf_counter()-started
        scores = artifact.decision_function(tune)
        m = metrics(tune.label,scores,0,probability=False)
        path = RUN/'models'/f'{name}.joblib';joblib.dump(artifact,path)
        entry = {**old.entries[name], 'path':str(path.relative_to(ROOT)), 'fit_split':'train',
                 'cap_audit':features.cap_audit_,
                 'train_prior':float(train.label.mean()),'metrics':m, 'fit_s':fit_s,
                 'train_rows':len(train),'model_sha256':sha256(path), 'candidate':f'{name}_mixed_sources'}
        registry[name] = entry;atomic_json(RUN/'candidates.json',registry)
        rows.append({'model':name,'fit_s':fit_s,'dimensions':x.shape[1],**m})
        print('Sparse complete',name, m,flush=True)
        del x,artifact,features,estimator,original;gc.collect()
    if rows:
        pd.DataFrame(rows).to_csv(RUN/'sparse_training.csv',index=False)


def train_neural(config):
    configure_local()
    import torch, signal, random
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    from .checkpoint import Checkpoints, capture_rng, restore_rng, atomic_json
    if not torch.cuda.is_available():raise RuntimeError('微调需要 CUDA')
    torch.set_num_threads(8)
    old=read_json(ROOT/'artifacts/evaluation_plan.json')['registry']['E2']
    identity=old['identity'];verify_identity(identity)
    tokenizer=AutoTokenizer.from_pretrained(identity['folder'],local_files_only=True)
    train,tune=development('train'),development('tune')
    encoded=batches(train,tokenizer,old['config'],'head_tail');valid=batches(tune,tokenizer,old['config'],'head_tail')
    labels=train.label.to_numpy();batch_size=8;accumulation=2
    store=Checkpoints(RUN/'resume',{'schema':1,'config':sha256(RUN/'config.json'),
        'membership':sha256(RUN/'fit_membership.parquet'),'batch_size':batch_size,'accumulation':accumulation})
    resume=store.load()
    registry=resume['registry'] if resume else read_json(RUN/'candidates.json')
    # Qwen is trained after MacBERT. Reopening a finished fit must retain it.
    persisted=read_json(RUN/'candidates.json')
    if 'E1' in persisted:
        if sha256(ROOT/persisted['E1']['path'])!=persisted['E1']['model_sha256']:
            raise ValueError('已完成的 Qwen 分类头文件改变')
        registry['E1']=persisted['E1']
    curves=resume['curves'] if resume else []
    if resume and 'E2' in registry:
        atomic_json(ROOT/registry['E2']['path'],registry['E2']['checkpoint'])
        registry['E2']['model_sha256']=sha256(ROOT/registry['E2']['path'])
    best=registry.get('E2',{}).get('metrics');budget=Budget(RUN)
    stop=[False];original_handler=signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True))
    try:
        for loss_mode in config['loss_modes']:
            for lr in config['macbert_learning_rates']:
                if sum(r['loss_mode']==loss_mode and r['lr']==lr for r in curves)==config['epochs']:continue
                if budget.remaining<2400:
                    atomic_json(RUN/'budget_stop.json',{'remaining':budget.remaining,'reason':'Reserve evaluation time'});return
                torch.manual_seed(config['seed']);np.random.seed(config['seed']);random.seed(config['seed'])
                model=AutoModelForSequenceClassification.from_pretrained(identity['folder'],num_labels=2,
                    local_files_only=True,trust_remote_code=False,attn_implementation='sdpa').to('cuda')
                optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01)
                batch_count=int(np.ceil(len(train)/batch_size));steps=int(np.ceil(batch_count/accumulation))*config['epochs']
                scheduler=get_linear_schedule_with_warmup(optimizer,int(.1*steps),steps)
                weights=torch.tensor(len(train)/(2*np.bincount(labels,minlength=2)) if loss_mode=='balanced' else np.ones(2),dtype=torch.float32,device='cuda')
                rng=np.random.default_rng(config['seed']);cursor=None
                if resume and resume['loss_mode']==loss_mode and resume['lr']==lr:
                    model.load_state_dict(resume['model']);optimizer.load_state_dict(resume['optimizer']);scheduler.load_state_dict(resume['scheduler'])
                    restore_rng(resume['rng'],rng);cursor=resume['cursor']
                    print('Resuming',loss_mode,lr,cursor['epoch'],cursor['next_batch'],flush=True)
                resume=None
                first_epoch=cursor['epoch']+(cursor['phase']=='validated') if cursor else 1
                for epoch in range(first_epoch,config['epochs']+1):
                    budget.start(f'MacBERT {loss_mode} lr={lr} epoch={epoch}',reserve=1200)
                    status='complete'
                    try:
                        model.train();optimizer.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats()
                        if cursor and cursor['epoch']==epoch and cursor['phase']=='train':
                            order=cursor['order'];next_batch=cursor['next_batch'];total=cursor['total'];seen=cursor['seen'];previous_s=cursor['training_s']
                        else:
                            order=rng.permutation(len(train));next_batch=0;total=0.;seen=0;previous_s=0.
                        cursor=None;started=time.monotonic();last_save=started
                        def checkpoint(phase='train'):
                            state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                                'rng':capture_rng(rng),'loss_mode':loss_mode,'lr':lr,'registry':registry,'curves':curves,
                                'cursor':{'epoch':epoch,'phase':phase,'order':order,'next_batch':next_batch,'total':total,'seen':seen,
                                          'training_s':previous_s+time.monotonic()-started}}
                            store.save(state,{'loss_mode':loss_mode,'lr':lr,'epoch':epoch,'phase':phase,'processed':seen,'total':len(train),
                                'saved_at':__import__('datetime').datetime.now().isoformat(timespec='seconds')})
                            budget.heartbeat()
                        # Even a fresh run has a valid recovery point before its first update.
                        if next_batch==0:checkpoint()
                        for bi in range(next_batch,batch_count):
                            ids=order[bi*batch_size:(bi+1)*batch_size]
                            batch=tokenizer.pad([encoded[i] for i in ids],padding=True,return_tensors='pt').to('cuda')
                            y=torch.tensor(labels[ids],dtype=torch.long,device='cuda')
                            with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(**batch).logits
                            loss=torch.nn.functional.cross_entropy(logits.float(),y,weight=weights)
                            group_size=min(accumulation,batch_count-(bi//accumulation)*accumulation)
                            (loss/group_size).backward();total+=float(loss.detach())*len(ids);seen+=len(ids)
                            boundary=(bi+1)%accumulation==0 or bi+1==batch_count
                            if boundary:
                                torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
                                next_batch=bi+1
                                if time.monotonic()-last_save>=60 or next_batch==batch_count or stop[0]:
                                    checkpoint();last_save=time.monotonic()
                                    if stop[0]:status='paused_by_interrupt';raise KeyboardInterrupt
                                    if budget.remaining-(time.monotonic()-budget.started)<1200:
                                        status='budget_stop';atomic_json(RUN/'budget_stop.json',{'reason':'Reserve evaluation time'});return
                            if bi%1000==0:print(f'Train {loss_mode} lr={lr} epoch={epoch} {seen}/{len(train)} loss={total/seen:.5f}',flush=True)
                        torch.cuda.synchronize();training_s=previous_s+time.monotonic()-started
                        scores=predict(model,tokenizer,valid,batch_size*2);m=metrics(tune.label,scores,0,probability=False)
                        item=dict(loss_mode=loss_mode,lr=lr,epoch=epoch,training_loss=total/seen,training_s=training_s,
                            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,train_rows=len(train),**m)
                        if best is None or rank(m)>rank(best):
                            folder=RUN/'models'/f'E2_{loss_mode}_{lr}_{epoch}'
                            model.save_pretrained(folder,safe_serialization=True);tokenizer.save_pretrained(folder)
                            descriptor={'folder':str(folder),'files':{p.name:sha256(p) for p in folder.iterdir() if p.is_file()},
                                'lr':lr,'epoch':epoch,'mode':'head_tail','loss_mode':loss_mode}
                            path=RUN/'models/E2_identity.json';atomic_json(path,descriptor)
                            registry['E2']={**old,'checkpoint':descriptor,'path':str(path.relative_to(ROOT)),
                                'metrics':m,'model_sha256':sha256(path),'fit_split':'train','train_rows':len(train),
                                'train_prior':float(train.label.mean()),'candidate':f'E2_{loss_mode}_{lr}_{epoch}'}
                            best=m
                        curves=[r for r in curves if not (r['loss_mode']==loss_mode and r['lr']==lr and r['epoch']==epoch)]+[item]
                        checkpoint('validated')
                        atomic_json(RUN/'candidates.json',registry);atomic_json(RUN/'neural_progress.json',curves)
                        pd.DataFrame(curves).to_csv(RUN/'training_curves.csv',index=False)
                        print('Tune',item,flush=True)
                    except BaseException:
                        if status=='complete':status='interrupted_or_failed'
                        raise
                    finally:budget.finish(status)
                del model,optimizer,scheduler;gc.collect();torch.cuda.empty_cache()
        # Republish derived metadata if a shutdown occurred immediately after a durable checkpoint.
        atomic_json(RUN/'candidates.json',registry);atomic_json(RUN/'neural_progress.json',curves)
        pd.DataFrame(curves).to_csv(RUN/'training_curves.csv',index=False)
    finally:signal.signal(signal.SIGINT,original_handler)


def scores_raw(name,entry,frame):
    if entry['type']=='sparse':
        return joblib.load(entry['path']).decision_function(frame)
    if name=='E2':
        from spam_analysis.macbert import macbert_scores
        return macbert_scores(entry,frame)
    if name=='E1':
        split=frame.split.iloc[0]
        x=qwen_vectors(frame,split)
        return joblib.load(entry['path']).decision_function(x)
    raise ValueError('Only changed branches need calibration')


def qwen_vectors(frame,split,encoder=None,budget=None):
    """Reuse immutable original embeddings and cache only supplemental vectors."""
    import hashlib
    if split not in {'train','tune','calibrate','policy'}:raise ValueError('No test embeddings in development')
    metadata=read_json(ROOT/'artifacts/embeddings'/split/'qwen.json')
    old=np.load(ROOT/'artifacts/embeddings'/split/'qwen.npy',mmap_mode='r')
    positions={doc:i for i,doc in enumerate(metadata['doc_ids'])}
    supplement=frame.loc[frame.source_id.ne('trec06c')]
    directory=RUN/'embeddings';directory.mkdir(exist_ok=True)
    path=directory/f'{split}.npy';meta_path=directory/f'{split}.json'
    identity=read_json(ROOT/'artifacts/evaluation_plan.json')['registry']['E1']
    expected={'doc_ids':supplement.doc_id.tolist(),
              'text_hashes':[hashlib.sha256(t.encode()).hexdigest() for t in supplement.semantic_text],
              'revision':identity['identity']['revision'],'config':identity['config']}
    if path.exists() and meta_path.exists():
        meta=read_json(meta_path)
        if {k:v for k,v in meta.items() if k!='array_sha256'}!=expected:raise ValueError('Qwen 缓存身份不一致')
        if meta.get('array_sha256') and sha256(path)!=meta['array_sha256']:raise ValueError('Qwen 缓存内容损坏')
        new=np.load(path)
    elif encoder is not None:
        order=np.argsort(supplement.semantic_text.str.len().to_numpy(),kind='stable')
        from .checkpoint import atomic_json
        chunk_dir=directory/(split+'_chunks');chunk_dir.mkdir(exist_ok=True)
        manifest=chunk_dir/'identity.json'
        if manifest.exists() and read_json(manifest)!=expected:raise ValueError('嵌入断点与数据不同')
        atomic_json(manifest,expected);parts=[]
        for start in range(0,len(order),128):
            piece=chunk_dir/f'{start:06d}.npy';checksum=piece.with_suffix('.json')
            if piece.exists() and checksum.exists() and sha256(piece)==read_json(checksum)['sha256']:
                part=np.load(piece)
            else:
                part=encoder.encode(supplement.semantic_text.iloc[order[start:start+128]].tolist())
                temporary=piece.with_suffix('.tmp')
                with temporary.open('wb') as handle:
                    np.save(handle,part);handle.flush();__import__('os').fsync(handle.fileno())
                temporary.replace(piece);atomic_json(checksum,{'sha256':sha256(piece)})
            parts.append(part)
            if budget:budget.heartbeat()
        encoded=np.concatenate(parts)
        new=np.empty_like(encoded);new[order]=encoded
        temporary=path.with_suffix('.tmp')
        with temporary.open('wb') as handle:
            np.save(handle,new);handle.flush();__import__('os').fsync(handle.fileno())
        temporary.replace(path);atomic_json(meta_path,{**expected,'array_sha256':sha256(path)})
    else:raise ValueError('请先完成 Qwen 辅助语料编码')
    new_positions={doc:i for i,doc in enumerate(supplement.doc_id)}
    rows=[]
    for record in frame.itertuples():
        if record.source_id=='trec06c':
            i=positions[record.doc_id]
            if hashlib.sha256(record.semantic_text.encode()).hexdigest()!=metadata['text_hashes'][i]:raise ValueError('原始嵌入文本身份不一致')
            rows.append(old[i])
        else:rows.append(new[new_positions[record.doc_id]])
    return np.stack(rows)


def train_qwen():
    from spam_analysis.embeddings import QwenEncoder
    registry=read_json(RUN/'candidates.json')
    if 'E1' in registry:return
    old=read_json(ROOT/'artifacts/evaluation_plan.json')['registry']['E1']
    budget=Budget(RUN);budget.start('Qwen supplemental encoding and classifier',reserve=900)
    status='interrupted_or_failed'
    try:
        encoder=QwenEncoder(old['identity'],old['config']);matrices={};frames={}
        for split in ['train','tune','calibrate','policy']:
            frames[split]=development(split);matrices[split]=qwen_vectors(frames[split],split,encoder,budget)
            print('Qwen vectors ready',split,len(frames[split]),flush=True)
        rows=[];best=None
        for c in [.1,1.,10.]:
            started=time.perf_counter();clf=LogisticRegression(C=c,solver='lbfgs',max_iter=3000,random_state=42)
            clf.fit(matrices['train'],frames['train'].label)
            m=metrics(frames['tune'].label,clf.decision_function(matrices['tune']),0,probability=False)
            rows.append({'C':c,'fit_s':time.perf_counter()-started,**m})
            if best is None or rank(m)>rank(best):
                best=m;path=RUN/'models/E1.joblib';temporary=path.with_suffix('.tmp')
                joblib.dump(clf,temporary);temporary.replace(path)
                registry['E1']={**old,'path':str(path.relative_to(ROOT)),'model_sha256':sha256(path),'metrics':m,
                    'C':c,'fit_split':'train','train_rows':len(frames['train']),'train_prior':float(frames['train'].label.mean()),'candidate':f'E1_mixed_C{c}'}
        pd.DataFrame(rows).to_csv(RUN/'qwen_training.csv',index=False);atomic_json(RUN/'candidates.json',registry)
        status='complete'
    finally:budget.finish(status)


def baseline_policy(engine,policy,name):
    path=RUN/'predictions'/f'baseline_policy_{name}.parquet'
    if path.exists():return pd.read_parquet(path).set_index('doc_id').loc[policy.doc_id].score_spam.to_numpy()
    old=policy.loc[policy.source_id.eq('trec06c')];new=policy.loc[policy.source_id.ne('trec06c')]
    saved=pd.read_parquet(ROOT/'artifacts/predictions'/f'policy_{name}.parquet').set_index('doc_id')
    p=np.empty(len(policy));p[old.index]=saved.loc[old.doc_id].score_spam.to_numpy()
    if name=='E1' and (RUN/'embeddings/policy.npy').exists():
        entry=engine.entries[name];x=qwen_vectors(policy,'policy')
        raw=joblib.load(entry['path']).decision_function(x)
        values=probability(raw,joblib.load(entry['calibrator_path']),policy,entry['train_prior'])
        p[new.index]=values[new.index]
    else:p[new.index]=engine.scores(new.drop(columns='label'),name)[0].score_spam.to_numpy()
    pd.DataFrame({'doc_id':policy.doc_id,'score_spam':p}).to_parquet(path,index=False)
    return p


def probability(raw,calibrator,frame,prior):
    p=calibrator.predict_proba(np.asarray(raw).reshape(-1,1))[:,1]
    p[frame.lexical_text.eq('').to_numpy()]=prior
    return p


def guarded_threshold(policy,p,baseline):
    """Vectorized per-source constraints; includes all candidate thresholds."""
    table=threshold_table(policy.label,p)
    table['accuracy']=(table.tn+table.tp)/(table.tn+table.fp+table.fn+table.tp)
    table=table.sort_values(['accuracy','ham_fpr','macro_f1','threshold'],ascending=[False,True,False,False])
    thresholds=table.threshold.to_numpy();allowed=np.ones(len(table),dtype=bool)
    for source,group in policy.groupby('source_id'):
        y=group.label.to_numpy();v=p[group.index];ham=np.sort(v[y==0]);spam=np.sort(v[y==1])
        tn=np.searchsorted(ham,thresholds,side='left');fn=np.searchsorted(spam,thresholds,side='left')
        fp=len(ham)-tn;tp=len(spam)-fn
        accuracy=(tn+tp)/len(group)
        macro=.5*(2*tp/np.maximum(2*tp+fp+fn,1)+2*tn/np.maximum(2*tn+fp+fn,1))
        b=baseline[source]
        allowed &= (fp/max(len(ham),1)<=b['ham_fpr']+1e-12)&(macro>=b['macro_f1']-1e-12)&(accuracy>=b['accuracy']-1e-12)
    if not allowed.any():return None
    threshold=float(thresholds[np.flatnonzero(allowed)[0]])
    return threshold,{s:metrics(g.label,p[g.index],threshold) for s,g in policy.groupby('source_id')}


def select_policy(config):
    from sklearn.metrics import log_loss
    cal,policy=development('calibrate'),development('policy')
    old=ExperimentSession(ROOT/'artifacts/evaluation_plan.json')
    candidates=read_json(RUN/'candidates.json');registry={k:old.entries[k] for k in CORE}
    specs={k:old.specifications[k] for k in CORE}; decisions=[]
    budget=Budget(RUN)
    for name,entry in candidates.items():
        budget.start('Calibration and policy '+name,reserve=300)
        try:
            if entry['type']=='sparse':
                entry['cap_audit']=joblib.load(entry['path']).features.cap_audit_
            raw_cal=scores_raw(name,entry,cal);raw_policy=scores_raw(name,entry,policy)
            old_p=baseline_policy(old,policy,name);old_threshold=old.specifications[name]['thresholds']['research']
            baseline={source:metrics(group.label,old_p[group.index],old_threshold)
                      for source,group in policy.groupby('source_id')}
            eligible=[]
            for c in config['calibration_C']:
                clf=LogisticRegression(C=c,max_iter=1000,solver='lbfgs').fit(raw_cal.reshape(-1,1),cal.label)
                p=probability(raw_policy,clf,policy,entry['train_prior'])
                winner=guarded_threshold(policy,p,baseline)
                if winner:
                    threshold,m_by=winner
                    eligible.append((metrics(policy.label,p,threshold),float(log_loss(policy.label,p,labels=[0,1])),c,clf,p,threshold,m_by))
            if eligible:
                eligible.sort(key=lambda x:(*rank(x[0]),-x[1],-x[2]),reverse=True)
                m,ll,c,clf,p,threshold,m_by=eligible[0]
                old_m=metrics(policy.label,old_p,old_threshold)
                promote=m['accuracy']>old_m['accuracy']+1e-12 or (m['accuracy']>=old_m['accuracy'] and ll<float(log_loss(policy.label,old_p,labels=[0,1]))-1e-6)
            else:
                promote=False;m=None;c=None;m_by=None
            if promote:
                path=RUN/'models'/f'{name}_calibrator.joblib';joblib.dump(clf,path)
                entry.update(calibrator_path=str(path.relative_to(ROOT)),calibrator_sha256=sha256(path),calibration_split='calibrate',calibration_C=c,calibrated=True)
                registry[name]=entry
                from spam_analysis.policy import operating_points
                points,_=operating_points(policy.label,p);points['research']=threshold
                specs[name]={'weights':{name:1.0},'thresholds':points}
            decisions.append({'model':name,'promoted':bool(promote),'calibration_C':c,'policy_metrics':m,
                              'by_source':m_by,'baseline_by_source':baseline})
            selected=p if promote else old_p
            pd.DataFrame({'doc_id':policy.doc_id,'score_spam':selected}).to_parquet(RUN/'predictions'/f'policy_{name}.parquet',index=False)
            save_json(RUN/'selection.json',decisions)
            print('Policy selection',name,'promoted=',promote,flush=True)
        finally:
            budget.finish()
            old.loaded.clear();gc.collect()
            import torch
            torch.cuda.empty_cache()
    # Main-model selection uses Policy only, with the original main as a floor.
    base_p=baseline_policy(old,policy,old.main);base_t=old.specifications[old.main]['thresholds']['research']
    baseline={s:metrics(g.label,base_p[g.index],base_t) for s,g in policy.groupby('source_id')}
    main=old.main;best=metrics(policy.label,base_p,base_t);ranking=[]
    for name in CORE:
        saved=RUN/'predictions'/f'policy_{name}.parquet'
        if saved.exists():p=pd.read_parquet(saved).set_index('doc_id').loc[policy.doc_id].score_spam.to_numpy()
        else:
            p=baseline_policy(old,policy,name)
            pd.DataFrame({'doc_id':policy.doc_id,'score_spam':p}).to_parquet(saved,index=False)
        threshold=specs[name]['thresholds']['research'];m=metrics(policy.label,p,threshold)
        by={s:metrics(g.label,p[g.index],threshold) for s,g in policy.groupby('source_id')}
        qualified=all(by[s]['accuracy']>=b['accuracy']-1e-12 and by[s]['macro_f1']>=b['macro_f1']-1e-12 and by[s]['ham_fpr']<=b['ham_fpr']+1e-12 for s,b in baseline.items())
        if qualified and rank(m)>rank(best):main=name;best=m
        ranking.append({'model':name,'eligible':qualified,**m})
    pd.DataFrame(ranking).to_csv(RUN/'main_selection.csv',index=False)
    plan={'run_id':RUN.name,'main':main,'registry':registry,'specifications':specs,
          'comparison_scope':'combined','config_sha256':sha256(RUN/'config.json'),
          'data_sha256':sha256(CORPUS),'source_hashes':code_identity(),
          'training_membership_sha256':sha256(RUN/'fit_membership.parquet'),
          'baseline_plan_sha256':sha256(ROOT/'artifacts/evaluation_plan.json'),
          'frozen_at':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
          'limitations':['Historical primarily English supplemental corpus; modern Chinese generalization remains unverified']}
    save_json(RUN/'evaluation_plan.json',plan)
    print('Experiment frozen. Run evaluate separately after UI and tests are ready.',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['train','qwen','select','fit'])
    args=parser.parse_args();config=initialize()
    if args.stage in {'train','fit'}:
        train_sparse();train_neural(config)
        if args.stage=='fit' and not (RUN/'budget_stop.json').exists():train_qwen()
    elif args.stage=='qwen':train_qwen()
    else:
        select_policy(config)


if __name__=='__main__':
    from .checkpoint import exclusive_run
    with exclusive_run(RUN):main()
