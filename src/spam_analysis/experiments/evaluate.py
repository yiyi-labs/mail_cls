"""Evaluate a sealed run once; publish only complete, identity-bound results."""
import argparse
import gc
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from spam_analysis.io import sha256,save_json
from spam_analysis.metrics import metrics
from spam_analysis.macbert import batches
from spam_analysis.analysis.results import source_records,CORE
from spam_analysis.analysis.tables import prediction_frame,reliability,paired_bootstrap
from .runtime import ROOT,ACTIVE,ExperimentSession,Budget,code_identity
from .train import RUN,read_json
from .corpus import CORPUS,SOURCE_NAMES


def verify_plan():
    plan=read_json(RUN/'evaluation_plan.json')
    if sha256(CORPUS)!=plan['data_sha256']:raise ValueError('测试数据身份改变')
    if sha256(RUN/'config.json')!=plan['config_sha256']:raise ValueError('实验配置改变')
    if sha256(RUN/'fit_membership.parquet')!=plan['training_membership_sha256']:raise ValueError('拟合样本记录改变')
    if code_identity()!=plan['source_hashes']:raise ValueError('冻结后代码改变，不能自动开封测试')
    membership=pd.read_parquet(RUN/'fit_membership.parquet')
    test=source_records('combined','test')
    if set(test.doc_id)&set(membership.doc_id) or set(test.template_group_id)&set(membership.template_group_id):
        raise ValueError('新测试集与开发数据交叉')
    return plan


def calculate(session,records,name,path):
    if path.exists():
        cached=pd.read_parquet(path)
        if not cached.doc_id.is_unique or set(cached.doc_id)!=set(records.doc_id):raise ValueError('已保存预测的分母改变')
        return cached.set_index('doc_id').loc[records.doc_id].reset_index()
    parts=[]
    for start in range(0,len(records),512):
        chunk=records.iloc[start:start+512].drop(columns='label')
        scores,_=session.scores(chunk,name);parts.append(scores)
    result=pd.concat(parts,ignore_index=True)
    temp=path.with_suffix('.partial.parquet');result.to_parquet(temp,index=False);temp.replace(path)
    return result


def truncation(session,records,name,scope):
    if name not in {'E1','E2'}:return
    session.load(name);base,_=session.loaded[name];entry=session.entries[name]
    tokenizer=base[0].tokenizer if name=='E1' else base[1]
    if name=='E2':
        original=np.array([len(tokenizer.encode(s,add_special_tokens=False))+len(tokenizer.encode(b,add_special_tokens=False))+tokenizer.num_special_tokens_to_add(pair=True)
                           for s,b in zip(records.subject_clean,records.body_clean)])
        kept=np.array([len(x['input_ids']) for x in batches(records,tokenizer,entry['config'],entry['checkpoint']['mode'])])
    else:
        original=np.array([len(tokenizer(t,truncation=False)['input_ids']) for t in records.semantic_text])
        kept=np.minimum(original,entry['config']['max_length'])
    pd.DataFrame({'doc_id':records.doc_id,'original_tokens':original,'retained_tokens':kept,
                  'truncation_ratio':(original-kept)/np.maximum(original,1)}).to_parquet(RUN/f'truncation_{scope}_{name}.parquet',index=False)


def release(session):
    session.loaded.clear();gc.collect()
    import torch
    if torch.cuda.is_available():torch.cuda.empty_cache()


def evaluate():
    plan=verify_plan();identity=sha256(RUN/'evaluation_plan.json')
    marker=RUN/'test_opened.json'
    if marker.exists() and read_json(marker)['plan_sha256']!=identity:raise ValueError('测试已按不同方案打开')
    if (RUN/'test_complete.json').exists():
        print('Evaluation already complete; saved results retained.');return
    save_json(marker,{'plan_sha256':identity,'started_at':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
                      'scope':'Supplemental source group-heldout tests; TREC06C historical review only'})
    current=ExperimentSession(RUN/'evaluation_plan.json');old=ExperimentSession(ROOT/'artifacts/evaluation_plan.json')
    budget=Budget(RUN);budget.start('Frozen evaluation and token retention',reserve=900)
    summaries=[]
    try:
        for scope in source_records('combined','test').source_id.unique():
            records=source_records(scope,'test');rows=[];baseline_rows=[]
            for name in CORE:
                path=RUN/'predictions'/f'{scope}_{name}.parquet'
                unchanged=all(current.entries[name][k]==old.entries[name][k] for k in ['model_sha256','calibrator_sha256'])
                if scope=='trec06c' and unchanged and not path.exists():
                    source=pd.read_parquet(ROOT/'artifacts/predictions'/f'test_{name}.parquet')
                    source.to_parquet(path,index=False)
                scores=calculate(current,records,name,path)
                threshold=current.specifications[name]['thresholds']['research']
                frame=prediction_frame(records,scores,threshold);bins,cal=reliability(frame)
                bins.to_csv(RUN/f'reliability_{scope}_{name}.csv',index=False)
                rows.append({'model':name,**metrics(frame.label,frame.score_spam,threshold),'ece':cal['ece'],'log_loss':cal['log_loss']})
                truncation(current,records,name,scope);release(current)
                if scope=='trec06c':baseline=pd.read_parquet(ROOT/'artifacts/predictions'/f'test_{name}.parquet')
                else:baseline=calculate(old,records,name,RUN/'predictions'/f'baseline_{scope}_{name}.parquet')
                bt=old.specifications[name]['thresholds']['research']
                baseline=baseline.set_index('doc_id').loc[frame.doc_id].reset_index()
                baseline_rows.append({'model':name,**metrics(frame.label,baseline.score_spam,bt)})
                interval=paired_bootstrap(frame,baseline,bt)
                summaries.append({'scope':scope,'model':name,**interval})
                release(old);print('Evaluated',scope,name,'accuracy',rows[-1]['accuracy'],flush=True)
            pd.DataFrame(rows).to_csv(RUN/f'results_{scope}.csv',index=False)
            pd.DataFrame(baseline_rows).to_csv(RUN/f'baseline_{scope}.csv',index=False)
        # Merge per-mail predictions, then recompute metrics; never average source accuracies.
        records=source_records('combined','test');rows=[]
        for name in CORE:
            scores=pd.concat([pd.read_parquet(RUN/'predictions'/f'{scope}_{name}.parquet') for scope in records.source_id.unique()],ignore_index=True)
            if set(scores.doc_id)!=set(records.doc_id):raise ValueError('汇总评价分母不完整')
            scores.to_parquet(RUN/'predictions'/f'combined_{name}.parquet',index=False)
            threshold=current.specifications[name]['thresholds']['research'];frame=prediction_frame(records,scores,threshold)
            bins,cal=reliability(frame);bins.to_csv(RUN/f'reliability_combined_{name}.csv',index=False)
            rows.append({'model':name,**metrics(frame.label,frame.score_spam,threshold),'ece':cal['ece'],'log_loss':cal['log_loss']})
            if name in {'E1','E2'}:
                pd.concat([pd.read_parquet(RUN/f'truncation_{scope}_{name}.parquet') for scope in records.source_id.unique()],ignore_index=True).to_parquet(RUN/f'truncation_combined_{name}.parquet',index=False)
        pd.DataFrame(rows).to_csv(RUN/'results_combined.csv',index=False)
        pd.DataFrame(summaries).to_csv(RUN/'accuracy_change_intervals.csv',index=False)
        outputs=(list(RUN.glob('results_*.csv'))+list(RUN.glob('baseline_*.csv'))+
                 list(RUN.glob('reliability_*.csv'))+list(RUN.glob('truncation_*.parquet'))+
                 [RUN/'accuracy_change_intervals.csv']+list((RUN/'predictions').glob('*.parquet')))
        save_json(RUN/'test_complete.json',{'plan_sha256':identity,'files':{str(p.relative_to(ROOT)):sha256(p) for p in outputs}})
    finally:budget.finish()


def parameter_counts(engine,name):
    base,cal=engine.loaded[name]
    if name=='E2':
        total=sum(p.numel() for p in base[0].parameters());return total+2,total+2
    if name=='E1':
        encoder,clf=base;head=clf.coef_.size+clf.intercept_.size+2
        return sum(p.numel() for p in encoder.model.parameters())+head,head
    clf=base.estimator
    count=(clf.coef_.size+clf.intercept_.size) if hasattr(clf,'coef_') else (clf.feature_log_prob_.size+clf.class_log_prior_.size)
    count+=sum(len(vec.idf_) for _,_,vec,_ in base.features.transforms_)+2
    if base.features.numeric:count+=base.features.scaler_.mean_.size+base.features.scaler_.scale_.size
    return count,count


def benchmark():
    import torch,psutil
    from email.message import EmailMessage
    from spam_analysis.inference import record_from_bytes
    from .train import development
    verify_plan();train=development('train')
    train['length_bin']=pd.qcut(train.body_chars,4,labels=False,duplicates='drop')
    samples=train.groupby(['source_id','label','length_bin'],group_keys=False).sample(n=8,random_state=42)
    samples[['doc_id','source_id','length_bin']].to_csv(RUN/'benchmark_samples.csv',index=False)
    raw=[]
    for row in samples.itertuples():
        mail=EmailMessage();mail['Subject']=row.subject_clean.replace('\r',' ').replace('\n',' ')
        mail.set_content(row.body_clean,charset='utf-8');raw.append(mail.as_bytes())
    # Transport encoding is normalized for a repeatable parser-to-score benchmark.
    rows=[];budget=Budget(RUN);budget.start('Same-machine inference benchmark',reserve=180)
    try:
        for name in CORE:
            engine=ExperimentSession(RUN/'evaluation_plan.json');started=time.perf_counter();engine.load(name)
            load_s=time.perf_counter()-started
            for payload in raw[:5]:engine.scores(record_from_bytes(payload),name)
            total,trainable=parameter_counts(engine,name)
            if torch.cuda.is_available():torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            timings=[];rss=0;process=psutil.Process()
            for _ in range(3):
                for payload in raw:
                    started=time.perf_counter();engine.scores(record_from_bytes(payload),name)
                    if name in {'E1','E2'} and torch.cuda.is_available():torch.cuda.synchronize()
                    timings.append((time.perf_counter()-started)*1000);rss=max(rss,process.memory_info().rss)
            rows.append({'model':name,'median_ms':float(np.median(timings)),'p95_ms':float(np.quantile(timings,.95)),
                         'cold_load_s':load_s,'observed_process_rss_mib':rss/2**20,
                         'gpu_allocated_peak_mib':torch.cuda.max_memory_allocated()/2**20 if name in {'E1','E2'} and torch.cuda.is_available() else None,
                         'total_parameters':total,'trainable_parameters':trainable,'samples':len(raw),'repeats':3,
                         'device':torch.cuda.get_device_name() if name in {'E1','E2'} and torch.cuda.is_available() else 'CPU',
                         'precision':'BF16 compute' if name in {'E1','E2'} else 'saved sklearn precision'})
            release(engine);print('Benchmark',rows[-1],flush=True)
        pd.DataFrame(rows).to_csv(RUN/'benchmark.csv',index=False)
    finally:budget.finish()


def publish():
    verify_plan()
    completed=read_json(RUN/'test_complete.json')
    if completed['plan_sha256']!=sha256(RUN/'evaluation_plan.json'):raise ValueError('测试结果不属于当前方案')
    for path,digest in completed['files'].items():
        if sha256(ROOT/path)!=digest:raise ValueError('正式结果被修改')
    if not (RUN/'benchmark.csv').exists():raise ValueError('性能比较尚未完成')
    pointer={'plan':str((RUN/'evaluation_plan.json').relative_to(ROOT)),'sha256':sha256(RUN/'evaluation_plan.json')}
    temporary=ACTIVE.with_suffix('.tmp');save_json(temporary,pointer);temporary.replace(ACTIVE)
    print('Published',pointer,flush=True)


if __name__=='__main__':
    from .checkpoint import exclusive_run
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['evaluate','benchmark','publish'])
    args=parser.parse_args()
    with exclusive_run(RUN):{'evaluate':evaluate,'benchmark':benchmark,'publish':publish}[args.stage]()
