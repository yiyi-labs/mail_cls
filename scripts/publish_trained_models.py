"""Release the completed mixed-source candidates, keeping the first release intact.

This revision compares the current trained methods. Calibrate fits probabilities;
Policy alone ranks thresholds and methods. Test is a retrospective holdout because
the baseline-only release was already evaluated before this revision was defined.
"""
import contextlib
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));os.chdir(ROOT)
from spam_analysis.experiments import train
from spam_analysis.experiments.checkpoint import atomic_json,exclusive_run
from spam_analysis.experiments.runtime import Budget,code_identity
from spam_analysis.io import sha256

SOURCE=train.RUN
RUN=ROOT/'artifacts/runs/course-20261016'
EVALUATION_NOTE='固定留出集回顾评估；本轮发布修订前已观察旧模型在这些邮件上的结果。候选、校准和阈值仅使用开发数据。'


def best_threshold(labels,probabilities):
    from spam_analysis.metrics import threshold_table
    table=threshold_table(labels,probabilities)
    table['accuracy']=(table.tn+table.tp)/len(labels)
    row=table.sort_values(['accuracy','ham_fpr','macro_f1','threshold'],ascending=[False,True,False,False]).iloc[0]
    return float(row.threshold)


def development_only(frame,expected):
    if expected not in {'calibrate','policy'} or not frame.split.eq(expected).all():
        raise ValueError('发布选择仅允许 Calibrate 和 Policy，不允许 Test')


def freeze():
    import gc
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    from spam_analysis.metrics import metrics
    from spam_analysis.policy import operating_points
    if (RUN/'evaluation_plan.json').exists():return
    candidates=train.read_json(SOURCE/'candidates.json')
    assert set(train.CORE)<=set(candidates)
    config={**train.read_json(SOURCE/'config.json'),'run_id':RUN.name,
        'training_run_id':SOURCE.name,
        'selection':'Current trained candidates only; Policy accuracy, lower ham FPR, Macro-F1; calibration ties by Policy log loss then smaller C',
        'evaluation_note':EVALUATION_NOTE,
        'prior_release_plan_sha256':sha256(SOURCE/'evaluation_plan.json'),
        'candidates_sha256':sha256(SOURCE/'candidates.json')}
    if (RUN/'config.json').exists():
        assert train.read_json(RUN/'config.json')==config,'Release definition changed'
    else:
        # Record the rule before evaluating any current-candidate Test predictions.
        atomic_json(RUN/'config.json',config)
    for name in ['fit_membership.parquet','training_curves.csv','sparse_training.csv','qwen_training.csv','neural_progress.json']:
        shutil.copy2(SOURCE/name,RUN/name)
    budget=Budget(RUN);budget.start('Current-candidate calibration and Policy release',reserve=900)
    status='interrupted_or_failed'
    try:
        cal,policy=train.development('calibrate'),train.development('policy')
        development_only(cal,'calibrate');development_only(policy,'policy')
        entries={};specs={};decisions=[];ranking=[]
        for name in train.CORE:
            entry=copy.deepcopy(candidates[name])
            assert sha256(entry['path'])==entry['model_sha256']
            raw_cal=train.scores_raw(name,entry,cal);raw_policy=train.scores_raw(name,entry,policy)
            options=[]
            for c in config['calibration_C']:
                clf=LogisticRegression(C=c,max_iter=1000,solver='lbfgs').fit(raw_cal.reshape(-1,1),cal.label)
                p=train.probability(raw_policy,clf,policy,entry['train_prior'])
                threshold=best_threshold(policy.label,p);m=metrics(policy.label,p,threshold)
                options.append((m,float(log_loss(policy.label,p,labels=[0,1])),c,clf,p,threshold))
            m,ll,c,clf,p,threshold=max(options,key=lambda x:(*train.rank(x[0]),-x[1],-x[2]))
            path=RUN/'models'/f'{name}_calibrator.joblib';joblib.dump(clf,path)
            entry.update(calibrator_path=str(path.relative_to(ROOT)),calibrator_sha256=sha256(path),
                         calibrated=True,calibration_split='calibrate',calibration_C=c)
            points,_=operating_points(policy.label,p);points['research']=threshold
            entries[name]=entry;specs[name]={'weights':{name:1.},'thresholds':points}
            by_source={s:metrics(g.label,p[g.index],threshold) for s,g in policy.groupby('source_id')}
            decisions.append({'model':name,'promoted':True,'calibration_C':c,'policy_metrics':m,'by_source':by_source,
                              'meaning':'Included as this run\'s trained method, not a claim of per-source dominance over baseline'})
            ranking.append({'model':name,'eligible':True,**m})
            pd.DataFrame({'doc_id':policy.doc_id,'score_spam':p}).to_parquet(RUN/'predictions'/f'policy_{name}.parquet',index=False)
            print('Policy-only release',name,m['accuracy'],flush=True)
            budget.heartbeat();gc.collect()
            import torch
            torch.cuda.empty_cache()
        main=max(ranking,key=train.rank)['model']
        pd.DataFrame(ranking).to_csv(RUN/'main_selection.csv',index=False)
        atomic_json(RUN/'selection.json',decisions)
        plan={'run_id':RUN.name,'training_run_id':SOURCE.name,'main':main,'registry':entries,'specifications':specs,
            'comparison_scope':'combined','evaluation_kind':'retrospective_holdout','evaluation_note':EVALUATION_NOTE,
            'config_sha256':sha256(RUN/'config.json'),'data_sha256':config['corpus_sha256'],
            'source_hashes':code_identity(),'release_script_sha256':sha256(Path(__file__)),
            'training_membership_sha256':sha256(RUN/'fit_membership.parquet'),
            'baseline_plan_sha256':sha256(ROOT/'artifacts/evaluation_plan.json'),
            'prior_release_plan_sha256':config['prior_release_plan_sha256'],
            'frozen_at':datetime.now(timezone.utc).isoformat(),
            'limitations':[EVALUATION_NOTE,'Historical corpus; modern Chinese generalization remains unverified']}
        atomic_json(RUN/'evaluation_plan.json',plan)
        status='complete'
    finally:budget.finish(status)


def stage(label):
    atomic_json(RUN/'pipeline_status.json',{'stage':label,'status':'running'})
    print('\n'+label,flush=True)


def main():
    RUN.mkdir(parents=True,exist_ok=True)
    for name in ['models','predictions']:(RUN/name).mkdir(exist_ok=True)
    with exclusive_run(RUN,'pipeline.lock'),exclusive_run(RUN),exclusive_run(SOURCE):
        if not (RUN/'budget.json').exists():
            old_budget=train.read_json(SOURCE/'budget.json')
            assert 'active' not in old_budget,'Previous GPU work still running'
            # Continue the same cumulative 8-hour ledger, not a fresh allowance.
            atomic_json(RUN/'budget.json',old_budget)
        with (RUN/'pipeline.log').open('a',encoding='utf-8',buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            try:
                stage('校准并冻结本轮训练模型');freeze()
                from spam_analysis.experiments import evaluate
                evaluate.RUN=RUN
                # Reuse only immutable old-model predictions for the baseline comparison.
                for source in ['enron','spamassassin']:
                    for name in train.CORE:
                        filename=f'baseline_{source}_{name}.parquet'
                        target=RUN/'predictions'/filename
                        if not target.exists():shutil.copy2(SOURCE/'predictions'/filename,target)
                stage('固定留出集回顾评价');evaluate.evaluate()
                if not (RUN/'benchmark.csv').exists():stage('同机推理比较');evaluate.benchmark()
                stage('发布本轮训练模型');evaluate.publish()
                for label,args in [
                    ('网页与神经推理检查',['scripts/validate_local_app.py','--neural']),
                    ('更新报告',['scripts/build_course_materials.py','report']),
                    ('执行课程 Notebook',['scripts/build_course_materials.py','notebooks','--execute'])]:
                    stage(label)
                    subprocess.run([sys.executable,'-u',*args],check=True,stdout=log,stderr=subprocess.STDOUT,
                                   env={**os.environ,'PYTHONUTF8':'1'})
                atomic_json(RUN/'pipeline_status.json',{'stage':'本轮模型与分析材料已更新','status':'complete',
                    'remaining':'浏览器检查、课件生成与视觉检查、材料同步及清理'})
            except BaseException:
                current=train.read_json(RUN/'pipeline_status.json');current['status']='interrupted_or_failed'
                atomic_json(RUN/'pipeline_status.json',current)
                raise


if __name__=='__main__':main()
