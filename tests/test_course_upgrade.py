"""Core behavior: honest denominators, local uploads and isolated experiments."""
import numpy as np
import pandas as pd
import pytest
from spam_analysis.analysis.tables import prediction_frame,reliability,length_edges,assign_slices,slice_errors,paired_bootstrap
from spam_analysis.analysis.uploads import read_upload,replace_span,export_csv
from spam_analysis.experiments.runtime import Budget,ExperimentSession
from spam_analysis.experiments.train import development
from spam_analysis.experiments.train import guarded_threshold
from spam_analysis.inference import LocalSession,record_from_text


def sample():
    records=pd.DataFrame({'doc_id':['a','b','c','d'],'label':[0,0,1,1],
        'body_chars':[0,10,100,1000],'replacement_ratio':[0,.1,0,0],'body_missing':[1,0,0,0],
        'url_count':[0,1,2,5],'subject_chars':[0,3,4,5],'template_group_id':['g1','g1','g2','g3']})
    predictions=pd.DataFrame({'doc_id':['d','b','a','c'],'score_spam':[1.,.8,0.,.2]})
    return prediction_frame(records,predictions,.5)


def test_probability_endpoints_and_every_row_counted():
    frame=sample();bins,stats=reliability(frame)
    assert bins['count'].sum()==4 and bins.iloc[0]['count']==1 and bins.iloc[-1]['count']==1
    assert bins.iloc[1].mean_probability!=bins.iloc[1].mean_probability  # empty bin stays missing
    assert stats['brier']==pytest.approx(.32)
    assert stats['ece']==pytest.approx(.4)
    assert frame.false_positive.sum()==frame.false_negative.sum()==1


def test_join_and_probability_fail_closed():
    records=pd.DataFrame({'doc_id':['a'],'label':[0]})
    with pytest.raises(ValueError,match='对应'):prediction_frame(records,pd.DataFrame({'doc_id':['x'],'score_spam':[.5]}),.5)
    with pytest.raises(ValueError,match='概率'):prediction_frame(records,pd.DataFrame({'doc_id':['a'],'score_spam':[np.nan]}),.5)
    with pytest.raises(ValueError,match='唯一'):prediction_frame(pd.concat([records,records]),pd.DataFrame({'doc_id':['a'],'score_spam':[.5]}),.5)
    with pytest.raises(ValueError,match='分母'):
        prediction_frame(pd.DataFrame({'doc_id':['a','b'],'label':[0,1]}),pd.DataFrame({'doc_id':['a'],'score_spam':[.5]}),.5)


def test_slices_reconcile_and_length_boundaries_use_training():
    frame=sample();edges=length_edges(pd.DataFrame({'body_chars':[0,1,2,3]}))
    for field in ['length','quality','links','subject']:
        table=slice_errors(frame,assign_slices(frame,field,edges))
        assert table['count'].sum()==4 and table.errors.sum()==2
        assert table.false_positive.sum()==1 and table.false_negative.sum()==1
    assert np.isinf(edges[-1]) and edges[-2]<4


def test_group_bootstrap_identical_predictions_have_zero_interval():
    frame=sample()
    output=paired_bootstrap(frame,frame[['doc_id','score_spam']],.5,repeats=50)
    assert output['difference']==output['low']==output['high']==0 and output['groups']==3


def test_uploads_do_not_turn_names_or_labels_into_features():
    raw='主题,正文,label\n通知,周四提交课程报告,spam\n通知,周四提交课程报告,ham\n'.encode('utf-8')
    records=read_upload('spam.csv',raw)
    assert len(records)==2 and records.doc_id.is_unique and 'label' not in records
    assert records.semantic_text.tolist()==['通知\n周四提交课程报告']*2
    text=read_upload('spam.txt','中文🙂'.encode('utf-8'))
    assert text.subject_clean.iloc[0]=='' and text.body_clean.iloc[0]=='中文🙂'
    with pytest.raises(ValueError):read_upload('x.csv',b'subject\nHello\n')
    with pytest.raises(ValueError):read_upload('x.csv',('body\n'+'a\n'*101).encode())
    with pytest.raises(ValueError):read_upload('x.zip',b'abc')


def test_span_offsets_match_unicode_and_changes_are_local():
    record=record_from_text('通知','甲🙂乙，周四交作业。')
    subject,body=replace_span(record,{'field':'正文','start':1,'end':2,'surface':'🙂'},'好')
    assert subject=='通知' and body=='甲好乙,周四交作业。'
    with pytest.raises(ValueError,match='变化'):replace_span(record,{'field':'正文','start':1,'end':2,'surface':'错'},'')
    subject,body=replace_span(record,{'field':'语义文本','start':3,'end':4,'surface':'甲'},'')
    assert body.startswith('🙂')


def test_download_csv_neutralizes_formulas():
    content=export_csv(pd.DataFrame({'主题':['=SUM(1,2)',' @evil'],'分数':[.2,.3]})).decode('utf-8-sig')
    assert "'=SUM" in content and "' @evil" in content


def test_test_partition_cannot_enter_development():
    with pytest.raises(ValueError,match='Test'):development('test')


def test_threshold_guard_preserves_each_source_and_matches_actual_confusion():
    from spam_analysis.metrics import metrics
    policy=pd.DataFrame({'label':[0,1,0,1],'source_id':['a','a','b','b']})
    p=np.array([.1,.7,.2,.8])
    baseline={s:metrics(g.label,p[g.index],.5) for s,g in policy.groupby('source_id')}
    threshold,by=guarded_threshold(policy,p,baseline)
    assert all(row['accuracy']==1 and row['ham_fpr']==0 for row in by.values())
    assert .2<threshold<=.7
    assert guarded_threshold(policy,np.array([.9,.7,.2,.8]),baseline) is None


def test_budget_enforced(tmp_path):
    budget=Budget(tmp_path,limit=5)
    with pytest.raises(RuntimeError):budget.start('too much',reserve=6)
    budget.start('job',reserve=1);budget.finish()
    assert Budget(tmp_path).remaining<=5


def test_legacy_session_is_numerically_compatible():
    one=record_from_text('课程通知','周四下午提交作业。')
    old=LocalSession();adapted=ExperimentSession('artifacts/evaluation_plan.json')
    a,_=old.scores(one,'B2');b,_=adapted.scores(one,'B2')
    np.testing.assert_array_equal(a.score_spam,b.score_spam)


def test_checkpoint_resume_preserves_optimizer_scheduler_and_randomness(tmp_path):
    import torch
    from spam_analysis.experiments.checkpoint import Checkpoints,capture_rng,restore_rng
    torch.manual_seed(9);rng=np.random.default_rng(9)
    model=torch.nn.Sequential(torch.nn.Linear(3,5),torch.nn.Dropout(.4),torch.nn.Linear(5,1))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=.9)
    def step(m,o,s):
        x=torch.rand(4,3);y=torch.rand(4,1);o.zero_grad();loss=((m(x)-y)**2).mean();loss.backward();o.step();s.step()
    step(model,optimizer,scheduler)
    store=Checkpoints(tmp_path,{'data':'fixed','config':'fixed'})
    store.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'rng':capture_rng(rng)}, {'processed':4})
    step(model,optimizer,scheduler);expected=[p.detach().clone() for p in model.parameters()]
    restored=torch.nn.Sequential(torch.nn.Linear(3,5),torch.nn.Dropout(.4),torch.nn.Linear(5,1))
    opt=torch.optim.AdamW(restored.parameters(),lr=.01);sched=torch.optim.lr_scheduler.StepLR(opt,step_size=1,gamma=.9)
    state=Checkpoints(tmp_path,{'data':'fixed','config':'fixed'}).load()
    restored.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);sched.load_state_dict(state['scheduler']);restore_rng(state['rng'],rng)
    step(restored,opt,sched)
    for a,b in zip(expected,restored.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert sched.state_dict()==scheduler.state_dict()


def test_checkpoint_corrupt_latest_falls_back_but_rejects_different_data(tmp_path):
    from spam_analysis.experiments.checkpoint import Checkpoints
    store=Checkpoints(tmp_path,{'dataset':'a'});store.save({'step':1},{'processed':1});store.save({'step':2},{'processed':2})
    (tmp_path/'resume_1.pt').write_bytes(b'interrupted write')
    assert Checkpoints(tmp_path,{'dataset':'a'}).load()['step']==1
    with pytest.raises(ValueError,match='配置'):Checkpoints(tmp_path,{'dataset':'b'}).load()


def test_checkpoint_missing_manifest_never_silently_restarts(tmp_path):
    from spam_analysis.experiments.checkpoint import Checkpoints
    store=Checkpoints(tmp_path,{'dataset':'a'});store.save({'step':1},{'processed':1})
    (tmp_path/'resume_0.json').write_text('{incomplete',encoding='utf-8')
    with pytest.raises(ValueError,match='不会自动从头覆盖'):
        Checkpoints(tmp_path,{'dataset':'a'}).load()


def test_training_lock_rejects_parallel_writer_and_releases_after_exit(tmp_path):
    import subprocess,sys
    from spam_analysis.experiments.checkpoint import exclusive_run
    code='from spam_analysis.experiments.checkpoint import exclusive_run; import sys\nwith exclusive_run(sys.argv[1]): print("acquired")'
    with exclusive_run(tmp_path):
        second=subprocess.run([sys.executable,'-c',code,str(tmp_path)],capture_output=True)
        assert second.returncode!=0 and b'RuntimeError' in second.stderr
    with exclusive_run(tmp_path):pass


def test_unified_dataset_keeps_frozen_membership_and_disjoint_sources():
    from spam_analysis.analysis.results import source_records
    frame=source_records('combined');old=pd.read_parquet('artifacts/splits/split_manifest.parquet').set_index('doc_id')
    legacy=frame.loc[frame.source_id.eq('trec06c')].set_index('doc_id').loc[old.index]
    assert legacy.split.equals(old.split) and legacy.template_group_id.equals(old.template_group_id)
    assert frame.doc_id.is_unique and frame.groupby('template_group_id').split.nunique().max()==1
    assert frame.loc[frame.split.eq('test'),'doc_id'].isin(development('train').doc_id).sum()==0
    assert frame.label.isin([0,1]).all() and frame.script_profile.notna().all()
