"""Resuming later stages must never reopen model fitting after a freeze."""
import importlib.util
import json
from pathlib import Path
import pytest


def workflow(tmp_path,monkeypatch):
    path=Path(__file__).resolve().parents[1]/'scripts/continue_training.py'
    spec=importlib.util.spec_from_file_location('course_resume',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'RUN',tmp_path)
    return module


def test_frozen_resume_skips_fitting_and_selection(tmp_path,monkeypatch):
    module=workflow(tmp_path,monkeypatch);(tmp_path/'evaluation_plan.json').write_text('{}')
    calls=[];monkeypatch.setattr(module,'run',lambda stage,*args:calls.append(args))
    module.main(False)
    assert all('spam_analysis.experiments.train' not in args for args in calls)
    assert calls[0]==('-m','spam_analysis.experiments.evaluate','evaluate')
    assert json.loads((tmp_path/'pipeline_status.json').read_text(encoding='utf-8'))['status']=='complete'


def test_failed_training_cannot_fall_through_to_test(tmp_path,monkeypatch):
    module=workflow(tmp_path,monkeypatch);calls=[]
    def fail(stage,*args):
        calls.append(args);module.atomic_json(tmp_path/'pipeline_status.json',{'stage':stage})
        raise RuntimeError('interrupted')
    monkeypatch.setattr(module,'run',fail)
    with pytest.raises(RuntimeError,match='interrupted'):module.main(False)
    assert calls==[('-m','spam_analysis.experiments.train','fit')]
    assert json.loads((tmp_path/'pipeline_status.json').read_text(encoding='utf-8'))['status']=='interrupted_or_failed'


def test_resume_uses_current_release_instead_of_republishing_old_plan(tmp_path,monkeypatch):
    module=workflow(tmp_path,monkeypatch)
    monkeypatch.setattr(module,'ROOT',tmp_path)
    source=tmp_path/'artifacts/runs/course-20261015'
    current=source.parent/'course-20261016';current.mkdir(parents=True)
    (current/'config.json').write_text('{}')
    monkeypatch.setattr(module,'RUN',source)
    calls=[]
    monkeypatch.setattr(module.subprocess,'run',lambda args,**kwargs:calls.append(args))
    module.main(False)
    assert calls==[[module.sys.executable,'-u','scripts/publish_trained_models.py']]
    assert not source.exists()
