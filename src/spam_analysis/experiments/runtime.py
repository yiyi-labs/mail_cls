"""One run context shared by offline experiments and read-only inference."""
import json
import time
import os
from pathlib import Path
from spam_analysis.io import sha256, save_json
from spam_analysis.inference import LocalSession

ROOT = Path(__file__).resolve().parents[3]
ACTIVE = ROOT/'artifacts/active_run.json'


def active_plan():
    if ACTIVE.exists():
        pointer = json.loads(ACTIVE.read_text(encoding='utf-8'))
        path = (ROOT/pointer['plan']).resolve()
        if not path.is_relative_to(ROOT/'artifacts/runs') or sha256(path) != pointer['sha256']:
            raise ValueError('当前实验配置身份不一致')
        return path
    return ROOT/'artifacts/evaluation_plan.json'


class ExperimentSession(LocalSession):
    def __init__(self, plan_path=None):
        path = Path(plan_path or active_plan())
        self.plan = json.loads(path.read_text(encoding='utf-8'))
        self.run_id = self.plan.get('run_id', 'frozen-v1')
        self.entries = self.plan['registry']
        self.specifications = self.plan['specifications']
        self.main = self.plan['main']
        self.loaded = {}
        self.load_seconds = {}


def code_identity():
    # Freeze computation recursively. Presentation-only edits do not invalidate scores.
    paths = sorted((ROOT/'src').rglob('*.py'))
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}


class Budget:
    """Conservative GPU-job wall time accounting, including associated overhead."""
    def __init__(self, folder, limit=8*3600):
        self.path = Path(folder)/'budget.json'
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {'limit_seconds': limit, 'jobs': []}
        active=self.data.pop('active',None)
        if active:
            import psutil
            try:alive=abs(psutil.Process(active['pid']).create_time()-active['process_created'])<1
            except psutil.Error:alive=False
            if alive:raise RuntimeError('已有 GPU 实验正在运行，请勿重复启动')
            self.data['jobs'].append({'name':active['name'],'seconds':active['elapsed_seconds']+90,
                'status':'recovered_after_shutdown; conservative 90s allowance'})
            self._save()

    def _save(self):
        from .checkpoint import atomic_json
        atomic_json(self.path,self.data)

    @property
    def remaining(self):
        return self.data['limit_seconds']-sum(x['seconds'] for x in self.data['jobs'])

    def start(self, name, reserve=180):
        if self.remaining < reserve:
            raise RuntimeError('本轮 GPU 预算不足')
        self.name, self.started = name, time.monotonic()
        import psutil
        self.data['active']={'name':name,'pid':os.getpid(),'process_created':psutil.Process().create_time(),'elapsed_seconds':0}
        self._save()

    def heartbeat(self):
        self.data['active']['elapsed_seconds']=time.monotonic()-self.started
        self._save()

    def finish(self, status='complete'):
        self.data['jobs'].append({'name': self.name, 'seconds': time.monotonic()-self.started, 'status': status})
        self.data.pop('active',None);self._save()
