"""Atomic, rotating local training checkpoints with data/configuration identity."""
import json
import os
import random
from contextlib import contextmanager
from pathlib import Path
import numpy as np
from spam_analysis.io import sha256


@contextmanager
def exclusive_run(folder, filename='execution.lock'):
    """OS-owned file lock: released even after a crash or machine shutdown."""
    path=Path(folder)/filename;path.parent.mkdir(parents=True,exist_ok=True)
    handle=path.open('a+b')
    try:
        handle.seek(0,2)
        if handle.tell()==0:handle.write(b'0');handle.flush()
        handle.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('已有本轮训练或评价正在运行，请勿重复启动') from exc
        yield
    finally:
        # Closing releases the OS lock; the harmless marker file may remain.
        handle.close()


def atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w',encoding='utf-8') as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2,allow_nan=False)
        handle.flush();os.fsync(handle.fileno())
    temporary.replace(path)


class Checkpoints:
    def __init__(self, folder, identity):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.identity=identity;self.sequence=-1

    def load(self):
        import torch
        candidates=[]
        for path in self.folder.glob('resume_*.json'):
            try:
                meta=json.loads(path.read_text(encoding='utf-8'));candidates.append((meta['sequence'],path,meta))
            except (ValueError,KeyError):continue
        failures=[]
        for sequence,path,meta in sorted(candidates,reverse=True):
            if meta['identity']!=self.identity:raise ValueError('训练断点的数据或配置不同，拒绝混用')
            tensor=path.with_suffix('.pt')
            try:
                if sha256(tensor)!=meta['sha256']:raise ValueError('checkpoint digest mismatch')
                # Only local files written by this store and bound to their SHA are loaded.
                state=torch.load(tensor,map_location='cpu',weights_only=False)
                self.sequence=sequence
                if failures:print('Recovered previous intact checkpoint:',tensor.name,flush=True)
                return state
            except (OSError,ValueError,RuntimeError,EOFError) as exc:failures.append(str(exc))
        if candidates or list(self.folder.glob('resume_*.*')):
            raise ValueError('训练断点不可读，请保留文件排查，不会自动从头覆盖：'+'; '.join(failures))
        return None

    def save(self, state, progress):
        import torch
        self.sequence+=1;path=self.folder/f'resume_{self.sequence%2}.pt'
        temporary=path.with_suffix('.tmp')
        with temporary.open('wb') as handle:
            torch.save(state,handle);handle.flush();os.fsync(handle.fileno())
        temporary.replace(path)
        atomic_json(path.with_suffix('.json'),dict(sequence=self.sequence,identity=self.identity,sha256=sha256(path),**progress))
        atomic_json(self.folder.parent/'training_status.json',dict(sequence=self.sequence,**progress))


def capture_rng(generator):
    import torch
    return dict(generator=generator.bit_generator.state,numpy=np.random.get_state(),python=random.getstate(),
                torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state,generator):
    import torch
    generator.bit_generator.state=state['generator'];np.random.set_state(state['numpy']);random.setstate(state['python'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])
