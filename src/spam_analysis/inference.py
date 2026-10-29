"""Label-free local inference session shared by the demo and performance checks."""
import hashlib
import json
import time
from pathlib import Path
from email.message import EmailMessage
import joblib
import numpy as np
import pandas as pd
from .io import sha256
from .parsing import parse_email


def record_from_bytes(raw):
    return pd.DataFrame([dict(doc_id='input_'+hashlib.sha256(raw).hexdigest(),**parse_email(raw))])


def record_from_text(subject,body):
    mail=EmailMessage(); mail['Subject']=subject.replace('\r',' ').replace('\n',' ')
    mail.set_content(body,charset='utf-8')
    return record_from_bytes(mail.as_bytes())


class LocalSession:
    def __init__(self):
        frozen=Path('artifacts/evaluation_plan.json')
        if frozen.exists():
            plan=json.loads(frozen.read_text(encoding='utf-8'))
            self.entries=plan['registry']; self.specifications=plan['specifications']; self.main=plan['main']
        else:
            self.entries=json.loads(Path('artifacts/calibrated_registry.json').read_text(encoding='utf-8'))
            plan=json.loads(Path('artifacts/policy_selection.json').read_text(encoding='utf-8'))
            self.specifications=plan['specifications']; self.main=plan['main']
        self.loaded={}; self.load_seconds={}

    def load(self,name):
        if name in self.loaded: return
        entry=self.entries[name]; started=time.perf_counter()
        if sha256(entry['path'])!=entry['model_sha256'] or sha256(entry['calibrator_path'])!=entry['calibrator_sha256']:
            raise RuntimeError('Saved model/calibrator identity mismatch')
        calibration=joblib.load(entry['calibrator_path'])
        if entry['type']=='sparse': base=joblib.load(entry['path'])
        elif name=='E1':
            from .embeddings import QwenEncoder
            base=(QwenEncoder(entry['identity'],entry['config']),joblib.load(entry['path']))
        elif name=='E2':
            from .model_assets import configure_local
            configure_local()
            import torch
            from transformers import AutoTokenizer,AutoModelForSequenceClassification
            folder=entry['checkpoint']['folder']
            for filename,digest in entry['checkpoint']['files'].items():
                if sha256(Path(folder)/filename)!=digest: raise RuntimeError('Checkpoint changed')
            tokenizer=AutoTokenizer.from_pretrained(folder,local_files_only=True)
            model=AutoModelForSequenceClassification.from_pretrained(folder,local_files_only=True,attn_implementation='sdpa').to('cuda' if torch.cuda.is_available() else 'cpu')
            base=(model,tokenizer)
        else: raise ValueError(name)
        self.loaded[name]=(base,calibration)
        self.load_seconds[name]=time.perf_counter()-started

    def branch_scores(self,name,records):
        entry=self.entries[name]; nonempty=records.lexical_text.ne('').to_numpy()
        probability=np.full(len(records),entry['train_prior'],dtype=float)
        if not nonempty.any(): return probability
        self.load(name); base,calibration=self.loaded[name]
        active=records.loc[nonempty]
        if entry['type']=='sparse': raw=base.decision_function(active)
        elif name=='E1':
            encoder,classifier=base
            raw=classifier.decision_function(encoder.encode(active.semantic_text.tolist()))
        else:
            from .macbert import batches,predict
            model,tokenizer=base
            raw=predict(model,tokenizer,batches(active,tokenizer,entry['config'],entry['checkpoint']['mode']),entry['config']['macbert_batch_size'])
        probability[nonempty]=calibration.predict_proba(np.asarray(raw).reshape(-1,1))[:,1]
        if not np.isfinite(probability).all(): raise ValueError('Non-finite inference result')
        return probability

    def scores(self,records,name=None):
        if not records.doc_id.is_unique: raise ValueError('Duplicate doc_id')
        name=name or self.main; details={}; score=np.zeros(len(records))
        for branch,weight in self.specifications[name]['weights'].items():
            if weight:
                details[branch]=self.branch_scores(branch,records); score+=weight*details[branch]
        return pd.DataFrame({'doc_id':records.doc_id.to_numpy(),'score_spam':score}),details

    def predict_bytes(self,raw,name=None,point='research'):
        records=record_from_bytes(raw); name=name or self.main
        predictions,branches=self.scores(records,name)
        threshold=self.specifications[name]['thresholds'][point]
        predictions['prediction']=(predictions.score_spam>=threshold).astype(int)
        predictions['threshold']=threshold
        return records,predictions,branches
