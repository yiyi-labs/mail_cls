"""Record the actual runtime and verify saved prediction arithmetic."""
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
from .io import save_json,sha256
from .metrics import align_scores,metrics


def environment_snapshot():
    names=['numpy','pandas','scipy','scikit-learn','jieba','datasketch','pyarrow','matplotlib','torch','transformers','huggingface-hub','streamlit','pytest']
    versions={}
    for name in names:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    result=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),logical_cpus=psutil.cpu_count(),
                ram_total_bytes=psutil.virtual_memory().total,versions=versions)
    try:
        import torch
        result.update(cuda_compiled=torch.version.cuda,cuda_available=torch.cuda.is_available(),
                      gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except ImportError: pass
    save_json('artifacts/environment.json',result)
    return result


def verify_predictions():
    plan=json.loads(Path('artifacts/evaluation_plan.json').read_text(encoding='utf-8'))
    df=pd.read_parquet('artifacts/splits/split_manifest.parquet').query("split=='test'")
    result=pd.read_csv('artifacts/test_results.csv'); checked=0
    for name in plan['comparisons']:
        scores=pd.read_parquet(f'artifacts/predictions/test_{name}.parquet')
        merged=align_scores(df,scores)
        for point,threshold in plan['specifications'][name]['thresholds'].items():
            for weighting in ['per_mail','per_group']:
                weights=None if weighting=='per_mail' else 1/merged.groupby('template_group_id').doc_id.transform('size').to_numpy()
                m=metrics(merged.label,merged.score_spam,threshold,weights)
                saved=result[(result.model==name)&(result.point==point)&(result.weighting==weighting)].iloc[0]
                for key in ['tn','fp','fn','tp','accuracy','macro_f1','balanced_accuracy','ham_fpr','spam_precision','spam_recall','spam_f1','ham_precision','ham_recall','ham_f1','average_precision','roc_auc','brier']:
                    np.testing.assert_allclose(m[key],saved[key],rtol=0,atol=1e-12)
                expected=len(df) if weighting=='per_mail' else merged.template_group_id.nunique()
                np.testing.assert_allclose(m['tn']+m['fp']+m['fn']+m['tp'],expected,rtol=0,atol=1e-9)
                checked+=1
    save_json('artifacts/prediction_reconciliation.json',dict(models=len(plan['comparisons']),operating_points=checked//2,weighted_result_rows=checked,rows=len(df),
        passed=True,method='Recompute per-mail and per-group metrics from doc_id-aligned saved predictions; no model inference or threshold fitting'))


if __name__=='__main__':
    environment_snapshot()
    if '--verify-predictions' in sys.argv:verify_predictions()
