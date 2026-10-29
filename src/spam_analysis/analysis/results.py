"""Version-aware result access; never associate old scores with new weights."""
import pandas as pd
from spam_analysis.experiments.runtime import ROOT
from .tables import prediction_frame
from spam_analysis.experiments.corpus import CORPUS, SOURCE_NAMES

CORE = ['B1','B2','S_char13','E1','E2']


def evaluation_scopes(run_id):
    if run_id == 'frozen-v1':return {'trec06c':'中文历史留出集'}
    root = ROOT/'artifacts/runs'/run_id
    return {key:label for key,label in [('combined','统一测试汇总'),*SOURCE_NAMES.items()]
            if (root/f'results_{key}.csv').exists()}


def result_table(run_id, scope='combined'):
    if run_id == 'frozen-v1':
        table = pd.read_csv(ROOT/'artifacts/test_results.csv')
        return table.loc[table.model.isin(CORE) & table.point.eq('research') & table.weighting.eq('per_mail')].copy()
    return pd.read_csv(ROOT/'artifacts/runs'/run_id/f'results_{scope}.csv')


def saved_predictions(run_id, scope, model):
    path = ROOT/'artifacts/predictions'/f'test_{model}.parquet' if run_id=='frozen-v1' else ROOT/'artifacts/runs'/run_id/'predictions'/f'{scope}_{model}.parquet'
    return pd.read_parquet(path)


def source_records(scope, split=None):
    if CORPUS.exists():
        filters=[]
        if scope!='combined':filters.append(('source_id','==',scope))
        if split is not None:filters.append(('split','==',split))
        return pd.read_parquet(CORPUS,filters=filters or None).reset_index(drop=True)
    if scope=='combined':
        return pd.concat([source_records(key,split) for key in SOURCE_NAMES if key=='trec06c' or (ROOT/'data/processed'/key/'records.parquet').exists()],ignore_index=True)
    if scope=='trec06c':
        records = pd.read_parquet(ROOT/'data/processed/records.parquet')
        manifest = pd.read_parquet(ROOT/'artifacts/splits/split_manifest.parquet')
        frame = records.merge(manifest[['doc_id','split','template_group_id']],on='doc_id',validate='one_to_one').assign(source_id='trec06c')
    elif scope in SOURCE_NAMES:frame = pd.read_parquet(ROOT/'data/processed'/scope/'records.parquet')
    else:raise ValueError(scope)
    return frame if split is None else frame.loc[frame.split.eq(split)].reset_index(drop=True)


def error_frame(run_id,scope,model,threshold):
    if run_id=='frozen-v1':scope='trec06c'
    records=source_records(scope,'test');predictions=saved_predictions(run_id,scope,model)
    if set(records.doc_id)!=set(predictions.doc_id):raise ValueError('评价分母不一致')
    frame=prediction_frame(records,predictions,threshold)
    path=ROOT/'artifacts/runs'/run_id/f'truncation_{scope}_{model}.parquet'
    if path.exists():frame=frame.merge(pd.read_parquet(path),on='doc_id',how='left',validate='one_to_one')
    return frame
