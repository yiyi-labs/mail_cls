"""Deterministic, private cases for qualitative error review; never tune Test."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from spam_analysis.io import load_records,save_json
from spam_analysis.metrics import align_scores

p=argparse.ArgumentParser()
p.add_argument('--split',choices=['policy','test'],required=True)
args=p.parse_args()
selection=json.loads(Path('artifacts/policy_selection.json').read_text(encoding='utf-8'))
if args.split=='test' and not Path('artifacts/test_complete.json').exists():
    raise RuntimeError('Test cases may be reviewed only after the frozen evaluation completes')
data=load_records(args.split)
sparse=selection['best_sparse']; semantic=selection['best_semantic']
if not semantic: raise RuntimeError('Semantic comparison is unavailable')
scores={name:align_scores(data[['doc_id']],pd.read_parquet(f'artifacts/predictions/{args.split}_{name}.parquet')).score_spam.to_numpy() for name in [sparse,semantic]}
y=data.label.to_numpy(); a=scores[sparse]>=.5; b=scores[semantic]>=.5
categories=np.select([(a!=y)&(b==y),(a==y)&(b!=y),(a!=y)&(b!=y)],['semantic_repairs','semantic_new_errors','both_wrong'],default='both_right')
result=data[['doc_id','label','template_group_id','subject_clean','body_clean','replacement_ratio','parser_status']].copy()
result['sparse_score']=scores[sparse];result['semantic_score']=scores[semantic];result['comparison']=categories
result['review_category']='unreviewed'
selected=[]
for category in ['semantic_repairs','semantic_new_errors','both_wrong']:
    for label in [0,1]:
        part=result[(result.comparison==category)&(result.label==label)].sort_values('doc_id')
        selected.append(part.sample(min(3,len(part)),random_state=42))
directory=Path('artifacts/private');directory.mkdir(exist_ok=True)
pd.concat(selected).to_parquet(directory/f'{args.split}_semantic_review_cases.parquet',index=False)
summary=result.groupby(['comparison','label']).size().rename('n').reset_index()
summary.to_csv(f'artifacts/tables/{args.split}_complementarity_by_class.csv',index=False)
save_json(f'artifacts/{args.split}_case_review_protocol.json',dict(sparse=sparse,semantic=semantic,threshold=.5,
    selection='Up to three per outcome and class, doc_id sorted then seed42 sample; no cherry-picked successful cases',
    status='Candidate selection only; no invented manual interpretation'))
print(summary.to_string(index=False))
