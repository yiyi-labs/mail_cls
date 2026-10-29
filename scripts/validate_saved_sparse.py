"""Validate real saved candidates with labels and collection metadata removed."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import joblib
import numpy as np
from spam_analysis.io import load_records,save_json
from spam_analysis.features import NUMERIC_FEATURES,TEXT_FEATURES
from spam_analysis.explain import linear_contributions

registry=json.loads(Path('artifacts/sparse_registry.json').read_text(encoding='utf-8'))
chosen=json.loads(Path('artifacts/sparse_selection.json').read_text(encoding='utf-8'))['best']
rows=load_records('tune').sample(32,random_state=42)
allowed=rows[['doc_id',*TEXT_FEATURES,*NUMERIC_FEATURES]].copy()
assert 'label' not in allowed and 'template_group_id' not in allowed
checks=[]
for name in dict.fromkeys(['B1','B2',chosen]):
    model=joblib.load(registry[name]['path'])
    original=model.decision_function(rows)
    no_labels=model.decision_function(allowed)
    np.testing.assert_allclose(original,no_labels,rtol=0,atol=1e-12)
    repeat=joblib.load(registry[name]['path']).decision_function(allowed)
    np.testing.assert_allclose(original,repeat,rtol=0,atol=1e-12)
    explanation_checks=[]
    for label in [0,1]:
        example=rows[rows.label.eq(label)].iloc[:1]
        if len(example):
            _,detail=linear_contributions(model,example)
            explanation_checks.append(dict(label=label,margin_reconstruction_delta=abs(detail['actual_score']-detail['all_features_sum_plus_intercept'])))
    checks.append(dict(model=name,rows=len(rows),max_delta=float(np.max(np.abs(original-repeat))),
        no_labels_or_collection_metadata=True,reload_equal=True,actual_explanation_checks=explanation_checks))
save_json('artifacts/saved_sparse_validation.json',dict(passed=True,checks=checks))
print('Saved sparse models: content-only inference and reload passed',list(dict.fromkeys(['B1','B2',chosen])))
