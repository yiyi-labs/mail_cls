"""Release choices stay on development data and handle tied probabilities."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from spam_analysis.metrics import metrics

spec=importlib.util.spec_from_file_location('trained_release',Path(__file__).resolve().parents[1]/'scripts/publish_trained_models.py')
release=importlib.util.module_from_spec(spec);spec.loader.exec_module(release)


def test_threshold_matches_brute_force_with_ties_and_endpoints():
    y=np.array([0,1,0,1,1,0,1]);p=np.array([0.,0.,.3,.3,1.,1.,.8])
    thresholds=np.r_[np.nextafter(p.max(),np.inf),np.unique(p)]
    expected=max(thresholds,key=lambda t:(metrics(y,p,t)['accuracy'],-metrics(y,p,t)['ham_fpr'],metrics(y,p,t)['macro_f1'],t))
    assert release.best_threshold(y,p)==expected


def test_release_refuses_test_or_mixed_membership():
    release.development_only(pd.DataFrame({'split':['policy']*2}),'policy')
    for actual,expected in [(['test'],'policy'),(['policy','test'],'policy'),(['test'],'test')]:
        with pytest.raises(ValueError):release.development_only(pd.DataFrame({'split':actual}),expected)
