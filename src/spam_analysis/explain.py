"""Actual linear feature contributions, never generated explanations."""
import numpy as np
import pandas as pd

def linear_contributions(artifact,record,top_k=12):
    X=artifact.features.transform(record)
    if artifact.ratio is not None: X=artifact.ratio.transform(X)
    clf=artifact.estimator
    if hasattr(clf,"coef_"):
        weights=clf.coef_[0]; intercept=float(clf.intercept_[0])
    else:
        weights=clf.feature_log_prob_[1]-clf.feature_log_prob_[0]
        intercept=float(clf.class_log_prior_[1]-clf.class_log_prior_[0])
    names=artifact.features.get_feature_names_out(); row=X.getrow(0)
    values=row.data*weights[row.indices]
    table=pd.DataFrame({"feature":names[row.indices],"value":row.data,"contribution":values})
    table["direction"]=np.where(table.contribution>=0,"spam","ham")
    table=table.iloc[np.argsort(-np.abs(values))[:top_k]]
    reconstructed=float(values.sum()+intercept)
    actual=float(artifact.decision_function(record)[0])
    if record.lexical_text.iloc[0]: np.testing.assert_allclose(reconstructed,actual,atol=1e-5)
    return table,{"score_space":"uncalibrated linear decision/log-odds", "intercept":intercept,"all_features_sum_plus_intercept":reconstructed,"actual_score":actual,
                  "note":"Overlapping n-grams are correlated contributions, not independent causal evidence; not an explanation of a semantic/fused branch"}
