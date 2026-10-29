"""Exact sparse feature products and TF-IDF ingredients for interactive teaching."""
from collections import Counter
import numpy as np


def linear_steps(artifact,record,limit=24):
    X=artifact.features.transform(record)
    if artifact.ratio is not None:X=artifact.ratio.transform(X)
    row=X.getrow(0);estimator=artifact.estimator
    if hasattr(estimator,'coef_'):
        weights=estimator.coef_[0];intercept=float(estimator.intercept_[0]);kind='linear'
    else:
        weights=estimator.feature_log_prob_[1]-estimator.feature_log_prob_[0]
        intercept=float(estimator.class_log_prior_[1]-estimator.class_log_prior_[0]);kind='naive_bayes_log_odds'
    products=row.data*weights[row.indices]
    selected=np.argsort(-np.abs(products))[:limit]
    names=artifact.features.get_feature_names_out();items=[]
    ingredients={};offset=0
    for block,field,vectorizer,block_weight in artifact.features.transforms_:
        counts=Counter(vectorizer.build_analyzer()(str(record[field].iloc[0])))
        vocabulary=vectorizer.get_feature_names_out()
        for position in selected:
            feature=int(row.indices[position])-offset
            if 0<=feature<len(vocabulary):
                term=vocabulary[feature];count=int(counts[term])
                ingredients[int(row.indices[position])]={'block':block,'term':term,'count':count,
                    'sublinear_tf':float(1+np.log(count)) if count else 0.,
                    'idf':float(vectorizer.idf_[feature]),'block_weight':float(block_weight)}
        offset+=len(vocabulary)
    for position in selected:
        index=int(row.indices[position])
        items.append({'feature':str(names[index]),'x':float(row.data[position]),'weight':float(weights[index]),
                      'product':float(products[position]),**ingredients.get(index,{})})
    omitted=float(products.sum()-products[selected].sum())
    score=float(products.sum()+intercept)
    return {'kind':kind,'features':items,'intercept':intercept,'remaining_contribution':omitted,
            'reconstructed_score':score,'total_active':int(row.nnz),
            'selection':'top absolute products, not computation order; all omitted products included as one remainder',
            'formula':'z = intercept + sum(x_j * w_j); x includes TF-IDF L2 normalization and block weight'}
