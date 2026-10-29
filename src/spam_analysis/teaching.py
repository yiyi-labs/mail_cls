"""Small, explicit NumPy lessons, independently checked against sklearn."""
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from .metrics import threshold_table

def tfidf_lesson():
    documents=["课程 通知 课程","免费 优惠","课程 作业 通知","免费 课程 优惠"]
    vocabulary=sorted({token for doc in documents for token in doc.split()})
    counts=np.array([[doc.split().count(token) for token in vocabulary] for doc in documents],dtype=float)
    document_frequency=(counts>0).sum(axis=0)
    idf=np.log((1+len(documents))/(1+document_frequency))+1
    weighted=counts*idf; tfidf=weighted/np.linalg.norm(weighted,axis=1,keepdims=True)
    vectorizer=TfidfVectorizer(tokenizer=str.split,token_pattern=None,lowercase=False,vocabulary=vocabulary,smooth_idf=True,sublinear_tf=False,norm="l2")
    # Only this four-document teaching matrix is deliberately dense.
    expected=vectorizer.fit_transform(documents).toarray()
    np.testing.assert_allclose(tfidf,expected,rtol=1e-12,atol=1e-12)
    return {"documents":documents,"vocabulary":vocabulary,"counts":counts,"document_frequency":document_frequency,"idf":idf,"tfidf":tfidf,"cosine_matrix":tfidf@tfidf.T,"sklearn_equal":True}

def threshold_benchmark(n=5000,seed=42):
    rng=np.random.default_rng(seed); y=rng.integers(0,2,n); scores=np.round(rng.random(n),3)
    start=time.perf_counter(); vector=threshold_table(y,scores); vector_s=time.perf_counter()-start
    start=time.perf_counter(); reference=[]
    for threshold in vector.threshold:
        pred=scores>=threshold
        reference.append([np.sum((y==0)&(~pred)),np.sum((y==0)&pred),np.sum((y==1)&(~pred)),np.sum((y==1)&pred)])
    loop_s=time.perf_counter()-start
    np.testing.assert_array_equal(vector[["tn","fp","fn","tp"]].to_numpy(),reference)
    return dict(n=n,thresholds=len(vector),vector_s=vector_s,loop_s=loop_s,measured_ratio=loop_s/vector_s,correct=True,
                interpretation="One local run; ratio is not a hardware-independent speed guarantee")
