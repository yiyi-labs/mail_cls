from functools import lru_cache
from pathlib import Path
import jieba
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator,TransformerMixin

NUMERIC_FEATURES=["subject_chars","body_chars","line_count","chinese_ratio","digit_ratio","latin_ratio","url_count", "email_address_count","exclamation_count","question_count","repeated_punctuation_ratio","actual_html_tag_count","actual_link_count","subject_missing","body_missing"]
TEXT_FEATURES=["subject_clean","body_clean","lexical_text","semantic_text"]
Path(".cache").mkdir(exist_ok=True)
jieba.dt.tmp_dir=str(Path(".cache").resolve())
jieba.setLogLevel(30)

@lru_cache(maxsize=70000)
def tokenize(text):
    return [t for t in jieba.cut(text,HMM=False) if t.strip()]

class AuditedTfidfVectorizer(TfidfVectorizer):
    def _limit_features(self,X,vocabulary,high=None,low=None,limit=None):
        df=np.asarray((X!=0).sum(axis=0)).ravel()
        valid=np.ones(len(df),dtype=bool)
        if high is not None: valid &= df<=high
        if low is not None: valid &= df>=low
        self.eligible_features_=int(valid.sum())
        return super()._limit_features(X,vocabulary,high,low,limit)

def word_vectorizer(max_features=100000):
    return AuditedTfidfVectorizer(tokenizer=tokenize,token_pattern=None,lowercase=False,ngram_range=(1,2),min_df=2,
                           sublinear_tf=True,max_features=max_features,dtype=np.float32)

class Blocks(BaseEstimator,TransformerMixin):
    """All fitted transforms are trained only on the caller's training rows."""
    def __init__(self, word=True,char=False,subject=False,numeric=False,baseline=False,subject_weight=1.0,char_range=(2,4),naive=False,word_max_features=100000,char_max_features=200000):
        self.word=word; self.char=char; self.subject=subject; self.numeric=numeric; self.baseline=baseline
        self.subject_weight=subject_weight; self.char_range=char_range; self.naive=naive
        self.word_max_features=word_max_features; self.char_max_features=char_max_features
    def fit(self,df,y=None):
        self.transforms_=[]; self.cap_audit_={}
        if self.word:
            field="naive_text" if self.naive else "lexical_text" if self.baseline else "body_clean"
            self.transforms_.append(("word",field,word_vectorizer(self.word_max_features),1.0))
        if self.char:
            self.transforms_.append(("char","body_clean",AuditedTfidfVectorizer(analyzer="char",ngram_range=self.char_range,min_df=2,
                                       max_features=self.char_max_features,sublinear_tf=True,dtype=np.float32,lowercase=False),1.0))
        if self.subject:
            self.transforms_.append(("subject","subject_clean",word_vectorizer(30000),self.subject_weight))
        for name,field,vec,weight in self.transforms_:
            vec.fit(df[field]); eligible=vec.eligible_features_
            self.cap_audit_[name]={"dimensions":len(vec.vocabulary_),"eligible":eligible,"cap":vec.max_features,
                                  "truncated_fraction":1-len(vec.vocabulary_)/max(eligible,1),"hit_cap":len(vec.vocabulary_)>=vec.max_features}
        if self.numeric: self.scaler_=StandardScaler().fit(np.log1p(df[NUMERIC_FEATURES].to_numpy(dtype=float)))
        return self
    def transform(self,df):
        blocks=[vec.transform(df[field])*weight for _,field,vec,weight in self.transforms_]
        if self.numeric: blocks.append(sparse.csr_matrix(self.scaler_.transform(np.log1p(df[NUMERIC_FEATURES].to_numpy(dtype=float))),dtype=np.float32))
        return sparse.hstack(blocks,format="csr",dtype=np.float32)
    def get_feature_names_out(self):
        names=[f"{name}:{token}" for name,_,vec,_ in self.transforms_ for token in vec.get_feature_names_out()]
        if self.numeric: names.extend("structure:"+x for x in NUMERIC_FEATURES)
        return np.array(names,dtype=object)

class NBRatio(BaseEstimator,TransformerMixin):
    def __init__(self,alpha=1.0): self.alpha=alpha
    def fit(self,X,y):
        B=X.copy(); B.data[:]=1; y=np.asarray(y)
        p=self.alpha+np.asarray(B[y==1].sum(axis=0)).ravel(); q=self.alpha+np.asarray(B[y==0].sum(axis=0)).ravel()
        self.r_=np.log((p/p.sum())/(q/q.sum())); return self
    def transform(self,X):
        B=X.copy(); B.data[:]=1
        return B.multiply(self.r_).tocsr()

class SparseArtifact:
    def __init__(self,features,estimator,prior,ratio=None):
        self.features=features; self.estimator=estimator; self.prior=prior; self.ratio=ratio
        self.classes_=np.array([0,1])
    def decision_function(self,df):
        X=self.features.transform(df)
        if self.ratio is not None: X=self.ratio.transform(X)
        if hasattr(self.estimator,"decision_function"): score=self.estimator.decision_function(X)
        else:
            logp=self.estimator.predict_log_proba(X); score=logp[:,1]-logp[:,0]
        empty=df.lexical_text.eq("").to_numpy()
        score[empty]=np.log(self.prior/(1-self.prior))
        return score
    def predict_scores(self,df):
        import pandas as pd
        return pd.DataFrame({"doc_id":df.doc_id.to_numpy(),"score_spam":self.decision_function(df)})
