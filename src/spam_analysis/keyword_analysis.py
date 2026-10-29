"""Separate total occurrences, document frequency and discriminative frequency."""
import re
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from .io import load_records
from .features import tokenize
from .plots import style,savefig
from matplotlib import pyplot as plt

def run():
    train=load_records("train"); artifact=joblib.load("artifacts/models/B2.joblib")
    _,_,vec,_=artifact.features.transforms_[0]; names=vec.get_feature_names_out()
    count=CountVectorizer(tokenizer=tokenize,token_pattern=None,lowercase=False,ngram_range=(1,2),vocabulary=vec.vocabulary_)
    X=count.transform(train.lexical_text); rows=[]; rates=[]
    for label in [0,1]:
        M=X[train.label.to_numpy()==label]; occurrence=np.asarray(M.sum(axis=0)).ravel(); df=np.asarray((M>0).sum(axis=0)).ravel(); rate=df/M.shape[0]; rates.append(rate)
        # Save full term-level aggregate, no mail-level text.
        for j in range(len(names)): rows.append((label,names[j],int(occurrence[j]),int(df[j]),float(rate[j])))
    table=pd.DataFrame(rows,columns=["label","term","total_occurrences","document_frequency","within_class_document_rate"])
    # Full vocabulary can contain addresses/contact fragments; keep the full
    # aggregate local. Only curated aggregate Chinese terms are public figures.
    Path('artifacts/private').mkdir(parents=True,exist_ok=True)
    table.to_csv("artifacts/private/train_term_counts.csv",index=False,encoding="utf-8-sig")
    diff=rates[1]-rates[0]
    terms=pd.DataFrame({"term":names,"ham_rate":rates[0],"spam_rate":rates[1],"spam_minus_ham":diff})
    # Plot readability filter only; the trained vocabulary remains unchanged.
    valid=terms.term.map(lambda x:bool(re.fullmatch(r"[\u4e00-\u9fff]{2,12}",x)))
    selected=pd.concat([terms[valid].nlargest(12,"spam_minus_ham"),terms[valid].nsmallest(12,"spam_minus_ham")])
    selected.to_csv("artifacts/tables/train_discriminative_terms.csv",index=False,encoding="utf-8-sig")
    style(); fig,axes=plt.subplots(1,2,figsize=(12,6))
    for ax,part,name,color in [(axes[0],selected.iloc[12:],"正常侧","#2563eb"),(axes[1],selected.iloc[:12],"垃圾侧","#ea580c")]:
        part=part.iloc[::-1]; ax.barh(part.term,part.spam_minus_ham.abs(),color=color); ax.set_title(name); ax.set_xlabel("两类内文档频率之差的绝对值")
    fig.suptitle(f"训练集区分词｜n={len(train):,}；仅绘图筛选中文词，模型不删词")
    savefig("06_discriminative_terms")

if __name__=="__main__":run()
