"""Training-only descriptive tables shared by the UI and notebooks."""
import numpy as np
import pandas as pd


def training_frame(records,manifest):
    if not manifest.doc_id.is_unique or not records.doc_id.is_unique:raise ValueError('邮件编号必须唯一')
    train=manifest.loc[manifest.split.eq('train'),['doc_id']]
    frame=train.merge(records,on='doc_id',how='left',validate='one_to_one',indicator=True)
    if not frame['_merge'].eq('both').all():raise ValueError('训练记录不完整')
    return frame.drop(columns='_merge')


def grouped_distribution(frame,field):
    values=frame[field].to_numpy(dtype=float)
    # All observations, including zero and the maximum, enter exactly one bin.
    if field.endswith('_ratio'):edges=np.linspace(0,1,11)
    else:edges=np.unique(np.r_[0,np.ceil(np.expm1(np.linspace(0,np.log1p(max(float(values.max()),1)),11)))])
    if len(edges)<2:edges=np.array([0,1])
    output=[]
    for label,name in [(0,'正常邮件'),(1,'垃圾邮件')]:
        subset=frame.loc[frame.label.eq(label),field].to_numpy(dtype=float)
        counts,_=np.histogram(subset,bins=edges)
        for i,count in enumerate(counts):
            lo,hi=edges[i:i+2];right=']' if i==len(counts)-1 else ')'
            bucket=f'[{lo:.0%}, {hi:.0%}{right}' if field.endswith('_ratio') else f'[{lo:g}, {hi:g}{right}'
            output.append({'类别':name,'区间':bucket,'顺序':i,'邮件数':int(count),'类内比例':float(count/max(len(subset),1)),'分母':len(subset)})
    return pd.DataFrame(output)


def english_terms(frame):
    from sklearn.feature_extraction.text import CountVectorizer
    vec=CountVectorizer(binary=True,min_df=3,max_features=20000,stop_words='english')
    x=vec.fit_transform(frame.body_clean)
    ham=np.asarray(x[frame.label.eq(0).to_numpy()].mean(axis=0)).ravel()
    spam=np.asarray(x[frame.label.eq(1).to_numpy()].mean(axis=0)).ravel()
    return pd.DataFrame({'term':vec.get_feature_names_out(),'ham_rate':ham,'spam_rate':spam,'spam_minus_ham':spam-ham})


def corpus_terms(frame):
    """One training-only vocabulary over Chinese and Latin text, with class denominators."""
    from sklearn.feature_extraction.text import CountVectorizer
    from spam_analysis.features import tokenize
    if 'split' in frame and not frame.split.eq('train').all():raise ValueError('词频统计只使用训练集')
    vec=CountVectorizer(tokenizer=tokenize,token_pattern=None,lowercase=True,binary=True,min_df=3,max_features=30000)
    x=vec.fit_transform(frame.body_clean)
    terms=vec.get_feature_names_out();keep=np.array([len(t)>1 and any(c.isalpha() for c in t) for t in terms])
    ham=np.asarray(x[frame.label.eq(0).to_numpy()].mean(axis=0)).ravel()
    spam=np.asarray(x[frame.label.eq(1).to_numpy()].mean(axis=0)).ravel()
    return pd.DataFrame({'term':terms[keep],'ham_rate':ham[keep],'spam_rate':spam[keep],'spam_minus_ham':(spam-ham)[keep]})
