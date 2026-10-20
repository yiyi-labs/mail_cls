"""Every figure is produced from saved data; no manually entered scores."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt,font_manager
from .io import load_records,save_json
from .features import word_vectorizer

def style():
    candidates=["Microsoft YaHei","SimHei","Noto Sans CJK SC"]
    available={f.name for f in font_manager.fontManager.ttflist}
    chosen=next((n for n in candidates if n in available),None)
    if chosen: plt.rcParams["font.sans-serif"]=[chosen]
    plt.rcParams.update({"axes.unicode_minus":False,"figure.dpi":130,"savefig.dpi":180,"axes.spines.top":False,"axes.spines.right":False})
    Path("artifacts/figures").mkdir(parents=True,exist_ok=True)
    return chosen

def savefig(name):
    plt.tight_layout(); plt.savefig(f"artifacts/figures/{name}.png",bbox_inches="tight"); plt.close()

def eda(config):
    font=style(); df=load_records("train"); Path("artifacts/tables").mkdir(parents=True,exist_ok=True)
    n=len(df); cls=df.label.map({0:"正常",1:"垃圾"})
    counts=cls.value_counts(); counts.to_csv("artifacts/tables/train_class_counts.csv",encoding="utf-8-sig")
    counts.reindex(['正常','垃圾']).plot.bar(color=["#2563eb","#ea580c"],rot=0,figsize=(6,4)); plt.xlabel('类别'); plt.ylabel("邮件数"); plt.title(f"训练集类别分布｜n={n:,}"); savefig("01_class_balance")
    q=df.groupby("label").agg(n=("doc_id","size"),body_missing=("body_missing","sum"),mean_body=("body_chars","mean"),median_body=("body_chars","median"),replacement_rows=("replacement_ratio",lambda s:int(s.gt(0).sum())))
    q.to_csv("artifacts/tables/train_quality.csv",encoding="utf-8-sig")
    df.groupby("label")[["body_chars","subject_chars","chinese_ratio","digit_ratio","url_count"]].describe(percentiles=[.5,.9,.95,.99]).to_csv("artifacts/tables/train_distributions.csv",encoding="utf-8-sig")
    plt.figure(figsize=(7,4))
    for label,name in [(0,"正常"),(1,"垃圾")]:
        vals=np.sort(df.loc[df.label==label,"body_chars"]); plt.plot(vals,np.arange(1,len(vals)+1)/len(vals),label=f"{name} n={len(vals):,}")
    plt.xscale("symlog",linthresh=1); plt.xlabel("正文字符数（对称对数坐标）"); plt.ylabel("累计比例"); plt.title(f"训练集长度长尾｜n={n:,}"); plt.legend(); savefig("02_lengths")
    sizes=df.groupby("template_group_id").size(); sizes.value_counts().sort_index().to_csv("artifacts/tables/train_group_sizes.csv")
    group_edges=np.array([1,2,5,10,20,50,100,200,500,1000,2000])
    group_edges=np.r_[group_edges[group_edges<=sizes.max()],sizes.max()+1]
    plt.figure(figsize=(7,4)); plt.hist(sizes,bins=group_edges,color="#2563eb",edgecolor='white',linewidth=.6)
    plt.xscale('log'); plt.yscale("log"); plt.xlabel("每组邮件数（对数；不等宽区间）"); plt.ylabel("区间内模板组数（对数）"); plt.title(f"训练集模板重复｜{len(sizes):,} 组 / {n:,} 封"); savefig("03_groups")
    cross=pd.crosstab(df.url_count.gt(0),cls); cross.to_csv("artifacts/tables/train_url_crosstab.csv",encoding="utf-8-sig")
    bins=np.unique(np.quantile(df.body_chars,[0,.25,.5,.75,.9,1])); bins[-1]+=1
    save_json("artifacts/length_bins.json",bins.tolist()); df["length_bin"]=pd.cut(df.body_chars,bins,include_lowest=True).astype(str)
    df.pivot_table(index="length_bin",columns="label",values="url_count",aggfunc=["mean","count"]).to_csv("artifacts/tables/train_structure_pivot.csv",encoding="utf-8-sig")
    vec=word_vectorizer(); X=vec.fit_transform(df.lexical_text); names=vec.get_feature_names_out()
    freq=[]
    for label in [0,1]:
        subset=X[df.label.to_numpy()==label]; dfreq=np.asarray((subset>0).sum(axis=0)).ravel(); rate=dfreq/subset.shape[0]
        for j in np.argsort(-rate)[:100]: freq.append({"label":label,"term":names[j],"document_frequency":int(dfreq[j]),"within_class_document_rate":float(rate[j])})
    freq=pd.DataFrame(freq); freq.to_csv("artifacts/tables/train_keywords.csv",index=False,encoding="utf-8-sig")
    fig,axes=plt.subplots(1,2,figsize=(12,6))
    for ax,label,name in zip(axes,[0,1],["正常","垃圾"]):
        top=freq[freq.label==label].head(14).iloc[::-1]; ax.barh(top.term,top.within_class_document_rate,color="#2563eb" if label==0 else "#ea580c")
        ax.set_title(f"{name}类内文档频率"); ax.set_xlabel("含该词的邮件比例")
    fig.suptitle(f"训练集高频词项（含标点；不等于区分词）｜n={n:,}"); savefig("04_keywords")
    quality_plot=q[['body_missing','replacement_rows']].rename(columns={'body_missing':'正文为空','replacement_rows':'含损坏替换字符'},index={0:'正常',1:'垃圾'})
    quality_plot.plot.bar(rot=0,figsize=(7,4)); plt.xlabel('类别'); plt.title(f"训练集解析质量｜n={n:,}"); plt.ylabel("邮件数"); savefig("05_quality")
    save_json("artifacts/eda_summary.json",dict(split="train",rows=n,font=font,majority_baseline=float(df.label.mean()),
              length_quantiles=df.body_chars.quantile([.5,.9,.99]).to_dict(),word_cap=vec.eligible_features_,
              explanation="All feature-forming EDA uses Train. Raw-structure audit uses full corpus separately. Keywords report class-normalized document frequency."))
    print("EDA saved from Train only",n,flush=True)
