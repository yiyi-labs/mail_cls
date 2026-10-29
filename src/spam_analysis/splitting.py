from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from .io import save_json,sha256

def assert_disjoint(manifest):
    assert manifest.doc_id.is_unique
    assert manifest.groupby("template_group_id").split.nunique().max()==1

def make_splits(records,groups,config):
    if Path("artifacts/evaluation_plan.json").exists():
        raise RuntimeError("Evaluation already frozen; cannot replace splits")
    df=records[["doc_id","label"]].merge(groups.drop(columns="grouping_text"),on="doc_id",validate="one_to_one")
    df["fold"]=-1
    cv=StratifiedGroupKFold(n_splits=20,shuffle=True,random_state=config["seed"])
    for fold,(_,valid) in enumerate(cv.split(np.zeros(len(df)),df.label,df.template_group_id)):
        df.loc[valid,"fold"]=fold
    df["split"]=np.select([df.fold<11,df.fold<13,df.fold<15,df.fold<17],["train","tune","calibrate","policy"],default="test")
    dev_groups=set(df.loc[df.doc_id.isin(config["development_examples"]),"template_group_id"])
    moved=df.template_group_id.isin(dev_groups)&df.split.ne("train")
    moved_count=int(moved.sum()); df.loc[moved,"split"]="train"
    assert_disjoint(df)
    edges=pd.read_parquet("artifacts/splits/near_edges.parquet")
    byid=df.set_index("doc_id")
    cross=sum(byid.loc[r.left,"split"]!=byid.loc[r.right,"split"] for r in edges.itertuples())
    assert cross==0
    df.to_parquet("artifacts/splits/split_manifest.parquet",index=False)
    summary=df.groupby("split").agg(rows=("doc_id","size"),groups=("template_group_id","nunique"),spam=("label","sum"))
    summary["ham"]=summary.rows-summary.spam; summary["spam_ratio"]=summary.spam/summary.rows
    summary["largest_group"]=df.groupby(["split","template_group_id"]).size().groupby("split").max()
    summary.to_csv("artifacts/split_summary.csv",encoding="utf-8-sig")
    save_json("artifacts/split_checks.json",dict(group_overlap=0,verified_near_edge_crossings=cross,forced_development_rows=moved_count,
             manifest_sha256=sha256("artifacts/splits/split_manifest.parquet"),seed=config["seed"],test_semantics_viewed=False))
    print(summary.to_string(),flush=True)
    return df
