import json
from pathlib import Path
import numpy as np
import pandas as pd
from .io import load_records,save_json
from .metrics import threshold_table,metrics,align_scores

def operating_points(y,p,limits=(.005,.01)):
    table=threshold_table(y,p)
    research=table.sort_values(["macro_f1","ham_fpr","threshold"],ascending=[False,True,False],kind="stable").iloc[0]
    points={"default":.5,"research":float(research.threshold)}
    for limit in limits:
        feasible=table[table.ham_fpr<=limit]
        best=feasible.sort_values(["spam_recall","ham_fpr","threshold"],ascending=[False,True,False],kind="stable").iloc[0]
        points[f"fpr_{limit}"]=float(best.threshold)
    return points,table


def choose_main(research,names,sparse,minimum_gain=.0005):
    """Apply both safeguards before ranking, so an unsafe winner cannot hide
    another eligible candidate. Input order supplies the simplicity tie break.
    """
    candidates=research.loc[names].copy()
    candidates['gain_vs_sparse']=candidates.macro_f1-research.loc[sparse,'macro_f1']
    candidates['eligible']=(candidates.gain_vs_sparse>=minimum_gain)&(candidates.ham_fpr<=research.loc[sparse,'ham_fpr'])
    eligible=candidates[candidates.eligible].sort_values(['macro_f1','ham_fpr'],ascending=[False,True],kind='stable')
    return (eligible.index[0] if len(eligible) else sparse),candidates

def select_policy(config):
    if Path("artifacts/evaluation_plan.json").exists(): raise RuntimeError("Already frozen")
    policy=load_records("policy"); entries=json.loads(Path("artifacts/calibrated_registry.json").read_text(encoding="utf-8"))
    sparse=json.loads(Path("artifacts/sparse_selection.json").read_text(encoding="utf-8"))["best"]
    semantic=[k for k,v in entries.items() if v["type"]=="semantic"]
    best_semantic=max(semantic,key=lambda k:entries[k]["metrics"]["macro_f1"]) if semantic else None
    arrays={k:align_scores(policy[["doc_id"]],pd.read_parquet(f"artifacts/predictions/policy_{k}.parquet")).score_spam.to_numpy() for k in entries}
    specifications={k:{"weights":{k:1.0}} for k in arrays}
    if best_semantic:
        for alpha in config["alpha_candidates"]:
            name=f"F_{alpha}"
            arrays[name]=(1-alpha)*arrays[sparse]+alpha*arrays[best_semantic]
            specifications[name]={"weights":{sparse:1-alpha,best_semantic:alpha}}
    rows=[]
    for name,p in arrays.items():
        points,table=operating_points(policy.label,p,config["ham_fpr_limits"])
        specifications[name]["thresholds"]=points
        table.to_csv(f"artifacts/tables/threshold_{name}.csv",index=False)
        for point,threshold in points.items(): rows.append(dict(model=name,point=point,**metrics(policy.label,p,threshold)))
    results=pd.DataFrame(rows); results.to_csv("artifacts/policy_results.csv",index=False)
    research=results[results.point=="research"].set_index("model")
    names=[sparse]+([best_semantic]+[f"F_{a}" for a in config["alpha_candidates"] if 0<a<1] if best_semantic else [])
    chosen,candidates=choose_main(research,names,sparse)
    candidates.to_csv('artifacts/policy_candidate_eligibility.csv',index=True)
    if best_semantic:
        s=arrays[sparse]>=.5; e=arrays[best_semantic]>=.5; y=policy.label.to_numpy()
        save_json("artifacts/complementarity_policy.json",dict(sparse=sparse,semantic=best_semantic,n=len(y),
                  semantic_repairs=int(((s!=y)&(e==y)).sum()),semantic_new_errors=int(((s==y)&(e!=y)).sum()),
                  both_wrong=int(((s!=y)&(e!=y)).sum()),both_right=int(((s==y)&(e==y)).sum()),threshold=.5))
    save_json("artifacts/policy_selection.json",dict(main=chosen,best_sparse=sparse,best_semantic=best_semantic,specifications=specifications,
              rationale="First require gain >=0.0005 over sparse and no ham_FPR increase, then rank by Macro-F1 / lower FPR / simpler model; retain sparse if none qualify; alpha endpoints included; Policy only"))
    print("Selected policy:",chosen,specifications[chosen],flush=True)
