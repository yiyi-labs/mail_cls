"""Freeze identities and execute the once-opened Test evaluation."""
import datetime
import json
import subprocess
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .io import load_records,save_json,sha256,read_config
from .metrics import metrics,counts,from_counts,align_scores
from .calibration import calibrated_scores
from .robustness import RULES,perturb,paired_metrics

def source_identity():
    paths=sorted(Path("src/spam_analysis").glob("*.py"))+sorted(Path("configs").glob("*.yaml"))
    return {str(p).replace("\\","/"):sha256(p) for p in paths}

def freeze_evaluation(config):
    target=Path("artifacts/evaluation_plan.json")
    if target.exists(): raise RuntimeError("Evaluation plan already exists; do not silently overwrite")
    selection=json.loads(Path("artifacts/policy_selection.json").read_text(encoding="utf-8"))
    entries=json.loads(Path("artifacts/calibrated_registry.json").read_text(encoding="utf-8"))
    formal=list(entries)
    main=selection["main"]
    if main not in formal: formal.append(main)
    # Include all finite fusion weights as pre-registered comparisons; main is
    # chosen on Policy. No final selection uses these Test results.
    formal+= [k for k in selection["specifications"] if k.startswith("F_") and k not in formal]
    git=subprocess.run(['git','rev-parse','HEAD'],text=True,capture_output=True)
    plan={"frozen_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"main":main,"comparisons":formal,
          "registry":entries,"specifications":selection["specifications"],"zip_sha256":read_config("configs/data.yaml")["expected_sha256"],
          "manifest_sha256":sha256("artifacts/splits/split_manifest.parquet"),"source_hashes":source_identity(),
          "processed_sha256":sha256("data/processed/records.parquet"),"seed":config["seed"],"bootstrap_repeats":config["bootstrap_repeats"],
          "metrics":["accuracy","spam_precision","spam_recall","spam_f1","ham_precision","ham_recall","ham_f1","macro_f1","balanced_accuracy","ham_fpr","average_precision","roc_auc","brier","tn","fp","fn","tp"],
          "points":["default","research","fpr_0.005","fpr_0.01"],"robustness_rules":RULES,
          "robustness_sampling":"all Test mail for selected main; full data denominator, each mutation paired",
          "error_selection":"high-confidence FP/FN; boundary; seed42 random errors; full error table first",
          "git_commit":git.stdout.strip() if git.returncode==0 else None,
          "git_note":"User requested no further commits; source SHA-256 binds current uncommitted implementation", "test_previously_evaluated":False}
    for name,entry in entries.items():
        assert sha256(entry["path"])==entry["model_sha256"]
        assert sha256(entry["calibrator_path"])==entry["calibrator_sha256"]
        if entry['type']=='semantic':
            from .model_assets import verify_identity
            verify_identity(entry['identity'])
    save_json(target,plan)
    print("Frozen evaluation plan",sha256(target),flush=True)

def group_bootstrap(df,arrays,specs,main,repeats=1000,seed=42):
    codes,unique=pd.factorize(df.template_group_id,sort=True); g=len(unique); y=df.label.to_numpy(); aggregated={}
    for name,p in arrays.items():
        pred=p>=specs[name]["thresholds"]["research"]
        masks=[(y==0)&(~pred),(y==0)&pred,(y==1)&(~pred),(y==1)&pred]
        aggregated[name]=np.stack([np.bincount(codes,weights=m.astype(float),minlength=g) for m in masks],axis=1)
    rng=np.random.default_rng(seed); samples={name:[] for name in arrays}; missing_class=0
    for _ in range(repeats):
        weights=np.bincount(rng.integers(0,g,g),minlength=g)
        for name,c in aggregated.items():
            tn,fp,fn,tp=weights@c
            if (tn+fp)==0 or (fn+tp)==0:
                if name==main: missing_class+=1
                samples[name].append(None)
            else: samples[name].append(from_counts(tn,fp,fn,tp))
    results={}
    for name,values in samples.items():
        valid=[v for v in values if v is not None]
        ci={k:np.quantile([v[k] for v in valid],[.025,.975]).tolist() for k in ["accuracy","macro_f1","ham_fpr","spam_recall"]}
        paired=[a["macro_f1"]-b["macro_f1"] for a,b in zip(samples[main],values) if a is not None and b is not None]
        results[name]={"ci95":ci,"main_minus_model_macro_f1_ci95":np.quantile(paired,[.025,.975]).tolist()}
    return dict(groups=g,repeats=repeats,missing_class_resamples=missing_class,models=results,
                limitation="Groups, not individual messages, are resampled. Intervals rely on grouping quality; zero-event bootstrap intervals can degenerate and do not imply zero population risk.")

def error_tables(df,p,threshold,split):
    pred=(p>=threshold).astype(int); y=df.label.to_numpy()
    out=df[["doc_id","label","template_group_id","body_chars","subject_chars","parser_status","replacement_ratio","fallback_reason"]].copy()
    out["score_spam"]=p; out["prediction"]=pred; out["threshold"]=threshold
    out["error_type"]=np.select([(y==0)&(pred==1),(y==1)&(pred==0)], ["FP","FN"],default="correct")
    bins=json.loads(Path("artifacts/length_bins.json").read_text(encoding="utf-8")); bins[0]=-float("inf"); bins[-1]=float("inf")
    out["length_bin"]=pd.cut(out.body_chars,bins).astype(str); out["manual_error_category"]="unreviewed"
    Path("artifacts/private").mkdir(parents=True,exist_ok=True)
    out.to_parquet(f"artifacts/predictions/{split}_error_analysis_all.parquet",index=False)
    errors=out[out.error_type.ne("correct")]
    errors.merge(df[["doc_id","subject_clean","body_clean"]],on="doc_id",validate="one_to_one").to_parquet(f"artifacts/private/{split}_errors_with_text.parquet",index=False)
    selection=pd.concat([errors[errors.error_type.eq("FP")].nlargest(5,"score_spam"),errors[errors.error_type.eq("FN")].nsmallest(5,"score_spam"),
                         errors.assign(distance=(errors.score_spam-threshold).abs()).nsmallest(5,"distance"),errors.sample(min(5,len(errors)),random_state=42)]).drop_duplicates("doc_id")
    selection.to_csv(f"artifacts/private/{split}_selected_errors.csv",index=False,encoding="utf-8-sig")
    for group in ["length_bin","parser_status","error_type"]:
        tab=out.assign(error=out.error_type.ne("correct")).groupby(group).agg(n=("doc_id","size"),errors=("error","sum"))
        tab["error_rate"]=tab.errors/tab.n; tab.to_csv(f"artifacts/tables/{split}_errors_by_{group}.csv",encoding="utf-8-sig")
    return out

def evaluate_frozen(plan_path):
    plan=json.loads(Path(plan_path).read_text(encoding="utf-8"))
    if Path("artifacts/test_opened.json").exists():
        raise RuntimeError("Test already opened: use saved predictions for reports. A bug-driven rerun requires documented invalidation.")
    assert sha256("artifacts/splits/split_manifest.parquet")==plan["manifest_sha256"]
    assert sha256("data/processed/records.parquet")==plan["processed_sha256"]
    assert source_identity()==plan["source_hashes"],"Code/config changed after freeze"
    assert sha256(read_config("configs/data.yaml")["zip_path"])==plan["zip_sha256"]
    for name,entry in plan["registry"].items():
        assert sha256(entry["path"])==entry["model_sha256"]
        assert sha256(entry["calibrator_path"])==entry["calibrator_sha256"]
    save_json("artifacts/test_opened.json",{"opened_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"plan_sha256":sha256(plan_path)})
    df=load_records("test"); arrays={}; performance=[]
    for name,entry in plan["registry"].items():
        started=time.perf_counter(); p=calibrated_scores(name,entry,df,"test"); elapsed=time.perf_counter()-started
        arrays[name]=p
        pd.DataFrame({"doc_id":df.doc_id.to_numpy(),"score_spam":p}).to_parquet(f"artifacts/predictions/test_{name}.parquet",index=False)
        performance.append({"model":name,"n":len(df),"load_transform_score_calibrate_s":elapsed,"mean_ms_per_mail":elapsed/len(df)*1000,"artifact_bytes":Path(entry["path"]).stat().st_size})
        print("Test predictions saved",name,len(p),flush=True)
    for name in plan["comparisons"]:
        if name not in arrays: arrays[name]=sum(w*arrays[k] for k,w in plan["specifications"][name]["weights"].items())
    prior=float(load_records("train").label.mean()); dummy=metrics(df.label,np.full(len(df),prior),.5); save_json("artifacts/dummy_test.json",dummy)
    rows=[]; group_sizes=df.groupby("template_group_id").doc_id.transform("size"); weights=1/group_sizes.to_numpy()
    for name in plan["comparisons"]:
        p=arrays[name]
        pd.DataFrame({"doc_id":df.doc_id.to_numpy(),"score_spam":p}).to_parquet(f"artifacts/predictions/test_{name}.parquet",index=False)
        for point,threshold in plan["specifications"][name]["thresholds"].items():
            rows.append({"model":name,"point":point,"weighting":"per_mail",**metrics(df.label,p,threshold)})
            rows.append({"model":name,"point":point,"weighting":"per_group",**metrics(df.label,p,threshold,weights)})
    pd.DataFrame(rows).to_csv("artifacts/test_results.csv",index=False)
    pd.DataFrame(performance).to_csv("artifacts/performance_test.csv",index=False)
    save_json("artifacts/bootstrap_test.json",group_bootstrap(df,arrays,plan["specifications"],plan["main"],plan["bootstrap_repeats"],plan["seed"]))
    main=plan["main"]; threshold=plan["specifications"][main]["thresholds"]["research"]
    error_tables(df,arrays[main],threshold,"test")
    robust=[]
    for rule in plan["robustness_rules"]:
        changed,applied=perturb(df,rule); new=np.zeros(len(df))
        for branch,w in plan["specifications"][main]["weights"].items():
            if w: new+=w*calibrated_scores(branch,plan["registry"][branch],changed,"test_"+rule)
        robust.append({"rule":rule,**paired_metrics(df.label,arrays[main],new,threshold,applied)})
        pd.DataFrame({"doc_id":df.doc_id.to_numpy(),"score_spam":new,"applied":applied}).to_parquet(f"artifacts/predictions/test_robust_{rule}.parquet",index=False)
        print("Paired Test robustness",rule,robust[-1],flush=True)
    pd.DataFrame(robust).to_csv("artifacts/robustness_test.csv",index=False)
    save_json("artifacts/test_complete.json",dict(rows=len(df),all_predictions_aligned=True,no_abstention=True,zip_unchanged=sha256(read_config("configs/data.yaml")["zip_path"])==plan["zip_sha256"]))
