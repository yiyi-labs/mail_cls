"""Supervised MacBERT branch, mail-level examples and bounded token budgets."""
import gc
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from .io import save_json,sha256
from .metrics import metrics

def batches(records,tokenizer,config,mode):
    from .semantic_models import truncate_tokens
    return [truncate_tokens(r.subject_clean,r.body_clean,tokenizer,config["macbert_max_length"],config["macbert_subject_max"],mode) for r in records.itertuples()]

def predict(model,tokenizer,encoded,batch_size=16):
    import torch
    model.eval(); result=[]
    with torch.inference_mode():
        for start in range(0,len(encoded),batch_size):
            batch=tokenizer.pad(encoded[start:start+batch_size],padding=True,return_tensors="pt").to(model.device)
            with torch.autocast(device_type=model.device.type,dtype=torch.bfloat16,enabled=model.device.type=="cuda"):
                logits=model(**batch).logits
            result.extend((logits[:,1]-logits[:,0]).float().cpu().numpy())
    return np.asarray(result)

def train_macbert(train,tune,config):
    from .semantic_models import configure_local,download_official
    configure_local()
    import torch
    from transformers import AutoTokenizer,AutoModelForSequenceClassification,get_linear_schedule_with_warmup
    status=json.loads(Path("artifacts/semantic_status.json").read_text(encoding="utf-8"))
    if not torch.cuda.is_available():
        status["E2"]="not run: CUDA unavailable; E1 CPU resource measurement retained"
        save_json("artifacts/semantic_status.json",status); return
    torch.manual_seed(config["seed"]); np.random.seed(config["seed"]); torch.set_num_threads(8)
    identity_path=Path("artifacts/macbert_identity.json")
    identity=json.loads(identity_path.read_text(encoding="utf-8")) if identity_path.exists() else download_official(config["macbert_repo"])
    from .model_assets import verify_identity
    verify_identity(identity)
    save_json(identity_path,identity)
    tokenizer=AutoTokenizer.from_pretrained(identity["folder"],local_files_only=True,trust_remote_code=False)
    mode=config["macbert_truncation"]
    started=time.perf_counter(); encoded=batches(train,tokenizer,config,mode); valid=batches(tune,tokenizer,config,mode)
    tokenization_s=time.perf_counter()-started; labels=train.label.to_numpy(); batch_size=config["macbert_batch_size"]; accumulation=config["macbert_accumulation"]
    # Audit original lengths separately; bounded tensors alone cannot reveal
    # how much input was omitted. Statistics use Train only.
    subject_lengths=np.array([len(tokenizer.encode(s,add_special_tokens=False)) for s in train.subject_clean])
    body_lengths=np.array([len(tokenizer.encode(b,add_special_tokens=False)) for b in train.body_clean])
    body_budget=config['macbert_max_length']-np.minimum(subject_lengths,config['macbert_subject_max'])-tokenizer.num_special_tokens_to_add(pair=True)
    length_audit=dict(rows=len(train),subject_truncated_fraction=float(np.mean(subject_lengths>config['macbert_subject_max'])),
        body_truncated_fraction=float(np.mean(body_lengths>body_budget)),any_truncated_fraction=float(np.mean((subject_lengths>config['macbert_subject_max'])|(body_lengths>body_budget))),
        original_total_token_quantiles={str(q):float(np.quantile(subject_lengths+body_lengths+tokenizer.num_special_tokens_to_add(pair=True),q)) for q in [.5,.9,.95,.99]},
        max_encoded_tokens=max(len(row['input_ids']) for row in encoded),scope='Train; subject/body truncation evaluated against actual per-mail token budget')
    save_json('artifacts/macbert_length_audit.json',length_audit)
    make_model=lambda: AutoModelForSequenceClassification.from_pretrained(identity["folder"],num_labels=2,
        id2label={0:"ham",1:"spam"},label2id={"ham":0,"spam":1},local_files_only=True,trust_remote_code=False,attn_implementation="sdpa").to("cuda")
    model=make_model(); tiny=np.r_[np.flatnonzero(labels==0)[:4],np.flatnonzero(labels==1)[:4]]
    tiny_batch=tokenizer.pad([encoded[i] for i in tiny],padding=True,return_tensors="pt").to("cuda")
    tiny_y=torch.tensor(labels[tiny],dtype=torch.long,device="cuda")
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
    losses=[]; torch.cuda.reset_peak_memory_stats(); started=time.perf_counter()
    for step in range(40):
        model.train(); optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16): output=model(**tiny_batch,labels=tiny_y)
        output.loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
        losses.append(float(output.loss.detach()))
    torch.cuda.synchronize(); smoke_s=time.perf_counter()-started
    scores=predict(model,tokenizer,[encoded[i] for i in tiny],8); smoke_accuracy=float(accuracy_score(labels[tiny],scores>=0))
    profile={"smoke_rows":len(tiny),"smoke_steps":40,"smoke_first_loss":losses[0],"smoke_final_loss":losses[-1],
             "smoke_training_accuracy":smoke_accuracy,"smoke_s":smoke_s,"tokenization_s":tokenization_s,
             "estimated_epoch_s_from_smoke":smoke_s/40*np.ceil(len(train)/batch_size),"peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20,
             "note":"Tiny training-set overfit smoke test is not a generalization result; estimate excludes validation and I/O"}
    save_json("artifacts/macbert_profile.json",profile); print(profile,flush=True)
    if smoke_accuracy<.99 or losses[-1]>=losses[0]: raise RuntimeError("MacBERT 8-mail overfit test failed")
    status['E2']='training';save_json('artifacts/semantic_status.json',status)
    del model,optimizer,tiny_batch,tiny_y; gc.collect(); torch.cuda.empty_cache()
    curves=[]; best=None; registry=json.loads(Path("artifacts/semantic_registry.json").read_text(encoding="utf-8"))
    for lr in config["macbert_learning_rates"]:
        torch.manual_seed(config["seed"]); model=make_model(); optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01)
        steps=int(np.ceil(np.ceil(len(train)/batch_size)/accumulation))*config["macbert_epochs"]
        scheduler=get_linear_schedule_with_warmup(optimizer,num_warmup_steps=int(.1*steps),num_training_steps=steps)
        rng=np.random.default_rng(config["seed"]); trained_steps=0
        for epoch in range(1,config["macbert_epochs"]+1):
            model.train(); order=rng.permutation(len(train)); optimizer.zero_grad(set_to_none=True)
            started=time.perf_counter(); total_loss=0.; seen=0; torch.cuda.reset_peak_memory_stats()
            for batch_index,start in enumerate(range(0,len(order),batch_size)):
                ids=order[start:start+batch_size]
                batch=tokenizer.pad([encoded[i] for i in ids],padding=True,return_tensors="pt").to("cuda")
                target=torch.tensor(labels[ids],dtype=torch.long,device="cuda")
                with torch.autocast("cuda",dtype=torch.bfloat16): output=model(**batch,labels=target)
                loss=output.loss/accumulation; loss.backward(); total_loss+=float(output.loss.detach())*len(ids); seen+=len(ids)
                if (batch_index+1)%accumulation==0 or start+batch_size>=len(order):
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); trained_steps+=1
                if batch_index%500==0: print(f"MacBERT lr={lr} epoch={epoch} mails={seen}/{len(train)} loss={total_loss/seen:.5f}",flush=True)
            torch.cuda.synchronize(); training_s=time.perf_counter()-started
            score=predict(model,tokenizer,valid,batch_size*2); m=metrics(tune.label,score,0,probability=False)
            curves.append({"lr":lr,"epoch":epoch,"training_loss":total_loss/seen,"training_s":training_s,"optimizer_steps":trained_steps,
                           "peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20,"train_rows":len(train),"train_groups":int(train.template_group_id.nunique()),**m})
            pd.DataFrame(curves).to_csv("artifacts/macbert_learning_curves.csv",index=False)
            pd.DataFrame({"doc_id":tune.doc_id.to_numpy(),"score_spam":score}).to_parquet(f"artifacts/predictions/tune_E2_{lr}_{epoch}.parquet",index=False)
            if best is None or (m["macro_f1"],-m["ham_fpr"])>(best["macro_f1"],-best["ham_fpr"]):
                best=m; folder=Path("artifacts/models/E2_checkpoint"); model.save_pretrained(folder,safe_serialization=True); tokenizer.save_pretrained(folder)
                checkpoint={"folder":str(folder.resolve()),"files":{p.name:sha256(p) for p in folder.glob("*") if p.is_file()},"lr":lr,"epoch":epoch,"mode":mode}
                save_json("artifacts/models/E2_identity.json",checkpoint)
                registry["E2"]={"path":"artifacts/models/E2_identity.json","identity":identity,"checkpoint":checkpoint,"config":config,"metrics":m,"fit_split":"train"}
                save_json("artifacts/semantic_registry.json",registry)
            print("MacBERT Tune",lr,epoch,m,flush=True)
        del model,optimizer,scheduler; gc.collect(); torch.cuda.empty_cache()
    # Evaluate a second deterministic inference truncation on Tune with the same
    # trained weights. This isolates inference truncation, not a retraining claim.
    entry=registry["E2"]
    restored=macbert_scores(entry,tune)
    saved=pd.read_parquet(f"artifacts/predictions/tune_E2_{entry['checkpoint']['lr']}_{entry['checkpoint']['epoch']}.parquet")
    assert saved.doc_id.tolist()==tune.doc_id.tolist()
    np.testing.assert_allclose(restored,saved.score_spam.to_numpy(),rtol=1e-4,atol=.001)
    save_json('artifacts/macbert_reload_validation.json',dict(rows=len(tune),passed=True,
        max_abs_delta=float(np.max(np.abs(restored-saved.score_spam.to_numpy()))),
        prediction_flips=int(np.sum((restored>=0)!=(saved.score_spam.to_numpy()>=0))),rtol=1e-4,atol=.001))
    alternative={**entry,"checkpoint":{**entry["checkpoint"],"mode":"head"}}
    head=macbert_scores(alternative,tune); head_m=metrics(tune.label,head,0,probability=False)
    save_json("artifacts/macbert_truncation_tune.json",{"head_tail":entry["metrics"],"head":head_m,"scope":"Same trained weights; inference truncation only"})
    if (head_m["macro_f1"],-head_m["ham_fpr"])>(best["macro_f1"],-best["ham_fpr"]):
        entry["checkpoint"]["mode"]="head"; entry["metrics"]=head_m
        save_json(entry["path"],entry["checkpoint"]); save_json("artifacts/semantic_registry.json",registry)
    status["E2"]="complete"; save_json("artifacts/semantic_status.json",status)

def macbert_scores(entry,df):
    from .semantic_models import configure_local
    configure_local(); import torch
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    folder=entry["checkpoint"]["folder"]
    for name,h in entry["checkpoint"]["files"].items():
        if sha256(Path(folder)/name)!=h: raise RuntimeError("MacBERT checkpoint identity changed")
    tokenizer=AutoTokenizer.from_pretrained(folder,local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(folder,local_files_only=True,attn_implementation="sdpa").to("cuda" if torch.cuda.is_available() else "cpu")
    encoded=batches(df,tokenizer,entry["config"],entry["checkpoint"]["mode"])
    result=predict(model,tokenizer,encoded,entry["config"]["macbert_batch_size"]*2)
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return result
