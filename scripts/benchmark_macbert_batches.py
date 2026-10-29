"""Train-only resource probe; weights are discarded and no Tune scores read."""
import gc,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from spam_analysis.io import load_records,read_config,save_json
from spam_analysis.macbert import batches
from spam_analysis.model_assets import configure_local,verify_identity

configure_local();torch.set_num_threads(8)
identity=json.loads(Path('artifacts/macbert_identity.json').read_text(encoding='utf-8'));verify_identity(identity)
config=read_config('configs/semantic.yaml')
# Long Train inputs exercise the maximum declared tensor size, not easy short mail.
train=load_records('train').sort_values(['body_chars','doc_id'],ascending=[False,True]).head(16)
tokenizer=AutoTokenizer.from_pretrained(identity['folder'],local_files_only=True)
encoded=batches(train,tokenizer,config,'head_tail');labels=train.label.to_numpy()
rows=[];reference=None
for micro,accum,fused in [(8,2,None),(8,2,False),(8,2,True),(16,1,False),(16,1,True)]:
    torch.manual_seed(42)
    model=AutoModelForSequenceClassification.from_pretrained(identity['folder'],num_labels=2,
        local_files_only=True,attn_implementation='sdpa').to('cuda').train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01,fused=fused)
    pieces=[]
    for start in range(0,16,micro):
        x=tokenizer.pad(encoded[start:start+micro],padding=True,return_tensors='pt').to('cuda')
        y=torch.tensor(labels[start:start+micro],device='cuda',dtype=torch.long)
        pieces.append((x,y))
    elapsed=[];losses=[];torch.cuda.reset_peak_memory_stats()
    try:
        for step in range(8):
            torch.cuda.synchronize();started=time.perf_counter();optimizer.zero_grad(set_to_none=True)
            for x,y in pieces:
                with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(**x,labels=y).loss/accum
                loss.backward()
                losses.append(float(loss.detach()))
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step();torch.cuda.synchronize()
            if step>=3:elapsed.append(time.perf_counter()-started)
        row=dict(micro_batch=micro,accumulation=accum,effective_batch=16,fused=fused,
            measured_updates=len(elapsed),median_update_s=float(np.median(elapsed)),mean_update_s=float(np.mean(elapsed)),
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,passed=bool(np.isfinite(losses).all()),
            first_micro_loss=losses[0],last_micro_loss=losses[-1])
        if micro==8:
            state={name:value.detach().cpu().clone() for name,value in model.named_parameters()}
            if reference is None:reference=state
            row['max_weight_delta_vs_default_after_8_updates']=max(float((state[name]-value).abs().max()) for name,value in reference.items())
    except torch.cuda.OutOfMemoryError:
        row=dict(micro_batch=micro,accumulation=accum,effective_batch=16,fused=fused,passed=False,error='CUDA OOM')
    rows.append(row);print(row,flush=True)
    del pieces,x,y,model,optimizer,loss;gc.collect();torch.cuda.empty_cache()
save_json('artifacts/macbert_batch_benchmark.json',dict(cases=rows,rows=16,max_tokens=max(len(e['input_ids']) for e in encoded),
    scope='Fixed longest Train mails, 3 warm-up and 5 measured optimizer updates; no Tune/Calibrate/Policy/Test data, all temporary weights discarded'))
