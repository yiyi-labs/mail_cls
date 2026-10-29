"""Local Qwen encoding, padding checks and split-specific text-hash caches."""
import time
import hashlib
import json
from pathlib import Path
import numpy as np
from .io import save_json
from .parsing import PARSER_VERSION
from .model_assets import configure_local,verify_identity
from .semantic_tokens import last_token_pool

class QwenEncoder:
    def __init__(self,identity,config):
        configure_local(); verify_identity(identity)
        if config['instruction'] or config['embedding_truncation']!='head' or config['pooling']!='last_nonpadding_token_l2':
            raise ValueError('This registered encoder supports only empty instruction, head truncation and last-nonpadding L2 pooling')
        import torch
        from transformers import AutoTokenizer,AutoModel
        self.torch=torch; self.identity=identity; self.config=config
        self.device="cuda" if torch.cuda.is_available() else "cpu"
        torch.set_num_threads(8); torch.manual_seed(config["seed"])
        started=time.perf_counter()
        self.tokenizer=AutoTokenizer.from_pretrained(identity["folder"],padding_side="left",truncation_side="right",local_files_only=True,trust_remote_code=False)
        self.model=AutoModel.from_pretrained(identity["folder"],torch_dtype=torch.bfloat16 if self.device=="cuda" else torch.float32,
                                           attn_implementation="sdpa",local_files_only=True,trust_remote_code=False).to(self.device).eval()
        self.load_s=time.perf_counter()-started
    def encode(self,texts,batch_size=None):
        torch=self.torch; batch_size=batch_size or self.config["embedding_batch_size"]
        out=[]
        with torch.inference_mode():
            for start in range(0,len(texts),batch_size):
                batch=self.tokenizer(texts[start:start+batch_size],padding=True,truncation=True,max_length=self.config["max_length"],return_tensors="pt").to(self.device)
                states=self.model(**batch).last_hidden_state
                emb=torch.nn.functional.normalize(last_token_pool(states,batch["attention_mask"]).float(),p=2,dim=1)
                out.append(emb.cpu().numpy())
        return np.concatenate(out)
    def profile(self,records):
        texts=records.semantic_text.tolist()
        a=self.encode([texts[0]],1)[0]; b=self.encode([texts[0],texts[1],texts[-1]],3)[0]
        cosine=float(a@b); delta=float(np.max(np.abs(a-b)))
        if cosine<.999 or delta>.015: raise RuntimeError(f"Padding consistency failed: cosine={cosine}, max_delta={delta}")
        if self.device=="cuda": self.torch.cuda.reset_peak_memory_stats(); self.torch.cuda.synchronize()
        started=time.perf_counter(); embeddings=self.encode(texts)
        if self.device=="cuda": self.torch.cuda.synchronize()
        elapsed=time.perf_counter()-started
        return dict(device=self.device,rows=len(texts),elapsed_s=elapsed,rows_per_second=len(texts)/elapsed,load_s=self.load_s,
                    padding_cosine=cosine,padding_max_abs_delta=delta,embedding_dim=embeddings.shape[1],
                    peak_allocated_mib=self.torch.cuda.max_memory_allocated()/2**20 if self.device=="cuda" else None)
    def cache(self,df,split):
        folder=Path("artifacts/embeddings")/split; folder.mkdir(parents=True,exist_ok=True)
        texts=df.semantic_text.tolist(); hashes=[hashlib.sha256(t.encode()).hexdigest() for t in texts]
        meta={"doc_ids":df.doc_id.tolist(),"text_hashes":hashes,"parser_version":PARSER_VERSION,"revision":self.identity["revision"],
              "instruction":self.config["instruction"],"max_length":self.config["max_length"],"pooling":self.config["pooling"],"truncation":self.config["embedding_truncation"],
              "device":self.device,"dtype":str(self.model.dtype),"torch_version":self.torch.__version__}
        signature=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()
        path=folder/"qwen.npy"; metapath=folder/"qwen.json"
        if path.exists() and metapath.exists() and json.loads(metapath.read_text(encoding="utf-8"))["signature"]==signature:
            return np.load(path)
        # Within-split exact text caching saves work without changing mail weights.
        unique={}
        for i,h in enumerate(hashes): unique.setdefault(h,i)
        indices=list(unique.values()); indices.sort(key=lambda i:len(texts[i]))
        partial=folder/"qwen_partial.npy"; progress=folder/"qwen_progress.json"
        resumed=0; matrix=None; previous_s=0.0
        if partial.exists() and progress.exists():
            checkpoint=json.loads(progress.read_text(encoding="utf-8"))
            if checkpoint.get("signature")==signature:
                matrix=np.load(partial,mmap_mode="r+"); resumed=checkpoint["completed"]; previous_s=checkpoint.get('elapsed_s',0.0)
        if matrix is None:
            matrix=np.lib.format.open_memmap(partial,mode="w+",dtype=np.float32,shape=(len(indices),1024))
        started=time.perf_counter(); bs=self.config["embedding_batch_size"]
        for start in range(resumed,len(indices),bs):
            ids=indices[start:start+bs]; enc=self.encode([texts[i] for i in ids],bs)
            matrix[start:start+len(ids)]=enc
            if start%(bs*100)==0 or start+bs>=len(indices):
                matrix.flush(); save_json(progress,{"signature":signature,"completed":start+len(ids),'elapsed_s':previous_s+time.perf_counter()-started})
                print(f"Qwen {split}: {start+len(ids)}/{len(indices)} distinct texts",flush=True)
        position={hashes[i]:j for j,i in enumerate(indices)}
        X=np.stack([matrix[position[h]] for h in hashes]); np.save(path,X)
        save_json(metapath,{**meta,"signature":signature,"unique_texts":len(indices),"elapsed_s":previous_s+time.perf_counter()-started,"resumed_completed_rows":resumed})
        return X

