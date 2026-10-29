"""Label-independent content components; LSH proposes, exact Jaccard decides."""
import hashlib
import re
import time
import zlib
from pathlib import Path
import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH
from .io import save_json

def compact(text):
    return re.sub(r"\s+", "", text)

def grouping_view(text):
    text = re.sub(r"https?://[^\s<>]+|www\.[^\s<>]+", " URL ", text, flags=re.I)
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", " EMAIL ", text)
    text = re.sub(r"\d{6,}", " NUMBER ", text)
    return compact(text)

def shingles(text, k=5):
    return {text[i:i+k] for i in range(max(0,len(text)-k+1))}

def jaccard(a,b):
    return len(a & b)/len(a | b) if a or b else 1.0

class UnionFind:
    def __init__(self,n):
        self.parent=list(range(n))
    def find(self,x):
        while x!=self.parent[x]:
            self.parent[x]=self.parent[self.parent[x]]; x=self.parent[x]
        return x
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b:
            self.parent[max(a,b)]=min(a,b)

def build_groups(records, config):
    started=time.perf_counter(); n=len(records); uf=UnionFind(n); exact={}; body_keys={}; views={}; exact_ids=[]
    for i,r in enumerate(records.itertuples()):
        key=compact(r.subject_clean)+"\n"+compact(r.body_clean)
        # Truly empty messages remain separate, never a giant empty template.
        eid=hashlib.sha256((key if key.strip() else "EMPTY:"+r.doc_id).encode()).hexdigest()
        exact_ids.append(eid)
        if key.strip():
            if key in exact: uf.union(i,exact[key])
            else: exact[key]=i
        body=compact(r.body_clean)
        if len(body)>=config["body_group_min_chars"]:
            if body in body_keys: uf.union(i,body_keys[body])
            else: body_keys[body]=i
            view=grouping_view(r.body_clean)
            if view in views: uf.union(i,views[view])
            else: views[view]=i
    texts=list(views); representatives=[views[t] for t in texts]
    sets=[shingles(t,config["shingle_size"]) for t in texts]
    edges=[]; tested=set(); candidates=0; secondary_new=0
    # Independent secondary candidate pass catches additional LSH misses before
    # any split/model exists. It still cannot prove exhaustive semantic recall.
    for pass_id, seed in enumerate([config["seed"],config["seed"]+1]):
        lsh=MinHashLSH(threshold=config["lsh_candidate_threshold"],num_perm=config["minhash_permutations"])
        for i,grams in enumerate(sets):
            mh=MinHash(num_perm=config["minhash_permutations"],seed=seed,hashfunc=zlib.crc32)
            gram_list=sorted(grams)
            for offset in range(0,len(gram_list),1024):
                mh.update_batch([g.encode("utf-8") for g in gram_list[offset:offset+1024]])
            for j in sorted(lsh.query(mh)):
                pair=(j,i)
                if pair in tested: continue
                tested.add(pair); candidates+=1
                ratio=min(len(texts[i]),len(texts[j]))/max(len(texts[i]),len(texts[j]),1)
                if ratio<config["length_ratio"]: continue
                sim=jaccard(grams,sets[j])
                if sim>=config["jaccard_threshold"]:
                    uf.union(representatives[i],representatives[j])
                    edges.append({"left":records.iloc[representatives[j]].doc_id,"right":records.iloc[representatives[i]].doc_id,"jaccard":sim,"length_ratio":ratio,"candidate_pass":pass_id})
                    secondary_new+=int(pass_id==1)
            lsh.insert(i,mh)
            if (i+1)%5000==0: print(f"LSH pass {pass_id+1}: {i+1}/{len(texts)}",flush=True)
    group_ids=["g"+str(uf.find(i)).zfill(6) for i in range(n)]
    result=pd.DataFrame({"doc_id":records.doc_id,"exact_content_id":exact_ids,"template_group_id":group_ids,
                         "grouping_text":[grouping_view(x) for x in records.body_clean]})
    Path("artifacts/splits").mkdir(parents=True,exist_ok=True)
    result.to_parquet("artifacts/splits/group_manifest.parquet",index=False)
    pd.DataFrame(edges,columns=["left","right","jaccard","length_ratio","candidate_pass"]).to_parquet("artifacts/splits/near_edges.parquet",index=False)
    joined=result[["doc_id","template_group_id"]].merge(records[["doc_id","label"]],on="doc_id",validate="one_to_one")
    sizes=joined.groupby("template_group_id").size(); mixed=joined.groupby("template_group_id").label.nunique().gt(1)
    # Audit component chaining: deterministic pair sample from large components.
    rng=np.random.default_rng(config["seed"]); chain=[]
    grouped={}
    for i,rep in enumerate(representatives): grouped.setdefault(group_ids[rep],[]).append(i)
    for gid,count in sizes.nlargest(20).items():
        members=grouped.get(gid,[]); sims=[]
        if len(members)>1:
            for _ in range(min(100,len(members)*(len(members)-1)//2)):
                a,b=rng.choice(members,2,replace=False); sims.append(jaccard(sets[a],sets[b]))
        chain.append({"group":gid,"mail_count":int(count),"distinct_views":len(members),"sample_min_pair_jaccard":float(min(sims)) if sims else None})
    summary=dict(rows=n,groups=int(sizes.size),exact_content_groups=len(set(exact_ids)),largest=int(sizes.max()),
                 mixed_label_groups=int(mixed.sum()),rows_in_mixed_groups=int(sizes[mixed].sum()),
                 long_unique_views=len(texts),candidate_pairs=candidates,verified_edges=len(edges),secondary_new_edges=secondary_new,
                 empty_text_rows=int(records.lexical_text.eq("").sum()),short_body_rows=int(records.body_clean.map(lambda t:len(compact(t))<80).sum()),
                 largest_component_audit=chain,elapsed_s=time.perf_counter()-started,
                 limitation="LSH can miss pairs; transitive components need not have all-pair Jaccard >= threshold; no label-based splitting")
    save_json("artifacts/group_summary.json",summary); print(summary,flush=True)
    return result
