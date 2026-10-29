"""Padding-safe pooling and deterministic per-mail token budgets."""

def last_token_pool(hidden,mask):
    import torch
    # Explicit position indices support both left and right padding.
    positions=torch.arange(mask.shape[1],device=mask.device)[None,:].expand_as(mask)
    indices=positions.masked_fill(mask==0,-1).max(dim=1).values
    if (indices<0).any(): raise ValueError("All-padding input")
    return hidden[torch.arange(len(hidden),device=hidden.device),indices]


def truncate_tokens(subject,body,tokenizer,max_length=512,subject_max=64,mode="head_tail"):
    s=tokenizer.encode(subject,add_special_tokens=False)[:subject_max]
    b=tokenizer.encode(body,add_special_tokens=False)
    budget=max_length-len(s)-tokenizer.num_special_tokens_to_add(pair=True)
    if budget<0: raise ValueError("Subject and special tokens exceed limit")
    if len(b)>budget:
        if mode=="head_tail":
            head=(budget+1)//2; b=b[:head]+(b[-(budget-head):] if budget>head else [])
        elif mode=="head": b=b[:budget]
        else: raise ValueError(mode)
    ids=tokenizer.build_inputs_with_special_tokens(s,b)
    types=tokenizer.create_token_type_ids_from_sequences(s,b)
    assert len(ids)<=max_length
    return {"input_ids":ids,"token_type_ids":types,"attention_mask":[1]*len(ids)}

