import pytest
import numpy as np

def test_last_nonpadding_pool():
    torch=pytest.importorskip("torch")
    from spam_analysis.semantic_models import last_token_pool
    hidden=torch.arange(24).reshape(2,4,3)
    assert torch.equal(last_token_pool(hidden,torch.tensor([[0,0,1,1],[1,1,0,0]])),torch.stack([hidden[0,3],hidden[1,1]]))

class ToyTokenizer:
    def encode(self,s,add_special_tokens=False): return list(range(len(s)))
    def num_special_tokens_to_add(self,pair=False): return 3 if pair else 2
    def build_inputs_with_special_tokens(self,a,b): return [100]+a+[101]+b+[101]
    def create_token_type_ids_from_sequences(self,a,b): return [0]*(len(a)+2)+[1]*(len(b)+1)

def test_length_budget_no_overlap():
    from spam_analysis.semantic_models import truncate_tokens
    tok=ToyTokenizer(); r=truncate_tokens("abc","de",tok,12,3)
    assert r["input_ids"]==[100,0,1,2,101,0,1,101]
    r=truncate_tokens("abc","x"*100,tok,12,3)
    assert len(r["input_ids"])==12 and r["input_ids"][-4:-1]==[97,98,99]
