"""Trace a small GPU resource check without fitting or reading held-out data."""
import json,time,sys
from pathlib import Path
import torch
from spam_analysis.io import load_records,read_config,save_json
print('Torch',torch.__version__,'CUDA available',torch.cuda.is_available(),flush=True)
if not torch.cuda.is_available() and '--allow-cpu' not in sys.argv: raise RuntimeError('GPU unavailable in this execution context')
if torch.cuda.is_available(): print('Device',torch.cuda.get_device_name(0),flush=True)
from spam_analysis.embeddings import QwenEncoder
identity=json.loads(Path('artifacts/qwen_identity.json').read_text(encoding='utf-8'))
started=time.perf_counter(); encoder=QwenEncoder(identity,read_config('configs/semantic.yaml'))
print('Loaded Qwen seconds',time.perf_counter()-started,'device',encoder.device,flush=True)
records=load_records('train').iloc[:3]
started=time.perf_counter(); result=encoder.encode(records.semantic_text.tolist(),batch_size=1)
print('Three records seconds',time.perf_counter()-started,'shape',result.shape,flush=True)
save_json('artifacts/semantic_device_probe.json',dict(device=encoder.device,n=3,elapsed_s=time.perf_counter()-started,load_s=encoder.load_s,max_length=1024,batch_size=1,note='Three fixed Train rows; too small to claim corpus throughput'))
