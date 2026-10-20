import argparse
from pathlib import Path
from .io import read_config,load_records

def main():
    p=argparse.ArgumentParser(description="Immutable-data Chinese spam course experiment")
    p.add_argument("command",choices=["audit","prepare","split","eda","train-sparse","train-semantic","calibrate","select-policy","freeze-evaluation","evaluate","deliver"])
    p.add_argument("--config",default="configs/experiment.yaml")
    p.add_argument("--split",default="test")
    p.add_argument("--frozen-plan",default="artifacts/evaluation_plan.json")
    args=p.parse_args(); config=read_config(args.config)
    if args.command=='evaluate' and args.split!='test':
        p.error('Formal frozen evaluation only supports --split test; development analysis is separate')
    Path("artifacts").mkdir(exist_ok=True)
    if args.command in {"audit","prepare"}:
        from .audit import audit,prepare
        if "zip_path" not in config: config=read_config(config["data_config"])
        (audit if args.command=="audit" else prepare)(config)
    elif args.command=="split":
        from .grouping import build_groups
        from .splitting import make_splits
        import pandas as pd
        records=pd.read_parquet("data/processed/records.parquet")
        group_path=Path("artifacts/splits/group_manifest.parquet")
        groups=pd.read_parquet(group_path) if group_path.exists() else build_groups(records,config)
        make_splits(records,groups,config)
    else:
        modules={"eda":("plots","eda"),"train-sparse":("sparse_models","train_sparse"),"train-semantic":("semantic_models","train_semantic"),
                 "calibrate":("calibration","calibrate"),"select-policy":("policy","select_policy"),"freeze-evaluation":("evaluation","freeze_evaluation"),
                 "evaluate":("evaluation","evaluate_frozen"),"deliver":("deliver","deliver")}
        import importlib
        mod,func=modules[args.command]
        getattr(importlib.import_module("spam_analysis."+mod),func)(config if args.command!="evaluate" else args.frozen_plan)

if __name__=="__main__": main()
