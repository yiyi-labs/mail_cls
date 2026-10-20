"""Official model acquisition and immutable revision/file identities."""
import os
from pathlib import Path,PurePosixPath
from .io import sha256


def verify_identity(identity):
    folder=Path(identity['folder'])
    if not folder.exists():
        raise FileNotFoundError('Pinned model assets are missing. Restore the saved identity with spam_analysis.model_assets first.')
    for name,digest in identity['files'].items():
        relative=PurePosixPath(name.replace('\\','/'))
        if relative.is_absolute() or '..' in relative.parts or ':' in name or not relative.parts:
            raise ValueError('Unsafe model manifest path: '+name)
        path=folder.joinpath(*relative.parts)
        # Hugging Face snapshots intentionally link to sibling cache blobs.
        # Validate the manifest name and exact content hash; this read-only
        # check must not mistake those normal links for changed model bytes.
        if not path.is_file() or sha256(path)!=digest:
            raise ValueError('Model file identity mismatch: '+name)
    return identity


def restore_identity(path):
    """Restore exactly the recorded revision, never silently follow latest."""
    import json
    from .io import save_json
    identity=json.loads(Path(path).read_text(encoding='utf-8')); configure_local()
    if identity.get('source')=='ModelScope':
        import requests
        folder=Path('.cache/modelscope')/identity['distribution_repo']/identity['revision'][:16]
        base='https://modelscope.cn/api/v1/models/'+identity['distribution_repo']+'/repo'
        for name,digest in identity['files'].items():
            target=folder/name
            if not target.resolve().is_relative_to(folder.resolve()): raise ValueError('Unsafe asset path')
            if target.exists() and sha256(target)==digest: continue
            target.parent.mkdir(parents=True,exist_ok=True)
            with requests.get(base,params={'Revision':identity['file_revisions'][name],'FilePath':name},stream=True,timeout=(30,120)) as response:
                response.raise_for_status()
                with target.open('wb') as output:
                    for chunk in response.iter_content(1024*1024): output.write(chunk)
            if sha256(target)!=digest: raise ValueError('Restored weight hash mismatch: '+name)
    else:
        from huggingface_hub import snapshot_download
        folder=Path(snapshot_download(identity['repo'],revision=identity['revision'],allow_patterns=list(identity['files']),max_workers=2))
    identity['folder']=str(folder.resolve()); verify_identity(identity); save_json(path,identity)
    return identity

def download_modelscope(repo, upstream_repo=None, expected_weight_hash=None):
    """Verified alternate distribution; immutable per-file revisions + SHA-256.

    Qwen's own GitHub explicitly links its ModelScope distribution. For a mirror
    of another author, require an independently checked upstream weight hash.
    """
    import requests
    from .io import save_json
    if not repo.startswith("Qwen/") and not expected_weight_hash:
        raise ValueError("Mirror requires independently verified upstream weight SHA-256")
    base="https://modelscope.cn/api/v1/models/"+repo
    info=requests.get(base,timeout=30).json()["Data"]
    if info.get("License")!="apache-2.0": raise ValueError("Unverified alternate-source license")
    data=requests.get(base+"/repo/files",params={"Revision":"master","Recursive":"true"},timeout=30).json()["Data"]
    files=[f for f in data["Files"] if f["Type"]=="blob" and (f["Path"].endswith((".json",".txt",".safetensors",".bin",".model")) or f["Path"] in {"LICENSE","README.md"}) and not f["Path"].startswith(("onnx/","openvino/","tf_","flax_"))]
    if any(f["Path"].endswith(".safetensors") for f in files): files=[f for f in files if not f["Path"].endswith(".bin")]
    if expected_weight_hash:
        weights=[f for f in files if f["Path"].endswith((".bin",".safetensors"))]
        if len(weights)!=1 or weights[0]["Sha256"]!=expected_weight_hash: raise ValueError("Mirror weights do not match the official upstream hash")
    import hashlib,json
    revision=hashlib.sha256(json.dumps([(f['Path'],f['Revision'],f['Sha256']) for f in files],sort_keys=True).encode()).hexdigest()
    folder=Path(".cache/modelscope")/repo/ revision[:16]; folder.mkdir(parents=True,exist_ok=True)
    for f in files:
        target=folder/f["Path"]
        if not target.resolve().is_relative_to(folder.resolve()): raise ValueError("Unsafe remote path")
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and sha256(target)==f["Sha256"]: continue
        print("Downloading verified model file",repo,f["Path"],f["Size"],flush=True)
        with requests.get(base+"/repo",params={"Revision":f["Revision"],"FilePath":f["Path"]},stream=True,timeout=(30,120)) as response:
            response.raise_for_status()
            with target.open("wb") as stream:
                for chunk in response.iter_content(1024*1024): stream.write(chunk)
        if sha256(target)!=f["Sha256"]: raise ValueError("Downloaded file failed SHA-256 verification: "+f["Path"])
    identity={"repo":upstream_repo or repo,"distribution_repo":repo,"source":"ModelScope", "revision":revision,
              "revision_definition":"SHA256 of immutable per-file revision/hash manifest", "license":info["License"],"folder":str(folder.resolve()),
              "upstream_weight_verified":bool(expected_weight_hash),"files":{f["Path"]:f["Sha256"] for f in files},
              "file_revisions":{f["Path"]:f["Revision"] for f in files}}
    save_json(folder/"download_identity.json",identity)
    return identity

def configure_local():
    os.environ["HF_HOME"]=str(Path(".cache/huggingface").resolve())
    os.environ["HF_HUB_DISABLE_TELEMETRY"]="1"
    os.environ["HF_HUB_DISABLE_XET"]="1"
    os.environ["TOKENIZERS_PARALLELISM"]="false"


def download_official(repo):
    configure_local()
    from huggingface_hub import HfApi,snapshot_download
    info=HfApi().model_info(repo)
    license_name=info.card_data.get("license") if info.card_data else None
    if license_name!="apache-2.0": raise RuntimeError(f"Unexpected license: {license_name}; inspect before downloading")
    revision=info.sha
    folder=snapshot_download(repo,revision=revision,allow_patterns=["*.json","*.txt","*.safetensors","*.bin","*.model","LICENSE","README.md"],ignore_patterns=["tf*","flax*","rust*","onnx/*","openvino/*"],max_workers=2)
    identity={"repo":repo,"revision":revision,"license":license_name,"folder":str(Path(folder).resolve()),
              "files":{p.name:sha256(p) for p in Path(folder).glob('*') if p.is_file() and p.suffix in {'.json','.txt','.safetensors','.bin','.model'}}}
    return identity


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Restore pinned local model files without uploading data')
    parser.add_argument('command',choices=['restore','verify']); parser.add_argument('--identity',required=True)
    args=parser.parse_args()
    if args.command=='restore': restore_identity(args.identity)
    else:
        import json
        verify_identity(json.loads(Path(args.identity).read_text(encoding='utf-8')))
    print('Pinned model files verified')

