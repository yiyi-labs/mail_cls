"""Read the original archive without extracting or changing it."""
import hashlib
import json
import posixpath
from pathlib import Path
from zipfile import ZipFile
import yaml

LABELS = {"ham": 0, "spam": 1}

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def read_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_json(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

def read_index(z):
    candidates = [n for n in z.namelist() if n.endswith("/full/index") and not n.startswith("__MACOSX/")]
    if len(candidates) != 1:
        raise ValueError("Exactly one non-metadata full/index is required; delay is never training data")
    idx = candidates[0]; root = idx[:-len("full/index")]
    rows = []; seen = set(); names = set(z.namelist())
    for order, line in enumerate(z.read(idx).decode("ascii").splitlines()):
        fields = line.split()
        if len(fields) != 2 or fields[0] not in LABELS:
            raise ValueError(f"Invalid index line {order}")
        label, relative = fields
        path = posixpath.normpath(posixpath.join(posixpath.dirname(idx), relative))
        if not path.startswith(root + "data/") or path in seen or path not in names:
            raise ValueError(f"Unsafe, duplicate or missing index path: {path}")
        seen.add(path)
        rows.append({"doc_id": path[len(root):], "relative_path": path, "index_order": order, "label": LABELS[label]})
    return rows

def load_records(split=None):
    import pandas as pd
    manifest = Path("artifacts/splits/split_manifest.parquet")
    if manifest.exists():
        m = pd.read_parquet(manifest)
        if split is not None:
            allowed = [split] if isinstance(split, str) else split
            m = m[m.split.isin(allowed)]
        df = pd.read_parquet("data/processed/records.parquet", filters=[("doc_id", "in", m.doc_id.tolist())])
        df = df.merge(m.drop(columns="label"), on="doc_id", validate="one_to_one")
    else:
        if split is not None:
            raise RuntimeError("Split manifest required before model development")
        df = pd.read_parquet("data/processed/records.parquet")
    return df.reset_index(drop=True)
