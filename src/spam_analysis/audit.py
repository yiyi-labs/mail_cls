import base64
import binascii
from collections import Counter, defaultdict
from email import policy
from email.parser import BytesParser
from pathlib import Path
from zipfile import ZipFile
import hashlib
import re
import time
import pandas as pd
from .io import read_index, save_json, sha256
from .parsing import raw_body, parse_email, naive_parse, LEAKY_LINE

def audit(config):
    start = time.perf_counter(); path = Path(config["zip_path"])
    identity = sha256(path)
    if identity != config["expected_sha256"]:
        raise ValueError("ZIP identity differs from the preregistered input")
    bodies = Counter(); raws = Counter(); counts = Counter(); defects = Counter(); headers = Counter(); types = Counter()
    body_labels = defaultdict(set); per_row = []
    with ZipFile(path) as z:
        rows = read_index(z); names = z.namelist()
        for row in rows:
            raw = z.read(row["relative_path"]); body = raw_body(raw)
            rh = hashlib.sha256(raw).hexdigest(); bh = hashlib.sha256(body).hexdigest()
            bodies[bh] += 1; raws[rh] += 1; body_labels[bh].add(row["label"])
            msg = BytesParser(policy=policy.default).parsebytes(raw)
            ct = msg.get_content_type(); cte = str(msg.get("Content-Transfer-Encoding", "")).strip().lower()
            types[ct] += 1; defects.update(type(d).__name__ for d in msg.defects)
            headers.update(set(k.lower() for k in msg.keys()))
            counts["raw_bytes"] += len(raw); counts["max_size"] = max(counts["max_size"], len(raw))
            counts["empty_body"] += not body.strip()
            counts["multipart_broken"] += ct.startswith("multipart/") and not msg.is_multipart()
            counts["light_html"] += bool(re.search(rb"<(?:html|body|div|p|br|table|a)\b", body, re.I))
            if cte == "base64":
                counts["base64"] += 1; counts["base64_nonascii"] += any(b > 127 for b in body)
                try:
                    base64.b64decode(re.sub(rb"\s+", b"", body), validate=True)
                except (ValueError, binascii.Error):
                    counts["base64_invalid"] += 1
            per_row.append({**row, "raw_sha256": rh, "raw_body_sha256": bh, "raw_size": len(raw)})
        root = rows[0]["relative_path"].split("data/")[0]
        delay = [line.split() for line in z.read(root + "delay/index").decode("ascii").splitlines()]
        dp = Counter(p for _, p in delay)
    summary = dict(zip_bytes=path.stat().st_size, sha256=identity, entries=len(names),
                   macosx=sum(n.startswith("__MACOSX/") for n in names), rows=len(rows), unique_paths=len(rows), missing=0,
                   labels={"spam":sum(r["label"] for r in rows),"ham":sum(1-r["label"] for r in rows)},
                   raw_unique=len(raws), body_unique=len(bodies), body_duplicate_groups=sum(v>1 for v in bodies.values()),
                   body_duplicate_rows=sum(v for v in bodies.values() if v>1), body_duplicate_extra=sum(v-1 for v in bodies.values()),
                   body_max=max(bodies.values()), body_mixed=sum(len(v)>1 for v in body_labels.values()), counts=dict(counts),
                   defects=dict(defects), types=dict(types), leaky_headers={k:headers[k] for k in ["x-proofpoint-spam-details","x-imss-scores","x-imss-result","x-eyou-spamvalue"]},
                   delay_rows=len(delay), delay_twice=all(v==2 for v in dp.values()),
                   delay_same={p.removeprefix("../") for p in dp}=={r["doc_id"] for r in rows}, elapsed_s=time.perf_counter()-start)
    assert len(rows) == config["expected_rows"]
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_row).to_parquet("data/processed/audit_rows.parquet", index=False)
    save_json("artifacts/audit_summary.json", summary)
    print(summary, flush=True)
    return summary

def prepare(config):
    if not Path("data/processed/audit_rows.parquet").exists():
        audit(config)
    start = time.perf_counter(); audit_rows = pd.read_parquet("data/processed/audit_rows.parquet")
    records = []; failures = []
    with ZipFile(config["zip_path"]) as z:
        for i, row in enumerate(audit_rows.to_dict("records")):
            raw = z.read(row["relative_path"])
            try:
                parsed = parse_email(raw)
                parsed["naive_text"] = naive_parse(raw)
            except Exception as exc:
                failures.append({"doc_id":row["doc_id"],"error":repr(exc)})
                # Preserve the row, with explicit quality failure; block P1 below.
                parsed = parse_email(b""); parsed.update(parser_status="failed", fallback_reason=repr(exc), naive_text="")
            records.append({**row, **parsed})
            if (i+1)%10000 == 0:
                print(f"Parsed {i+1}/{len(audit_rows)}", flush=True)
    df = pd.DataFrame(records)
    assert len(df)==len(audit_rows) and df.doc_id.is_unique
    assert df[["doc_id","label"]].equals(audit_rows[["doc_id","label"]])
    assert sha256(config["zip_path"])==config["expected_sha256"]
    df.to_parquet(config["processed_path"], index=False)
    summary = {"rows":len(df), "failed":len(failures), "empty_body":int(df.body_missing.sum()),
               "empty_text":int(df.lexical_text.eq("").sum()),"replacement_rows":int(df.replacement_ratio.gt(0).sum()),
               "header_leak_rows":int(df.lexical_text.map(lambda t:bool(LEAKY_LINE.search(t))).sum()),
               "removed_header_lines":int(df.removed_header_lines.sum()),
               "fallback_counts":df.fallback_reason.str.split(";").explode().value_counts().to_dict(),
               "elapsed_s":time.perf_counter()-start, "zip_unchanged":True}
    save_json("artifacts/parser_summary.json", summary); save_json("artifacts/parser_failures.json", failures)
    print(summary, flush=True)
    if failures or summary["header_leak_rows"]:
        raise RuntimeError("P1 parser acceptance failed; fix before splitting/training")
    return df
