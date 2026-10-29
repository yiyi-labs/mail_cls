from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'app'),str(ROOT/'src')]
from isolation_audit import audit_isolation
from spam_analysis.io import save_json

if __name__=='__main__':
    audit=audit_isolation()
    save_json('artifacts/web_isolation_audit.json',audit)
    print(f"Isolation audit: {sum(c['通过'] for c in audit['checks'])}/{len(audit['checks'])} checks passed")
    for check in audit['checks']:
        if not check['通过']:print(check)
    if not audit['passed']:raise SystemExit(1)
