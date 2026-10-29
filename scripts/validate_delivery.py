"""Final acceptance: tests, frozen identities, arithmetic and delivered artifacts."""
import datetime
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import os
os.chdir(ROOT)
import nbformat
import yaml
from spam_analysis.io import sha256,read_config,save_json
from spam_analysis.evaluation import source_identity
from spam_analysis.reproducibility import verify_predictions

unit=subprocess.run([sys.executable,'-m','pytest','-q'],capture_output=True,text=True,encoding='utf-8')
Path('artifacts/private/final_pytest.txt').write_text(unit.stdout+unit.stderr,encoding='utf-8')
print(unit.stdout,flush=True)
if unit.returncode:raise RuntimeError('Unit tests failed; see private final_pytest.txt')
matched=re.search(r'(\d+) passed',unit.stdout)
assert matched
plan=json.loads(Path('artifacts/evaluation_plan.json').read_text(encoding='utf-8'))
current_sources=source_identity()
amendment_path=Path('artifacts/reporting_amendment.json')
if amendment_path.exists():
    amendment=json.loads(amendment_path.read_text(encoding='utf-8'))
    allowed={'src/spam_analysis/notebooks.py','src/spam_analysis/reporting.py','src/spam_analysis/result_plots.py'}
    assert set(amendment['files'])==allowed
    expected=dict(plan['source_hashes'])
    for name,identity in amendment['files'].items():
        assert identity['before']==expected[name]
        expected[name]=identity['after']
    for name,digest in amendment['unchanged_evidence'].items():assert sha256(name)==digest
else:
    expected=dict(plan['source_hashes'])
location_path=Path('artifacts/data_location_amendment.json')
if location_path.exists():
    location=json.loads(location_path.read_text(encoding='utf-8'))
    name='configs/data.yaml'
    assert location['file']==name and location['before']==expected[name]
    assert hashlib.sha256(location['before_text'].encode('utf-8')).hexdigest()==location['before']
    assert hashlib.sha256(location['after_text'].encode('utf-8')).hexdigest()==location['after']
    old_config=yaml.safe_load(location['before_text']); new_config=yaml.safe_load(location['after_text'])
    assert old_config['zip_path']=='题6_中文垃圾邮件_trec06c.zip'
    assert new_config['zip_path']=='data/raw/题6_中文垃圾邮件_trec06c.zip'
    assert {k:v for k,v in old_config.items() if k!='zip_path'}=={k:v for k,v in new_config.items() if k!='zip_path'}
    assert new_config['expected_sha256']==plan['zip_sha256']==location['zip_sha256']
    expected[name]=location['after']
assert current_sources==expected,'Changes exceed the documented reporting repair and dataset relocation'
assert sha256('data/processed/records.parquet')==plan['processed_sha256']
assert sha256('artifacts/splits/split_manifest.parquet')==plan['manifest_sha256']
assert sha256(read_config('configs/data.yaml')['zip_path'])==plan['zip_sha256']
for entry in plan['registry'].values():
    assert sha256(entry['path'])==entry['model_sha256']
    assert sha256(entry['calibrator_path'])==entry['calibrator_sha256']
    if entry['type']=='semantic':
        from spam_analysis.model_assets import verify_identity
        verify_identity(entry['identity'])
        if 'checkpoint' in entry:
            checkpoint=entry['checkpoint']
            for name,digest in checkpoint['files'].items():
                assert sha256(Path(checkpoint['folder'])/name)==digest
verify_predictions()
from spam_analysis.experiments.runtime import ACTIVE,ExperimentSession
new_run=None
if ACTIVE.exists():
    import numpy as np
    from spam_analysis.experiments import evaluate as evaluation_module
    from spam_analysis.experiments.runtime import active_plan
    from spam_analysis.analysis.results import source_records,result_table,saved_predictions,CORE
    from spam_analysis.analysis.tables import prediction_frame
    from spam_analysis.metrics import metrics
    evaluation_module.RUN=active_plan().parent
    updated=evaluation_module.verify_plan();engine=ExperimentSession();run=Path('artifacts/runs')/engine.run_id
    if 'release_script_sha256' in updated:
        assert sha256('scripts/publish_trained_models.py')==updated['release_script_sha256']
    assert engine.plan==updated
    completed=json.loads((run/'test_complete.json').read_text(encoding='utf-8'))
    assert completed['plan_sha256']==sha256(run/'evaluation_plan.json')
    for name,digest in completed['files'].items():assert sha256(name)==digest,name
    for name,entry in engine.entries.items():
        assert sha256(entry['path'])==entry['model_sha256']
        assert sha256(entry['calibrator_path'])==entry['calibrator_sha256']
        if 'checkpoint' in entry:
            for filename,digest in entry['checkpoint']['files'].items():
                assert sha256(Path(entry['checkpoint']['folder'])/filename)==digest
    records=source_records('combined','test');expected_rows=result_table(engine.run_id).set_index('model')
    for name in CORE:
        threshold=engine.specifications[name]['thresholds']['research']
        joined=prediction_frame(records,saved_predictions(engine.run_id,'combined',name),threshold)
        calculated=metrics(joined.label,joined.score_spam,threshold)
        for field in ['n','tn','fp','fn','tp','accuracy','macro_f1','brier']:
            assert np.isclose(calculated[field],expected_rows.loc[name,field],rtol=0,atol=1e-12),(name,field)
    new_run=dict(run_id=engine.run_id,test_rows=len(records),main=engine.main,results_reconciled=True)
checks={}
for name in ['test_complete.json','app_validation.json','saved_sparse_validation.json','macbert_reload_validation.json']:
    content=json.loads((Path('artifacts')/name).read_text(encoding='utf-8'))
    checks[name]=content
    if 'passed' in content:assert content['passed']
assert checks['test_complete.json']['all_predictions_aligned'] and checks['test_complete.json']['zip_unchanged']
status=json.loads(Path('artifacts/semantic_status.json').read_text(encoding='utf-8'))
assert status.get('E1')=='complete' and status.get('E2')=='complete',status
notebooks=[]
for path in sorted(Path('notebooks').glob('*.ipynb')):
    book=nbformat.read(path,as_version=4)
    for cell in book.cells:
        if cell.cell_type!='code' or not cell.source.strip():continue
        assert cell.execution_count is not None,path
        assert not any(out.output_type=='error' for out in cell.outputs),path
    notebooks.append(str(path))
assert len(notebooks)==4
for name in ['results.md','results.html','Topic6_课程汇报.pptx','五分钟视频讲稿.md']:
    assert (Path('reports')/name).stat().st_size>0,name
for name in ['分工.md']:
    assert Path(name).stat().st_size>0,name
with ZipFile('reports/Topic6_课程汇报.pptx') as deck:
    assert deck.testzip() is None
    slides=[n for n in deck.namelist() if re.fullmatch(r'ppt/slides/slide\d+\.xml',n)]
    # OPC relationships/content types define chart parts; exporters may legally
    # place them under ppt/slides/charts rather than ppt/charts.
    types=ET.fromstring(deck.read('[Content_Types].xml'))
    charts=[node.attrib['PartName'].lstrip('/') for node in types
            if node.attrib.get('ContentType')=='application/vnd.openxmlformats-officedocument.drawingml.chart+xml']
    for name in charts:
        assert ET.fromstring(deck.read(name)).tag=='{http://schemas.openxmlformats.org/drawingml/2006/chart}chartSpace'
    assert len(slides)==8 and len(charts)>=4
qa=json.loads(Path('artifacts/presentation_visual_qa.json').read_text(encoding='utf-8'))
assert qa['passed'] and qa['reviewed_slides']==list(range(1,9))
assert qa['pptx_sha256']==sha256('reports/Topic6_课程汇报.pptx')
for name,digest in qa['slide_hashes'].items():assert sha256(Path('artifacts/figures/slides')/name)==digest
save_json('artifacts/final_validation.json',dict(passed=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    unit_tests_passed=int(matched[1]),frozen_computational_source_and_assets_unchanged=True,
    reporting_only_amendment=amendment_path.exists(),data_location_amendment=location_path.exists(),notebooks=notebooks,
    semantic_status=status,published_run=new_run,slides=len(slides),native_charts=len(charts),app_checks=checks['app_validation.json'],
    prediction_reconciliation=json.loads(Path('artifacts/prediction_reconciliation.json').read_text(encoding='utf-8'))))
print('Delivery checks passed against saved results. This check did not train models or generate new Test predictions.')
