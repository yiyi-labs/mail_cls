"""Exercise the three presentation pages and read-only local prediction flow."""
import sys,os,json,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));os.chdir(ROOT)
from streamlit.testing.v1 import AppTest
from spam_analysis.io import sha256,save_json
from spam_analysis.experiments.runtime import active_plan


def no_errors(app):
    assert not app.exception,[str(e.value) for e in app.exception]
    assert not app.error,[str(e.value) for e in app.error]


def snapshot():
    paths=[Path('artifacts/evaluation_plan.json'),Path('artifacts/test_results.csv'),active_plan(),Path('data/processed/corpus.parquet')]
    paths+=list(Path('artifacts/predictions').glob('test_*.parquet'))
    return {str(p):sha256(p) for p in paths if p.exists()}


parser=argparse.ArgumentParser();parser.add_argument('--neural',action='store_true');args=parser.parse_args()
before=snapshot();checks=[]
app=AppTest.from_file(str(ROOT/'app/app.py'),default_timeout=240).run();no_errors(app)
app.selectbox(key='chosen_model').set_value('B2').run()
app.button(key='classify').click().run();no_errors(app)
trace=app.session_state['last_result']['trace'];assert trace['branches']['B2']['contributions']
checks.append('Real sparse inference and contributions')
app.text_area(key='edit_body').set_value('请明天下午提交课程报告。').run()
app.button(key='recalculate').click().run();no_errors(app)
assert app.session_state['edit_result']['trace']['model']=='B2'
app.button(key='undo_edit').click().run();no_errors(app)
assert 'edit_result' not in app.session_state
checks.append('Local edit, same-model recomputation and restore')
app.multiselect(key='compare_models').set_value(['B1','B2','S_char13']).run()
app.button(key='compare_mail').click().run();no_errors(app)
assert len(app.session_state['mail_comparison']['rows'])==3
checks.append('Same-message comparison')
app.text_area(key='body').set_value('更新输入。').run();no_errors(app)
assert app.button(key='recalculate').disabled
app.text_input(key='subject').set_value('').run();app.text_area(key='body').set_value('').run()
app.button(key='classify').click().run();no_errors(app)
assert app.session_state['last_result']['trace']['branches']['B2']['prior_only']
checks.append('Input invalidation and empty-input prior')
app.radio(key='input_mode').set_value('数据集邮件').run();no_errors(app)
assert app.selectbox(key='mail_split').value=='test'
assert 'mail_source' not in [x.key for x in app.selectbox]
app.text_input(key='mail_query').set_value('NONEXISTENT_9999_TEST').run();no_errors(app)
assert app.button(key='classify').disabled
checks.append('Unified catalog, independent split filter and empty search')
app.radio(key='input_mode').set_value('上传文件').run();no_errors(app)
assert app.button(key='classify').disabled
for page in ['模型对比','数据分析']:
    app.radio(key='page').set_value(page).run();no_errors(app)
checks.append('Version-matched model comparison and unified data analysis')
for field in ['source','language','length','quality']:
    app.selectbox(key='slice_field').set_value(field).run();no_errors(app)
checks.append('Source, script, length and quality slices')
if args.neural:
    app.radio(key='page').set_value('邮件实验室').run();app.radio(key='input_mode').set_value('自己输入').run()
    app.text_input(key='subject').set_value('通知').run();app.text_area(key='body').set_value('请提交课程项目。').run()
    for name in ['E1','E2']:
        app.selectbox(key='chosen_model').set_value(name).run();app.button(key='classify').click().run();no_errors(app)
        assert app.session_state['last_result']['trace']['model']==name
    checks.append('Both neural model inference traces')
assert snapshot()==before,'UI modified frozen evidence or corpus'
checks.append('Read-only data and evidence identities')
published=active_plan()
save_json('artifacts/app_validation.json',dict(passed=True,checks=checks,plan_sha256=sha256(published),
    run_id=json.loads(published.read_text(encoding='utf-8')).get('run_id','frozen-v1')))
print('Local app validation:',len(checks),'checks passed')
