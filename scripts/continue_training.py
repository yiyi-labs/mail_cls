"""Resume the ordered local experiment; never refit after the plan is frozen."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));os.chdir(ROOT)
from spam_analysis.experiments.train import RUN
from spam_analysis.experiments.checkpoint import exclusive_run,atomic_json


def run(stage,*arguments):
    atomic_json(RUN/'pipeline_status.json',{'stage':stage,'status':'running'})
    print(stage,flush=True)
    with (RUN/'pipeline.log').open('a',encoding='utf-8') as log:
        log.write('\n'+stage+'\n');log.flush()
        # Console Ctrl+C reaches the child too. Give its checkpoint handler time
        # to finish instead of letting subprocess.run kill it immediately.
        previous=signal.signal(signal.SIGINT,lambda *_:None)
        try:
            subprocess.run([sys.executable,'-u',*arguments],stdout=log,stderr=subprocess.STDOUT,
                           check=True,env={**os.environ,'PYTHONUTF8':'1'})
        finally:signal.signal(signal.SIGINT,previous)


def main(wait):
    # A completed-training release has its own immutable plan and resumable stages.
    if RUN==ROOT/'artifacts/runs/course-20261015' and (RUN.parent/'course-20261016/config.json').exists():
        subprocess.run([sys.executable,'-u','scripts/publish_trained_models.py'],check=True)
        return
    with exclusive_run(RUN,'pipeline.lock'):
        while True:
            try:
                with exclusive_run(RUN):pass
                break
            except RuntimeError:
                if not wait:raise
                atomic_json(RUN/'pipeline_status.json',{'stage':'等待当前训练完成','status':'waiting'})
                time.sleep(10)
        try:
            if not (RUN/'evaluation_plan.json').exists():
                candidates=json.loads((RUN/'candidates.json').read_text(encoding='utf-8')) if (RUN/'candidates.json').exists() else {}
                progress=json.loads((RUN/'neural_progress.json').read_text(encoding='utf-8')) if (RUN/'neural_progress.json').exists() else []
                if not ({'B1','B2','S_char13','E1','E2'}<=set(candidates) and len(progress)==12):
                    run('恢复或完成训练','-m','spam_analysis.experiments.train','fit')
                run('核心检查','-m','pytest','-q')
                run('校准、选择并冻结方案','-m','spam_analysis.experiments.train','select')
            run('冻结方案评价','-m','spam_analysis.experiments.evaluate','evaluate')
            if not (RUN/'benchmark.csv').exists():
                run('同机推理比较','-m','spam_analysis.experiments.evaluate','benchmark')
            run('发布已完成结果','-m','spam_analysis.experiments.evaluate','publish')
            run('网页与神经推理检查','scripts/validate_local_app.py','--neural')
            run('同步结果报告','scripts/build_course_materials.py','report')
            run('执行课程 Notebook','scripts/build_course_materials.py','notebooks','--execute')
            atomic_json(RUN/'pipeline_status.json',{'stage':'模型、网页结果与分析材料已更新','status':'complete',
                'remaining':'课件需按新结果生成并逐页检查。'})
            print('模型与分析材料已更新。课件仍需生成并逐页检查。',flush=True)
        except BaseException:
            status=json.loads((RUN/'pipeline_status.json').read_text(encoding='utf-8'))
            status['status']='interrupted_or_failed';atomic_json(RUN/'pipeline_status.json',status)
            print('流程已停止；已完成阶段与断点保留，详情见本轮 pipeline.log。',flush=True)
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--wait-for-running',action='store_true')
    main(parser.parse_args().wait_for_running)
