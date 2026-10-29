"""Create a fresh local run directory without altering existing results or Git."""
import argparse
import re
import shutil
from pathlib import Path
from spam_analysis.io import read_config,sha256

parser=argparse.ArgumentParser();parser.add_argument('--name',required=True);args=parser.parse_args()
if not re.fullmatch(r'[A-Za-z0-9_-]{1,60}',args.name): raise ValueError('Use a short alphanumeric/hyphen name')
source=Path(__file__).resolve().parents[1]; target=source/'reproductions'/args.name
if not target.resolve().is_relative_to(source.resolve()) or target.exists(): raise ValueError('Target must be a new directory within this project')
config=read_config(source/'configs/data.yaml')
archive=Path(config['zip_path'])
if archive.is_absolute() or '..' in archive.parts: raise ValueError('Dataset path must stay within the project')
if sha256(source/archive)!=config['expected_sha256']: raise ValueError('Original ZIP identity mismatch')
target.mkdir(parents=True)
for name in ['src','configs','tests','app','scripts','examples','.streamlit']:
    shutil.copytree(source/name,target/name,ignore=shutil.ignore_patterns('__pycache__','*.egg-info','secrets.toml'))
for name in ['pyproject.toml','requirements-lock.txt','.gitignore','.gitattributes','README.md','启动演示.cmd','分工.md']:
    shutil.copy2(source/name,target/name)
(target/'reports').mkdir()
for name in ['experiment_plan.md','course_alignment.md','contributions_and_ai_use.md']:
    shutil.copy2(source/'reports'/name,target/'reports'/name)
(target/'reports/stage_log.md').write_text('# 新复现运行\n\n此目录只复制实现、计划与原始 ZIP，尚未执行实验。原实现的修复和阶段记录保留在主工作区；本目录应按实际运行追加自己的证据，不能复制旧成绩当新结果。\n',encoding='utf-8')
(target/'artifacts').mkdir()
for name in ['qwen_identity.json','macbert_identity.json']:
    if (source/'artifacts'/name).exists():shutil.copy2(source/'artifacts'/name,target/'artifacts'/name)
(target/archive).parent.mkdir(parents=True,exist_ok=True)
shutil.copy2(source/archive,target/archive)
print('Fresh local reproduction directory:',target)
print('No previous predictions/models or Git metadata copied. Restore pinned model assets before semantic training.')
