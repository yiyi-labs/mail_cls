"""Build course notebooks and current-run reports from shared analysis functions."""
import argparse
import json
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT);sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'app'))
import nbformat as nbf
import pandas as pd
from spam_analysis.experiments.runtime import ExperimentSession
from spam_analysis.analysis.results import source_records,result_table,error_frame,CORE
from spam_analysis.analysis.tables import reliability,assign_slices,slice_errors,length_edges
from spam_analysis.experiments.corpus import SOURCE_NAMES
from spam_analysis.io import save_json,sha256
from model_catalog import short_name,METHODS

SETUP="""from pathlib import Path
import os,json
if not Path('configs').exists() and Path('../configs').exists():os.chdir('..')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from spam_analysis.analysis.results import source_records,result_table,error_frame
from spam_analysis.experiments.runtime import ExperimentSession
plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus']=False
engine=ExperimentSession()
print('结果版本：',engine.run_id)
"""


def notebooks():
    books=[('01_parse_and_quality',[
        ('md','# 可靠邮件与统一数据表\n三个公开来源统一存放，来源、标签依据、原文哈希与用途仍可追溯。这里只看聚合质量和合成解析示例，不打印真实邮件全文。'),
        ('code',SETUP),
        ('code',"frame=source_records('combined')\nassert frame.doc_id.is_unique\nassert frame.groupby('template_group_id').split.nunique().max()==1\ndisplay(pd.crosstab(frame.source_id,frame.label))\ndisplay(pd.crosstab(frame.source_id,frame.split))"),
        ('code',"quality=frame.groupby('source_id').agg(邮件数=('doc_id','size'),正文为空=('body_missing','sum'),含损坏字符=('replacement_ratio',lambda s:s.gt(0).sum()))\ndisplay(quality)\ndisplay(pd.crosstab(frame.script_profile,frame.label))"),
        ('md','文字类型来自正文比例规则，不是人工语言标注。原 TREC06C 内容与划分冻结，新来源先排除跨来源重复，再按模板组划分。来源可靠不等于全部标注绝无错误。'),
        ('code',"from spam_analysis.parsing import parse_email\nraw=b'Content-Type: text/plain; charset=gb2312\\nContent-Transfer-Encoding: base64\\n\\n'+'课程通知，请按时参加。'.encode('gbk')\nparsed=parse_email(raw)\ndisplay({k:parsed[k] for k in ['body_clean','used_charset','fallback_reason']})\nassert parsed['body_clean']=='课程通知,请按时参加。'"),
        ('md','原始字节与声明可能不一致。验证传输编码后再解码，保留无法恢复的损坏标记；不通过猜字补齐内容。')]),
        ('02_eda_and_features',[
        ('md','# Pandas 探索与 NumPy 文本特征\n探索统计与词表只从训练邮件形成，和网页共用分析函数。'),('code',SETUP),
        ('code',"train=source_records('combined','train')\ndisplay(train.groupby(['source_id','label']).agg(邮件数=('doc_id','size'),长度均值=('body_chars','mean'),长度中位数=('body_chars','median'),P90=('body_chars',lambda s:s.quantile(.9))))\ndisplay(pd.crosstab(train.label,train.url_count.gt(0),normalize='index'))"),
        ('code',"from spam_analysis.analysis.distributions import grouped_distribution\ndistribution=grouped_distribution(train,'body_chars')\nassert distribution.groupby('类别')['邮件数'].sum().sum()==len(train)\nchart=distribution.pivot(index='顺序',columns='类别',values='类内比例')\nchart.plot.bar(figsize=(9,3),color=['#6265d9','#be794f']);plt.ylabel('占本类别邮件的比例');plt.tight_layout();plt.show()\ndisplay(distribution.head())"),
        ('code',"from spam_analysis.analysis.distributions import corpus_terms\nterms=corpus_terms(train)\ndisplay(pd.concat([terms.nlargest(6,'spam_minus_ham'),terms.nsmallest(6,'spam_minus_ham')]))"),
        ('md','每封邮件只计一次词语出现。比较类内比例，避免类别数量影响；常见词不一定更有区分力，关联也不等于因果。'),
        ('code',"from spam_analysis.teaching import tfidf_lesson\nlesson=tfidf_lesson()\nfor key in ['documents','vocabulary','counts','document_frequency','idf','tfidf','cosine_matrix']:\n    print(key);display(lesson[key])\nassert lesson['sklearn_equal']"),
        ('md','教学例采用计数 TF、平滑 IDF 和 L2 归一化。正式词模型启用 sublinear TF，即非零计数使用 1+ln(count)，二者设置不同。')]),
        ('03_baselines_and_semantics',[
        ('md','# 五类分类方法与训练过程\n朴素贝叶斯和逻辑回归共享词 TF-IDF，SVM 加入字符与主题。Qwen 冻结编码器学习分类头，MacBERT 全量微调。以下是开发过程，不把 Tune 当 Test。'),('code',SETUP),
        ('code',"run=Path('artifacts/runs')/engine.run_id\nfor filename in ['sparse_training.csv','qwen_training.csv','main_selection.csv']:\n    path=run/filename\n    if path.exists():print(filename);display(pd.read_csv(path))\n    else:print(filename,'尚未完成')"),
        ('code',"path=run/'training_curves.csv'\nif path.exists():\n    curves=pd.read_csv(path);display(curves)\n    fig,axes=plt.subplots(1,2,figsize=(10,3))\n    for (mode,lr),g in curves.groupby(['loss_mode','lr']):\n        axes[0].plot(g.epoch,g.training_loss,marker='o',label=f'{mode} {lr:g}')\n        axes[1].plot(g.epoch,g.accuracy,marker='o',label=f'{mode} {lr:g}')\n    axes[0].set_title('训练损失');axes[1].set_title('Tune 准确率');axes[1].legend(fontsize=8);plt.tight_layout();plt.show()\nelse:print('神经训练尚未生成完整轮次记录')"),
        ('code',"conditions=[]\nfor name in ['B1','B2','S_char13','E1','E2']:\n    item=engine.entries[name]\n    conditions.append({'方法':name,'训练样本':item.get('train_rows','见基线记录'),'训练先验':item.get('train_prior'),'模型身份':item['model_sha256'][:12]})\ndisplay(pd.DataFrame(conditions))"),
        ('md','比较完整方案，参数量和输入长度并不相同。模型选择依据开发数据；数据增多不保证效果提高。GPU 断点保存优化器、随机状态与批次位置，避免中断后把不同训练过程混为一次。')]),
        ('04_evaluation_and_policy',[
        ('md','# 相同邮件上的评价与错误分析\n读取当前发布版本的保存预测。固定留出集用于回顾评估：发布修订前已观察旧模型成绩，不能视作全新独立确认。模型拟合、校准与阈值选择均未使用 Test。'),('code',SETUP),
        ('code',"from spam_analysis.metrics import counts,metrics,threshold_table\nfrom sklearn.metrics import confusion_matrix\ny=np.array([0,0,1,1]);p=np.array([.2,.5,.5,.8])\ndisplay(counts(y,p>=.5),confusion_matrix(y,p>=.5,labels=[0,1]))\ndisplay(threshold_table(y,p))"),
        ('code',"scope='trec06c' if engine.run_id=='frozen-v1' else 'combined'\nresults=result_table(engine.run_id,scope)\ndisplay(results[['model','n','accuracy','macro_f1','fp','fn','ham_fpr']])\nthreshold=engine.specifications[engine.main]['thresholds']['research']\nevaluated=error_frame(engine.run_id,scope,engine.main,threshold)\nassert len(evaluated)==int(results.set_index('model').loc[engine.main,'n'])"),
        ('code',"from spam_analysis.analysis.tables import reliability,assign_slices,slice_errors,length_edges\ntrain=source_records('combined','train')\nfor field in ['source','language','length','quality']:\n    print(field);display(slice_errors(evaluated,assign_slices(evaluated,field,length_edges(train))))"),
        ('code',"bins,calibration=reliability(evaluated)\ndisplay(bins,calibration)\nassert bins['count'].sum()==len(evaluated)\nused=bins.query('count>0');plt.figure(figsize=(5,4));plt.plot([0,1],[0,1],'--',color='gray');plt.plot(used.mean_probability,used.spam_fraction,'o-',color='#6265d9');plt.xlabel('平均预测垃圾概率');plt.ylabel('实际垃圾比例');plt.tight_layout();plt.show()"),
        ('md','高分不保证单封正确。可靠性曲线比较区间平均预测与实际垃圾比例，Brier 衡量概率误差；小分组误差率不稳定。旧公开语料高分也不能证明现代中文邮箱效果。')])]
    for name,cells in books:
        nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(t) if k=='md' else nbf.v4.new_code_cell(t) for k,t in cells])
        nb.metadata.kernelspec={'display_name':'Python (project .venv)','language':'python','name':'python3'}
        nbf.write(nb,ROOT/'notebooks'/f'{name}.ipynb')


def report():
    engine=ExperimentSession()
    if engine.run_id=='frozen-v1':raise RuntimeError('新实验尚未发布，不能覆盖最终成果报告')
    run=ROOT/'artifacts/runs'/engine.run_id
    corpus=source_records('combined');train=source_records('combined','train');results=result_table(engine.run_id)
    main=results.set_index('model').loc[engine.main];threshold=engine.specifications[engine.main]['thresholds']['research']
    evaluated=error_frame(engine.run_id,'combined',engine.main,threshold);bins,cal=reliability(evaluated)
    source_errors=slice_errors(evaluated,assign_slices(evaluated,'source'))
    language_errors=slice_errors(evaluated,assign_slices(evaluated,'language'))
    perf=pd.read_csv(run/'benchmark.csv');selection=json.loads((run/'selection.json').read_text(encoding='utf-8'))
    coverage=corpus.groupby(['source_id','label']).size().unstack(fill_value=0)
    terms=pd.read_parquet(ROOT/'artifacts/tables/corpus_terms.parquet')
    keywords=pd.concat([terms.nsmallest(3,'spam_minus_ham'),terms.nlargest(3,'spam_minus_ham')])
    data=dict(run_id=engine.run_id,main=engine.main,main_name=short_name(engine.main),rows=len(corpus),train_rows=len(train),
        test_rows=len(evaluated),sources=[{'name':SOURCE_NAMES[s],'ham':int(row[0]),'spam':int(row[1])} for s,row in coverage.iterrows()],
        results=results.fillna('').to_dict('records'),benchmark=perf.fillna('').to_dict('records'),
        by_source=source_errors.fillna('').to_dict('records'),by_language=language_errors.fillna('').to_dict('records'),
        calibration=cal,reliability=bins.fillna('').to_dict('records'),selection=selection,
        names={name:short_name(name) for name in CORE},methods=METHODS,
        keywords=keywords.to_dict('records'),split_counts=corpus.groupby('split').size().to_dict(),
        evaluation_note=engine.plan.get('evaluation_note','原中文 Test 为历史回顾，新增来源为分组留出。'),
        plan_sha256=sha256(run/'evaluation_plan.json'))
    save_json('artifacts/presentation_data.json',data)
    def md_table(headers,rows):return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+'\n'.join('| '+' | '.join(str(v) for v in row)+' |' for row in rows)
    scores=md_table(['方法','准确率','Macro-F1','正常误拦','垃圾漏判'],[[short_name(r.model),f'{r.accuracy:.3%}',f'{r.macro_f1:.3%}',int(r.fp),int(r.fn)] for r in results.itertuples()])
    sources=md_table(['来源','正常','垃圾','合计'],[[SOURCE_NAMES[s],int(r[0]),int(r[1]),int(r.sum())] for s,r in coverage.iterrows()])
    groups=md_table(['来源','评价数','错误率','正常误拦','垃圾漏判'],[[r.group,int(r.count),f'{r.error_rate:.3%}',int(r.false_positive),int(r.false_negative)] for r in source_errors.itertuples()])
    cost=md_table(['方法','推理中位数','P95','GPU 峰值'],[[short_name(r.model),f'{r.median_ms:.1f} ms',f'{r.p95_ms:.1f} ms',f'{r.gpu_allocated_peak_mib:.0f} MiB' if pd.notna(r.gpu_allocated_peak_mib) else 'CPU'] for r in perf.itertuples()])
    text=f'''# 邮件分类与 Python 数据分析

本报告对应已发布版本 `{engine.run_id}`。主模型为 **{short_name(engine.main)}**，由开发数据选择。统一留出汇总 {len(evaluated):,} 封，准确率 {main.accuracy:.3%}，正常误拦 {int(main.fp)} 封，垃圾漏判 {int(main.fn)} 封。

## 数据与处理

{sources}

共 {len(corpus):,} 封，训练 {len(train):,} 封。邮件来自 TREC06C、Apache SpamAssassin 和 AUEB Enron-Spam 的官方发布。统一表保留原始标签、来源、哈希和用途。新增来源排除重复与已知重叠模板，同组不跨集合。文字类型按正文比例统计，不能把来源直接当成逐封语言标签。

## Python 分析流程

Pandas 统一表与分组统计，NumPy 计算分箱、分位数、归一化和混淆计数，scikit-learn 建立 TF-IDF 与分类器。网页与 Notebook 共用分析函数。数据分析连接来源、长度、质量、词语和错误，而不只展示模型榜单。

## 分类比较

{scores}

五种方法覆盖词统计、线性分类与语义表示。B1/B2 共用词 TF-IDF；SVM 加入字符和主题；Qwen 冻结 0.6B 编码器，MacBERT 监督微调。参数量、输入长度与训练方式不同，比较的是完整方案，不是等参数模型。

## 泛化与错误分布

{groups}

{data['evaluation_note']} 训练与留出邮件按已识别模板组分开，训练中使用同来源的其他邮件，因此不属于全新来源测试。公开数据主要来自 1999–2006 年，现代中文邮箱效果仍待独立检验。AI 辅助抽样复核没有替代人工全量标注。

浏览器验收中的合成课程通知“课程项目答疑时间”（正文为本周四项目答疑安排）仍被当前 MacBERT 误判为垃圾，分数约 0.9983。该例未加入训练，也不计入正式成绩；它说明历史语料高分不能保证现代课程邮件的效果，后续需要独立人工核对的现代邮件集，而不能围绕这个例子临时修改阈值。

## 概率与成本

主模型 Brier 为 {cal['brier']:.6f}，10 区间 ECE 为 {cal['ece']:.6f}。可靠性曲线比较平均预测垃圾概率与实际垃圾占比，高置信度仍可能误判。

{cost}

性能测试从相同训练邮件分层抽样，统一 UTF-8 封装，单封端到端解析与推理；预热后重复三次，模型加载另计。硬件、样本与参数量保存在本轮 benchmark.csv。网页采集计算轨迹会带来额外耗时，不能与该基准直接混用。

## 交互与复现

邮件实验室支持文字、EML/TXT/CSV、本地批量分类。真实词元、TF-IDF、线性贡献与中间张量可回放；修改片段后以同模型同阈值重算。动画不是额外训练，也不意味着因果解释。

训练更新参数，Tune 选候选，Calibrate 校准，Policy 定阈值与主模型。Test 只用于回顾评价。原发布规则要求各来源指标同时不降，导致全部新候选被拒绝；该次结果完整保留。本次展示已完成训练的五种方法，按 Policy 总体准确率选择，并报告各来源的得失。没有重新拟合测试邮件，也没有根据测试成绩调整参数或阈值。训练与阶段恢复代码保留在 experiments/ 和 scripts/，本轮已全部完成。

来源：[NIST](https://trec.nist.gov/pubs/trec15/papers/SPAM06.OVERVIEW.pdf)、[Apache](https://spamassassin.apache.org/old/publiccorpus/readme.html)、[AUEB](https://www2.aueb.gr/users/ion/data/enron-spam/readme.txt)。完整参数、预测、来源哈希及不确定性分析见 `artifacts/runs/{engine.run_id}/`。
'''
    (ROOT/'reports/results.md').write_text(text,encoding='utf-8')
    from markdown_it import MarkdownIt
    html='<!doctype html><meta charset="utf-8"><title>邮件分类课程报告</title><style>body{max-width:1000px;margin:50px auto;padding:0 30px;font:16px/1.85 "Microsoft YaHei",sans-serif;color:#222a40;background:#f5f6fa}h1,h2{color:#484db2}table{border-collapse:collapse;width:100%;background:white}td,th{padding:10px;border-bottom:1px solid #dfe3ee;text-align:left}a{color:#6265d9}code{overflow-wrap:anywhere}</style>'+MarkdownIt().enable('table').render(text)
    (ROOT/'reports/results.html').write_text(html,encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['notebooks','report']);parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if args.stage=='notebooks':
        notebooks()
        if args.execute:
            from spam_analysis.notebooks import execute
            execute()
    else:report()
