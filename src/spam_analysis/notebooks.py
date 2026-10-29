"""Generate and execute the four course notebooks without hidden state."""
from pathlib import Path
import sys
import nbformat as nbf

def build():
    setup="""from pathlib import Path
import os, json
if not Path('configs').exists() and Path('../configs').exists():
    os.chdir('..')
import numpy as np
import pandas as pd
from IPython.display import display, Image
from spam_analysis.io import load_records
"""
    notebooks=[
        ("01_parse_and_quality",[
            ("md","# 邮件解析与质量复核\n从原始字节到可追溯派生表。此 Notebook 不打印真实邮件正文；开发示例使用合成文本。全量只展示结构审计。"),
            ("code",setup),
            ("code","audit=json.loads(Path('artifacts/audit_summary.json').read_text(encoding='utf-8'))\nquality=json.loads(Path('artifacts/parser_summary.json').read_text(encoding='utf-8'))\ndisplay(pd.Series(audit['counts']),pd.Series(quality).drop('fallback_counts'))"),
            ("md","## 伪 Base64 与损坏 multipart\n声明与结构不一致时保留可验证的原始字节。替换字符明确记录，不能假装完全恢复损坏内容。"),
            ("code","from spam_analysis.parsing import parse_email\nraw=b'Content-Type: text/plain; charset=gb2312\\nContent-Transfer-Encoding: base64\\n\\n'+'课程通知，请按时参加。'.encode('gbk')\nparsed=parse_email(raw)\ndisplay({k:parsed[k] for k in ['body_clean','used_charset','fallback_reason']})\nassert parsed['body_clean']=='课程通知,请按时参加。'"),
            ("md","## 非标准主题续行的受约束恢复\n下面是合成样例。只有负载完整、Base64 严格有效且声明字符集可严格解码时才拼接，不猜缺字、不跨其他头。真实修复影响由全量审计确认，并在 Test 前重跑比较。"),
            ("code","import base64\npayload=base64.b64encode('主题折行也应完整保留'.encode('utf-8'))\nraw=b'Subject: =?utf-8?B?'+payload[:20]+b'\\n'+payload[20:]+b'?=\\nContent-Type: text/plain; charset=utf-8\\n\\n'+'正文保留'.encode('utf-8')\nparsed=parse_email(raw)\ndisplay({k:parsed[k] for k in ['subject_clean','body_clean','fallback_reason']})\nassert parsed['subject_clean']=='主题折行也应完整保留' and parsed['body_clean']=='正文保留'"),
            ("code","splits=pd.read_parquet('artifacts/splits/split_manifest.parquet')\nassert splits.doc_id.is_unique\nassert splits.groupby('template_group_id').split.nunique().max()==1\ndisplay(splits.groupby('split').agg(mails=('doc_id','size'),groups=('template_group_id','nunique'),spam=('label','sum')))"),
            ("md","原 ZIP 未修改；delay 不作为新样本。近重复的连通分量存在链式扩展，LSH 也可能漏检。详细局限见阶段日志。")]),
        ("02_eda_and_features",[
            ("md","# pandas EDA 与 NumPy 文本特征\n所有形成特征选择的统计仅来自 Train。"),("code",setup),
            ("code","train=load_records('train')\ndisplay(train.groupby('label').agg(n=('doc_id','size'),median_body=('body_chars','median'),mean_body=('body_chars','mean'),missing_body=('body_missing','sum')))\ndisplay(train[['body_chars','subject_chars','digit_ratio']].describe(percentiles=[.5,.9,.99]))"),
            ("code","display(pd.crosstab(train.label,train.url_count.gt(0),normalize='index'))\ntrain['length_bin']=pd.qcut(train.body_chars,4,duplicates='drop')\ndisplay(train.pivot_table(index='length_bin',columns='label',values='url_count',aggfunc=['mean','count'],observed=True))"),
            ("md","## 手算 TF-IDF 的小例子\n四条合成文本已人工分词，教学例使用原始计数 TF、平滑 IDF 与 L2 归一化。正式模型使用 jieba 并启用 sublinear TF：对非零词次使用 1+log(count)，降低重复词的影响；不要把两套 TF 设置混为同一矩阵。下面的 NumPy 与 sklearn 比较使用完全相同的教学配置。"),
            ("code","from spam_analysis.teaching import tfidf_lesson\nlesson=tfidf_lesson()\nfor key in ['documents','vocabulary','counts','document_frequency','idf','tfidf','cosine_matrix']:\n    print(key); display(lesson[key])\nassert lesson['sklearn_equal']"),
            ("code","display(Image(filename='artifacts/figures/02_lengths.png'))\ndisplay(pd.read_csv('artifacts/tables/train_discriminative_terms.csv').head(12))"),
            ("md","总词次数与含词文档数是不同统计量。常见词不一定区分类别；比例与回归系数不能直接解释为因果。")]),
        ("03_baselines_and_semantics",[
            ("md","# 同特征基线与语义分支\nTune 用于候选选择；以下表格不是正式 Test 成绩。"),("code",setup),
            ("code","results=pd.read_csv('artifacts/sparse_tune_results.csv')\ndisplay(results[['candidate','dimensions','macro_f1','fp','fn','fit_s']])\ndisplay(pd.DataFrame(json.loads(Path('artifacts/convergence_audit.json').read_text(encoding='utf-8'))))\ndisplay(pd.read_csv('artifacts/sparse_supplement_results.csv'))"),
            ("md","NB 与 LR 共用同一 TF-IDF；词、字、主题与结构的贡献分开比较。初始收敛审计对应保存的各族最佳模型；未保存候选没有逐项迭代记录，不能虚填为已收敛。补充实验另有逐项记录，正式选择只在通过审计的保存模型中进行。"),
            ("code","from spam_analysis.features import NBRatio\nfrom scipy.sparse import csr_matrix\nB=csr_matrix([[1,0,2],[0,3,1],[2,0,1],[0,1,2]],dtype=float)\nratio=NBRatio().fit(B,np.array([0,1,0,1]))\ndisplay(ratio.r_)\nassert ratio.transform(B).shape==B.shape"),
            ("code","status=json.loads(Path('artifacts/semantic_status.json').read_text(encoding='utf-8'))\ndisplay(status)\nfor filename in ['semantic_tune_results.csv','macbert_learning_curves.csv']:\n    path=Path('artifacts')/filename\n    if path.exists(): display(pd.read_csv(path))\ndisplay(json.loads(Path('artifacts/qwen_profile.json').read_text(encoding='utf-8')))"),
            ("md","冻结嵌入加 LR 仍是监督分类。模型的通用榜单不代表此语料成绩。预训练数据与旧公开语料是否重叠无法完全核实。")]),
        ("04_evaluation_and_policy",[
            ("md","# 全量评价与策略\n本 Notebook 在正式 Test 已完成后读取保存的预测与冻结计划，不重新训练、不调正式阈值。"),("code",setup),
            ("code","from spam_analysis.metrics import counts,metrics,threshold_table\nfrom sklearn.metrics import confusion_matrix,f1_score\ny=np.array([0,0,1,1]); s=np.array([.2,.5,.5,.8]); pred=s>=.5\ndisplay(counts(y,pred),confusion_matrix(y,pred,labels=[0,1]))\nassert metrics(y,s)['macro_f1']==f1_score(y,pred,average='macro')\ndisplay(threshold_table(y,s))"),
            ("code","from spam_analysis.teaching import threshold_benchmark\nbenchmark=threshold_benchmark(); display(benchmark)\nfrom spam_analysis.io import save_json\nsave_json('artifacts/numpy_benchmark.json',benchmark)"),
            ("code","plan=json.loads(Path('artifacts/evaluation_plan.json').read_text(encoding='utf-8'))\nresults=pd.read_csv('artifacts/test_results.csv')\ndisplay(results.query(\"weighting=='per_mail' and point=='research'\")[[\"model\",\"n\",\"accuracy\",\"macro_f1\",\"fp\",\"fn\",\"ham_fpr\",\"spam_recall\"]])\ndisplay(pd.read_csv('artifacts/robustness_test.csv'))"),
            ("code","main=plan['main']\nm=pd.read_parquet('artifacts/splits/split_manifest.parquet').query(\"split=='test'\")\npredictions=pd.read_parquet(f'artifacts/predictions/test_{main}.parquet')\njoined=m.merge(predictions,on='doc_id',validate='one_to_one')\nassert len(joined)==len(m)\ndisplay(metrics(joined.label,joined.score_spam,plan['specifications'][main]['thresholds']['research']))\ndisplay(pd.read_csv('artifacts/tables/test_errors_by_length_bin.csv'))"),
            ("md","主分母是全部 Test 邮件。每模板等权和组 bootstrap 是敏感性及不确定性分析。Policy 满足低误拦约束不保证 Test 或新邮件满足；普通 bootstrap 的零事件区间也不意味着真实风险为零。")])]
    Path("notebooks").mkdir(exist_ok=True)
    for name,cells in notebooks:
        nb=nbf.v4.new_notebook(); nb.cells=[nbf.v4.new_markdown_cell(s) if kind=="md" else nbf.v4.new_code_cell(s) for kind,s in cells]
        nb.metadata.kernelspec={"display_name":"Python (project .venv)","language":"python","name":"python3"}
        nbf.write(nb,Path("notebooks")/(name+".ipynb"))

def execute(paths=None):
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    for path in (sorted(Path("notebooks").glob("*.ipynb")) if paths is None else paths):
        nb=nbf.read(path,as_version=4); km=KernelManager(kernel_name="python3")
        km.kernel_spec.argv=[sys.executable,"-m","ipykernel_launcher","-f","{connection_file}"]
        client=NotebookClient(nb,km=km,timeout=300,resources={"metadata":{"path":str(Path.cwd())}})
        try:
            client.execute()
        finally:
            if client.kc is not None:client.kc.stop_channels()
            if km.has_kernel:km.shutdown_kernel(now=True)
        nbf.write(nb,path); print("Executed notebook",path,flush=True)

if __name__=="__main__":
    build()
    if "--execute" in sys.argv: execute()
