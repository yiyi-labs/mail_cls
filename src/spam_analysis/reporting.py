"""Evidence-based Chinese report chapters; no manually invented experiment data."""
import json
import html
from pathlib import Path
import pandas as pd


def read_json(name): return json.loads((Path('artifacts')/name).read_text(encoding='utf-8'))


def table(frame,percent=()):
    data=frame.copy()
    for column in data:
        if column in percent: data[column]=data[column].map(lambda x:f'{float(x)*100:.4f}%')
        elif pd.api.types.is_float_dtype(data[column]): data[column]=data[column].map(lambda x:f'{x:.6g}' if pd.notna(x) else '—')
    escape=lambda value:str(value).replace('|','\\|').replace('\n',' ')
    rows=['| '+' | '.join(map(escape,data.columns))+' |','|'+'|'.join(['---']*len(data.columns))+'|']
    rows+=['| '+' | '.join(map(escape,row))+' |' for row in data.itertuples(index=False,name=None)]
    return '\n'.join(rows)


def build_report():
    if not Path('artifacts/test_complete.json').exists(): raise RuntimeError('Report requires completed frozen evaluation')
    audit=read_json('audit_summary.json'); parser=read_json('parser_summary.json'); groups=read_json('group_summary.json')
    plan=read_json('evaluation_plan.json'); policy=read_json('policy_selection.json'); boot=read_json('bootstrap_test.json')
    result=pd.read_csv('artifacts/test_results.csv'); research=result.query("point=='research' and weighting=='per_mail'")
    main=research.set_index('model').loc[plan['main']]
    splits=pd.read_csv('artifacts/split_summary.csv'); grid=pd.read_csv('artifacts/sparse_tune_results.csv')
    supplement=pd.read_csv('artifacts/sparse_supplement_results.csv'); stability=read_json('sparse_stability.json')
    semantic=pd.read_csv('artifacts/semantic_tune_results.csv') if Path('artifacts/semantic_tune_results.csv').exists() else pd.DataFrame()
    semstatus=read_json('semantic_status.json'); robust=pd.read_csv('artifacts/robustness_test.csv')
    performance=pd.read_csv('artifacts/performance_development.csv')
    random=read_json('random_split_diagnostic.json'); profile=read_json('qwen_profile.json') if Path('artifacts/qwen_profile.json').exists() else {}
    limits=boot['models'][plan['main']]['ci95']; formal=list(plan['registry'])
    if plan['main'] not in formal: formal.append(plan['main'])
    baseline=research.set_index('model').loc['B2']
    paired_rows=[]
    for name in formal:
        if name==plan['main']:continue
        interval=boot['models'][name]['main_minus_model_macro_f1_ci95']
        paired_rows.append(dict(comparator=name,main_minus_comparator=float(main.macro_f1-research.set_index('model').loc[name,'macro_f1']),ci95_low=interval[0],ci95_high=interval[1]))
    indexed=grid.set_index('candidate'); sparse_registry=read_json('sparse_registry.json')
    cap_rows=[]
    for family in ['B2',policy['best_sparse'],'B2_cap200k','S_cap']:
        if family not in sparse_registry: continue
        for block,values in sparse_registry[family]['cap_audit'].items(): cap_rows.append(dict(model=family,block=block,**values))
    cap_table=pd.DataFrame(cap_rows)
    quality=pd.read_csv('artifacts/tables/train_quality.csv')
    quality['replacement_rate']=quality.replacement_rows/quality.n
    quality['class']=quality.label.map({0:'正常 ham',1:'垃圾 spam'})
    url_counts=pd.read_csv('artifacts/tables/train_url_crosstab.csv',index_col=0)
    url_counts.index=url_counts.index.map(lambda value:str(value).lower())
    url_analysis=pd.DataFrame([{'类别':name,'含链接封数':int(url_counts.loc['true',name]),'总封数':int(url_counts[name].sum()),
        '含链接比例':float(url_counts.loc['true',name]/url_counts[name].sum())} for name in ['正常','垃圾']])
    eda=read_json('eda_summary.json')
    company=pd.read_csv('artifacts/tables/train_discriminative_terms.csv').set_index('term').loc['公司']
    repair=read_json('solver_repair.json') if Path('artifacts/solver_repair.json').exists() else None
    repair_summary=(f"收敛={repair['converged']}，迭代 {repair['iterations']}/{repair['max_iter']}，拟合 {repair['fit_s']:.3f} 秒，Tune FP={int(repair['metrics']['fp'])}、FN={int(repair['metrics']['fn'])}。" if repair else '原结构 SVM 已收敛，未触发额外求解器修复。')
    review_path=Path('reports/错误案例复核.md')
    case_review=review_path.read_text(encoding='utf-8').split('## 最终模型复核',1)[-1].strip() if review_path.exists() else '案例复核尚未完成。'
    chapters=[f'''# 题目六：中文垃圾邮件分类——课程报告章节

本文件由保存的审计、实验表和逐封预测自动生成。学校 Word 模板、成员姓名尚未提供，因此这是可套版的完整章节，不冒充学校模板终稿。全部主结果使用同一封存 Test；旧解析版本实验只作为修复记录，不用于结论。

## 摘要

本项目对原始 TREC06C ZIP 中的 {audit['rows']:,} 封邮件实施只读审计、编码与 MIME 恢复、内容分组、NumPy/pandas 探索分析和监督分类。垃圾邮件为 1，正常邮件为 0；不删除重复邮件、不改标签。最终按 Policy 预先选择的主模型为 **{plan['main']}**。封存 Test 共 {int(main.n):,} 封、{boot['groups']:,} 个内容组，研究工作点准确率 **{main.accuracy:.4%}**、Macro-F1 **{main.macro_f1:.4%}**，正常误拦 **{int(main.fp)} / {int(main.tn+main.fp):,}**，垃圾漏检 **{int(main.fn)} / {int(main.fn+main.tp):,}**。每封邮件都进入分母，没有拒判后剔除样本。

现代语义是独立候选，不预设一定胜出。最终选择依据、负结果、性能代价和分组不确定性均在下文给出；这些旧语料结果不等于现代真实邮箱部署效果。

**关键词：** 中文垃圾邮件；数据清洗；内容分组；TF-IDF；语义表示；概率校准。

## 1. 任务、课程目标与原始数据

原始文件是工作区内的 `题6_中文垃圾邮件_trec06c.zip`，SHA-256 为 `{audit['sha256']}`，字节数 {audit['zip_bytes']:,}。唯一清单是 `full/index`。正常邮件 {audit['labels']['ham']:,} 封，垃圾邮件 {audit['labels']['spam']:,} 封。{audit['delay_rows']:,} 条 delay 索引是同一批路径各两次，不当成新增样本；{audit['macosx']:,} 个 macOS 元数据条目排除在样本外。

课程主线顺序是：定义问题 → 收集与理解 → 清洗 → 探索 → 词/字特征与传统分类 → 评价解释 → 展示复现。NumPy 实现小样本 TF-IDF、布尔混淆计数与排序累积阈值扫描；pandas 完成聚合、交叉表、长宽表与一对一合并；sklearn 完成实际模型拟合与校准，现代语义分支在这些主线完成之后比较。

## 2. 数据审计与可靠解析

全部原始文件不同并不代表内容独立：整封字节哈希有 {audit['raw_unique']:,} 个，原始正文哈希只有 {audit['body_unique']:,} 个；{audit['body_duplicate_rows']:,} 封落在重复正文组，最大原始正文组 {audit['body_max']} 封。原始标签保留，重复样本仍计入邮件级评价，通过分组隔离依赖。

原始声明中，multipart 结构损坏 {audit['counts']['multipart_broken']:,} 封；Base64 声明 {audit['counts']['base64']:,} 封，其中严格语法不合法 {audit['counts']['base64_invalid']:,} 封。不能把这样的声明直接交给盲解码流程。实现以原始 bytes 为准，验证 Base64 语法、字符解码与质量后只解码一次；伪声明保留原字节，合法 multipart/alternative 只选一个等价正文，附件不执行。HTML 只提取可见内容，不执行脚本、不访问链接。

派生表对账 {parser['rows']:,} 行、解析崩溃 {parser['failed']}、空正文 {parser['empty_body']}、全空主题正文 {parser['empty_text']}。空正文邮件保留主题；真正全空输入使用 Train 先验。{parser['replacement_rows']:,} 封保留损坏替换标记，不能声称全部恢复。815 封畸形头块用显式空行恢复边界；移除 {parser['removed_header_lines']} 行指定传输/过滤头，指定残留模式扫描为 {parser['header_leak_rows']}。主模型只接受内容白名单，不读取路径、组、标签、解析状态或旧反垃圾评分。

开发回归与错例复核发现并修复三项缺陷：① 合法 UTF-8 中原有 U+FFFD/私用字被误作新解码错误；修复的全量影响扫描为 0 个原始模型文本变化。② RFC2047 编码片段与普通英文之间的空格丢失；原语料 35 封主题受影响，含 Train 23、Tune 2。③ 编码词内部不规范折行且缺少缩进，导致后续主题被丢弃；parser 1.3.0 严格验证完整 Base64/终止符/声明 charset 后恢复 603 封，含 Train 339、Tune 54。不能严格验证的片段仍保留，不猜测缺字。修复后重新解析、复核分组并完整重跑传统候选；旧结果归档，Test 预测当时尚未打开。

## 3. 内容分组与五用途数据协议

精确分组使用主题+正文去空白键，较长正文还单独精确分组；短正文/空白有单列规则。对正文中联系方式做保守归一化，以字符 5-gram 提出近重复候选：两轮 128 维 MinHash/LSH，候选阈值 0.80；实际 Jaccard≥0.90 且长度比≥0.80 才连边。最终有 {groups['groups']:,} 组、{groups['verified_edges']:,} 条已验证近重复边，最大组 {groups['largest']} 封，混合标签组 {groups['mixed_label_groups']}。标签只用于审计及分层，不用于构造相似度或拆开不喜欢的组。

{table(splits[['split','rows','groups','spam','ham','largest_group']])}

种子固定为 42，20 折分组分层后按 11/2/2/2/3 折分配。已阅读的两封开发样例所在组固定 Train。Train 拟合全部词表、IDF、缩放、频比及模型；Tune 选参数；Calibrate 拟合 sigmoid；Policy 选有限融合和阈值；Test 冻结后才预测。不重拟合 Train+Tune。

所有组、所有已验证近重复边均无跨集合。LSH 不保证找出全部语义模板；连通分量存在链式扩展，较大组抽样最低成对 Jaccard 约 0.771，不能把“每条边≥0.90”写成“组内任意两封≥0.90”。分组评估减少已识别模板泄漏，不能消除老语料采集偏差或未知近重复。

仅在 Train+Tune 上做的固定 B2 C=1 随机切分诊断中，验证集 {random['n']:,} 封，{random['validation_rows_with_seen_group']:,} 封的内容组在其训练侧出现，重叠组 {random['overlap_groups']:,} 个；Macro-F1 为 {random['macro_f1']:.4%}。它与正式分组 Tune 的验证组成不同，不能当成严格配对的因果增益，更不是 Test 成绩。

## 4. pandas EDA 与课程特征

所有形成特征选择的 EDA 只读取 Train。类别不平衡、长度长尾、重复组大小、含 URL 交叉表、长度分箱结构均保存为表。高频统计区分总词次数、文档频率和类内归一化频率；区分词展示两类文档比例差，模型权重与词频分别解释。完整词表可能含联系方式片段，仅保存在私有目录；公开图采用聚合中文词，不公开邮件正文。

实际 Train 质量与长度聚合如下。两类的平均正文长度都高于中位数，说明只报均值会掩盖长尾；全体 Train 正文字符数中位数为 {eda['length_quantiles']['0.5']:.0f}，P90 为 {eda['length_quantiles']['0.9']:.0f}，P99 为 {eda['length_quantiles']['0.99']:.0f}。较长邮件是否被语义模型截断需要另外按 tokenizer 统计，不能用字符数直接代替 tokens。

{table(quality[['class','n','mean_body','median_body','body_missing','replacement_rows','replacement_rate']],percent=['replacement_rate'])}

`crosstab` 的链接统计说明，URL 是有用线索，但不是垃圾邮件的充分条件：正常类也有链接，而相当一部分垃圾邮件没有链接。质量标志的类间差异还提示可能存在采集偏差，因此主模型不直接使用编码声明或解析缺陷判断类别。

{table(url_analysis,percent=['含链接比例'])}

![训练集长度](../artifacts/figures/02_lengths.png)

![训练集区分词](../artifacts/figures/06_discriminative_terms.png)

例如“公司”出现在 {company.ham_rate:.2%} 的正常邮件和 {company.spam_rate:.2%} 的垃圾邮件中。它有区分性，但正常邮件也会讨论公司，不能据此写一条“一见公司就判垃圾”的规则。这里使用的是各类内部文档比例，不是未经归一化的总词次数。

词级 unigram/bigram 使用 jieba，保留单字、英文、数字、否定词与标点；`min_df=2`、sublinear TF、L2 归一化。初始词/字符上限分别 100000/200000，主题独立 30000。字符 (2,4) 与 (1,3)、词字组合、主题权重 0.5/1/2、结构分块均分别比较。稀疏矩阵使用 CSR，不把全文特征稠密化。结构特征为内容派生白名单，log1p 后仅在 Train 标准化；NB 不接受其负值缩放分块。

## 5. 传统候选与受控优化

B1 是 MultinomialNB，B2 是 LogisticRegression，两者共享同一组词 TF-IDF，以保证分类器对比公平。强稀疏 S 逐步加入正文字符和独立主题分块；N 使用实际训练集 NB log-count ratio 后拟合线性 SVM。全部候选结果保留在 `sparse_tune_results.csv` 和补充表，正式选择排除未收敛候选。

{table(grid[['candidate','dimensions','macro_f1','fp','fn','fit_s']],percent=['macro_f1'])}

初始网格的收敛审计针对保存的各族最佳模型，表列如下；其他未保存候选没有逐项迭代记录，不能据此虚填“全部收敛”。正式选择仅在已审计通过的保存模型中进行，补充实验则逐项记录收敛状态。

{table(pd.DataFrame(read_json('convergence_audit.json')))}

补充实验用于检查特征截断、主题权重和收敛，不根据 Test 结果增加候选。完整配置总数 {read_json('sparse_selection.json')['configurations']}；约 30 的初始预算增加了必要的收敛修复及同分类器对照，扩展发生于 Test 前且全部披露。

{table(supplement[['candidate','converged','dimensions','macro_f1','fp','fn','fit_s']],percent=['macro_f1'])}

以下是实际满足 min_df 的候选特征数量与截断比例；它不是被删邮件比例：

{table(cap_table,percent=['truncated_fraction'])}

在同一个 LR C=1 对照中，校验解析相对盲信声明的 Macro-F1 差为 {(indexed.loc['B2_1_plain','macro_f1']-indexed.loc['naive_1_plain','macro_f1'])*100:.4f} 个百分点，FP 从 {int(indexed.loc['naive_1_plain','fp'])} 到 {int(indexed.loc['B2_1_plain','fp'])}，FN 从 {int(indexed.loc['naive_1_plain','fn'])} 到 {int(indexed.loc['B2_1_plain','fn'])}。这一结论针对这两个清洗流程的完整差异，不把所有改善只归因于一种 MIME 异常。

在同一个 SVM C=1 下，词+字相对正文词的 Macro-F1 差为 {(indexed.loc['word_char_1_plain','macro_f1']-indexed.loc['body_word_1_plain','macro_f1'])*100:.4f} 个百分点；再加入独立主题块的差为 {(indexed.loc['S_1_plain','macro_f1']-indexed.loc['word_char_1_plain','macro_f1'])*100:.4f} 个百分点。因此“增加特征必然提升”并不成立。特征上限、主题权重和结构的选择应以同条件表中的真实错误分布与成本为准。

在相同词/主题分块、C=1 和字符特征上限下，将字符范围从 2–4 gram 改为 1–3 gram，Tune FP 从 {int(indexed.loc['S_1_plain','fp'])} 到 {int(indexed.loc['S_char13_1_plain','fp'])}，FN 从 {int(indexed.loc['S_1_plain','fn'])} 到 {int(indexed.loc['S_char13_1_plain','fn'])}，Macro-F1 差为 {(indexed.loc['S_char13_1_plain','macro_f1']-indexed.loc['S_1_plain','macro_f1'])*100:.4f} 个百分点。它检验的是字符粒度及受限词表的联合表示变化，不能仅凭两封差异声称普遍显著。

结构 SVM 的本次收敛状态为 {sparse_registry['S_structure']['converged']}；达到迭代上限的候选只能作为诊断，排除出正式选择。另建纯文本 LR 与结构 LR 的相同分类器对照，避免同时更换分类器与特征后误归因。最终稀疏候选 {stability['model']} 的优化器种子 42/43 重复预测翻转 {stability['prediction_flips']} 封，最大分数差 {stability['max_score_delta']:.6g}，重复拟合收敛状态 {stability['repeated_converged']}；这不是跨划分交叉验证，也不保证 GPU 完全确定性。

有界求解修复保持原 C、L2 squared-hinge 目标、tol、类别权重及迭代上限，复用 Train 变换，只增加 dual=False 的 primal 候选。实测：{repair_summary}完整记录在 `solver_repair.json`。该补充在 Test 前登记；未收敛仍排除，没有根据 Test 数字追加搜索。

![开发消融](../artifacts/figures/12_development_ablation.png)

## 6. 现代语义候选与资源实测

E1 使用官方 Qwen3-Embedding-0.6B，本地冻结编码器，主题+正文、无标签指令、1024 tokens、最后一个非 padding token 池化与 L2 归一化，再由 Train LR 拟合。只比较 C=0.1/1/10。下载 revision/逐文件哈希和许可保存，推理不上传邮件，也不执行远端自定义代码。缓存绑定文本哈希、doc_id、解析版本、模型 revision、长度/指令/池化规则；同文本编码可复用，但每封的训练和评价权重不删除。

状态：{json.dumps(semstatus,ensure_ascii=False)}。

{table(semantic[['C','converged','iterations','fit_s','accuracy','macro_f1','fp','fn']],percent=['accuracy','macro_f1']) if len(semantic) else '没有完成 E1 分类实验；不得填入效果数字。'}

资源与 padding 记录：`{json.dumps(profile,ensure_ascii=False)}`。

这里 `peak_allocated_mib` 指 PyTorch 记录的已分配张量峰值，不包含 CUDA 驱动、其他进程和全部缓存预留；不能将它与系统显示的整块显卡占用直接等同。

E2 使用 hfl/chinese-macbert-base，本地监督微调，先 8 封 Train 过拟合冒烟验证，再从预训练权重重新开始正式训练。批量 8、梯度累积 2、最多 3 epochs、学习率 2e-5/3e-5、512 token 预算、主题最多 64 tokens；头尾截断与同权重仅头部推理在 Tune 比较。完整 epoch 曲线、峰值显存和耗时单独保存；没有把冒烟训练准确率当泛化成绩。未采用长文多窗口时不声称测过窗口收益。

训练流程冒烟与资源记录：`{json.dumps(read_json('macbert_profile.json'),ensure_ascii=False) if Path('artifacts/macbert_profile.json').exists() else '未完成'}`。其中 estimated 字段只是由初始小样本外推，含暖机影响；实际 epoch 耗时以下表为准。

保持有效 batch=16 的资源探测只用最长的 16 封 Train，3 次暖机后测 5 次更新；探测权重全部丢弃。微批量增大占用更多显存、收益有限，因此保留原 batch8/accum2 方案，不增加效果搜索预算：

{table(pd.DataFrame(read_json('macbert_batch_benchmark.json')['cases'])[['micro_batch','accumulation','fused','median_update_s','peak_allocated_mib','passed']]) if Path('artifacts/macbert_batch_benchmark.json').exists() else '未执行额外批量资源探测。'}

fused 的空值表示 PyTorch 默认后端，False/True 是显式设置；不同后端未被假定为逐位一致。微批量、累积次数及单次更新计时不是整轮训练时间，全部探测明细保留在独立 JSON。

{table(pd.read_csv('artifacts/macbert_learning_curves.csv')[['lr','epoch','training_loss','training_s','peak_allocated_mib','macro_f1','fp','fn']],percent=['macro_f1']) if Path('artifacts/macbert_learning_curves.csv').exists() else '尚无完成的 MacBERT epoch；不生成效果数字。'}

{'![MacBERT 学习曲线](../artifacts/figures/14_macbert_learning_curves.png)' if Path('artifacts/figures/14_macbert_learning_curves.png').exists() else ''}

实际长度审计：`{json.dumps(read_json('macbert_length_audit.json'),ensure_ascii=False) if Path('artifacts/macbert_length_audit.json').exists() else '未完成'}`。

同权重截断对照：`{json.dumps(read_json('macbert_truncation_tune.json'),ensure_ascii=False) if Path('artifacts/macbert_truncation_tune.json').exists() else '未完成'}`。

保存再加载验收：E1 每次保存后对全部 Tune 嵌入的 LR 分数作绝对误差 ≤1e-12 断言；E2 验收记录为 `{json.dumps(read_json('macbert_reload_validation.json'),ensure_ascii=False) if Path('artifacts/macbert_reload_validation.json').exists() else '未完成'}`。

设备中途暂时只显示 Intel 核显，CUDA 初始化实际报告无 GPU；独显恢复后重新确认 RTX 5070 Laptop GPU。CPU/GPU 小批次探测属于环境记录，不代表全量吞吐。E1/E2 的通用榜单与本项目结果无关；旧公开语料与预训练数据是否重叠不能完全核实。

## 7. 独立校准、有限融合与阈值

基础模型固定后在 Calibrate 拟合 sigmoid。Policy 比较最佳稀疏与 Tune 最佳语义的 alpha=[0,0.25,0.5,0.75,1]，含两个单模型端点，不再叠第三分支。研究点最大化 Macro-F1；低误拦点要求 Policy 经验 ham_FPR≤0.5% 或 1%，在可行阈值中最大化 spam Recall。

在读取 Policy 前预定：先筛出相对稀疏的 Macro-F1 增益至少 0.0005、且正常误拦率不增加的语义/融合候选，再按 Macro-F1、较低误拦率和较简单模型排序；没有合格替代者才保留稀疏。这个界限是决策规则，不是统计显著性检验。候选资格表如下，稀疏基准自身增益为 0，但始终保留为默认回退：

{table(pd.read_csv('artifacts/policy_candidate_eligibility.csv')[['model','gain_vs_sparse','ham_fpr','eligible']],percent=['ham_fpr'])}

Policy 选择主模型 **{plan['main']}**，权重和阈值如下：

```json
{json.dumps(plan['specifications'][plan['main']],ensure_ascii=False,indent=2)}
```

{('互补性计数（Policy，双方默认阈值 0.5）：`'+json.dumps(read_json('complementarity_policy.json'),ensure_ascii=False)+'`。') if Path('artifacts/complementarity_policy.json').exists() else '没有可用语义比较，不生成融合收益结论。'}

![Policy 阈值权衡](../artifacts/figures/10_policy_threshold.png)

## 8. 封存 Test 的全部邮件结果

冻结时间 `{plan['frozen_at']}`。模型/校准器、原 ZIP、派生表、划分、代码及配置身份写入 `evaluation_plan.json`；用户要求本轮不继续提交 Git，因此同时保存当前源码 SHA-256，不把旧 Git HEAD 冒充当前全部实现。`test_opened.json` 在访问前记录，重复调用正式评价被拒绝；报告只读取保存结果。

{('生成材料时修复了 Notebook/报告误读收敛列的展示错误，并将错误长度图按数值区间排序、移动遮挡条形的图例。仅三个展示模块发生冻结后的更正，旧/新文件哈希与原因记录于 `reporting_amendment.json`；冻结计划、数据、训练、推理、阈值、逐封预测均未修改，没有重新评估 Test。') if Path('artifacts/reporting_amendment.json').exists() else ''}

{table(research[research.model.isin(formal)][['model','n','accuracy','macro_f1','balanced_accuracy','ham_fpr','fp','fn']],percent=['accuracy','macro_f1','balanced_accuracy','ham_fpr'])}

逐类精确率与召回（不能只看 spam，也不能把误拦率当 precision）：

{table(research[research.model.isin(formal)][['model','spam_precision','spam_recall','spam_f1','ham_precision','ham_recall','ham_f1']],percent=['spam_precision','spam_recall','spam_f1','ham_precision','ham_recall','ham_f1'])}

排序与概率评价（AP、ROC-AUC 越高越好；Brier 越低越好）：

{table(research[research.model.isin(formal)][['model','average_precision','roc_auc','brier']])}

只使用 Train 多数类的 Dummy 对照在相同 Test 上准确率为 {read_json('dummy_test.json')['accuracy']:.4%}、Macro-F1 为 {read_json('dummy_test.json')['macro_f1']:.4%}；它不读取 Test 决定多数类。

主模型相对 B2 的研究点 Macro-F1 差为 {(main.macro_f1-baseline.macro_f1)*100:.4f} 个百分点。这是两个事先冻结工作点的比较，不能单独归因于表示学习；默认阈值结果和全部有限融合结果见机器表。

![主比较](../artifacts/figures/07_model_comparison.png)

![混淆矩阵](../artifacts/figures/08_confusion.png)

![PR 与校准](../artifacts/figures/09_pr_calibration.png)

同一主模型的不同冻结工作点：

{table(result.query("weighting=='per_mail'")[lambda x:x.model.eq(plan['main'])][['point','threshold','accuracy','macro_f1','ham_fpr','spam_recall','fp','fn']],percent=['accuracy','macro_f1','ham_fpr','spam_recall'])}

Policy 满足经验误拦约束不保证 Test 或新邮件满足。本报告照实展示 Test 的实际 FP、FN，没有为了追平业务目标再调阈值。spam Precision 与 ham FPR 分母不同；Average Precision 也不是梯形积分 PR-AUC。

## 9. 模板权重与不确定性

主指标按邮件计数；辅助敏感性分析让每个模板组总权重为 1，保留组内所有邮件。以下是研究点的组等权结果，不能与原始计数混在一起：

{table(result.query("point=='research' and weighting=='per_group'")[lambda x:x.model.isin(formal)][['model','accuracy','macro_f1','ham_fpr','spam_recall','fp','fn']],percent=['accuracy','macro_f1','ham_fpr','spam_recall'])}

以 {boot['groups']:,} 个 Test 组为抽样单位，配对 bootstrap {boot['repeats']} 次，每次对所有模型使用相同组重采样。类别消失重采样 {boot['missing_class_resamples']} 次。主模型 Macro-F1 95% percentile 区间 [{limits['macro_f1'][0]:.4%}, {limits['macro_f1'][1]:.4%}]，ham FPR 区间 [{limits['ham_fpr'][0]:.4%}, {limits['ham_fpr'][1]:.4%}]。它依赖分组质量与独立组假设；若事件为 0，普通 bootstrap 可给出退化零区间，不能据此宣称总体风险为零。

主模型减比较模型的 Macro-F1 差值与配对区间如下，单位是 **百分点**。区间依赖这次分组与数据抽样，不等于跨时间或真实邮箱保证；不能凭一个小数差异就宣称普遍显著。

{table(pd.DataFrame(paired_rows).assign(**{k:lambda frame,k=k:frame[k]*100 for k in ['main_minus_comparator','ci95_low','ci95_high']}))}

## 10. 配对鲁棒性与错误解释

冻结前还对固定种子抽取的 256 封 Tune 邮件，分别检查所有已完成分支，使用各自已在 Policy 保存的研究工作点。完整开发比较见 `development_model_robustness.csv`；它用于呈现各分支的包装/文本敏感性，不替代封存 Test，也不以开发集分数声称泛化。

规则在 Test 前确定，在 Tune 上验证。格式封装（HTML、合法 MIME/Base64）与文本变体（全角、零宽、每第 7 个中文字符后插空格、感叹/问号重复 3 次）分开解释。每一项都由同一批 Test 原邮件生成，标签不变，包含 ham 与 spam；适用比例与全量分母一起报告，变体不是新独立样本。

{table(robust[['rule','n','applied_rows','application_rate','original_macro_f1','perturbed_macro_f1','flip_rate']],percent=['original_macro_f1','perturbed_macro_f1','application_rate','flip_rate'])}

扰动前后实际错误计数：

{table(robust[['rule','fp_before','fp_after','fn_before','fn_after']])}

![配对扰动](../artifacts/figures/11_robustness.png)

错误表先保存全部邮件，再标记 FP/FN、长度分箱、解析质量、内容组和人工分析类别；类别不回写训练标签。按预定高置信 FP/FN、边界与固定种子随机规则选案例，完整正文仅在本地私有目录。公开汇总不挑选只对新模型有利的样例。

### 实际案例复核

{case_review}

线性解释计算真实激活特征 x_j*w_j，NB ratio 使用实际变换后的 X，所有特征和截距重构真实 decision score 后才展示。字符 n-gram 可能重叠，不是独立因果证据。语义解释只对两个事前固定 Train 示例删除首/尾 120 字，记录实际概率变化，是局部敏感性而非模型“真正理由”，不把稀疏解释冒充融合全解释。

![错误长度层](../artifacts/figures/13_error_lengths.png)

## 11. 性能、复现与交互演示

以下在固定种子选出的 32 封 Train 邮件上逐封重封装为 UTF-8 MIME，模型预热一次；端到端时间包括解析、特征变换、打分、校准和决策，初始加载另列。计时仅对本机当次实测负责；文件大小包含该分支运行所需模型/tokenizer/校准文件，不用一个小 LR 文件冒充完整语义模型大小。

{table(performance[['model','n','load_s','median_e2e_ms','p95_e2e_ms','mean_e2e_ms','local_artifact_bytes']])}

批量 Test 时间另见 `performance_test.csv`，包含模型加载/变换/校准、排除原始解析，口径不同不能互换。嵌入批量缓存耗时和 MacBERT epoch 耗时单独记录。

本地 Streamlit 页面支持主题正文、单封 .eml、解析质量、各分支概率、冻结阈值和真实线性贡献。Policy 滑块只重算开发集混淆矩阵；Test 页只读，不自动重新预测。不连接真实邮箱、不删除邮件、不执行上传 HTML/附件、不公开部署。四个 Notebook 从独立空 kernel 顺序执行，不依赖隐藏变量。

## 12. 总结、局限与课程责任

本项目的主要成果是可追溯的分析与评价闭环：全量保留 {audit['rows']:,} 封邮件的原始标签，以结构验证恢复文本，按内容组控制已识别模板泄漏；在相同划分上完成课程要求的两种传统分类方法及有限的词/字/主题、结构、语义对照。最终保留 **{plan['main']}**，独立 Test 准确率 {main.accuracy:.4%}、Macro-F1 {main.macro_f1:.4%}，实际错误为 {int(main.fp)} 封正常误拦和 {int(main.fn)} 封垃圾漏检。

优化并非越多越好。解析修复有逐条保真和完整重跑证据，特征及语义改动则必须接受真实负结果；模型、阈值和融合在 Test 前决定。课程层面的新意来自可靠解析、多粒度表示、词汇/语义受限融合、低误拦决策与配对验证的组合及问题诊断，不声称发明了 TF-IDF、Qwen、MacBERT 或新的基础网络。

损坏字符未全部恢复；编码回退可能仍有边界。高阈值内容分组可能漏掉语义近似，也可能链式连接不完全相似的邮件。旧采集环境与旧主题分布可能容易分离；严分组的高分不证明现代复杂真实邮箱泛化。预训练污染无法完全排除。合成扰动保留标签是可检查的局部假设，不涵盖社会工程语义改写、附件或图像垃圾邮件。

学校 Word 模板、三名成员姓名/最终责任和 GitHub 仓库未给出；此处交付完整可套版章节、可编辑 PPT 和五分钟讲稿素材，未声称已录制视频或完成成员个人提交。AI 参与实现和实验执行；团队须真实理解、复核、修改后以本人身份提交，不伪造作者、日期或贡献。详见 `功能分工与分批提交指南.md` 与 `contributions_and_ai_use.md`。

## 附录：代码、证据与成员填写位置

- 四个课程 Notebook 对应解析、EDA/NumPy、传统/语义比较、评价/策略，完整能力映射见 `course_alignment.md`。
- 审计与阶段验收见 `audit.md`、`stage_log.md`；全部实验参数及负结果见 `artifacts/*_results.csv`，正式逐封预测留在本地受保护目录。
- 软件测试、Notebook、图表与冻结身份的最终机器验收写入 `artifacts/final_validation.json`。不能将本报告的生成时间当作所有验收已经完成。
- 60 个功能责任单元详见 `功能分工与分批提交指南.md`。实际成员姓名、真实承担内容和对应提交链接由团队完成后填写，当前没有伪造个人承担项。
- 环境安装、直接演示和新目录复现步骤见根目录 `README.md`；正式 Test 不允许随意重开。

## 技术资料

- Python email：<https://docs.python.org/3/library/email.parser.html>
- sklearn 分组划分与泄漏：<https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html>；<https://scikit-learn.org/stable/common_pitfalls.html>
- Qwen 官方模型与实现：<https://huggingface.co/Qwen/Qwen3-Embedding-0.6B>；<https://github.com/QwenLM/Qwen3-Embedding>
- HFL MacBERT：<https://huggingface.co/hfl/chinese-macbert-base>
- sklearn 校准、阈值、AP：<https://scikit-learn.org/stable/modules/calibration.html>；<https://scikit-learn.org/stable/modules/classification_threshold.html>；<https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html>

课程事实以本地课程要求 HTML、W1 数字教材与用户任务书为依据；本报告全部项目数字来自本工作区真实实验，不引用外部模型成绩替代。
''']
    content='\n'.join(chapters); Path('reports').mkdir(exist_ok=True)
    Path('reports/results.md').write_text(content,encoding='utf-8')
    try:
        from markdown_it import MarkdownIt
        rendered=MarkdownIt('commonmark').enable('table').render(content)
    except ImportError:
        rendered='<pre>'+html.escape(content)+'</pre>'
    css='body{font-family:Microsoft YaHei,sans-serif;max-width:1120px;margin:40px auto;padding:0 24px;color:#142b3d;line-height:1.85;overflow-wrap:anywhere}h1{font-size:32px}h2{margin-top:48px;border-bottom:2px solid #0f766e;padding-bottom:8px}table{font-size:13px;border-collapse:collapse;display:block;overflow:auto}td,th{padding:8px;border:1px solid #cbd5e1}th{background:#e8f3f0}img{max-width:100%}pre{white-space:pre-wrap;background:#f1f5f9;padding:16px}code{font-size:.9em}a{color:#0f766e}@media print{body{margin:0}h2{break-after:avoid}table,img{break-inside:avoid}}'
    Path('reports/results.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>题目六课程报告</title><style>'+css+'</style><body>'+rendered+'</body></html>',encoding='utf-8')
    # Structured deck input, derived from the same exact evidence tables.
    deck=dict(audit=audit,groups=groups,parser=parser,splits=splits.to_dict('records'),main=plan['main'],main_metrics=main.to_dict(),
        formal=research[research.model.isin(formal)].to_dict('records'),policy=policy,bootstrap=boot,robustness=robust.to_dict('records'),
        keywords=pd.read_csv('artifacts/tables/train_discriminative_terms.csv').to_dict('records'),performance=performance.to_dict('records'))
    Path('artifacts/presentation_data.json').write_text(json.dumps(deck,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__': build_report()
