# 预注册实施与评估计划

2026-10-13，在任何模型训练前记录。任务书全文及数据审计已读；当前工作区初始只有 ZIP 和课程 HTML，无现有源码、Git 历史或用户未提交修改。原始文件保持原位只读。

## 固定协议

- spam=1、ham=0，full/index 是唯一清单；不合并 delay、不删重复、不改标签。
- 解析函数只接收 bytes。主模型输入允许主题、正文及显式允许的正文结构统计；邮件头、路径、标签、组、解析缺陷和编码声明不作为分类输入。
- 精确主题+正文分组；长度至少 80 的正文精确分组；联系方式归一化的字符 5-gram，128 个 MinHash、LSH 候选阈值 0.80，实际 Jaccard≥0.90 且长度比例≥0.80 才连边。短正文与空白例外单列。种子 42。检查链式组，不宣称召回所有近重复。
- 固定 20 折 StratifiedGroupKFold，前 11/2/2/2/3 折分配 Train/Tune/Calibrate/Policy/Test。已阅读 data/000/000、001 所在组强制放 Train。比例偏差据实报告，不搜索种子。
- 基础模型**不重新拟合 Train+Tune**。所有词表、IDF、缩放、NB 频比及学习参数仅来自 Train；Tune 只选择候选。Calibrate 拟合 sigmoid；Policy 选择有限融合及阈值。Test 内容与预测不用于开发。
- B1 alpha=[0.1,0.5,1]；B2 C=[0.5,1,2,4]；其余顺序消融 word/char/subject/structure，最多约 30 个完整配置。默认保留停用词。Tune 按 Macro-F1，再更低 ham_FPR 排序。
- E1 官方 Qwen3-Embedding-0.6B，无指令先行，1024 tokens、last-token pooling、L2 归一化，LR C=[0.1,1,10]；先测开发小批次吞吐和 padding 一致性。E2 按可测资源决定，最多 3 epochs，512 tokens，头/头尾截取，学习率 2e-5/3e-5。环境失败记录实际错误和有限修复，不能写成完成。
- Policy 使用 alpha=[0,.25,.5,.75,1]，研究阈值最大化 Macro-F1；低误拦阈值约束经验 ham_FPR≤0.5%/1%，再最大化 recall。包括单分支端点；融合无强制保留。
- 在访问 Policy 前细化简洁性规则：相对稀疏模型的 Policy 研究点 Macro-F1 增益小于 0.0005，或 ham_FPR 上升，则保留稀疏模型。此界限是预设决策规则，不是统计显著性标准。
- 保存冻结评估文件（输入/划分/模型/代码哈希、阈值、全部比较清单）后才执行 Test。主指标逐封全量；额外每模板等权、1000 次配对组 bootstrap。不以拒判剔除分母。
- 扰动预先固定：NFKC 全角、零宽、HTML/MIME 等价封装、中文间稀疏空白、标点重复。格式不变性与文本鲁棒性分开。开发确认后冻结，对 Test 同封配对，不把变体当独立样本。
- 错例选择：高置信 FP/FN、最接近阈值、seed=42 随机错误。先汇总全体；正文只在本地私有表，公开报告不附原始邮件。

## Test 开封前的有界收敛修复补充

parser 1.3.0 重跑再次实测结构 SVM 达到 5000 次迭代上限（Tune 0 FP/3 FN 的数值不作为已收敛成果）。在读取 Calibrate/Policy/Test 前登记一个额外候选：复用同一 Train 拟合的变换，保持 C=1、L2、squared_hinge、tol、类别权重和迭代上限，只改为 `dual=False` 的 primal 求解。无论是否改善都保留结果；未收敛仍排除。最终稳定性复验 clone 完整估计器，仅改 seed，避免丢失求解器配置。包含原有同 LR 结构消融后，总候选上限由 32 增至 33，不继续追加求解器搜索。

## 资源与当前缺口

### 首次访问 Policy 前的选择逻辑细化

代码复核用合成反例发现：先挑最高 Macro-F1 再对它作安全否决，可能漏掉第二名中增益足够且不增加误拦的候选。在 `policy_results.csv`、`calibrated_registry.json`、`evaluation_plan.json` 均不存在时，细化为：先筛选相对稀疏的 Macro-F1 增益≥0.0005 且 ham_FPR 不上升的候选，再按 Macro-F1、较低 FPR、较简单模型排序；没有合格者才保留稀疏。门槛、alpha 集合、划分和阈值目标均不变。增加候选资格审计表和合成回归，未查看 Policy 或 Test 分数。

Windows，Python 3.10.5；16 逻辑处理器；RAM 63.59 GiB（检查时约 35.80 GiB 可用）；D 盘约 744 GiB 可用；RTX 5070 系列 8151 MiB 显存。原默认 torch 2.14.0+cpu、CUDA 不可用，不修改它；创建工作区 .venv。训练资源限本项目，先测再估，不承诺固定耗时。

课程要求已定位并核对；w1/W1_T1_数字教材.html 支持 NumPy/pandas/EDA/sklearn 主线及 PyTorch 自学拓展。缺少学校 Word 模板、三名成员姓名/实际责任、GitHub 目标仓库。可先制作通用章节和汇报素材，不声称完成学校套版、真实成员提交或录制视频。附件 audit_trec06c.py/audit_summary.json 未找到，使用独立脚本复核相同统计定义。
