# Python 数据分析课程对应

项目围绕邮件表的处理与分析展开，语义模型用于补充比较，不能替代 Pandas、NumPy 与 scikit-learn 的课程主线。

| 课程能力 | 项目实践 | 实现位置 |
|---|---|---|
| 文件与字符串处理 | 原始邮件、编码、MIME、HTML 可见文字、异常记录 | `io.py`、`parsing.py`、编码与 MIME 模块 |
| Pandas 合并与清洗 | 统一三个可靠来源，唯一编号、缺失值、标签依据、内容分组 | `experiments/corpus.py` |
| Pandas 分组与交叉表 | 来源 × 类别、文字类型 × 类别、长度与质量统计 | `analysis/distributions.py`、`app/data_analysis.py` |
| NumPy 数组统计 | 分位数边界、直方图、类内比例、布尔混淆计数 | `analysis/tables.py`、`metrics.py` |
| 文本特征 | Jieba、词与字符 n-gram、TF-IDF、稀疏矩阵 | `features.py`、`teaching.py` |
| 监督分类比较 | NB、LR、SVM；Qwen 向量加 LR；MacBERT 微调 | `sparse_models.py`、`experiments/train.py` |
| 数据驱动诊断 | 不同来源、文字类型、长度、编码质量与截断比例上的错误率 | `analysis/tables.py`、`app/evidence_views.py` |
| 概率分析 | 可靠性曲线、Brier、ECE，解释高置信度误判 | `analysis/tables.py` |
| 可视化与交互 | 统计图、词元与特征联动、真实张量回放、改写重算 | `app/` |
| 可复现 | 固定来源与划分、测试隔离、断点恢复、成绩绑定模型版本 | `experiments/`、`tests/` |

Notebook 与页面复用分析函数，避免重复实现；四个 Notebook 分别负责解析质量、探索与特征、模型比较、评价与错误分析。训练与正式评价由独立命令控制，Notebook 不暗中重新训练或调测试阈值。

讲解重点是三个问题：哪些数据特点影响判断？哪些方法在哪些邮件上出错？增加准确率付出了多少成本？不把大参数量或动画本身当作贡献，也不把数据量增加当作泛化已经提高。

来源限制：公开语料主要形成于 1999–2006 年，现代中文邮箱效果仍待独立真实数据验证。当前新增标签复核为 AI 辅助抽样，不冒充人工黄金标注。具体收益只引用已完成的冻结评价结果。

学校模板格式报告已完成摘要及第 1–7 章，见 `项目报告.docx`。封面已填写用户提供的信息，团队仍需填写成员姓名与真实责任，并实际完成附录、最终视频录制及后续仓库推送；这些内容没有代填为已完成。
