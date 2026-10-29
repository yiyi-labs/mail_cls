# 邮件分类 · Python 数据分析

以邮件清洗、Pandas 统计、NumPy 数值计算、文本特征和分类比较为课程主线，提供本地网页、真实分词及模型计算回放。主模型由开发数据选择，网页只展示已完成评价并发布的版本。五类对照为朴素贝叶斯、逻辑回归、多粒度 SVM、Qwen 语义迁移和 MacBERT 微调。

## 快速入口

- **启动网页**：双击根目录 `启动演示.cmd`，访问 <http://127.0.0.1:8601/>。
- **当前状态**：五种方法已完成训练、评价与发布，直接启动网页即可。
- **模块与要点**：[分工.md](分工.md)，不预设人数或人员分配。
- **使用与方法**：[网页说明](reports/模型设计与网页使用说明.md)、[课程对应](reports/course_alignment.md)。
- **实验成果**：[结果报告](reports/results.md)、[HTML 报告](reports/results.html)、[课程课件](reports/Topic6_课程汇报.pptx)、[演示讲稿](reports/五分钟视频讲稿.md)。
- **学校格式报告**：[项目报告](reports/项目报告.docx)，已填写摘要及第 1–7 章；封面信息已填写，分工和附录留空。根目录保留原始 Word 模板。

## 项目结构

```text
mail_cls/
├─ 分工.md                   模块清单与功能要点
├─ 启动演示.cmd              网页启动入口
├─ app/                      网页、分词与真实计算可视化
├─ src/spam_analysis/         数据处理、分析、训练和评价核心
├─ configs/                  数据路径与固定实验配置
├─ data/
│  ├─ raw/                   官方原始 ZIP / TAR 邮件包，只读
│  └─ processed/             corpus.parquet 统一邮件表与来源记录
├─ artifacts/                模型、划分、预测、聚合结果与验证证据
│  ├─ figures/               图表、课件预览及当前网页截图
│  └─ runs/                  训练记录、已发布模型和逐次评价依据
├─ notebooks/                四个课程分析 Notebook
├─ reports/                  报告、课件、说明与实验记录
├─ scripts/                  验证、制图、复现和交付工具
├─ tests/                    与解析、算法和网页直接相关的回归测试
└─ examples/                 合成演示邮件
```

`.venv/` 为本地运行环境，`.cache/` 保存模型下载及运行缓存，`.streamlit/` 保存网页配置。它们是本地运行所需的辅助目录。首次创建独立复现工作区时会生成 `reproductions/`。

数据统一入口为 **`data/processed/corpus.parquet`，共 113,715 封**。包含 TREC06C 64,620 封、Apache SpamAssassin 5,913 封、AUEB Enron-Spam 43,182 封。新增邮件经过精确去重、跨来源重叠排除和模板分组，原 TREC06C 的内容与划分保持冻结。原始包位于 `data/raw/`，源表保留用于复现，网页统一浏览而不要求先选数据集。

来源与范围见 [数据说明](data/README.md)。这些主要是历史中英文邮件，现代中文邮箱效果仍待独立检验。网页通过 `artifacts/active_run.json` 读取已发布模型及对应成绩。

## 训练状态与恢复机制

本轮训练已经完成，累计 GPU 账本约 5.26 小时，未超出 8 小时预算。当前发布版本为 `course-20261016`，主模型 MacBERT 在 19,820 封固定留出邮件上的回顾准确率为 99.405%，SVM 为 99.324%。

训练代码具备完整断点恢复：MacBERT 约每 60 秒保存模型、AdamW、学习率、随机状态与批次位置，交替两份并校验哈希；Qwen 按 128 封保存编码块。正常中止应按 Ctrl+C 等待保存，关机期间不会运行。本轮全部完成后，已移除不再需要的优化器断点和编码分块，保留最终权重、合并向量、训练曲线及恢复验证记录；这些已清理的中途状态不能再用于续训。

五种方法的训练已完成。`scripts/continue_training.py` 保留发布、性能比较、网页检查和分析材料更新的阶段衔接。`course-20261015` 保留训练与首次发布记录，`course-20261016` 展示已训练的五种方法，累计 GPU 预算从原账本延续。独占锁防止重复启动，日志与阶段状态位于对应目录的 `pipeline.log`、`pipeline_status.json`。

仅在需要重新生成后续报告和检查时使用（已冻结的模型不会重新拟合）：

```powershell
& .venv/Scripts/python.exe scripts/publish_trained_models.py
```

本轮发布修订前已经观察旧模型测试结果，成绩因此属于固定留出集的回顾评估。候选由 Tune 确定，Calibrate 拟合校准器，Policy 选择阈值与主模型；Test 从未用于模型拟合或选型。首次发布规则要求每个来源的指标同时不降，拒绝了全部新模型；该过程完整保留，修订依据与局限记录在 `reports/next_stage.md`。

## 核心功能

| 内容 | 实现要点 |
|---|---|
| 数据清洗 | MIME、字符编码、主题、HTML 可见文本、异常记录与内容分组 |
| Python 数据分析 | Train 数据一对一合并、分组统计、分箱、类别词语出现率与图表 |
| 传统分类 | 词 TF-IDF + 朴素贝叶斯 / 逻辑回归、多粒度 SVM 及特征消融 |
| 语义分类 | 冻结 Qwen 向量 + 逻辑回归、MacBERT 监督微调，统一开发数据选择 |
| 真实可视化 | 实际分词编号和位置、TF-IDF 数值、线性贡献、神经模型中间张量 |
| 结果分析 | 相同邮件多方法对照、校准概率、冻结阈值、错误案例及训练档案 |

网页使用统一石墨色与雾紫色主题。结果区域保留稳定空间，提供分词、进度、热图和贡献条的过渡效果。动画回放实际测得的数据；播放速度不表示计算耗时，张量切片也不等于完整网络或因果解释。

阅读视图支持编码切换与损坏位置标记，不改变冻结分类输入。垃圾分数接近或等于 1 可能来自数值饱和，不代表判断一定正确；训练集高分不能证明泛化效果或不存在泄漏。

## 本地运行与验证

本项目使用独立 Python 3.10 环境；版本记录见 `requirements-lock.txt`。已有本地环境和模型时，直接运行：

```powershell
& .venv/Scripts/python.exe -m streamlit run app/app.py --server.address 127.0.0.1 --server.port 8601
```

可输入文字、上传 `.eml`、`.txt` 或 `.csv` 或浏览数据集邮件。`examples/course_notice.eml` 和 `examples/marketing_demo.eml` 是合成示例，不计入正式成绩。本地演示不连接真实邮箱，不执行邮件 HTML 或附件。

验证现有项目，不重新训练或开封 Test：

```powershell
& .venv/Scripts/python.exe -m pytest -q
& .venv/Scripts/python.exe scripts/validate_local_app.py
& .venv/Scripts/python.exe scripts/audit_web_isolation.py
& .venv/Scripts/python.exe scripts/validate_delivery.py
```

清理核验记录位于 `artifacts/cleanup_validation.json`；数据路径迁移记录位于 `artifacts/data_location_amendment.json`。路径迁移只改变原始 ZIP 的存放位置，正式冻结计划、数据内容、模型、阈值及预测保持不变。

## 从新目录复现

不要在已封存的当前目录覆盖训练和正式评价产物。创建独立工作区：

```powershell
& .venv/Scripts/python.exe scripts/create_reproduction_workspace.py --name rerun_01
Set-Location reproductions/rerun_01
```

辅助脚本复制源码、配置、说明、原始 ZIP 和模型身份，保留 `data/raw/` 布局，不复制旧模型、预测、成绩或 Git。按顺序完成：

1. 建立 Python 3.10 虚拟环境，安装 `requirements-lock.txt` 与项目的 `dev,demo,semantic` 依赖；根据设备准备匹配的 PyTorch。
2. 执行 `spam_analysis.cli` 的 `audit`、`prepare` 和 `split`；配置分别使用 `configs/data.yaml` 和 `configs/experiment.yaml`。
3. 完成 EDA、传统训练、预定消融与稳定性检查；恢复固定身份的语义资产并执行语义训练。
4. 执行独立校准及 Policy 策略选择，通过测试后冻结评价计划。
5. 仅在新工作区执行一次正式 Test 评价，再由保存分数生成 Notebook、图表与报告。

以上辅助脚本复现原 TREC06C 基线；扩充语料的流程位于 `src/spam_analysis/experiments/`，需要同时携带 `data/raw/` 的官方来源包与本轮配置，在独立目录重新建立实验。具体顺序和预算见 `reports/experiment_plan.md`、`reports/next_stage.md`。PPT 生成使用 `scripts/build_presentation.mjs`，依赖该脚本注明的本地 Node / artifact-tool 运行环境。

## 数据用途与公开范围

- Train 拟合词表、IDF、模型等；Tune 选择候选；Calibrate 拟合概率校准；Policy 选择阈值与融合；Test 只做冻结后的最终评价。
- 每封邮件一个 `doc_id`，重复内容按组隔离；全部测试样本计入分母，不以窗口或分组重复计数。
- 正式结果与完整错误分析见报告；旧语料成绩不保证现代真实邮箱效果。
- 原始邮件、逐封派生数据、完整词表、模型、嵌入、逐封预测和私有错误案例不进入公开 Git。
- `tests/` 保留有效回归测试；实验冻结与修复证据保留在 `artifacts/` 和报告中，用于核验结果来源。
- 学校格式报告的摘要及第 1–7 章已完成，封面信息已填写；真实成员贡献、附录、视频录制及发布记录按实际完成情况补充；AI 使用说明见 `reports/contributions_and_ai_use.md`。
