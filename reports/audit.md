# 附件审计复核

独立复算全部 64,620 封邮件，ZIP SHA-256 为 `57e7baf0009d516d4360f7dd5879f9a404c5ae87ae223ad1c7f600d902cdd509`，大小 118139295 字节。

附件计数全部得到复核：spam 42854 / ham 21766；缺失路径 0；全文哈希 64620；原始正文哈希 31881；正文重复组 4823、涉及 37562 封、额外行 32739、最大 827；元数据 64842；delay 129240 行，路径集合相同且各两次。

声明异常：multipart 非 multipart 11901；Base64 声明 7551、严格语法失败 7541、含非 ASCII 7516；原始正文空白 335；轻量 HTML 正则 273。声明类型不是实际内容。邮件头中的已有反垃圾结果必须排除。

可复现命令：`.venv\Scripts\python.exe -m spam_analysis.cli audit --config configs/data.yaml`。
原始正文定义是首个 CRLFCRLF 或 LFLF 之后的原始字节；正式解析后的分组统计是另一口径，不能混为一个数字。逐封审计在本地 `data/processed/audit_rows.parquet`，不纳入 Git。

未发现原附件的 audit_trec06c.py 或 audit_summary.json；本报告基于本工程独立审计，不声称执行过缺失脚本。
