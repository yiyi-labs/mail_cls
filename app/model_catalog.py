"""Presentation names only: frozen experiment IDs and model files stay intact."""
NAMES = {
    'B1': '词频基线 · TF-IDF + 朴素贝叶斯',
    'B2': '线性基线 · TF-IDF + 逻辑回归',
    'body_word': '词特征消融 · 正文词 TF-IDF + SVM',
    'char': '字特征消融 · 正文字 TF-IDF + SVM',
    'word_char': '双特征消融 · 正文词字 TF-IDF + SVM',
    'S': '三路词汇 · 正文词字 + 主题 SVM',
    'S_structure': '结构增强 · 三路词汇 + 结构 SVM',
    'S_char13': '多粒度词汇 · 词 + 字 1–3 元组 + 主题 SVM',
    'naive': '解析对照 · 简化解码 + 逻辑回归',
    'N': '词频比增强 · NB 对数频比 + SVM',
    'B2_cap200k': '大词表线性 · 20 万词 TF-IDF + 逻辑回归',
    'S_cap': '大词表词汇 · 扩展词字词表 + SVM',
    'S_subject05': '主题降权 · 主题权重 0.5 + SVM',
    'S_subject2': '主题增权 · 主题权重 2 + SVM',
    'S_structure_lr': '结构线性 · 三路词汇 + 结构逻辑回归',
    'S_structure_primal': '结构求解优化 · 结构 SVM 原始问题求解',
    'S_text_lr': '三路线性 · 正文词字 + 主题逻辑回归',
    'E1': 'Qwen 语义迁移 · 冻结向量 + 逻辑回归',
    'E2': 'MacBERT 邮件专训 · 主题保留 + 正文首尾',
}
METHODS = {
    'B1': '从 Train 学习词表与 TF-IDF，训练多项式朴素贝叶斯。',
    'B2': '保持词 TF-IDF 不变，改用逻辑回归，比较分类器的作用。',
    'S_char13': '正文词、字符 1–3 元组、主题独立表示，交给线性 SVM。',
    'E1': 'Qwen3-Embedding-0.6B 不更新参数；1024 维语义向量交给训练好的逻辑回归。',
    'E2': '以 hfl/chinese-macbert-base 为起点，使用邮件标签微调编码器与二分类头。',
}


SHORT_NAMES={'B1':'朴素贝叶斯','B2':'逻辑回归','S_char13':'多粒度 SVM','E1':'Qwen 语义迁移','E2':'MacBERT 专项微调'}

def short_name(model):
    if model in SHORT_NAMES:return SHORT_NAMES[model]
    if model.startswith('F_'):return f'融合 · MacBERT {float(model[2:]):.0%}'
    return model_name(model)

def model_name(model):
    if model in NAMES:
        return NAMES[model]
    if model.startswith('F_'):
        alpha = float(model[2:])
        return f'概率融合 · MacBERT {alpha:.0%} + 多粒度 SVM {1-alpha:.0%}'
    raise KeyError(f'Unmapped model: {model}')


def candidate_name(candidate):
    # Longest prefix first prevents S from swallowing S_char13 etc.
    for key in sorted(NAMES, key=len, reverse=True):
        if candidate == key or candidate.startswith(key+'_'):
            suffix = candidate[len(key):].lstrip('_')
            return model_name(key) + (f' / 参数 {suffix}' if suffix else '')
    return candidate
