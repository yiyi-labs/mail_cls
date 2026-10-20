"""Denominator-preserving joins, error slices and probability reliability."""
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss


def prediction_frame(records, predictions, threshold):
    """Every requested prediction must have exactly one source row."""
    if not records.doc_id.is_unique or not predictions.doc_id.is_unique:
        raise ValueError('邮件编号必须唯一')
    if set(records.doc_id)!=set(predictions.doc_id):
        raise ValueError('预测与对应邮件的评价分母不一致')
    frame = predictions.merge(records, on='doc_id', how='left', validate='one_to_one', indicator=True)
    if not frame['_merge'].eq('both').all():
        raise ValueError('预测缺少对应邮件')
    frame = frame.drop(columns='_merge')
    p = frame.score_spam.to_numpy(float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('概率必须在 0 到 1 之间')
    if not frame.label.isin([0, 1]).all():
        raise ValueError('标签必须为 0 或 1')
    frame['prediction'] = (p >= threshold).astype(int)
    frame['wrong'] = frame.prediction.ne(frame.label)
    frame['false_positive'] = frame.label.eq(0) & frame.prediction.eq(1)
    frame['false_negative'] = frame.label.eq(1) & frame.prediction.eq(0)
    return frame


def reliability(frame, bins=10):
    """Mean spam probability versus observed spam fraction, including 0 and 1."""
    if len(frame) == 0:
        raise ValueError('没有可统计的预测')
    p = frame.score_spam.to_numpy(float)
    y = frame.label.to_numpy(int)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('概率范围无效')
    index = np.minimum((p * bins).astype(int), bins - 1)
    rows = []
    for i in range(bins):
        selected = index == i
        n = int(selected.sum())
        rows.append(dict(bin=i, lower=i/bins, upper=(i+1)/bins, count=n,
                         mean_probability=float(p[selected].mean()) if n else np.nan,
                         spam_fraction=float(y[selected].mean()) if n else np.nan))
    table = pd.DataFrame(rows)
    ece = float((table['count'] * (table.mean_probability-table.spam_fraction).abs()).sum()/len(frame))
    return table, dict(brier=float(brier_score_loss(y, p)),
                       log_loss=float(log_loss(y, p, labels=[0, 1])), ece=ece, n=len(frame))


def length_edges(training):
    """Learn display boundaries from training data, never evaluated samples."""
    inner = np.unique(np.quantile(training.body_chars, [.25, .5, .75]))
    return np.r_[-np.inf, inner, np.inf]


def assign_slices(frame, field, edges=None):
    if field=='source':
        from spam_analysis.experiments.corpus import SOURCE_NAMES
        return frame.source_id.map(SOURCE_NAMES)
    if field=='language':return frame.script_profile
    if field == 'length':
        if edges is None:
            raise ValueError('长度分组需要训练集边界')
        return pd.cut(frame.body_chars, bins=edges, include_lowest=True).astype(str)
    if field == 'quality':
        return pd.Series(np.where(frame.replacement_ratio.gt(0), '含损坏字符',
                         np.where(frame.body_missing.eq(1), '正文为空', '可读正文')), index=frame.index)
    if field == 'links':
        return pd.cut(frame.url_count, [-np.inf, 0, 2, np.inf], labels=['无链接', '1–2 个链接', '3 个及以上']).astype(str)
    if field == 'subject':
        return pd.Series(np.where(frame.subject_chars.eq(0), '主题缺失', '有主题'), index=frame.index)
    if field == 'truncation':
        return pd.cut(frame.truncation_ratio, [-np.inf, 0, .25, .5, np.inf],
                      labels=['完整保留', '截去 ≤25%', '截去 25–50%', '截去 >50%']).astype(str)
    raise ValueError(field)


def slice_errors(frame, groups):
    joined = frame.assign(group=groups)
    table = joined.groupby('group', observed=True, sort=True).agg(
        count=('doc_id', 'size'), errors=('wrong', 'sum'),
        false_positive=('false_positive', 'sum'), false_negative=('false_negative', 'sum'),
        spam=('label', 'sum')).reset_index()
    table['ham'] = table['count']-table.spam
    table['error_rate'] = table.errors/table['count']
    table['ham_fpr'] = table.false_positive/table.ham.replace(0, np.nan)
    table['spam_miss_rate'] = table.false_negative/table.spam.replace(0, np.nan)
    return table


def paired_bootstrap(frame, other_predictions, threshold, repeats=1000, seed=42):
    """Paired group bootstrap for accuracy difference, not independent mail draws."""
    if not other_predictions.doc_id.is_unique or set(frame.doc_id) != set(other_predictions.doc_id):
        raise ValueError('比较必须覆盖同一组邮件')
    other = other_predictions.set_index('doc_id').loc[frame.doc_id].score_spam.to_numpy()
    delta = frame.prediction.eq(frame.label).to_numpy(int) - ((other >= threshold) == frame.label.to_numpy()).astype(int)
    grouped = pd.DataFrame({'group': frame.template_group_id.to_numpy(), 'delta': delta}).groupby('group').delta.agg(['sum', 'size'])
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repeats):
        chosen = rng.integers(len(grouped), size=len(grouped))
        draws.append(float(grouped['sum'].to_numpy()[chosen].sum()/grouped['size'].to_numpy()[chosen].sum()))
    return {'difference': float(delta.mean()), 'low': float(np.quantile(draws, .025)),
            'high': float(np.quantile(draws, .975)), 'groups': len(grouped), 'repeats': repeats}
