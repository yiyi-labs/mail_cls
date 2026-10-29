"""Observe the actual frozen inference, without changing its computations."""
from threading import RLock
from time import perf_counter
from contextlib import ExitStack
import numpy as np
from scipy.special import expit
from spam_analysis.explain import linear_contributions
from spam_analysis.macbert import batches
from neural_capture import NeuralCapture
from linear_capture import linear_steps

LOCK = RLock()


class ObservedCalibration:
    def __init__(self, delegate, trace):
        self.delegate, self.trace = delegate, trace

    def predict_proba(self, raw):
        self.trace['raw_score'] = float(raw[0,0])
        self.trace['calibration_coefficient'] = float(self.delegate.coef_[0,0])
        self.trace['calibration_intercept'] = float(self.delegate.intercept_[0])
        logit = float(raw[0,0]*self.delegate.coef_[0,0]+self.delegate.intercept_[0])
        self.trace['calibrated_logit'] = logit
        self.trace['stable_ham_tail'] = float(expit(-logit))
        return self.delegate.predict_proba(raw)


class ObservedClassifier:
    def __init__(self, delegate, trace):
        self.delegate, self.trace = delegate, trace

    def decision_function(self, vectors):
        self.trace['vector_dimensions'] = int(vectors.shape[1])
        self.trace['vector_norm'] = float(np.linalg.norm(vectors[0]))
        self.trace['vector_preview'] = vectors[0,:24].astype(float).tolist()
        return self.delegate.decision_function(vectors)


def classify_with_trace(engine, record, name, point='research', on_event=None):
    if len(record)!=1 or 'label' in record:
        raise ValueError('演示推理只接受一封不带标签的邮件')
    traces = {}; restored = {}
    started = perf_counter()
    with LOCK, ExitStack() as captures:
        try:
            for branch,weight in engine.specifications[name]['weights'].items():
                if not weight:
                    continue
                trace = traces[branch] = {'weight':float(weight)}
                if not record.lexical_text.iloc[0]:
                    trace['prior_only'] = True
                    continue
                engine.load(branch)
                if on_event:on_event({'branch':branch,'phase':'loaded','title':'模型已加载，开始本次计算'})
                base,calibrator = restored[branch] = engine.loaded[branch]
                if branch=='E1':
                    encoder,classifier = base
                    base = (encoder,ObservedClassifier(classifier,trace))
                    token_ids = encoder.tokenizer(record.semantic_text.iloc[0],truncation=True,
                                                   max_length=encoder.config['max_length'])['input_ids']
                    trace['tokens'] = encoder.tokenizer.convert_ids_to_tokens(token_ids[:36])
                    trace['token_count'] = len(token_ids)
                    trace['representation'] = '最多 1024 tokens → 最后有效位置池化 → L2 归一化'
                    captures.enter_context(NeuralCapture(encoder.model,branch,encoder.tokenizer.convert_ids_to_tokens(token_ids),trace,on_event))
                elif branch=='E2':
                    model,tokenizer = base
                    entry = engine.entries[branch]
                    encoded = batches(record,tokenizer,entry['config'],entry['checkpoint']['mode'])[0]
                    trace['tokens'] = tokenizer.convert_ids_to_tokens(encoded['input_ids'][:36])
                    trace['token_count'] = len(encoded['input_ids'])
                    trace['representation'] = '主题最多 64 tokens；正文首尾拼接；总长最多 512'
                    captures.enter_context(NeuralCapture(model,branch,tokenizer.convert_ids_to_tokens(encoded['input_ids']),trace,on_event))
                else:
                    table,detail = linear_contributions(base,record)
                    features = base.features.transform(record)
                    trace.update(vector_dimensions=int(features.shape[1]),active_features=int(features.nnz),
                                 contributions=table.to_dict('records'),intercept=detail['intercept'],
                                 reconstructed_score=detail['all_features_sum_plus_intercept'],
                                 representation='TF-IDF 稀疏特征 → 训练好的线性判别函数')
                    trace['linear_capture'] = linear_steps(base,record)
                    if on_event:on_event({'branch':branch,'phase':'features','title':f'已计算 {features.nnz:,} 项实际特征乘积'})
                engine.loaded[branch] = (base,ObservedCalibration(calibrator,trace))
            scores,branches = engine.scores(record,name)
            for branch,values in branches.items():
                traces[branch]['score_spam'] = float(values[0])
        finally:
            engine.loaded.update(restored)
    probability = float(scores.score_spam.iloc[0])
    threshold = float(engine.specifications[name]['thresholds'][point])
    row = record.iloc[0]
    return {'model':name,'point':point,'probability':probability,'threshold':threshold,
            'prediction':int(probability>=threshold),'branches':traces,
            'elapsed_seconds':perf_counter()-started,
            'subject_chars':len(row.subject_clean),'body_chars':len(row.body_clean),
            'parser_status':str(row.parser_status),'fallback_reason':str(row.fallback_reason),
            'removed_header_lines':int(row.removed_header_lines),'charset':str(row.used_charset)}
