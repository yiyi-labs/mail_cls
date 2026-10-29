import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'app'))
from text_analysis import local_tokenizer,neural_tokenization,sparse_tokenization,text_payload
from text_view import text_html
from data_analysis import training_frame,grouped_distribution
from spam_analysis.inference import LocalSession,record_from_text
from spam_analysis.macbert import batches


@pytest.mark.parametrize('branch',['E2','E1'])
def test_token_ids_match_frozen_preprocessing_and_unicode_offsets(branch):
    engine=LocalSession();entry=engine.entries[branch]
    identity=entry['checkpoint'] if branch=='E2' else entry['identity']
    tokenizer=local_tokenizer(branch,identity['folder'],json.dumps(identity['files'],sort_keys=True))
    record=record_from_text('主题：中文测试 A '+('课程'*70),'正文🙂ABC 联系会议 '+('学习邮件内容。'*350)+'末尾保留标记')
    result=neural_tokenization(branch,tokenizer,record,entry)
    expected=batches(record,tokenizer,entry['config'],entry['checkpoint']['mode'])[0]['input_ids'] if branch=='E2' else tokenizer(record.semantic_text.iloc[0],truncation=True,max_length=entry['config']['max_length'])['input_ids']
    assert result['input_ids']==expected
    assert result['omitted']>0 and result['total']<=result['limit']
    for token in result['tokens']:
        if token['special']:continue
        source=record.iloc[0][{'主题':'subject_clean','正文':'body_clean','语义文本':'semantic_text'}[token['field']]]
        assert source[token['start']:token['end']]==token['surface']
    if branch=='E2' and entry['checkpoint']['mode']=='head_tail':
        assert max(t['end'] for t in result['tokens'] if t['field']=='正文')==len(record.body_clean.iloc[0])


def test_sparse_views_use_actual_saved_vector_and_count():
    engine=LocalSession();engine.load('S_char13');artifact=engine.loaded['S_char13'][0]
    record=record_from_text('课程通知','课程 课程 通知，请提交作业。')
    blocks=sparse_tokenization(artifact,record)
    for shown,(name,field,vec,weight) in zip(blocks,artifact.features.transforms_):
        actual=vec.transform(record[field]).getrow(0)
        assert shown['active']==actual.nnz
        for feature in shown['features']:
            assert feature['tfidf']==pytest.approx(float(actual[0,feature['id']]*weight),rel=1e-6)
            assert feature['count']==vec.build_analyzer()(record[field].iloc[0]).count(feature['term'])
            assert feature['tf']*feature['idf']/feature['norm']*feature['block_weight']==pytest.approx(feature['tfidf'],rel=2e-6)
    assert any(b['algorithm']=='字符 n-gram' for b in blocks)


def test_empty_mail_bypasses_tokenizer_and_mismatch_is_rejected():
    engine=LocalSession()
    assert not text_payload(engine,record_from_text('',''),'E2')['branches']
    assert not engine.loaded
    with pytest.raises(ValueError,match='实际输入不同'):
        text_payload(engine,record_from_text('主题','正文'),'E2',{'branches':{'E2':{'model_input_ids':[0]}}})


def test_text_payload_cannot_execute_mail_html():
    html=text_html({'branches':[],'state':'</script><img src=x onerror=alert(1)>'})
    assert '</script><img' not in html and '\\u003c/script' in html


def test_training_join_excludes_test_and_distribution_accounts_for_every_row():
    records=pd.DataFrame({'doc_id':['a','b','c','d','test'],'label':[0,0,1,1,1],
                          'body_chars':[0,1,20,999,10000000],'digit_ratio':[0,.1,.5,1,.9]})
    manifest=pd.DataFrame({'doc_id':records.doc_id,'split':['train']*4+['test']})
    train=training_frame(records,manifest)
    assert set(train.doc_id)=={'a','b','c','d'}
    for field in ['body_chars','digit_ratio']:
        table=grouped_distribution(train,field)
        assert table['邮件数'].sum()==4
        np.testing.assert_allclose(table.groupby('类别')['类内比例'].sum(),1)
    with pytest.raises(ValueError,match='完整'):
        training_frame(records.iloc[:2],manifest)
