import json
import sys
from pathlib import Path
from email.message import EmailMessage
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'app'))
from mail_display import readable_mail,decode_lossless
from score_display import score_text
from computation_view import computation_html
from inference_trace import classify_with_trace
from spam_analysis.inference import LocalSession,record_from_text


@pytest.mark.parametrize('charset',['utf-8','gb18030','big5'])
@pytest.mark.parametrize('cte',['base64','quoted-printable','8bit'])
def test_readable_charsets_and_transfers(charset,cte):
    msg=EmailMessage();msg['Subject']='中文主題測試'
    msg.set_content('中文郵件正文\n第二行',charset=charset,cte=cte)
    display=readable_mail(msg.as_bytes())
    assert display['subject']=='中文主題測試'
    assert '中文郵件正文' in display['body'] and '第二行' in display['body']
    assert display['invalid_bytes']==0


def test_damage_is_accounted_not_silently_invented():
    text,event=decode_lossless(b'abc\xff\xfe',override='utf-8')
    assert event['invalid_bytes']==2 and '字节 FF' in text and '字节 FE' in text
    text,event=decode_lossless('原文�'.encode(),declared='utf-8')
    assert event['source_replacements']==1 and '原文替换符' in text


def test_broken_subject_continuation_display():
    import base64
    word=base64.b64encode('中文主题恢复'.encode('gb18030'))
    raw=b'Subject: =?gb2312?B?'+word[:8]+b'\n'+word[8:]+b'?=\nContent-Type: text/plain; charset=gb2312\n\n'+'正文'.encode('gb18030')
    display=readable_mail(raw)
    assert display['subject']=='中文主题恢复' and display['body']=='正文'


def test_header_keeps_encoded_plain_boundary_spaces():
    from mail_display import display_header
    assert display_header('=?UTF-8?B?5Lit5paH?=\r\n plain')[0]=='中文 plain'
    assert display_header('=?UTF-8?B?5Lit5paH?=\r\n =?UTF-8?B?5Lit5paH?=')[0]=='中文中文'


def test_html_alternative_and_attachment():
    msg=EmailMessage();msg.set_content('正常正文')
    msg.add_alternative('<script>danger()</script><p>另一正文</p>',subtype='html')
    msg.add_attachment(b'PRIVATE ATTACHMENT',maintype='text',subtype='plain',filename='secret.txt')
    text=readable_mail(msg.as_bytes())['body']
    assert '正常正文' in text and 'PRIVATE' not in text and 'danger' not in text


def test_scores_do_not_round_to_false_certainty():
    assert score_text(1-1e-12).startswith('1 − ')
    assert '饱和' in score_text(1.) and '饱和' in score_text(0.)
    assert 'e-' in score_text(1e-20)


@pytest.mark.parametrize('branch',['B1','B2','S_char13'])
def test_full_feature_sum_includes_omitted_features(branch):
    engine=LocalSession();mail=record_from_text('中文课程通知','请在星期四提交课程项目与代码。产品信息会议安排。')
    expected,_=engine.scores(mail,branch)
    trace=classify_with_trace(engine,mail,branch)
    details=trace['branches'][branch];capture=details['linear_capture']
    assert trace['probability']==float(expected.score_spam.iloc[0])
    assert capture['reconstructed_score']==pytest.approx(details['raw_score'],abs=2e-5)
    assert capture['intercept']+sum(x['product'] for x in capture['features'])+capture['remaining_contribution']==pytest.approx(details['raw_score'],abs=2e-5)


def test_untrusted_tokens_cannot_escape_visualization():
    trace=classify_with_trace(LocalSession(),record_from_text('',''),'E2')
    trace['branches']['E2']['tokens']=['</script><img src=x onerror=alert(1)>']
    html=computation_html(trace)
    assert '</script><img' not in html and '\\u003c/script' in html


def test_neural_hook_matches_real_tensor_and_is_removed_on_error():
    import torch
    from transformers import BertConfig,BertForSequenceClassification
    from neural_capture import NeuralCapture
    model=BertForSequenceClassification(BertConfig(vocab_size=30,hidden_size=16,num_hidden_layers=2,num_attention_heads=2,intermediate_size=24)).eval()
    tokens=torch.tensor([[1,2,3,4]])
    with torch.inference_mode():expected=model(input_ids=tokens,output_hidden_states=True)
    trace={}
    with NeuralCapture(model,'E2',['a','b','c','d'],trace):
        with torch.inference_mode():observed=model(input_ids=tokens)
    torch.testing.assert_close(expected.logits,observed.logits,rtol=0,atol=0)
    frames=trace['neural_capture']['frames']
    assert len(frames)==7
    np.testing.assert_allclose(frames[-1]['values'],expected.hidden_states[-1][0].numpy(),rtol=0,atol=0)
    assert trace['class_logits']==observed.logits[0].tolist()
    assert trace['model_input_ids']==tokens[0].tolist()
    with pytest.raises(RuntimeError):
        with NeuralCapture(model,'E2',['a','b','c','d'],{}):raise RuntimeError('abort')
    assert not any(module._forward_hooks for module in model.modules())
    assert not any(module._forward_pre_hooks for module in model.modules())
