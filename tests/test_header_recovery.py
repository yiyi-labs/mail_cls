import base64
from spam_analysis.header_recovery import recover_subject_lines
from spam_analysis.text_encoding import subject_text


def test_valid_nonindented_encoded_subject_is_recovered():
    text='网络和信息安全技术研究以及相关软件系统'
    data=base64.b64encode(text.encode('gb18030'))
    original=[b'Subject: =?gb2312?B?'+data[:20],data[20:]+b'?=',b'Date: today']
    lines,n=recover_subject_lines(original)
    assert n==1 and lines[1]==b'Date: today'
    assert subject_text(lines[0].split(b': ',1)[1].decode())==text


def test_never_consume_other_header_or_invent_missing_terminator():
    for lines in [[b'Subject: =?utf-8?B?aGVs',b'Date: today',b'bG8=?='],
                  [b'Subject: =?utf-8?B?aGVs',b'bG8=']]:
        assert recover_subject_lines(lines)==(lines,0)


def test_invalid_base64_charset_and_controls_remain_unchanged():
    for lines in [[b'Subject: =?gb2312?B?a',b'bc?='],
                  [b'Subject: =?unknown?B?aGVs',b'bG8=?='],
                  [b'Subject: =?utf-8?B?AA',b'AA?=']]:
        assert recover_subject_lines(lines)==(lines,0)


def test_parser_repairs_subject_without_consuming_body_or_headers():
    from spam_analysis.parsing import parse_email
    text='这是带有换行的邮件主题'
    data=base64.b64encode(text.encode('gb18030'))
    raw=b'Subject: =?gb2312?B?'+data[:20]+b'\n'+data[20:]+b'?=\nContent-Type: text/plain; charset=utf-8\n\n'+ '正文完全保留'.encode()
    result=parse_email(raw)
    assert result['subject_clean']==text and result['body_clean']=='正文完全保留'
    assert 'unindented_encoded_subject_recovered' in result['fallback_reason']
