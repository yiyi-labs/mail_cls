import base64
from email.message import EmailMessage
import inspect
import pytest
from spam_analysis.parsing import parse_email, normalize, structure

@pytest.mark.parametrize("charset,text", [("gb2312","中文课程通知"),("gbk","扩展字镕"),("utf-8","中文与 English"),("big5","繁體中文測試"),("ascii","English notice")])
def test_charsets(charset,text):
    raw=f'Content-Type: text/plain; charset="{charset}"\n\n'.encode()+text.encode(charset)
    result=parse_email(raw)
    assert result["body_clean"]==text

def test_pseudo_base64_regression():
    raw=b'Content-Type: text/plain; charset="gb2312"\nContent-Transfer-Encoding: base64\n\n'+"讲的是孔子后人的故事。".encode("gbk")
    r=parse_email(raw)
    assert r["body_clean"]=="讲的是孔子后人的故事。"
    assert "invalid_base64_kept_raw" in r["fallback_reason"]

def test_real_base64_and_encoded_subject():
    text="课程安排与答疑"; encoded=base64.b64encode(text.encode())
    raw=b'Subject: =?utf-8?B?'+encoded+b'?=\nContent-Type: text/plain; charset=utf-8\nContent-Transfer-Encoding: base64\n\n'+encoded
    r=parse_email(raw)
    assert r["subject_clean"]==r["body_clean"]==text

def test_broken_multipart_regression():
    r=parse_email(b'Content-Type: multipart/related; boundary="absent"\n\n'+"非财务经理的财务管理".encode("gbk"))
    assert r["body_clean"]=="非财务经理的财务管理"
    assert "broken_multipart_raw_body" in r["fallback_reason"]

def test_alternative_no_double_count():
    m=EmailMessage(); m.set_content("课程说明"); m.add_alternative('<p>课程说明</p><script>secret()</script>',subtype="html")
    r=parse_email(m.as_bytes()); assert r["body_clean"]=="课程说明"

def test_html_visible_link_no_scripts():
    r=parse_email(b'Content-Type: text/html; charset=utf-8\n\n'+b'<p>Hello <a href="https://invalid.example">visible</a></p><style>hidden</style><script>fetch()</script>')
    assert r["body_clean"]=="Hello visible" and r["actual_link_count"]==1

def test_malformed_headers_no_leak():
    r=parse_email(b'Subject: notice\nmalformed line\nReceived: PRIVATE_SERVER\nX-IMSS-Scores: 99\n\nbody')
    assert r["body_clean"]=="body" and "PRIVATE_SERVER" not in r["lexical_text"] and "99" not in r["lexical_text"]

def test_empty_and_normalization():
    assert parse_email(b'Subject: notice\n\n')["lexical_text"]=="notice"
    assert parse_email(b"")["body_clean"]==""
    assert normalize("ＡＢＣ\u200b　！")=="ABC !"
    assert all(v==0 for k,v in structure("","").items() if k.endswith("ratio"))
    assert list(inspect.signature(parse_email).parameters)==["raw"]

def test_quoted_printable_requires_evidence():
    r=parse_email(b'Content-Transfer-Encoding: quoted-printable\n\nx = y')
    assert r["body_clean"]=="x = y"
    r=parse_email(b'Content-Transfer-Encoding: quoted-printable\nContent-Type: text/plain; charset=utf-8\n\nhello=20world')
    assert r["body_clean"]=="hello world"

def test_sparse_byte_damage_not_western_mojibake():
    text="中文课程说明与教学安排。"*80
    raw=text.encode("gbk")+b'\x81'
    r=parse_email(b'Content-Type: multipart/related; boundary=absent\n\n'+raw)
    assert "中文课程说明" in r["body_clean"]
    assert r["used_charset"]=="gb18030" and r["replacement_ratio"]>0

def test_actual_development_examples():
    from pathlib import Path
    from zipfile import ZipFile
    from spam_analysis.io import read_config
    archive=Path(read_config('configs/data.yaml')['zip_path'])
    if not archive.exists(): pytest.skip("Local private corpus unavailable")
    with ZipFile(archive) as z:
        a=parse_email(z.read('trec06c/data/000/000'))
        b=parse_email(z.read('trec06c/data/000/001'))
    assert a['subject_clean']=='非财务经理的财务管理-(沙盘模拟)'
    assert '每一位管理' in a['body_clean'] and 'gb18030' in a['used_charset']
    assert '讲的是孔子后人的故事' in b['body_clean']


def test_valid_utf8_roundtrip_keeps_existing_damage_markers():
    body='合法 UTF-8 保留旧损坏标记：\ufffd 与私用字 \ue495。'
    m=EmailMessage(); m.set_content(body,charset='utf-8',cte='base64')
    assert parse_email(m.as_bytes())['body_clean']==normalize(body)
    m=EmailMessage(); m.set_content('<p>'+body+'</p>',subtype='html',charset='utf-8',cte='quoted-printable')
    assert parse_email(m.as_bytes())['body_clean']==normalize(body)


def test_folded_encoded_subject_preserves_plain_boundary_space():
    subject='法兰克福办公用品博览会Paper World'
    m=EmailMessage(); m['Subject']=subject; m.set_content('正文')
    assert parse_email(m.as_bytes())['subject_clean']==subject
    from spam_analysis.text_encoding import subject_text
    assert subject_text('=?utf-8?B?5Lit?= \n =?utf-8?B?5paH?=')=='中文'
