"""Charset-aware, loss-accounted mail reading; never changes inference inputs."""
from email import policy
from email.parser import BytesParser
from email.header import decode_header
import re
import unicodedata
from spam_analysis.text_encoding import canonical_charset,decode_bytes
from spam_analysis.mime_transfer import transfer_decode
from spam_analysis.html_text import clean_visible
from spam_analysis.header_recovery import recover_subject_lines

ENCODINGS={'自动':'','GB18030 / GBK':'gb18030','UTF-8':'utf-8','Big5':'big5','Windows-1252':'cp1252'}
SEPARATOR=re.compile(rb'\r?\n\r?\n')


def decode_lossless(raw,declared='',override=''):
    """Invalid bytes remain visible as \\xNN instead of being discarded or invented."""
    if not raw:return '',{'encoding':'empty','invalid_bytes':0,'source_replacements':0,'nontext_characters':0}
    codec=override or decode_bytes(raw,declared)[1]
    if codec=='empty':codec='utf-8'
    escaped=raw.decode(codec,errors='surrogateescape')
    invalid=sum(0xDC80<=ord(c)<=0xDCFF for c in escaped)
    literals=escaped.count('\ufffd');nontext=0;out=[]
    for char in escaped:
        code=ord(char)
        if 0xDC80<=code<=0xDCFF:out.append(f'⟦字节 {code-0xDC00:02X}⟧')
        elif char=='\ufffd':out.append('⟦原文替换符 U+FFFD⟧')
        elif unicodedata.category(char) in {'Cs','Co','Cn','Cc'} and char not in '\r\n\t':
            out.append(f'⟦U+{code:04X}⟧');nontext+=1
        else:out.append(char)
    return ''.join(out),{'encoding':codec,'invalid_bytes':invalid,'source_replacements':literals,'nontext_characters':nontext}


def _header_fragments(value,declared='',override=''):
    try:fragments=decode_header(value)
    except (ValueError,TypeError):fragments=[(value,None)]
    text=[];events=[]
    for fragment,charset in fragments:
        if isinstance(fragment,str):
            try:fragment=fragment.encode('ascii','surrogateescape')
            except UnicodeError:
                text.append(fragment.replace('\ufffd','⟦原文替换符 U+FFFD⟧'));continue
        decoded,event=decode_lossless(fragment,charset or declared,override)
        text.append(decoded);events.append(event)
    return re.sub(r'\r?\n[ \t]+',' ',''.join(text)),events


def display_header(value,declared='',override=''):
    # Adjacent encoded words discard separating whitespace; encoded/plain
    # boundaries must preserve it. decode_header alone strips folded spaces.
    parts=re.split(r'(=\?[^?]+\?[bBqQ]\?[^?]*\?=)',str(value))
    if len(parts)==1:return _header_fragments(value,declared,override)
    output=[];events=[]
    for i,part in enumerate(parts):
        if not i%2 and part.isspace() and 0<i<len(parts)-1:continue
        decoded,items=_header_fragments(re.sub(r'\r?\n[ \t]+',' ',part),declared,override)
        output.append(decoded);events.extend(items)
    return ''.join(output),events


def readable_mail(raw,encoding='自动'):
    if encoding not in ENCODINGS:raise ValueError('Unsupported display encoding')
    override=ENCODINGS[encoding]
    split=SEPARATOR.search(raw);head=raw[:split.start()] if split else raw
    lines,recovered=recover_subject_lines(head.splitlines())
    # Recover the same malformed header block as the frozen parser, before
    # applying display-only byte accounting. Never replace the source bytes.
    valid=[];prior=False
    for line in lines:
        if re.match(rb'^[!-9;-~]+:',line):valid.append(line);prior=True
        elif prior and line[:1] in (b' ',b'\t'):valid.append(line)
        else:prior=False
    wire=b'\n'.join(valid)+b'\n\n'+raw[split.end():] if split else raw
    message=BytesParser(policy=policy.compat32).parsebytes(wire)
    match=re.search(rb'=\?([^?\s]+)\?[bBqQ]\?',head)
    hint=message.get_content_charset() or (match.group(1).decode('ascii','ignore') if match else '')
    headers=[];events=[];recoveries=['subject_continuation'] if recovered else []
    for name,value in message.raw_items():
        decoded,parts=display_header(value,hint,override)
        headers.append((name,decoded));events.extend(parts)

    def body(part,top=False):
        if part.get_content_disposition()=='attachment':return ''
        if part.is_multipart():
            children=[p for p in part.get_payload() if p.get_content_disposition()!='attachment']
            if part.get_content_subtype()=='alternative':
                children=sorted(children,key=lambda p:p.get_content_type()!='text/plain')
                for child in children:
                    result=body(child)
                    if result.strip():return result
                return ''
            return '\n\n'.join(filter(None,(body(child) for child in children)))
        if not part.get_content_type().startswith(('text/','multipart/')):return ''
        payload=part._payload
        if top and split:payload=raw[split.end():]
        elif isinstance(payload,str):payload=payload.encode('ascii','surrogateescape')
        else:payload=b''
        declared=part.get_content_charset() or hint
        decoded_bytes=transfer_decode(payload,part.get('Content-Transfer-Encoding',''),declared,recoveries)
        decoded,event=decode_lossless(decoded_bytes,declared,override);events.append(event)
        return clean_visible(decoded)[0].strip()

    text=body(message,True)
    subject=next((v for k,v in headers if k.lower()=='subject'),'')
    source,source_event=decode_lossless(raw,hint,override)
    return {'headers':headers,'subject':subject,'body':text,'source':source,
            'encodings':sorted({e['encoding'] for e in events if e['encoding']!='empty'}),
            'invalid_bytes':sum(e['invalid_bytes'] for e in events),
            'source_replacements':sum(e['source_replacements'] for e in events),
            'nontext_characters':sum(e['nontext_characters'] for e in events),
            'wire_invalid_bytes':source_event['invalid_bytes'],'recovery':sorted(set(recoveries)),
            'display_only':True}
