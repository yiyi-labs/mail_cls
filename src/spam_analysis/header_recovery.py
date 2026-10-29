"""Recover one strictly verifiable encoded Subject split without indentation."""
import base64
import binascii
import re
from .text_encoding import canonical_charset,text_penalty

START=re.compile(rb'^(Subject:[ \t]*)(=\?([^? \t]+)\?[bB]\?)([A-Za-z0-9+/=]+)$',re.I)
CONTINUE=re.compile(rb'^([A-Za-z0-9+/=]+)(\?=)?$')


def recover_subject_lines(lines):
    """Join only an incomplete B word followed by valid payload + terminator.

    Never cross another header, infer missing bytes, add padding or accept a
    word that cannot be decoded strictly in its declared charset.
    """
    result=[]; recovered=0; i=0
    while i<len(lines):
        match=START.fullmatch(lines[i])
        if match:
            payload=match[4]; end=i+1; terminated=False
            while end<len(lines) and len(payload)<8192:
                continuation=CONTINUE.fullmatch(lines[end].strip())
                if not continuation: break
                payload+=continuation[1];end+=1
                if continuation[2]: terminated=True;break
            if terminated:
                try:
                    charset=canonical_charset(match[3].decode('ascii'))
                    decoded=base64.b64decode(payload,validate=True).decode(charset,errors='strict')
                    if not decoded or text_penalty(decoded)>.02: raise ValueError('Uncertain decoded content')
                except (ValueError,UnicodeError,LookupError,binascii.Error): pass
                else:
                    result.append(match[1]+match[2]+payload+b'?=');recovered+=1;i=end;continue
        result.append(lines[i]);i+=1
    return result,recovered
