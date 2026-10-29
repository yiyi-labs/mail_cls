"""Validate content-transfer encoding before changing the original bytes."""
import base64
import binascii
import quopri
import re
import unicodedata
from .text_encoding import decode_bytes,text_penalty,canonical_charset

def transfer_decode(raw, cte, charset, reasons):
    cte = (cte or "").strip().lower()
    if cte == "base64":
        compact = re.sub(rb"\s+", b"", raw)
        try:
            if not compact or len(compact) % 4:
                raise ValueError("empty or invalid padding")
            decoded = base64.b64decode(compact, validate=True)
            if canonical_charset(charset) in {"utf-8","utf-8-sig"}:
                try:
                    utf8=decoded.decode("utf-8-sig",errors="strict")
                    controls=sum(unicodedata.category(c)=="Cc" and c not in "\r\n\t" for c in utf8)
                    if utf8 and controls/max(len(utf8),1)<=.02:
                        reasons.append("base64_validated_once")
                        return decoded
                except UnicodeError:
                    pass
            text, _, method = decode_bytes(decoded, charset)
            if method != "strict" or text_penalty(text) > 0.02:
                raise ValueError("decoded text quality")
            reasons.append("base64_validated_once")
            return decoded
        except (ValueError, binascii.Error):
            reasons.append("invalid_base64_kept_raw")
    elif cte == "quoted-printable":
        if re.search(rb"=[0-9A-Fa-f]{2}|=\r?\n", raw):
            reasons.append("quoted_printable_evidence")
            return quopri.decodestring(raw)
        reasons.append("quoted_printable_without_evidence_kept_raw")
    return raw

