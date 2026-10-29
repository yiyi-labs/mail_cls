"""Charset evidence, replacement accounting, RFC2047 and Unicode normalization."""
import codecs
import re
import unicodedata
from email.header import decode_header

def canonical_charset(value):
    s = (value or "").strip(" \t\"'").lower()
    if s in {"gb2312", "gb_2312-80", "gbk", "cp936", "x-gbk"}:
        return "gb18030"
    try:
        return codecs.lookup(s).name
    except LookupError:
        return ""


def text_penalty(text):
    if not text:
        return 0.0
    controls = sum(unicodedata.category(c) in {"Cc", "Cs", "Co", "Cn"} and c not in "\n\r\t" for c in text)
    return (text.count("\ufffd") * 10 + controls * 4) / len(text)


def decode_bytes(raw, declared=None):
    if not raw:
        return "", "empty", "empty"
    declared = canonical_charset(declared)
    # A valid explicitly declared UTF-8 stream can intentionally contain U+FFFD
    # or private-use codepoints inherited from an earlier damaged source. They
    # are real encoded characters here, not decoding errors. Reinterpreting
    # their bytes as a legacy charset would corrupt a legal round trip.
    if declared in {"utf-8", "utf-8-sig"}:
        try:
            return raw.decode("utf-8-sig", errors="strict"), "utf-8-sig", "strict"
        except UnicodeError:
            pass
    # UTF-8 validity is strong evidence; declarations on this corpus are unreliable.
    encodings = list(dict.fromkeys(["utf-8-sig", declared, "gb18030", "big5", "cp1252"]))
    candidates = []
    for rank, enc in enumerate(encodings):
        if not enc:
            continue
        try:
            text = raw.decode(enc, errors="strict")
        except (UnicodeError, LookupError):
            continue
        # No label lexicon and no Chinese-ratio maximization.
        cost = text_penalty(text) + rank * 0.0001
        if enc == "cp1252" and declared != "cp1252":
            cost += 0.25
        candidates.append((cost, rank, text, enc, "strict"))
    # Historical messages can contain a few damaged bytes among valid GBK.
    # A universal single-byte codec must not win merely because it never fails.
    # Compare replacement/controls penalties; retain and report every replacement.
    for rank, enc in enumerate(encodings[:-1]):
        if not enc or enc in {"ascii", "iso8859-1", "cp1252"}:
            continue
        try:
            text = raw.decode(enc, errors="replace")
        except LookupError:
            continue
        if "\ufffd" in text:
            candidates.append((text_penalty(text)+0.015+rank*0.0001,rank,text,enc,"replacement_fallback"))
    if candidates:
        _, _, text, enc, method = min(candidates)
        return text, enc, method
    enc = declared or "gb18030"
    return raw.decode(enc, errors="replace"), enc, "replacement_fallback"


def _decode_header_fragments(header):
    if not header:
        return ""
    out = []
    try:
        fragments = decode_header(header)
    except (ValueError, TypeError):
        fragments = [(header, None)]
    for value, charset in fragments:
        if isinstance(value, bytes):
            out.append(decode_bytes(value, charset)[0])
        else:
            try:
                recovered = value.encode("ascii", "surrogateescape")
                out.append(decode_bytes(recovered, charset)[0])
            except (UnicodeError, LookupError):
                out.append(value)
    return "".join(out)


def subject_text(header):
    """Decode encoded words without dropping encoded/plain boundary spaces.

    RFC2047 ignores whitespace between adjacent encoded words, but preserves
    whitespace between an encoded word and ordinary text. decode_header alone
    strips the latter after a folded header, so simply joining its outputs is
    insufficient. Each word is decoded with the same byte-recovery policy.
    """
    if not header: return ""
    parts=re.split(r"(=\?[^?]+\?[bBqQ]\?[^?]*\?=)",str(header))
    if len(parts)==1: return _decode_header_fragments(header)
    output=[]
    for i,part in enumerate(parts):
        if i%2:
            output.append(_decode_header_fragments(part))
        elif part.isspace() and i>0 and i<len(parts)-1:
            continue
        else:
            output.append(_decode_header_fragments(re.sub(r"\r?\n[ \t]+"," ",part)))
    return "".join(output)


def normalize(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\u200b-\u200d\ufeff\u2060]", "", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[^\S\n]+", " ", text)
    return re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", text).strip()

