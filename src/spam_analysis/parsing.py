"""Label-blind, byte-first MIME recovery. Never executes HTML or attachments."""
import base64
import binascii
import codecs
import hashlib
import quopri
import re
import unicodedata
from email import policy
from email.header import decode_header
from email.parser import BytesParser
from html.parser import HTMLParser

PARSER_VERSION = "1.3.0"
SEPARATOR = re.compile(rb"\r?\n\r?\n")
HEADER_LINE = re.compile(rb"^[!-9;-~]+:")
LEAKY_LINE = re.compile(r"(?im)^\s*(?:Received|Message-ID|Return-Path|X-Proofpoint-[\w-]+|X-IMSS-[\w-]+|X-Eyou-SpamValue|X-Spam-[\w-]+):")
HTML_TAG = re.compile(r"</?(?:html|body|div|p|br|table|tr|td|a|span|script|style|h[1-6]|font|img|ul|li)\b[^>]*>", re.I)

def raw_body(raw):
    match = SEPARATOR.search(raw)
    return raw[match.end():] if match else b""










def _parts(message, fallback, reasons, charsets):
    if message.is_multipart():
        parts = [p for p in message.iter_parts() if p.get_content_disposition() != "attachment"]
        if message.get_content_subtype() == "alternative":
            # One equivalent representation; prefer nonempty plain text.
            ordered = sorted(parts, key=lambda p: p.get_content_type() != "text/plain")
            for p in ordered:
                result = _parts(p, b"", reasons, charsets)
                if result.strip() and text_penalty(result) <= 0.02:
                    return result
            return ""
        return "\n\n".join(filter(None, (_parts(p, b"", reasons, charsets) for p in parts)))
    ct = message.get_content_type()
    if not ct.startswith(("text/", "multipart/")):
        return ""
    if ct.startswith("multipart/"):
        reasons.append("broken_multipart_raw_body")
    # BytesParser stores byte-surrogate text in _payload. get_payload(False)
    # may already charset-decode 8bit data with replacement, losing the bytes
    # needed to validate a false CTE declaration. Keep this access localized.
    payload = message._payload
    if fallback:
        raw = fallback
    elif isinstance(payload, str):
        raw = payload.encode("ascii", "surrogateescape")
    else:
        raw = b""
    charset = message.get_content_charset()
    raw = transfer_decode(raw, message.get("Content-Transfer-Encoding", ""), charset, reasons)
    text, used, method = decode_bytes(raw, charset); charsets.append(used)
    if method == "replacement_fallback":
        reasons.append(method)
    return text


def parse_email(raw):
    if not isinstance(raw, bytes):
        raise TypeError("parse_email expects bytes only")
    reasons = []; charsets = []
    original = BytesParser(policy=policy.default).parsebytes(raw)
    defects = sorted({type(d).__name__ for part in original.walk() for d in part.defects})
    # Reconstruct headers before the explicit blank separator. A malformed line
    # must not cause later Received/filter headers to become the model body.
    sep = SEPARATOR.search(raw)
    body = raw[sep.end():] if sep else b""
    message = original
    recovered_header_lines=[]
    if sep:
        from .header_recovery import recover_subject_lines
        recovered_header_lines,recovered_subjects=recover_subject_lines(raw[:sep.start()].splitlines())
        if recovered_subjects:
            message=BytesParser(policy=policy.default).parsebytes(b'\n'.join(recovered_header_lines)+b'\n\n'+body)
            reasons.append('unindented_encoded_subject_recovered')
    if "MissingHeaderBodySeparatorDefect" in defects and sep:
        header_lines = []; prior = False
        for line in recovered_header_lines:
            if HEADER_LINE.match(line):
                header_lines.append(line); prior = True
            elif prior and line[:1] in {b" ", b"\t"}:
                header_lines.append(line)
            else:
                prior = False
        message = BytesParser(policy=policy.default).parsebytes(b"\n".join(header_lines) + b"\n\n" + body)
        reasons.append("malformed_header_block_recovered")
    elif not sep:
        # No explicit separator: use standard parser payload, then strip known
        # transport lines; never append the entire raw message as body.
        reasons.append("no_explicit_separator")
    header = next((v for k,v in message.raw_items() if k.lower() == "subject"), "")
    subject = subject_text(header)
    decoded = _parts(message, body, reasons, charsets)
    clean, html_count, link_count = clean_visible(decoded)
    clean, removed = remove_header_leaks(clean)
    if removed:
        reasons.append("transport_header_lines_removed")
    subject_clean = normalize(subject); body_clean = normalize(clean)
    text = (subject_clean + "\n" + body_clean).strip()
    return dict(subject_decoded=subject, body_decoded=decoded, subject_clean=subject_clean,
                body_clean=body_clean, lexical_text=text, semantic_text=text,
                parser_version=PARSER_VERSION, parser_status="recovered" if reasons or defects else "ok",
                parser_defects=";".join(defects), fallback_reason=";".join(sorted(set(reasons))),
                declared_charset=str(original.get_content_charset() or ""), used_charset=";".join(sorted(set(charsets))),
                declared_cte=str(original.get("Content-Transfer-Encoding", "")),
                declared_content_type=original.get_content_type(), replacement_ratio=decoded.count("\ufffd")/max(len(decoded),1),
                removed_header_lines=removed, **structure(subject_clean, body_clean, html_count, link_count))

def naive_parse(raw):
    """Pre-registered ablation: trust the standard parser's MIME/CTE declarations."""
    msg = BytesParser(policy=policy.default).parsebytes(raw)
    subject = str(msg.get("Subject", "")); texts = []
    for part in msg.walk():
        if part.get_content_maintype() != "text" or part.get_content_disposition() == "attachment":
            continue
        value = part.get_payload(decode=True) or b""
        texts.append(decode_bytes(value, part.get_content_charset())[0])
    body = clean_visible("\n".join(texts))[0]
    body, _ = remove_header_leaks(body)
    return normalize(subject + "\n" + body)

from .text_encoding import canonical_charset, text_penalty, decode_bytes, subject_text, normalize

from .html_text import VisibleHTML, clean_visible, remove_header_leaks

from .mime_transfer import transfer_decode

from .structure import structure
