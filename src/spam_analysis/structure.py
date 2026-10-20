"""Explicit content-derived structural features; zero-denominator convention."""
import re

def structure(subject, body, html_count=0, link_count=0):
    text = subject + "\n" + body; n = max(len(text.strip()), 1)
    return dict(subject_chars=len(subject), body_chars=len(body), line_count=len(body.splitlines()),
                chinese_ratio=len(re.findall(r"[\u4e00-\u9fff]", text))/n,
                digit_ratio=sum(c.isdigit() for c in text)/n,
                latin_ratio=len(re.findall(r"[A-Za-z]", text))/n,
                url_count=len(re.findall(r"https?://|www\.", text, re.I)),
                email_address_count=len(re.findall(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text)),
                exclamation_count=text.count("!"), question_count=text.count("?"),
                repeated_punctuation_ratio=sum(len(m.group()) for m in re.finditer(r"([!?.,])\1+", text))/n,
                actual_html_tag_count=html_count, actual_link_count=link_count,
                subject_missing=int(not subject), body_missing=int(not body))

