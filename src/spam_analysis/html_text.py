"""Offline visible HTML and explicit transport/filter-header removal."""
import re
from html.parser import HTMLParser
from .text_encoding import normalize
HTML_TAG = re.compile(r"</?(?:html|body|div|p|br|table|tr|td|a|span|script|style|h[1-6]|font|img|ul|li)\b[^>]*>", re.I)
LEAKY_LINE = re.compile(r"(?im)^\s*(?:Received|Message-ID|Return-Path|X-Proofpoint-[\w-]+|X-IMSS-[\w-]+|X-Eyou-SpamValue|X-Spam-[\w-]+):")

class VisibleHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out = []; self.hidden = 0; self.tags = 0; self.links = 0
    def handle_starttag(self, tag, attrs):
        self.tags += 1
        if tag in {"script", "style", "head"}:
            self.hidden += 1
        if tag == "a":
            self.links += 1
        if not self.hidden and tag in {"p", "div", "br", "tr", "li", "h1", "h2"}:
            self.out.append("\n")
    def handle_endtag(self, tag):
        if tag in {"script", "style", "head"}:
            self.hidden = max(0, self.hidden - 1)
        if not self.hidden and tag in {"p", "div", "tr", "li"}:
            self.out.append("\n")
    def handle_data(self, data):
        if not self.hidden:
            self.out.append(data)


def clean_visible(text):
    if HTML_TAG.search(text):
        p = VisibleHTML(); p.feed(text); p.close()
        return normalize("".join(p.out)), p.tags, p.links
    return normalize(text), 0, 0


def remove_header_leaks(text):
    """Strip only recognized transport/filter header lines plus their continuations.

    This also conservatively removes quoted original transport headers. It does
    not strip arbitrary English colon lines or ordinary email addresses in prose.
    """
    lines = text.splitlines(); kept = []; removed = 0; dropping = False
    for line in lines:
        if LEAKY_LINE.match(line):
            dropping = True; removed += 1
        elif dropping and line[:1] in {" ", "\t"}:
            removed += 1
        else:
            dropping = False; kept.append(line)
    return "\n".join(kept), removed

