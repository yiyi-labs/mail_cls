"""Bounded local file inputs. Labels and file names never become model features."""
from io import StringIO
from pathlib import Path
import pandas as pd
from spam_analysis.inference import record_from_bytes, record_from_text
from spam_analysis.text_encoding import decode_bytes

MAX_BYTES = 5*1024*1024
MAX_ROWS = 100


def read_upload(name, content):
    if len(content) > MAX_BYTES:
        raise ValueError('单个文件不能超过 5 MiB')
    suffix = Path(name).suffix.lower()
    if suffix == '.eml':
        return record_from_bytes(content)
    text, _, _ = decode_bytes(content)
    if suffix == '.txt':
        return record_from_text('', text)
    if suffix != '.csv':
        raise ValueError('支持 EML、TXT 或 CSV 文件')
    table = pd.read_csv(StringIO(text), keep_default_na=False, dtype=str, nrows=MAX_ROWS+1)
    table = table.rename(columns={'主题':'subject','正文':'body'})
    if not table.columns.is_unique:
        raise ValueError('CSV 列名重复')
    if 'body' not in table:
        raise ValueError('CSV 需要 body（正文）列；subject（主题）列可选')
    if not 0 < len(table) <= MAX_ROWS:
        raise ValueError('CSV 需要包含 1～100 封邮件')
    frames = [record_from_text(str(row.get('subject','')),str(row['body'])) for row in table.to_dict('records')]
    records = pd.concat(frames, ignore_index=True)
    records['doc_id'] = [f'{doc}_{i}' for i,doc in enumerate(records.doc_id)]
    return records


def export_csv(frame):
    """Prevent spreadsheet formula execution in downloaded user-supplied fields."""
    result = frame.copy()
    for column in result.select_dtypes(include=['object','string']):
        result[column] = result[column].map(lambda x: "'"+x if isinstance(x,str) and x.lstrip().startswith(('=','+','-','@')) else x)
    return result.to_csv(index=False).encode('utf-8-sig')


def replace_span(record, span, replacement):
    """Validate Python Unicode offsets against the exact displayed source."""
    fields = {'主题':'subject_clean','正文':'body_clean','subject_clean':'subject_clean',
              'body_clean':'body_clean','语义文本':'semantic_text','lexical_text':'lexical_text'}
    field = fields.get(span.get('field'))
    if field is None:
        raise ValueError('该词元没有可编辑的原文位置')
    row = record.iloc[0];text = str(row[field])
    start,end = span.get('start'),span.get('end')
    if not isinstance(start,int) or not isinstance(end,int) or not 0 <= start < end <= len(text):
        raise ValueError('片段位置无效')
    if text[start:end] != span.get('surface'):
        raise ValueError('原文已经变化，请重新选择片段')
    subject,body = str(row.subject_clean),str(row.body_clean)
    if field in {'lexical_text','semantic_text'}:
        if text == body:
            field='body_clean'
        elif text == subject:
            field='subject_clean'
        elif text == subject+'\n'+body:
            if end <= len(subject):
                field='subject_clean'
            elif start >= len(subject)+1:
                field='body_clean';start-=len(subject)+1;end-=len(subject)+1
            else:
                raise ValueError('跨主题和正文的片段请在下方分别编辑')
        else:
            raise ValueError('该片段无法映射到单一输入字段，请手动编辑')
    if field=='subject_clean':subject=subject[:start]+replacement+subject[end:]
    else:body=body[:start]+replacement+body[end:]
    return subject,body
