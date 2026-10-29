"""Read-only, indexed access to original mail; never feed labels to inference."""
from pathlib import Path
from zipfile import ZipFile
import hashlib
import re
import pandas as pd
from spam_analysis.io import read_config, read_index, sha256
from spam_analysis.inference import record_from_bytes


def load_catalog():
    config = read_config('configs/data.yaml')
    if sha256(config['zip_path']) != config['expected_sha256']:
        raise ValueError('原始 ZIP 校验失败，停止展示。')
    columns = ['doc_id','subject_clean','label','relative_path','raw_sha256']
    rows = pd.read_parquet('data/processed/records.parquet', columns=columns)
    manifest = pd.read_parquet('artifacts/splits/split_manifest.parquet',
                              columns=['doc_id','split','template_group_id'])
    with ZipFile(config['zip_path']) as archive:
        index = pd.DataFrame(read_index(archive))
    checked = rows.merge(index[['doc_id','label','relative_path']], on='doc_id',
                         suffixes=('','_index'), validate='one_to_one')
    if len(checked) != config['expected_rows'] or not checked.label.equals(checked.label_index):
        raise ValueError('索引标签不一致')
    if not checked.relative_path.equals(checked.relative_path_index):
        raise ValueError('索引路径不一致')
    return rows.merge(manifest,on='doc_id',validate='one_to_one')


def search_catalog(catalog, split='train', label='全部', query=''):
    result = catalog if split == '全部' else catalog[catalog.split.eq(split)]
    if label != '全部':
        result = result[result.label.eq({'正常邮件':0,'垃圾邮件':1}[label])]
    if query.strip():
        q = query.strip()
        result = result[result.subject_clean.str.contains(q,case=False,regex=False)
                        | result.doc_id.str.contains(q,case=False,regex=False)]
    return result.reset_index(drop=True)


def read_mail(row):
    # Only accept the canonical data/... member corresponding to this catalog ID.
    path = str(row.relative_path)
    if '..' in Path(path).parts or not path.endswith('/'+str(row.doc_id)):
        raise ValueError('邮件索引路径无效')
    with ZipFile(read_config('configs/data.yaml')['zip_path']) as archive:
        raw = archive.read(path)
    if hashlib.sha256(raw).hexdigest() != row.raw_sha256:
        raise ValueError('邮件原文校验失败')
    record = record_from_bytes(raw)
    assert 'label' not in record and 'relative_path' not in record
    return raw, record


def mask_contacts(text):
    """Display-only convenience; not anonymization and never a model transform."""
    text = re.sub(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}', '[邮箱]', str(text))
    return re.sub(r'(?<!\d)1[3-9]\d{9}(?!\d)', '[手机号]', text)
