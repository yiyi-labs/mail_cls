"""Acquire public mail without extraction; group before any model sees labels."""
import hashlib
import json
import re
import tarfile
import urllib.request
import zlib
from pathlib import Path
import pandas as pd
from datasketch import MinHash, MinHashLSH
from sklearn.model_selection import StratifiedGroupKFold
from spam_analysis.parsing import parse_email
from spam_analysis.grouping import grouping_view, compact, shingles, jaccard, UnionFind
from spam_analysis.io import sha256, save_json
from .runtime import ROOT

BASE = 'https://spamassassin.apache.org/old/publiccorpus/'
ARCHIVES = {
    '20030228_easy_ham.tar.bz2': 0, '20030228_easy_ham_2.tar.bz2': 0,
    '20030228_hard_ham.tar.bz2': 0, '20030228_spam.tar.bz2': 1,
    '20050311_spam_2.tar.bz2': 1,
}
DIRECTORY = ROOT/'data/raw/spamassassin'
OUTPUT = ROOT/'data/processed/spamassassin'
CORPUS = ROOT/'data/processed/corpus.parquet'
ENRON = ROOT/'data/processed/enron'
ENRON_ARCHIVES = {name+'.tar.gz':0 for name in ['beck-s','farmer-d','kaminski-v','kitchen-l','lokay-m','williams-w3']}
ENRON_ARCHIVES.update({name+'.tar.gz':1 for name in ['BG','GP','SH']})
SOURCE_NAMES = {'trec06c':'TREC06C', 'spamassassin':'Apache SpamAssassin', 'enron':'Enron-Spam'}


def download():
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    for filename in ['readme.html', *ARCHIVES]:
        destination = DIRECTORY/filename
        if not destination.exists():
            with urllib.request.urlopen(BASE+filename, timeout=90) as response:
                content = response.read(12*1024*1024+1)
            if len(content) > 12*1024*1024:
                raise ValueError('公开语料包超过预期大小')
            temporary = destination.with_suffix(destination.suffix+'.part')
            temporary.write_bytes(content)
            temporary.replace(destination)
        print('Source acquired:', filename, sha256(destination), flush=True)


def fingerprint(text, seed):
    grams = sorted(shingles(text))
    mh = MinHash(num_perm=128, seed=seed, hashfunc=zlib.crc32)
    for i in range(0, len(grams), 1024):
        mh.update_batch([g.encode('utf-8') for g in grams[i:i+1024]])
    return mh


def group_and_filter(frame, legacy, output=OUTPUT, prefix='sa_'):
    # Existing full-dataset text is used only for overlap exclusion, never fitting.
    old_exact = {compact(s)+'\n'+compact(b) for s, b in zip(legacy.subject_clean, legacy.body_clean)}
    frame = frame.assign(exact=[compact(s)+'\n'+compact(b) for s, b in zip(frame.subject_clean, frame.body_clean)])
    duplicate_raw = int(frame.raw_sha256.duplicated().sum())
    frame = frame.drop_duplicates('raw_sha256')
    overlap_exact = int(frame.exact.isin(old_exact).sum())
    frame = frame.loc[~frame.exact.isin(old_exact)].copy()
    conflict = frame.groupby('exact').label.nunique()
    conflicts = set(conflict[conflict > 1].index)
    # Contradictory source labels are quarantined before fitting; no model involved.
    frame.loc[frame.exact.isin(conflicts)].drop(columns='exact').to_parquet(output/'label_conflicts.parquet', index=False)
    frame = frame.loc[~frame.exact.isin(conflicts)].drop_duplicates('exact').reset_index(drop=True)
    texts = [grouping_view(b) if len(compact(b)) >= 80 else compact(s)+'\n'+compact(b)
             for s, b in zip(frame.subject_clean, frame.body_clean)]
    old_views = set(grouping_view(b) for b in legacy.body_clean if len(compact(b)) >= 80)
    uf = UnionFind(len(frame)); overlapping = set(); verified = set()
    # Two candidate passes, exact Jaccard confirms edges; no exhaustive recall claim.
    for seed in [42, 43]:
        lsh = MinHashLSH(threshold=.8, num_perm=128)
        candidates = {}
        for i, text in enumerate(sorted(old_views)):
            key = f'old:{i}'; candidates[key] = text
            lsh.insert(key, fingerprint(text, seed))
        for i, text in enumerate(texts):
            if text in old_views:
                overlapping.add(i)
            mh = fingerprint(text, seed)
            for key in lsh.query(mh):
                other = candidates[key]
                if min(len(text), len(other))/max(len(text), len(other), 1) < .8:
                    continue
                if jaccard(shingles(text), shingles(other)) < .9:
                    continue
                if key.startswith('old:'):
                    overlapping.add(i)
                else:
                    j = int(key.split(':')[1]); uf.union(i, j); verified.add(tuple(sorted((i, j))))
            key = f'new:{i}'; candidates[key] = text; lsh.insert(key, mh)
        print('Overlap grouping pass complete:', seed, flush=True)
    excluded_groups = {uf.find(i) for i in overlapping}
    frame['template_group_id'] = [prefix+str(uf.find(i)).zfill(6) for i in range(len(frame))]
    keep = [uf.find(i) not in excluded_groups for i in range(len(frame))]
    frame = frame.loc[keep].drop(columns='exact').reset_index(drop=True)
    audit = dict(duplicate_raw=duplicate_raw, overlap_exact=overlap_exact,
                 overlap_near_or_template=int(sum(not k for k in keep)), conflicting_content_groups=len(conflicts),
                 verified_near_edges=len(verified), groups=int(frame.template_group_id.nunique()),
                 limitation='Two MinHash passes; exact Jaccard >=0.9 verifies candidates, not exhaustive semantic duplicate proof')
    return frame, audit


def prepare():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT/'source.json').exists():
        meta = json.loads((OUTPUT/'source.json').read_text(encoding='utf-8'))
        if sha256(OUTPUT/'records.parquet') != meta['records_sha256']:
            raise ValueError('补充语料身份不一致')
        return meta
    download()
    rows = []
    for filename, label in ARCHIVES.items():
        with tarfile.open(DIRECTORY/filename, 'r:bz2') as archive:
            for member in archive.getmembers():
                if not member.isfile() or not re.match(r'^\d+\.', Path(member.name).name):
                    continue
                if member.size > 5*1024*1024:
                    raise ValueError('原始邮件超过预期大小')
                raw = archive.extractfile(member).read()
                digest = hashlib.sha256(raw).hexdigest()
                rows.append(dict(doc_id='sa_'+digest, raw_sha256=digest, label=label,
                                 source='SpamAssassin 2002–2005', source_id='spamassassin',
                                 archive=filename, member=member.name, label_origin='publisher_folder',
                                 **parse_email(raw)))
    source_count = len(rows)
    legacy = pd.read_parquet(ROOT/'data/processed/records.parquet', columns=['subject_clean','body_clean'])
    frame, audit = group_and_filter(pd.DataFrame(rows), legacy)
    folds = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    frame['split'] = ''
    names = ['train']*5+['tune','calibrate','policy','test','test']
    for fold, (_, indices) in enumerate(folds.split(frame, frame.label, frame.template_group_id)):
        frame.loc[indices, 'split'] = names[fold]
    if frame.groupby('template_group_id').split.nunique().max() != 1:
        raise AssertionError('模板跨集合')
    frame['language_scope'] = '历史英文邮件为主'
    frame.to_parquet(OUTPUT/'records.parquet', index=False)
    frame[['doc_id','label','split','template_group_id']].to_parquet(OUTPUT/'manifest.parquet', index=False)
    summary = frame.groupby(['split','label']).size().unstack(fill_value=0).rename(columns={0:'ham',1:'spam'})
    summary.to_csv(OUTPUT/'summary.csv')
    meta = dict(source_id='spamassassin', source_url=BASE+'readme.html', source_rows=source_count,
                eligible_rows=len(frame), **audit, purpose='Historical cross-source English auxiliary corpus; not modern Chinese evidence',
                rights='Offered by publisher for spam-filter research; message copyright remains with original senders',
                archives={name:sha256(DIRECTORY/name) for name in ARCHIVES},
                records_sha256=sha256(OUTPUT/'records.parquet'), manifest_sha256=sha256(OUTPUT/'manifest.parquet'),
                split_seed=42, split_groups=True, split_counts=frame.split.value_counts().to_dict())
    save_json(OUTPUT/'source.json', meta)
    print(json.dumps(meta, ensure_ascii=False), flush=True)
    return meta


def read_raw(row):
    source = row.source_id
    archives, directory = (ENRON_ARCHIVES, ROOT/'data/raw/enron') if source=='enron' else (ARCHIVES, DIRECTORY)
    if source not in {'enron','spamassassin'} or row.archive not in archives:
        raise ValueError('未知来源包')
    with tarfile.open(directory/row.archive, 'r:*') as archive:
        raw = archive.extractfile(str(row.member)).read()
    if hashlib.sha256(raw).hexdigest() != row.raw_sha256:
        raise ValueError('邮件身份不一致')
    return raw


def prepare_enron():
    """Official raw Enron-Spam; source labels preserved, cross-source duplicates excluded."""
    ENRON.mkdir(parents=True, exist_ok=True)
    if (ENRON/'source.json').exists():
        meta=json.loads((ENRON/'source.json').read_text(encoding='utf-8'))
        if sha256(ENRON/'records.parquet')!=meta['records_sha256']:raise ValueError('Enron 身份不一致')
        return meta
    directory=ROOT/'data/raw/enron';rows=[]
    for filename,label in ENRON_ARCHIVES.items():
        with tarfile.open(directory/filename,'r:gz') as archive:
            for member in archive.getmembers():
                if not member.isfile():continue
                if member.size>5*1024*1024:raise ValueError('邮件超过 5 MiB')
                raw=archive.extractfile(member).read()
                # Raw archives contain numbered messages, not preprocessed variants.
                digest=hashlib.sha256(raw).hexdigest()
                rows.append(dict(doc_id='en_'+digest,raw_sha256=digest,label=label,
                    source='Enron-Spam 1999–2005',source_id='enron',archive=filename,member=member.name,
                    label_origin='publisher_archive',**parse_email(raw)))
        print('Parsed',filename,flush=True)
    legacy=pd.concat([pd.read_parquet(ROOT/'data/processed/records.parquet'),pd.read_parquet(OUTPUT/'records.parquet')],ignore_index=True)
    frame,audit=group_and_filter(pd.DataFrame(rows),legacy,ENRON,'en_')
    folds=StratifiedGroupKFold(n_splits=10,shuffle=True,random_state=42)
    frame['split']='';names=['train']*5+['tune','calibrate','policy','test','test']
    for fold,(_,indices) in enumerate(folds.split(frame,frame.label,frame.template_group_id)):
        frame.loc[indices,'split']=names[fold]
    frame['language_scope']='历史英文邮件为主'
    frame.to_parquet(ENRON/'records.parquet',index=False)
    frame[['doc_id','label','split','template_group_id']].to_parquet(ENRON/'manifest.parquet',index=False)
    meta=dict(source_id='enron',source_url='https://www2.aueb.gr/users/ion/data/enron-spam/readme.txt',
        source_rows=len(rows),eligible_rows=len(frame),**audit,
        purpose='Historical office and spam email; not contemporary Chinese evidence',
        label_basis='Publisher raw ham/spam archives; no inferred labels or preprocessing copies',
        archives={name:sha256(directory/name) for name in ENRON_ARCHIVES},
        records_sha256=sha256(ENRON/'records.parquet'),manifest_sha256=sha256(ENRON/'manifest.parquet'),
        split_seed=42,split_groups=True,split_counts=frame.split.value_counts().to_dict())
    save_json(ENRON/'source.json',meta);print(json.dumps(meta,ensure_ascii=False),flush=True)
    return meta


def script_profile(frame):
    """Transparent script proxy, not a claim of validated language identification."""
    import numpy as np
    zh=frame.chinese_ratio.to_numpy(float);latin=frame.latin_ratio.to_numpy(float)
    return np.select([frame.body_chars.to_numpy()==0,(zh>=.1)&(latin>=.2),zh>=.1,latin>=.2],
                     ['空正文','中拉混合','中文字符为主','拉丁字母为主'],default='其他 / 难判')


def merge_corpus():
    """One active table. Frozen source artifacts remain provenance, never a new split."""
    from spam_analysis.io import load_records
    old=load_records().assign(source_id='trec06c',source='TREC06C 2006',label_origin='publisher_index')
    frames=[old];sources={'trec06c':sha256(ROOT/'data/processed/records.parquet')}
    for key,folder in [('spamassassin',OUTPUT),('enron',ENRON)]:
        if (folder/'source.json').exists():
            meta=json.loads((folder/'source.json').read_text(encoding='utf-8'))
            if sha256(folder/'records.parquet')!=meta['records_sha256']:raise ValueError('源表身份改变')
            frames.append(pd.read_parquet(folder/'records.parquet'));sources[key]=meta['records_sha256']
    frame=pd.concat(frames,ignore_index=True)
    if not frame.doc_id.is_unique or not frame.label.isin([0,1]).all():raise ValueError('合并身份或标签无效')
    if frame.groupby('template_group_id').split.nunique().max()!=1:raise ValueError('合并引入模板泄漏')
    frame['script_profile']=script_profile(frame)
    temporary=CORPUS.with_suffix('.tmp.parquet');frame.to_parquet(temporary,index=False);temporary.replace(CORPUS)
    save_json(CORPUS.with_suffix('.json'),dict(rows=len(frame),sources=sources,sha256=sha256(CORPUS),
        split_counts=frame.split.value_counts().to_dict(),
        script_note='Script proportion heuristic on cleaned bodies; Latin text is not necessarily English',
        source_rows=frame.source_id.value_counts().to_dict()))
    return frame


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('stage',nargs='?',default='spamassassin',choices=['spamassassin','enron','merge'])
    stage=parser.parse_args().stage
    {'spamassassin':prepare,'enron':prepare_enron,'merge':merge_corpus}[stage]()
