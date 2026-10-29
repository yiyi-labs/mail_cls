"""Read-only views of the exact saved tokenizer and fitted feature transforms."""
from collections import Counter
from functools import lru_cache
from pathlib import Path
import json
import numpy as np
from spam_analysis.io import sha256
from spam_analysis.macbert import batches
from inference_trace import LOCK
from model_catalog import short_name


@lru_cache(maxsize=2)
def local_tokenizer(branch,folder,identities):
    from transformers import AutoTokenizer
    for name,digest in json.loads(identities).items():
        if name.startswith(('tokenizer','vocab','merges','special_tokens','added_tokens')):
            if sha256(Path(folder)/name)!=digest:raise ValueError('分词器文件身份发生变化')
    kwargs={'padding_side':'left','truncation_side':'right'} if branch=='E1' else {}
    return AutoTokenizer.from_pretrained(folder,local_files_only=True,trust_remote_code=False,**kwargs)


def token_item(text,start,end,token,token_id,index,field,special=False):
    left=max(0,start-45);right=min(len(text),end+60)
    return {'index':index,'token':str(token),'id':int(token_id) if token_id is not None else None,
            'field':field,'start':int(start),'end':int(end),'special':special,
            'surface':text[start:end] if not special else str(token),
            'before':text[left:start],'after':text[end:right],
            'prefix_omitted':left>0,'suffix_omitted':right<len(text)}


def neural_tokenization(branch,tokenizer,record,entry):
    row=record.iloc[0]
    if branch=='E2':
        subject=tokenizer(row.subject_clean,add_special_tokens=False,return_offsets_mapping=True)
        body=tokenizer(row.body_clean,add_special_tokens=False,return_offsets_mapping=True)
        encoded=batches(record,tokenizer,entry['config'],entry['checkpoint']['mode'])[0]
        ns=min(len(subject['input_ids']),entry['config']['macbert_subject_max'])
        budget=entry['config']['macbert_max_length']-ns-tokenizer.num_special_tokens_to_add(pair=True)
        indices=list(range(len(body['input_ids'])))
        if len(indices)>budget:
            head=(budget+1)//2
            indices=(indices[:head]+(indices[-(budget-head):] if budget>head else [])) if entry['checkpoint']['mode']=='head_tail' else indices[:budget]
        mapping=[('主题',row.subject_clean,subject,i) for i in range(ns)]+[('正文',row.body_clean,body,i) for i in indices]
        # Negative markers let the tokenizer itself place its special tokens.
        markers=tokenizer.build_inputs_with_special_tokens(list(range(-1,-ns-1,-1)),list(range(-ns-1,-len(mapping)-1,-1)))
        ids=encoded['input_ids'];items=[]
        if len(markers)!=len(ids):raise ValueError('词元边界与推理输入长度不一致')
        for j,(marker,token_id) in enumerate(zip(markers,ids)):
            token=tokenizer.convert_ids_to_tokens(token_id)
            if marker>=0:items.append(token_item('',0,0,token,token_id,j,'特殊符号',True))
            else:
                field,text,source,i=mapping[-marker-1];start,end=source['offset_mapping'][i]
                if token_id!=source['input_ids'][i]:raise ValueError('词元预览与推理输入不一致')
                items.append(token_item(text,start,end,token,token_id,j,field))
        original=len(subject['input_ids'])+len(body['input_ids'])+tokenizer.num_special_tokens_to_add(pair=True)
        rule=f'主题最多 {entry["config"]["macbert_subject_max"]}；正文'+('首尾保留' if entry['checkpoint']['mode']=='head_tail' else '保留开头')
        limit=entry['config']['macbert_max_length']
    else:
        text=row.semantic_text
        source=tokenizer(text,return_offsets_mapping=True,truncation=False)
        encoded=tokenizer(text,return_offsets_mapping=True,truncation=True,max_length=entry['config']['max_length'])
        ids=encoded['input_ids'];items=[]
        for j,(token_id,(start,end)) in enumerate(zip(ids,encoded['offset_mapping'])):
            items.append(token_item(text,start,end,tokenizer.convert_ids_to_tokens(token_id),token_id,j,'语义文本',start==end))
        original=len(source['input_ids']);rule='拼接主题与正文，仅保留开头';limit=entry['config']['max_length']
    backend=json.loads(tokenizer.backend_tokenizer.to_str())['model']['type']
    return {'name':short_name(branch),'kind':'neural','algorithm':backend,'rule':rule,
            'total':len(ids),'original':original,'omitted':original-len(ids),'limit':limit,
            'tokens':items,'input_ids':list(ids),'unique':len(set(ids)),
            'frequency':[{'term':tokenizer.convert_ids_to_tokens(i) if branch=='E2' else next(t['surface'] or t['token'] for t in items if t['id']==i)+f' · #{i}',
                          'count':n} for i,n in Counter(ids).most_common(10)]}


def sparse_tokenization(artifact,record):
    blocks=[];row=record.iloc[0];offset=0;classifier=artifact.estimator
    coefficients=classifier.coef_[0] if hasattr(classifier,'coef_') else classifier.feature_log_prob_[1]-classifier.feature_log_prob_[0]
    for block,field,vec,weight in artifact.features.transforms_:
        text=str(row[field]);terms=vec.build_analyzer()(text);counts=Counter(terms)
        matrix=vec.transform([text]).getrow(0)
        names=vec.get_feature_names_out();selected=np.argsort(-np.abs(matrix.data))[:12]
        raw=np.array([(1+np.log(counts[names[i]]))*vec.idf_[i] for i in matrix.indices])
        norm=float(np.linalg.norm(raw))
        features=[{'term':str(names[matrix.indices[i]]),'count':int(counts[names[matrix.indices[i]]]),
                   'id':int(matrix.indices[i]),'idf':float(vec.idf_[matrix.indices[i]]),
                   'tf':float(1+np.log(counts[names[matrix.indices[i]]])),
                   'norm':norm,'block_weight':float(weight),'tfidf':float(matrix.data[i]*weight)} for i in selected]
        for feature in features:
            index=offset+feature['id'];ratio=float(artifact.ratio.r_[index]) if artifact.ratio is not None else 1.
            feature['coefficient']=float(coefficients[index])*ratio
            feature['contribution']=feature['tfidf']*feature['coefficient']
        offset+=len(names)
        sequence=vec.build_tokenizer()(vec.build_preprocessor()(text)) if vec.analyzer=='word' else terms
        items=[];cursor=0
        for i,term in enumerate(sequence[:400]):
            start=text.find(term,cursor) if vec.analyzer=='word' else -1
            if start>=0:
                item=token_item(text,start,start+len(term),term,vec.vocabulary_.get(term),i,field)
                cursor=start+len(term)
            else:
                item=token_item('',0,0,term,vec.vocabulary_.get(term),i,field)
                item['surface']=term
            item['known']=term in vec.vocabulary_;items.append(item)
        blocks.append({'name':{'word':'词语','char':'字符片段','subject':'主题词语'}[block],
            'kind':'sparse','algorithm':'Jieba · HMM=False' if vec.analyzer=='word' else '字符 n-gram',
            'rule':f'{field} · {vec.ngram_range[0]}–{vec.ngram_range[1]} gram · 区块权重 {weight:g}',
            'tokens':items,'total':len(sequence),'original':len(sequence),'omitted':0,'limit':None,
            'unique':len(set(sequence)),'active':int(matrix.nnz),'features':features,
            'frequency':[{'term':t,'count':n} for t,n in Counter(sequence).most_common(10)]})
    return blocks


def text_payload(engine,record,chosen,trace=None):
    if record is None:return {'branches':[],'state':'请先选择或输入邮件'}
    if len(record)!=1 or 'label' in record:raise ValueError('分词视图仅接受一封不带标签的邮件')
    if not record.lexical_text.iloc[0]:return {'branches':[],'state':'空文本使用训练集先验，不执行分词分类'}
    branches=[]
    with LOCK:
        for branch,weight in engine.specifications[chosen]['weights'].items():
            if not weight:continue
            entry=engine.entries[branch]
            if branch in ('E1','E2'):
                identity=entry['checkpoint'] if branch=='E2' else entry['identity']
                tokenizer=local_tokenizer(branch,identity['folder'],json.dumps(identity['files'],sort_keys=True))
                result=neural_tokenization(branch,tokenizer,record,entry)
                actual=trace['branches'][branch].get('model_input_ids') if trace else None
                if actual is not None and result['input_ids']!=actual:raise ValueError('展示词元与本次模型实际输入不同')
                result['verified']=actual is not None;branches.append(result)
            else:
                engine.load(branch)
                for block in sparse_tokenization(engine.loaded[branch][0],record):
                    block['name']=short_name(branch)+' / '+block['name'];branches.append(block)
    row=record.iloc[0]
    return {'branches':branches,'state':'就绪','stats':{'主题字符':int(row.subject_chars),'正文字符':int(row.body_chars),
            '链接':int(row.url_count),'数字占比':f'{row.digit_ratio:.1%}'}}
