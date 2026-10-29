"""Course presentation: local classification, model comparison and data analysis."""
from pathlib import Path
import sys
import os
import pandas as pd
import streamlit as st

ROOT=Path(__file__).resolve().parents[1]
for folder in [ROOT/'src',ROOT/'app']:
    if str(folder) not in sys.path:sys.path.insert(0,str(folder))
os.chdir(ROOT)
from spam_analysis.inference import record_from_text,record_from_bytes
from spam_analysis.experiments.runtime import ExperimentSession,active_plan
from spam_analysis.analysis.uploads import read_upload,export_csv
from spam_analysis.analysis.results import source_records,CORE
from spam_analysis.experiments.corpus import CORPUS,SOURCE_NAMES
from model_catalog import short_name,METHODS
from mail_catalog import load_catalog,search_catalog,read_mail
from inference_trace import classify_with_trace,LOCK
from theme import apply_theme,hero,section,result_card,empty_result
from evidence_views import comparison_view,data_view
from mail_preview import preview_mail
from computation_view import render_computation
from lab_views import compare_mail
from text_view import render_text_lens
from what_if import render_what_if

st.set_page_config(page_title='邮件研判 · Python 数据分析',page_icon='✉',layout='wide')
apply_theme()


@st.cache_resource
def session(path,version):return ExperimentSession(path)


@st.cache_data(show_spinner=False)
def catalog(version):return source_records('combined')


plan=active_plan();engine=session(str(plan),plan.stat().st_mtime_ns)
with st.container(key='navigation'):
    page=st.radio('实验导航',['邮件实验室','模型对比','数据分析'],horizontal=True,label_visibility='collapsed',key='page')
if page=='模型对比':
    comparison_view(engine)
elif page=='数据分析':
    data_view(engine)
else:
    hero('MAIL INTELLIGENCE','邮件研判实验室','读懂一封邮件，也看见判断的依据')
    record=None;truth=None;batch_records=None
    left,right=st.columns([1.25,1],gap='medium')
    with left,st.container(key='compose',height=510,border=False):
        section('01 / INPUT','邮件内容')
        mode=st.radio('邮件来源',['自己输入','上传文件','数据集邮件'],horizontal=True,label_visibility='collapsed',key='input_mode')
        if mode=='自己输入':
            subject=st.text_input('主题','课程项目答疑时间',key='subject')
            body=st.text_area('正文','同学们好，本周四下午安排项目答疑，请准备好代码与问题。',height=225,key='body')
            record=record_from_text(subject,body)
            st.caption('可直接粘贴邮件内容 · 默认文本为合成演示示例')
        elif mode=='上传文件':
            upload=st.file_uploader('EML / TXT / CSV · 最大 5 MiB',type=['eml','txt','csv'],key='mail_upload')
            st.caption('CSV：body（正文）必填，subject（主题）可选；最多 100 封。文件仅用于本地分类。')
            st.download_button('下载 CSV 模板','subject,body\n课程安排,请按时提交本周课程作业。\n'.encode('utf-8-sig'),file_name='邮件模板.csv',mime='text/csv')
            if upload is not None:
                try:
                    batch_records=read_upload(upload.name,upload.getvalue())
                    if len(batch_records)>1:
                        index=st.selectbox('预览邮件',range(len(batch_records)),format_func=lambda i:f'{i+1}. {batch_records.subject_clean.iloc[i] or "（无主题）"}',key='upload_preview')
                        record=batch_records.iloc[[index]].reset_index(drop=True)
                        st.caption(f'已读取 {len(batch_records)} 封；可分类当前邮件或批量分类全部。')
                    else:record=batch_records
                    if upload.name.lower().endswith('.eml'):preview_mail(upload.getvalue(),record,'uploaded_mail')
                    else:st.text(record.body_clean.iloc[0][:1200])
                except (ValueError,UnicodeError,pd.errors.ParserError,pd.errors.EmptyDataError) as exc:
                    st.error(str(exc));record=None;batch_records=None
        else:
            pending=st.session_state.pop('open_mail',None)
            if pending:st.session_state.update(mail_split='test',mail_query=pending['doc_id'],mail_category='全部',mail_page=1)
            c1,c2=st.columns(2)
            split=c1.selectbox('集合',['test','train','tune','calibrate','policy','全部'],format_func=lambda x:{'train':'训练集','tune':'调参集','calibrate':'校准集','policy':'策略集','test':'留出集','全部':'全部'}[x],key='mail_split')
            category=c2.selectbox('标签',['全部','正常邮件','垃圾邮件'],key='mail_category')
            query=st.text_input('搜索主题或邮件编号',key='mail_query')
            found=search_catalog(catalog(CORPUS.stat().st_mtime_ns if CORPUS.exists() else 0),split,category,query)
            if len(found):
                selector,pager=st.columns([4,1]);pages=range(1,(len(found)+29)//30+1)
                if st.session_state.get('mail_page',1) not in pages:st.session_state['mail_page']=1
                page_no=pager.selectbox('页码',pages,key='mail_page')
                subset=found.iloc[(page_no-1)*30:page_no*30].set_index('doc_id',drop=False)
                selected=selector.selectbox('邮件',list(subset.index),format_func=lambda x:str(subset.loc[x,'subject_clean'])[:42] or '（无主题）',key='mail_selected')
                metadata=subset.loc[selected];truth=int(metadata.label)
                if metadata.source_id=='trec06c':raw,record=read_mail(metadata)
                else:
                    from spam_analysis.experiments.corpus import read_raw
                    raw=read_raw(metadata);record=record_from_bytes(raw)
                st.caption(f'{metadata.split.upper()} · {SOURCE_NAMES[metadata.source_id]} · 参考标签：{"正常" if truth==0 else "垃圾"} · {len(found):,} 封')
                if metadata.split=='train':st.caption('训练样本用于观察计算过程，不能衡量泛化效果。')
                preview_mail(raw,record,selected)
            else:st.info('没有匹配邮件，请调整筛选条件。')
    with right,st.container(key='decision',height=510,border=False):
        section('02 / PREDICTION','分类判断')
        choices=[engine.main]+[k for k in CORE if k!=engine.main]
        chosen=st.selectbox('分类方法',choices,format_func=lambda x:short_name(x)+(' · 主模型' if x==engine.main else ''),key='chosen_model')
        point='research'
        st.caption(METHODS[chosen])
        key=None if record is None else (str(record.doc_id.iloc[0]),chosen,point,mode,engine.run_id)
        run=st.button('开始分类',type='primary',width='stretch',disabled=record is None,key='classify')
        saved=st.session_state.get('last_result');trace=saved['trace'] if saved and saved['key']==key else None
        with st.container(height=260,border=False,key='result-slot'):
            progress=st.progress(1. if trace else 0.,text='分类完成' if trace else '等待开始')
            surface=st.empty()
            if run:
                def on_event(event):
                    if event['phase']=='forward' and event['frame']['kind'] in {'embedding','layer_output','final_norm'}:
                        progress.progress(event['layer']/max(1,event['layers']),text=f'正在计算 · {event["layer"]}/{event["layers"]} 层')
                    elif event['phase']!='forward':progress.progress(0,text=event['title'])
                with st.spinner('正在计算…'):
                    trace=classify_with_trace(engine,record,chosen,point,on_event=on_event)
                st.session_state['last_result']={'key':key,'trace':trace}
                progress.progress(1.,text='分类完成')
            with surface.container():
                if trace:
                    result_card(trace)
                    st.caption(f'本次 {trace["elapsed_seconds"]:.3f} 秒，含计算记录采集。分数表示模型倾向，不是正确率。')
                    if truth is not None:st.caption('与参考标签一致' if truth==trace['prediction'] else '与参考标签不同，可查看关键证据。')
                else:empty_result()
    if batch_records is not None and len(batch_records)>1:
        batch_key=(engine.run_id,chosen,tuple(batch_records.doc_id))
        if st.button(f'批量分类全部 {len(batch_records)} 封',key='classify_batch'):
            rows=[];bar=st.progress(0,text='准备批量分类')
            for i in range(len(batch_records)):
                one=batch_records.iloc[[i]]
                with LOCK:scores,_=engine.scores(one,chosen)
                p=float(scores.score_spam.iloc[0]);threshold=engine.specifications[chosen]['thresholds'][point]
                rows.append({'序号':i+1,'主题':one.subject_clean.iloc[0],'垃圾分数':p,'结论':'垃圾邮件' if p>=threshold else '正常邮件'})
                bar.progress((i+1)/len(batch_records),text=f'已完成 {i+1}/{len(batch_records)} 封')
            st.session_state['batch_result']={'key':batch_key,'rows':rows}
        batch=st.session_state.get('batch_result')
        if batch and batch['key']==batch_key:
            table=pd.DataFrame(batch['rows']);st.dataframe(table,hide_index=True,width='stretch')
            st.download_button('下载分类结果',export_csv(table),file_name='邮件分类结果.csv',mime='text/csv')
    span=render_text_lens(engine,record,chosen,trace)
    render_what_if(engine,record,chosen,point,trace,span)
    with st.container(key='compare-panel',border=False):
        section('03 / COMPARE','同一封邮件，多种判断');compare_mail(engine,record,point)
    with st.expander('查看本次真实计算过程',expanded=False):render_computation(trace)
