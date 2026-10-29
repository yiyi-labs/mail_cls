"""Secondary lab tasks, kept out of the primary reading and decision flow."""
import streamlit as st
import pandas as pd
from inference_trace import LOCK
from model_catalog import short_name
from score_display import score_text


def compare_mail(engine,record,point):
    models=st.multiselect('对照方法',[k for k in ['B1','B2','S_char13','E1','E2'] if k in engine.entries],default=['B1','B2','S_char13','E2'],format_func=short_name,key='compare_models',label_visibility='collapsed')
    key=None if record is None else (record.doc_id.iloc[0],point,tuple(models),getattr(engine,'run_id','frozen-v1'))
    if st.button('对照分类这封邮件',disabled=record is None or not models,key='compare_mail'):
        rows=[]
        progress=st.progress(0,text='等待首个方法完成')
        for index,name in enumerate(models):
            with LOCK:scores,_=engine.scores(record,name)
            probability=float(scores.score_spam.iloc[0]);threshold=float(engine.specifications[name]['thresholds'][point])
            rows.append({'方法':short_name(name),'垃圾分数':score_text(probability),'阈值':score_text(threshold),'结论':'垃圾邮件' if probability>=threshold else '正常邮件'})
            progress.progress((index+1)/len(models),text=f'{short_name(name)} 完成 · {index+1}/{len(models)}')
        progress.empty()
        st.session_state['mail_comparison']={'key':key,'rows':rows}
    comparison=st.session_state.get('mail_comparison')
    if comparison and comparison['key']==key:
        st.html('<div class="results-table compact-table">'+pd.DataFrame(comparison['rows']).to_html(index=False,escape=True,border=0)+'</div>')
        if len({r['结论'] for r in comparison['rows']})>1:st.caption('● 方法间存在分歧，请结合原文核对。')
