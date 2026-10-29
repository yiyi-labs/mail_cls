"""Local text edits, evaluated with identical model identity and threshold."""
import streamlit as st
from spam_analysis.inference import record_from_text
from spam_analysis.analysis.uploads import replace_span
from inference_trace import classify_with_trace
from theme import result_card
from score_display import score_text


def render_what_if(engine, record, name, point, trace, span):
    key=None if record is None else (getattr(engine,'run_id','frozen-v1'),str(record.doc_id.iloc[0]),name,point)
    if st.session_state.get('edit_source')!=key:
        st.session_state.update(edit_source=key,edit_subject='' if record is None else str(record.subject_clean.iloc[0]),
                                edit_body='' if record is None else str(record.body_clean.iloc[0]))
        st.session_state.pop('edit_result',None)
    if span and record is not None:
        st.session_state['chosen_span']={'key':key,'span':span}
    st.subheader('改写一句，判断会怎样变化？')
    selected=st.session_state.get('chosen_span')
    if selected and selected['key']==key:
        st.caption('选中片段：'+selected['span'].get('surface',''))
        replacement=st.text_input('替换为（留空即删除）',key='span_replacement')
        if st.button('应用片段修改',key='apply_span'):
            try:
                subject,body=replace_span(record,selected['span'],replacement)
                st.session_state.update(edit_subject=subject,edit_body=body)
            except ValueError as exc:
                st.info(str(exc))
    left,right=st.columns([1.25,1],gap='medium')
    with left,st.container(height=340,border=True):
        st.text_input('修改后的主题',key='edit_subject',disabled=record is None)
        st.text_area('修改后的正文',height=145,key='edit_body',disabled=record is None)
        run,reset=st.columns(2)
        clicked=run.button('重算并对照',type='primary',width='stretch',disabled=not trace,key='recalculate')
        def undo():
            st.session_state.update(edit_subject=str(record.subject_clean.iloc[0]),edit_body=str(record.body_clean.iloc[0]))
            st.session_state.pop('edit_result',None)
        reset.button('恢复原文',width='stretch',disabled=record is None,on_click=undo,key='undo_edit')
    with right,st.container(height=340,border=True):
        edited=record_from_text(st.session_state.edit_subject,st.session_state.edit_body)
        edit_key=(key,edited.doc_id.iloc[0])
        if clicked:
            with st.spinner('用同一模型重新计算…'):
                changed=classify_with_trace(engine,edited,name,point)
            st.session_state['edit_result']={'key':edit_key,'trace':changed}
        saved=st.session_state.get('edit_result')
        if trace and saved and saved['key']==edit_key:
            changed=saved['trace'];result_card(changed)
            st.metric('垃圾分数变化',f'{changed["probability"]-trace["probability"]:+.6f}')
            before=trace['branches'].get(name,{});after=changed['branches'].get(name,{})
            if 'raw_score' in before and 'raw_score' in after:
                st.caption(f'原始分数 {before["raw_score"]:.4f} → {after["raw_score"]:.4f}')
        else:
            st.markdown('**原文 → 修改 → 重新分类**')
            st.caption('先完成原文分类，再修改一段文字。模型、阈值保持相同；修改不会进入训练。')
            if trace:st.write('原文垃圾分数：'+score_text(trace['probability']))
    st.caption('对照反映本次修改的影响，不代表某个词始终决定分类。')
