"""Train-only descriptive analysis; never fits a model or selects a threshold."""
from html import escape
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from spam_analysis.analysis.distributions import training_frame,grouped_distribution,corpus_terms
corpus_terms=st.cache_data(show_spinner=False)(corpus_terms)

FIELDS={'body_chars':'正文字符数','digit_ratio':'数字占比','url_count':'链接数量','chinese_ratio':'中文占比'}




@st.cache_data(show_spinner=False)
def analysis_records():
    columns=['doc_id','label',*FIELDS,'subject_chars','body_missing','replacement_ratio']
    return training_frame(pd.read_parquet('data/processed/records.parquet',columns=columns),
                          pd.read_parquet('artifacts/splits/split_manifest.parquet',columns=['doc_id','split']))




def render_analysis(frame=None):
    frame=analysis_records() if frame is None else frame
    st.html('<div class="analysis-flow"><div><b>01 · Pandas</b><span>邮件表与缺失检查</span></div><div><b>02 · NumPy</b><span>分箱、比例与分位数</span></div><div><b>03 · TF-IDF</b><span>文本转为稀疏向量</span></div><div><b>04 · scikit-learn</b><span>分类与误差评估</span></div></div>')
    st.subheader('训练数据里，哪些差异看得见？')
    st.caption(f'训练集 {len(frame):,} 封 · 按类别比例比较，避免类别数量差异造成偏差。')
    field=st.selectbox('观察特征',list(FIELDS),format_func=FIELDS.get,key='analysis_feature')
    left,right=st.columns([1.5,1])
    distribution=grouped_distribution(frame,field)
    with left,st.container(height=365,border=False):
        chart=alt.Chart(distribution).mark_bar(cornerRadiusTopLeft=3,cornerRadiusTopRight=3).encode(
            x=alt.X('区间:N',sort=distribution.drop_duplicates('顺序').sort_values('顺序')['区间'].tolist(),title=FIELDS[field],axis=alt.Axis(labelAngle=-35,labelFontSize=10)),
            y=alt.Y('类内比例:Q',title='占该类邮件的比例',axis=alt.Axis(format='%')),
            xOffset='类别:N',color=alt.Color('类别:N',scale=alt.Scale(domain=['正常邮件','垃圾邮件'],range=['#6265d9','#be794f']),legend=alt.Legend(title=None,orient='top')),
            tooltip=['类别','区间','邮件数','分母',alt.Tooltip('类内比例:Q',format='.2%')]).properties(height=260,background='transparent').configure_view(stroke=None)
        st.altair_chart(chart,width='stretch',theme=None)
    with right,st.container(height=365,border=False):
        summary=frame.groupby('label')[field].agg(邮件数='size',均值='mean',中位数='median',P90=lambda x:float(np.quantile(x,.9)))
        summary.index=summary.index.map({0:'正常邮件',1:'垃圾邮件'})
        for col in ['均值','中位数','P90']:summary[col]=summary[col].map(lambda v:f'{v:.2%}' if field.endswith('_ratio') else f'{v:,.2f}')
        st.html('<div class="results-table">'+summary.reset_index(names='类别').to_html(index=False,escape=True,border=0)+'</div>')
        missing=int(frame.body_missing.sum());damaged=int(frame.replacement_ratio.gt(0).sum())
        st.caption(f'质量检查：正文为空 {missing:,} 封；含替换字符 {damaged:,} 封。两类可重叠，仍保留在统计中。')
        st.download_button('下载当前分组统计',distribution.to_csv(index=False).encode('utf-8-sig'),file_name=f'train_{field}_distribution.csv',mime='text/csv',key='download_analysis')
    st.subheader('常出现的词，一定更能区分类别吗？')
    saved=Path('artifacts/tables/corpus_terms.json');identity=Path('data/processed/corpus.json')
    if saved.exists() and identity.exists() and json.loads(saved.read_text())['corpus_sha256']==json.loads(identity.read_text(encoding='utf-8'))['sha256']:
        words=pd.read_parquet('artifacts/tables/corpus_terms.parquet')
    else:
        words=corpus_terms(frame[['doc_id','body_clean','label']]) if 'body_clean' in frame else pd.read_csv('artifacts/tables/train_discriminative_terms.csv')
    top=pd.concat([words.nlargest(6,'spam_minus_ham'),words.nsmallest(6,'spam_minus_ham')]).copy()
    top['偏向']=np.where(top.spam_minus_ham>=0,'垃圾邮件中更多','正常邮件中更多')
    left,right=st.columns([1.5,1])
    with left:
        bars=alt.Chart(top).mark_bar(cornerRadiusEnd=3).encode(
            y=alt.Y('term:N',title=None,sort=top.term.tolist()),
            x=alt.X('spam_minus_ham:Q',title='垃圾类出现率 − 正常类出现率',axis=alt.Axis(format='%')),
            color=alt.Color('偏向:N',scale=alt.Scale(domain=['正常邮件中更多','垃圾邮件中更多'],range=['#6265d9','#be794f']),legend=None),
            tooltip=[alt.Tooltip('term:N',title='词语'),alt.Tooltip('ham_rate:Q',title='正常类出现率',format='.2%'),alt.Tooltip('spam_rate:Q',title='垃圾类出现率',format='.2%')]
        ).properties(height=280,background='transparent').configure_view(stroke=None)
        st.altair_chart(bars,width='stretch',theme=None)
    with right:
        strongest=top.iloc[0]
        st.markdown('**看词频，也看比较的分母。**')
        st.write(f'“{strongest.term}”出现在 {strongest.ham_rate:.1%} 的正常邮件、{strongest.spam_rate:.1%} 的垃圾邮件中。')
        st.caption('每封邮件最多计一次出现，展示两类出现率差异最大的词语。')
        st.markdown('**从统计到模型**')
        st.caption('词频与 IDF 构成向量，分类器学习词语组合与类别的关系。原文、词元和特征可在实验室联动查看。')
