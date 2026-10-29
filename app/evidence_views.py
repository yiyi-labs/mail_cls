"""Concise model and data views over version-matched experiment artifacts."""
from pathlib import Path
import pandas as pd
import streamlit as st
import altair as alt
from model_catalog import short_name,METHODS
from theme import hero
from spam_analysis.analysis.results import evaluation_scopes,result_table,source_records,error_frame,CORE
from spam_analysis.analysis.tables import reliability,length_edges,assign_slices,slice_errors
from spam_analysis.experiments.corpus import CORPUS,SOURCE_NAMES

PURPLE='#6265d9';CLAY='#be794f'


def metric_table(frame):
    table=frame[['model','accuracy','macro_f1','fp','fn']].copy();table['model']=table.model.map(short_name)
    for c in ['accuracy','macro_f1']:table[c]=table[c].map(lambda x:f'{x:.3%}')
    for c in ['fp','fn']:table[c]=table[c].astype(int)
    table=table.rename(columns={'model':'方法','accuracy':'准确率','macro_f1':'Macro-F1','fp':'正常误拦','fn':'垃圾漏判'})
    st.html('<div class="results-table">'+table.to_html(index=False,escape=True,border=0)+'</div>')


def comparison_view(engine):
    hero('MODEL COMPARISON','模型对比','相同邮件，比较判断质量与使用成本')
    scopes=evaluation_scopes(engine.run_id)
    scope='combined' if 'combined' in scopes else 'trec06c'
    st.caption('固定留出集回顾评估 · 模型、校准与阈值仅使用开发数据。' if engine.plan.get('evaluation_kind')=='retrospective_holdout' else ('统一留出邮件汇总 · 原 TREC06C 为历史回顾，新增来源为分组留出评价。' if scope=='combined' else '当前已发布版本的 TREC06C 历史留出成绩。'))
    rows=result_table(engine.run_id,scope);main=rows.set_index('model').loc[engine.main]
    a,b,c,d=st.columns(4);a.metric('主模型准确率',f'{main.accuracy:.3%}')
    b.metric('正常误拦',f'{int(main.fp)} 封');c.metric('垃圾漏判',f'{int(main.fn)} 封');d.metric('评价邮件',f'{int(main.n):,} 封')
    left,right=st.columns([1.25,1],gap='medium')
    with left:
        st.subheader('分类表现');metric_table(rows)
        st.caption('准确率衡量总体正确比例；Macro-F1 同等关注正常与垃圾两类。')
    with right:
        st.subheader('错误构成')
        errors=rows[['model','fp','fn']].melt('model',var_name='kind',value_name='count')
        errors['方法']=errors.model.map(short_name);errors['错误类型']=errors.kind.map({'fp':'正常误拦','fn':'垃圾漏判'})
        chart=alt.Chart(errors).mark_bar(cornerRadiusEnd=3).encode(y=alt.Y('方法:N',sort=[short_name(k) for k in CORE],title=None),
            x=alt.X('count:Q',title='邮件数'),color=alt.Color('错误类型:N',scale=alt.Scale(domain=['正常误拦','垃圾漏判'],range=[CLAY,PURPLE]),legend=alt.Legend(title=None,orient='top')),
            tooltip=['方法','错误类型',alt.Tooltip('count:Q',title='邮件数')]).properties(height=235,background='transparent').configure_view(stroke=None)
        st.altair_chart(chart,width='stretch',theme=None)
    st.subheader('模型如何处理邮件')
    conditions=[];perf_path=Path('artifacts/runs')/engine.run_id/'benchmark.csv'
    perf=pd.read_csv(perf_path).set_index('model') if perf_path.exists() else None
    for name in CORE:
        row={'方法':short_name(name),'特征与训练':METHODS[name]}
        if name=='E2':row.update(输入上限='512 词元',参数='约 1.02 亿 · 全量微调')
        elif name=='E1':row.update(输入上限='1024 词元',参数='0.6B 编码器冻结 · 训练分类器')
        else:row.update(输入上限='完整清洗文本',参数='取决于训练词表维数')
        if perf is not None and name in perf.index:
            p=perf.loc[name];row['推理中位数']=f'{p.median_ms:.1f} ms';row['P95']=f'{p.p95_ms:.1f} ms'
            row['参数']=f'{int(p.total_parameters):,} / {int(p.trainable_parameters):,} 可训练'
            row['GPU 峰值']=f'{p.gpu_allocated_peak_mib:.0f} MiB' if pd.notna(p.gpu_allocated_peak_mib) else 'CPU 推理'
        conditions.append(row)
    st.dataframe(pd.DataFrame(conditions),hide_index=True,width='stretch')
    st.caption('比较完整方案：参数量、输入长度与训练方式不同。Qwen 提取语义向量，MacBERT 学习邮件分类。')
    with st.expander('训练过程与课程方法'):training_view(engine)


def training_view(engine):
    path=Path('artifacts/runs')/engine.run_id/'training_curves.csv'
    curves=pd.read_csv(path if path.exists() else 'artifacts/macbert_learning_curves.csv')
    if 'loss_mode' not in curves:curves['loss_mode']='ordinary'
    curves['实验']=curves.loss_mode.map({'ordinary':'普通损失','balanced':'类别加权'}).fillna(curves.loss_mode)+' · '+curves.lr.map(lambda x:f'{x:g}')
    chosen=st.selectbox('训练配置',curves['实验'].unique(),key='curve_configuration');selected=curves.loc[curves['实验'].eq(chosen)]
    a,b=st.columns(2)
    for column,field,title in [(a,'training_loss','训练损失'),(b,'accuracy','调参集准确率')]:
        with column:
            chart=alt.Chart(selected).mark_line(point=True,color=PURPLE).encode(x=alt.X('epoch:O',title='轮次'),
                y=alt.Y(field+':Q',title=title,scale=alt.Scale(zero=field=='training_loss')),
                tooltip=['epoch',field,'training_s']).properties(height=200,background='transparent').configure_view(stroke=None)
            st.altair_chart(chart,width='stretch',theme=None)
    st.caption('训练集更新参数，调参集选择候选，校准集调整概率，策略集确定阈值，测试集评价最终效果。')


@st.cache_data(show_spinner=False)
def analysis_data(run_id,scope,model,threshold):return error_frame(run_id,scope,model,threshold)


@st.cache_data(show_spinner=False)
def source_data(version):return source_records('combined')


def open_mail(scope,doc_id):
    st.session_state.update(open_mail={'source':scope,'doc_id':str(doc_id)},page='邮件实验室',input_mode='数据集邮件')


def data_view(engine):
    hero('PYTHON DATA ANALYSIS','数据分析','清洗与探索 → 特征构建 → 分类与误差分析')
    scope='combined' if engine.run_id!='frozen-v1' else 'trec06c'
    frame=source_data(CORPUS.stat().st_mtime_ns if CORPUS.exists() else 0);train=frame.loc[frame.split.eq('train')]
    a,b,c,d=st.columns(4);a.metric('有效邮件',f'{len(frame):,}');b.metric('正常邮件',f'{int(frame.label.eq(0).sum()):,}')
    c.metric('垃圾邮件',f'{int(frame.label.eq(1).sum()):,}');d.metric('用于训练',f'{len(train):,}')
    st.caption('统一邮件表 · 保留来源与原始标签 · 训练和测试按模板组隔离。公开语料主要来自 1999–2006 年。')
    st.subheader('数据覆盖与类别分布')
    left,right=st.columns([1.25,1],gap='medium')
    with left:
        coverage=frame.groupby(['source_id','label']).size().rename('邮件数').reset_index()
        coverage['来源']=coverage.source_id.map(SOURCE_NAMES);coverage['类别']=coverage.label.map({0:'正常',1:'垃圾'})
        chart=alt.Chart(coverage).mark_bar(cornerRadiusEnd=3).encode(y=alt.Y('来源:N',title=None),x=alt.X('邮件数:Q'),
            color=alt.Color('类别:N',scale=alt.Scale(domain=['正常','垃圾'],range=[PURPLE,CLAY]),legend=alt.Legend(title=None,orient='top')),
            tooltip=['来源','类别','邮件数']).properties(height=180,background='transparent').configure_view(stroke=None)
        st.altair_chart(chart,width='stretch',theme=None)
    with right:
        language=frame.groupby('script_profile').agg(邮件数=('doc_id','size'),垃圾占比=('label','mean')).reset_index(names='正文文字类型')
        st.dataframe(language,hide_index=True,width='stretch',column_config={'垃圾占比':st.column_config.NumberColumn(format='percent')})
        st.caption('按中文与拉丁字母比例观察语言构成；拉丁字母为主不等同于全部为英语。')
    from data_analysis import render_analysis
    render_analysis(train)
    if scope not in evaluation_scopes(engine.run_id):return
    st.subheader('模型在哪些邮件上容易出错？')
    if engine.run_id=='frozen-v1':st.caption('以下错误与可靠性分析对应当前已发布模型的中文历史留出成绩。')
    model=st.selectbox('分析模型',CORE,index=CORE.index(engine.main),format_func=short_name,key='error_model')
    threshold=engine.specifications[model]['thresholds']['research'];evaluated=analysis_data(engine.run_id,scope,model,threshold)
    fields={'source':'邮件来源','language':'正文文字类型','length':'邮件长度','quality':'编码与正文质量','links':'链接数量','subject':'主题完整性'}
    if 'truncation_ratio' in evaluated:fields['truncation']='模型实际截断比例'
    field=st.selectbox('分组维度',list(fields),format_func=fields.get,key='slice_field')
    groups=assign_slices(evaluated,field,length_edges(train));table=slice_errors(evaluated,groups)
    shown=table[['group','count','false_positive','false_negative','error_rate']].rename(columns={'group':'分组','count':'邮件数','false_positive':'正常误拦','false_negative':'垃圾漏判','error_rate':'错误率'})
    left,right=st.columns([1.25,1],gap='medium')
    with left:
        event=st.dataframe(shown,hide_index=True,width='stretch',on_select='rerun',selection_mode='single-row',key='slice_table',column_config={'错误率':st.column_config.NumberColumn(format='percent')})
        st.caption('点击一行查看邮件。长度边界来自训练集；小样本分组仅供观察。')
    with right:
        index=event.selection.rows[0] if event.selection.rows else 0
        group=table.iloc[min(index,len(table)-1)].group;cases=evaluated.loc[groups.eq(group)].sort_values(['wrong','score_spam'],ascending=[False,False]).set_index('doc_id',drop=False)
        selected=st.selectbox('该组邮件（错分优先）',cases.doc_id.tolist(),format_func=lambda x:('错分 · ' if bool(cases.loc[x,'wrong']) else '正确 · ')+str(cases.loc[x,'subject_clean'])[:45],key='case_pick')
        case=cases.loc[selected]
        st.caption(f'参考标签：{"垃圾" if case.label else "正常"} · 模型：{"垃圾" if case.prediction else "正常"} · 垃圾分数 {case.score_spam:.6g}')
        st.button('在实验室查看这封邮件',on_click=open_mail,args=(case.source_id,selected),key='open_case',width='stretch')
    st.subheader('垃圾分数与实际比例是否一致？');bins,quality=reliability(evaluated)
    left,right=st.columns([1.25,1],gap='medium')
    with left:
        line=alt.Chart(pd.DataFrame({'x':[0.,1.],'y':[0.,1.]})).mark_line(color='#b6bdd0',strokeDash=[5,5]).encode(x='x:Q',y='y:Q')
        points=alt.Chart(bins.loc[bins['count'].gt(0)]).mark_line(point=True,color=PURPLE).encode(x=alt.X('mean_probability:Q',title='平均垃圾概率',scale=alt.Scale(domain=[0,1])),
            y=alt.Y('spam_fraction:Q',title='实际垃圾占比',scale=alt.Scale(domain=[0,1])),tooltip=[alt.Tooltip('count:Q',title='区间邮件数'),alt.Tooltip('mean_probability:Q',title='平均预测',format='.3f'),alt.Tooltip('spam_fraction:Q',title='实际比例',format='.3f')])
        st.altair_chart((line+points).properties(height=250,background='transparent').configure_view(stroke=None),width='stretch',theme=None)
    with right:
        st.metric('Brier 分数',f'{quality["brier"]:.5f}')
        st.caption('越低表示概率预测整体更好；曲线越接近虚线，区间预测与实际比例越一致。')
        st.caption(f'固定 10 个区间 · 共 {quality["n"]:,} 封 · ECE {quality["ece"]:.4f}')
        st.caption('分数接近 1 表示模型强烈倾向垃圾类别，并不保证每封邮件正确。')
