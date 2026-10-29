"""Shared theme and escaped presentation primitives."""
from pathlib import Path
import streamlit as st
from html import escape
from score_display import score_text

CSS="<style>"+Path(__file__).with_name("styles.css").read_text(encoding="utf-8")+"</style>"

MAIL_ICON='<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.3" aria-hidden="true"><rect x="3" y="5" width="18" height="14" rx="3"/><path d="m4 7 8 6 8-6"/></svg>'

def apply_theme():
    st.markdown(CSS,unsafe_allow_html=True)
    st.html(f'<div class="brand"><div class="brand-name">{MAIL_ICON}邮件研判实验室</div><div class="brand-meta">PYTHON · DATA ANALYSIS</div></div>')

def hero(kicker,title,subtitle):
    st.html(f'<section class="hero"><div class="eyebrow">{escape(kicker)}</div><h1>{escape(title)}</h1><p>{escape(subtitle)}</p></section>')

def section(kicker,title):
    st.html(f'<div class="section-label">{escape(kicker)}</div><div class="section-title">{escape(title)}</div>')

def note(title,body):
    st.html(f'<div class="note-card"><h3>{escape(title)}</h3><p>{escape(body)}</p></div>')

def mail_card(subject,body):
    st.html(f'<article class="read-mail"><h3>{escape(subject or "（无主题）")}</h3><div class="mail-body">{escape(body or "（正文为空）")}</div></article>')

def result_card(trace):
    value=trace['probability'];threshold=trace['threshold']
    st.html(f'<section class="result {"spam" if trace["prediction"] else ""}"><div class="result-label">分类结论</div><strong>{"垃圾邮件" if trace["prediction"] else "正常邮件"}</strong><div class="result-values"><div><span>垃圾分数</span><b>{escape(score_text(value))}</b></div><div><span>判定阈值</span><b>{escape(score_text(threshold))}</b></div></div><div class="result-meter" aria-label="分数与判定阈值"><i style="width:{value*100:.5f}%"></i><em style="left:{min(99.5,threshold*100):.5f}%"></em></div></section>')

def empty_result():
    st.html(f'<div class="empty-result">{MAIL_ICON}<p><strong>等待一封邮件的答案</strong></p><p>分类后查看结论与真实计算过程</p></div>')

def split_strip(frame):
    labels=[('train','训练','更新参数'),('tune','调参','选择候选'),('calibrate','校准','校准分数'),('policy','策略','选择阈值'),('test','测试','最终评估')]
    st.html('<div class="split-strip">'+''.join(f'<div><span>{title}</span><b>{int(frame.loc[key,"rows"]):,}</b><small>{purpose}</small></div>' for key,title,purpose in labels)+'</div>')
