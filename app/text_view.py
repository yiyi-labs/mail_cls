"""Bounded interactive text lens; all values originate from the saved pipeline."""
import json
import re
from pathlib import Path
import streamlit as st
from text_analysis import text_payload


def text_html(payload):
    data=json.dumps(payload,ensure_ascii=False,allow_nan=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    return Path(__file__).with_name('text_lens.html').read_text(encoding='utf-8').replace('__TEXT_DATA__',data)


@st.cache_resource
def lens_component():
    """Reuse the same renderer in a bidirectional, style-isolated component."""
    source=Path(__file__).with_name('text_lens.html').read_text(encoding='utf-8')
    css=re.search(r'<style>(.*?)</style>',source,re.S).group(1).replace('body{','.lens{')
    html=source.split('</style>',1)[1].split('<script',1)[0]
    js=re.findall(r'<script>(.*?)</script>',source,re.S)[0]
    js=js.replace("const data=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id);",
                  "const data=component.data,$=id=>component.parentElement.querySelector('#'+id);")
    js=js.replace("document.querySelectorAll('button,select')","component.parentElement.querySelectorAll('button,select')")
    js=js.replace('data.branches.forEach',"$('branch').replaceChildren();component.parentElement.querySelectorAll('button,select').forEach(n=>n.disabled=false);$('tokens').className='tokens';$('tokens').replaceChildren();$('stats').replaceChildren();$('detail').replaceChildren();$('analysis').replaceChildren();data.branches.forEach")
    js=js.replace("$('tokens').className='empty';", "$('rule').textContent='';$('footer').textContent='';$('page').textContent='';$('progress').style.width='0%';$('progress').parentElement.setAttribute('aria-valuenow','0');$('tokens').className='empty';")
    js=js.replace("$('tokens').querySelectorAll('button').forEach", "if(!t.special && t.end>t.start){const edit=node('button','用此片段做改写');edit.onclick=()=>{stop();component.setTriggerValue('span',{field:t.field,start:t.start,end:t.end,surface:t.surface,index:t.index})};box.append(edit)}$('tokens').querySelectorAll('button').forEach")
    js=js.replace('已与本次模型输入逐项核对','本次实际输入').replace('同一推理流程 · 预览','模型分词')
    js=js.replace("['词项','次数','IDF','向量值']","['词项','次数','IDF','向量值','分数贡献']")
    js=js.replace("[f.count,f.idf.toFixed(2),f.tfidf.toFixed(4)]","[f.count,f.idf.toFixed(2),f.tfidf.toFixed(4),(f.contribution>=0?'+':'')+f.contribution.toFixed(4)]")
    js=js.replace('分母是该区块全部命中项的 L2 范数。','分母是该区块全部命中项的 L2 范数。再乘模型权重 ${f.coefficient.toFixed(4)}，得到原始分数贡献 ${f.contribution.toFixed(4)}。')
    js=js.replace('向量值含 L2 归一化和区块权重；IDF 来自训练词表。','贡献为向量值 × 学到的权重：正值推向垃圾，负值推向正常；不是概率变化。')
    js=js.replace("window.addEventListener('pagehide',stop);",'')
    js=js.replace("window.addEventListener('resize',()=>{",'const resize=()=>{')
    js=js.replace('showPage()}});','showPage()}};window.addEventListener(\'resize\',resize);')
    return st.components.v2.component('mail_text_lens',html='<div class="lens">'+html+'</div>',css=css,
        js='export default function(component){'+js+";return ()=>{stop();window.removeEventListener('resize',resize)};}")


def render_text_lens(engine,record,chosen,trace):
    payload=text_payload(engine,record,chosen,trace)
    result=lens_component()(data=payload,key='text_lens',height='content',on_span_change=lambda:None)
    return result.span
