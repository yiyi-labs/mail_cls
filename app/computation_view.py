"""Interactive replay driven exclusively by this inference's measured values."""
import json
from html import escape
from pathlib import Path
import streamlit as st
from model_catalog import short_name


def live_heatmap(event):
    frame=event['frame'];values=frame['values']
    scale=max(abs(v) for row in values for v in row) or 1
    cells=''.join(f'<span title="{v:.7g}" style="height:7px;background:{"#be794f" if v>=0 else "#6265d9"};opacity:{.15+.85*abs(v)/scale:.4f}"></span>' for row in values for v in row)
    return f'<div style="padding:16px;background:#f0f1fb;color:#20283f;border-radius:10px;font-size:12px"><b>{escape(event["title"])}</b><p>实测 RMS {frame["sample_rms"]:.5f}</p><div style="display:grid;grid-template-columns:repeat({len(values[0])},1fr);gap:2px">{cells}</div><small>当前帧色标 · 完整回放使用统一色标</small></div>'


def computation_html(trace):
    payload={'branches':[{'id':key,'name':short_name(key),**value} for key,value in trace['branches'].items()],
             'probability':trace['probability'],'threshold':trace['threshold']} if trace else {'branches':[]}
    # The payload is a script data block. A mail token must never close it.
    data=json.dumps(payload,ensure_ascii=False,allow_nan=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    return Path(__file__).with_name('computation.html').read_text(encoding='utf8').replace('__TRACE_JSON__',data)


def render_computation(trace):
    st.iframe(computation_html(trace),height=636,alt='本次推理真实计算：逐层激活热图或特征乘积累加，支持暂停、步进和数值检查')
