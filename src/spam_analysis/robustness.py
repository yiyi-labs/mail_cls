"""Fixed label-preserving perturbations, no mail link/network access."""
from email.message import EmailMessage
import html
import re
import numpy as np
import pandas as pd
from .parsing import parse_email,normalize,structure
from .metrics import metrics

RULES=["fullwidth","zero_width","html_wrapper","mime_base64","cjk_spaces","repeat_punctuation"]

def mutate_text(text,rule):
    if rule=="fullwidth": return "".join(chr(ord(c)+0xfee0) if 0x21<=ord(c)<=0x7e else c for c in text)
    if rule=="zero_width": return re.sub(r"([\u4e00-\u9fff])",lambda m:m.group()+"\u200b",text)
    if rule=="cjk_spaces":
        count=0
        def insert(m):
            nonlocal count
            count+=1
            return m.group()+(" " if count%7==0 else "")
        return re.sub(r"[\u4e00-\u9fff]",insert,text)
    if rule=="repeat_punctuation": return re.sub(r"[!?]",lambda m:m.group()*3,text)
    return text

def perturb(df,rule):
    rows=[]; applicable=[]
    for r in df.to_dict("records"):
        sub,body=r["subject_clean"],r["body_clean"]
        if rule in {"html_wrapper","mime_base64"}:
            m=EmailMessage(); m["Subject"]=sub.replace("\n"," ")
            if rule=="html_wrapper":
                m.set_content("<html><body>"+"<br>".join(html.escape(body).split("\n"))+"</body></html>",subtype="html",charset="utf-8")
            else: m.set_content(body,charset="utf-8",cte="base64")
            parsed=parse_email(m.as_bytes()); r.update(parsed); applied=True
        else:
            newsub=mutate_text(sub,rule); newbody=mutate_text(body,rule); applied=newsub!=sub or newbody!=body
            newsub=normalize(newsub); newbody=normalize(newbody)
            r.update(subject_clean=newsub,body_clean=newbody,lexical_text=(newsub+"\n"+newbody).strip(),semantic_text=(newsub+"\n"+newbody).strip())
            r.update(structure(newsub,newbody,r.get("actual_html_tag_count",0),r.get("actual_link_count",0)))
        rows.append(r); applicable.append(applied)
    return pd.DataFrame(rows),np.array(applicable)

def paired_metrics(y,before,after,threshold,applied):
    a=metrics(y,before,threshold); b=metrics(y,after,threshold)
    return dict(original_macro_f1=a["macro_f1"],perturbed_macro_f1=b["macro_f1"],delta_macro_f1=b["macro_f1"]-a["macro_f1"],
                applied_rows=int(applied.sum()),n=len(y),application_rate=float(applied.mean()),flip_rate=float(np.mean((before>=threshold)!=(after>=threshold))),
                fp_before=a["fp"],fp_after=b["fp"],fn_before=a["fn"],fn_after=b["fn"])
