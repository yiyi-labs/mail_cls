"""Readable MIME preview with explicit source damage and frozen-input comparison."""
import streamlit as st
from mail_display import readable_mail, ENCODINGS
from mail_catalog import mask_contacts
from theme import mail_card


def preview_mail(raw,record,identity):
    options,privacy=st.columns([1,1])
    encoding=options.selectbox('阅读编码',list(ENCODINGS),key='display_encoding',help='仅改变阅读视图，不改变模型输入。')
    with privacy:
        st.caption('展示隐私')
        masked=st.checkbox('隐藏邮箱与手机号',value=True,key='mask_contacts')
    display=readable_mail(raw,encoding)
    show=mask_contacts if masked else str
    mail_card(show(display['subject']),show(display['body']))
    damage=display['invalid_bytes']+display['source_replacements']+display['nontext_characters']
    if damage:
        st.warning(f'原文有 {damage} 处损坏或非文本字符，已标记；可切换阅读编码核对。')
    with st.expander('原文与分类输入'):
        st.caption('阅读编码：'+(' / '.join(display['encodings']) or '空文本')+'。编码和遮盖仅影响展示；缺失文字无法可靠还原。')
        st.markdown('**模型实际读取的文本**')
        st.text(show(record.lexical_text.iloc[0]))
        st.markdown('**按所选编码查看原始封装**')
        st.text(show(display['source']))
        st.download_button('下载这封原始邮件',raw,file_name=identity.replace('/','_')+'.eml',mime='message/rfc822',key='download_mail')
    return display
