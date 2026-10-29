"""Generate explicitly synthetic local .eml inputs for the upload demo."""
from email.message import EmailMessage
from email.policy import SMTP
from pathlib import Path


def main():
    root=Path(__file__).resolve().parent
    examples={
        'course_notice.eml':('课程项目答疑时间','同学们好，本周四下午安排课程项目答疑。请准备数据分析代码、实验记录和问题。'),
        'marketing_demo.eml':('限时优惠推广','您好！本公司现推出限时优惠服务，批量订购可享额外折扣。请回复邮件咨询套餐和报价。此邮件为未经请求的推广演示文本。'),
    }
    for filename,(subject,body) in examples.items():
        mail=EmailMessage(policy=SMTP)
        mail['Subject']=subject
        mail['From']='demo@example.invalid'
        mail['To']='student@example.invalid'
        mail['X-Demo-Source']='Synthetic course example; not part of TREC06C'
        mail.set_content(body,charset='utf-8',cte='base64')
        (root/filename).write_bytes(mail.as_bytes())
        print(filename)


if __name__=='__main__':main()
