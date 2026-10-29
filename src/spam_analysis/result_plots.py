"""Read-only reporting figures from frozen saved predictions and tables."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.calibration import calibration_curve
from .plots import style,savefig
from .metrics import align_scores


def build():
    if not Path('artifacts/test_complete.json').exists(): raise RuntimeError('Formal evaluation is not complete')
    style(); plan=json.loads(Path('artifacts/evaluation_plan.json').read_text(encoding='utf-8'))
    all_results=pd.read_csv('artifacts/test_results.csv')
    formal=list(plan['registry'])
    if plan['main'] not in formal: formal.append(plan['main'])
    result=all_results.query("point=='research' and weighting=='per_mail'").set_index('model').loc[formal]
    n=int(result.n.iloc[0])
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    result[['fp','fn']].rename(columns={'fp':'正常误拦 FP','fn':'垃圾漏检 FN'}).plot.bar(ax=axes[0],color=['#2563eb','#ea580c'],rot=15)
    axes[0].set_ylabel('实际错误封数'); axes[0].set_xlabel('模型（各自冻结的研究阈值）')
    axes[0].set_title('错误数量，不放大微小小数差异')
    result.macro_f1.plot.bar(ax=axes[1],color='#0f766e',rot=15,ylim=(0,1))
    axes[1].set_ylabel('Macro-F1（0–1）'); axes[1].set_xlabel('模型')
    for i,value in enumerate(result.macro_f1): axes[1].text(i,value-.055,f'{value:.4f}',ha='center',color='white',fontsize=10)
    fig.suptitle(f'封存 Test 全量比较｜n={n:,}；主模型 {plan["main"]} 在 Policy 已选定')
    savefig('07_model_comparison')
    main=result.loc[plan['main']]; matrix=np.array([[main.tn,main.fp],[main.fn,main.tp]],dtype=int)
    plt.figure(figsize=(5.5,4.8)); plt.imshow(matrix,cmap='Blues',vmin=0)
    for i in range(2):
        for j in range(2): plt.text(j,i,f'{matrix[i,j]:,}',ha='center',va='center',fontsize=25,color='white' if matrix[i,j]>matrix.max()/2 else '#0f172a')
    plt.xticks([0,1],['预测正常 ham','预测垃圾 spam']); plt.yticks([0,1],['实际正常 ham','实际垃圾 spam'])
    plt.title(f'{plan["main"]}｜Test n={n:,}｜原始计数'); savefig('08_confusion')
    manifest=pd.read_parquet('artifacts/splits/split_manifest.parquet').query("split=='test'")
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for name in formal:
        p=align_scores(manifest[['doc_id']],pd.read_parquet(f'artifacts/predictions/test_{name}.parquet')).score_spam.to_numpy()
        precision,recall,_=precision_recall_curve(manifest.label,p)
        axes[0].plot(recall,precision,label=f'{name} AP={result.loc[name,"average_precision"]:.4f}')
        observed,predicted=calibration_curve(manifest.label,p,n_bins=10,strategy='quantile')
        axes[1].plot(predicted,observed,'o-',markersize=4,label=name)
    axes[0].set(xlabel='spam Recall',ylabel='spam Precision',xlim=(0,1),ylim=(0,1.02),title='PR 曲线；图例为 Average Precision')
    axes[1].plot([0,1],[0,1],'--',color='#64748b'); axes[1].set(xlabel='分箱平均预测概率',ylabel='分箱实际 spam 比例',title='独立校准后可靠性图（等频分箱）')
    for ax in axes: ax.legend(fontsize=8)
    fig.suptitle(f'Test 全量保存分数｜n={n:,}；未据此调整模型或阈值'); savefig('09_pr_calibration')
    table=pd.read_csv(f'artifacts/tables/threshold_{plan["main"]}.csv').sort_values('threshold')
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    axes[0].plot(table.threshold,table.macro_f1,label='Macro-F1'); axes[0].plot(table.threshold,table.spam_recall,label='spam Recall')
    axes[0].set(xlabel='阈值',ylabel='指标',title='Policy 阈值变化（不是 Test 调参）'); axes[0].legend()
    axes[1].plot(table.ham_fpr,table.spam_recall,color='#0f766e')
    axes[1].axvline(.005,color='#2563eb',linestyle='--',label='0.5%'); axes[1].axvline(.01,color='#ea580c',linestyle=':',label='1.0%')
    axes[1].set(xlabel='ham FPR',ylabel='spam Recall',title='经验误拦约束与召回的代价',xlim=(0,.03),ylim=(0,1.01)); axes[1].legend()
    fig.suptitle(f'{plan["main"]}｜Policy n={int(pd.read_csv("artifacts/split_summary.csv").set_index("split").loc["policy","rows"]):,}')
    savefig('10_policy_threshold')
    robustness=pd.read_csv('artifacts/robustness_test.csv')
    names={'fullwidth':'全角变体','zero_width':'零宽字符','html_wrapper':'HTML 封装','mime_base64':'Base64 封装','cjk_spaces':'中文间稀疏空格','repeat_punctuation':'标点重复'}
    robustness['名称']=robustness.rule.map(names)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    robustness.set_index('名称')[['fp_after','fn_after']].rename(columns={'fp_after':'正常误拦 FP','fn_after':'垃圾漏检 FN'}).plot.barh(ax=axes[0],color=['#2563eb','#ea580c'])
    axes[0].set(xlabel='扰动后实际错误封数',ylabel='',title='每次仍是相同的全部 Test 邮件')
    axes[0].legend(loc='upper center',bbox_to_anchor=(.5,-.16),ncol=2,fontsize=8)
    axes[1].barh(robustness['名称'],robustness.flip_rate*100,color='#0f766e'); axes[1].set(xlabel='预测翻转率（%）',title='同一邮件扰动前后配对')
    fig.suptitle(f'{plan["main"]}｜Test n={n:,}；变体不作为新增独立样本'); savefig('11_robustness')
    grid=pd.read_csv('artifacts/sparse_tune_results.csv').set_index('candidate')
    keys=['naive_1_plain','B2_1_plain','body_word_1_plain','char_1_plain','word_char_1_plain','S_1_plain']
    names=['朴素解析 LR','校验解析 LR','正文词 SVM','正文字 SVM','词+字 SVM','词+字+主题 SVM']
    sub=grid.loc[keys,['fp','fn']]; sub.index=names
    sub.rename(columns={'fp':'正常误拦 FP','fn':'垃圾漏检 FN'}).plot.barh(figsize=(9,5),color=['#2563eb','#ea580c'])
    plt.xlabel('Tune 错误封数'); plt.ylabel(''); plt.title(f'开发消融｜Tune n={int(grid.n.iloc[0]):,}｜C=1；前两行 LR，其余 SVM'); savefig('12_development_ablation')
    errors=pd.read_csv('artifacts/tables/test_errors_by_length_bin.csv')
    # CSV stores interval labels as text; order by their numerical lower edge,
    # not lexicographically (which would place 1848 before 228).
    errors=errors.assign(_lower=errors.length_bin.map(lambda value:float(value[1:].split(',')[0]))).sort_values('_lower')
    plt.figure(figsize=(8,4.5)); plt.bar(errors.length_bin,errors.errors,color='#ea580c')
    plt.xticks(rotation=15); plt.ylabel('错误封数'); plt.xlabel('Train 确定的正文长度分箱'); plt.title(f'{plan["main"]}｜Test 错误分布｜n={n:,}'); savefig('13_error_lengths')
    curve_path=Path('artifacts/macbert_learning_curves.csv')
    if curve_path.exists():
        curves=pd.read_csv(curve_path)
        fig,axes=plt.subplots(1,2,figsize=(12,4.8))
        for i,(lr,part) in enumerate(curves.groupby('lr',sort=True)):
            part=part.sort_values('epoch');color=['#2563eb','#ea580c'][i%2]
            axes[0].plot(part.epoch,part.training_loss,'o-',color=color,label=f'LR={lr:.0e}')
            axes[1].plot(part.epoch,part.fp,'o-',color=color,label=f'LR={lr:.0e} FP')
            axes[1].plot(part.epoch,part.fn,'s--',color=color,label=f'LR={lr:.0e} FN')
        for ax in axes:ax.set_xlabel('epoch');ax.set_xticks(sorted(curves.epoch.unique()));ax.set_ylim(bottom=0);ax.legend(fontsize=9)
        axes[0].set(ylabel='Train 平均交叉熵',title='训练损失（不等于泛化）')
        axes[1].set(ylabel='Tune 实际错误封数',title='同一 Tune 上选择 checkpoint')
        fig.suptitle(f'MacBERT 正式学习曲线｜Train {int(curves.train_rows.iloc[0]):,}；Tune {int(curves.n.iloc[0]):,}；不含冒烟测试')
        savefig('14_macbert_learning_curves')


if __name__=='__main__': build()
