"""Fresh read-only checks of all saved split identities and training evidence."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from spam_analysis.io import sha256, read_config
from spam_analysis.grouping import compact


def audit_isolation():
    plan=json.loads(Path('artifacts/evaluation_plan.json').read_text(encoding='utf8'))
    manifest=pd.read_parquet('artifacts/splits/split_manifest.parquet')
    records=pd.read_parquet('data/processed/records.parquet',columns=['doc_id','label','raw_sha256','subject_clean','body_clean'])
    checks=[]
    def check(name,passed,detail):checks.append({'检查':name,'通过':bool(passed),'证据':str(detail)})
    check('冻结划分身份',sha256('artifacts/splits/split_manifest.parquet')==plan['manifest_sha256'],plan['manifest_sha256'])
    check('冻结派生数据身份',sha256('data/processed/records.parquet')==plan['processed_sha256'],plan['processed_sha256'])
    check('原始 ZIP 身份',sha256(read_config('configs/data.yaml')['zip_path'])==plan['zip_sha256'],plan['zip_sha256'])
    check('每封邮件唯一归属',manifest.doc_id.is_unique and records.doc_id.is_unique and set(manifest.doc_id)==set(records.doc_id),f'{len(manifest):,} 封，五个集合覆盖完整')
    joined=records.merge(manifest,on='doc_id',validate='one_to_one',suffixes=('','_manifest'))
    check('标签一致',joined.label.equals(joined.label_manifest),'派生数据标签逐行对照划分标签')
    check('集合名称',set(manifest.split)=={'train','tune','calibrate','policy','test'},manifest.split.value_counts().to_dict())
    overlaps=int(manifest.groupby('template_group_id').split.nunique().gt(1).sum())
    check('内容组跨集合',overlaps==0,f'{overlaps} 个跨集合内容组')
    for key,title in [('exact_content_id','精确内容组'),('raw_sha256','原文字节重复')]:
        count=int(joined.groupby(key).split.nunique().gt(1).sum())
        check(title,count==0,f'{count} 个跨集合重复项')
    # Independently recompute exact content identities, rather than trusting group IDs.
    ids=[hashlib.sha256((compact(r.subject_clean)+'\n'+compact(r.body_clean)).encode()).hexdigest()
         if (r.subject_clean+r.body_clean).strip() else 'EMPTY:'+r.doc_id for r in joined.itertuples()]
    count=int(joined.assign(recomputed=ids).groupby('recomputed').split.nunique().gt(1).sum())
    check('重新计算文本重复',count==0,f'{count} 个跨集合主题+正文重复项；空邮件独立')
    min_chars=read_config('configs/experiment.yaml')['body_group_min_chars']
    bodies=joined.body_clean.map(compact);long=joined.loc[bodies.str.len()>=min_chars].assign(body_key=bodies[bodies.str.len()>=min_chars])
    count=int(long.groupby('body_key').split.nunique().gt(1).sum())
    check('重新计算长正文重复',count==0,f'正文 ≥ {min_chars} 字符，{count} 个跨集合重复项')
    edges=pd.read_parquet('artifacts/splits/near_edges.parquet');lookup=manifest.set_index('doc_id').split
    left=edges.left.map(lookup);right=edges.right.map(lookup)
    cross=int((left.ne(right)|left.isna()|right.isna()).sum())
    check('已发现近重复边隔离',cross==0,f'{len(edges):,} 条边，{cross} 条跨集合或无归属')
    train=manifest[manifest.split.eq('train')];test=manifest[manifest.split.eq('test')]
    check('Train/Test 邮件交集',not(set(train.doc_id)&set(test.doc_id)),f'Train {len(train):,}，Test {len(test):,}，交集 {len(set(train.doc_id)&set(test.doc_id))}')
    for name,entry in plan['registry'].items():
        intact=sha256(entry['path'])==entry['model_sha256'] and sha256(entry['calibrator_path'])==entry['calibrator_sha256']
        if 'checkpoint' in entry:
            intact= intact and all(sha256(Path(entry['checkpoint']['folder'])/f)==h for f,h in entry['checkpoint']['files'].items())
        check(name+' 模型与校准器身份',intact,'逐文件 SHA-256 与冻结计划一致')
        check(name+' 拟合用途',entry.get('fit_split')=='train' and entry.get('calibration_split')=='calibrate','基础模型 train；分数校准 calibrate')
    nb=joblib.load(plan['registry']['B1']['path'])
    check('朴素贝叶斯真实拟合样本数',np.array_equal(nb.estimator.class_count_,np.bincount(train.label,minlength=2)),f'模型 class_count_={nb.estimator.class_count_.tolist()}；Train 标签计数={np.bincount(train.label,minlength=2).tolist()}')
    qwen=json.loads(Path('artifacts/embeddings/train/qwen.json').read_text(encoding='utf8'))
    check('Qwen 训练向量的邮件 ID',len(qwen['doc_ids'])==len(train) and set(qwen['doc_ids'])==set(train.doc_id),'逐封核对 Train 缓存 ID，且与 Test 无交集')
    curves=pd.read_csv('artifacts/macbert_learning_curves.csv')
    check('MacBERT 各轮训练计数',curves.train_rows.eq(len(train)).all() and curves.train_groups.eq(train.template_group_id.nunique()).all(),f'{len(curves)} 轮记录，每轮 {len(train):,} 封 / {train.template_group_id.nunique():,} 组')
    critical=['io','features','sparse_models','sparse_ablation','sparse_validation','sparse_convergence','semantic_models','macbert','embeddings','calibration','policy','parsing','text_encoding','splitting','inference']
    for module in critical:
        path=f'src/spam_analysis/{module}.py'
        check(module+' 源码身份',sha256(path)==plan['source_hashes'][path],'与正式评价冻结时的源码一致')
    return {'checked_at':datetime.now(timezone.utc).isoformat(),'passed':all(c['通过'] for c in checks),
            'rows':len(manifest),'train_rows':len(train),'test_rows':len(test),'checks':checks,
            'scope':'只读审计；没有训练、选择模型、修改阈值或重新生成 Test 预测。',
            'limitation':'核对的是本项目划分、已发现的近重复关系、冻结源码与训练产物；LSH 不保证发现所有语义近重复，也无法审计公开预训练语料。历史运行未保存每一步梯度所用邮件 ID，不能把现有证据说成逐步训练溯源。'}
