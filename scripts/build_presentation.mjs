// Eight editable course slides; every result comes from the published run.
import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const ROOT=process.cwd();
const RUNTIME='C:/Users/30243/.cache/codex-runtimes/codex-primary-runtime/dependencies';
process.env.RUNTIME_NODE_MODULES=path.join(RUNTIME,'node/node_modules');
const SKILL='C:/Users/30243/.codex/plugins/cache/openai-primary-runtime/presentations/26.1013.11914/skills/presentations';
const {Presentation,PresentationFile}=await import(pathToFileURL(path.join(RUNTIME,'node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs')).href);
const {finalizePresentation}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')).href);
const d=JSON.parse(await fs.readFile(path.join(ROOT,'artifacts/presentation_data.json'),'utf8'));
if(!d.run_id || d.run_id==='frozen-v1' || !d.plan_sha256)throw Error('Publish and build the current report first.');
const p=Presentation.create({slideSize:{width:1280,height:720}});
const font='Microsoft YaHei';p.theme.defaultFont=font;
const C={bg:'#F5F6FA',ink:'#222A40',accent:'#6265D9',muted:'#657086',clay:'#BE794F',white:'#FFFFFF',line:'#DEE2EC'};
const num=n=>Number(n).toLocaleString('en-US'),pct=n=>(100*Number(n)).toFixed(2)+'%';
const main=d.results.find(r=>r.model===d.main),speed=d.benchmark.find(r=>r.model===d.main);
const labels={'B1':'NB','B2':'LR','S_char13':'SVM','E1':'Qwen + LR','E2':'MacBERT'};
function txt(s,t,x,y,w,h,size=26,color=C.ink,bold=false){
  const a=s.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  a.text=String(t);a.text.style={fontSize:size,typeface:font,color,bold,verticalAlignment:'middle',insets:0,wrap:'square'};return a;
}
function page(k,title,subtitle){
  const s=p.slides.add();s.background.fill=C.bg;
  txt(s,title,80,42,1120,65,42,C.ink,true);txt(s,subtitle,80,114,1120,46,22,C.muted);
  txt(s,'题目六 · 邮件分类与 Python 数据分析',80,674,800,26,16,C.muted);
  txt(s,String(k).padStart(2,'0')+' / 08',1116,674,90,26,16,C.muted);return s;
}
function table(s,values,x,y,w,h,widths,size=23){
  const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,values,columnWidths:widths});
  for(let r=0;r<values.length;r++)for(let c=0;c<values[0].length;c++){
    const cell=t.getCell(r,c);cell.fill=r===0?C.ink:(r%2?C.white:'#EBEDF6');
    cell.text.style={fontSize:size,typeface:font,color:r===0?C.white:C.ink,bold:r===0,insets:12,verticalAlignment:'middle'};
  }t.borders.assign({style:'solid',fill:C.bg,width:2});return t;
}
function bars(s,categories,series,x,y,w,h,title,format='0'){
  return s.charts.add('bar',{position:{left:x,top:y,width:w,height:h},title,
    titleTextStyle:{fontSize:23,fill:C.ink,bold:true},categories,series,hasLegend:series.length>1,
    legend:{position:'bottom',textStyle:{fontSize:19,fill:C.muted}},barOptions:{direction:'column',grouping:'clustered',gapWidth:85},chartFill:C.white,plotAreaFill:C.white,
    xAxis:{textStyle:{fontSize:19,fill:C.ink},line:{fill:C.line,width:1}},
    yAxis:{min:0,numberFormatCode:format,textStyle:{fontSize:18,fill:C.muted},majorGridlines:{fill:C.line,width:1,style:'solid'}},
    dataLabels:{showValue:true,position:'outEnd',numberFormatCode:format,textStyle:{fontSize:19,fill:C.ink}}});
}
const notes=[];
function note(s,seconds,text,sources){s.speakerNotes.text=`建议 ${seconds} 秒。\n${text}\n\n依据：${sources}`;notes.push({seconds,text,sources});}
{
 const s=page(1,'邮件分类与 Python 数据分析','从原始邮件，到统计特征，再到可解释的分类结果');
 txt(s,'用数据说明\n模型为何这样判断',80,207,800,155,57,C.ink,true);
 const stats=[[num(d.rows),'统一邮件数据'],[num(d.train_rows),'仅用于拟合的训练邮件'],[pct(main.accuracy),'主模型留出汇总准确率']];
 stats.forEach(([value,label],i)=>{txt(s,value,80+i*384,421,355,72,45,C.accent,true);txt(s,label,80+i*384,501,355,42,24,C.muted);});
 txt(s,'当前主模型：'+d.main_name,80,597,1120,43,29,C.ink,true);
 note(s,25,`课程题目是邮件分类，重点是用 Python 完成数据清洗、统计分析、特征构建和结果解释。统一语料有 ${d.rows} 封，主模型 ${d.main_name} 由开发数据选择。后面的成绩来自相同的 ${d.test_rows} 封留出邮件，并保留来源差异。`,'corpus.json；evaluation_plan.json；results_combined.csv');
}
{
 const s=page(2,'数据统一，来源与用途仍然可区分','官方语料 → 校验与解析 → 去重分组 → 统一邮件表');
 bars(s,d.sources.map(r=>r.name),[{name:'正常',values:d.sources.map(r=>r.ham),fill:C.accent},{name:'垃圾',values:d.sources.map(r=>r.spam),fill:C.clay}],80,194,755,388,'各来源的真实邮件数');
 txt(s,'中文与拉丁文本',882,207,310,42,29,C.accent,true);
 txt(s,'按正文字符比例统计\n来源不直接等于语言\n类内比例避免数量误导',882,267,315,130,24);
 txt(s,'训练与测试分开',882,425,310,42,29,C.accent,true);
 txt(s,'同一已识别模板组\n只进入一个用途集合',882,485,310,80,24);
 txt(s,'历史公开语料主要来自 1999–2006 年；现代中文邮箱仍需独立验证。',80,610,1120,40,23,C.muted);
 note(s,40,`TREC06C 保留原有划分，新增 Apache SpamAssassin 与 AUEB Enron-Spam。原始包统一放在 data/raw，课程分析只读一张 corpus 表。新增数据先检查精确重复和已知近重复，再按组划分。合并存储并不代表混合训练与测试。拉丁字母为主不一定全部是英语，语料年代也限制了结论。`,'NIST TREC15 overview；Apache publiccorpus/readme.html；AUEB enron-spam/readme.txt；data/README.md');
}
{
 const s=page(3,'从词频观察到可计算的特征','类内文档频率：某词出现过的邮件数 ÷ 该类邮件总数');
 bars(s,d.keywords.map(r=>r.term),[{name:'正常类内比例',values:d.keywords.map(r=>+r.ham_rate.toFixed(6)),fill:C.accent},{name:'垃圾类内比例',values:d.keywords.map(r=>+r.spam_rate.toFixed(6)),fill:C.clay}],80,194,755,411,'训练集中的区分词','0%');
 txt(s,'pandas',882,216,310,40,31,C.accent,true);txt(s,'分组、交叉统计\n按编号连接预测与原文',882,272,315,85,24);
 txt(s,'NumPy',882,401,310,40,31,C.accent,true);txt(s,'计数 → DF → IDF\nTF-IDF 与 sklearn 核对',882,457,315,90,24);
 txt(s,'词频与词表只从 Train 学习；词语相关性不等于单封邮件的因果解释。',80,614,1120,34,23,C.muted);
 note(s,40,'这里的词来自统一训练集，不是手工挑选的营销词。按正常与垃圾各自的邮件数归一化，避免垃圾邮件更多造成所有词都显得偏垃圾。Notebook 用 NumPy 计算一个小语料的 TF-IDF 并核对 sklearn，真实分析则使用 pandas 聚合和严格编号连接。模型还会使用字符片段和主题，补充词分割的局限。','corpus_terms.parquet；Notebook 02；analysis/distributions.py');
}
{
 const s=page(4,'五种方法，各自解决什么问题','比较完整方案；参数量、输入长度与训练方式并不相同');
 table(s,[['方法','邮件如何表示','哪些部分学习'],['朴素贝叶斯','词 TF-IDF','每类的词统计'],['逻辑回归','相同词 TF-IDF','词权重与偏置'],['多粒度 SVM','词 + 字符 1–3 元组 + 主题','最大间隔分类边界'],['Qwen + LR','1024 维语义向量','冻结 0.6B 编码器，训练分类头'],['MacBERT','主题 + 正文首尾，共 512 词元','编码器与二分类头一起微调']],80,193,1120,365,[230,420,470],23);
 txt(s,'Qwen 输入上限 1024 词元；Embedding 是把文本编码为可比较的数值向量。',80,586,1120,50,23,C.muted);
 note(s,40,'朴素贝叶斯与逻辑回归共享词 TF-IDF，形成清楚的分类器对照。SVM 使用词、字符和主题的多粒度特征。Qwen3-Embedding-0.6B 使用官方预训练编码器提取向量，只训练逻辑回归头。MacBERT 约一亿参数，使用邮件标签进行全量微调。选择 0.6B 是本地资源与语义能力的折中，并不是为了与 MacBERT 做等参数比较。','model_catalog.py；benchmark.csv；evaluation_plan.json');
}
{
 const s=page(5,'用错误封数看清模型差异',`相同的 ${num(d.test_rows)} 封留出邮件；阈值在 Policy 中预先确定`);
 bars(s,d.results.map(r=>labels[r.model]),[{name:'正常误拦',values:d.results.map(r=>r.fp),fill:C.clay},{name:'垃圾漏判',values:d.results.map(r=>r.fn),fill:C.accent}],80,195,755,410,'实际分类错误');
 txt(s,'当前主模型',882,218,310,38,25,C.muted);txt(s,d.main_name,882,269,310,82,34,C.accent,true);
 txt(s,pct(main.accuracy),882,383,310,63,47,C.ink,true);txt(s,'准确率',882,453,310,35,23,C.muted);
 txt(s,`单封推理中位数 ${Number(speed.median_ms).toFixed(1)} ms`,882,528,315,62,24);
 txt(s,'整体分数之外，同时比较正常误拦、垃圾漏判、耗时与显存。',80,617,1120,34,23,C.muted);
 note(s,40,`这张图使用相同邮件上的真实错误封数。当前主模型正常误拦 ${main.fp} 封、垃圾漏判 ${main.fn} 封。模型及阈值由 Policy 选择，同时报告来源分组的得失。当前成绩是固定留出集的回顾评估，不能称为一次新的独立确认。性能在同一机器、相同邮件上测量，预热后重复三次，模型加载另计。`,'results_combined.csv；selection.json；benchmark.csv');
}
{
 const s=page(6,'分析错误落在哪里','来源、文字类型、长度、编码质量与实际截断比例');
 table(s,[['邮件来源','留出邮件数','错误率','正常误拦 / 垃圾漏判'],...d.by_source.map(r=>[r.group,num(r.count),pct(r.error_rate),`${r.false_positive} / ${r.false_negative}`])],80,200,1120,252,[340,220,220,340],24);
 txt(s,'同来源的新邮件',80,494,500,45,31,C.accent,true);txt(s,'新增来源按模板组留出；\n训练已见过该来源的其他邮件。',80,551,510,82,25);
 txt(s,'结论的适用范围',685,494,500,45,31,C.accent,true);txt(s,'固定留出集回顾评估\n现代中文与全新来源仍待检验',685,551,510,82,25);
 note(s,40,'数据分析需要从整体分数回到错误分布。网页按来源、文字类型、长度和损坏情况分组，再打开具体邮件查看；语义模型还显示真实截断比例。训练见过同来源的其他邮件。首次发布过严地拒绝全部新模型，修订前已观察旧模型测试成绩，因此这里是回顾评估。修订仅用开发数据选择模型和阈值，模型拟合没有使用 Test。','results_*.csv；reports/next_stage.md；analysis/tables.py');
}
{
 const s=page(7,'高垃圾分数，是否真的可信？','比较区间平均预测概率与实际垃圾比例；空区间不补造数据');
 const bins=d.reliability.filter(r=>r.count>0);
 s.charts.add('scatter',{position:{left:80,top:196,width:755,height:410},title:'主模型概率可靠性',titleTextStyle:{fontSize:23,fill:C.ink,bold:true},
   series:[{name:'理想情况',xValues:[0,1],values:[0,1],line:{fill:C.line,width:2,style:'dash'},marker:{symbol:'none'}},
   {name:'实际区间',xValues:bins.map(r=>+r.mean_probability.toFixed(8)),values:bins.map(r=>+r.spam_fraction.toFixed(8)),line:{fill:C.accent,width:3},marker:{symbol:'circle',size:8,fill:C.accent}}],
   scatterOptions:{style:'lineWithMarkers'},hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:19,fill:C.muted}},chartFill:C.white,plotAreaFill:C.white,
   xAxis:{title:'平均预测垃圾概率',min:0,max:1,numberFormatCode:'0%',textStyle:{fontSize:18,fill:C.muted}},
   yAxis:{title:'实际垃圾比例',min:0,max:1,numberFormatCode:'0%',textStyle:{fontSize:18,fill:C.muted},majorGridlines:{fill:C.line,width:1}}});
 txt(s,'Brier',882,229,310,36,25,C.muted);txt(s,Number(d.calibration.brier).toFixed(5),882,278,310,64,46,C.accent,true);
 txt(s,'越低，整体概率误差越小',882,355,310,47,23,C.muted);txt(s,'分数 1 ≠ 必然正确',882,458,310,55,29,C.ink,true);
 txt(s,'训练邮件高分不能证明泛化；独立校准也无法保证每封邮件都可靠。',80,616,1120,36,23,C.muted);
 note(s,40,'垃圾分数接近一表示模型强烈倾向垃圾类别，可能还包含数值饱和，并不等于百分之百正确。可靠性图将概率划成十个区间，对照平均预测与真实垃圾比例，点偏离对角线说明区间内存在偏差。Brier 使用每封邮件的概率误差计算。校准数据与模型训练数据分开，但历史数据上的校准仍不能保证现代邮箱可靠。','reliability_combined_*.csv；Notebook 04；analysis/tables.py');
}
{
 const s=page(8,'把分析过程连回具体邮件','文字或文件 → 真实词元与特征 → 模型分数 → 修改片段再比较');
 const steps=[['读取','粘贴邮件或上传\nEML / TXT / CSV'],['解释','词元、TF-IDF、线性贡献\n实际张量与计算回放'],['比较','同一邮件切换方法\n修改片段后重新推理']];
 steps.forEach(([title,body],i)=>{txt(s,String(i+1).padStart(2,'0'),80+i*384,216,350,60,44,C.accent,true);txt(s,title,80+i*384,294,350,50,34,C.ink,true);txt(s,body,80+i*384,367,350,119,25);});
 txt(s,'课程主线：清洗、统计、特征、建模、评价与解释',80,552,1120,49,32,C.ink,true);
 txt(s,'动画回放真实计算；断点保存训练状态，关机后可继续。',80,613,1120,35,24,C.muted);
 note(s,35,'最后在网页输入一封合成邮件，展示实际词元与特征贡献，再替换一个片段重新分类。中间张量来自真实推理，动画用于理解计算顺序，不把动画当作重新训练或因果解释。上传文件支持本地批量分类。所有课程材料共用分析函数；GPU 训练保存模型、优化器、随机状态和批次位置，电脑关闭后可恢复。','app/app.py；text_analysis.py；experiments/checkpoint.py；四份 Notebook');
}
const build=path.join(ROOT,'artifacts/private/presentation_build');await fs.mkdir(build,{recursive:true});
const previews=path.join(ROOT,'artifacts/figures/slides');await fs.mkdir(previews,{recursive:true});
for(let i=0;i<p.slides.items.length;i++){
 const s=p.slides.items[i];const img=await p.export({slide:s,format:'png',scale:1});
 await fs.writeFile(path.join(previews,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await img.arrayBuffer()));
 const layout=await s.export({format:'layout'});await fs.writeFile(path.join(build,`slide-${i+1}.layout.json`),await layout.text());console.log('Rendered slide',i+1);
}
const candidate=path.join(build,'candidate.pptx');await(await PresentationFile.exportPptx(p)).save(candidate);
const versionedFinal=path.join(build,`final-${Date.now()}.pptx`);
await finalizePresentation({workspaceDir:ROOT,candidatePath:candidate,finalPath:versionedFinal,
 pythonExecutable:path.join(RUNTIME,'python/python.exe'),integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),
 explicitTotalSlideCount:8,layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit','--require-native-table-slide','4','--require-native-table-slide','6'],
 requiredNativeTableOwnerSlides:[4,6],requiredNativeChartOwnerSlides:[2,3,5,7],materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:[font]},verifyArtifactToolImport:true,receiptPath:path.join(ROOT,'artifacts/private/presentation_validation.json')});
const finalPath=path.join(ROOT,'reports/Topic6_课程汇报.pptx');await fs.copyFile(versionedFinal,finalPath);
let elapsed=0;const script=['# 五分钟视频讲稿与录制顺序','可录制素材；未声称已录制。实际姓名与贡献由成员填写。'];
for(let i=0;i<notes.length;i++){const n=notes[i],stamp=x=>`${Math.floor(x/60)}:${String(x%60).padStart(2,'0')}`;
 script.push(`\n## ${stamp(elapsed)}–${stamp(elapsed+n.seconds)}｜第 ${i+1} 页\n\n${n.text}\n\n对应依据：${n.sources}`);elapsed+=n.seconds;}
script.push('\n## 问答准备\n\n1. 为什么存储合并后仍要保留数据来源和用途？\n2. NumPy 与 pandas 各承担哪些计算？\n3. NB 与 LR 的控制变量是什么？SVM 用了哪些特征？\n4. Qwen 与 MacBERT 为什么不是等参数对照？\n5. 训练集高分为什么不能证明效果？\n6. 分组留出与未见来源测试有什么区别？\n7. 断点包含哪些状态，如何防止测试参与训练？');
await fs.writeFile(path.join(ROOT,'reports/五分钟视频讲稿.md'),script.join('\n'),'utf8');
console.log('Validated presentation saved:',finalPath);
