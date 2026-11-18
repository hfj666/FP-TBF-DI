"""Promote the pre-specified DI candidate only after its classification audit."""
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from paper.audit_revision_evidence import metrics,paired

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/paper_c_joint_reduction'


def metric(y,z):
    r=metrics(y,z);pred=z>=0;pos=y==1
    r.update(tp=int((pred&pos).sum()),fp=int((pred&~pos).sum()),
             tn=int((~pred&~pos).sum()),fn=int((~pred&pos).sum()))
    r['precision']=r['tp']/(r['tp']+r['fp']) if r['tp']+r['fp'] else 0.
    return r


def aggregate(rows,field):
    groups=defaultdict(list)
    for r in rows:groups[int(r['seed']),int(r['batch'])].append(r)
    configs=[]
    for (seed,batch),values in sorted(groups.items()):
        y=np.array([int(r['label']) for r in values]);z=np.array([field(r) for r in values])
        configs.append(dict(seed=seed,batch=batch,**metric(y,z)))
    means={k:float(np.mean([r[k] for r in configs])) for k in
           ('accuracy','precision','recall','specificity','f1','balanced_accuracy','auc')}
    y=np.array([int(r['label']) for r in rows]);z=np.array([field(r) for r in rows])
    return dict(configurations=configs,means=means,pooled=metric(y,z))


def table(title,summary,names):
    lines=[f'**{title}**','', '|方法|准确率/%|敏感度/%|特异度/%|F1/%|BA/%|ROC-AUC|',
           '|---|---:|---:|---:|---:|---:|---:|']
    for key,label in names:
        v=summary[key]['means']
        nums=[f'{100*v[k]:.2f}' for k in ('accuracy','recall','specificity','f1','balanced_accuracy')]+[f'{v["auc"]:.4f}']
        lines.append('|'+label+'|'+'|'.join(nums)+'|')
    return '\n'.join(lines)


def main():
    summary=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    assert summary['decision']['adopt_simplified']
    assert summary['decision']['all_reported_fusion_labels_identical']
    assert all(hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==v for k,v in manifest['sources'].items())
    with (OUT/'predictions.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    checks=json.loads((OUT/'checks.json').read_text(encoding='utf-8'))
    definitions={k:(lambda r,key=k:float(r[key])) for k in ('base','A','B','logistic','simple_c','full_c','simple_fusion','full_fusion')}
    definitions.update(coverage_A=lambda r:(float(r['base'])+float(r['A']))/2,
                       AB=lambda r:(float(r['A'])+float(r['B']))/2,
                       AC=lambda r:(float(r['A'])+float(r['simple_c']))/2,
                       BC=lambda r:(float(r['B'])+float(r['simple_c']))/2,
                       D_fusion=lambda r:(float(r['A'])+float(r['B'])+float(r['after_d']))/3)
    updated={}
    for evaluation in ('stream','holdout_300','holdout_500','holdout_700','holdout_849'):
        selected=[r for r in rows if r['evaluation']==evaluation]
        updated[evaluation]={k:aggregate(selected,fn) for k,fn in definitions.items()}
    (OUT/'updated_method_metrics.json').write_text(json.dumps(updated,ensure_ascii=False,indent=2),encoding='utf-8')
    exported=[]
    for ev,arms in updated.items():
        for name,result in arms.items():
            exported.extend(dict(evaluation=ev,method=name,**r) for r in result['configurations'])
    with (OUT/'configuration_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(exported[0]));writer.writeheader();writer.writerows(exported)

    joint=['**表14 原五层C路与联合删减版的融合比较（六配置均值）**','',
           '|评价|原版准确率/%|精简准确率/%|原版AUC|精简AUC|融合标签变化/次|C路净增加正确/次|',
           '|---|---:|---:|---:|---:|---:|---:|']
    names={'stream':'逐批','holdout_300':'留出：300条','holdout_500':'留出：500条','holdout_700':'留出：700条','holdout_849':'留出：849条'}
    for ev,label in names.items():
        v=summary[ev];a=v['full_fusion']['means'];b=v['simple_fusion']['means']
        joint.append(f'|{label}|{100*a["accuracy"]:.2f}|{100*b["accuracy"]:.2f}|{a["auc"]:.6f}|{b["auc"]:.6f}|{v["comparisons"]["fusion"]["label_disagreements"]}|{v["comparisons"]["branch"]["net"]:+d}|')
    joint='\n'.join(joint)
    report='''# C路P/E/U联合删减实验与精简决定

## 1. 结论与范围

同时去掉P历史阴性面板、E历史误判精确复现、U前缀重采样一致性，仅保留基础分数→D单字段删除稳定性→I稳定交互后，3294次逐批预测以及四个留出阶段合计5112次预测，最终融合标签全部逐例相同。75个批次或留出检查点均完成，满足运行前规定的分类保持判据。

据此采用FP-TBF-DI作为当前精简主配置，原五层C路保留为对照。该决定来自同一开发队列，不能称为新独立队列上的等效验证。8406次评价包含跨配置和跨阶段重复，不等于8406名患者。

'''+joint+'''

## 2. 分数与分支变化

融合标签相同并不表示方法完全等价。逐批融合平均AUC从0.990003085降为0.989989048，最终留出从0.988725852升为0.988756614；所有AUC均重新计算。逐批有8次融合分数发生变化，最大绝对变化1.3935，没有跨过零阈值。

C路单独分类存在损失：逐批新增6次漏判，涉及3条原始记录，平均准确率由96.11%降为95.93%，BA由95.79%降为95.65%。最终849条训练的留出中，C路修复6次误报、新增2次漏判，净增加4次正确，准确率由96.24%升为96.56%。500条留出阶段C路净少判对2次，700条阶段净多判对2次。不能把融合层面的分类保持推广到C路单独使用。

## 3. 实际简化

- 移除基于历史阴性预测的SVM面板。
- 移除历史漏判/正确阴性组合的精确复现步骤。
- 移除U层的全特征模型及32次分类重采样一致性检查。
- 保留D层、I层及其已有资格：基础判阴、评分组合未在当前训练良性记录中出现。该良性组合排除属于D/I原有保护条件，与移除的历史误判复现不同。
- 不再把历史基础预测或历史误判子集作为方法输入与持久状态。仍保留累计训练评分与标签，因此不是无历史样本学习。

新实现位于`src/fp_tbf_simplified.py`，提供`SimplifiedFPTBF.fit`、`decision_scores`和`decision_function`。新运行入口为`run_fp_tbf_simplified.py`，只从已揭示累计前缀训练。原完整方法和先前所有结果未改写。

D层最多19次、I层最多33次辅助逻辑模型拟合，合计上限52次；移除的U层原本最多33次，另移除了历史SVM及误判匹配。实际次数仍由查询资格触发。这是模型调用结构的减少，不是已经证明的端到端加速比。

## 4. 核查

36个不同前缀重新拟合基础、A、B模型。51个逐批批次与6个最终留出的完整分数对照原归档，最大误差低于10^-7；全部24个固定留出检查点的原版指标对照既有阶段表一致。在末批800条及最终849条前缀的12个配置上，精简D/I新实现与直接调用原D/I辅助函数逐例完全一致。

输入原始数据、划分、算法源文件和旧结果均按SHA-256核对。精简入口另行运行一个完整配置，输出与本实验对应配置核对。所有阈值、正则化、规则预算、编码与重采样参数保持不变，没有依据结果校准分数。

## 5. 论文修订

方法主体改为D+I的C路；删除历史预测缓存状态，并重写流程图、算法步骤、摘要与结论。单分支C、含C的双路、最终三路及全部留出阶段的指标均重新计算。旧单层消融明确标为原五层参考方法的实验，避免与精简版混用。

论文将“保持效果”限定为本轮零阈值融合分类保持；同时披露AUC微小变化和C路自身漏判增加。后续验证应固定这一精简版本，再获取患者级隔离或独立来源数据，不继续在当前队列搜索删减组合。

复现材料位于`results/paper_c_joint_reduction/`，包括`predictions.csv`、`summary.json`、`checks.json`、`configuration_metrics.csv`及`updated_method_metrics.json`。实验协议为`paper/C路联合删减实验协议.md`，运行脚本为`run_paper_c_joint_reduction.py`。
'''
    (ROOT/'paper/C路联合删减与方法精简报告.md').write_text(report,encoding='utf-8')

    text=(ROOT/'paper/FP_TBF_中文论文_补充实验修订稿.md').read_text(encoding='utf-8')
    def replace(a,b):
        nonlocal text
        assert a in text,a[:90]
        text=text.replace(a,b)
    def section(start,end,body):
        nonlocal text
        a=text.index(start);b=text.index(end,a)
        text=text[:a]+body.strip()+'\n\n'+text[b:]
    def put_table(number,new):
        nonlocal text
        pattern=r'^\*\*表'+str(number)+r' [^\n]+\n\n(?:\|[^\n]+\n)+'
        text,count=re.subn(pattern,lambda _:new+'\n',text,flags=re.M)
        assert count==1,(number,count)

    abstract='''针对乳腺结构化评分的局部模式维护与组合判别问题，构建基于FP树增量统计和三路融合的精简分类方法FP-TBF-DI。以带类别计数的FP树提取列一致双聚类模式，分别构造投影筛选Sugeno推理、类别条件距离修正和D/I稳定性复核分支，最终对原始分数等权平均。C路仅保留单字段删除稳定性与稳定二阶交互，不再使用历史预测面板、误判组合复现及前缀重采样一致性。在1062条病灶记录的三个划分、两种批量内部评价中，精简版平均准确率为96.39%，平衡准确率为96.08%，ROC-AUC为0.989989。联合删减实验中，3294次逐批预测及四阶段合计5112次留出预测的最终融合标签均与原五层版本逐例相同；连续分数和AUC存在变化，C路单独分类的逐批准确率由96.11%降至95.93%。统一Sugeno拟合预算后，投影筛选相较覆盖筛选的平均准确率提高约0.33个百分点，但AUC下降。最终固定留出中，精简版准确率为96.71%，仍低于累计逻辑回归的96.87%。结果支持在当前内部评价中去除三个历史复核模块、保持最终分类表现，但不支持将该结论推广为分支等价或独立临床泛化。'''
    section('## 摘要','**关键词：**','## 摘要\n\n'+abstract)
    replace('利用历史预测补充识别信息时，又需要约束修正条件，避免只降低漏判而过多增加误报。',
            '引入补充判别模型时，需要约束其修正条件，并检验新增模块的局部收益能否传递到最终融合，避免将复杂结构本身视为必要贡献。')
    replace('基于此，本文提出FP-TBF方法，围绕',
            '基于此，本文构建精简方法FP-TBF-DI；原五层复核版本称为完整FP-TBF，作为组件实验的参照。本文围绕')
    replace('（2）将局部规则选择、类别条件距离和历史阴性复核组织为三个判别分支，给出共享模式统计、分支拟合及预测后更新的完整流程。',
            '（2）将局部规则选择、类别条件距离和稳定性复核组织为三个分支，通过D单字段删除与I二阶交互检查补充阴性判别；经联合删减实验移除三个历史复核层，减少辅助模型及历史预测状态。')
    replace('通过匹配事件分析报告收益与代价。','通过匹配事件及联合删减分析报告收益、代价与可简化范围。')
    replace('FP-TBF的总体结构如图1所示。','FP-TBF-DI的总体结构如图1所示。')
    replace('C路以覆盖互补推理为基础，利用历史预测及特征稳定性条件复核阴性结果。',
            'C路以覆盖互补推理为基础，通过单字段删除和二阶交互稳定性复核阴性结果。')
    replace('    V[历史逐批基础分数及随后揭示标签] --> C\n','')
    replace('C路：覆盖互补推理与阴性复核','C路：覆盖互补推理与D/I复核')
    replace('**图1 FP-TBF逐批预测与更新框架。**','**图1 FP-TBF-DI逐批预测与更新框架。**')

    c_method=r'''### 3.6 C路：单字段删除与二阶交互稳定性复核

C路从覆盖互补Sugeno基础分数\(h_0(x)\)出发，依次采用D单字段删除稳定性和I稳定二阶交互复核模型阴性结果。其输入仅为当前已揭示的评分、标签和查询评分，不需要历史预测或历史误判子集。

定义当前训练良性评分组合集合为\(\mathcal N_t=\{x_i:y_i=-1,(x_i,y_i)\in D_t\}\)。复核对象必须满足当前分数小于0且完整向量不属于\(\mathcal N_t\)。这是D/I原有的良性组合排除条件，用于限制修正范围，与原版根据历史漏判匹配组合的E层不同。

**D：单字段删除稳定性。** 在当前训练前缀上拟合独热编码，未见取值按全零处理；用全特征拟合逻辑回归，再分别删除18个原始字段对应的全部独热列并重新拟合。只有全特征及全部18个删除模型均给出非负分数时，才以全特征分数替换当前负分数，否则保持基础分数。该层最多拟合19个辅助模型。

**I：稳定二阶交互。** 对D后仍满足资格的记录，在当前训练前缀上标准化18项评分，并加入153个两两交互项。交互统一乘以\(\sqrt{2/17}\)，不包含平方项。除全前缀模型外，按类别分层进行32次有放回重采样，随机种子为1105，标准化参数在重采样中固定。仅当全前缀与全部重采样模型均判阳时，使用全前缀交互模型分数；若某次已排除所有查询对象，可提前停止其余重采样。该层至多拟合33个模型。

D、I均采用C=1、liblinear及1000次迭代上限的逻辑回归。它们只对仍判阴且满足条件的记录进行正向替换，基础判阳结果保持不变。最终得到\(h_{C,\mathrm{DI}}(x)\)。这一过程仍访问累计训练评分与标签；取消历史预测缓存并不等于无需历史训练数据。

原五层C路在D、I之前依次执行P历史阴性SVM面板、E历史误判组合复现及U前缀重采样一致性。它们仅作为5.9—5.10节的参考版本。精简主方法在运行时不拟合或执行P、E、U。
'''
    section('### 3.6 C路：','### 3.7 ',c_method)
    replace('H(x)=\\frac{h_A(x)+h_B(x)+h_C(x)}3','H(x)=\\frac{h_A(x)+h_B(x)+h_{C,\\mathrm{DI}}(x)}3')
    replace('**算法1 FP-TBF逐批预测与更新**','**算法1 FP-TBF-DI逐批预测与更新**')
    replace('**状态：** 累计FP树、已揭示评分及标签、三路分类模型、历史逐批基础分数及其随后揭示的标签。',
            '**状态：** 累计FP树、已揭示评分及标签、A/B/基础分类模型，以及当前前缀上的D/I辅助模型。')
    replace('历史逐批分数集合初始化为空。','不初始化历史预测缓存。')
    replace('3. 按“历史阴性面板—精确组合复现—前缀一致性—逐项删除—稳定交互”的顺序计算C路。各阶段均只修正仍符合条件的记录；条件不满足时保留上一阶段分数。',
            '3. 从基础分数开始，依次执行D单字段删除、I稳定交互复核。每层重新计算阴性及良性组合排除资格；不满足条件时保留上一阶段分数。')
    replace('5. 揭示整批标签；将评分、标签与本批预测前得到的基础分数加入历史。历史分数不以更新后模型回算替换。',
            '5. 揭示整批标签，将评分与标签加入训练前缀。预测分数可保存用于评价，但不作为后续C路拟合输入。')
    replace('C路阴性复核模型','C路D/I复核模型')
    replace('原主配置的迭代上限不同；为隔离这一因素，本研究另外开展4.6节的统一预算对照，原主配置保持不变。',
            'A路与基础模型的主配置迭代上限不同，4.6节另外采用共同上限隔离这一因素。联合删减仅改变C路，A、B与基础模型的参数不变。')
    replace('C路依次定义P历史阴性面板、E精确组合复现、U前缀一致性、D单字段删除稳定性和I稳定交互五个组件。',
            '原完整C路依次定义P历史阴性面板、E精确组合复现、U前缀一致性、D单字段删除稳定性和I稳定交互五个组件。')
    replace('不依据结果调整参数或更换主方法。','不依据结果调整参数；该轮完整配置作为联合删减的参考方法。')
    joint_design='''### 4.7 P/E/U联合删减与精简判据

在单项消融之后，预先固定仅保留基础→D→I这一候选，同时去掉P、E、U。沿用全部参数与原始分数平均，重新拟合36个不同前缀的A、B及基础模型，并重新评价51个逐批批次和四阶段24个留出检查点。原完整路径逐批及最终留出对照已有分数，四阶段原版指标对照已有阶段表；精简D/I新实现另与直接调用原D/I函数核对。

将全部已评价融合标签逐例一致规定为保持当前分类效果的强判据，并单独报告分支标签、分数及AUC变化。通过后将精简版作为本文当前方法，原版作为对照。该选择发生在同一开发队列的补充分析中，仍需固定精简方法后开展独立验证，不构成患者级等效检验。

'''
    replace('## 5 实验结果与分析',joint_design+'## 5 实验结果与分析')

    stream=updated['stream'];final=updated['holdout_849']
    put_table(4,table('表4 相同六配置上的方法比较（配置等权均值）',stream,
                    [('base','覆盖互补Sugeno'),('A','A路：投影筛选'),('B','B路：距离修正'),('simple_c','C路：D/I复核'),
                     ('coverage_A','覆盖互补＋A路'),('logistic','累计逻辑回归'),('full_fusion','原完整FP-TBF'),('simple_fusion','FP-TBF-DI')]))
    put_table(5,table('表5 精简C路下的全部分支组合（逐批六配置均值）',stream,
                    [('A','A'),('B','B'),('simple_c','C_DI'),('AB','A+B'),('AC','A+C_DI'),('BC','B+C_DI'),('simple_fusion','A+B+C_DI')]))
    put_table(9,table('表9 849条训练时的匹配固定留出结果（六配置均值）',final,
                    [('base','覆盖互补Sugeno'),('logistic','累计逻辑回归'),('A','A'),('B','B'),('simple_c','C_DI'),('AB','A+B'),
                     ('AC','A+C_DI'),('BC','B+C_DI'),('full_fusion','原完整FP-TBF'),('simple_fusion','FP-TBF-DI')]))
    t2=['**表2 FP-TBF-DI逐配置结果（除AUC外单位为%）**','',
        '|种子|批量|准确率|精确率|敏感度|特异度|F1|BA|ROC-AUC|',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in stream['simple_fusion']['configurations']:
        nums=[f'{100*r[k]:.2f}' for k in ('accuracy','precision','recall','specificity','f1','balanced_accuracy')]+[f'{r["auc"]:.4f}']
        t2.append(f'|{r["seed"]}|{r["batch"]}|'+'|'.join(nums)+'|')
    put_table(2,'\n'.join(t2))
    t7=['**表7 FP-TBF-DI不同训练规模的固定留出结果（六配置均值）**','',
        '|累计训练/条|本阶段新增/条|准确率/%|精确率/%|敏感度/%|特异度/%|F1/%|BA/%|ROC-AUC|',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    previous=0
    for n in (300,500,700,849):
        r=updated[f'holdout_{n}']['simple_fusion']['means']
        nums=[f'{100*r[k]:.2f}' for k in ('accuracy','precision','recall','specificity','f1','balanced_accuracy')]+[f'{r["auc"]:.4f}']
        t7.append(f'|{n}|{n-previous}|'+'|'.join(nums)+'|');previous=n
    put_table(7,'\n'.join(t7))
    stream_rows=[r for r in rows if r['evaluation']=='stream']
    y=np.array([int(r['label']) for r in stream_rows]);candidate=np.array([float(r['simple_fusion']) for r in stream_rows])
    t6=['**表6 FP-TBF-DI相较各参照的事件正确性变化**','',
        '|参照方法|纠正原错误/次|新增错误/次|净增加正确/次|','|---|---:|---:|---:|']
    for key,label in [('base','覆盖互补Sugeno'),('coverage_A','覆盖互补＋A路'),('A','A路'),('B','B路'),
                      ('simple_c','C_DI'),('logistic','累计逻辑回归'),('D_fusion','D层后融合'),('full_fusion','原完整FP-TBF')]:
        ref=np.array([definitions[key](r) for r in stream_rows]);q=paired(y,ref,candidate)
        t6.append(f'|{label}|{q["repaired"]}|{q["introduced"]}|{q["net"]}|')
    put_table(6,'\n'.join(t6))
    # Refresh every main-result interpretation affected by the new C route.
    replace('### 5.1 整体分类表现','### 5.1 精简方法的整体分类表现')
    replace('表2给出FP-TBF在六种配置下的逐批预测表现。','表2给出FP-TBF-DI在六种配置下的逐批预测表现。')
    replace('**表3 FP-TBF的3294次内部预测事件混淆矩阵**','**表3 FP-TBF-DI的3294次内部预测事件混淆矩阵**')
    replace('表4比较各判别分支与组合方法。FP-TBF的平均准确率','表4比较各判别分支与组合方法。FP-TBF-DI的平均准确率')
    replace('表6从相同预测事件上的正确性变化分析融合收益。相较覆盖互补Sugeno，FP-TBF纠正',
            '表6从相同预测事件上的正确性变化分析融合收益。相较覆盖互补Sugeno，FP-TBF-DI纠正')
    replace('不能据此推断同时删去P、E、U仍保持结果。',
            '单项实验本身不足以判断P、E、U同时删除的结果，联合检查另见5.10节。')
    a=text.index('C路的敏感度为97.22%');b=text.index('**表5 ',a)
    text=text[:a]+'''精简C路的平均敏感度为96.92%，高于A路的96.85%，低于B路的97.12%；其特异度为94.37%。三路融合的敏感度和特异度分别为97.41%和94.76%，说明当前组合收益不能由C路单独性能判断。主方法平均AUC为0.989989，原完整版本为0.990003，两者保留更高精度后并不相同。

'''+text[b:]
    a=text.index('加入稳定二阶交互后，三路模型');b=text.index('### 5.4 ',a)
    text=text[:a]+'''在精简路径中，从D层后继续加入I层，逐批融合净多判对2次，涉及同一条记录在两种配置中的预测，平均准确率由96.3267%增至96.3874%。这一局部收益仍不足以证明稳定交互在独立病例上的普遍必要性。与原完整五层版本比较，精简版的最终融合分类完全相同，连续分数变化另见5.10节。

'''+text[b:]
    replace('FP-TBF的AUC为0.9887，高于逻辑回归的0.9870；','精简版AUC为0.9888，高于逻辑回归的0.9870；')
    replace('完整FP-TBF还包含累计规则筛选、多模型拟合及重采样复核。',
            '精简版仍包含累计规则筛选、D/I辅助拟合及交互重采样，尚不能由组件时间推断整体部署效率。')
    replace('### 5.9 C路逐层加入与逐项删除','### 5.9 原五层参考C路的逐层与单项消融')
    replace('**表11 C路逐层加入结果','**表11 原五层参考C路逐层加入结果')
    replace('**表12 C路逐项删除结果','**表12 原五层参考C路逐项删除结果')
    replace('**表13 C路逐项删除的最终固定留出结果','**表13 原五层参考C路逐项删除的最终留出结果')
    a=text.index('### 5.9 ');b=text.index('**表11 ',a)
    text=text[:a]+text[a:b]+'''本节以原P+E+U+D+I为完整参照，记录删减前的组件证据；表11—13中的“完整C路”均指原五层版本，不是本文精简主方法。

'''+text[b:]
    joint_result='''### 5.10 P/E/U联合删减与方法精简

'''+joint+'''

共同去掉P、E、U之后，3294次逐批和四阶段5112次留出事件的最终融合标签均逐例相同，每个配置的准确率、敏感度、特异度、F1与BA保持不变。由此将基础→D→I确定为本文当前精简配置。取消的是历史预测面板、历史误判匹配与U层，保留D/I原有的训练良性组合排除条件。

分类保持并非分数等价。逐批有8次融合分数发生变化，最大绝对变化1.3935；平均AUC由0.990003085降为0.989989048，最终留出由0.988725852升为0.988756614。融合差分等于C路差分的三分之一，本轮变化未跨过零阈值，但不能由此保证新病例也具有足够判别余量。

C路单独的逐批分类新增6次漏判、涉及3条记录，准确率由96.11%降至95.93%；最终留出则修复6次误报并新增2次漏判，净多判对4次。保持最终融合不等于保持C路单支表现。新方法不再接收历史预测分数，D/I辅助拟合次数上限由原U+D+I的85次降至52次，并取消历史SVM；实际计算仍由资格触发，尚不作为端到端加速结论。

'''
    replace('## 6 讨论',joint_result+'## 6 讨论')
    discussion='''### 6.2 联合删减揭示的条件冗余

单项删除不改变结果时，可能存在组件间替代关系，因而不能直接同时移除多个组件。本研究先进行单项消融，再固定D+I候选开展联合删减，并覆盖逐批与全部四个留出阶段。最终融合标签逐例保持，支持在当前数据和阈值下减少三个历史复核模块。

这一结果没有使C路本身与原版等价。精简版在C路单独分类中新增部分漏判，而A、B提供的分数使这些变化未跨过最终决策边界。因此，简化依据是目标组合分类器的输出保持，而不是每个子分类器各项指标均保持。连续分数和AUC的改变也要求重新报告精简方法的全部排名指标。

精简版将辅助拟合集中于评分字段删除和交互稳定性，不再维护历史误判面板与历史基础分数。它仍使用累计训练评分、类别条件分布及多个逻辑模型，不能等同于有界内存或只更新新批次参数的学习。模型简化降低了结构与状态负担，但跨患者泛化、概率校准和完整成本仍需独立评价。
'''
    section('### 6.2 ','### 6.3 ',discussion)
    replace('FP-TBF的完整决策还包括支持向量机、逻辑回归、交互项及多阶段分数替换。',
            'FP-TBF-DI的决策仍包括逻辑回归、交互项及D/I分数替换。')
    replace('逐项删除也检验了C路各层在其余层保留时的作用，但尚未覆盖不同组件顺序及所有联合删减；后续不能把当前11种配置的结果推广为任意组合的最优结构。',
            '逐项及P/E/U联合删除检验了特定组件组合，但尚未覆盖不同顺序及任意联合删减；当前精简决定仍基于同一开发队列，不确立全局最优结构。')
    replace('C路多阶段复核采用的稳定性条件仍可能适应当前队列；',
            'D/I稳定性条件及本轮结构选择仍可能适应当前队列；')
    conclusion='''## 7 结论

本文在FP树增量模式维护与三路判别框架下，经统一预算、单项和联合消融，将C路精简为基础推理后的D单字段删除稳定性与I稳定交互复核。新方法FP-TBF-DI不再依赖历史基础预测缓存及历史误判子集，保留原有训练前缀、规则参数与融合方式。

在3294次逐批及四阶段5112次留出预测事件中，精简版与原五层版本的最终融合标签全部相同，逐批平均准确率为96.39%、BA为96.08%、AUC为0.989989。C路自身存在分类变化，融合分数与AUC也并非完全相同；最终留出准确率96.71%仍低于逻辑回归的96.87%。这些结果支持当前内部评价中的结构精简，不能替代患者级隔离及独立队列上的确认。

后续应固定精简配置，优先开展直接近邻的同协议比较、端到端成本测量及独立来源验证，避免继续在同一开发队列上搜索微小改进。
'''
    section('## 7 结论','## 参考文献',conclusion)
    target=ROOT/'paper/FP_TBF_DI_中文论文精简修订稿.md'
    assert list(map(int,re.findall(r'^\*\*表(\d+) ',text,re.M)))==list(range(1,15))
    assert list(map(int,re.findall(r'\\tag\{(\d+)\}',text)))==list(range(1,16))
    target.write_text(text,encoding='utf-8')
    print('Saved joint-reduction report and simplified manuscript',len(text),'characters')


if __name__=='__main__':main()
