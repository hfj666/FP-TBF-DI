"""Report the fixed supplement and revise the manuscript without changing models."""
import csv
import json
import re
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/paper_budget_c_ablation'
NAMES={'base':'仅基础分数','plus_P':'P','plus_PE':'P+E','plus_PEU':'P+E+U',
       'plus_PEUD':'P+E+U+D','full':'完整C路',
       'without_P':'去P','without_E':'去E','without_U':'去U','without_D':'去D','without_I':'去I'}
CUMULATIVE=['base','plus_P','plus_PE','plus_PEU','plus_PEUD','full']
DELETIONS=['full','without_P','without_E','without_U','without_D','without_I']


def budget_table(result,evaluation,title):
    lines=[f'**{title}**','', '|筛选|拟合上限|准确率/%|敏感度/%|特异度/%|F1/%|BA/%|ROC-AUC|',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for budget in (500,2000):
        for strategy,label in [('coverage','覆盖互补'),('projection','联合投影')]:
            v=result[evaluation][f'{strategy}_{budget}']['means']
            values=[f'{100*v[k]:.2f}' for k in ('accuracy','recall','specificity','f1','balanced_accuracy')]+[f'{v["auc"]:.4f}']
            lines.append(f'|{label}|{budget}|'+'|'.join(values)+'|')
    return '\n'.join(lines)


def c_table(result,variants,evaluation,title):
    lines=[f'**{title}**','', '|C路配置|C路准确率/%|C路BA/%|融合准确率/%|融合BA/%|融合AUC|相较完整融合净增加正确/次|',
           '|---|---:|---:|---:|---:|---:|---:|']
    for name in variants:
        b=result['branch'][evaluation][name]['means'];f=result['fusion'][evaluation][name]['means']
        net=0 if name=='full' else result['vs_full'][name]['fusion_score'][evaluation]['net']
        lines.append(f'|{NAMES[name]}|{b["accuracy"]*100:.2f}|{b["balanced_accuracy"]*100:.2f}|{f["accuracy"]*100:.2f}|{f["balanced_accuracy"]*100:.2f}|{f["auc"]:.4f}|{net:+d}|')
    return '\n'.join(lines)


def main():
    selection=json.loads((OUT/'selection_summary.json').read_text(encoding='utf-8'))
    c=json.loads((OUT/'c_summary.json').read_text(encoding='utf-8'))
    verify=json.loads((OUT/'verification.json').read_text(encoding='utf-8'))
    assert verify['source_hashes_unchanged'] and verify['full_route_archive_replays']==57
    assert verify['full_route_max_error']<=1e-7
    budget_stream=budget_table(selection,'stream','表10 统一Sugeno拟合上限的筛选比较（逐批六配置均值）')
    budget_holdout=budget_table(selection,'holdout','补充表S1 统一预算的最终固定留出比较（六配置均值）')
    cumulative=c_table(c,CUMULATIVE,'stream','表11 C路逐层加入结果（逐批六配置均值）')
    deletion=c_table(c,DELETIONS,'stream','表12 C路逐项删除结果（逐批六配置均值）')
    holdout=c_table(c,DELETIONS,'holdout','表13 C路逐项删除的最终固定留出结果（六配置均值）')
    rows=[]
    for ev in ('stream','holdout'):
        for arm,r in selection[ev].items():
            for config in r['configurations']:rows.append(dict(experiment='selection',evaluation=ev,target='standalone',arm=arm,**config))
        for target in ('branch','fusion'):
            for arm,r in c[target][ev].items():
                for config in r['configurations']:rows.append(dict(experiment='c_ablation',evaluation=ev,target=target,arm=arm,**config))
    with (OUT/'configuration_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fits=[json.loads(p.read_text(encoding='utf-8')) for p in (OUT/'selection').glob('*.json')]
    fit_time={arm:float(np.mean([r['models'][arm]['fit_seconds'] for r in fits])) for arm in selection['fit_status']}
    selection_time={name:float(np.mean([r['selection_seconds'][name] for r in fits])) for name in ('coverage','projection')}
    time_table=['|策略|平均筛选耗时/s|500步拟合平均耗时/s|2000步拟合平均耗时/s|','|---|---:|---:|---:|']
    for name,label in [('coverage','覆盖互补'),('projection','联合投影')]:
        time_table.append(f'|{label}|{selection_time[name]:.3f}|{fit_time[name+"_500"]:.3f}|{fit_time[name+"_2000"]:.3f}|')
    stages=json.loads((OUT/'c_shared_costs.json').read_text(encoding='utf-8'))
    shared_seconds=sum(r['seconds'] for r in stages)
    component_fits=sum(sum(v['model_fits'] for v in r['auxiliary'].values()) for r in stages)
    sequential=[]
    with (OUT/'c_predictions.csv').open(encoding='utf-8-sig',newline='') as f:predictions=list(csv.DictReader(f))
    for ev in ('stream','holdout'):
        vectors={}
        for name in CUMULATIVE:
            values=[r for r in predictions if r['evaluation']==ev and r['variant']==name]
            vectors[name]=np.array([int(r['label']) for r in values]),np.array([float(r['c_score']) for r in values]),np.array([float(r['fusion_score']) for r in values])
        for previous,current in zip(CUMULATIVE[:-1],CUMULATIVE[1:]):
            y,a,fa=vectors[previous];_,b,fb=vectors[current]
            sequential.append(dict(evaluation=ev,added=current,branch_net=int(((b>=0)==(y==1)).sum()-((a>=0)==(y==1)).sum()),
                                   fusion_net=int(((fb>=0)==(y==1)).sum()-((fa>=0)==(y==1)).sum())))
    (OUT/'derived_statistics.json').write_text(json.dumps(dict(fit_mean_seconds=fit_time,selection_mean_seconds=selection_time,
                      c_shared_seconds=shared_seconds,c_shared_auxiliary_fits=component_fits,
                      sequential_changes=sequential),ensure_ascii=False,indent=2),encoding='utf-8')

    report='''# 统一拟合预算与C路逐层消融：补充实验报告

## 1. 结论

本轮已完成36个不同训练前缀的144次Sugeno拟合、C路11种配置的逐批与最终固定留出评价。六配置逐批各3294次事件，最终留出各1278次事件。完整C路在57个批次或留出检查点与原归档分数完全一致。主方法与原数据未更换，所有阈值和参数固定。

统一预算后，投影筛选相较覆盖筛选的逐批准确率提高0.334个百分点、BA提高0.292个百分点，但AUC降低约0.0027。该收益在500与2000两个共同上限下均存在，不能仅解释为原稿迭代上限不同；同时也不代表排序能力或各划分均改善。

C路消融表明，P、E、U各自单独删除时，在逐批与最终留出中均不改变最终融合的分类标签。去掉D和I各使逐批融合少判对2次，分别涉及1条不同原始记录；最终留出中只有去I少判对2次，涉及1条记录。当前结果不支持将五层复核全部写成不可替代的分类贡献。

## 2. 实验控制与验证

筛选采用2×2设计：覆盖/投影，拟合上限500/2000。每类32条规则，候选、初始化、损失、边界、训练前缀、预处理和分类阈值相同。相同上限并非相同实际耗时，选择优化成本单列。

C路组件为P历史阴性面板、E精确组合复现、U前缀一致性、D单字段删除稳定性、I稳定交互。逐层加入6种配置，逐项删除5种配置；其中去I与P+E+U+D相同，但保留别名。删除后重新执行其余层的资格判断，A、B与基础历史预测固定。共享的辅助拟合只使用当前训练前缀，不把查询评分加入训练。

“仅基础分数”配置仍将覆盖互补基础分数作为C路，与固定A、B进行三项平均，并非删除整个C分支或改成A+B双路。

除全部完整路径归档重放外，还在种子43、批量100、训练前缀800上，用原生辅助函数独立重算全部11种配置，与共享实现逐例完全一致。源文件校验值保持不变。

## 3. 统一预算筛选结果

'''+budget_stream+'\n\n'+budget_holdout+'''

共同上限2000时，逐批BA在6配置中4个提高、2个降低。种子17的两个配置分别降低0.293和0.977个百分点，其他四配置提高。平均收益并不代表所有设置下均优于覆盖筛选。最终留出中4配置提高、2配置持平，但相同种子的两个配置共用同一筛选模型，不能视为独立重复。

两个预算下，逐批配对结果均为纠正48次、引入37次错误，净增11次正确；最终留出均为纠正10次、引入6次错误，净增4次正确。预算500与2000在所有已评价事件上的分类标签相同，但投影分数并非完全相同，最大分数差约0.1301。

覆盖筛选的两个预算均为36/36次收敛；投影500为34/36次，未收敛状态为种子29的450和700条训练前缀达到迭代上限，仍保留其有限输出；投影2000为36/36次收敛。覆盖两预算的平均实际迭代数均为54.0，投影分别为191.5和196.2。

'''+ '\n'.join(time_table)+'''

耗时为本次单线程、固定顺序执行的36个不同前缀的算术平均，不是独立重复基准。投影包含额外筛选计算，不能因共同Sugeno上限相同便称为总计算预算相同。

## 4. C路逐层加入

'''+cumulative+'''

按P、E、U、D、I顺序加入时，C路单独分类分别净增3、4、4、3、2次逐批正确事件；固定A、B后的最终融合分别净增0、0、0、2、2次。局部分支的分类收益并不自动传递到最终融合。

最终留出中，P、E分别使C路单独分类净少判对2次，U、D没有净变化，I净多判对2次；完整C路准确率96.24%，低于基础C分数的96.40%。但最终融合从96.56%提高至96.71%，这两种评价目标应分别解释。

## 5. C路逐项删除

'''+deletion+'\n\n'+holdout+'''

P、E、U的逐项删除均未改变最终融合标签，但P和E的删除会改变C路单独分类：逐批各少判对2次、4次，最终留出则各净多判对2次。U删除在两类评价的C路及融合分类上均没有变化。部分变体的AUC仍有细微变化，因此“标签不变”不等于“连续分数完全相同”。

上述是每次只删除一个组件、其余保留时的条件结果，不能推出P、E、U同时删除仍无影响。D、I的逐批边际作用均只涉及1条不同记录；I的最终留出作用也只涉及1条记录，不将重复事件数写成独立患者数。

## 6. 成本与范围

'''+f'C路共享组件计算总耗时约{shared_seconds:.2f} s，执行{component_fits}次辅助逻辑模型拟合，覆盖57个预测批次或最终留出检查点。'+'''

该计算在相同前缀下缓存辅助分数供11种配置使用，预测对象取各配置资格的并集。它不等于任何一个变体独立执行的端到端时间，也不用于声称删除某层后可加速多少。本文只据直接计数讨论复杂度，不将缓存后的总耗时当作临床部署成本。

该补充实验仍使用历史开发队列和既有留出；没有新增患者级隔离或外部队列。主要指标预定为BA，其他指标同时报告；未调阈值、权重或预算以追求特定结果。未据消融胜者更换主方法。

## 7. 对论文的修改

- 取消“筛选效果尚被不同迭代上限混淆”的未完成表述，改成统一预算下的实测结果；保留固定阈值收益与AUC下降的区别。
- 增加C路逐层加入及逐项删除，区分C路本身和最终融合的作用，不再暗示五层均具有独立必要性。
- 将模型简化列为后续固定验证方向，不用单一开发队列的消融结果直接发布新优胜方法。
- 保留原三路96.39%的内部结果及其最终留出不及逻辑回归的事实。

复现文件：根目录`run_paper_budget_c_ablation.py`、`verify_paper_budget_c_ablation.py`、`summarize_paper_budget_c_ablation.py`；事先协议位于`paper/统一预算与C路消融实验协议.md`；分配置指标位于`results/paper_budget_c_ablation/configuration_metrics.csv`；原始预测、拟合状态、完整重放与校验记录保存在同一结果目录。
'''
    (ROOT/'paper/统一预算与C路消融实验报告.md').write_text(report,encoding='utf-8')

    text=(ROOT/'paper/FP_TBF_中文论文修订稿.md').read_text(encoding='utf-8')
    def replace(a,b):
        nonlocal text
        assert a in text,a[:80]
        text=text.replace(a,b)
    abstract='''针对乳腺超声结构化评分随新增记录变化的模式维护与判别问题，构建FP树增量模式维护和三路判别融合方法FP-TBF。以带类别计数的FP树提取列一致双聚类模式，结合投影筛选Sugeno推理、类别条件距离修正和历史阴性复核，对三路分数等权融合。在1062条病灶记录上采用三个固定划分及两种批量大小开展先预测后更新的内部评价，并补充统一拟合预算和C路逐层消融。主方法平均准确率为96.39%，平衡准确率为96.08%，ROC-AUC为0.9900。在共同2000步拟合上限下，投影筛选相较覆盖筛选的准确率由95.63%提高至95.96%，平衡准确率由95.40%提高至95.69%，但AUC由0.9865降至0.9838；500步共同上限下的分类标签与2000步结果相同。C路单独去掉历史面板、精确组合复现或前缀一致性层，逐批与最终留出中的融合分类标签均保持不变；去掉单字段删除或稳定交互层，逐批融合各少判对2次，分别仅涉及1条原始记录。最终固定留出中，完整三路准确率为96.71%，低于累计逻辑回归的96.87%。结果表明，投影筛选在当前内部评价中具有有限的固定阈值收益，复杂复核层的分支作用与最终融合贡献并不一致，现有证据尚不足以确立各层必要性及独立泛化优势。'''
    a=text.index('## 摘要');b=text.index('**关键词：**')
    text=text[:a]+'## 摘要\n\n'+abstract+'\n\n'+text[b:]
    replace('由于A路与基础模型的迭代上限不同，两者差异不作为仅改变筛选机制的严格消融。',
            '原主配置的迭代上限不同；为隔离这一因素，本研究另外开展4.6节的统一预算对照，原主配置保持不变。')
    replace('由于两者筛选方式和拟合迭代上限均有差异，这一结果描述两个完整配置的表现，尚不能单独归因于投影筛选。',
            '表4描述原主配置，筛选因素在5.8节通过共同拟合上限进一步检验。')
    replace('本文补充分析读取固定数据、划分与归档分数完成，核心结果与原汇总逐例核对；不新增模型拟合。各输入文件的校验值及分配置统计随复算材料保存。',
            '本节组合与评分分层分析读取固定数据、划分与归档分数完成，核心结果与原汇总逐例核对，不新增模型拟合。4.6节另行开展统一预算重新拟合与组件消融；各输入文件的校验值及分配置统计随复算材料保存。')
    new_design='''### 4.6 统一拟合预算与C路组件消融

为隔离Sugeno迭代上限的影响，采用“覆盖筛选/投影筛选×500/2000步共同拟合上限”的2×2设计。相同前缀下使用一致候选、每类32条规则、初始尺度、冷初始化、损失和参数边界，仅改变筛选方式或预定迭代上限；记录实际迭代数、优化状态和耗时。不同批量共用相同前缀的拟合，得到36个不同前缀、144次模型拟合。达到迭代上限但输出有限的模型不剔除，单独报告停止状态。相同拟合上限不包含投影筛选的额外计算，不称为相同总计算预算。

C路依次定义P历史阴性面板、E精确组合复现、U前缀一致性、D单字段删除稳定性和I稳定交互五个组件。首先比较从基础分数依次加入P、E、U、D、I的六种路径；其次在完整路径中每次去掉一个组件，其余组件保持原顺序并重新计算资格。两组共11个具名配置，其中去I与P+E+U+D相同。A、B及基础历史预测保持原值，分别报告C路本身和三路融合结果。

“仅基础分数”的C路配置仍取覆盖互补分数h0，与A、B进行三项平均，不等于直接移除C路后采用A+B双路。

辅助模型的拟合目标只依赖已揭示前缀和相应历史，故可在同一配置的同一训练前缀上共享各变体的模型分数，再重新执行资格判断；不共享未来标签，也不将查询对象加入训练。全部57个逐批或最终留出路径的完整C路均与原归档逐例核对，并以原生辅助函数独立复算一个前缀上的全部变体。实验协议在本轮运行前固定，不依据结果调整参数或更换主方法。共享组件耗时仅描述补充实验的计算，不替代每个变体的独立端到端成本。

'''
    replace('## 5 实验结果与分析',new_design+'## 5 实验结果与分析')
    new_results='''### 5.8 统一拟合上限的规则筛选比较

'''+budget_stream+'''

共同上限2000时，投影筛选较覆盖筛选的准确率提高0.334个百分点，BA提高0.292个百分点；六配置BA为4个提高、2个降低。配对统计为纠正48次、引入37次错误，净增加11次正确预测。共同上限500时，分类标签和配对计数相同，AUC略有变化。这表明原固定阈值差异不能仅由迭代上限不同解释，但投影并未改善所有划分，也未改善整体分数排序。

覆盖模型在两预算下均36/36次收敛，平均实际迭代54.0次；投影500为34/36次收敛，投影2000为36/36次收敛，平均实际迭代分别为191.5和196.2次。两个未收敛状态均为达到500步上限，其有限分数仍纳入评价。预算提高导致的投影分数最大差约0.1301，但未改变本轮任何已评价事件的分类标签。

最终固定留出中，两预算下投影与覆盖的准确率分别为96.71%和96.40%，BA分别为96.59%和96.34%，AUC分别约为0.9837和0.9873。固定阈值收益与排序指标下降同时存在。两种筛选的平均选择耗时约为0.733 s和1.576 s，2000步上限下平均拟合耗时约为0.170 s和0.648 s，前者为覆盖、后者为投影。该耗时来自单次固定顺序运行，仅说明投影存在额外计算，不能据此建立稳定加速比。

### 5.9 C路逐层加入与逐项删除

'''+cumulative+'''

按既定顺序加入P、E、U、D、I，C路单独分类分别净增3、4、4、3、2次逐批正确事件；最终融合分别净增0、0、0、2、2次。前几层在分支层面的收益未改变融合后的类别，说明组件作用需要在局部分支与最终模型两个层次评价。

'''+deletion+'''

在完整C路中，每次单独去掉P、E或U，均不改变最终融合的逐批分类标签，AUC仅有细微变化；去掉D或I均少判对2次，各涉及1条原始记录。这里检验的是其他组件保留时的边际作用，不能据此推断同时删去P、E、U仍保持结果。

'''+holdout+'''

最终留出中，单独去掉P、E、U或D均不改变融合标签，去I则少判对2次、涉及1条记录。P和E各自删除后，C路单独分类反而净增加2次正确预测，表明其逐批局部收益未在回顾性留出中一致复现。全部完整路径与原归档分数一致，独立原生函数核对也一致。因此，上述差异来自按协议改变组件，而非共享计算改变完整方法。

'''
    replace('## 6 讨论',new_results+'## 6 讨论')
    replace('现有结果不足以确立这些复核层全部必要，后续应在固定协议下逐层删减，检验更简洁的方案能否保持表现；不宜依据当前表格继续增添修正层。',
            '本轮逐项删除进一步显示，P、E、U各自在其他层保留时未影响最终融合标签，D和I的逐批边际作用各仅涉及1条记录。现有结果不支持将五层复核都视为不可替代；任意多层联合删除和独立队列中的稳定性仍未得到检验，不宜据此继续增添修正层。')
    replace('当前匹配对照尚未覆盖文献[1,2]直接近邻的同协议复现，也未完全分离C路各复核层的贡献。投影分支与基础模型的迭代预算不同，使当前对照尚不能独立检验规则筛选机制。',
            '当前匹配对照尚未覆盖文献[1,2]直接近邻的同协议复现。统一预算对照隔离了拟合上限差异，逐项删除也检验了C路各层在其余层保留时的作用，但尚未覆盖不同组件顺序及所有联合删减；后续不能把当前11种配置的结果推广为任意组合的最优结构。')
    replace('当前证据支持逐批固定阈值下的有限组合收益，尚不支持在不同评价设置中稳定占优。',
            '当前证据支持逐批固定阈值下的有限组合收益，尚不支持在不同评价设置中稳定占优。统一预算后的投影比较同样体现这一点：平均BA提高，但AUC下降且两个配置BA降低，故筛选收益具有指标与配置依赖性。')
    a=text.index('## 7 结论');b=text.index('## 参考文献')
    text=text[:a]+'''## 7 结论

本文构建FP树增量模式维护与三路判别融合流程，将局部规则推理、类别条件距离及历史阴性复核用于乳腺结构化评分分类。在六种内部逐批配置中，主方法平均准确率为96.39%、BA为96.08%、ROC-AUC为0.9900。统一Sugeno拟合上限后，投影筛选相较覆盖筛选仍取得约0.33个百分点的准确率收益及0.29个百分点的BA收益，但AUC降低，且未在全部配置获益。

C路逐层加入和逐项删除表明，分支层面的改进不必然改变最终融合分类。P、E、U分别单独删除时，逐批与最终留出融合标签均保持不变；D、I的逐批边际收益各仅涉及1条原始记录，I的最终留出收益也仅涉及1条记录。完整三路在最终固定留出中的准确率为96.71%，仍低于累计逻辑回归的96.87%。

上述证据支持当前模式与判别组合的有限内部收益，同时揭示复杂复核结构中的条件冗余。后续应优先固定更简洁的候选结构并开展独立验证，补充近邻方法同协议对照、完整运行成本及患者级隔离评价，而不是依据同一开发队列的微小增益继续增加模块。

'''+text[b:]
    assert list(map(int,re.findall(r'^\*\*表(\d+) ',text,re.M)))==list(range(1,14))
    assert list(map(int,re.findall(r'\\tag\{(\d+)\}',text)))==list(range(1,16))
    target=ROOT/'paper/FP_TBF_中文论文_补充实验修订稿.md'
    target.write_text(text,encoding='utf-8')
    print('Written report and manuscript',len(text),'characters')


if __name__=='__main__':main()
