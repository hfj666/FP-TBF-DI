"""Generate the evidence revision from the previous manuscript and saved results."""
import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PAPER = ROOT / 'paper'
OUT = ROOT / 'results/paper_evidence_extension'


def read(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))


def values(row):
    m = row['means']
    return f"{100*m['accuracy']:.2f} / {100*m['balanced_accuracy']:.2f} / {m['auc']:.5f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-manuscript', action='store_true',
                        help='Also build the historical manuscript revision from a local draft.')
    args = parser.parse_args()
    matched, grouped = read('matched_summary.json'), read('grouped_summary.json')
    frozen = read('frozen_update/summary.json')
    if not args.write_manuscript:
        summary = {
            'stream': {name: row['means'] for name, row in matched['stream'].items()},
            'holdout_849': {name: row['means'] for name, row in matched['holdout_849'].items()},
            'grouped': {name: row['means'] for name, row in grouped.items()},
            'frozen_stream': frozen['frozen']['means'],
        }
        print(json.dumps(summary, indent=2))
        return
    old = PAPER / 'FP_TBF_DI_中文论文精简修订稿.md'
    s = old.read_text(encoding='utf-8')

    def replace(start, end, body):
        nonlocal s
        i = s.index(start)
        j = s.index(end, i + len(start))
        s = s[:i] + body.strip() + '\n\n' + s[j:]

    s = '# 乳腺病灶结构化评分的增量模式分类：三路融合与结构简化的实证研究\n' + s.split('\n', 1)[1]
    replace('## 摘要', '**关键词', """
## 摘要

针对乳腺病灶评分中局部征象组合的逐批更新问题，本文在固定顺序的FP树中维护带类别计数的列一致模式，并构建精简的三路融合模型FP-TBF-DI。三路分别采用联合筛选的Sugeno规则、类别条件距离修正和D/I稳定性复核，等权融合。基于1062条病灶记录，比较模型在模拟逐批、固定留出和完整评分向量分组五折中的表现。逐批平均准确率、平衡准确率和ROC-AUC分别为96.39%、96.08%和0.98999；更新相较冻结初始模型净增15次正确预测（3294次配对事件）。替换联合筛选分支后净减少19次正确预测，替换距离修正或D/I复核的影响较小。删除原C路三个历史复核模块后，8406次匹配预测的最终标签保持一致，但分数与AUC发生变化。最终留出中，模型准确率为96.71%，低于随机森林的97.50%；评分向量分组后，模型平衡准确率为96.14%，与RBF-SVM的96.18%接近。结果支持在本队列中简化历史复核结构，并表明联合规则筛选的作用主要体现在逐批融合。数据缺少患者标识及采集时间，结果不能作为患者级泛化或真实时间验证的依据。
""")
    replace('新增记录使这些问题更加突出。', '## 2 数据表示', """
随着带标签记录逐批增加，规则的覆盖数、相对支持度和类别纯度随之变化，满足阈值的规则集合也可能改变。FP树通过共享事务前缀压缩项目组合[6]；CanTree已使用固定项目顺序维护增量频繁模式[10]。另一方面，Huang等已将乳腺诊断双聚类用于Boosting和模糊分类[1,2]，FleBiC也将具有更一般一致性形式的双聚类用于生物医学关联分类[11]。本文处理的是离散评分中的列一致模式，研究重点是累计模式统计如何接入逐批分类，以及哪些判别组件在这一协议下提供可辨认的收益。

本文的工作包括：

（1）构建面向逐批评分记录的分类流程FP-TBF-DI。模式层累计维护类别计数，学习层在已揭示前缀上重筛规则和重新拟合；联合筛选、距离修正和稳定性复核分别给出判别分数。

（2）在匹配事件上检验结构简化。先对照相同Sugeno拟合上限下的筛选表现，再通过单项、逐层及联合删减，确定可移除的历史复核模块，分别检查标签、分数和排序指标。

（3）通过冻结模型、固定配置的常规模型、分支替换和评分向量分组分析，区分更新收益、组件收益和评价划分的影响。研究贡献是具体任务中的流程适配和可复核证据。固定顺序维护、Sugeno推理和约束投影均有已有方法作为基础[6–10]。
""")
    s = s.replace('FP-TBF相较该分支', 'FP-TBF-DI相较该分支')
    mark = '## 5 实验结果与分析'
    s = s.replace(mark, """### 4.8 扩展对照与评分向量分组

保持FP-TBF-DI的模型、阈值和预测事件不变，在相同累计训练前缀上比较六种固定配置：原始编码逻辑回归、独热逻辑回归、含18个主效应与153个两两交互的逻辑回归、RBF-SVM、300棵树的随机森林及100轮直方图梯度提升。交互逻辑回归沿用C路交互缩放，但不进行稳定性筛选；其余配置见[扩展实验协议](创新复核与扩展论证实验协议.md)。各基线不搜索超参数；森林与梯度提升以恶性概率0.5为阈值，其余以决策分数0为阈值。AUC按各模型自身的连续分数计算。

分别进行四种替换，其余两路及等权平均保持不变：A路替换为覆盖互补Sugeno分数；B路替换为累计逻辑分数，或改为以数值均方距离生成的Sugeno修正（规则与拟合上限保持一致，修正系数重新拟合）；C路替换为基础Sugeno分数。A路替换检验完整筛选分支的融合价值；其拟合上限与基础模型不同，筛选本身的同预算对照见第5.8节。另对各划分冻结初始300条记录所拟合的完整精简模型，在同一3294次逐批事件上对照累计更新。

将1062条记录按完整18维评分向量分为741组，实施一次分层分组五折。各组完整分配，保留同向量但标签不同的记录。每折训练顺序独立随机排列，前300条用于确定初始尺度，使用整折训练集拟合；各模型共享训练与测试索引。报告五折指标的等权均值。该分析检验完全相同评分向量跨训练和测试的问题，不能隔离患者；它也与逐批实验在训练规模和划分方式上不同，故不将两组指标差异归因于重复向量。配对计数均为预测事件，不用于推断独立患者的统计显著性。

""" + mark, 1)

    stream, holdout = matched['stream'], matched['holdout_849']
    baseline_names = dict(fp_tbf_di='FP-TBF-DI', raw_lr='原始编码逻辑回归',
                          onehot_lr='独热逻辑回归', interaction_lr='交互逻辑回归',
                          rbf_svm='RBF-SVM', random_forest='随机森林',
                          hist_gradient_boosting='直方图梯度提升')
    baselines = '\n'.join(f"| {name} | {values(stream[key])} | {values(holdout[key])} | {values(grouped[key])} |"
                          for key, name in baseline_names.items())
    replacement_names = dict(fp_tbf_di='FP-TBF-DI', A_coverage='A→覆盖互补Sugeno',
                             B_logistic='B→原始逻辑分数', B_numeric='B→数值距离修正',
                             C_base='C→基础Sugeno')

    def net(row):
        return f"{row['candidate_vs_main']['net']:+d}"

    replacements = '\n'.join(f"| {name} | {values(stream[key])} | {net(stream[key])} | {values(holdout[key])} | {net(holdout[key])} |"
                             for key, name in replacement_names.items())
    section = f"""### 5.11 常规模型与评分向量分组对照

**表15 同协议基线比较（每格依次为准确率/%、BA/%、AUC）**

| 模型 | 逐批：六配置均值 | 最终留出：六配置均值 | 评分向量分组：五折均值 |
|---|---:|---:|---:|
{baselines}

在逐批评价中，FP-TBF-DI的BA为96.08%，最接近的固定配置是交互逻辑回归（96.01%），融合净增5次正确预测。最终留出中，随机森林的准确率和BA分别为97.50%和97.17%，高于FP-TBF-DI的96.71%和96.59%；配对时随机森林纠正14次融合错误，同时新增4次错误。融合在最终留出的AUC较高（0.98876对0.98659），表明当前阈值的判断与整体排序得出了不同的模型比较结果。

评分向量分组后，五折的训练、测试均没有共同的完整向量，全部1062条记录各作为测试样本一次。FP-TBF-DI的BA为96.14%，略低于RBF-SVM的96.18%，两者的五折BA分别在2折和3折占优。融合准确率96.33%、AUC 0.98893，说明在本队列中，即使排除完全相同评分向量跨集合出现，分类仍有较高表现；这一结果不足以建立模型间的稳定优越性。

### 5.12 分支替换与冻结模型对照

**表16 固定其余分支后的替换结果（指标同表15）**

| 替换方案 | 逐批指标 | 逐批净增正确事件 | 最终留出指标 | 最终留出净增正确事件 |
|---|---:|---:|---:|---:|
{replacements}

净增量以FP-TBF-DI为参照。A路换为覆盖互补Sugeno后，逐批纠正13次错误，但新增32次错误，净减少19次正确预测；BA下降约0.51个百分点，六配置中五个下降、一个持平。最终留出净减少2次。B路去掉距离修正或改为数值距离，逐批均只净减少2次，最终留出标签均不变；去掉修正后的逐批AUC还略有提高。C路换为基础Sugeno后，逐批和最终留出分别净减少4次和2次正确预测。由此可见，A路作为完整分支对逐批融合的影响最清楚；现有数据仅支持B路和D/I提供有限的边际纠错，不能证明它们在其他队列中不可替代。

冻结模型的逐批准确率、BA和AUC分别为{values(frozen['frozen'])}（前两项单位为%）。相较冻结模型，累计更新纠正39次错误并新增24次，净增15次，BA提高约0.51个百分点。初始训练规模为300时，两种策略的分数在10⁻⁷容差内一致。该比较衡量增加带标签记录并重拟合的总体收益，无法将收益单独归因于FP树的维护方式。

"""
    s = s.replace('## 6 讨论', section + '## 6 讨论', 1)
    replace('### 6.4 研究局限', '## 参考文献', """
### 6.4 研究局限

首先，本研究使用单一且已用于开发的评分队列。固定留出及新增分组均是内部评价。评分向量分组排除了完全相同输入在训练与测试中同时出现，但不同向量仍可能来自同一患者。缺少患者标识、采集时间与独立来源数据，无法验证患者级泛化、跨中心迁移或真实时间漂移。完整字段字典和评分一致性也有待确认。

其次，常规模型比较只覆盖固定配置，未在相同协议下复现文献[1,2]的原始方法，不能据此声称优于最接近的研究。B路系数在训练内分数上拟合，D/I的稳定性条件及组件保留决定均使用了同一开发队列；已完成的消融支持特定删减，不证明D/I是跨队列必要组件。较小的配对差异应视为探索性观察，不能将重复预测事件当作独立患者来缩窄不确定性。

最后，累计树的精确计数、冻结对照的更新收益、以及计算效率是不同问题。本研究未测得完整三路系统的端到端时间或内存优势。模型仍需累计训练前缀和重拟合；输出为分类分数，尚未开展概率校准及临床效用验证。

## 7 结论

本文在固定顺序FP树和局部规则推理的已有框架上，构建了针对结构化病灶评分的逐批分类流程。P/E/U联合删减后，8406次配对预测的最终标签保持一致，但分支判断与连续分数仍可变化。冻结模型对照显示累计更新净增15次正确预测；分支替换显示联合筛选分支对逐批融合的作用较清楚，距离修正和D/I复核只提供小幅边际纠错。

FP-TBF-DI在逐批评价中具有较好的准确率和排序表现，但最终留出准确率低于随机森林，评分向量分组后的BA也未超过RBF-SVM。现有证据支持当前队列内的结构精简及流程可行性。后续应固定方法，优先开展患者级或独立队列验证、直接近邻的同协议复现与端到端成本评价。
""")
    s += """
[10] Leung C K S, Khan Q I, Li Z, Hoque T. CanTree: a canonical-order tree for incremental frequent-pattern mining[J]. Knowledge and Information Systems, 2007, 11(3): 287–311. [DOI:10.1007/s10115-006-0032-8](https://doi.org/10.1007/s10115-006-0032-8).

[11] Henriques R, Madeira S C. FleBiC: Learning classifiers from high-dimensional biomedical data using discriminative biclusters with non-constant patterns[J]. Pattern Recognition, 2021, 115: 107900. [DOI:10.1016/j.patcog.2021.107900](https://doi.org/10.1016/j.patcog.2021.107900).
"""
    target = PAPER / 'FP_TBF_DI_中文论文_证据复核稿.md'
    if target.exists() and target.read_text(encoding='utf-8') != s:
        raise FileExistsError(f'Refusing to overwrite manuscript edits: {target}')
    target.write_text(s, encoding='utf-8')


if __name__ == '__main__':
    main()
