# 02 标准minimum聚合：事前固定协议

2026-11-04，用户要求保留HVDM并继续研究，争取超过累计逻辑基线。HVDM全部文件保留。此项单独从原SparseSugeno数值距离基线出发，不与HVDM或其他新组件组合。

源：Farahbod & Eftekhari, Comparison of different T-norm operators in classification problems, arXiv1208.1955v1，§2 Table1 minimum、§1中两条件规则、§3对照。全文已保存在literature_archive/tnorm_comparison.pdf并记录hash；AdaTSK arXiv2201.03187v1 §II-B作为minimum/softmin的背景。原论文并非本地管线完整复现，也没有核验到作者源码。只移植标准minimum，不借用其整个系统的性能排序。

唯一变化：原D_r²=mean_j[(z_j-v_rj)²]变为max_j[(z_j-v_rj)²]，仍logfire=logconfidence-.5*D_r²*exp(-2logwidth)，softmax归一、零阶后件。等价min_j Gaussian隶属。动机是L2/3规则中的单个违背条件不被均值稀释；不是高维防下溢。每条规则宽度共享且中心固定，max距离与可训练宽度独立，所以仍可使用原width梯度；不引入softmin温度。L1必须等价。

已有activation.py numeric_max实现及run_component.py minimum入口未做过病例实测，此次直接使用，保留既有文件和HVDM源hash。通过合成min/max及L1等价后运行。全流程保留L3/support.1/purity>.65/min5/64每类32、CoveragePool alpha.5、初始300冻结尺度、宽度.25/[.03,2]、后件2label/[-10,10]、原BCE/L2和冷拟合500。只解析每划分授权849，seed17/29/43×batch50/100，各549整批先预测后更新，不用各自213。阈值固定0，绝不按结果选阈值或网格。末批后拟合仍计成本。

比较原coverage_cumulative并复核保存评分；并列旧HVDM、累计逻辑的同ID/标签评分。不改变任何对照设置，不重训累计逻辑。报告全部六配置accuracy、BA、sensitivity、specificity、AUC、配对纠错/新增错/新增漏报及成本，新增漏报同时相对原基线及累计逻辑计。

延续工程筛选：相对原基线同一方法平均accuracy/BA均升且BA>=4/6胜出才为候选；另外明确是否平均accuracy和BA超过累计逻辑，并报告六配置胜负。AUC下降、任何新增漏报均突出，不因通过一个指标而隐去。该标准不构成独立验证或临床非劣性。若失败，不改max或宽度再试；只有新的独立文献机制才另写协议。若有超过逻辑的候选，完成审计及必要匹配控制后保留，不叠加追分。

验证源码/输入/状态hash、每状态同规则、初始尺度不变、预测顺序/标签/混淆矩阵、评分及训练目标独立重放。不能宣称本次复用规则筛选消除其前置成本。新结果写results_02，存在即拒绝覆盖。
