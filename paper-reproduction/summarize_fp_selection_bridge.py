"""Audit bridge artifacts only; do not parse raw cases."""
import csv,gzip,hashlib,json
import numpy as np
from run_experiments import ROOT,metrics

def main():
    out=ROOT/'results/fp_selection_bridge'
    read=lambda n:list(csv.DictReader((out/(n+'.csv')).open(encoding='utf-8-sig',newline='')))
    protocol=json.loads((out/'protocol.json').read_text())
    for p,h in protocol['hashes'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
    preds=read('predictions');summary=read('summary');updates=read('updates');initial=read('initial')
    splits=json.loads((ROOT/'results/online_study/splits.json').read_text());groups={};agg={}
    names=sorted({r['policy'] for r in summary})
    for sp in splits:
        for b in (50,100):
            for p in names:
                rr=[r for r in preds if int(r['seed'])==sp['seed'] and int(r['batch_size'])==b and r['policy']==p]
                assert [int(r['row_index']) for r in rr]==sp['train_stream'][300:]
                assert not set(int(r['row_index']) for r in rr)&set(sp['holdout'])
                m=metrics(np.array([int(r['label']) for r in rr]),np.array([float(r['score']) for r in rr]))
                saved=next(r for r in summary if int(r['seed'])==sp['seed'] and int(r['batch_size'])==b and r['policy']==p)
                for k,v in m.items():np.testing.assert_allclose(v,float(saved[k]),atol=1e-12)
                m['ba']=(m['sensitivity']+m['specificity'])/2;groups[sp['seed'],b,p]=m
    for a in json.loads((out/'audit.json').read_text()):
        raw=gzip.open(out/f"state_s{a['seed']}_b{a['batch_size']}_t{a['stage']}.json.gz",'rt',encoding='utf-8').read();assert hashlib.sha256(raw.encode()).hexdigest()==a['sha256']
        record=json.loads(raw)
        for p,m in record['models'].items():assert np.isfinite(m['theta']).all() and len(m['rules'])==64
        for sel in ('original','coverage'):
            assert record['models'][sel+'_local']['rules']==record['models'][sel+'_cumulative']['rules']
    lines=['# 已有覆盖互补筛选的统一FP接回对照','', '固定α=0.5，不调参。两种筛选各自初始化、保留冻结参照；共同初始300例冻结缩放。local沿用已有lr/active、仅新批次更新参数；cumulative冷启动联合拟合500次。两条训练路径同时改变历史使用、优化器与初始化，不能当作纯数据窗口因果效应。覆盖筛选扫描已揭示历史，local整条管线也不是无需历史。','', '|方法|平均BA|敏感度|特异度|AUC|','|---|---:|---:|---:|---:|']
    for p in names:
        rr=[v for (s,b,n),v in groups.items() if n==p];agg[p]={k:float(np.mean([m[k] for m in rr])) for k in ('ba','sensitivity','specificity','auc')};v=agg[p]
        lines.append(f"|{p}|{v['ba']:.4%}|{v['sensitivity']:.4%}|{v['specificity']:.4%}|{v['auc']:.6f}|")
    lines+=['','|比较（前者减后者）|BA平均差/百分点|胜出配置|','|---|---:|---:|']
    for a,b in [('coverage_local','original_local'),('coverage_cumulative','original_cumulative'),('coverage_frozen','original_frozen'),('coverage_local','coverage_frozen'),('coverage_cumulative','coverage_frozen'),('original_local','original_frozen'),('original_cumulative','original_frozen')]:
        ds=[groups[sp['seed'],bb,a]['ba']-groups[sp['seed'],bb,b]['ba'] for sp in splits for bb in (50,100)]
        lines.append(f'|{a} - {b}|{100*np.mean(ds):+.4f}|{sum(d>0 for d in ds)}/6|')
    failures=[r for r in initial+updates if r.get('success')=='False']
    lines+=['','## 六配置BA','', '|seed/batch|'+'|'.join(names)+'|','|---|'+'---:|'*len(names)]
    for sp in splits:
        for b in (50,100):lines.append(f"|{sp['seed']}/{b}|"+'|'.join(f"{groups[sp['seed'],b,p]['ba']:.4%}" for p in names)+'|')
    lines+=['',f'{len(preds)}条预测，{len(updates)}次更新，51份状态；优化未成功记录{len(failures)}次（含初始化），保留有限结果，未追加预算。单次运行，不作速度优势结论。', '', '本轮每划分213例未解析/训练/预测/评分；历史全表交叉验证及部分旧留出评价已经使用过这些病例，因此不是从未接触的独立测试集。划分重叠、数据反复开发，结果限于内部探索。覆盖互补是已有方法整合，不是本轮新发明。']
    # Check original arms reproduce prior unified direct comparison scores exactly.
    old=list(csv.DictReader((ROOT/'results/direct_comparison_v2/predictions.csv').open(encoding='utf-8-sig')))
    for new,previous in [('original_local','fp_local'),('original_cumulative','fp_weighted_cumulative'),('original_frozen','fp_frozen')]:
        for sp in splits:
            for b in (50,100):
                a=[float(r['score']) for r in preds if r['policy']==new and int(r['seed'])==sp['seed'] and int(r['batch_size'])==b]
                z=[float(r['score']) for r in old if r['policy']==previous and int(r['seed'])==sp['seed'] and int(r['batch_size'])==b]
                if new!='original_cumulative':np.testing.assert_array_equal(a,z)
                else:np.testing.assert_array_equal(np.array(a)>=0,np.array(z)>=0)
    lines+=['','原始local与frozen逐例评分精确重现直接对照；原始累计原始行拟合与旧FP分组拟合仅核对分类标签一致，不要求浮点分数一致。源hash、状态hash、同筛选两训练路径规则一致、病例顺序与指标均已核验。']
    (out/'接回对照报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (out/'aggregate.json').write_text(json.dumps(agg,indent=2),encoding='utf-8')
    (out/'verification.json').write_text(json.dumps(dict(verified=True,predictions=len(preds),updates=len(updates),failures=failures,holdout_evaluated_this_run=0),indent=2),encoding='utf-8')
    print(json.dumps(agg,indent=2));print('verified')

if __name__=='__main__':main()
