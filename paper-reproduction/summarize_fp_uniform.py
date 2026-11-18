"""Independent state-score replay and paired UR report."""
import csv,gzip,hashlib,json
import numpy as np
from src.models import Rule,SparseSugeno
from run_original_closeout import load_stream
from run_experiments import ROOT,metrics
from run_fp_uniform import OUT

def main():
    read=lambda folder,n:list(csv.DictReader((folder/(n+'.csv')).open(encoding='utf-8-sig',newline='')))
    p=json.loads((OUT/'protocol.json').read_text())
    for path,h in p['hashes'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h
    pred=read(OUT,'predictions');summary=read(OUT,'summary');fits=read(OUT,'fits');old=read(ROOT/'results/fp_selection_bridge','predictions');replayed=0;diffs=[]
    for sp in json.loads((ROOT/'results/online_study/splits.json').read_text()):
        x,y=load_stream(ROOT/'data/dataset_1062.txt',sp['train_stream'])
        for b in (50,100):
            ba={}
            for arm in ('baseline','ur'):
                rr=[r for r in pred if int(r['seed'])==sp['seed'] and int(r['batch_size'])==b and r['policy']==arm]
                assert [int(r['row_index']) for r in rr]==sp['train_stream'][300:];np.testing.assert_array_equal([int(r['label']) for r in rr],y[300:])
                s=np.array([float(r['score']) for r in rr]);calc=metrics(y[300:],s);r=next(r for r in summary if int(r['seed'])==sp['seed'] and int(r['batch_size'])==b and r['policy']==arm)
                for k,v in calc.items():np.testing.assert_allclose(v,float(r[k]),atol=1e-12)
                r['ba']=(calc['sensitivity']+calc['specificity'])/2;ba[arm]=r['ba']
                if arm=='baseline':np.testing.assert_array_equal(s,[float(r['score']) for r in old if int(r['seed'])==sp['seed'] and int(r['batch_size'])==b and r['policy']=='coverage_cumulative'])
            diffs.append(ba['ur']-ba['baseline'])
            for a in json.loads((OUT/'audit.json').read_text()):
                if a['seed']!=sp['seed'] or a['batch_size']!=b:continue
                raw=gzip.open(OUT/f"state_s{sp['seed']}_b{b}_t{a['stage']}.json.gz",'rt',encoding='utf-8').read();assert hashlib.sha256(raw.encode()).hexdigest()==a['sha256'];record=json.loads(raw);end=record['prefix']
                assert record['models']['ur']['rules']==record['models']['baseline']['rules']
                for arm,s in record['models'].items():
                    m=SparseSugeno([Rule(tuple(r['columns']),np.array(r['values']),r['label'],r['confidence'],r['count']) for r in s['rules']]);m.minimum=np.array(s['minimum']);m.span=np.array(s['span']);m.theta=np.array(s['theta'])
                    if end<849:
                        rr=[r for r in pred if int(r['seed'])==sp['seed'] and int(r['batch_size'])==b and int(r['stage'])==a['stage']+1 and r['policy']==arm]
                        np.testing.assert_allclose(m.decision_function(x[end:min(end+b,849)]),[float(r['score']) for r in rr],atol=1e-12,rtol=1e-12);replayed+=len(rr)
    agg={a:{k:float(np.mean([float(r[k]) for r in summary if r['policy']==a])) for k in ('ba','sensitivity','specificity','auc')} for a in ('baseline','ur')}
    lines=['# 文献组件UR移植：固定对照结果','', '仅将Cui等2019预印本/2020论文的均匀正则Eq7接入现有零阶BCE Sugeno；不是作者完整模型复现。论文使用mini-batch/AdaBound/一阶后件等，本轮保留原FP/覆盖α.5/初始尺度/500 L-BFGS/原后件正则，只加UR。系数.1固定为论文候选网格最小值，不称最优，不按本轮分数调参。','', '|组|BA|敏感度|特异度|AUC|','|---|---:|---:|---:|---:|']
    for a,v in agg.items():lines.append(f"|{a}|{v['ba']:.4%}|{v['sensitivity']:.4%}|{v['specificity']:.4%}|{v['auc']:.6f}|")
    lines+=['',f'UR相对基线BA均差{100*np.mean(diffs):+.4f}个百分点；{sum(d>1e-12 for d in diffs)}胜/{sum(abs(d)<=1e-12 for d in diffs)}平/{sum(d< -1e-12 for d in diffs)}负。','', '|组|训练前缀平均有效规则数|最大平均激活|','|---|---:|---:|']
    for a in ('baseline','ur'):
        rr=[r for r in fits if r['policy']==a];lines.append(f"|{a}|{np.mean([float(r['effective_rules']) for r in rr]):.4f}|{np.mean([float(r['max_mass']) for r in rr]):.6f}|")
    failure=[r for r in fits if r['success']!='True']
    lines+=['',f'{len(fits)}拟合，未报告成功{len(failure)}次；{replayed}预测由保存参数全部重放通过，基线与旧覆盖累计分数精确一致。源/state hash、规则一致、病例顺序和指标核验。有效规则数=1/sum(mean firing²)，为训练描述，不是患者泛化证据。单次耗时保存在fits.csv，不作速度声明。','', 'tau=.5按原文1/C，并非1/R；因归一化权重和为1，改成1/R仅改变常数，梯度等价。UR控制平均规则使用，并不保证每病例正确或条件冗余消失。各自213本轮不解析/评分，历史已接触；内部重复开发，不能用此结果宣称独立或临床优越性。']
    (OUT/'文献UR移植报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (OUT/'verification.json').write_text(json.dumps(dict(verified=True,replayed=replayed,failures=failure,aggregate=agg,ba_deltas=diffs),indent=2),encoding='utf-8')
    print(json.dumps(agg,indent=2));print('delta',float(np.mean(diffs)))

if __name__=='__main__':main()
