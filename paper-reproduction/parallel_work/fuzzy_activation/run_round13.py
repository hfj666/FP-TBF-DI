"""Compare fixed logistic/HVDM output heads on the existing online splits."""
import json
import hashlib
from collections import Counter

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from diagnose_round05 import HERE, ROOT, OLD, LOGIC, load_state, read_predictions
from run_component import digest, load_stream, write_csv, metrics
from activation import restore
from audit_round12 import antecedents
from src.reference_comparators import CumulativeLogistic

OUT = HERE/'results_13'
MODES = ('logistic_offset', 'logistic_score_feature', 'logistic_rule_weights')


def offset_fit(base, feature, y):
    n = len(y)

    def objective(a):
        score = base + feature*a[0]
        loss = np.mean(np.logaddexp(0,score)-y*score) + .5*a[0]**2/n
        grad = np.mean((expit(score)-y)*feature) + a[0]/n
        return loss,np.array([grad])

    return minimize(objective,[0.],jac=True,method='L-BFGS-B',options={'maxiter':500})


def main():
    assert not (OUT/'fits.json').exists()
    OUT.mkdir(exist_ok=True)
    hvdm = read_predictions(OLD/'predictions.csv','hvdm_mean')
    logic = read_predictions(LOGIC,'logistic_cumulative')
    shared = read_predictions(HERE/'results_12/predictions.csv','shared_linear')
    assert hvdm.keys()==logic.keys()==shared.keys()
    disagreements = []
    for k,(y,s) in shared.items():
        ly,l = logic[k]
        hy,h = hvdm[k]
        assert y==ly==hy
        if (s>=0)!=(l>=0):
            disagreements.append(dict(key=k,label=y,shared=s,logistic=l,hvdm=h))
    (OUT/'shared_logistic_disagreements.json').write_text(json.dumps(disagreements,indent=2))
    audit = {(r['seed'],r['batch_size'],r['stage']):r['sha256']
             for r in json.loads((OLD/'state_audit.json').read_text())}
    files = [HERE/'PROTOCOL_13.md', HERE/'run_round13.py', HERE/'activation.py',
             HERE/'audit_round12.py', ROOT/'src/reference_comparators.py', LOGIC,
             OLD/'state_audit.json', OLD/'predictions.csv', HERE/'results_12/predictions.csv']
    (OUT/'protocol.json').write_text(json.dumps(dict(threshold=0, modes=MODES, hashes={
        str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}),indent=2))
    predictions = {k:{} for k in MODES}
    fits, events = [], []
    max_ref_error, max_replay_error = 0.,0.
    for split in json.loads((ROOT/'results/online_study/splits.json').read_text()):
        seed = split['seed']
        ids = np.array(split['train_stream'])
        assert len(ids)==len(set(ids))==849 and not set(ids)&set(split['holdout'])
        x,y = load_stream(ROOT/'data/dataset_1062.txt',ids)
        for batch in (50,100):
            for stage,end in enumerate(range(300,849,batch)):
                stop = min(end+batch,849)
                record = load_state(seed,batch,stage,audit)
                assert record['prefix_input_sha256']==digest(x[:end],y[:end],ids[:end])
                snap = record['models']['hvdm_mean']
                z = (x[:stop]-snap['minimum'])/snap['span']
                model = restore(snap)
                w = model.firing_weights(x[:stop])
                score = w@model.theta[64:]
                feature = score/10
                joint = [np.c_[z,feature],np.c_[z,w]]
                if stage or batch==50:
                    log = CumulativeLogistic(x[:end],y[:end],(snap['minimum'],snap['span']))
                    res = offset_fit(log.decision_function(x[:end]),feature[:end],y[:end]==1)
                    assert res.success
                    assert np.linalg.norm(res.jac,np.inf)<1e-5
                    heads = [LogisticRegression(C=1.,solver='lbfgs',max_iter=500).fit(d[:end],y[:end]) for d in joint]
                    assert all(h.n_iter_.max()<500 for h in heads)
                    fits.append(dict(seed=seed,batch=batch if stage else 0,stage=stage,prefix=end,
                                     offset=float(res.x[0]),offset_iterations=int(res.nit),
                                     logistic_beta=log.model.coef_[0].tolist(),logistic_bias=float(log.model.intercept_[0]),
                                     heads=[dict(beta=h.coef_[0].tolist(),bias=float(h.intercept_[0]),iterations=int(h.n_iter_[0])) for h in heads]))
                    if not stage:
                        initial = snap,log,res,heads
                else:
                    previous,log,res,heads = initial
                    assert previous==snap
                baseline = log.decision_function(x[end:stop])
                tested = [baseline+feature[end:stop]*res.x[0],
                          *(h.decision_function(d[end:stop]) for h,d in zip(heads,joint,strict=True))]
                independent = antecedents(snap,x[end:stop])
                independent_score = independent @ np.array(snap['theta'][64:])/10
                linear = z[end:stop] @ log.model.coef_[0] + log.model.intercept_[0]
                replayed = [linear + independent_score*res.x[0],
                            *(d @ h.coef_[0] + h.intercept_[0] for h,d in
                              zip(heads,(np.c_[z[end:stop],independent_score],np.c_[z[end:stop],independent]),strict=True))]
                max_replay_error = max(max_replay_error,float(np.max(np.abs(np.array(tested)-replayed))))
                for j,i in enumerate(range(end,stop)):
                    key = seed,batch,stage+1,int(ids[i])
                    label = int(y[i])
                    assert label==hvdm[key][0]==logic[key][0]
                    max_ref_error = max(max_ref_error,abs(baseline[j]-logic[key][1]),abs(score[i]-hvdm[key][1]))
                    for mode,values in zip(MODES,tested,strict=True):
                        value = float(values[j])
                        predictions[mode][key] = label,value
                        events.append(dict(seed=seed,batch_size=batch,stage=stage+1,row_index=int(ids[i]),
                                           label=label,score=value,policy=mode))
    assert len(fits)==48 and len(events)==9882 and max_ref_error<1e-10 and max_replay_error<1e-12, (len(fits),len(events),max_ref_error,max_replay_error)
    (OUT/'fits.json').write_text(json.dumps(fits,separators=(',',':')))
    write_csv(OUT/'predictions.csv',events)
    groups = sorted({k[:2] for k in logic})
    arms = {'累计逻辑':logic,'原 HVDM':hvdm,'共享线性':shared,**predictions}
    results = {}
    for name,arm in arms.items():
        assert arm.keys()==logic.keys()
        config = []
        for g in groups:
            vals = np.array([v for k,v in arm.items() if k[:2]==g])
            assert len(vals)==549
            m = metrics(*vals.T)
            m['ba'] = (m['sensitivity']+m['specificity'])/2
            config.append(dict(seed=g[0],batch=g[1],**m))
        average = {metric:float(np.mean([c[metric] for c in config])) for metric in ('accuracy','ba','auc')}
        pairs = {}
        for ref in ('累计逻辑','原 HVDM','共享线性'):
            base = arms[ref]
            pairs[ref] = dict(new_fn=[k for k,(v,s) in arm.items() if v==1 and s<0<=base[k][1]],
                              corrected_fn=[k for k,(v,s) in arm.items() if v==1 and base[k][1]<0<=s])
        results[name] = dict(average=average,configs=config,pairs=pairs)
    (OUT/'verification.json').write_text(json.dumps(dict(results=results,max_ref_error=max_ref_error,
             max_replay_error=max_replay_error,unique_fits_per_head=len(fits)),ensure_ascii=False,indent=2),encoding='utf-8')
    counts = Counter((r['label'],(r['logistic']>=0)==(r['label']==1)) for r in disagreements)
    lines = ['# 累计逻辑与 HVDM 输出层实验','',
             f"共享线性与累计逻辑分歧 {len(disagreements)} 次。累计逻辑独对的正/负类事件为 "
             f"{counts[1,True]}/{counts[-1,True]}，共享线性独对为 {counts[1,False]}/{counts[-1,False]}。",'',
             '|方法|Accuracy|BA|AUC|对累计逻辑新增/修正漏报|对共享线性新增/修正漏报|',
             '|---|---:|---:|---:|---|---|']
    for name,r in results.items():
        a=r['average']
        pairs=['/'.join(str(len(r['pairs'][ref][k])) for k in ('new_fn','corrected_fn')) for ref in ('累计逻辑','共享线性')]
        lines.append(f"|{name}|{a['accuracy']:.6%}|{a['ba']:.6%}|{a['auc']:.8f}|"+'|'.join(pairs)+'|')
    lines += ['',f'固定阈值 0。预测前缀内拟合 {len(fits)} 次每臂，预测重放最大误差 {max_replay_error:.3g}。'
              '累计逻辑与 HVDM 逐事件复现。未读取各划分 213 条留出记录。结果是内部探索。']
    (OUT/'输出层深入对照.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))


if __name__=='__main__':
    main()


