"""Replay each alternative and compare aligned threshold-zero outcomes."""
import hashlib
import json

import numpy as np

from scipy.special import softmax
from diagnose_round05 import LOGIC, OLD, load_state, read_predictions
from diagnose_round10 import OUT as DIAG, RESERVED, RESERVED_POLICY
from diagnose_round08 import OUT as HARMONIC
from run_component import ROOT, digest, load_stream, metrics
from run_round12 import OUT, MODES as METHODS


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def antecedents(snap, x):
    q = {(j,v):p/n for j,v,n,p in snap['table']}
    distance = []
    for rule in snap['raw_rules']:
        parts = [[(q[j,int(v)]-q[j,int(center)])**2 if (j,int(v)) in q else 1.
                  for v in x[:,j]] for j,center in zip(rule['columns'],rule['values'],strict=True)]
        distance.append(np.mean(parts,axis=0))
    return softmax(np.log([r['confidence'] for r in snap['raw_rules']])
                   - .5*np.column_stack(distance)/np.exp(2*np.array(snap['theta'][:64])),axis=1)

def main():
    paths = {'累计逻辑': (LOGIC,'logistic_cumulative'),
             '原 HVDM': (OLD/'predictions.csv','hvdm_mean'),
             '上轮固定负类': (DIAG/'experiment/predictions.csv','hvdm_fixed_negative'),
             **{m: (OUT/'predictions.csv',m) for m in METHODS}}
    predictions = {k:read_predictions(*v) for k,v in paths.items()}
    ref = predictions['原 HVDM']
    for arm in predictions.values():
        assert arm.keys() == ref.keys() and len(arm) == 3294
        assert all(arm[k][0] == ref[k][0] for k in ref)
    audit = {(r['seed'],r['batch_size'],r['stage']):r['sha256'] for r in read(OLD/'state_audit.json')}
    fits = {}
    checks = {m: dict(scores=0, objectives=0, score_error=0., loss_error=0., failures=[]) for m in METHODS}
    for m in METHODS:
        proto = read(OUT/'protocol.json')
        assert proto['threshold']==0
        for name,h in proto['hashes'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h
        stored = [r for r in read(OUT/'fits.json') if r['mode']==m]
        fits[m] = {(r['seed'],r['batch'],r['stage']):r for r in stored}
        assert len(fits[m]) == len(stored) == 54
        checks[m]['failures'] = [(r['seed'],r['batch'],r['stage'],r['message'])
                                for r in stored if not r['success']]
    configs = []
    for sp in read(ROOT/'results/online_study/splits.json'):
        seed = sp['seed']
        ids = np.array(sp['train_stream'])
        assert len(ids)==len(set(ids))==849 and not set(ids)&set(sp['holdout'])
        x,y = load_stream(ROOT/'data/dataset_1062.txt',ids)
        for batch in (50,100):
            configs.append((seed,batch))
            for stage,end in enumerate([*range(300,849,batch),849]):
                state = load_state(seed,batch,stage,audit)
                assert state['prefix_input_sha256'] == digest(x[:end],y[:end],ids[:end])
                old = state['models']['hvdm_mean']
                fit_key = seed,batch if stage else 0,stage
                stop = min(end+batch,849)
                weights = antecedents(old, x[:stop])
                for m in METHODS:
                    fit = fits[m][fit_key]
                    beta = np.array(fit['beta'])
                    assert fit['iterations'] <= 500 and np.isfinite(beta).all() and np.all(np.abs(beta)<=10)
                    z = (x[:stop] - old['minimum']) / old['span']
                    if m == 'local_linear':
                        params = beta.reshape(64,19)
                        scores = np.sum(weights * (params[:,0] + z @ params[:,1:].T), axis=1)
                        penalty = .001/64 * np.sum(beta**2)
                    else:
                        features = z if m=='shared_linear' else np.c_[z,(z-.5)**2]
                        scores = weights @ beta[:64] + features @ beta[64:]
                        penalty = .001/64 * np.sum(beta[:64]**2) + .001 * np.sum(beta[64:]**2)
                    train = scores[:end]
                    loss = np.mean(np.logaddexp(0,train)-(y[:end]==1)*train)+penalty
                    check = checks[m]
                    check['objectives'] += 1
                    check['loss_error'] = max(check['loss_error'],abs(loss-fit['training_after']))
                    for i in range(end,stop):
                        label,score = predictions[m][seed,batch,stage+1,int(ids[i])]
                        assert label==y[i]
                        check['scores'] += 1
                        check['score_error'] = max(check['score_error'],abs(scores[i]-score))
    for check in checks.values():
        assert check['scores']==3294 and check['objectives']==57
        assert max(check['loss_error'],check['score_error'])<1e-12
    results = {}
    for name,arm in predictions.items():
        results[name] = {}
        for seed,batch in configs:
            y,s = np.array([v for k,v in arm.items() if k[:2]==(seed,batch)]).T
            assert len(y)==549
            mm = metrics(y,s)
            results[name][f'{seed}/{batch}'] = dict(accuracy=mm['accuracy'],ba=(mm['sensitivity']+mm['specificity'])/2,
                                                    auc=mm['auc'],fn=mm['fn'])
    averages = {name:{k:float(np.mean([v[k] for v in rows.values()])) for k in ('accuracy','ba','auc')}
                for name,rows in results.items()}
    lines = ['# 固定 HVDM 激活的输出层准确率对照', '',
             '三种方案各自沿用原 HVDM 规则与激活，单独替换输出层。'
             '六配置宏平均，阈值固定为 0。漏报按配对预测事件计，重叠配置可能包含同一数据行。', '',
             '|方法|平均 Accuracy|平均 BA|平均 AUC|总漏报|', '|---|---:|---:|---:|---:|']
    for name,v in averages.items():
        lines.append(f"|{name}|{v['accuracy']:.6%}|{v['ba']:.6%}|{v['auc']:.8f}|{sum(r['fn'] for r in results[name].values())}|")
    lines += ['', '|方案|参照|Accuracy 胜/平/负|ΔAccuracy（百分点）|ΔAUC|新增漏报|修正漏报|',
              '|---|---|---|---:|---:|---:|---:|']
    pairs = {}
    for name in (*METHODS,'原 HVDM'):
        arm = predictions[name]
        pairs[name] = {}
        for compare in ('原 HVDM','累计逻辑','上轮固定负类'):
            if name==compare:
                continue
            baseline = predictions[compare]
            added = sorted(k for k,(y,s) in arm.items() if y==1 and s<0<=baseline[k][1])
            fixed = sorted(k for k,(y,s) in arm.items() if y==1 and baseline[k][1]<0<=s)
            d = np.array([v['accuracy']-results[compare][c]['accuracy'] for c,v in results[name].items()])
            wins = f'{sum(d>0)}/{sum(d==0)}/{sum(d<0)}'
            db = 100*(averages[name]['accuracy']-averages[compare]['accuracy'])
            da = averages[name]['auc']-averages[compare]['auc']
            pairs[name][compare] = dict(new_fn=added,corrected_fn=fixed,wins=wins,accuracy_delta_pp=db,auc_delta=da)
            lines.append(f'|{name}|{compare}|{wins}|{db:+.6f}|{da:+.8f}|{len(added)}|{len(fixed)}|')
    lines += ['', '|配置|方法|Accuracy|BA|AUC|漏报|','|---|---|---:|---:|---:|---:|']
    for c in results['原 HVDM']:
        for name,rows in results.items():
            m = rows[c]
            lines.append(f"|{c}|{name}|{m['accuracy']:.6%}|{m['ba']:.6%}|{m['auc']:.8f}|{m['fn']}|")
    lines += ['', '|此前去重事件组|事件数|'+'|'.join(METHODS)+'|',
              '|---|---:|---:|---:|---:|']
    diagnostic = read(DIAG/'diagnosis.json')
    for group in ('persistent','lost_fix','new_fn'):
        keys = [tuple(r['key']) for r in diagnostic if group in r['groups']]
        lines.append(f'|{group}|{len(keys)}|'+'|'.join(str(sum(predictions[m][k][1]>=0 for k in keys)) for m in METHODS)+'|')
    lines += ['', '上表列的是各方案判阳事件数，组间可能重叠。各方案的逐事件新增和修正漏报键见 verification.json。'
              '各划分 213 条留出未读取。本轮是既有流上的探索。']
    for m,c in checks.items():
        lines.append(f"- {m} 重放 {c['scores']} 条，最大误差 {c['score_error']:.3g}。"
                     f"重算 {c['objectives']} 个训练目标，未报告收敛成功 {len(c['failures'])} 次。")
    (OUT/'输出层准确率对照.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (OUT/'verification.json').write_text(json.dumps(dict(averages=averages,results=results,pairs=pairs,checks=checks),
                                                   ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(averages=averages, paired={m:{r:[len(t['new_fn']),len(t['corrected_fn']),t['wins']]
          for r,t in pairs[m].items()} for m in METHODS},checks=checks),ensure_ascii=False))


if __name__=='__main__':
    main()

