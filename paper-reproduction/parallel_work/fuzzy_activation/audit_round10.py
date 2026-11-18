"""Independently replay fitted scores and compare matched false negatives."""
import hashlib
import json
from collections import Counter

import numpy as np

from diagnose_round05 import LOGIC, OLD, load_state
from diagnose_round10 import OUT, RESERVED, RESERVED_POLICY, read_predictions
from run_component import ROOT, digest, load_stream, metrics
from run_round10 import POLICY


def replay(snapshot, theta, x):
    posterior = {(j, value): pos / n for j, value, n, pos in snapshot['table']}
    rules = snapshot['raw_rules']
    distances = []
    for rule in rules:
        parts = []
        for j, center in zip(rule['columns'], rule['values'], strict=True):
            q = posterior[j, int(center)]
            parts.append([1. if (j, int(v)) not in posterior else (posterior[j, int(v)] - q)**2
                          for v in x[:, j]])
        distances.append(np.mean(parts, axis=0))
    logfire = np.log([r['confidence'] for r in rules]) - (
        np.column_stack(distances) / (2 * np.exp(2 * theta[:64])))
    strengths = np.exp(logfire - logfire.max(axis=1, keepdims=True))
    return (strengths @ theta[64:]) / strengths.sum(axis=1)

def main():
    run = OUT / 'experiment'
    protocol = json.loads((run / 'protocol.json').read_text(encoding='utf-8'))
    assert protocol['threshold'] == 0
    for name, expected in protocol['hashes'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
    paths = {'HVDM': (OLD / 'predictions.csv', 'hvdm_mean'),
             '累计逻辑': (LOGIC, 'logistic_cumulative'),
             'HVDM + 固定负类': (run / 'predictions.csv', POLICY),
             '保留半数': (RESERVED / 'experiment/predictions.csv', RESERVED_POLICY)}
    preds = {method: read_predictions(*source) for method, source in paths.items()}
    hvdm, logic, candidate, reserved = preds.values()
    assert hvdm.keys() == logic.keys() == candidate.keys() == reserved.keys() and len(candidate) == 3294
    assert all(hvdm[k][0] == logic[k][0] == candidate[k][0] == reserved[k][0] for k in candidate)
    stored = json.loads((run / 'fits.json').read_text(encoding='utf-8'))
    reserved_fits = {(r['seed'],r['batch'],r['stage']):r['snapshot']
                     for r in json.loads((RESERVED / 'experiment/fits.json').read_text())}
    fits = {(r['seed'], r['batch'], r['stage']): r for r in stored}
    assert len(fits) == len(stored) == 54
    audit = {(r['seed'], r['batch_size'], r['stage']): r['sha256']
             for r in json.loads((OLD / 'state_audit.json').read_text(encoding='utf-8'))}
    count, max_error, max_loss_error = 0, 0., 0.
    config = []
    diagnostic_matches = {}
    diagnosis = json.loads((OUT / 'diagnosis.json').read_text(encoding='utf-8'))
    diagnostic_keys = {tuple(r['key']) for r in diagnosis}
    for split in json.loads((ROOT / 'results/online_study/splits.json').read_text(encoding='utf-8')):
        seed = split['seed']
        ids = np.array(split['train_stream'])
        assert len(ids) == len(set(ids)) == 849 and not set(ids) & set(split['holdout'])
        x, y = load_stream(ROOT / 'data/dataset_1062.txt', ids)
        for batch in (50, 100):
            config.append((seed, batch))
            for stage, end in enumerate([*range(300, 849, batch), 849]):
                record = load_state(seed, batch, stage, audit)
                assert record['prefix_input_sha256'] == digest(x[:end], y[:end], ids[:end])
                fit = fits[seed, batch if stage else 0, stage]
                snap = fit['snapshot']
                opt = snap['optimization']
                assert fit['prefix'] == end and opt['iterations'] <= 500 and opt['transferred_rules'] == 0
                theta = np.array(snap['theta'])
                assert np.isfinite(theta).all() and theta.shape == (128,)
                assert np.all(theta[:64] >= np.log(.03)) and np.all(theta[:64] <= np.log(2))
                assert np.all(np.abs(theta[64:]) <= 10)
                original = record['models']['hvdm_mean']
                assert all(snap[k] == original[k] for k in ('minimum','span','table','mode'))
                assert sum(r['label'] == 1 for r in snap['raw_rules']) == 32
                assert len(snap['raw_rules']) == 64
                reserve = reserved_fits[seed, batch if stage else 0, stage]
                assert [r for r in snap['raw_rules'] if r['label'] == -1] == [r for r in original['raw_rules'] if r['label'] == -1]
                assert [r for r in snap['raw_rules'] if r['label'] == 1] == [r for r in reserve['raw_rules'] if r['label'] == 1]
                for rule in snap['raw_rules']:
                    exact = np.all(x[:end, rule['columns']] == rule['values'], axis=1)
                    n = int(exact.sum())
                    own = int(np.sum(exact & (y[:end] == rule['label'])))
                    assert n == rule['count'] and n >= max(5, np.ceil(.1 * end))
                    assert own / n == rule['confidence'] > .65
                train = replay(snap, theta, x[:end])
                loss = np.mean(np.logaddexp(0, train) - (y[:end] == 1) * train) + .001 * np.mean(theta[64:]**2)
                max_loss_error = max(max_loss_error, abs(loss - opt['loss']))
                stop = min(end + batch, 849)
                if end == stop:
                    continue
                scores = replay(snap, theta, x[end:stop])
                for i, score in zip(range(end, stop), scores, strict=True):
                    key = seed, batch, stage + 1, int(ids[i])
                    label, expected = candidate[key]
                    if key in diagnostic_keys:
                        matched = [r for r in snap['raw_rules'] if np.array_equal(x[i, r['columns']], r['values'])]
                        diagnostic_matches[key] = [sum(r['label'] == label for r in matched) for label in (1,-1)]
                    assert label == y[i]
                    max_error = max(max_error, abs(score - expected))
                    count += 1
    assert count == 3294 and max_error < 1e-12 and max_loss_error < 1e-12

    results = {}
    for method, rows in preds.items():
        groups = {}
        for seed, batch in config:
            keys = [k for k in rows if k[:2] == (seed, batch)]
            y, score = np.array([rows[k] for k in keys]).T
            assert len(keys) == 549
            m = metrics(y, score)
            groups[f'{seed}/{batch}'] = dict(ba=(m['sensitivity'] + m['specificity']) / 2,
                                            auc=m['auc'], fn=m['fn'])
        results[method] = dict(config=groups, average={k: float(np.mean([m[k] for m in groups.values()]))
                                                       for k in ('ba', 'auc')})
    assert diagnostic_keys == diagnostic_matches.keys()
    lines = ['# 保留半数漏报复核与固定负类选择实验', '',
             '事件键为划分、批量、阶段和数据行。仍漏掉的 4 个均与相对原 HVDM 新增的 6 个重叠。'
             '与能判阳的模型比较。回退修正用完整重复覆盖作参照，其余用原 HVDM。'
             '有效规则指参照中精确匹配且后件为正的正类规则。', '',
             '|事件组|划分/批量/阶段/行|参照|原/重复/半数分数|原/重复/半数正类权重|参照→半数精确正/负|有效正类规则 换出/保留|保留有效规则贡献 参照→半数|负类组贡献 参照→半数|固定负类分数|',
             '|---|---|---|---|---|---|---|---|---|---:|']
    strata = Counter()
    names = {'persistent':'仍漏掉', 'lost_fix':'修正后再漏', 'new_fn':'相对原新增'}
    for r in diagnosis:
        key = tuple(r['key'])
        new = candidate[key][1]
        for group in r['groups']:
            strata[names[group], new >= 0] += 1
        pos,neg = r['changes']
        original,harmonic,half = r['models']
        ref = original if r['reference'] == 'HVDM' else harmonic
        scores = '/'.join(f"{m['score']:+.4f}" for m in r['models'])
        masses = '/'.join(f"{m['mass'][0]:.3f}" for m in r['models'])
        groups = '、'.join(names[g] for g in r['groups'])
        lines.append(f"|{groups}|{'/'.join(map(str,key))}|{r['reference']}|{scores}|{masses}|"
                     f"{pos['exact_old']}/{neg['exact_old']}→{pos['exact_new']}/{neg['exact_new']}|"
                     f"{pos['useful_removed']}/{pos['useful_retained']}|"
                     f"{pos['useful_retained_old']:+.3f}→{pos['useful_retained_new']:+.3f}|"
                     f"{ref['contributions'][1]:+.3f}→{half['contributions'][1]:+.3f}|{new:+.4f}|")
    lines += ['', '|事件|类别|换出分数项|换入分数项|共同规则权重变化|共同规则后件变化|',
              '|---|---|---:|---:|---:|---:|']
    for r in diagnosis:
        for label,g in zip(('正','负'),r['changes'],strict=True):
            terms = '|'.join(f'{g[k]:+.4f}' for k in ('removed','added','weight','head'))
            lines.append(f"|{'/'.join(map(str,r['key']))}|{label}|{terms}|")
    lines += ['', '共同规则权重和后件变化用两侧均值分配交互项。四项之和与该类净分数变化精确一致。'
              '换入和换出使用各自模型的拟合参数，分解只用于归因线索。', '',
              '|事件组（组间有重叠）|事件数|固定负类判阳|', '|---|---:|---:|']
    for g,name in names.items():
        lines.append(f"|{name}|{sum(g in r['groups'] for r in diagnosis)}|{strata[name, True]}|")
    lines += ['', '|方法|平均 BA|平均 AUC|总漏报|', '|---|---:|---:|---:|']
    for method, res in results.items():
        m = res['average']
        lines.append(f"|{method}|{m['ba']:.6%}|{m['auc']:.8f}|{sum(g['fn'] for g in res['config'].values())}|")
    lines += ['', '|对照|划分/批量|ΔBA（百分点）|ΔAUC|新增漏报|纠正漏报|',
              '|---|---|---:|---:|---:|---:|']
    pairs = {}
    for name, ref in [('HVDM', hvdm), ('累计逻辑', logic), ('保留半数', reserved)]:
        new = {k for k, (y, s) in candidate.items() if y == 1 and s < 0 <= ref[k][1]}
        fixed = {k for k, (y, s) in candidate.items() if y == 1 and ref[k][1] < 0 <= s}
        pairs[name] = dict(new_fn=sorted(new), corrected_fn=sorted(fixed))
        for seed, batch in config:
            group = f'{seed}/{batch}'
            a, b = (results[method]['config'][group] for method in ('HVDM + 固定负类', name))
            n = sum(k[:2] == (seed, batch) for k in new)
            f = sum(k[:2] == (seed, batch) for k in fixed)
            lines.append(f"|{name}|{group}|{100*(a['ba']-b['ba']):+.6f}|{a['auc']-b['auc']:+.8f}|{n}|{f}|")
        a, b = (results[method]['average'] for method in ('HVDM + 固定负类', name))
        lines.append(f"|{name}|总计/宏平均|{100*(a['ba']-b['ba']):+.6f}|{a['auc']-b['auc']:+.8f}|{len(new)}|{len(fixed)}|")
    failures = [dict(seed=r['seed'], batch=r['batch'], stage=r['stage'], message=r['snapshot']['optimization']['message'])
                for r in stored if not r['snapshot']['optimization']['success']]
    lines += ['', f'阈值固定为 0。{len(stored)} 次冷拟合中 {len(failures)} 次未报告收敛成功。'
              f'独立重放 {count} 条新分数，最大误差 {max_error:.3g}。重算 57 份前缀训练目标，最大误差 {max_loss_error:.3g}。'
              '各划分的 213 条留出样本未解析。本轮按已见漏报选择组件，属于内部探索。']
    (OUT / '去重漏报复核与实验报告.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    summary = dict(results=results, pairs=pairs, failures=failures, replay_count=count,
                   replay_error=max_error, objective_error=max_loss_error,
                   strata=[dict(stratum=k, corrected=bool(v), count=n) for (k, v), n in strata.items()],
                   prediction_hashes={name: hashlib.sha256(p.read_bytes()).hexdigest()
                                      for name, (p, _) in paths.items()})
    (run / 'verification.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(averages={k: v['average'] for k, v in results.items()},
                          pairs={k: {m: len(ids) for m, ids in v.items()} for k, v in pairs.items()},
                          failures=failures, strata=summary['strata'], max_error=max_error), ensure_ascii=False))


if __name__ == '__main__':
    main()





