"""Compare four frozen detector designs on the approved prequential streams."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from rescue_geometry_learning import METHODS, predict
from run_prequential_evidence import metrics, paired_counts


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from run_original_closeout import load_stream

SHADOW = HERE / 'experiment_hard_width_bank1701'
def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    split_path = ROOT/'results/online_study/splits.json'
    source = SHADOW/'predictions.csv'
    manifest = SHADOW/'manifest.json'
    prior = {row['seed']: row for row in json.loads(manifest.read_text())['inputs']}
    with source.open(encoding='utf-8-sig',newline='') as stream:
        rows = [row for row in csv.DictReader(stream) if row['policy']=='baseline']
    splits = {row['seed']: row for row in json.loads(split_path.read_text())}
    configurations, predictions, diagnostics = [], [], []
    for seed in (17,29,43):
        ids = splits[seed]['train_stream']
        assert len(ids)==len(set(ids))==849 and not set(ids)&set(splits[seed]['holdout'])
        raw, labels = load_stream(ROOT/'data/dataset_1062.txt',ids)
        assert digest(raw.tobytes()) == prior[seed]['allowed_x_sha256']
        assert digest(labels.tobytes()) == prior[seed]['allowed_y_sha256']
        scaled = (raw[300:]-raw[:300].min(axis=0)) / np.maximum(np.ptp(raw[:300],axis=0),1.)
        y = labels[300:]
        for batch in (50,100):
            base = [r for r in rows if (int(r['seed']),int(r['batch_size']))==(seed,batch)]
            assert [int(r['row_index']) for r in base] == ids[300:]
            np.testing.assert_array_equal([int(r['label']) for r in base],y)
            stages = np.array([int(r['stage']) for r in base])
            np.testing.assert_array_equal(stages,np.arange(549)//batch+1)
            b = np.array([float(r['score']) for r in base])
            features = np.column_stack((scaled,b))
            final = {name:b.copy() for name in METHODS}
            for start in range(0,len(y),batch):
                stop = min(start+batch,len(y))
                scores, status = predict(features[:start],y[:start],features[start:stop])
                for name in METHODS:
                    final[name][start:stop] = scores[name]
                diagnostics.append(dict(seed=seed,batch_size=batch,stage=start//batch+1,history=start,**status))
            for stage in np.unique(stages):
                past, now = stages < stage, stages == stage
                altered = np.where(past,y,-y)
                replay, _ = predict(features[past],altered[past],features[now])
                for name in METHODS:
                    np.testing.assert_array_equal(replay[name],final[name][now])
            record = dict(seed=seed,batch_size=batch,baseline=metrics(y,b),screened=int(np.sum(b<0)),methods={})
            for name, s in final.items():
                np.testing.assert_array_equal(b[b>=0],s[b>=0])
                fixed_fn,new_fn = paired_counts(y,b,s,1)
                fixed_fp,new_fp = paired_counts(y,b,s,-1)
                assert new_fn==fixed_fp==0
                record['methods'][name] = dict(metrics=metrics(y,s),new_fn=new_fn,resolved_fn=fixed_fn,
                                               new_fp=new_fp,resolved_fp=fixed_fp,flips=fixed_fn+new_fp)
                predictions.extend(dict(row,score=str(value),policy=name) for row,value in zip(base,s,strict=True))
            configurations.append(record)
    out = HERE/'experiment_rescue_geometry'
    out.mkdir(exist_ok=False)
    hashes = {str(p.relative_to(ROOT)):digest(p.read_bytes()) for p in
              (source,split_path,manifest,Path(__file__),HERE/'protocol_rescue_geometry.md',
               HERE/'run_prequential_evidence.py',HERE/'rescue_geometry_learning.py',ROOT/'run_original_closeout.py')}
    result = dict(configurations=configurations,diagnostics=diagnostics,source_hashes=hashes,
                  replayed_scores=len(predictions),holdout_parsed_or_evaluated=0)
    (out/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    with (out/'predictions.csv').open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    for name in METHODS:
        print(name,[sum(x['methods'][name][key] for x in configurations) for key in ('resolved_fn','new_fp')])
    print('Replayed',len(predictions),'scores')


if __name__=='__main__':
    main()
