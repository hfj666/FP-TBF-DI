"""Apply predeclared add-only policies to saved out-of-sample proposals."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from run_prequential_evidence import metrics, paired_counts

HERE=Path(__file__).resolve().parent
PROPOSALS=HERE/'experiment_rescue_geometry'/'predictions.csv'
RBF=HERE/'experiment_rescue_panel'/'predictions.csv'
METHODS=('exact_recurrence','supported_anomaly')


def load(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    raw,reference=load(PROPOSALS),load(RBF)
    outputs,records=[],[]
    for seed in (17,29,43):
        for batch in (50,100):
            def select(rows, policy):
                return [r for r in rows if (int(r['seed']),int(r['batch_size']),r['policy'])==(seed,batch,policy)]
            base=select(reference,'rbf_margin')
            y=np.array([int(r['label']) for r in base])
            ref=np.array([float(r['score']) for r in base])
            identity=lambda r:(r['stage'],r['row_index'],r['label'])
            proposals=[]
            for method in METHODS:
                current=select(raw,method)
                assert list(map(identity,current))==list(map(identity,base))
                proposals.append(np.array([float(r['score']) for r in current]))
            for name,scores in zip(METHODS,proposals,strict=True):
                chosen=np.where((ref<0)&(scores>=0),scores,ref)
                rf,nf=paired_counts(y,ref,chosen,1)
                rp,np_=paired_counts(y,ref,chosen,-1)
                assert nf==rp==0
                records.append(dict(method=name,seed=seed,batch_size=batch,baseline=metrics(y,ref),
                                    candidate=metrics(y,chosen),resolved_fn=rf,new_fn=nf,new_fp=np_,resolved_fp=rp))
                outputs.extend(dict(r,policy=name,score=str(s)) for r,s in zip(base,chosen,strict=True))
    audit=HERE/'experiment_rescue_geometry'
    with (audit/'add_only_predictions.csv').open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(outputs[0]))
        writer.writeheader()
        writer.writerows(outputs)
    loaded=load(audit/'add_only_predictions.csv')
    assert loaded==outputs
    sources={p.name+'@'+p.parent.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (PROPOSALS,RBF,Path(__file__),HERE/'protocol_rescue_geometry.md')}
    (audit/'add_only_audit.json').open('x',encoding='utf-8').write(json.dumps(dict(records=records,
        verified_scores=len(loaded),source_hashes=sources),indent=2))
    for name in METHODS:
        group=[r for r in records if r['method']==name]
        print(name,sum(r['resolved_fn'] for r in group),sum(r['new_fp'] for r in group),
              sum(r['candidate']['balanced_accuracy']>r['baseline']['balanced_accuracy'] for r in group))


if __name__=='__main__':
    main()
