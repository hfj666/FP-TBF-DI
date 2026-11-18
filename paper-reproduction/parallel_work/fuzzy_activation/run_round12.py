"""Evaluate three score heads with frozen HVDM antecedents."""
import hashlib
import json
import numpy as np

from diagnose_round05 import HERE, ROOT, OLD, LOGIC, load_state, read_predictions
from run_component import digest, load_stream, write_csv, metrics
from activation import restore
from src.fp_linear_consequent import LinearConsequent, design_matrix, objective
from audit_round10 import replay

OUT = HERE / 'results_12'
MODES = ('shared_linear', 'shared_quadratic', 'local_linear')


class QuadraticConsequent(LinearConsequent):
    def _design(self, x):
        z = (x-self.base.minimum)/self.base.span
        return design_matrix(self.base.firing_weights(x),
                             np.column_stack((z, (z-.5)**2)), 'shared_linear')


def main():
    OUT.mkdir(exist_ok=False)
    paths = [HERE / 'PROTOCOL_12.md', HERE / 'run_round12.py', HERE / 'activation.py',
             ROOT / 'src/fp_linear_consequent.py', OLD / 'state_audit.json', LOGIC]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (OUT/'protocol.json').write_text(json.dumps(dict(hashes=hashes, threshold=0, modes=MODES),indent=2))
    audit = {(r['seed'],r['batch_size'],r['stage']):r['sha256']
             for r in json.loads((OLD/'state_audit.json').read_text())}
    preds, fits = [], []
    for split in json.loads((ROOT/'results/online_study/splits.json').read_text()):
        seed = split['seed']
        ids = np.array(split['train_stream'])
        assert len(ids)==len(set(ids))==849 and not set(ids)&set(split['holdout'])
        x,y = load_stream(ROOT/'data/dataset_1062.txt',ids)
        for batch in (50,100):
            for stage,end in enumerate([*range(300,849,batch),849]):
                state = load_state(seed,batch,stage,audit)
                assert state['prefix_input_sha256']==digest(x[:end],y[:end],ids[:end])
                snap = state['models']['hvdm_mean']
                if stage or batch==50:
                    base = restore(snap)
                    heads = {}
                    for mode in MODES:
                        head = (QuadraticConsequent if mode=='shared_quadratic' else LinearConsequent)(base,mode)
                        head.fit(x[:end],y[:end])
                        heads[mode] = head
                        fits.append(dict(seed=seed,batch=batch if stage else 0,stage=stage,
                                         mode=mode,beta=head.beta.tolist(),**head.optimization))
                    if not stage:
                        initial = snap,heads
                else:
                    original,heads = initial
                    assert snap==original
                if end==849:
                    continue
                stop = min(end+batch,849)
                for mode,head in heads.items():
                    scores = head.decision_function(x[end:stop])
                    assert np.isfinite(scores).all()
                    preds.extend(dict(seed=seed,batch_size=batch,stage=stage+1,row_index=int(i),policy=mode,
                                      label=int(v),score=float(s)) for i,v,s in zip(ids[end:stop],y[end:stop],scores,strict=True))
            print(f'{seed}/{batch} complete',flush=True)
    assert len(fits)==162 and len(preds)==9882
    (OUT/'fits.json').write_text(json.dumps(fits,separators=(',',':')),encoding='utf-8')
    write_csv(OUT/'predictions.csv',preds)


if __name__=='__main__':
    main()
