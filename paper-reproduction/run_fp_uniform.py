"""One predeclared literature component, matched control, no tuning."""
import gzip,hashlib,json
from time import perf_counter
import numpy as np
from src.fptree import FPTree
from src.selection import CoveragePool
from src.models import SparseSugeno
from src.fp_uniform import UniformSugeno,objective
from src.fp_online import rule_records
from run_original_closeout import load_stream
from run_experiments import ROOT,metrics,write_csv

OUT=ROOT/'results/fp_uniform'
def main():
    OUT.mkdir(exist_ok=False)
    sources=['run_fp_uniform.py','src/fp_uniform.py','tests/test_fp_uniform.py','src/models.py','src/selection.py','src/fptree.py','run_original_closeout.py','run_experiments.py','results/online_study/splits.json','data/dataset_1062.txt']
    (OUT/'protocol.json').write_text(json.dumps(dict(source='https://arxiv.org/html/1908.00636v2 Eq7',coefficient=.1,rationale='Smallest published CV candidate, fixed here with no search; not author recommended optimum. Tau=.5 exact source C2; objective constant nonzero at uniform. Only UR component; no BN/AdaBound/first-order/FCM; not full author reproduction.',settings='Coverage alpha.5 FP support.1 L3 64/32perclass min5 purity>.65. Same frozen initial scaling,cold BCE+old consequent penalty,500 optimizer,bounds,initialization. initial300/849 suffix549,seeds17/29/43 batches50/100,each213 excluded this run. Reused internal development.',hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}),indent=2),encoding='utf-8')
    predictions=[];fits=[];summaries=[];audit=[]
    for sp in json.loads((ROOT/'results/online_study/splits.json').read_text()):
        seed=sp['seed'];ids=np.array(sp['train_stream']);assert len(set(ids))==849 and not set(ids)&set(sp['holdout']);x,y=load_stream(ROOT/'data/dataset_1062.txt',ids);scale=(x[:300].min(0),np.maximum(np.ptp(x[:300],axis=0),1.))
        for batch in (50,100):
            tree=FPTree();previous=0;scores={'baseline':[],'ur':[]}
            for stage,end in enumerate([300,*range(300+batch,849,batch),849]):
                tree.update(x[previous:end],y[previous:end]);rules=CoveragePool(tree.mine(.1,3),x[:end],y[:end]).select(.5,64);record=dict(seed=seed,batch_size=batch,stage=stage,prefix=end,models={})
                for arm in ('baseline','ur'):
                    t=perf_counter();m=SparseSugeno(rules,maxiter=500).fit(x[:end],y[:end],scaling=scale) if arm=='baseline' else UniformSugeno(rules).fit(x[:end],y[:end],scale,.1)
                    elapsed=perf_counter()-t;mean=m.firing_weights(x[:end]).mean(0)
                    fits.append(dict(seed=seed,batch_size=batch,stage=stage,policy=arm,seconds=elapsed,effective_rules=float(1/(mean@mean)),max_mass=float(mean.max()),**{k:v for k,v in m.optimization.items() if k not in ('effective_rules','max_mass')}))
                    record['models'][arm]=dict(rules=rule_records(m.rules),theta=m.theta.tolist(),minimum=m.minimum.tolist(),span=m.span.tolist())
                    if end<849:
                        s=m.decision_function(x[end:min(end+batch,849)]);scores[arm].extend(s);predictions.extend(dict(seed=seed,batch_size=batch,stage=stage+1,policy=arm,row_index=int(i),label=int(v),score=float(z)) for i,v,z in zip(ids[end:min(end+batch,849)],y[end:min(end+batch,849)],s))
                raw=json.dumps(record,sort_keys=True)
                with gzip.open(OUT/f'state_s{seed}_b{batch}_t{stage}.json.gz','wt',encoding='utf-8') as f:f.write(raw)
                audit.append(dict(seed=seed,batch_size=batch,stage=stage,sha256=hashlib.sha256(raw.encode()).hexdigest()));previous=end
            for arm,s in scores.items():summaries.append(dict(seed=seed,batch_size=batch,policy=arm,**metrics(y[300:],np.array(s))))
            for name,rr in [('predictions',predictions),('summary',summaries),('fits',fits)]:write_csv(OUT/f'{name}.csv',rr)
            (OUT/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8');print(f'UR seed={seed} batch={batch} complete',flush=True)
    (OUT/'completion.json').write_text(json.dumps(dict(completed=True,predictions=len(predictions),fits=len(fits))),encoding='utf-8')

if __name__=='__main__':main()
