"""Independent fixed comparison; only approved cohort rows are parsed."""
import sys,csv,copy,gzip,hashlib,json,argparse
from pathlib import Path
from time import perf_counter
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.models import Rule,SparseSugeno
from src.fptree import FPTree
from src.selection import CoveragePool
from src.fp_online import rule_records
from run_original_closeout import load_stream
from run_experiments import metrics,write_csv
from activation import DistanceSugeno,header_table
HERE=Path(__file__).resolve().parent
PARENT=ROOT/'results/fp_selection_bridge'

def digest(x,y,ids):
    return hashlib.sha256(np.ascontiguousarray(np.column_stack([ids,y,x]),dtype='<f8').tobytes()).hexdigest()

def rules_from(records):return [Rule(tuple(r['columns']),np.array(r['values']),r['label'],r['confidence'],r['count']) for r in records]
def baseline_restore(s):
    m=SparseSugeno(rules_from(s['rules']),maxiter=500)
    m.minimum=np.array(s['minimum']);m.span=np.array(s['span']);m.theta=np.array(s['theta']);return m

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--component',choices=['categorical','minimum'],default='categorical');args=ap.parse_args()
    modes=('hvdm_mean','overlap_mean') if args.component=='categorical' else ('numeric_max',)
    proto=HERE/('PROTOCOL_01.md' if args.component=='categorical' else 'PROTOCOL_02.md')
    assert proto.exists();out=HERE/('results_01' if args.component=='categorical' else 'results_02');out.mkdir(exist_ok=False)
    files=[HERE/'activation.py',HERE/'run_component.py',HERE/'check_synthetic.py',proto,ROOT/'src/models.py',ROOT/'src/fptree.py',ROOT/'src/selection.py',ROOT/'run_original_closeout.py',ROOT/'results/online_study/splits.json',PARENT/'audit.json',PARENT/'predictions.csv',PARENT/'protocol.json']
    provenance={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (out/'protocol.json').write_text(json.dumps(dict(component=args.component,modes=modes,source_hashes=provenance,clinical_threshold=0,scale='initial300 frozen',parsed_rows_per_seed=849,holdout_parsed_or_evaluated=0),indent=2),encoding='utf-8')
    source=out/'source_snapshot';source.mkdir()
    for p in (HERE/'activation.py',HERE/'run_component.py',HERE/'check_synthetic.py',proto): (source/p.name).write_bytes(p.read_bytes())
    parent_audit={(a['seed'],a['batch_size'],a['stage']):a['sha256'] for a in json.loads((PARENT/'audit.json').read_text())}
    oldpred=list(csv.DictReader((PARENT/'predictions.csv').open(encoding='utf-8-sig',newline='')))
    preds=[];summaries=[];fits=[];state_audit=[];io_audit=[]
    splits=json.loads((ROOT/'results/online_study/splits.json').read_text())
    for sp in splits:
        seed=sp['seed'];ids=np.array(sp['train_stream']);assert len(ids)==len(set(ids))==849 and not set(ids)&set(sp['holdout'])
        x,y=load_stream(ROOT/'data/dataset_1062.txt',ids);scale=(x[:300].min(0),np.maximum(np.ptp(x[:300],axis=0),1.))
        t=perf_counter();tree0=FPTree().update(x[:300],y[:300]);patterns=tree0.mine(.1,3);initial_rules=CoveragePool(patterns,x[:300],y[:300]).select(.5,64);initial_structure_seconds=perf_counter()-t
        t=perf_counter();base0=SparseSugeno(copy.deepcopy(initial_rules),maxiter=500).fit(x[:300],y[:300],scaling=scale);baseline_initial_seconds=perf_counter()-t
        initial={};table0=header_table(tree0)
        for mode in modes:
            t=perf_counter();m=DistanceSugeno(copy.deepcopy(initial_rules),mode,table0).fit(x[:300],y[:300],scaling=scale);elapsed=perf_counter()-t;initial[mode]=m
            fits.append(dict(seed=seed,batch_size=0,stage=0,prefix=300,policy=mode,fit_seconds=elapsed,**m.optimization))
        io_audit.append(dict(seed=seed,allowed_stream_sha256=digest(x,y,ids),initial_structure_seconds=initial_structure_seconds,baseline_initial_seconds=baseline_initial_seconds,holdout_rows_parsed=0))
        for batch in (50,100):
            tree=copy.deepcopy(tree0);models={'baseline':copy.deepcopy(base0),**copy.deepcopy(initial)};scores={a:[] for a in models}
            for stage,end in enumerate([300,*range(300+batch,849,batch),849]):
                structure_seconds=0.;parent_hash=None
                if stage:
                    prev=max(300,end-batch) if end!=849 else list([300,*range(300+batch,849,batch),849])[stage-1]
                    t=perf_counter();tree.update(x[prev:end],y[prev:end]);table=header_table(tree);structure_seconds=perf_counter()-t
                    # Exact cumulative header counts, including categories below mining support.
                    for (j,v),(n,p) in table.items():
                        mask=x[:end,j]==v;assert n==int(mask.sum()) and p==int(np.sum(mask&(y[:end]==1)))
                    path=PARENT/f'state_s{seed}_b{batch}_t{stage}.json.gz'
                    raw=gzip.open(path,'rt',encoding='utf-8').read();record=json.loads(raw)
                    parent_hash=hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest();assert parent_hash==parent_audit[seed,batch,stage] and record['prefix']==end
                    raw_rules=rules_from(record['selected']['coverage']);models={'baseline':baseline_restore(record['models']['coverage_cumulative'])}
                    np.testing.assert_array_equal(models['baseline'].minimum,scale[0]);np.testing.assert_array_equal(models['baseline'].span,scale[1])
                    for mode in modes:
                        t=perf_counter();m=DistanceSugeno(copy.deepcopy(raw_rules),mode,table).fit(x[:end],y[:end],scaling=scale);elapsed=perf_counter()-t;models[mode]=m
                        fits.append(dict(seed=seed,batch_size=batch,stage=stage,prefix=end,policy=mode,fit_seconds=elapsed,**m.optimization))
                else: raw_rules=initial_rules
                for mode in modes:
                    assert rule_records(models[mode].raw_rules)==rule_records(raw_rules)
                    assert rule_records(models[mode].rules)==rule_records(models['baseline'].rules)
                snap=dict(seed=seed,batch_size=batch,stage=stage,prefix=end,prefix_input_sha256=digest(x[:end],y[:end],ids[:end]),parent_state_sha256=parent_hash,structure_seconds=structure_seconds,
                          baseline=dict(rules=rule_records(models['baseline'].rules),minimum=models['baseline'].minimum.tolist(),span=models['baseline'].span.tolist(),theta=models['baseline'].theta.tolist()),
                          models={mode:models[mode].snapshot() for mode in modes})
                raw=json.dumps(snap,sort_keys=True)
                with gzip.open(out/f'state_s{seed}_b{batch}_t{stage}.json.gz','wt',encoding='utf-8') as f:f.write(raw)
                state_audit.append(dict(seed=seed,batch_size=batch,stage=stage,sha256=hashlib.sha256(raw.encode()).hexdigest()))
                if end<849:
                    nxt=min(end+batch,849)
                    # All arms predict complete batch before yy is used for the following update.
                    staged={arm:m.decision_function(x[end:nxt]) for arm,m in models.items()}
                    old=[r for r in oldpred if r['policy']=='coverage_cumulative' and int(r['seed'])==seed and int(r['batch_size'])==batch and int(r['stage'])==stage+1]
                    assert [int(r['row_index']) for r in old]==ids[end:nxt].tolist()
                    np.testing.assert_array_equal([int(r['label']) for r in old],y[end:nxt]);np.testing.assert_array_equal([float(r['score']) for r in old],staged['baseline'])
                    for arm,s in staged.items():
                        scores[arm].extend(s);preds.extend(dict(seed=seed,batch_size=batch,stage=stage+1,policy=arm,row_index=int(i),label=int(v),score=float(z)) for i,v,z in zip(ids[end:nxt],y[end:nxt],s))
            for arm,ss in scores.items():
                assert len(ss)==549;mm=metrics(y[300:],np.array(ss));mm['balanced_accuracy']=(mm['sensitivity']+mm['specificity'])/2;summaries.append(dict(seed=seed,batch_size=batch,policy=arm,**mm))
            for name,rows in [('predictions',preds),('summary',summaries),('fits',fits)]:write_csv(out/f'{name}.csv',rows)
            (out/'state_audit.json').write_text(json.dumps(state_audit,indent=2),encoding='utf-8');(out/'input_audit.json').write_text(json.dumps(io_audit,indent=2),encoding='utf-8')
            print(f'{args.component} seed={seed} batch={batch} completed',flush=True)
    (out/'completion.json').write_text(json.dumps(dict(completed=True,states=len(state_audit),fits=len(fits),predictions=len(preds),holdout_parsed_or_evaluated=0)),encoding='utf-8')
if __name__=='__main__':main()
