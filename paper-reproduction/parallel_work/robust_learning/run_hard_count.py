"""Run one predeclared component in an isolated output directory."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import sys, json, gzip, hashlib, csv, argparse, shutil
from pathlib import Path
from time import perf_counter
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT))
import numpy as np
from src.models import Rule,SparseSugeno
from src.fp_online import rule_records
from run_original_closeout import load_stream
from run_experiments import metrics,write_csv
from hard_count_learning import fit_component


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def restore(s):
    m=SparseSugeno([Rule(tuple(r['columns']),np.array(r['values']),r['label'],r['confidence'],r['count']) for r in s['rules']])
    m.minimum=np.array(s['minimum']);m.span=np.array(s['span']);m.theta=np.array(s['theta']);return m


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['hard_consequent','hard_width']);ap.add_argument('--bank',type=int,default=1701);args=ap.parse_args()
    protocol=HERE/f'protocol_{args.mode}.md';assert protocol.exists()
    out=HERE/f'experiment_{args.mode}_bank{args.bank}';out.mkdir(exist_ok=False)
    parent=ROOT/'results/fp_uniform'
    parent_audit={(a['seed'],a['batch_size'],a['stage']):a['sha256'] for a in json.loads((parent/'audit.json').read_text())}
    bridge=ROOT/'results/fp_selection_bridge'
    parent_protocol=json.loads((bridge/'protocol.json').read_text())
    for p,h in parent_protocol['hashes'].items():
        if not p.startswith('data/'):assert sha(ROOT/p)==h,(p,'shared source changed')
    with (bridge/'predictions.csv').open(encoding='utf-8-sig',newline='') as f:
        saved={(int(r['seed']),int(r['batch_size']),int(r['row_index'])):r for r in csv.DictReader(f) if r['policy']=='coverage_cumulative'}
    files=[protocol,HERE/'protocol_hard_consequent.md',HERE/'run_hard_count.py',HERE/'hard_count_learning.py',HERE/'robust_learning.py',HERE/'test_objectives.py',ROOT/'src/models.py',ROOT/'run_original_closeout.py',ROOT/'results/online_study/splits.json',parent/'audit.json',bridge/'predictions.csv']
    manifest=dict(mode=args.mode,bank=args.bank,protocol=str(protocol.relative_to(ROOT)),source_hashes={str(p.relative_to(ROOT)):sha(p) for p in files},inputs=[],baseline='audited fp_uniform baseline, checked against fp_selection_bridge predictions',holdout_parsed=0)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    snap=out/'source_snapshot';snap.mkdir()
    for p in files:
        if p.suffix in ('.py','.md'):shutil.copyfile(p,snap/p.name)
    predictions=[];fits=[];summaries=[];audit=[]
    for sp in json.loads((ROOT/'results/online_study/splits.json').read_text()):
        seed=sp['seed'];ids=np.array(sp['train_stream']);holdout=set(sp['holdout'])
        assert len(ids)==len(set(ids))==849 and not set(ids)&holdout
        x,y=load_stream(ROOT/'data/dataset_1062.txt',ids)
        manifest['inputs'].append(dict(seed=seed,allowed_ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),allowed_x_sha256=hashlib.sha256(x.tobytes()).hexdigest(),allowed_y_sha256=hashlib.sha256(y.tobytes()).hexdigest(),rows=849))
        scale=(x[:300].min(0),np.maximum(np.ptp(x[:300],axis=0),1.))
        for batch in (50,100):
            scores={a:[] for a in ('baseline',args.mode)}
            for stage,end in enumerate([300,*range(300+batch,849,batch),849]):
                raw=gzip.open(parent/f'state_s{seed}_b{batch}_t{stage}.json.gz','rt',encoding='utf-8').read()
                assert hashlib.sha256(raw.encode()).hexdigest()==parent_audit[seed,batch,stage]
                parsed=json.loads(raw);assert parsed['prefix']==end
                base=restore(parsed['models']['baseline'])
                np.testing.assert_array_equal(base.minimum,scale[0]);np.testing.assert_array_equal(base.span,scale[1]);assert len(base.rules)==64
                bank_seed=args.bank+seed*100000+batch*1000+stage
                tic=perf_counter();m=fit_component(base,x[:end],y[:end],args.mode,bank_seed);seconds=perf_counter()-tic
                fits.append(dict(seed=seed,batch_size=batch,stage=stage,prefix=end,policy=args.mode,seconds=seconds,**m.optimization))
                rec=dict(seed=seed,batch_size=batch,stage=stage,prefix=end,parent_sha256=parent_audit[seed,batch,stage],models={a:dict(rules=rule_records(v.rules),theta=v.theta.tolist(),minimum=v.minimum.tolist(),span=v.span.tolist()) for a,v in [('baseline',base),(args.mode,m)]})
                encoded=json.dumps(rec,sort_keys=True)
                with gzip.open(out/f'state_s{seed}_b{batch}_t{stage}.json.gz','wt',encoding='utf-8') as f:f.write(encoded)
                audit.append(dict(seed=seed,batch_size=batch,stage=stage,sha256=hashlib.sha256(encoded.encode()).hexdigest()))
                if end<849:
                    stop=min(end+batch,849)
                    # Both entire batches are predicted before suffix labels enter records or future fits.
                    pp={a:v.decision_function(x[end:stop]) for a,v in [('baseline',base),(args.mode,m)]}
                    for a,s in pp.items():
                        scores[a].extend(s)
                        if a=='baseline':
                            for i,z in zip(ids[end:stop],s):np.testing.assert_allclose(z,float(saved[seed,batch,int(i)]['score']),rtol=0,atol=1e-11)
                        predictions.extend(dict(seed=seed,batch_size=batch,stage=stage+1,policy=a,row_index=int(i),label=int(v),score=float(z)) for i,v,z in zip(ids[end:stop],y[end:stop],s))
            for a,s in scores.items():
                assert len(s)==549
                mm=metrics(y[300:],np.array(s));mm['balanced_accuracy']=(mm['sensitivity']+mm['specificity'])/2
                summaries.append(dict(seed=seed,batch_size=batch,policy=a,**mm))
            for name,rr in [('predictions',predictions),('summary',summaries),('fits',fits)]:write_csv(out/f'{name}.csv',rr)
            (out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
            (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
            print(args.mode,seed,batch,'complete',flush=True)
    (out/'completion.json').write_text(json.dumps(dict(completed=True,states=len(audit),fits=len(fits),predictions=len(predictions),holdout_parsed_or_evaluated=0)),encoding='utf-8')

if __name__=='__main__':main()
