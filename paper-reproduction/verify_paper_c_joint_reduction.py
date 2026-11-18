"""Verify joint-reduction artifacts, standalone entry, and manuscript tables."""
import argparse
import csv
import hashlib
import inspect
import json
import re
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from src.fp_tbf_simplified import predict_c

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/paper_c_joint_reduction'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-manuscript', action='store_true',
                        help='Also verify table and equation numbering in the local manuscript.')
    args = parser.parse_args()
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha for name,sha in manifest['sources'].items())
    assert list(inspect.signature(predict_c).parameters)==['x','y','query','base']
    summary=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    with (OUT/'predictions.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    counts=Counter(r['evaluation'] for r in rows)
    assert counts=={'stream':3294,'holdout_300':1278,'holdout_500':1278,'holdout_700':1278,'holdout_849':1278}
    assert len({(r['evaluation'],r['seed'],r['batch'],r['row_index']) for r in rows})==8406
    groups=defaultdict(list)
    for r in rows:
        assert (float(r['full_fusion'])>=0)==(float(r['simple_fusion'])>=0)
        assert abs((float(r['simple_c'])-float(r['full_c']))/3-(float(r['simple_fusion'])-float(r['full_fusion'])))<1e-12
        groups[r['evaluation'],int(r['seed']),int(r['batch'])].append(r)
    assert len(groups)==30
    for (evaluation,seed,batch),values in groups.items():
        target=next(r for r in summary[evaluation]['simple_fusion']['configurations'] if (r['seed'],r['batch'])==(seed,batch))
        y=np.array([int(r['label'])==1 for r in values]);score=np.array([float(r['simple_fusion']) for r in values])
        assert abs(roc_auc_score(y,score)-target['auc'])<1e-12
        assert int(((score>=0)==y).sum())==target['correct']
    with (ROOT/'results/fp_tbf_di_entry_check/predictions.csv').open(encoding='utf-8-sig',newline='') as f:entry=list(csv.DictReader(f))
    expected={(r['evaluation'],r['row_index']):r for r in rows if r['seed']=='17' and r['batch']=='100' and r['evaluation'] in ('stream','holdout_849')}
    assert len(entry)==len(expected)==762
    entry_error=0.
    for r in entry:
        original=expected[r['evaluation'],r['row_index']]
        for key,field in [('C','simple_c'),('fusion','simple_fusion'),('base','base'),('A','A'),('B','B')]:
            entry_error=max(entry_error,abs(float(r[key])-float(original[field])))
    assert entry_error<=1e-7
    manuscript = {}
    if args.check_manuscript:
        text=(ROOT/'paper/FP_TBF_DI_中文论文精简修订稿.md').read_text(encoding='utf-8')
        assert list(map(int,re.findall(r'^\*\*表(\d+) ',text,re.M)))==list(range(1,15))
        assert list(map(int,re.findall(r'\\tag\{(\d+)\}',text)))==list(range(1,16))
        assert text.count('\\[')==text.count('\\]') and text.count('\\(')==text.count('\\)')
        assert '\ufffd' not in text
        manuscript = dict(manuscript_tables=14, manuscript_equations=15)
    checks=json.loads((OUT/'checks.json').read_text(encoding='utf-8'))
    assert len(checks)==75
    archive=[r['archive_error'] for r in checks if r['archive_error'] is not None]
    native=[r['native_di_error'] for r in checks if r['native_di_error'] is not None]
    assert len(archive)==57 and len(native)==12 and max(archive)<=1e-7 and max(native)<=1e-7
    proof=dict(source_hashes_unchanged=True,counts=dict(counts),configuration_stage_groups=len(groups),
               fusion_label_disagreements=0,independent_auc_checks=30,
               native_DI_checks=len(native),native_DI_max_error=max(native),
               original_archive_checks=len(archive),original_archive_max_error=max(archive),
               standalone_entry_predictions=len(entry),standalone_entry_max_error=entry_error,
               production_C_history_prediction_inputs=False, **manuscript)
    (OUT/'verification.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(proof,ensure_ascii=False))


if __name__=='__main__':main()
