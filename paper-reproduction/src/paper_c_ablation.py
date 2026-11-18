"""Fixed C-branch ablations with shared, training-prefix-only auxiliary fits.

The original stage learners depend only on the revealed prefix; previous stages
change eligibility, not their fitting objective. This module caches predictions
on the union of eligible query rows and re-executes every variant's gates.
Full-route archive replay is mandatory in the experiment driver.
"""
from time import perf_counter

import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from run_three_way_retrospective_holdout import panel, geometry
from run_rescue_prefix import fit_score
from run_rescue_interactions import interact

ORDER = ('P', 'E', 'U', 'D', 'I')
VARIANTS = {
    'base': (), 'plus_P': ('P',), 'plus_PE': ('P','E'),
    'plus_PEU': ('P','E','U'), 'plus_PEUD': ('P','E','U','D'),
    'full': ORDER,
    **{f'without_{stage}':tuple(s for s in ORDER if s != stage) for stage in ORDER},
}


def evaluate(x, y, query, base, previous_x, previous_y, previous_scores, scale):
    """No current-query labels are accepted by this fitting/prediction function."""
    started=perf_counter()
    past=np.column_stack(((previous_x-scale[0])/scale[1],previous_scores))
    current=np.column_stack(((query-scale[0])/scale[1],base))
    tick=perf_counter()
    panel_output,panel_status=panel(past,previous_y,current)
    panel_score=panel_output['rbf_margin']
    panel_seconds=perf_counter()-tick
    tick=perf_counter()
    geometry_output,geometry_status=geometry(past,previous_y,current)
    exact_score=geometry_output['exact_recurrence']
    exact_seconds=perf_counter()-tick
    negative=set(map(tuple,x[y == -1]))
    unseen_benign=np.array([tuple(row) not in negative for row in query])
    eligible=np.flatnonzero((base < 0) & unseen_benign)
    components={name:(base.copy(),np.zeros(len(query),dtype=bool)) for name in ('U','D','I')}
    details=dict(panel_seconds=panel_seconds,exact_seconds=exact_seconds,
                 panel_status=panel_status,exact_status=geometry_status,
                 eligible_union=int(len(eligible)),auxiliary={})
    if len(eligible):
        encoder=OneHotEncoder(handle_unknown='ignore').fit(x)
        design,test=encoder.transform(x),encoder.transform(query[eligible])
        tick=perf_counter()
        full=fit_score(design,y,test)
        stable=full>=0
        rng=np.random.default_rng(1105)
        groups=[np.flatnonzero(y==label) for label in (-1,1)]
        for _ in range(32):
            sample=np.concatenate([rng.choice(group,size=len(group),replace=True) for group in groups])
            stable &= fit_score(design[sample],y[sample],test)>=0
        components['U'][0][eligible]=full
        components['U'][1][eligible]=stable
        details['auxiliary']['U']=dict(seconds=perf_counter()-tick,model_fits=33)

        tick=perf_counter()
        full=fit_score(design,y,test)
        stable=full>=0
        borders=np.cumsum([0,*[len(v) for v in encoder.categories_]])
        for first,last in zip(borders[:-1],borders[1:],strict=True):
            keep=(np.arange(design.shape[1])<first)|(np.arange(design.shape[1])>=last)
            stable &= fit_score(design[:,keep],y,test[:,keep])>=0
        components['D'][0][eligible]=full
        components['D'][1][eligible]=stable
        details['auxiliary']['D']=dict(seconds=perf_counter()-tick,model_fits=19)

        tick=perf_counter()
        scaler=StandardScaler().fit(x)
        numeric,numeric_test=interact(scaler.transform(x),scaler.transform(query[eligible]))
        full=fit_score(numeric,y,numeric_test)
        stable=full>=0
        rng=np.random.default_rng(1105)
        fits=1
        for _ in range(32):
            if not stable.any():break
            sample=np.concatenate([rng.choice(group,size=len(group),replace=True) for group in groups])
            stable[stable]=fit_score(numeric[sample],y[sample],numeric_test[stable])>=0
            fits+=1
        components['I'][0][eligible]=full
        components['I'][1][eligible]=stable
        details['auxiliary']['I']=dict(seconds=perf_counter()-tick,model_fits=fits)
    outputs={}
    stages={}
    for variant,enabled in VARIANTS.items():
        reference=base.copy()
        trace={}
        for stage in ORDER:
            if stage not in enabled:continue
            before=reference.copy()
            if stage=='P':
                reference=panel_score.copy()
            elif stage=='E':
                take=(reference<0)&(exact_score>=0)
                reference[take]=exact_score[take]
            else:
                scores,stable=components[stage]
                take=(reference<0)&unseen_benign&stable
                reference[take]=scores[take]
            trace[stage]=int(np.count_nonzero(reference != before))
        assert np.isfinite(reference).all()
        assert np.array_equal(reference[base>=0],base[base>=0])
        outputs[variant]=reference
        stages[variant]=trace
    np.testing.assert_array_equal(outputs['without_I'],outputs['plus_PEUD'])
    details.update(seconds=perf_counter()-started,stage_changes=stages)
    return outputs,details
