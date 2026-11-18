"""Exact recurrent cases and nearest-neighbor anomaly evidence."""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import expit

METHODS=('exact_recurrence','supported_anomaly')


def predict(history, labels, current):
    outputs={name:current[:,-1].copy() for name in METHODS}
    screened=current[:,-1]<0
    past_negative=history[:,-1]<0
    misses=history[past_negative & (labels==1)]
    controls=history[past_negative & (labels==-1)]
    status=dict(screened=int(screened.sum()),historical_negatives=len(misses)+len(controls),
                historical_misses=len(misses),fitted=[])
    if not screened.any() or not len(misses) or not len(controls):
        return outputs,status
    observed_misses=set(map(tuple,misses[:,:-1]))
    observed_controls=set(map(tuple,controls[:,:-1]))
    query=current[screened].copy()
    exact=np.array([tuple(row) in observed_misses and tuple(row) not in observed_controls
                    for row in query[:,:-1]])
    outputs['exact_recurrence'][screened]=np.where(exact,1.,query[:,-1])
    status['fitted'].append('exact_recurrence')
    if len(controls)<19:
        return outputs,status
    original=query[:,-1].copy()
    for matrix in (controls,misses,query):
        matrix[:,-1]=expit(matrix[:,-1])
    d=cdist(controls,controls,'sqeuclidean')
    np.fill_diagonal(d,np.inf)
    original_loo=d.min(axis=1)
    to_control=cdist(query,controls,'sqeuclidean')
    nearest_control=to_control.min(axis=1)
    nearest_miss=cdist(query,misses,'sqeuclidean').min(axis=1)
    updated=np.minimum(to_control,original_loo[None,:])
    numerator=1+(updated>=nearest_control[:,None]).sum(axis=1)
    take=(nearest_miss<nearest_control)&(20*numerator<=len(controls)+1)
    outputs['supported_anomaly'][screened]=np.where(take,nearest_control-nearest_miss,original)
    status['fitted'].append('supported_anomaly')
    return outputs,status
