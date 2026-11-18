"""Direct synthetic gradient and equivalence checks, no clinical data."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from robust_learning import objective,mask_bank


def main():
    rng=np.random.default_rng(717); n,k=11,5
    d=rng.random((n,k)); y=rng.integers(0,2,n); conf=rng.uniform(.65,1,k)
    theta=np.r_[np.log(rng.uniform(.2,.8,k)),rng.normal(size=k)]
    for mode in ('baseline','droprule','gce','focal'):
        masks=mask_bank(n,k,27)
        value,grad=objective(theta,d,y,conf,mode,masks)
        numeric=[]
        for j in range(2*k):
            step=np.eye(2*k)[j]*1e-6
            numeric.append((objective(theta+step,d,y,conf,mode,masks)[0]-objective(theta-step,d,y,conf,mode,masks)[0])/2e-6)
        np.testing.assert_allclose(grad,numeric,rtol=2e-5,atol=1e-8)
        print(mode,'gradient passed',float(np.max(np.abs(grad-numeric))))
    a=objective(theta,d,y,conf,'baseline')
    b=objective(theta,d,y,conf,'droprule',np.ones((16,n,k),dtype=bool))
    np.testing.assert_allclose(a[0],b[0],rtol=1e-14)
    np.testing.assert_allclose(a[1],b[1],rtol=1e-13,atol=1e-14)
    assert mask_bank(n,1,8,keep=0).all()
    assert mask_bank(n,k,9).any(axis=2).all()
    assert np.array_equal(mask_bank(n,k,9),mask_bank(n,k,9))
    print('All-keep BCE equivalence, all-zero restore, determinism passed')

if __name__=='__main__':main()
