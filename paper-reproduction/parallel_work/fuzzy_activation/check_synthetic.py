"""Small direct checks: no patient IO, model experiments, or test discovery."""
import sys,json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp,expit
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.models import Rule
from src.fptree import FPTree
from activation import DistanceSugeno,header_table,restore

def main():
    x=np.array([[0,1],[0,2],[2,1],[2,2],[2,2],[5,2]],float);y=np.array([-1,1,1,1,-1,-1])
    rules=[Rule((0,1),np.array([0.,1.]),-1,.7,2),Rule((0,),np.array([2.]),1,.8,3)]
    table=header_table(FPTree().update(x,y));assert table[0,2]==(3,2)
    m=DistanceSugeno(rules,'hvdm_mean',table);d=m._distances(x)
    np.testing.assert_allclose(d[2,0],((2/3-.5)**2)/2)
    assert m._distances(np.array([[99.,99.]]))[0,0]==1
    rename={0:77,2:-4,5:100};xx=x.copy();xx[:,0]=[rename[int(v)] for v in x[:,0]]
    rr=[Rule(r.columns,np.array([rename[int(v)] if j==0 else v for j,v in zip(r.columns,r.values)]),r.label,r.confidence,r.count) for r in rules]
    other=DistanceSugeno(rr,'hvdm_mean',header_table(FPTree().update(xx,y)))
    np.testing.assert_array_equal(d,other._distances(xx))
    # Finite differences for exactly the frozen-distance BCE/width objective inherited by both arms.
    theta=np.array([-.7,-1.2,-.8,1.]);conf=np.array([.7,.8]);target=(y==1).astype(float)
    def fun(t):
        a=d*np.exp(-2*t[:2]);lf=-.5*a+np.log(conf);w=np.exp(lf-logsumexp(lf,axis=1)[:,None]);s=w@t[2:];e=(expit(s)-target)/len(y)
        loss=np.mean(np.logaddexp(0,s)-target*s)+.001*np.mean(t[2:]**2)
        grad=np.r_[np.sum(e[:,None]*w*(t[2:]-s[:,None])*a,axis=0),w.T@e+.002*t[2:]/2]
        return loss,grad
    numeric=[]
    for j in range(4):
        off=np.eye(4)[j]*1e-6;numeric.append((fun(theta+off)[0]-fun(theta-off)[0])/2e-6)
    error=float(np.max(np.abs(np.array(numeric)-fun(theta)[1])));assert error<1e-8
    scale=(np.zeros(2),np.ones(2)*5)
    m.fit(x,y,scaling=scale);snap=m.snapshot();np.testing.assert_array_equal(m.decision_function(x),restore(snap).decision_function(x))
    # Shared rule width => min of Gaussian degrees equals Gaussian of max squared distance.
    z=np.array([[.1,.8],[.6,.2]]);sig=.25
    np.testing.assert_allclose(np.min(np.exp(-.5*z*z/(sig*sig)),axis=1),np.exp(-.5*np.max(z*z,axis=1)/(sig*sig)))
    a=np.array([[.2],[.9]]);np.testing.assert_array_equal(a.mean(1),a.max(1))
    out=dict(passed=True,checks=['header counts','binary N2 simplification','unknown values','category relabel invariance','width and consequent finite difference','snapshot score replay','Gaussian min/max equivalence','L1 equivalence'],gradient_max_error=error,patient_rows_read=0)
    (Path(__file__).resolve().parent/'synthetic_verification.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out))
if __name__=='__main__':main()
