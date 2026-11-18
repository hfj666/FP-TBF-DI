import unittest
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from src.fp_uniform import objective,UniformSugeno
from src.models import Rule,SparseSugeno

class Uniform(unittest.TestCase):
    def test_gradient_and_target_constant(self):
        rng=np.random.default_rng(84);d=rng.random((15,4));y=rng.integers(0,2,15);c=np.full(4,.8);theta=np.r_[np.full(4,np.log(.25)),rng.normal(size=4)]
        numerical=approx_derivative(lambda t:objective(t,d,y,c,.1)[0],theta).ravel()
        np.testing.assert_allclose(objective(theta,d,y,c,.1)[1],numerical,atol=1e-6)
    def test_zero_weight_matches_existing(self):
        x=np.array([[0.],[1.]]*12);y=np.array([-1,1]*12);r=[Rule((0,),np.array([v]),2*v-1,1.,12) for v in (0,1)];scale=(np.zeros(1),np.ones(1))
        a=SparseSugeno(r,maxiter=500).fit(x,y,scaling=scale);b=UniformSugeno(r).fit(x,y,scale,0.)
        np.testing.assert_allclose(a.decision_function(x),b.decision_function(x),atol=1e-10)

if __name__=='__main__':unittest.main()
