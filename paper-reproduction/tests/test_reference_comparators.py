import unittest
import numpy as np
from src.models import StaticBiclusterBoost
from src.reference_comparators import CumulativeLogistic


class ComparatorTests(unittest.TestCase):
    def test_static_pair_distance_and_training(self):
        x=np.array([[0.,0.]]*10+[[2.,2.]]*10);y=np.array([-1]*10+[1]*10)
        m=StaticBiclusterBoost().fit(x,y)
        np.testing.assert_array_equal(np.where(m.decision_function(x)>=0,1,-1),y)
        rules=m.raw_rules();self.assertTrue(all(set(r.values)<=set([0.,2.]) for r in rules))
        np.testing.assert_array_equal(m.training_max,[2.,2.])

    def test_cumulative_logistic_preserves_initial_scaling(self):
        x=np.array([[0.,0.]]*10+[[2.,2.]]*10);y=np.array([-1]*10+[1]*10)
        m=CumulativeLogistic(x,y,(np.zeros(2),np.ones(2)))
        m.fit(np.vstack([x,[[9.,9.]]]),np.r_[y,1])
        np.testing.assert_array_equal(m.span,np.ones(2));self.assertTrue(np.isfinite(m.decision_function(x)).all())


if __name__=='__main__':unittest.main()
