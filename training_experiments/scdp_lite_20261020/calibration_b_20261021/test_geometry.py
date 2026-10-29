import unittest
import numpy as np
from scipy.optimize import least_squares
from fit_camera import predict,Correspondences

class GeometryTests(unittest.TestCase):
    def test_arkit_optical_axis_and_transform_inverse(self):
        p=np.r_[np.log([180.,240.]),[160.,120.],np.zeros(6)]
        xyz=np.array([[0,0,-1],[.1,.1,-1]])
        uv,cam,r=predict(p,xyz)
        np.testing.assert_allclose(uv,[[160,120],[178,96]])
        np.testing.assert_allclose(cam@r,xyz)

    def test_known_synthetic_parameters_are_recoverable(self):
        rng=np.random.default_rng(12)
        xyz=rng.uniform([-.25,-.4,-.9],[.4,.1,-.3],(90,3))
        known=np.r_[np.log([174.,224.]),[194.,164.],[.11,-.05,-.04],[-.04,-.05,-.07]]
        observed=predict(known,xyz)[0]
        initial=np.r_[np.log([180.,240.]),[160.,120.],np.zeros(6)]
        result=least_squares(lambda p:(predict(p,xyz)[0]-observed).ravel(),initial,max_nfev=150)
        np.testing.assert_allclose(predict(result.x,xyz)[0],observed,atol=1e-5)
        np.testing.assert_allclose(result.x,known,atol=1e-4)

    def test_fractional_time_offset_sign(self):
        c=Correspondences.__new__(Correspondences)
        c.t0=np.array([10.]);c.near_times=np.arange(19)[None]*.1+9.1
        c.near_xyz=np.zeros((1,19,3));c.near_xyz[0,:,0]=c.near_times[0]*2
        np.testing.assert_allclose(c.points(.075),[[20.15,0,0]])
        np.testing.assert_allclose(c.points(-.075),[[19.85,0,0]])

if __name__=='__main__':unittest.main()
