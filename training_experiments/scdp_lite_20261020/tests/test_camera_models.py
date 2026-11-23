import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import runtime
import json,unittest
import cv2,numpy as np,torch
from spatial_conditioner import project_wrist,resize_crop_affine
from camera_geometry import interpolate_poses,intrinsics_at_stored_resolution
from hat_adapter import SCDPHAT
from test_adapter import FakeHAT
from test_spatial import camera


class CameraModelTests(unittest.TestCase):
    def test_rational_distortion_matches_opencv(self):
        config=json.loads((runtime.ROOT/'configs/cameras_20261021.json').read_text())['robot_zed_left']
        points=np.array([[-.3,-.1,.8],[.2,.2,1.],[-.1,.3,.6],[0,0,1.]])
        expected,_=cv2.projectPoints(points,np.zeros(3),np.zeros(3),np.array(config['K']),np.array(config['D']))
        cam=camera();cam['K']=torch.tensor(config['K'])[None]
        cam['D']=torch.tensor(config['D'])[None]
        cam['raw_hw']=torch.tensor([[240,320]])
        cam['pixel_from_raw']=resize_crop_affine((240,320),(240,320))[None]
        grid,valid=project_wrist(torch.tensor(points,dtype=torch.float32).reshape(1,2,2,3),cam)
        pixels=(grid[0].numpy()+1)*[304,224]/2-.5
        uv=np.c_[expected.reshape(-1,2),np.ones(4)]@cam['pixel_from_raw'][0].numpy().T
        np.testing.assert_allclose(pixels.reshape(-1,2)[valid.flatten()],uv[:,:2][valid.flatten()],atol=3e-5)

    def test_reference_intrinsics_survive_actual_stored_resize(self):
        k=np.array([[132.8966,0,156.1407],[0,177.2074,123.5014],[0,0,1.]])
        stored=intrinsics_at_stored_resolution(k,(240,320),(480,640))
        to_ref=np.array([[.5,0,-.25],[0,.5,-.25],[0,0,1.]])
        np.testing.assert_allclose(to_ref@stored,k)

    def test_virtual_and_urdf_head_transform_are_consistent(self):
        cfg=json.loads((runtime.ROOT/'configs/cameras_20261021.json').read_text())['robot_zed_left']
        head_from_cam=np.array(cfg['T_head_link_from_left_optical'])
        virtual_from_head=np.eye(4);virtual_from_head[:3,3]=[.0039635,0,-.444]
        virtual_from_cam=virtual_from_head@head_from_cam
        np.testing.assert_allclose(virtual_from_cam[:3,3],cfg['t_virtual_head_from_left_optical'],atol=1e-8)
        points=np.array([[.1,-.2,-.5,1.]]).T
        np.testing.assert_allclose(np.linalg.inv(virtual_from_cam)@points,
            np.linalg.inv(head_from_cam)@np.linalg.inv(virtual_from_head)@points)

    def test_temporal_interpolation_and_outside_mask(self):
        poses=np.broadcast_to(np.eye(4),(3,4,4)).copy();poses[:,0,3]=[0,1,2]
        shifted,valid=interpolate_poses([0,1,2],poses,np.array([.097428,1.097428,2.097428]))
        np.testing.assert_allclose(shifted[:2,0,3],[.097428,1.097428])
        np.testing.assert_array_equal(valid,[True,True,False])

    def test_right_only_does_not_modify_left_wrist(self):
        torch.manual_seed(4)
        model=SCDPHAT(FakeHAT(),dict(enabled=True))
        model.conditioner.film[-1].bias.data.fill_(.2)
        cam=camera();cam['hand_valid']=torch.tensor([[False,True]])
        cond=dict(scdp_camera=cam,scdp_action_mean=torch.zeros(1,128),scdp_action_std=torch.ones(1,128))
        result=model(None,torch.randn(1,4,14,19),None,conditioning_dict=cond)[0]
        original=model.debug['coarse']
        self.assertTrue(torch.equal(result[...,80:83],original[...,80:83]))
        self.assertGreater((result[...,30:33]-original[...,30:33]).abs().sum().item(),0)
        self.assertFalse(model.debug['valid'][...,0].any())


if __name__=='__main__':
    torch.set_num_threads(2);unittest.main()
