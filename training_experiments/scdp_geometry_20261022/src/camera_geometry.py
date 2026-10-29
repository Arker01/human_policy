"""Explicit transforms and pixel-centre conventions shared by offline adapters."""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def rotation_6d_rows(values):
    values=np.asarray(values,dtype=np.float64)
    first=values[...,:3]; second=values[...,3:]
    first=first/np.maximum(np.linalg.norm(first,axis=-1,keepdims=True),1e-12)
    second=second-(first*second).sum(-1,keepdims=True)*first
    second=second/np.maximum(np.linalg.norm(second,axis=-1,keepdims=True),1e-12)
    return np.stack((first,second,np.cross(first,second)),axis=-2)


def poses_from_state(state):
    state=np.asarray(state)
    poses=np.broadcast_to(np.eye(4),(len(state),4,4)).copy()
    poses[:,:3,:3]=rotation_6d_rows(state[:,3:9]);poses[:,:3,3]=state[:,:3]
    return poses


def interpolate_poses(times,poses,query):
    times=np.asarray(times);query=np.asarray(query)
    if np.any(np.diff(times)<=0):
        raise ValueError('Head timestamps must be strictly increasing')
    valid=(query>=times[0])&(query<=times[-1])
    clipped=np.clip(query,times[0],times[-1])
    result=np.broadcast_to(np.eye(4),(len(query),4,4)).copy()
    result[:,:3,:3]=Slerp(times,Rotation.from_matrix(poses[:,:3,:3]))(clipped).as_matrix()
    result[:,:3,3]=np.stack([np.interp(clipped,times,poses[:,k,3]) for k in range(3)],axis=-1)
    return result,valid


def intrinsics_at_stored_resolution(K,reference_hw,stored_hw):
    """Undo cv2/torch pixel-centre resize so reprojection at reference size is K."""
    rh,rw=reference_hw; sh,sw=stored_hw
    sx,sy=rw/sw,rh/sh
    to_reference=np.array([[sx,0,(sx-1)/2],[0,sy,(sy-1)/2],[0,0,1.]])
    return np.linalg.solve(to_reference,np.asarray(K,dtype=np.float64))
