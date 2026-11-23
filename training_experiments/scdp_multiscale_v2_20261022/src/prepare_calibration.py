"""Build read-only-data sidecars from supplied camera parameters; no inferred calibration."""
import runtime
import argparse,json
from pathlib import Path
import cv2,h5py,numpy as np
from camera_geometry import poses_from_state,interpolate_poses,intrinsics_at_stored_resolution


def stored_image(f,key):
    image=f[key][0]
    return cv2.imdecode(image,cv2.IMREAD_COLOR) if image.ndim==1 else image


def human_transforms(f,parameters):
    source=Path(str(f.attrs['source']))
    data=json.loads(source.read_text());frames=data['frames']
    state=np.asarray(f['observation.state']);n=len(state)
    if len(frames)!=n:
        raise ValueError(f'{source}: pose/HDF timeline mismatch')
    world_head=np.array([fr['head'] for fr in frames],np.float64).reshape(-1,4,4)
    kp_head=poses_from_state(state)
    # Recover ONLY the existing converter's constant coordinate change from head
    # frames, then verify it against the complete recorded trajectory.
    kp_from_world=kp_head[0]@np.linalg.inv(world_head[0])
    predicted=kp_from_world[None]@world_head
    head_residual=np.max(np.abs(predicted-kp_head))
    if head_residual>3e-5:
        raise ValueError(f'{source}: HAT/ARKit coordinate mapping is not constant: {head_residual}')
    wrist_residual=0.
    for side,ids in [('left',slice(80,83)),('right',slice(30,33))]:
        wrist=np.array([fr[f'{side}Wrist'] for fr in frames]).reshape(-1,4,4)
        valid=np.array([fr.get(f'{side}Wrist_valid',True) for fr in frames],bool)
        expected=wrist[:,:3,3]@kp_from_world[:3,:3].T+kp_from_world[:3,3]
        residual=np.max(np.abs(expected[valid]-state[valid,ids])) if valid.any() else 0.
        wrist_residual=max(wrist_residual,float(residual))
    if wrist_residual>3e-5:
        raise ValueError(f'{source}: wrist position frame disagrees with head-derived transform: {wrist_residual}')
    times=np.array([fr['timestamp'] for fr in frames])
    shifted,valid=interpolate_poses(times,world_head,times+parameters['pose_time_offset_seconds'])
    cam_from_head=np.array(parameters['T_optical_from_arkit_head'])
    cam_from_kp=cam_from_head[None]@np.linalg.inv(shifted)@np.linalg.inv(kp_from_world)[None]
    audit=dict(source_pose=str(source),converter_head_max_abs=head_residual,
               converter_wrist_position_max_abs_m=wrist_residual,
               kp_from_arkit_world=kp_from_world.tolist(),
               pose_time_offset_seconds=parameters['pose_time_offset_seconds'],
               time_outside_recording_frames=int((~valid).sum()))
    return cam_from_kp,valid,audit


def robot_transforms(f,parameters):
    state=np.asarray(f['observation.state'])
    if str(f.attrs.get('coordinate_convention',''))!='initial_heading_frame_x_forward_y_left_z_up':
        raise ValueError('Robot KP frame convention needs a separate audit')
    # This HAT state uses torso orientation and torso+[0,0,.4] virtual head,
    # matching G1CurrentPoseFK. User supplied a distinct transform for that frame.
    kp_from_virtual=poses_from_state(state)
    # Audit the virtual-head definition from the original stored root/joints.
    # These audit-only joint values are NEVER passed into the conditioner.
    from scipy.spatial.transform import Rotation
    root_rotation=Rotation.from_quat(state[:,103:107][:,[1,2,3,0]]).as_matrix()
    yaw=Rotation.from_euler('z',state[:,119,None]).as_matrix()
    roll=Rotation.from_euler('x',state[:,120,None]).as_matrix()
    pitch=Rotation.from_euler('y',state[:,121,None]).as_matrix()
    torso=root_rotation@yaw@roll@pitch
    virtual_position=state[:,100:103]+root_rotation@yaw@np.array([-.0039635,0,.044])+torso@np.array([0,0,.4])
    fk_position_error=float(np.abs(virtual_position-state[:,:3]).max())
    fk_rotation_error=float(np.abs(torso-kp_from_virtual[:,:3,:3]).max())
    if max(fk_position_error,fk_rotation_error)>1e-4:
        raise ValueError(f'Robot virtual head does not match stored FK: {fk_position_error}, {fk_rotation_error}')
    virtual_from_camera=np.array(parameters['T_head_link_from_left_optical'])
    virtual_from_camera[:3,3]=parameters['t_virtual_head_from_left_optical']
    cam_from_kp=np.linalg.inv(virtual_from_camera)[None]@np.linalg.inv(kp_from_virtual)
    valid=np.isfinite(kp_from_virtual).all(axis=(1,2))
    return cam_from_kp,valid,dict(head_frame='virtual_torso_plus_0.4m',
        audit_fk_head_max_abs_m=fk_position_error,audit_fk_rotation_max_abs=fk_rotation_error,
        camera_status=parameters['status'],pose_source='unaugmented observation.state[:9]',
        root_origin_handled='head and wrist already share the translated initial-heading KP frame')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=runtime.ROOT/'calibration_sidecars_20261021')
    args=parser.parse_args();out=args.output.resolve()
    if not out.is_relative_to(runtime.ROOT):raise ValueError('Sidecars must stay inside the experiment')
    out.mkdir(parents=True,exist_ok=True)
    configs=json.loads((runtime.ROOT/'configs/cameras_20261021.json').read_text())
    manifest={};audit={};counts={}
    for split,entries in json.loads(runtime.DATASET.read_text()).items():
        for entry in entries:
            group=Path(entry['dataset_path']).name
            if group=='human_old':
                print(f'{split}/human_old: uncalibrated baseline fallback ({len(entry["file_list"])} episodes)',flush=True)
                continue
            for filename in entry['file_list']:
                path=(Path(entry['dataset_path'])/filename).resolve()
                with h5py.File(path,'r') as f:
                    if group=='human_new':
                        parameters=configs['human_new_visionpro_screen'];key='observation.image.top'
                        transforms,valid,detail=human_transforms(f,parameters)
                        hand_valid=np.array(parameters['hand_valid_left_right'],bool)
                        rectified=True;distortion_model='pinhole_equivalent_screen'
                    elif group=='robot_new':
                        parameters=configs['robot_zed_left'];key='observation.image.left'
                        transforms,valid,detail=robot_transforms(f,parameters)
                        hand_valid=np.ones(2,bool);rectified=False;distortion_model='opencv_rational_8'
                    else:raise ValueError(group)
                    image=stored_image(f,key);raw_hw=image.shape[:2]
                    K=intrinsics_at_stored_resolution(parameters['K'],parameters['image_hw'],raw_hw)
                    filename_out=f'{group}_{path.stem}.npz'
                    np.savez_compressed(out/filename_out,K=K.astype(np.float32),D=np.array(parameters['D'],np.float32),
                        T_cam_from_kp=transforms.astype(np.float32),valid=valid,raw_hw=np.array(raw_hw),hand_valid=hand_valid)
                    manifest[str(path)]=dict(npz=str(out/filename_out),rgb_key=key,embodiment=str(f.attrs['embodiment']),
                        kp_frame='existing_original_HAT_episode_frame',kp_units='m',rectified=rectified,
                        distortion_model=distortion_model,provenance=str(runtime.ROOT/'configs/cameras_20261021.json'),
                        calibration_scope='human_new_right_only' if group=='human_new' else 'robot_coarse_candidate')
                    audit[str(path)]=detail
                    count=counts.setdefault(split+'/'+group,[0,0,0]);count[0]+=1;count[1]+=len(valid);count[2]+=int(valid.sum())
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (out/'transform_provenance.json').write_text(json.dumps(audit,indent=2))
    print('sidecar coverage [episodes,raw frames,valid camera frames]',counts,flush=True)
    print('manifest',out/'manifest.json',flush=True)


if __name__=='__main__':main()
