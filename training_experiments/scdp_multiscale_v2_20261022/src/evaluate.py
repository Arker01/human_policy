"""Prior-only checkpoint check on original held-out episodes; console output only."""
import runtime
import argparse,json
import torch
from load_policy import load_scdp_policy
from calibrated_data import CalibrationStore,CalibratedDataset
from data_utils_hdt import EpisodicDataset,gather_hdf_paths,get_all_episode_len
from modeling.utils import make_visual_encoder
from train import offline_check


def main():
    parser=argparse.ArgumentParser();parser.add_argument('checkpoint');args=parser.parse_args()
    torch.set_num_threads(4)
    policy,stats,config=load_scdp_policy(args.checkpoint)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    entries=sorted(json.loads(runtime.DATASET.read_text())['val'],key=lambda x:x['dataset_path'])
    paths,counts=gather_hdf_paths(str(runtime.REPO/'data'),entries,['top'])
    paths=sorted(paths)
    _,preprocessor=make_visual_encoder('ACT',{})
    base=EpisodicDataset(config['data'],paths,[None]*len(entries),['top'],100,
        get_all_episode_len(paths),counts,preprocessor,0.,train=False,slow_down_factor=4,
        action_representation='original',robot_target_fps=30.)
    dataset=CalibratedDataset(base,CalibrationStore(saved['calibration_manifest']),stats)
    print('Reloaded checkpoint step',saved['step'],flush=True)
    offline_check(policy,policy.model,dataset,'cuda')


if __name__=='__main__':main()
