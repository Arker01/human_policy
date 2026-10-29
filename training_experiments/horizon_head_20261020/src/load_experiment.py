"""Load isolated weights; returned policy accepts the original normalized inputs.

For readiness, policy.predict_with_readiness(image, normalized_qpos) returns
(action_chunk, probabilities), ordered left-close, left-open, right-close,
right-open. The auxiliary probability is available for deployment gating; this
loader never sends actions to a robot.
"""
import sys,json,pathlib,pickle,torch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent/'hdt'))
from policy import ACTPolicy

def load_experiment(checkpoint_dir, device='cuda', weights='policy_last.ckpt'):
    checkpoint_dir=pathlib.Path(checkpoint_dir)
    config=json.load(open(checkpoint_dir/'training_config.json'))
    args=config['config']['policy_config'].copy()
    args['experiment_variant']=config['variant']
    policy=ACTPolicy(args).to(device)
    state=torch.load(checkpoint_dir/weights,map_location=device,weights_only=True)
    policy.load_state_dict(state,strict=True)
    policy.eval()
    stats=pickle.load(open(checkpoint_dir/'dataset_stats.pkl','rb'))
    return policy,stats,config
