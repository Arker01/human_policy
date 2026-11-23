"""Resolve original HAT imports without modifying that repository."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get('HUMAN_POLICY_ROOT', ROOT.parents[1]))
sys.path[:0] = [str(REPO / 'hdt'), str(REPO)]
BASE = REPO / 'b0930_20261004_replace50_nostate_vjepa_w0_lagvis_ckpt'
DATASET = REPO / 'experiments/bottle0930_16runs_20261004/replace50.json'


def build_policy():
    import yaml
    import torch
    from policy import ACTPolicy
    cfg = yaml.safe_load((BASE / 'model.yaml').read_text())
    args = dict(cfg['model'], **cfg['common'], lr=1e-5, chunk_size=100, num_queries=100)
    args['zero_state_dims'] = (98, 128)
    args['lr_backbone'] = float(args['lr_backbone'])
    args['future_dino_config'] = cfg['model']['future_dino']
    # torch.hub's unpinned DINO repository lookup can make a network request to
    # discover its default branch even when this exact source is already cached.
    # Use that existing local cache during construction; no shared files change.
    original_hub_load=torch.hub.load
    cached_dino=Path(torch.hub.get_dir())/'facebookresearch_dinov2_main'
    def cached_hub_load(repository,model,*positional,**keywords):
        if repository=='facebookresearch/dinov2' and cached_dino.is_dir():
            keywords.pop('trust_repo',None);keywords.pop('skip_validation',None)
            return original_hub_load(str(cached_dino),model,*positional,source='local',**keywords)
        return original_hub_load(repository,model,*positional,**keywords)
    torch.hub.load=cached_hub_load
    try:
        policy = ACTPolicy(args)
    finally:
        torch.hub.load=original_hub_load
    policy.load_state_dict(torch.load(BASE / 'policy_last.ckpt', map_location='cpu', weights_only=True), strict=True)
    return policy, cfg
