"""Offline-compatible model loader; live deployment is never changed here."""
import runtime
import pickle
from pathlib import Path
import torch
from hat_adapter import SCDPHAT


def load_scdp_policy(checkpoint,device='cuda'):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if saved.get('base_variant')!=runtime.BASE_VARIANT:
        raise ValueError('Set SCDP_BASE_VARIANT='+str(saved.get('base_variant'))+' before importing this loader')
    if Path(saved['base']).resolve()!=runtime.BASE.resolve():
        raise ValueError('Adapter checkpoint belongs to another baseline')
    policy,config=runtime.build_policy()
    wrapper=SCDPHAT(policy.model,saved['conditioner_config'])
    wrapper.conditioner.load_state_dict(saved['conditioner'],strict=True)
    policy.model=wrapper;policy.to(device).eval()
    stats=pickle.load(open(runtime.BASE/'dataset_stats.pkl','rb'))
    return policy,stats,config
