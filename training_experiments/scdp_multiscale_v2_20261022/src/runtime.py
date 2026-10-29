"""Choose the exact isolated geometry source/checkpoint before any HAT imports."""
import os,sys,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EXPERIMENT=json.loads((ROOT/'configs/experiment.json').read_text())
REPO=Path(os.environ.get('HUMAN_POLICY_ROOT',ROOT.parents[1]))
BASE_VARIANT=os.environ.get('SCDP_BASE_VARIANT','geometry')
BASE_CONFIG=json.loads((ROOT/'configs/bases.json').read_text())[BASE_VARIANT]
SOURCE_ROOT=Path(BASE_CONFIG['source_root'])
sys.path[:0]=[str(SOURCE_ROOT/'hdt'),str(SOURCE_ROOT),str(REPO)]
os.environ['HDT_EXPERIMENT']=BASE_CONFIG['experiment_variant']
BASE=Path(BASE_CONFIG['checkpoint_dir'])
DATASET=Path(BASE_CONFIG['dataset_json'])
CALIBRATION_MANIFEST=ROOT.parent/'scdp_lite_20261020/calibration_sidecars_20261021/manifest.json'


def build_policy():
    import yaml,torch
    from policy import ACTPolicy
    cfg=yaml.safe_load((BASE/'model.yaml').read_text())
    if BASE_VARIANT == 'original_20261004':
        args=dict(cfg['model'], **cfg['common'], lr=1e-5, chunk_size=100, num_queries=100)
        args['zero_state_dims']=(98,128)
        args['lr_backbone']=float(args['lr_backbone'])
        args['future_dino_config']=cfg['model']['future_dino']
    else:
        saved=json.loads((BASE/'training_config.json').read_text())
        args=saved['config']['policy_config'].copy()
        args['experiment_variant']=saved['variant']
    original=torch.hub.load;cached=Path(torch.hub.get_dir())/'facebookresearch_dinov2_main'
    def cached_load(repository,model,*a,**kw):
        if repository=='facebookresearch/dinov2' and cached.is_dir():
            kw.pop('trust_repo',None);kw.pop('skip_validation',None)
            return original(str(cached),model,*a,source='local',**kw)
        return original(repository,model,*a,**kw)
    torch.hub.load=cached_load
    try:policy=ACTPolicy(args)
    finally:torch.hub.load=original
    policy.load_state_dict(torch.load(BASE/'policy_last.ckpt',map_location='cpu',weights_only=True),strict=True)
    expected=torch.ones(128,device=policy.model.state_keep_mask.device)
    for lo,hi in BASE_CONFIG['input_mask_ranges']:expected[lo:hi]=0
    if not torch.equal(policy.model.state_keep_mask,expected):
        raise RuntimeError('Checkpoint input mask differs from the selected base experiment')
    if getattr(policy.model,'experiment_variant','baseline')!=BASE_CONFIG['experiment_variant'] or policy.model.future_dino_weight!=0.:
        raise RuntimeError('Wrong geometry/world-head variant')
    print('BASE LOADED',json.dumps(dict(name=BASE_VARIANT,checkpoint=str(BASE),source=str(SOURCE_ROOT),
        input_mask_ranges=BASE_CONFIG['input_mask_ranges'],geometry_loss_weight=.1 if BASE_VARIANT!='original_20261004' else 0.,future_visual_weight=0.)),flush=True)
    return policy,cfg
