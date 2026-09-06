import re
import csv
import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
import wandb
import time
import yaml

import accelerate
from accelerate import Accelerator
from data_utils_hdt import load_data # data functions
from data_utils_hdt import compute_dict_mean, set_seed, detach_dict # helper functions
from modeling.utils import make_visual_encoder

def _to_float(x):
    try:
        if hasattr(x, "detach"):
            return float(x.detach().cpu().item())
        return float(x)
    except Exception:
        return None

def _append_metrics_csv(csv_path, step, metrics: dict):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    row = {"step": int(step)}
    for k, v in metrics.items():
        fv = _to_float(v)
        if fv is not None:
            row[k] = fv
    fieldnames = ["step"] + sorted([k for k in row.keys() if k != "step"])
    file_exists = os.path.exists(csv_path)
    if file_exists:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            existing = reader.fieldnames or []
        fieldnames = sorted(set(existing).union(fieldnames), key=lambda x: (x != "step", x))
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def _plot_metrics(csv_path, out_png_path, keys):
    if not os.path.exists(csv_path):
        return
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return
    steps = [int(r["step"]) for r in rows if r.get("step") not in (None, "")]
    if not steps:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for k in keys:
        ys = []
        xs = []
        for r in rows:
            if r.get(k) is None or r.get(k) == "":
                continue
            try:
                xs.append(int(r["step"]))
                ys.append(float(r[k]))
            except Exception:
                continue
        if xs:
            ax.plot(xs, ys, label=k)
    ax.set_xlabel("step")
    ax.set_ylabel("value")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)
    fig.savefig(out_png_path, dpi=150)
    plt.close(fig)

def _iter_ckpt_weight_paths(ckpt_dir):
    entries = []
    if not os.path.isdir(ckpt_dir):
        return []
    for name in os.listdir(ckpt_dir):
        m = re.search(r"policy_iter_(\d+)_seed_(\d+)", name)
        if not m:
            continue
        step = int(m.group(1))
        entries.append((step, os.path.join(ckpt_dir, name)))
    entries.sort(key=lambda x: x[0])

    results = []
    for step, path in entries:
        weights_path = os.path.join(path, "pytorch_model.bin")
        if os.path.exists(weights_path):
            results.append((step, weights_path))

    last_path = os.path.join(ckpt_dir, "policy_last.ckpt")
    if os.path.exists(last_path):
        last_step = results[-1][0] + 1 if results else 0
        results.append((last_step, last_path))
    return results

def eval_checkpoints(accelerator, val_dataloader, policy, ckpt_dir):
    out_csv = os.path.join(ckpt_dir, "retro_metrics.csv")
    out_png = os.path.join(ckpt_dir, "retro_metrics.png")

    ckpts = _iter_ckpt_weight_paths(ckpt_dir)
    if not ckpts:
        raise RuntimeError(f"No checkpoints found under {ckpt_dir}")

    device = next(policy.parameters()).device
    for step, weights_path in ckpts:
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        accelerator.unwrap_model(policy).load_state_dict(state_dict, strict=False)
        with torch.no_grad():
            policy.eval()
            validation_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                validation_dicts.append(forward_dict)
                if batch_idx > 20:
                    break
            validation_summary = compute_dict_mean(validation_dicts)
            for k in list(validation_summary.keys()):
                validation_summary[f"val/{k}"] = validation_summary.pop(k)
        if accelerator.is_main_process:
            _append_metrics_csv(out_csv, step, validation_summary)
            _plot_metrics(out_csv, out_png, keys=["val/loss", "val/l1", "val/eef_loss", "val/kl"])
            summary_string = " ".join([f"{k}: {_to_float(v):.6f}" for k, v in validation_summary.items() if _to_float(v) is not None])
            print(f"[eval ckpt step={step}] {os.path.basename(weights_path)} {summary_string}")

def make_policy(policy_class, policy_config, visual_encoder, USE_PRETRAINED=True):
    if policy_class == 'ACT':
        from policy import ACTPolicy
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        from policy import CNNMLPPolicy
        policy = CNNMLPPolicy(policy_config)
    elif policy_class == 'RDT':
        from modeling.modeling_hdt import HumanDiffusionTransformer
        policy = HumanDiffusionTransformer(
            action_dim=policy_config["common"]["state_dim"],
            pred_horizon=policy_config["common"]["action_chunk_size"],
            config=policy_config,
            lang_token_dim=policy_config["model"]["lang_token_dim"],
            img_token_dim=policy_config["model"]["img_token_dim"],
            state_token_dim=policy_config["model"]["state_token_dim"],
            max_lang_cond_len=policy_config["dataset"]["tokenizer_max_length"],
            visual_encoder=visual_encoder,
            lang_pos_embed_config=[
                # Similarly, no initial pos embed for language
                ("lang", -policy_config["dataset"]["tokenizer_max_length"]),
            ],
            dtype=torch.float32,
        )
        if USE_PRETRAINED:
            RDTS_DIR = '/data/pretrained_weights/rdt-170m'
            POLICY_PATH = os.path.join(RDTS_DIR, 'pytorch_model.bin')
            state_dict = torch.load(POLICY_PATH, map_location=next(policy.parameters()).device, weights_only=True)
            # remove pos embeddings
            state_dict = {k: v for k, v in state_dict.items() if 'pos_embed' not in k}
            policy.load_state_dict(state_dict, strict=False)  # type: ignore
    elif policy_class == "DP":
        from modeling.modeling_vanilla_dp import DiffusionPolicy
        policy = DiffusionPolicy(action_dim=policy_config["action_dim"],
            chunk_size=policy_config["chunk_size"],
            img_token_dim=visual_encoder.hidden_size,
            state_token_dim=policy_config["state_dim"],
            num_inference_timesteps=policy_config["num_inference_timesteps"],
            visual_encoder=visual_encoder)
    else:
        raise NotImplementedError
    return policy

def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'RDT' or policy_class == 'DP':
        parameter_list = []
        for name, param in policy.named_parameters():
            if name.startswith("vision_encoder"):
                continue
            parameter_list.append(param)

        optimizer = torch.optim.AdamW(
            parameter_list,
            lr=1e-4,  # from RDT pretrain
        )
    else:
        raise NotImplementedError
    return optimizer

def main(args, base_dir, processed_dir=None):
    set_seed(1)
    with open(args["model_cfg_path"], "r") as fp:
        trainer_config = yaml.safe_load(fp)

    # Override Future-DINO config from CLI if specified
    if args.get('use_future_dino_head'):
        if 'future_dino' not in trainer_config.get('model', {}):
            trainer_config.setdefault('model', {})['future_dino'] = {}
        trainer_config['model']['future_dino']['enabled'] = True

    if args.get('future_dino_weight') is not None:
        if 'future_dino' not in trainer_config.get('model', {}):
            trainer_config.setdefault('model', {})['future_dino'] = {}
        trainer_config['model']['future_dino']['weight'] = args['future_dino_weight']

    if args.get('future_dino_warmup_steps') is not None:
        if 'future_dino' not in trainer_config.get('model', {}):
            trainer_config.setdefault('model', {})['future_dino'] = {}
        trainer_config['model']['future_dino']['warmup_steps'] = args['future_dino_warmup_steps']

    if args.get('future_dino_horizon') is not None:
        if 'future_dino' not in trainer_config.get('model', {}):
            trainer_config.setdefault('model', {})['future_dino'] = {}
        trainer_config['model']['future_dino']['horizon'] = args['future_dino_horizon']

    if args.get('future_dino_ablation') is not None:
        if 'future_dino' not in trainer_config.get('model', {}):
            trainer_config.setdefault('model', {})['future_dino'] = {}
        trainer_config['model']['future_dino']['ablation'] = args['future_dino_ablation']

    # Second future-target species (frozen Wan VAE). Same CLI-overrides-yaml pattern as
    # future_dino above. It shares future_dino's horizon and future frame by design, so
    # there is deliberately no --future_vae_horizon.
    if args.get('use_future_vae_head'):
        trainer_config.setdefault('model', {}).setdefault('future_vae', {})['enabled'] = True

    if args.get('future_vae_weight') is not None:
        trainer_config.setdefault('model', {}).setdefault('future_vae', {})['weight'] = args['future_vae_weight']

    if args.get('future_vae_ablation') is not None:
        trainer_config.setdefault('model', {}).setdefault('future_vae', {})['ablation'] = args['future_vae_ablation']

    if args.get('future_vae_normalize_target') is not None:
        trainer_config.setdefault('model', {}).setdefault('future_vae', {})['normalize_target'] = \
            bool(args['future_vae_normalize_target'])

    # Third future-target species (EgoWAM 3D point flow). Same CLI-overrides-yaml
    # pattern. Unlike future_vae this one does NOT ride on future_dino's future frame
    # -- its target is precomputed on disk -- so a flow-only arm is legal.
    if args.get('use_future_flow_head'):
        trainer_config.setdefault('model', {}).setdefault('future_flow', {})['enabled'] = True

    if args.get('future_flow_weight') is not None:
        trainer_config.setdefault('model', {}).setdefault('future_flow', {})['weight'] = args['future_flow_weight']

    if args.get('future_flow_ablation') is not None:
        trainer_config.setdefault('model', {}).setdefault('future_flow', {})['ablation'] = args['future_flow_ablation']

    if args.get('future_flow_dir') is not None:
        trainer_config.setdefault('model', {}).setdefault('future_flow', {})['target_dir'] = args['future_flow_dir']

    if args.get('zero_state_dims') is not None:
        trainer_config.setdefault('model', {})['zero_state_dims'] = args['zero_state_dims']

    # "100:128" -> (100, 128). None means "leave the state alone", the default.
    zero_state_dims = trainer_config.get('model', {}).get('zero_state_dims', None)
    if isinstance(zero_state_dims, str):
        lo, hi = zero_state_dims.split(':')
        zero_state_dims = (int(lo), int(hi))
    elif zero_state_dims is not None:
        zero_state_dims = tuple(int(v) for v in zero_state_dims)
    if zero_state_dims is not None:
        print(f"\n=== State ablation: normalized qpos[{zero_state_dims[0]}:{zero_state_dims[1]}] "
              f"forced to 0 for every embodiment ===")

    future_vae_cfg = trainer_config.get('model', {}).get('future_vae', {}) or {}
    future_vae_enabled = bool(future_vae_cfg.get('enabled', False))

    future_dino_cfg = trainer_config.get('model', {}).get('future_dino', {}) or {}
    future_dino_enabled = bool(future_dino_cfg.get('enabled', False))
    # The horizon drives the dataloader, so it has to be resolved here, not in the model.
    future_dino_horizon = int(future_dino_cfg.get('horizon', 16)) if future_dino_enabled else 0
    # Same for the clip shape: a video teacher (V-JEPA 2) needs the dataloader to read
    # K frames per side instead of 1. Defaults keep the single-frame path unchanged.
    future_dino_clip_frames = int(future_dino_cfg.get('clip_frames', 1)) if future_dino_enabled else 1
    future_dino_clip_stride = int(future_dino_cfg.get('clip_stride', 1)) if future_dino_enabled else 1

    # Print Future-DINO config if enabled
    if future_dino_enabled:
        print("Future-DINO World Head enabled:")
        print(f"  Weight: {future_dino_cfg.get('weight', 0.3)}")
        print(f"  Warmup steps: {future_dino_cfg.get('warmup_steps', 1000)}")
        print(f"  Num layers: {future_dino_cfg.get('num_layers', 4)}")
        print(f"  Num heads: {future_dino_cfg.get('num_heads', 8)}")
        print(f"  Horizon (frames): {future_dino_horizon}")
        print(f"  Target encoder (frozen): {future_dino_cfg.get('target_encoder', 'dinov2_vits14')}")
        print(f"  Clip frames / stride: {future_dino_clip_frames} / {future_dino_clip_stride}")
        print(f"  Ablation: {future_dino_cfg.get('ablation', 'none')}")

    if future_vae_enabled:
        assert future_dino_enabled, (
            "future_vae reuses the future frame the Future-DINO path requests from the "
            "dataloader. Enable future_dino too (--future_dino_weight 0.0 gives a "
            "VAE-only arm with the DINO head attached but silent).")
        print("Future-VAE (Wan) target enabled:")
        print(f"  Weight: {future_vae_cfg.get('weight', 1.0)}")
        print(f"  Target encoder (frozen): {future_vae_cfg.get('target_encoder', 'wan22_vae')}")
        print(f"  Normalize target: {future_vae_cfg.get('normalize_target', False)}")
        print(f"  Huber weight: {future_vae_cfg.get('huber_weight', 1.0)}")
        print(f"  Ablation: {future_vae_cfg.get('ablation', 'none')}")

    future_flow_cfg = trainer_config.get('model', {}).get('future_flow', {}) or {}
    future_flow_enabled = bool(future_flow_cfg.get('enabled', False))
    future_flow_dir = future_flow_cfg.get('target_dir', None) if future_flow_enabled else None
    future_flow_grid_hw = tuple(future_flow_cfg.get('grid_hw', (30, 40)))
    future_flow_horizon = int(future_flow_cfg.get('horizon', args['chunk_size']))

    if future_flow_enabled:
        assert future_flow_dir, (
            "future_flow.target_dir must point at the output of "
            "scripts/preprocess/flow_target.py (one h5 per episode).")
        assert future_flow_horizon == args['chunk_size'], (
            f"future_flow.horizon ({future_flow_horizon}) must equal chunk_size "
            f"({args['chunk_size']}): the head cross-attends to hat_memory, whose "
            f"token k IS chunk step k, so the two trajectories are indexed together.")
        print("Future-Flow (EgoWAM 3D point flow) target enabled:")
        print(f"  Weight: {future_flow_cfg.get('weight', 1.0)}")
        print(f"  Target dir: {future_flow_dir}")
        print(f"  Anchor grid: {future_flow_grid_hw} "
              f"({future_flow_grid_hw[0] * future_flow_grid_hw[1]} anchors)")
        print(f"  Horizon (chunk steps): {future_flow_horizon}")
        print(f"  Huber beta (m): {future_flow_cfg.get('huber_beta', 0.01)}")
        print(f"  Ablation: {future_flow_cfg.get('ablation', 'none')}")

    policy_class = trainer_config["common"]["policy_class"]
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']

    # get task parameters
    task_name = "hdt"
    ckpt_dir = args["exptid"] + "_ckpt"

    camera_names = trainer_config['common']['camera_names']

    # fixed parameters
    # TODO(roger): consolidate these to just loading from yaml
    state_dim = 128
    action_dim = 128
    if policy_class == 'ACT':
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': trainer_config['model']['kl_weight'],
                         'hidden_dim': trainer_config['model']['hidden_dim'],
                         'chunk_size': args['chunk_size'],
                         'dim_feedforward': trainer_config['model']['dim_feedforward'],
                         'lr_backbone': float(trainer_config['model']['lr_backbone']),
                         'backbone': trainer_config['model']['backbone'],
                         'enc_layers': trainer_config['model']['enc_layers'],
                         'dec_layers': trainer_config['model']['dec_layers'],
                         'nheads': trainer_config['model']['nheads'],
                         'camera_names': camera_names,
                         'state_dim': state_dim,
                         'action_dim': action_dim,
                         'image_feature_strategy': trainer_config['model']['image_feature_strategy'],
                         'use_language_conditioning': trainer_config['model']['use_language_conditioning'],
                         'query0_extra_weight': args.get('query0_extra_weight', 0.0),
                         }
        # Add Future-DINO config if enabled
        if future_dino_enabled:
            policy_config['future_dino_config'] = future_dino_cfg
        if future_vae_enabled:
            policy_config['future_vae_config'] = future_vae_cfg
        if future_flow_enabled:
            policy_config['future_flow_config'] = future_flow_cfg
        if zero_state_dims is not None:
            policy_config['zero_state_dims'] = zero_state_dims
    elif policy_class == 'RDT':
        assert "visual_backbone" not in trainer_config
        trainer_config["visual_backbone"] = trainer_config["model"]["backbone"]
        policy_config = trainer_config
    elif policy_class == 'DP':
        policy_config = {
            'action_dim': action_dim,
            'state_dim': state_dim,
            'chunk_size': args['chunk_size'],
            'visual_backbone': 'MASKCLIP',
            'num_inference_timesteps': 20,
        }
    else:
        raise NotImplementedError
    
    from accelerate.utils import DistributedDataParallelKwargs
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(project_dir=ckpt_dir, kwargs_handlers=[kwargs])

    #!
    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        # 'episode_len': episode_len,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'policy_config': policy_config,
        'seed': args['seed'],
        'camera_names': camera_names,
        'val_and_jit_trace': args['val_and_jit_trace'],
        'task_name': task_name,
        'exptid': args['exptid'],
        'load_pretrained_path': args['load_pretrained_path'],
    }
    mode = "disabled" if args["no_wandb"] or args["val_and_jit_trace"] else "online"
    if True:
        # NOTE(roger): disable wandb for public release
        mode = "disabled"
    if accelerator.is_main_process and mode != "disabled":
        wandb_id = re.sub(r"[^A-Za-z0-9_.-]", "_", str(args["exptid"]))
        wandb_name = str(args["exptid"]).split("/")[-1]
        wandb.init(
            project="human2robot",
            name=wandb_name,
            group="RogerQiu",
            entity="RogerQiu",
            mode=mode,
            dir="../data/logs",
            id=wandb_id,
            resume="allow",
        )
        wandb.config.update(config)
    
    visual_encoder, visual_preprocessor = make_visual_encoder(policy_class, policy_config)
    policy = make_policy(policy_class, policy_config, visual_encoder)
    optimizer = make_optimizer(policy_class, policy)

    train_dataloader, val_dataloader, stats = load_data(base_dir, 
                                                        trainer_config["data"],
                                                        args["dataset_json_path"],
                                                        camera_names, 
                                                        args["chunk_size"],
                                                        batch_size_train, 
                                                        batch_size_val, 
                                                        visual_preprocessor,
                                                        args['cond_mask_prob'],
                                                        args['human_slow_down_factor'],
                                                        [processed_dir] if processed_dir else None,
                                                        auto_trim_dirty_start=True,
                                                        dirty_start_check_frames=args.get('dirty_start_check_frames', 20),
                                                        dirty_start_jump_threshold_m=args.get('dirty_start_jump_threshold_m', 0.3),
                                                        dirty_start_settle_frames=args.get('dirty_start_settle_frames', 0),
                                                        future_image_enabled=future_dino_enabled,
                                                        future_horizon=future_dino_horizon,
                                                        future_clip_frames=future_dino_clip_frames,
                                                        future_clip_stride=future_dino_clip_stride,
                                                        flow_target_enabled=future_flow_enabled,
                                                        flow_target_dir=future_flow_dir,
                                                        flow_grid_hw=future_flow_grid_hw,
                                                        flow_horizon=future_flow_horizon)

    if args.get("eval_ckpts", False):
        val_dataloader, policy = accelerator.prepare(val_dataloader, policy)
        eval_checkpoints(accelerator, val_dataloader, policy, ckpt_dir)
        return

    # save dataset stats
    if accelerator.is_main_process:
        if not os.path.isdir(ckpt_dir):
            os.makedirs(ckpt_dir)
        stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
        with open(stats_path, 'wb') as f:
            pickle.dump(stats, f)

    train_fn(accelerator, train_dataloader, val_dataloader, policy, optimizer, config)

def maybe_to_tensor(element, to_target):
    if isinstance(element, torch.Tensor):
        return element.to(to_target)
    elif isinstance(element, dict):
        for k, v in element.items():
            if isinstance(v, torch.Tensor):
                element[k] = v.to(to_target)
        return element

#!!! we also change it to tensor in the forward pass
def forward_pass(data, policy, training_step=0):
    device = next(policy.parameters()).device
    if isinstance(data, dict):
        for k, v in data.items():
            data[k] = maybe_to_tensor(v, device)
    elif isinstance(data, list):
        for i in range(len(data)):
            data[i] = maybe_to_tensor(data[i], device)

    # Set training step for Future-DINO loss warmup.
    # Must unwrap DDP first: under accelerate `policy` is a DistributedDataParallel,
    # so `policy.model` is not the ACT model and the step never landed -> warmup
    # factor stayed at 0 -> effective_weight stayed at 0 for the whole run.
    inner = policy.module if hasattr(policy, 'module') else policy
    for candidate in (inner, getattr(inner, 'model', None)):
        if candidate is None:
            continue
        if hasattr(candidate, 'set_training_step'):
            candidate.set_training_step(training_step)

    return policy(*(data))

class WarmupMultiplicativeLR(torch.optim.lr_scheduler._LRScheduler):
    """Custom scheduler that multiplies lr by 10 every 1000 steps until reaching max_lr"""
    def __init__(self, optimizer, initial_lr=1e-7, max_lr=1e-4, warmup_period=1000, last_epoch=-1):
        self.initial_lr = initial_lr
        self.max_lr = max_lr
        self.warmup_period = warmup_period
        # Set initial lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = initial_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        # Calculate current multiplier based on epoch
        multiplier = 10 ** (self.last_epoch // self.warmup_period)
        new_lr = min(self.initial_lr * multiplier, self.max_lr)
        return [new_lr for _ in self.base_lrs]

def maybe_load_ckpt(ckpt_dir, seed, train_from_iter):
    ckpt_names = os.listdir(ckpt_dir)
    max_ckpt_name = None
    for ckpt_name in ckpt_names:
        match = re.search(r'policy_iter_(\d+)_seed_(\d+)', ckpt_name)
        if match:
            loaded_iter = int(match.group(1)) 
            cur_seed = int(match.group(2))
            assert cur_seed == seed, f"seed mismatch: {cur_seed} vs {seed}"
            if loaded_iter > train_from_iter:
                train_from_iter = loaded_iter
                max_ckpt_name = ckpt_name
    return train_from_iter, max_ckpt_name

def train_fn(accelerator, train_dataloader, val_dataloader, policy, optimizer, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    metrics_csv = os.path.join(ckpt_dir, "metrics.csv")
    metrics_png = os.path.join(ckpt_dir, "metrics.png")

    state = accelerate.state.AcceleratorState()
    process_idx = state.process_index

    set_seed(process_idx * 1000 + seed)

    min_val_loss = np.inf

    if config['load_pretrained_path'] is not None:
        print(f"Loading pretrained model from {config['load_pretrained_path']}")
        state_dict = torch.load(config['load_pretrained_path'], map_location=next(policy.parameters()).device, weights_only=True)
        policy.load_state_dict(state_dict, strict=False)
        # Create custom scheduler with configurable learning rate
        # 使用配置中的学习率，默认为较低的值以保护 finetune
        finetune_lr = config.get('finetune_lr', 1e-6)
        finetune_warmup_period = config.get('finetune_warmup_period', 1000)
        print(f"Finetune mode: max_lr={finetune_lr}, warmup_period={finetune_warmup_period}")
        scheduler = WarmupMultiplicativeLR(
            optimizer,
            initial_lr=finetune_lr / 1000,  # 初始学习率为 max_lr 的 1/1000
            max_lr=finetune_lr,
            warmup_period=finetune_warmup_period
        )
    else:
        # use constant LR scheduler
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=num_epochs)

    train_dataloader, policy, optimizer, scheduler = accelerator.prepare(train_dataloader, policy, optimizer, scheduler)

    if config['load_pretrained_path'] is not None:
        print(f"Loading pretrained model from {config['load_pretrained_path']}")
        state_dict = torch.load(config['load_pretrained_path'], map_location=next(policy.parameters()).device, weights_only=True)
        policy.load_state_dict(state_dict, strict=False)

    train_from_iter = 0

    train_from_iter, max_ckpt_name = maybe_load_ckpt(ckpt_dir, seed, train_from_iter)
    if train_from_iter > 0:
        print(f"Resuming from iter {train_from_iter}")
        ckpt_path = os.path.join(ckpt_dir, max_ckpt_name)
        try:
            accelerator.load_state(ckpt_path)
        except Exception as e:
            print(f"Failed to load full checkpoint state from {ckpt_path}: {e}")
            weights_path = os.path.join(ckpt_path, "pytorch_model.bin")
            print(f"Falling back to loading model weights only from {weights_path}")
            state_dict = torch.load(
                weights_path,
                map_location=next(policy.parameters()).device,
                weights_only=True,
            )
            accelerator.unwrap_model(policy).load_state_dict(state_dict, strict=False)

    policy.train()
    cur_iter = train_from_iter

    with tqdm(total=num_epochs, initial=train_from_iter) as pbar:
        for data in train_dataloader:
            if cur_iter >= num_epochs or config['val_and_jit_trace']:
                break

            if cur_iter % 1000 == 0:
            # validation
                with torch.no_grad():
                    policy.eval()
                    validation_dicts = []
                    for batch_idx, data in enumerate(val_dataloader):
                        forward_dict = forward_pass(data, policy, training_step=cur_iter)
                        validation_dicts.append(forward_dict)
                        if batch_idx > 20:
                            break

                    validation_summary = compute_dict_mean(validation_dicts)

                    epoch_val_loss = validation_summary['loss']
                if accelerator.is_main_process:
                    print(f'\n Iter {cur_iter}')
                    for k in list(validation_summary.keys()):
                        validation_summary[f'val/{k}'] = validation_summary.pop(k)

                    if wandb.run is not None:
                        wandb.log(validation_summary, step=cur_iter)
                    _append_metrics_csv(metrics_csv, cur_iter, validation_summary)
                    _plot_metrics(
                        metrics_csv,
                        metrics_png,
                        keys=["val/loss", "val/l1", "val/eef_loss", "val/kl", "train/loss", "train/l1", "train/eef_loss", "train/kl", "train/future_dino_loss", "val/future_dino_loss", "train/future_dino_cosine_loss", "val/future_dino_cosine_loss"],
                    )
                    print(f'Val loss:   {epoch_val_loss:.5f}')
                    summary_string = ''
                    for k, v in validation_summary.items():
                        summary_string += f'{k}: {v.item():.3f} '
                    print(summary_string)

                    if config['val_and_jit_trace']:
                        break

                policy.train()

            forward_dict = forward_pass(data, policy, training_step=cur_iter)
            # backward
            loss = forward_dict['loss']

            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            pbar.update(1)
            cur_iter += 1
            
            if accelerator.is_main_process:
                epoch_summary = detach_dict(forward_dict)
                epoch_summary['lr'] = torch.tensor(scheduler.get_last_lr()[0])
                summary_string = ''
                for k, v in epoch_summary.items():
                    summary_string += f'{k}: {v.item():.3f} '
                # print(summary_string)
                train_log = {f"train/{k}": v for k, v in epoch_summary.items() if k != "lr"}
                train_log["lr"] = epoch_summary["lr"]
                if wandb.run is not None:
                    wandb.log(train_log, step=cur_iter)
                _append_metrics_csv(metrics_csv, cur_iter, train_log)

                #! save ckpt
                if cur_iter % 10000 == 0 and cur_iter != 0:
                    ckpt_path = os.path.join(ckpt_dir, f'policy_iter_{cur_iter}_seed_{seed}')
                    accelerator.save_state(ckpt_path, safe_serialization=False)
                    # torch.save(policy.state_dict(), ckpt_path)
                    print(f'Saved ckpt at iter {cur_iter}')
                
                
    
    if config['val_and_jit_trace']:
        # JIT trace
        class polciy_wrapper(torch.nn.Module):
            def __init__(self, policy):
                super().__init__()
                self.policy = policy

            def forward(self, image, qpos):
                return self.policy(image, qpos, conditioning_dict={})

        # TRACING_DEVICE = 'cuda'
        TRACING_DEVICE = 'cuda'
        # Rollout validation
        my_policy_wrapper = polciy_wrapper(policy)
        my_policy_wrapper.eval().to(TRACING_DEVICE)

        # Benchmark
        # With Future-DINO enabled the loader yields 6 items (trailing future_image), which
        # the 5-way unpack could not take -- tracing crashed exactly on the configs that
        # need tracing. The head is training-only and the traced graph is eval-mode, so the
        # extra element is simply dropped here. Disabled path is byte-identical (5 items).
        image, qpos, _, _, conditioning_dict = data[:5]
        image = image.to(TRACING_DEVICE)
        qpos = qpos.to(TRACING_DEVICE)
        conditioning_dict = {k: maybe_to_tensor(v, TRACING_DEVICE) for k, v in conditioning_dict.items()}
        # warm up
        for _ in range(10):
            trajectory = my_policy_wrapper(image[0:1], qpos[0:1])
        # benchmark speed
        start = time.time()
        N_iters = 50
        for _ in range(N_iters):
            trajectory = my_policy_wrapper(image[0:1], qpos[0:1])
        end = time.time()
        print("Total time: ", end - start)
        print(f"Rollout speed: {N_iters / (end - start)} Hz")

        # Jit trace
        image_data = torch.rand(image[0:1].shape, device=TRACING_DEVICE)
        qpos_data = torch.rand(qpos[0:1].shape, device=TRACING_DEVICE)
        input_data = (image_data, qpos_data)

        traced_policy = torch.jit.trace(my_policy_wrapper, input_data)

        traced_path = os.path.join(ckpt_dir, f'policy_traced.pt')
        traced_policy.save(traced_path)
        del traced_policy
        torch.cuda.empty_cache()

        loaded_policy = torch.jit.load(traced_path)
        # Manually set seed to make diffusion sampling deterministic in comparisons
        torch.random.manual_seed(0)
        jit_output = loaded_policy(image_data, qpos_data)
        torch.random.manual_seed(0)
        vanilla_output = my_policy_wrapper(image_data, qpos_data)

        l1_err = torch.nn.functional.l1_loss(jit_output, vanilla_output, reduction='none').cpu().detach().numpy()
        assert (l1_err < 1e-3).all(), f"JIT trace error: {l1_err.max()}"

        exit(0)
            
    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=False, default=0)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--val_and_jit_trace', action='store_true')
    parser.add_argument('--exptid', action='store', type=str, help='experiment id', required=True)
    parser.add_argument('--epoch', action='store', type=str, help='epoch num', required=False)
    parser.add_argument('--cond_mask_prob', type=float, default=0.1, help='cond_mask_prob', required=False)
    parser.add_argument('--query0_extra_weight', type=float, default=0.0, help='extra L1 weight anchoring a_hat[:,0] to actions[:,0], to counter chunk-loss query0 lag', required=False)
    parser.add_argument('--dataset_json_path', type=str, help='dataset_json_path', required=True)
    parser.add_argument('--model_cfg_path', type=str, help='path to model cfg yaml', required=True)
    parser.add_argument('--human_slow_down_factor', type=int, default=4, help='human demonstrations slow_down_factor', required=False)
    parser.add_argument('--no_auto_trim_dirty_start', action='store_true', help='disable automatic trimming of early dirty motion frames')
    parser.add_argument('--dirty_start_check_frames', type=int, default=20, help='number of early frames to scan for dirty motion jumps')
    parser.add_argument('--dirty_start_jump_threshold_m', type=float, default=0.3, help='position jump threshold in meters for dirty-start detection')
    parser.add_argument('--dirty_start_settle_frames', type=int, default=0, help='extra frames to skip after the last detected early dirty jump')
    parser.add_argument('--load_pretrained_path', type=str, help='path to load pretrained model', required=False)
    parser.add_argument('--eval_ckpts', action='store_true', help='evaluate all checkpoints under ckpt_dir and write retro_metrics.csv/png')
    parser.add_argument('--base_dir', type=str, help='base directory for data', required=False)
    parser.add_argument('--processed_dir', type=str, help='additional base directory for processed data', required=False)
    parser.add_argument('--use_future_dino_head', action='store_true', help='enable Future-DINO World Head for auxiliary training')
    parser.add_argument('--future_dino_weight', type=float, default=None, help='weight for Future-DINO loss (overrides config)')
    parser.add_argument('--future_dino_warmup_steps', type=int, default=None, help='warmup steps for Future-DINO loss (overrides config)')
    parser.add_argument('--future_dino_horizon', type=int, default=None, help='future frame offset H in raw frames (overrides config)')
    parser.add_argument('--future_dino_ablation', type=str, default=None, choices=['none', 'shuffled', 'current'], help='Future-DINO ablation mode (overrides config)')
    parser.add_argument('--use_future_vae_head', action='store_true', help='enable the second future-target head with a frozen Wan VAE teacher (ST-WAM style); requires --use_future_dino_head')
    parser.add_argument('--future_vae_weight', type=float, default=None, help='weight for the Future-VAE loss (overrides config)')
    parser.add_argument('--future_vae_ablation', type=str, default=None, choices=['none', 'shuffled', 'current'], help='Future-VAE ablation mode (overrides config)')
    parser.add_argument('--future_vae_normalize_target', type=int, default=None, choices=[0, 1], help='1 = normalize each Wan latent token before the loss; default 0 keeps the latent magnitude')
    # EgoWAM 3D point flow (third world target). No --future_dino_head requirement:
    # the target is precomputed offline, so this head stands alone.
    parser.add_argument('--use_future_flow_head', action='store_true', help="enable EgoWAM's 3D point-flow world head; target comes from --future_flow_dir, not a teacher")
    parser.add_argument('--future_flow_weight', type=float, default=None, help='weight for the Future-Flow loss (overrides config)')
    parser.add_argument('--future_flow_ablation', type=str, default=None, choices=['none', 'shuffled'], help="Future-Flow ablation mode; 'shuffled' is the negative control (no 'current': the current displacement is identically 0)")
    parser.add_argument('--future_flow_dir', type=str, default=None, help='directory of per-episode flow-target h5 files from scripts/preprocess/flow_target.py (overrides config)')
    parser.add_argument('--zero_state_dims', type=str, default=None, help='"lo:hi" half-open range of normalized qpos dims to force to 0, e.g. "100:128" to hide the dex5 robot-configuration block and match the human episodes. Default: keep the whole state.')
    args = vars(parser.parse_args())

    if args.get('base_dir'):
        base_dir = args['base_dir']
    else:
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/recordings/processed')
    assert os.path.exists(base_dir)

    processed_dir = args.get('processed_dir')

    main(args, base_dir, processed_dir)
