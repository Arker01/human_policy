import re
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

from accelerate import Accelerator
from data_utils_hdt import load_data # data functions
from data_utils_hdt import compute_dict_mean, set_seed, detach_dict # helper functions
from modeling.utils import make_visual_encoder

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

def main(args, base_dir):
    set_seed(1)
    with open(args["model_cfg_path"], "r") as fp:
        trainer_config = yaml.safe_load(fp)
    
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
                         }
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
    if accelerator.is_main_process:
        wandb.init(project="human2robot", name=args['exptid'], group="RogerQiu",
                   entity="RogerQiu", mode=mode, dir="../data/logs",
                   id=args['exptid'], resume="allow")
        # os.makedirs("./wandb", exist_ok=True)
        # wandb.init(project="cross_embodiment", name=args['exptid'], mode=mode, dir="./wandb")
        # wandb.config.update(config)
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
                                                        args['human_slow_down_factor'])

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
def forward_pass(data, policy):
    device = next(policy.parameters()).device
    if isinstance(data, dict):
        for k, v in data.items():
            data[k] = maybe_to_tensor(v, device)
    elif isinstance(data, list):
        for i in range(len(data)):
            data[i] = maybe_to_tensor(data[i], device)
    
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

    set_seed(seed)

    min_val_loss = np.inf

    if config['load_pretrained_path'] is not None:
        # TODO(roger): currently it does not respect --lr
        print(f"Loading pretrained model from {config['load_pretrained_path']}")
        state_dict = torch.load(config['load_pretrained_path'], map_location=next(policy.parameters()).device, weights_only=True)
        policy.load_state_dict(state_dict, strict=False)
        # Create custom scheduler
        scheduler = WarmupMultiplicativeLR(
            optimizer,
            initial_lr=1e-7,
            max_lr=1e-4,
            warmup_period=1000
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
        accelerator.load_state(os.path.join(ckpt_dir, max_ckpt_name))

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
                        forward_dict = forward_pass(data, policy)
                        validation_dicts.append(forward_dict)
                        if batch_idx > 20:
                            break

                    validation_summary = compute_dict_mean(validation_dicts)
                    
                    epoch_val_loss = validation_summary['loss']
                if accelerator.is_main_process:
                    print(f'\n Iter {cur_iter}')
                    for k in list(validation_summary.keys()):
                        validation_summary[f'val/{k}'] = validation_summary.pop(k)     

                    wandb.log(validation_summary, step=cur_iter)
                    print(f'Val loss:   {epoch_val_loss:.5f}')
                    summary_string = ''
                    for k, v in validation_summary.items():
                        summary_string += f'{k}: {v.item():.3f} '
                    print(summary_string)

                    if config['val_and_jit_trace']:
                        break

                policy.train()

            forward_dict = policy(*(data))
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
                wandb.log(epoch_summary, step=cur_iter)

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

            def forward(self, image, qpos, language_conditioning):
                cond_dict = {'language_embeddings': language_conditioning}
                return self.policy(image, qpos, conditioning_dict=cond_dict)

        # TRACING_DEVICE = 'cuda'
        TRACING_DEVICE = 'cuda'
        # Rollout validation
        my_policy_wrapper = polciy_wrapper(policy)
        my_policy_wrapper.eval().to(TRACING_DEVICE)

        # Benchmark
        image, qpos, _, _, conditioning_dict = data
        image = image.to(TRACING_DEVICE)
        qpos = qpos.to(TRACING_DEVICE)
        conditioning_dict = {k: v.to(TRACING_DEVICE) for k, v in conditioning_dict.items()}
        # warm up
        for _ in range(10):
            trajectory = my_policy_wrapper(image[0:1], qpos[0:1], conditioning_dict['language_embeddings'][0:1])
        # benchmark speed
        start = time.time()
        N_iters = 50
        for _ in range(N_iters):
            trajectory = my_policy_wrapper(image[0:1], qpos[0:1], conditioning_dict['language_embeddings'][0:1])
        end = time.time()
        print("Total time: ", end - start)
        print(f"Rollout speed: {N_iters / (end - start)} Hz")

        # Jit trace
        image_data = torch.rand(image[0:1].shape, device=TRACING_DEVICE)
        qpos_data = torch.rand(qpos[0:1].shape, device=TRACING_DEVICE)
        language_conditioning = torch.rand(conditioning_dict['language_embeddings'][0:1].shape, device=TRACING_DEVICE)
        input_data = (image_data, qpos_data, language_conditioning)

        traced_policy = torch.jit.trace(my_policy_wrapper, input_data)

        traced_path = os.path.join(ckpt_dir, f'policy_traced.pt')
        traced_policy.save(traced_path)
        del traced_policy
        torch.cuda.empty_cache()

        loaded_policy = torch.jit.load(traced_path)
        # Manually set seed to make diffusion sampling deterministic in comparisons
        torch.random.manual_seed(0)
        jit_output = loaded_policy(image_data, qpos_data, language_conditioning)
        torch.random.manual_seed(0)
        vanilla_output = my_policy_wrapper(image_data, qpos_data, language_conditioning)

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
    parser.add_argument('--dataset_json_path', type=str, help='dataset_json_path', required=True)
    parser.add_argument('--model_cfg_path', type=str, help='path to model cfg yaml', required=True)
    parser.add_argument('--human_slow_down_factor', type=int, default=4, help='human demonstrations slow_down_factor', required=False)
    parser.add_argument('--load_pretrained_path', type=str, help='path to load pretrained model', required=False)
    args = vars(parser.parse_args())

    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/recordings/processed')
    assert os.path.exists(base_dir)

    main(args, base_dir)
