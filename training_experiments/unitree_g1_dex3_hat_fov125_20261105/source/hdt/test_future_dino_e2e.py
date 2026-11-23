"""
End-to-end regression test for the Future-DINO world head (ACT path).

The pre-existing tests (test_act_future_dino.py / test_future_dino.py) hand-build a
`future_image` and call the model directly. They passed while the feature was in fact
completely dead in training, because nothing they touch is on the real training path.
This test covers the three things that were actually broken:

  1. the dataloader emits `future_image` and `future_valid` and `collate_fn` batches
     them into the 6-tuple that `policy(*data)` expects,
  2. the world gradient reaches the SHARED TRANSFORMER TRUNK, not just the backbone,
  3. `set_training_step()` actually reaches the model through `forward_pass()`, so the
     warmup weight becomes non-zero instead of pinning the loss contribution at 0.

Run:
    /home/aigc/miniconda/envs/human_policy/bin/python hdt/test_future_dino_e2e.py
"""
import os
import sys

import torch

# hdt/ only -- do NOT add hdt/detr, its main.py would shadow hdt/main.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FUTURE_DINO_CONFIG = {
    'enabled': True,
    'weight': 0.3,
    'target_encoder': 'dinov2_vits14',
    'horizon': 16,
    'num_layers': 2,
    'num_heads': 8,
    'num_patches': 1024,
    'dropout': 0.1,
    'huber_weight': 0.1,
    'warmup_steps': 100,
    'normalize_target': True,
    'ablation': 'none',
}


def build_policy(future_dino_config=FUTURE_DINO_CONFIG):
    from policy import ACTPolicy
    return ACTPolicy({
        'lr': 1e-5,
        'num_queries': 8,
        'kl_weight': 10,
        'hidden_dim': 256,
        'chunk_size': 8,
        'dim_feedforward': 512,
        'lr_backbone': 1e-5,
        'backbone': 'resnet18',
        'enc_layers': 2,
        'dec_layers': 2,
        'nheads': 8,
        'camera_names': ['top'],
        'state_dim': 128,
        'action_dim': 128,
        'image_feature_strategy': 'ACT_linear',
        'use_language_conditioning': False,
        'future_dino_config': future_dino_config,
    })


def fake_batch(bs=2, device='cpu'):
    image = torch.rand(bs, 1, 3, 240, 320, device=device)
    future_image = torch.rand(bs, 1, 3, 240, 320, device=device)
    qpos = torch.randn(bs, 128, device=device)
    actions = torch.randn(bs, 8, 128, device=device)
    is_pad = torch.zeros(bs, 8, dtype=torch.bool, device=device)
    cond = {'future_valid': torch.ones(bs, device=device)}
    return image, qpos, actions, is_pad, cond, future_image


def test_collate_returns_future_tuple():
    from data_utils_hdt import collate_fn
    bs = 3
    batch = [(
        torch.rand(1, 3, 240, 320),          # image
        torch.randn(128),                    # qpos
        torch.randn(8, 128),                 # action
        torch.zeros(8, dtype=torch.bool),    # is_pad
        {'language_embeddings': None, 'plain_text': '', 'future_valid': torch.tensor(1.0 if i else 0.0)},
        torch.rand(1, 3, 240, 320),          # future image
    ) for i in range(bs)]
    out = collate_fn(batch)
    assert len(out) == 6, f"collate_fn dropped the future image: got {len(out)} elements"
    assert out[5].shape == (bs, 1, 3, 240, 320), out[5].shape
    assert out[4]['future_valid'].tolist() == [0.0, 1.0, 1.0], out[4]['future_valid']

    # the original 5-tuple path must be untouched
    legacy = [tuple(item[:5]) for item in batch]
    assert len(collate_fn(legacy)) == 5
    print("PASS  collate_fn batches future_image + future_valid, 5-tuple path unchanged")


def test_world_gradient_reaches_trunk():
    """
    The original code used the PRE-trunk `src` as decoder memory, so backwarding the
    world loss alone produced exactly zero gradient on self.transformer. This asserts
    the fix.
    """
    policy = build_policy()
    device = next(policy.parameters()).device
    image, qpos, actions, is_pad, cond, future_image = fake_batch(device=device)

    policy.model.set_training_step(10 ** 6)
    loss_dict = policy(image, qpos, actions, is_pad, cond, future_image)
    assert 'future_dino_loss' in loss_dict, "world loss missing from loss_dict"

    policy.zero_grad(set_to_none=True)
    loss_dict['future_dino_loss'].backward()

    buckets = {}
    for name, p in policy.model.named_parameters():
        if p.grad is None:
            continue
        buckets[name.split('.')[0]] = buckets.get(name.split('.')[0], 0.0) + p.grad.abs().sum().item()

    print("      world-loss grad by module:", {k: f"{v:.3e}" for k, v in sorted(buckets.items())})
    assert buckets.get('transformer', 0.0) > 0, (
        "SHARED TRUNK GOT NO WORLD GRADIENT -- memory is still pre-trunk (doc S3 violated)")
    assert buckets.get('future_dino_head', 0.0) > 0, "world head got no gradient"
    assert buckets.get('backbones', 0.0) > 0, "trunk backbone got no gradient"
    print("PASS  world loss reaches the shared transformer trunk")


def test_target_encoder_is_frozen_and_off_ledger():
    policy = build_policy()
    enc = policy.model._future_dino_target_encoder[0]
    assert all(not p.requires_grad for p in enc.parameters()), "target encoder is trainable"

    # It must not appear in state_dict (checkpoint size / strict-load compatibility)
    # nor in the optimizer, or it is de facto part of the trained model.
    assert not any('target_encoder' in k for k in policy.state_dict()), \
        "frozen teacher leaked into state_dict"
    opt_params = {id(p) for g in policy.optimizer.param_groups for p in g['params']}
    assert not any(id(p) in opt_params for p in enc.parameters()), \
        "frozen teacher leaked into the optimizer"
    print("PASS  target encoder is frozen and stays out of state_dict/optimizer")


def test_warmup_reaches_model_through_forward_pass():
    """`forward_pass` used `policy.model`, which under DDP is the wrapped module, so
    the step never landed and effective_weight stayed 0 forever."""
    from main import forward_pass

    policy = build_policy()
    device = next(policy.parameters()).device
    data = list(fake_batch(device=device))

    out0 = forward_pass(list(data), policy, training_step=0)
    assert float(out0['future_dino_effective_weight']) == 0.0

    out_mid = forward_pass(list(data), policy, training_step=50)
    assert abs(float(out_mid['future_dino_effective_weight']) - 0.15) < 1e-6, \
        float(out_mid['future_dino_effective_weight'])

    out_full = forward_pass(list(data), policy, training_step=10 ** 6)
    assert abs(float(out_full['future_dino_effective_weight']) - 0.3) < 1e-6, \
        float(out_full['future_dino_effective_weight'])

    # and through a DDP-like wrapper, which is what actually broke
    class FakeDDP(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.module = m

        def forward(self, *a, **kw):
            return self.module(*a, **kw)

    policy.model.set_training_step(0)
    wrapped = FakeDDP(policy)
    out_ddp = forward_pass(list(data), wrapped, training_step=10 ** 6)
    assert abs(float(out_ddp['future_dino_effective_weight']) - 0.3) < 1e-6, \
        f"warmup step did not survive DDP wrapping: {float(out_ddp['future_dino_effective_weight'])}"
    print("PASS  warmup step propagates through forward_pass, incl. DDP wrapping")


def test_invalid_future_is_masked_out():
    policy = build_policy()
    device = next(policy.parameters()).device
    image, qpos, actions, is_pad, _, future_image = fake_batch(device=device)
    policy.model.set_training_step(10 ** 6)

    all_invalid = policy(image, qpos, actions, is_pad,
                         {'future_valid': torch.zeros(image.shape[0], device=device)}, future_image)
    assert float(all_invalid['future_dino_loss']) == 0.0, \
        f"clamped-past-end samples still contribute: {float(all_invalid['future_dino_loss'])}"
    print("PASS  samples whose t+H ran past the episode end are masked out")


def test_disabled_path_is_unchanged():
    policy = build_policy(future_dino_config=None)
    assert not policy.use_future_dino
    device = next(policy.parameters()).device
    image, qpos, actions, is_pad, _, _ = fake_batch(device=device)
    loss_dict = policy(image, qpos, actions, is_pad, {})
    assert not any(k.startswith('future_dino') for k in loss_dict), loss_dict.keys()
    print("PASS  feature-disabled path produces the original loss dict")


if __name__ == '__main__':
    torch.manual_seed(0)
    test_collate_returns_future_tuple()
    test_disabled_path_is_unchanged()
    test_target_encoder_is_frozen_and_off_ledger()
    test_world_gradient_reaches_trunk()
    test_warmup_reaches_model_through_forward_pass()
    test_invalid_future_is_masked_out()
    print("\nAll Future-DINO end-to-end checks passed.")
