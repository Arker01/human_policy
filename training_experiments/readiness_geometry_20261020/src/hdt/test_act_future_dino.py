#!/usr/bin/env python3
"""
Test script for ACT + Future-DINO World Head implementation.
Verifies that:
1. ACT without Future-DINO works (original functionality preserved)
2. ACT with Future-DINO works correctly
3. Both modes produce valid outputs
"""

import os
import sys
import yaml
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from detr.models.detr_vae import (
    DETRVAE,
    FutureDINOHead,
    FutureDINOLoss,
    build as build_vae,
)
from detr.models.backbone import build_backbone
from detr.models.transformer import build_transformer
from detr.models.position_encoding import PositionEmbeddingSine
from detr.models.detr_vae import build_encoder, get_sinusoid_encoding_table


def create_test_args(enable_future_dino=False):
    """Create test arguments for building ACT model."""
    import argparse
    parser = argparse.ArgumentParser('DETR test')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_backbone', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--position_embedding', type=str, default='sine')
    parser.add_argument('--camera_names', type=list, default=['top'])
    parser.add_argument('--enc_layers', type=int, default=4)
    parser.add_argument('--dec_layers', type=int, default=7)
    parser.add_argument('--dim_feedforward', type=int, default=3200)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--nheads', type=int, default=8)
    parser.add_argument('--num_queries', type=int, default=64)
    parser.add_argument('--pre_norm', action='store_true', default=False)
    parser.add_argument('--kl_weight', type=float, default=10)
    parser.add_argument('--chunk_size', type=int, default=64)
    parser.add_argument('--state_dim', type=int, default=128)
    parser.add_argument('--action_dim', type=int, default=128)
    parser.add_argument('--image_feature_strategy', type=str, default='ACT_linear')
    parser.add_argument('--use_language_conditioning', action='store_true', default=False)
    parser.add_argument('--masks', action='store_true', default=False)
    
    args = parser.parse_args([])
    
    # Add Future-DINO config if enabled
    if enable_future_dino:
        args.future_dino_config = {
            'enabled': True,
            'weight': 0.3,
            'num_layers': 2,
            'num_heads': 4,
            'dropout': 0.1,
            'huber_weight': 0.1,
            'warmup_steps': 100,
            # Upper bound on the position table. Patch count now comes from the frozen
            # DINOv2 target encoder (352 at the real 224x308 input), not from the
            # resnet feature map, so 150 no longer fits.
            'num_patches': 1024,
        }
    
    return args


def test_future_dino_head():
    """Test FutureDINOHead module."""
    print("\n" + "=" * 60)
    print("Testing FutureDINOHead...")
    print("=" * 60)

    dino_dim = 512
    hidden_dim = 512
    num_patches = 150

    head = FutureDINOHead(
        dino_dim=dino_dim,
        hidden_dim=hidden_dim,
        num_layers=2,
        num_heads=4,
        num_patches=num_patches,
    )

    # Test forward pass
    current_dino_tokens = torch.randn(2, num_patches, dino_dim)
    hat_memory = torch.randn(2, 100, hidden_dim)

    output = head(current_dino_tokens, hat_memory)

    assert output.shape == (2, num_patches, dino_dim), f"Expected (2, {num_patches}, {dino_dim}), got {output.shape}"
    print(f"  ✓ Forward pass works: input {current_dino_tokens.shape} -> output {output.shape}")

    # Count parameters
    num_params = sum(p.numel() for p in head.parameters())
    print(f"  ✓ Number of parameters: {num_params:,}")

    print("FutureDINOHead tests passed!\n")


def test_future_dino_loss():
    """Test FutureDINOLoss module."""
    print("\n" + "=" * 60)
    print("Testing FutureDINOLoss...")
    print("=" * 60)

    loss_fn = FutureDINOLoss(huber_weight=0.1)

    # Test with simple inputs
    predicted = torch.randn(2, 100, 512)
    target = torch.randn(2, 100, 512)

    loss_dict = loss_fn(predicted, target)

    assert 'cosine_loss' in loss_dict
    assert 'huber_loss' in loss_dict
    assert 'loss' in loss_dict
    assert loss_dict['loss'].item() > 0

    print(f"  ✓ Loss computation works")
    print(f"    Cosine loss: {loss_dict['cosine_loss'].item():.4f}")
    print(f"    Huber loss: {loss_dict['huber_loss'].item():.4f}")
    print(f"    Total loss: {loss_dict['loss'].item():.4f}")

    # Test identical inputs (should give low loss)
    loss_dict_identical = loss_fn(predicted, predicted.clone())
    assert loss_dict_identical['cosine_loss'].item() < 0.01
    print(f"  ✓ Identical input gives low loss: cosine={loss_dict_identical['cosine_loss'].item():.6f}")

    print("FutureDINOLoss tests passed!\n")


def test_act_without_future_dino():
    """Test ACT model without Future-DINO Head (original functionality)."""
    print("\n" + "=" * 60)
    print("Testing ACT without Future-DINO Head...")
    print("=" * 60)

    device = torch.device('cpu')

    # Create args without Future-DINO
    args = create_test_args(enable_future_dino=False)

    # Build backbone
    backbone = build_backbone(args)
    backbones = [backbone]

    # Build transformer
    transformer = build_transformer(args)

    # Build encoder
    encoder = build_encoder(args)

    # Create DETRVAE model
    model = DETRVAE(
        backbones=backbones,
        transformer=transformer,
        encoder=encoder,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names,
        image_feature_strategy=args.image_feature_strategy,
        use_language_conditioning=args.use_language_conditioning,
    )

    model = model.to(device)

    print(f"  Model created (without Future-DINO)")
    print(f"    use_future_dino_head: {model.use_future_dino_head}")
    assert model.use_future_dino_head == False, "Should not have Future-DINO head"

    # Test inference (should work without Future-DINO)
    batch_size = 2
    num_cams = 1
    img_height = 240
    img_width = 320

    image = torch.randn(batch_size, num_cams, 3, img_height, img_width).to(device)
    qpos = torch.randn(batch_size, 128).to(device)

    print("  Testing inference (without Future-DINO)...")
    model.eval()
    with torch.no_grad():
        a_hat, is_pad_hat, (mu, logvar), future_dino_loss_dict = model(qpos, image, None)
    
    assert a_hat.shape == (batch_size, args.num_queries, args.action_dim)
    assert future_dino_loss_dict is None
    print(f"  ✓ Inference works: output shape {a_hat.shape}")

    # Test training (should work without Future-DINO)
    print("  Testing training (without Future-DINO)...")
    model.train()
    actions = torch.randn(batch_size, args.num_queries, args.action_dim).to(device)
    is_pad = torch.zeros(batch_size, args.num_queries, dtype=torch.bool).to(device)

    a_hat, is_pad_hat, (mu, logvar), future_dino_loss_dict = model(
        qpos, image, None, actions, is_pad
    )

    assert a_hat.shape == (batch_size, args.num_queries, args.action_dim)
    assert future_dino_loss_dict is None
    print(f"  ✓ Training works: output shape {a_hat.shape}")
    print(f"    No Future-DINO loss (correct)")

    print("ACT without Future-DINO tests passed!\n")


def test_act_with_future_dino():
    """Test ACT model with Future-DINO Head enabled."""
    print("\n" + "=" * 60)
    print("Testing ACT with Future-DINO Head...")
    print("=" * 60)

    device = torch.device('cpu')

    # Create args with Future-DINO enabled
    args = create_test_args(enable_future_dino=True)
    future_dino_config = args.future_dino_config

    # Build backbone
    backbone = build_backbone(args)
    backbones = [backbone]

    # Build transformer
    transformer = build_transformer(args)

    # Build encoder
    encoder = build_encoder(args)

    # Create DETRVAE model with Future-DINO config
    model = DETRVAE(
        backbones=backbones,
        transformer=transformer,
        encoder=encoder,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names,
        image_feature_strategy=args.image_feature_strategy,
        use_language_conditioning=args.use_language_conditioning,
        future_dino_config=future_dino_config,
    )

    model = model.to(device)

    print(f"  Model created (with Future-DINO)")
    print(f"    use_future_dino_head: {model.use_future_dino_head}")
    print(f"    future_dino_weight: {model.future_dino_weight}")
    assert model.use_future_dino_head == True

    # Test training without future_image (should not compute Future-DINO loss)
    batch_size = 2
    num_cams = 1
    img_height = 240
    img_width = 320

    image = torch.randn(batch_size, num_cams, 3, img_height, img_width).to(device)
    qpos = torch.randn(batch_size, 128).to(device)

    print("  Testing training without future_image (Future-DINO not computed)...")
    model.train()
    actions = torch.randn(batch_size, args.num_queries, args.action_dim).to(device)
    is_pad = torch.zeros(batch_size, args.num_queries, dtype=torch.bool).to(device)

    a_hat, is_pad_hat, (mu, logvar), future_dino_loss_dict = model(
        qpos, image, None, actions, is_pad
    )

    assert a_hat.shape == (batch_size, args.num_queries, args.action_dim)
    assert future_dino_loss_dict is None, "Should not have Future-DINO loss without future_image"
    print(f"  ✓ Training without future_image works: output shape {a_hat.shape}")

    # Test training with future_image (should compute Future-DINO loss)
    print("  Testing training with future_image (Future-DINO computed)...")
    future_image = torch.randn(batch_size, num_cams, 3, img_height, img_width).to(device)

    # Set training step for warmup
    model.set_training_step(50)  # Half of warmup_steps=100

    a_hat, is_pad_hat, (mu, logvar), future_dino_loss_dict = model(
        qpos, image, None, actions, is_pad,
        future_image=future_image
    )

    assert a_hat.shape == (batch_size, args.num_queries, args.action_dim)
    assert future_dino_loss_dict is not None, "Should have Future-DINO loss with future_image"

    print(f"  ✓ Training with future_image works:")
    print(f"    Output shape: {a_hat.shape}")
    print(f"    Future-DINO cosine loss: {future_dino_loss_dict['cosine_loss'].item():.4f}")
    print(f"    Future-DINO huber loss: {future_dino_loss_dict['huber_loss'].item():.4f}")
    print(f"    Future-DINO total loss: {future_dino_loss_dict['loss'].item():.4f}")
    effective_weight = future_dino_loss_dict['effective_weight']
    print(f"    Effective weight: {effective_weight:.4f}")

    # Test that inference still works without future_image
    print("  Testing inference with Future-DINO (should not use it)...")
    model.eval()
    with torch.no_grad():
        a_hat, is_pad_hat, (mu, logvar), future_dino_loss_dict = model(qpos, image, None)
    
    assert a_hat.shape == (batch_size, args.num_queries, args.action_dim)
    assert future_dino_loss_dict is None
    print(f"  ✓ Inference still works: output shape {a_hat.shape}")

    # Test warmup effect
    print("  Testing warmup effect...")
    model.train()

    # No warmup (step=0)
    model.set_training_step(0)
    _, _, _, loss_no_warmup = model(
        qpos, image, None, actions, is_pad,
        future_image=future_image
    )
    weight_no_warmup = loss_no_warmup['effective_weight']

    # Full warmup (step >= warmup_steps)
    model.set_training_step(1000)
    _, _, _, loss_full_warmup = model(
        qpos, image, None, actions, is_pad,
        future_image=future_image
    )
    weight_full_warmup = loss_full_warmup['effective_weight']

    print(f"  ✓ Warmup works:")
    print(f"    No warmup (step=0) effective weight: {weight_no_warmup:.4f}")
    print(f"    Full warmup (step=1000) effective weight: {weight_full_warmup:.4f}")
    assert weight_no_warmup < weight_full_warmup, "Warmup should increase weight"

    print("ACT with Future-DINO tests passed!\n")


def test_config_file():
    """Test the ACT Future-DINO configuration file."""
    print("\n" + "=" * 60)
    print("Testing ACT configuration file...")
    print("=" * 60)

    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "configs", "models", "act_with_future_dino.yaml"
    )

    assert os.path.exists(config_path), f"Config file not found: {config_path}"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Verify Future-DINO config
    assert 'future_dino' in config['model']
    assert config['model']['future_dino']['enabled'] == True
    assert config['common']['policy_class'] == 'ACT'
    print(f"  ✓ Config file exists and is valid")
    print(f"    policy_class: {config['common']['policy_class']}")
    print(f"    Future-DINO enabled: {config['model']['future_dino']['enabled']}")
    print(f"    Weight: {config['model']['future_dino']['weight']}")
    print(f"    Warmup steps: {config['model']['future_dino']['warmup_steps']}")

    print("Configuration file tests passed!\n")


def main():
    """Run all tests."""
    print("\n" + "#" * 60)
    print("# ACT + Future-DINO World Head Implementation Tests")
    print("#" * 60)

    try:
        # Test FutureDINOHead module
        test_future_dino_head()

        # Test FutureDINOLoss module
        test_future_dino_loss()

        # Test configuration file
        test_config_file()

        # Test ACT without Future-DINO (original functionality)
        test_act_without_future_dino()

        # Test ACT with Future-DINO
        test_act_with_future_dino()

        print("\n" + "=" * 60)
        print("ALL ACT + Future-DINO TESTS PASSED!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
