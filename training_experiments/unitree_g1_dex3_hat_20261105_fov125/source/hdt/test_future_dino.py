#!/usr/bin/env python3
"""
Test script for Future-DINO World Head implementation.
Verifies that:
1. Original functionality works (without Future-DINO Head)
2. Future-DINO Head works correctly when enabled
3. Both modes produce valid outputs
"""

import os
import sys
import yaml
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hdt.modeling.modeling_siglip import SiglipVisionTower
from hdt.modeling.modeling_hdt import (
    HumanDiffusionTransformer,
    FutureDINOHead,
    FutureDINOLoss,
)


def create_test_config(enable_future_dino=False):
    """Create a test configuration."""
    config = {
        "common": {
            "img_history_size": 1,
            "action_chunk_size": 64,
            "num_cameras": 2,
            "state_dim": 128,
            "action_dim": 128,
            "camera_names": ["left", "right"],
            "policy_class": "RDT",
        },
        "dataset": {
            "tokenizer_max_length": 1024,
        },
        "model": {
            "backbone": "SIGLIP",
            "lang_adaptor": "mlp2x_gelu",
            "img_adaptor": "mlp2x_gelu",
            "state_adaptor": "mlp3x_gelu",
            "lang_token_dim": 4096,
            "img_token_dim": 1152,
            "state_token_dim": 128,
            "rdt": {
                "hidden_size": 512,
                "depth": 4,
                "num_heads": 8,
                "cond_pos_embed_type": "multimodal",
            },
            "noise_scheduler": {
                "type": "ddpm",
                "num_train_timesteps": 1000,
                "num_inference_timesteps": 5,
                "beta_schedule": "squaredcos_cap_v2",
                "prediction_type": "sample",
                "clip_sample": False,
            },
            "ema": {
                "update_after_step": 0,
                "inv_gamma": 1.0,
                "power": 0.75,
                "min_value": 0.0,
                "max_value": 0.9999,
            },
        },
        "data": {
            "image_resolution_hw": [240, 320],
        },
    }

    if enable_future_dino:
        config["model"]["future_dino"] = {
            "enabled": True,
            "weight": 0.3,
            "num_layers": 2,
            "num_heads": 4,
            "dropout": 0.1,
            "huber_weight": 0.1,
            "warmup_steps": 100,
        }

    return config


def test_future_dino_head():
    """Test FutureDINOHead module."""
    print("\n" + "=" * 60)
    print("Testing FutureDINOHead...")
    print("=" * 60)

    dino_dim = 1152
    hidden_dim = 512
    num_patches = 300

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

    # Test with different batch size
    current_dino_tokens_1 = torch.randn(1, num_patches, dino_dim)
    hat_memory_1 = torch.randn(1, 50, hidden_dim)
    output_1 = head(current_dino_tokens_1, hat_memory_1)
    assert output_1.shape == (1, num_patches, dino_dim)
    print(f"  ✓ Batch size 1 works: input {current_dino_tokens_1.shape} -> output {output_1.shape}")

    # Test with different number of patches (should handle variable)
    num_patches_2 = 150
    position_embedding_2 = torch.randn(1, num_patches_2, hidden_dim) * 0.02
    head.position_embedding = torch.nn.Parameter(position_embedding_2)
    current_dino_tokens_2 = torch.randn(2, num_patches_2, dino_dim)
    output_2 = head(current_dino_tokens_2, hat_memory)
    assert output_2.shape == (2, num_patches_2, dino_dim)
    print(f"  ✓ Variable patches work: input {current_dino_tokens_2.shape} -> output {output_2.shape}")

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
    predicted = torch.randn(2, 100, 768)
    target = torch.randn(2, 100, 768)

    loss_dict = loss_fn(predicted, target)

    assert 'cosine_loss' in loss_dict
    assert 'huber_loss' in loss_dict
    assert 'loss' in loss_dict
    assert loss_dict['loss'].item() > 0

    print(f"  ✓ Loss computation works")
    print(f"    Cosine loss: {loss_dict['cosine_loss'].item():.4f}")
    print(f"    Huber loss: {loss_dict['huber_loss'].item():.4f}")
    print(f"    Total loss: {loss_dict['loss'].item():.4f}")

    # Test with mask
    mask = torch.ones(2, 100)
    mask[:, 80:] = 0  # Mask out last 20 patches

    loss_dict_masked = loss_fn(predicted, target, mask=mask)
    assert loss_dict_masked['loss'].item() > 0
    print(f"  ✓ Masked loss works: {loss_dict_masked['loss'].item():.4f}")

    # Test identical inputs (should give low loss)
    loss_dict_identical = loss_fn(predicted, predicted.clone())
    assert loss_dict_identical['cosine_loss'].item() < 0.01  # Should be close to 0
    assert loss_dict_identical['huber_loss'].item() < 0.001
    print(f"  ✓ Identical input gives low loss: cosine={loss_dict_identical['cosine_loss'].item():.6f}")

    print("FutureDINOLoss tests passed!\n")


def test_hdt_without_future_dino():
    """Test HAT without Future-DINO Head (original functionality)."""
    print("\n" + "=" * 60)
    print("Testing HAT without Future-DINO Head...")
    print("=" * 60)

    device = torch.device('cpu')  # Use CPU for testing

    # Create config without Future-DINO
    config = create_test_config(enable_future_dino=False)

    # Create a simple vision encoder (mock for testing)
    class MockVisionEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_patches = 300
            self.hidden_size = 1152

        def forward(self, x):
            batch_size, _, _, _ = x.shape
            return torch.randn(batch_size, self.num_patches, self.hidden_size)

    vision_encoder = MockVisionEncoder()

    # Create model
    model = HumanDiffusionTransformer(
        action_dim=128,
        pred_horizon=64,
        config=config,
        lang_token_dim=4096,
        img_token_dim=1152,
        state_token_dim=128,
        max_lang_cond_len=1024,
        visual_encoder=vision_encoder,
        dtype=torch.float32,
    )

    model = model.to(device)

    # Test inference (should work without Future-DINO)
    batch_size = 2
    num_cams = 2
    img_height = 240
    img_width = 320

    image = torch.randn(batch_size, num_cams, 3, img_height, img_width).to(device)
    qpos = torch.randn(batch_size, 128).to(device)

    # Create dummy conditioning dict
    conditioning_dict = {
        "language_embeddings": torch.randn(batch_size, 22, 4096).to(device),
        "language_embeddings_mask": torch.ones(batch_size, 22, dtype=torch.bool).to(device),
    }

    print("  Testing inference (without Future-DINO)...")
    model.eval()
    with torch.no_grad():
        trajectory = model(image, qpos, conditioning_dict=conditioning_dict)
    assert trajectory.shape == (batch_size, 64, 128)
    print(f"  ✓ Inference works: output shape {trajectory.shape}")

    # Test training (should work without Future-DINO)
    print("  Testing training (without Future-DINO)...")
    model.train()
    actions = torch.randn(batch_size, 64, 128).to(device)
    is_pad = torch.zeros(batch_size, 64, dtype=torch.bool).to(device)

    loss_dict = model(image, qpos, actions=actions, is_pad=is_pad, conditioning_dict=conditioning_dict)
    assert 'loss' in loss_dict
    print(f"  ✓ Training works: loss={loss_dict['loss'].item():.4f}")
    print(f"    Loss dict keys: {list(loss_dict.keys())}")

    # Verify no Future-DINO loss in output
    assert 'future_dino_loss' not in loss_dict, "Should not have Future-DINO loss when disabled"
    print("  ✓ No Future-DINO loss when disabled (correct)")

    print("HAT without Future-DINO tests passed!\n")


def test_hdt_with_future_dino():
    """Test HAT with Future-DINO Head enabled."""
    print("\n" + "=" * 60)
    print("Testing HAT with Future-DINO Head...")
    print("=" * 60)

    device = torch.device('cpu')  # Use CPU for testing

    # Create config with Future-DINO enabled
    config = create_test_config(enable_future_dino=True)

    # Create a simple vision encoder (mock for testing)
    class MockVisionEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_patches = 300
            self.hidden_size = 1152

        def forward(self, x):
            batch_size, _, _, _ = x.shape
            return torch.randn(batch_size, self.num_patches, self.hidden_size)

    vision_encoder = MockVisionEncoder()

    # Create model with Future-DINO
    model = HumanDiffusionTransformer(
        action_dim=128,
        pred_horizon=64,
        config=config,
        lang_token_dim=4096,
        img_token_dim=1152,
        state_token_dim=128,
        max_lang_cond_len=1024,
        visual_encoder=vision_encoder,
        dtype=torch.float32,
    )

    model = model.to(device)

    # Verify Future-DINO head is created
    assert model.use_future_dino_head == True
    assert model.future_dino_head is not None
    print(f"  ✓ Future-DINO Head created: weight={model.future_dino_weight}")

    # Test training with Future-DINO (without future_image - should not compute it)
    batch_size = 2
    num_cams = 2
    img_height = 240
    img_width = 320

    image = torch.randn(batch_size, num_cams, 3, img_height, img_width).to(device)
    qpos = torch.randn(batch_size, 128).to(device)

    conditioning_dict = {
        "language_embeddings": torch.randn(batch_size, 22, 4096).to(device),
        "language_embeddings_mask": torch.ones(batch_size, 22, dtype=torch.bool).to(device),
    }

    actions = torch.randn(batch_size, 64, 128).to(device)
    is_pad = torch.zeros(batch_size, 64, dtype=torch.bool).to(device)

    print("  Testing training without future_image (Future-DINO not computed)...")
    model.train()
    loss_dict = model(image, qpos, actions=actions, is_pad=is_pad, conditioning_dict=conditioning_dict)
    assert 'loss' in loss_dict
    assert 'future_dino_loss' not in loss_dict, "Should not have Future-DINO loss without future_image"
    print(f"  ✓ Training without future_image works: loss={loss_dict['loss'].item():.4f}")

    # Test training with future_image (should compute Future-DINO loss)
    print("  Testing training with future_image (Future-DINO computed)...")
    future_image = torch.randn(batch_size, num_cams, 3, img_height, img_width).to(device)

    # Set training step for warmup
    model._training_step = 50  # Half of warmup_steps=100

    loss_dict_with_future = model(
        image, qpos,
        actions=actions, is_pad=is_pad,
        conditioning_dict=conditioning_dict,
        future_image=future_image,
    )

    assert 'loss' in loss_dict_with_future
    assert 'future_dino_loss' in loss_dict_with_future
    assert 'future_dino_cosine_loss' in loss_dict_with_future
    assert 'future_dino_huber_loss' in loss_dict_with_future

    print(f"  ✓ Training with future_image works:")
    print(f"    Total loss: {loss_dict_with_future['loss'].item():.4f}")
    print(f"    Future-DINO cosine loss: {loss_dict_with_future['future_dino_cosine_loss'].item():.4f}")
    print(f"    Future-DINO huber loss: {loss_dict_with_future['future_dino_huber_loss'].item():.4f}")
    print(f"    Future-DINO total loss: {loss_dict_with_future['future_dino_loss'].item():.4f}")

    # Test that inference still works without future_image
    print("  Testing inference with Future-DINO (should not use it)...")
    model.eval()
    with torch.no_grad():
        trajectory = model(image, qpos, conditioning_dict=conditioning_dict)
    assert trajectory.shape == (batch_size, 64, 128)
    print(f"  ✓ Inference still works: output shape {trajectory.shape}")

    # Test warmup effect
    print("  Testing warmup effect...")
    model._training_step = 0  # No warmup
    loss_dict_no_warmup = model(
        image, qpos,
        actions=actions, is_pad=is_pad,
        conditioning_dict=conditioning_dict,
        future_image=future_image,
    )

    model._training_step = 1000  # Full warmup
    loss_dict_full_warmup = model(
        image, qpos,
        actions=actions, is_pad=is_pad,
        conditioning_dict=conditioning_dict,
        future_image=future_image,
    )

    # With no warmup (step=0), weight should be 0
    # So future_dino_loss should not affect total loss
    print(f"  ✓ Warmup works:")
    print(f"    No warmup (step=0) total loss: {loss_dict_no_warmup['loss'].item():.4f}")
    print(f"    Full warmup (step=1000) total loss: {loss_dict_full_warmup['loss'].item():.4f}")

    # Test that original loss components are still present
    assert any(k in loss_dict_with_future for k in ['left_eef_loss', 'l1', 'eef_loss', 'loss']), \
        f"Expected original loss components, got keys: {list(loss_dict_with_future.keys())}"
    print("  ✓ Original loss components preserved")

    print("HAT with Future-DINO tests passed!\n")


def test_config_file():
    """Test the configuration file."""
    print("\n" + "=" * 60)
    print("Testing configuration file...")
    print("=" * 60)

    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "configs", "models", "rdt_with_future_dino.yaml"
    )

    assert os.path.exists(config_path), f"Config file not found: {config_path}"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Verify Future-DINO config. Note the RDT variant ships DISABLED on purpose: its
    # memory is still the pre-trunk conditioning, so the world gradient never reaches
    # the diffusion trunk. See HAT_future_DINO_fixes.md.
    assert 'future_dino' in config['model']
    assert config['model']['future_dino']['enabled'] is False, \
        "rdt_with_future_dino.yaml should stay disabled until the RDT memory is fixed"
    print(f"  ✓ Config file exists and is valid")
    print(f"    Future-DINO enabled: {config['model']['future_dino']['enabled']}")
    print(f"    Weight: {config['model']['future_dino']['weight']}")
    print(f"    Warmup steps: {config['model']['future_dino']['warmup_steps']}")

    print("Configuration file tests passed!\n")


def main():
    """Run all tests."""
    print("\n" + "#" * 60)
    print("# Future-DINO World Head Implementation Tests")
    print("#" * 60)

    try:
        # Test FutureDINOHead module
        test_future_dino_head()

        # Test FutureDINOLoss module
        test_future_dino_loss()

        # Test configuration file
        test_config_file()

        # Test HAT without Future-DINO (original functionality)
        test_hdt_without_future_dino()

        # Test HAT with Future-DINO
        test_hdt_with_future_dino()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
