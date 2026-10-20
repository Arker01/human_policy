import re
import os
from pathlib import Path

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import \
    DPMSolverMultistepScheduler

from hdt.modeling.rdt_trunk import RDT
from PIL import Image

import hdt.constants

CTRL_FREQS = 30


class FutureDINOHead(nn.Module):
    """
    Future-DINO Head for predicting future visual latent features.
    Uses Transformer Decoder to predict future DINO patch tokens from
    current DINO tokens (as queries) and HAT shared features (as memory).
    """
    def __init__(
        self,
        dino_dim: int = 1152,
        hidden_dim: int = 2048,
        num_layers: int = 4,
        num_heads: int = 8,
        num_patches: int = 300,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dino_dim = dino_dim
        self.hidden_dim = hidden_dim
        self.num_patches = num_patches

        # Project current DINO tokens to hidden_dim
        self.query_proj = nn.Linear(dino_dim, hidden_dim)

        # Position embedding for patches
        self.position_embedding = nn.Parameter(
            torch.randn(1, num_patches, hidden_dim) * 0.02
        )

        # Horizon embedding (single horizon for now)
        self.horizon_embedding = nn.Parameter(
            torch.randn(1, 1, hidden_dim) * 0.02
        )

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            norm_first=True,
            dropout=dropout,
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )

        # Output projection back to DINO dimension
        self.output_proj = nn.Linear(hidden_dim, dino_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.normal_(self.horizon_embedding, std=0.02)

    def forward(
        self,
        current_dino_tokens: torch.Tensor,
        hat_memory: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            current_dino_tokens: [B, P, D_dino] - Current frame DINO patch features
            hat_memory: [B, N, D_hidden] - HAT shared trunk features (from RDT blocks)

        Returns:
            pred_future_dino: [B, P, D_dino] - Predicted future DINO patch features
        """
        # Project queries
        queries = self.query_proj(current_dino_tokens)  # [B, P, hidden_dim]

        # Add position and horizon embeddings
        queries = queries + self.position_embedding[:, :queries.shape[1], :]
        queries = queries + self.horizon_embedding

        # Transformer Decoder
        # queries: [B, P, hidden_dim]
        # hat_memory: [B, N, hidden_dim]
        future_tokens = self.decoder(
            tgt=queries,
            memory=hat_memory,
        )

        # Project back to DINO dimension
        return self.output_proj(future_tokens)


class FutureDINOLoss(nn.Module):
    """
    Loss for Future-DINO prediction: cosine similarity + Smooth L1 loss.
    """
    def __init__(self, huber_weight: float = 0.1):
        super().__init__()
        self.huber_weight = huber_weight

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            predicted: [B, P, D] - Predicted future DINO tokens
            target: [B, P, D] - Ground truth future DINO tokens
            mask: [B, P] or None - Optional mask for valid patches

        Returns:
            Dictionary with 'cosine_loss', 'huber_loss', and 'loss'
        """
        # Cosine loss
        predicted_norm = F.normalize(predicted, dim=-1)
        target_norm = F.normalize(target.detach(), dim=-1)

        if mask is not None:
            # Masked cosine loss
            mask_expanded = mask.unsqueeze(-1)  # [B, P, 1]
            cosine_sim = (predicted_norm * target_norm).sum(dim=-1)  # [B, P]
            cosine_loss = (1.0 - cosine_sim) * mask.float()
            cosine_loss = cosine_loss.sum() / mask.float().sum().clamp(min=1.0)
        else:
            cosine_loss = 1.0 - (predicted_norm * target_norm).sum(dim=-1).mean()

        # Huber (Smooth L1) loss
        if mask is not None:
            huber_loss = F.smooth_l1_loss(predicted, target.detach(), reduction='none')
            huber_loss = (huber_loss * mask_expanded).sum() / mask.float().sum().clamp(min=1.0)
        else:
            huber_loss = F.smooth_l1_loss(predicted, target.detach())

        loss = cosine_loss + self.huber_weight * huber_loss

        return {
            'cosine_loss': cosine_loss,
            'huber_loss': huber_loss,
            'loss': loss,
        }


class HumanDiffusionTransformer(nn.Module):
    # Adapted from RDT https://github.com/thu-ml/RoboticsDiffusionTransformer/blob/main/models/rdt/model.py
    def __init__(self, *, action_dim, pred_horizon, config, 
                 lang_token_dim, img_token_dim, state_token_dim, 
                 max_lang_cond_len, visual_encoder, lang_pos_embed_config=None, 
                 dtype=torch.bfloat16,
                 ):
        super(HumanDiffusionTransformer, self).__init__()
        assert state_token_dim == action_dim
        self.config = config

        self.vision_encoder = visual_encoder.eval()

        img_cond_len = (config["common"]["img_history_size"] 
                        * config["common"]["num_cameras"] 
                        * self.vision_encoder.num_patches)
        
        img_pos_embed_config=[
            # No initial pos embed in the last grid size
            # since we've already done in ViT
            ("image", (config["common"]["img_history_size"], 
                config["common"]["num_cameras"], 
                -self.vision_encoder.num_patches)),  
        ]

        # Create diffusion model
        hidden_size = config['model']['rdt']['hidden_size']
        self.model = RDT(
            output_dim=action_dim,
            horizon=pred_horizon,
            hidden_size=hidden_size,
            depth=config['model']['rdt']['depth'],
            num_heads=config['model']['rdt']['num_heads'],
            max_lang_cond_len=max_lang_cond_len,
            img_cond_len=img_cond_len,
            lang_pos_embed_config=lang_pos_embed_config,
            img_pos_embed_config=img_pos_embed_config,
            dtype=dtype,
        )

        # Create adpators for various conditional inputs
        self.lang_adaptor = self.build_condition_adapter(
            config['model']['lang_adaptor'], 
            in_features=lang_token_dim, 
            out_features=hidden_size
        )
        self.img_adaptor = self.build_condition_adapter(
            config['model']['img_adaptor'], 
            in_features=img_token_dim, 
            out_features=hidden_size
        )
        # A `state` refers to an action or a proprioception vector
        self.state_adaptor = self.build_condition_adapter(
            config['model']['state_adaptor'], 
            in_features=state_token_dim * 2,    # state + state mask (indicator)
            out_features=hidden_size
        )
        
        # Create the noise scheduler
        noise_scheduler_config = config['model']['noise_scheduler']
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=noise_scheduler_config['num_train_timesteps'],
            beta_schedule=noise_scheduler_config['beta_schedule'],
            prediction_type=noise_scheduler_config['prediction_type'],
            clip_sample=noise_scheduler_config['clip_sample'],
        )
        self.noise_scheduler_sample = DPMSolverMultistepScheduler(
            num_train_timesteps=noise_scheduler_config['num_train_timesteps'],
            beta_schedule=noise_scheduler_config['beta_schedule'],
            prediction_type=noise_scheduler_config['prediction_type'],
        )

        self.num_train_timesteps = noise_scheduler_config['num_train_timesteps']
        self.num_inference_timesteps = noise_scheduler_config['num_inference_timesteps']
        self.prediction_type = noise_scheduler_config['prediction_type']

        self.pred_horizon = pred_horizon
        self.action_dim = action_dim

        # Future-DINO Head configuration (optional)
        # NOTE: the RDT/HDT variant of this head is NOT validated -- all real runs use
        # the ACT path. Its memory is still the pre-trunk conditioning rather than the
        # RDT block output, so the world gradient does not reach the diffusion trunk.
        # See HAT_future_DINO_fixes.md before enabling it.
        self._training_step = 0  # must exist up front: forward_pass() sets it via
                                 # set_training_step(), and without it the warmup
                                 # factor stayed 0 for the whole run.
        self.use_future_dino_head = config.get('model', {}).get('future_dino', {}).get('enabled', False)
        if self.use_future_dino_head:
            future_dino_config = config['model']['future_dino']
            num_patches = self.vision_encoder.num_patches * config['common'].get('num_cameras', 1)
            self.future_dino_head = FutureDINOHead(
                dino_dim=img_token_dim,  # Use same dimension as image tokens
                hidden_dim=hidden_size,
                num_layers=future_dino_config.get('num_layers', 4),
                num_heads=future_dino_config.get('num_heads', 8),
                num_patches=num_patches,
                dropout=future_dino_config.get('dropout', 0.1),
            )
            self.future_dino_loss_fn = FutureDINOLoss(
                huber_weight=future_dino_config.get('huber_weight', 0.1)
            )
            self.future_dino_weight = future_dino_config.get('weight', 0.3)
            self.future_dino_warmup_steps = future_dino_config.get('warmup_steps', 1000)
            print(f"Future-DINO Head enabled: weight={self.future_dino_weight}, warmup={self.future_dino_warmup_steps} steps")
        else:
            self.future_dino_head = None
            self.future_dino_loss_fn = None
            self.future_dino_weight = 0.0

        print("Diffusion params: %e" % sum(
            [p.numel() for p in self.model.parameters()] + 
            [p.numel() for p in self.lang_adaptor.parameters()] + 
            [p.numel() for p in self.img_adaptor.parameters()] + 
            [p.numel() for p in self.state_adaptor.parameters()]))
    
    def build_condition_adapter(
        self, projector_type, in_features, out_features):
        projector = None
        if projector_type == 'linear':
            projector = nn.Linear(in_features, out_features)
        else:
            mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
            if mlp_gelu_match:
                mlp_depth = int(mlp_gelu_match.group(1))
                modules = [nn.Linear(in_features, out_features)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU(approximate="tanh"))
                    modules.append(nn.Linear(out_features, out_features))
                projector = nn.Sequential(*modules)

        if projector is None:
            raise ValueError(f'Unknown projector type: {projector_type}')

        return projector
    
    def adapt_conditions(self, lang_tokens, img_tokens, state_tokens):
        '''
        lang_tokens: (batch_size, lang_len, lang_token_dim)
        img_tokens: (batch_size, img_len, img_token_dim)
        state_tokens: (batch_size, state_len, state_token_dim)
        
        return: adpated (..., hidden_size) for all input tokens
        '''
        adpated_lang = self.lang_adaptor(lang_tokens)
        adpated_img = self.img_adaptor(img_tokens)
        adpated_state = self.state_adaptor(state_tokens)

        return adpated_lang, adpated_img, adpated_state

    def conditional_sample(self, lang_cond, lang_attn_mask, img_cond, 
                           state_traj, action_mask, ctrl_freqs):
        '''
        lang_cond: language conditional data, (batch_size, lang_len, hidden_size).
        lang_attn_mask: (batch_size, lang_len), a mask for valid language tokens,
            which should be True-False bool tensor.
        img_cond: image conditional data, (batch_size, img_len, hidden_size).
        state_traj: (batch_size, 1, hidden_size), state trajectory.
        action_mask: (batch_size, 1, action_dim), a 0-1 **float** tensor
            indicating the valid action dimensions.
        ctrl_freqs: (batch_size,), control frequency for each sample.
        
        return: (batch_size, horizon, action_dim)
        '''
        device = state_traj.device
        dtype = state_traj.dtype
        noisy_action = torch.randn(
            size=(state_traj.shape[0], self.pred_horizon, self.action_dim), 
            dtype=dtype, device=device)
        action_mask = action_mask.expand(-1, self.pred_horizon, -1)
    
        # Set step values
        self.noise_scheduler_sample.set_timesteps(self.num_inference_timesteps)
        
        for t in self.noise_scheduler_sample.timesteps:
            # Prepare state-action trajectory
            action_traj = torch.cat([noisy_action, action_mask], dim=2)
            action_traj = self.state_adaptor(action_traj)
            state_action_traj = torch.cat([state_traj, action_traj], dim=1)
            
            # Predict the model output
            model_output = self.model(state_action_traj, ctrl_freqs,
                                    t.unsqueeze(-1).to(device),
                                    lang_cond, img_cond, lang_mask=lang_attn_mask)
            
            # Compute previous actions: x_t -> x_t-1
            noisy_action = self.noise_scheduler_sample.step(
                model_output, t, noisy_action).prev_sample
            noisy_action = noisy_action.to(state_traj.dtype)
        
        # Finally apply the action mask to mask invalid action dimensions
        noisy_action = noisy_action * action_mask

        return noisy_action
    
    # ========= Inference  ============
    def predict_action(self, lang_tokens, lang_attn_mask, img_tokens, state_tokens,
                       action_mask, ctrl_freqs):
        '''
        lang_tokens: (batch_size, lang_len, lang_token_dim)
        lang_attn_mask: (batch_size, lang_len), a mask for valid language tokens,
            which should be True-False bool tensor.
        img_tokens: (batch_size, img_len, img_token_dim)
        state_tokens: (batch_size, 1, state_token_dim)
        action_mask: (batch_size, 1, action_dim),
            which should be a 0-1 **float** tensor.
        ctrl_freqs: (batch_size,), control frequency for each sample.
        
        return: (batch_size, horizon, action_dim), predicted action sequence
        '''
        # Prepare the state and conditions
        state_tokens = torch.cat([state_tokens, action_mask], dim=2)
        lang_cond, img_cond, state_traj = self.adapt_conditions(
            lang_tokens, img_tokens, state_tokens)
        
        # Run sampling
        action_pred = self.conditional_sample(
            lang_cond, lang_attn_mask, img_cond, 
            state_traj, action_mask, ctrl_freqs,
        )
        
        return action_pred
    
    def forward(self, image, qpos, actions=None, is_pad=None, conditioning_dict=None, future_image=None):
        '''
        image: (batch_size, num_cameras, 3, img_height, img_width)
        qpos: (batch_size, state_dim)
        actions: (batch_size, max_pad_len, state_dim)
        is_pad: (batch_size, max_pad_len), a mask for INvalid actions (True for invalid)
        future_image: (batch_size, num_cameras, 3, img_height, img_width), optional future image for Future-DINO loss
        '''
        # Preprocess inputs
        qpos = qpos.unsqueeze(1)  # (batch_size, 1, state_dim)  # adapter wants (numbers of tokens) dim in the middle
        # Get image embeddings
        language_embeddings_bnc = conditioning_dict["language_embeddings"]
        batch_size, num_cams, _, img_height, img_width = image.shape
        device = image.device
        ctrl_freqs_tensor = torch.tensor([CTRL_FREQS] * batch_size).to(device)  # TODO(roger): load CTRL FREQS from some parameters

        with torch.no_grad():
            image = image.view(batch_size * num_cams, -1, img_height, img_width)
            image_features = self.vision_encoder(image)  # (B, num_patches, hidden_size)
            num_patches = image_features.shape[1]
            image_embeds = image_features.reshape(batch_size, num_cams * num_patches, self.vision_encoder.hidden_size).detach()
        
            # Assemble the mask indicating each dimension's availability 
            # For actions, we use constant masking, but mask the loss using `is_pad` during training time.
            # TODO(roger): check this part. We may need to modify this for different embodiments?
            valid_action_masks = torch.ones(
                (batch_size, 1, qpos.shape[2]),
                device=qpos.device, dtype=qpos.dtype
            )
            lang_token_valid_mask = conditioning_dict['language_embeddings_mask']

        if actions is None:
            # Inference time
            trajectory = self.predict_action(
                lang_tokens=language_embeddings_bnc,
                lang_attn_mask=lang_token_valid_mask,
                img_tokens=image_embeds,
                state_tokens=qpos,
                action_mask=valid_action_masks,
                ctrl_freqs=ctrl_freqs_tensor,
            )
            return trajectory
        else:
            # Training time
            loss_dict = {}

            # Sample noise that we'll add to the actions
            noise = torch.randn(
                actions.shape, dtype=actions.dtype, device=device
            )
            # Sample random diffusion timesteps
            timesteps = torch.randint(
                0, self.num_train_timesteps, 
                (batch_size,), device=device
            ).long()
            # Add noise to the clean actions according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_action = self.noise_scheduler.add_noise(
                actions, noise, timesteps)
            
            # Concatenate the state and action tokens to form the input sequence
            state_action_traj = torch.cat([qpos, noisy_action], dim=1)
            # Append the action mask to the input sequence
            action_mask = valid_action_masks
            action_mask = action_mask.expand(-1, state_action_traj.shape[1], -1)
            state_action_traj = torch.cat([state_action_traj, action_mask], dim=2)
            # Align the dimension with the hidden size
            lang_cond, img_cond, state_action_traj = self.adapt_conditions(
                language_embeddings_bnc, image_embeds, state_action_traj)
            # Predict the denoised result
            pred = self.model(state_action_traj, ctrl_freqs_tensor, 
                            timesteps, lang_cond, img_cond, 
                            lang_mask=lang_token_valid_mask)

            pred_type = self.prediction_type
            if pred_type == 'epsilon':
                target = noise
            elif pred_type == 'sample':
                target = actions
            else:
                raise ValueError(f"Unsupported prediction type {pred_type}")
        
            valid_pred = pred[~is_pad]
            valid_target = target[~is_pad]

            LOSS_BALANCING = True

            if LOSS_BALANCING:
                # Separate loss for hand and finger keypoints
                # to avoid excessive attention of finger keypoints loss
                left_eef_loss = F.l1_loss(valid_pred[:, hdt.constants.OUTPUT_LEFT_EEF], valid_target[:, hdt.constants.OUTPUT_LEFT_EEF])
                right_eef_loss = F.l1_loss(valid_pred[:, hdt.constants.OUTPUT_RIGHT_EEF], valid_target[:, hdt.constants.OUTPUT_RIGHT_EEF])
                left_kpts_loss = F.l1_loss(valid_pred[:, hdt.constants.OUTPUT_LEFT_KEYPOINTS], valid_target[:, hdt.constants.OUTPUT_LEFT_KEYPOINTS])
                right_kpts_loss = F.l1_loss(valid_pred[:, hdt.constants.OUTPUT_RIGHT_KEYPOINTS], valid_target[:, hdt.constants.OUTPUT_RIGHT_KEYPOINTS])
                head_kpts_loss = F.l1_loss(valid_pred[:, hdt.constants.OUTPUT_HEAD_EEF], valid_target[:, hdt.constants.OUTPUT_HEAD_EEF])
                # Increase head loss weight
                loss_dict['loss'] = left_eef_loss + right_eef_loss + 1.0 * head_kpts_loss + 0.5 * (left_kpts_loss + right_kpts_loss)
            else:
                # Compute loss against the entire trajectory
                loss_dict['l2'] = F.mse_loss(valid_pred, valid_target)
                loss_dict['l1'] = F.l1_loss(valid_pred, valid_target)
                # loss_dict['initial_state_loss']  = F.l1_loss(pred[:, 0], target[:, 0])
                loss_dict['loss'] = loss_dict['l1']

            # Future-DINO loss (optional auxiliary loss)
            if self.use_future_dino_head and future_image is not None and self.future_dino_head is not None:
                # Extract shared features from RDT model
                # We need to get the last layer's output from the RDT model
                # For now, we'll use the image_embeds as the memory for Future-DINO head
                # and predict future DINO tokens

                # Encode future image to get target DINO features
                with torch.no_grad():
                    future_batch_size, future_num_cams, _, future_h, future_w = future_image.shape
                    future_image_flat = future_image.view(future_batch_size * future_num_cams, -1, future_h, future_w)
                    future_image_features = self.vision_encoder(future_image_flat)
                    future_num_patches = future_image_features.shape[1]
                    future_image_embeds = future_image_features.reshape(
                        future_batch_size, future_num_cams * future_num_patches, self.vision_encoder.hidden_size
                    ).detach()

                # Current DINO tokens (from image_embeds)
                current_dino_tokens = image_embeds  # [B, P, D_dino]

                # Get HAT shared features as memory
                # `state_action_traj` carries the NOISY GROUND-TRUTH FUTURE ACTIONS.
                # Feeding it to the world head let the head read the answer instead of
                # inferring it, so the world loss could drop without the trunk learning
                # anything. Only the image conditioning is used now.
                hat_memory = img_cond  # [B, N_img, hidden_size]

                # Predict future DINO tokens
                pred_future_dino = self.future_dino_head(
                    current_dino_tokens=current_dino_tokens,  # [B, P, D_dino]
                    hat_memory=hat_memory,  # [B, N, hidden_size]
                )  # [B, P, D_dino]

                # Compute Future-DINO loss
                future_dino_loss_dict = self.future_dino_loss_fn(
                    predicted=pred_future_dino,
                    target=future_image_embeds,  # [B, P, D_dino]
                )

                # Warmup the Future-DINO loss weight
                # Get current step from config or use a default
                current_step = self.get_current_step()
                warmup_factor = min(1.0, current_step / max(self.future_dino_warmup_steps, 1))
                effective_weight = self.future_dino_weight * warmup_factor

                loss_dict['future_dino_cosine_loss'] = future_dino_loss_dict['cosine_loss']
                loss_dict['future_dino_huber_loss'] = future_dino_loss_dict['huber_loss']
                loss_dict['future_dino_loss'] = future_dino_loss_dict['loss']
                loss_dict['loss'] = loss_dict['loss'] + effective_weight * future_dino_loss_dict['loss']

            return loss_dict

    def get_current_step(self):
        """Get current training step (override in subclass or set externally)."""
        return getattr(self, '_training_step', 0)

    def set_training_step(self, step):
        """Set current training step. Called by forward_pass() every iteration."""
        self._training_step = step

if __name__ == "__main__":
    import yaml
    import time
    from modeling.modeling_t5 import T5Embedder

    BATCH_SIZE = 4
    CAM_COUNT = 6

    T5_PATH = "/data/pretrained_weights/t5-v1_1-xxl"
    WEIGHT_DTYPE = torch.float32

    # Load config from template from official RDT repo
    CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../configs/models/rdt_tiny.yaml")

    with open(CONFIG_PATH, "r") as fp:
        config = yaml.safe_load(fp)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get text embedding
    if True:
        text_embeds = torch.zeros((BATCH_SIZE, 22, 4096), dtype=torch.float32, device=device)
    else:
        INSTRUCTION = "Push the ADA door opening button to open the door."
        text_embedder = T5Embedder(
            from_pretrained=T5_PATH, 
            model_max_length=T5Embedder.TOKENIZER_MAX_LENGTH,
            device=device,
            use_offload_folder="/tmp",
            torch_dtype=WEIGHT_DTYPE
        )
        tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model

        tokens = tokenizer(
            INSTRUCTION, return_tensors="pt",
            padding="longest",
            truncation=True
        )["input_ids"].to(device)

        tokens = tokens.view(1, -1)
        with torch.no_grad():
            text_embeds = text_encoder(tokens).last_hidden_state.detach()
        
        del text_embedder
        torch.cuda.empty_cache()

        text_embeds = text_embeds.repeat(BATCH_SIZE, 1, 1)
        
    conditioning_dict = {
        "language_embeddings": text_embeds,
    }

    # Set up multi-modal RDT model
    hdt = HumanDiffusionTransformer(
        action_dim=config["common"]["state_dim"],
        pred_horizon=config["common"]["action_chunk_size"],
        config=config,
        lang_token_dim=config["model"]["lang_token_dim"],
        img_token_dim=config["model"]["img_token_dim"],
        state_token_dim=config["model"]["state_token_dim"],
        max_lang_cond_len=config["dataset"]["tokenizer_max_length"],
        lang_pos_embed_config=[
            # Similarly, no initial pos embed for language
            ("lang", -config["dataset"]["tokenizer_max_length"]),
        ],
        dtype=WEIGHT_DTYPE,
    )
    hdt = hdt.to(device)
    hdt.eval()

    # Make random states
    state = torch.randn(
        (BATCH_SIZE, config["common"]["state_dim"]), dtype=WEIGHT_DTYPE, device=device
    )

    example_image = Image.open(os.path.join(os.path.dirname(__file__), "test_wrist.png")).convert("RGB")
    example_np_hw3 = np.array(example_image)
    example_np_chw = np.transpose(example_np_hw3, (2, 0, 1))
    example_np_chw = example_np_chw / 255.0
    example_tensor_chw = torch.tensor(example_np_chw, dtype=torch.float32)
    example_tensor_nchw = example_tensor_chw.unsqueeze(0)
    example_tensor_bnchw = example_tensor_nchw.unsqueeze(0)
    # Repeat the image for 4 batch size and 2 cameras
    example_tensor_bnchw = example_tensor_bnchw.repeat(BATCH_SIZE, CAM_COUNT, 1, 1, 1).cuda()

    # Predict action (inference)
    with torch.no_grad():
        # warm up
        for _ in range(10):
            trajectory = hdt(
                image=example_tensor_bnchw,
                qpos=state,
                actions=None,
                is_pad=None,
                conditioning_dict=conditioning_dict
            )
        # profile
        start_cp = time.time()
        TEST_ITERS = 100
        for _ in range(TEST_ITERS):
            trajectory = hdt(
                image=example_tensor_bnchw,
                qpos=state,
                actions=None,
                is_pad=None,
                conditioning_dict=conditioning_dict
            )
        end_cp = time.time()
        print("Inference time:", end_cp - start_cp)
        print("Average inference time:", (end_cp - start_cp) / TEST_ITERS)

    print("[Inference-only] DP Predicted trajectory shape:", trajectory.shape)
    print(trajectory.shape)

    # Test training
    hdt.train()
    trajectory_gt = torch.randn(
        trajectory.shape, dtype=trajectory.dtype, device=trajectory.device
    )
    is_pad = torch.zeros(
        (BATCH_SIZE, trajectory.shape[1]), dtype=torch.bool, device=trajectory.device
    )

    loss_dict = hdt(
        image=example_tensor_bnchw,
        qpos=state,
        actions=trajectory_gt,
        is_pad=is_pad,
        conditioning_dict=conditioning_dict
    )

    print("[Training] Loss dict:", loss_dict)
