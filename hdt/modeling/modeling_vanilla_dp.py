import torch.nn as nn
from torch.nn import functional as F
import torch
import numpy as np

from collections import OrderedDict
from robomimic.models.base_nets import ResNet18Conv, SpatialSoftmax
from robomimic.algo.diffusion_policy import replace_bn_with_gn, ConditionalUnet1D

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel

NUM_CAMERAS = 2
OBS_HORIZON = 1

class DiffusionPolicy(nn.Module):
    def __init__(self, action_dim, chunk_size, img_token_dim, state_token_dim, num_inference_timesteps, visual_encoder):
        super().__init__()

        assert state_token_dim == action_dim

        self.state_token_dim = state_token_dim
        self.prediction_horizon = chunk_size
        self.num_inference_timesteps = num_inference_timesteps
        self.ema_power = 0.75
        self.weight_decay = 0

        self.feature_dimension = img_token_dim
        self.ac_dim = action_dim
        self.obs_dim = self.feature_dimension * NUM_CAMERAS + self.state_token_dim

        self.visual_encoder = visual_encoder.eval()

        pool = SpatialSoftmax(**{'input_shape': [512, 15, 20], 'num_kp': self.feature_dimension, 'temperature': 1.0, 'learnable_temperature': False, 'noise_std': 0.0})
        linear = torch.nn.Linear(int(np.prod([self.feature_dimension, 2])), self.feature_dimension)

        noise_pred_net = ConditionalUnet1D(
            input_dim=self.ac_dim,
            global_cond_dim=self.obs_dim*OBS_HORIZON
        )

        nets = nn.ModuleDict({
            'policy': nn.ModuleDict({
                'pool': pool,
                'linear': linear,
                'noise_pred_net': noise_pred_net
            })
        })

        nets = nets.float().cuda()
        # TODO: deal with loading
        ENABLE_EMA = False
        if ENABLE_EMA:
            ema = EMAModel(parameters=nets.parameters(), power=self.ema_power)
        else:
            ema = None
        self.nets = nets
        self.ema = ema

        self.prediction_type = 'epsilon'  # 'epsilon' or 'sample'

        # setup noise scheduler
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=100,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type=self.prediction_type,
        )

        n_parameters = sum(p.numel() for p in self.parameters())
        print("number of parameters: %.2fM" % (n_parameters/1e6,))

    def __call__(self, image, qpos, actions=None, is_pad=None, conditioning_dict=None):
        batch_size, num_cams, _, img_height, img_width = image.shape

        assert num_cams == NUM_CAMERAS

        nets = self.nets
        if self.ema is not None and actions is None:
            # inference time. Use EMA model
            nets = self.ema.averaged_model

        cam_images = image.view(batch_size * num_cams, -1, img_height, img_width)
        with torch.no_grad():
            cam_features = self.visual_encoder(cam_images)
            cam_features = cam_features.permute((0, 2, 1)).reshape(batch_size * num_cams, self.feature_dimension, img_height // 16, img_width // 16)
        pool_features = nets['policy']['pool'](cam_features)
        pool_features = torch.flatten(pool_features, start_dim=1)
        image_embeds = nets['policy']['linear'](pool_features)
        image_embeds = image_embeds.view(batch_size, num_cams * image_embeds.shape[1])

        obs_cond = torch.cat([image_embeds, qpos], dim=1)

        if actions is not None: # training time
            loss_dict = {}

            # sample noise to add to actions
            noise = torch.randn(actions.shape, device=obs_cond.device)
            
            # sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, 
                (batch_size,), device=obs_cond.device
            ).long()
            
            # add noise to the clean actions according to the noise magnitude at each diffusion iteration
            # (this is the forward diffusion process)
            noisy_actions = self.noise_scheduler.add_noise(
                actions, noise, timesteps)
            
            # predict the noise residual
            noise_pred = nets['policy']['noise_pred_net'](noisy_actions, timesteps, global_cond=obs_cond)

            if self.prediction_type == 'epsilon':
                target = noise
            elif self.prediction_type == 'sample':
                target = actions
            else:
                raise ValueError(f"Unsupported prediction type {self.prediction_type}")
        
            valid_pred = noise_pred[~is_pad]
            valid_target = target[~is_pad]

            loss_dict['l2'] = F.mse_loss(valid_pred, valid_target)
            loss_dict['l1'] = F.l1_loss(valid_pred, valid_target)
            loss_dict['loss'] = loss_dict['l1']

            if self.training and self.ema is not None:
                self.ema.step(nets.parameters())

            return loss_dict

        else:
            # inference time
            # initialize action from Guassian noise
            noisy_action = torch.randn(
                (batch_size, self.prediction_horizon, self.ac_dim), device=obs_cond.device)
            naction = noisy_action
            
            # init scheduler
            self.noise_scheduler.set_timesteps(self.num_inference_timesteps)

            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = nets['policy']['noise_pred_net'](
                    sample=naction, 
                    timestep=k,
                    global_cond=obs_cond
                )

                # inverse diffusion step (remove noise)
                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample

            return naction

    def serialize(self):
        return {
            "nets": self.nets.state_dict(),
            "ema": self.ema.averaged_model.state_dict() if self.ema is not None else None,
        }

    def deserialize(self, model_dict):
        status = self.nets.load_state_dict(model_dict["nets"])
        print('Loaded model')
        if model_dict.get("ema", None) is not None:
            print('Loaded EMA')
            status_ema = self.ema.averaged_model.load_state_dict(model_dict["ema"])
            status = [status, status_ema]
        return status
