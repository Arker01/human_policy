import torch.nn as nn
from torch.nn import functional as F
from torchvision.transforms import v2
import torch

from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
import IPython
e = IPython.embed

import hdt.constants

class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        self.query0_extra_weight = args_override.get('query0_extra_weight', 0.0)
        # patch_h = 24
        # patch_w = 42
        patch_h = 16
        patch_w = 22
        # FIXME(roger): RSS temp. EEF loss
        self.USE_EEF_LOSS = True
        self.transform = v2.Compose([
            v2.Resize((patch_h * 14, patch_w * 14)),
            # v2.CenterCrop((patch_h * 14, patch_w * 14)),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        print(f'KL Weight {self.kl_weight}')
        # Check if Future-DINO is enabled
        self.use_future_dino = getattr(self.model, 'use_future_dino_head', False)
        if self.use_future_dino:
            print(f'Future-DINO Head enabled in ACTPolicy')

    def __call__(self, image, qpos, actions=None, is_pad=None, conditioning_dict=None, future_image=None):
        env_state = None
            
        image = self.transform(image)
        future_image_transformed = None
        if future_image is not None:
            future_image_transformed = self.transform(future_image)

        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar), future_dino_loss_dict = self.model(
                qpos, image, env_state, actions, is_pad, conditioning_dict,
                future_image=future_image_transformed
            )
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            all_l1 = (all_l1 * ~is_pad.unsqueeze(-1))
            # neck is not needed downstream; zero its contribution everywhere
            # (base L1 and any later slices) so the model spends no capacity on it.
            all_l1[:, :, hdt.constants.OUTPUT_NECK] = 0.0
            loss_dict['l1'] = all_l1.mean()
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            if self.USE_EEF_LOSS:
                loss_dict['eef_loss'] = all_l1[:, :, hdt.constants.OUTPUT_LEFT_EEF].mean() + all_l1[:, :, hdt.constants.OUTPUT_RIGHT_EEF].mean()
                loss_dict['loss'] = loss_dict['loss'] + loss_dict['eef_loss'] * 2

                # head loss: position + 6D rotation (full OUTPUT_HEAD_EEF, was position-only)
                loss_dict['head_loss'] = all_l1[:, :, hdt.constants.OUTPUT_HEAD_EEF].mean()
                loss_dict['loss'] = loss_dict['loss'] + loss_dict['head_loss'] * 2

                # waist loss: position + 6D rotation, same extra weight as EEF
                loss_dict['waist_loss'] = all_l1[:, :, hdt.constants.OUTPUT_WAIST].mean()
                loss_dict['loss'] = loss_dict['loss'] + loss_dict['waist_loss'] * 2

            if self.query0_extra_weight > 0:
                # anchor the query actually used at inference (a_hat[:,0]) to the
                # immediate ground-truth frame; chunk-wise L1 alone lets query0
                # drift into a ~1-frame-lagged/smoothed estimate of the trajectory.
                loss_dict['query0_loss'] = all_l1[:, 0, :].mean()
                loss_dict['loss'] = loss_dict['loss'] + loss_dict['query0_loss'] * self.query0_extra_weight

            # Add Future-DINO loss if available
            if future_dino_loss_dict is not None:
                effective_weight = future_dino_loss_dict.get('effective_weight', 0.0)
                loss_dict['future_dino_cosine_loss'] = future_dino_loss_dict['cosine_loss']
                loss_dict['future_dino_huber_loss'] = future_dino_loss_dict['huber_loss']
                loss_dict['future_dino_loss'] = future_dino_loss_dict['loss']
                # log the warmed-up weight too: if this stays 0 the world head is
                # attached but contributing nothing, which is exactly how the
                # original run silently trained a dead head for 100k steps.
                loss_dict['future_dino_effective_weight'] = torch.as_tensor(
                    effective_weight, device=loss_dict['loss'].device, dtype=loss_dict['loss'].dtype)
                loss_dict['loss'] = loss_dict['loss'] + effective_weight * future_dino_loss_dict['loss']

                # Second future-target species (frozen Wan VAE, ST-WAM style). Present
                # only when future_vae is enabled; carried in the same dict so nothing
                # about the model's return contract changed.
                if 'vae_loss' in future_dino_loss_dict:
                    vae_weight = future_dino_loss_dict.get('vae_effective_weight', 0.0)
                    loss_dict['future_vae_cosine_loss'] = future_dino_loss_dict['vae_cosine_loss']
                    loss_dict['future_vae_huber_loss'] = future_dino_loss_dict['vae_huber_loss']
                    loss_dict['future_vae_loss'] = future_dino_loss_dict['vae_loss']
                    loss_dict['future_vae_effective_weight'] = torch.as_tensor(
                        vae_weight, device=loss_dict['loss'].device, dtype=loss_dict['loss'].dtype)
                    loss_dict['loss'] = loss_dict['loss'] + vae_weight * future_dino_loss_dict['vae_loss']

            return loss_dict
        else: # inference time
            a_hat, _, (_, _), _ = self.model(qpos, image, env_state, conditioning_dict=conditioning_dict) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model # decoder
        self.optimizer = optimizer

    def __call__(self, image, qpos, actions=None, is_pad=None, conditioning_dict=None):
        env_state = None # TODO
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else: # inference time
            a_hat = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
