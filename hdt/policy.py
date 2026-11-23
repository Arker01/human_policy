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
        # History frames must go through the SAME resize+normalize as the current
        # frame, or the trunk sees two different image distributions and the extra
        # slots are noise. Copied rather than mutated so the caller's dict is intact.
        if conditioning_dict is not None and 'hist_image' in conditioning_dict:
            conditioning_dict = dict(conditioning_dict)
            conditioning_dict['hist_image'] = self.transform(conditioning_dict['hist_image'])
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
            variant = self.model.experiment_variant
            if variant not in ('sonic_native', 'sonic_native_geometry'):
                # neck is not needed downstream; zero its contribution everywhere
                # (base L1 and any later slices) so the model spends no capacity on it.
                all_l1[:, :, hdt.constants.OUTPUT_NECK] = 0.0
            if variant in ('wo_root', 'hat_native'):
                # Labels and posterior input stay untouched. Only supervised loss
                # terms for the nine root dimensions are removed.
                all_l1[:, :, hdt.constants.OUTPUT_WAIST] = 0.0
            elif variant == 'wo_fingertips':
                # Keep the repurposed absolute grip scalars at dims 10 and 40;
                # suppress every other coordinate in the two 18-D fingertip blocks.
                all_l1[:, :, 11:28] = 0.0
                all_l1[:, :, 41:58] = 0.0
            loss_dict['l1'] = all_l1.mean()
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            if conditioning_dict is not None and 'semantic_missing_mask' in conditioning_dict:
                missing = conditioning_dict['semantic_missing_mask'].float()
                names = ('head', 'root', 'right_wrist', 'right_hand', 'vision', 'left_wrist', 'left_hand')
                for i, name in enumerate(names):
                    loss_dict[f'missing_{name}_frac'] = missing[:, i].mean()
                loss_dict['missing_block_count'] = missing.sum(dim=1).mean()
                loss_dict['missing_head_root_joint_frac'] = (
                    (missing[:, 0] > 0) & (missing[:, 1] > 0)).float().mean()
            if conditioning_dict is not None and 'history_present' in conditioning_dict:
                loss_dict['history_present_frac'] = conditioning_dict['history_present'].float().mean()
                loss_dict['plan_present_frac'] = conditioning_dict['plan_present'].float().mean()
            if self.USE_EEF_LOSS and variant not in ('sonic_native', 'sonic_native_geometry'):
                loss_dict['eef_loss'] = all_l1[:, :, hdt.constants.OUTPUT_LEFT_EEF].mean() + all_l1[:, :, hdt.constants.OUTPUT_RIGHT_EEF].mean()
                loss_dict['loss'] = loss_dict['loss'] + loss_dict['eef_loss'] * 2

                # head loss: position + 6D rotation (full OUTPUT_HEAD_EEF, was position-only)
                loss_dict['head_loss'] = all_l1[:, :, hdt.constants.OUTPUT_HEAD_EEF].mean()
                loss_dict['loss'] = loss_dict['loss'] + loss_dict['head_loss'] * 2

                # waist loss: position + 6D rotation, same extra weight as EEF
                if variant not in ('wo_root', 'hat_native'):
                    loss_dict['waist_loss'] = all_l1[:, :, hdt.constants.OUTPUT_WAIST].mean()
                    loss_dict['loss'] = loss_dict['loss'] + loss_dict['waist_loss'] * 2

            if self.query0_extra_weight > 0:
                # anchor the query actually used at inference (a_hat[:,0]) to the
                # immediate ground-truth frame; chunk-wise L1 alone lets query0
                # drift into a ~1-frame-lagged/smoothed estimate of the trajectory.
                loss_dict['query0_loss'] = all_l1[:, 0, :].mean()
                loss_dict['loss'] = loss_dict['loss'] + loss_dict['query0_loss'] * self.query0_extra_weight

            # Add Future-DINO loss if available
            if future_dino_loss_dict is not None and 'loss' in future_dino_loss_dict:
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

                # Third future-target species (EgoWAM 3D point flow). Same carrier
                # dict, same contract. flow_valid_frac is the one to watch: the
                # movement threshold that builds the target throws away static
                # anchors, so if this collapses toward 0 the head is training on an
                # empty mask and the threshold is wrong for this data.
                if 'flow_loss' in future_dino_loss_dict:
                    flow_weight = future_dino_loss_dict.get('flow_effective_weight', 0.0)
                    loss_dict['future_flow_loss'] = future_dino_loss_dict['flow_loss']
                    loss_dict['future_flow_valid_frac'] = future_dino_loss_dict['flow_valid_frac']
                    loss_dict['future_flow_gt_mag'] = future_dino_loss_dict['flow_gt_mag']
                    loss_dict['future_flow_pred_mag'] = future_dino_loss_dict['flow_pred_mag']
                    loss_dict['future_flow_effective_weight'] = torch.as_tensor(
                        flow_weight, device=loss_dict['loss'].device, dtype=loss_dict['loss'].dtype)
                    loss_dict['loss'] = loss_dict['loss'] + flow_weight * future_dino_loss_dict['flow_loss']

            from auxiliary_tasks import balanced_readiness_loss, geometry_loss, fingertip_geometry_loss
            geom = None
            if self.model.experiment_variant == 'readiness':
                logits = self.model.readiness_logits
                target = conditioning_dict['readiness_target']
                mask = conditioning_dict['readiness_mask']
                ready_loss = balanced_readiness_loss(logits, target, mask)
                loss_dict['readiness_loss'] = ready_loss
                loss_dict['readiness_valid_frac'] = mask.mean()
                loss_dict['readiness_positive_frac'] = (mask*target).sum()/mask.sum().clamp_min(1)
                loss_dict['readiness_accuracy'] = (((logits>0)==(target>0)).float()*mask).sum()/mask.sum().clamp_min(1)
                loss_dict['loss'] = loss_dict['loss'] + .1*ready_loss
            elif self.model.experiment_variant in ('geometry', 'wo_root'):
                geom, cm, per_hand = geometry_loss(a_hat, actions, conditioning_dict['action_mean'],
                                        conditioning_dict['action_std'],
                                        conditioning_dict['hand_event_weights'][:,:self.model.num_queries], is_pad)
            elif variant == 'sonic_native_geometry':
                geom, cm, per_hand = fingertip_geometry_loss(
                    future_dino_loss_dict['sonic_scene_tips'],
                    conditioning_dict['native_scene_tips'][:, :self.model.num_queries],
                    conditioning_dict['hand_event_weights'][:, :self.model.num_queries],
                    is_pad,
                )
            if geom is not None:
                loss_dict['geometry_loss'] = geom
                loss_dict['geometry_left_loss'], loss_dict['geometry_right_loss'] = per_hand
                loss_dict['geometry_error_cm'] = cm
                loss_dict['loss'] = loss_dict['loss'] + .1 * geom
            return loss_dict
        else: # inference time
            a_hat, _, (_, _), _ = self.model(qpos, image, env_state, conditioning_dict=conditioning_dict) # no action, sample from prior
            return a_hat

    @torch.no_grad()
    def predict_with_readiness(self, image, qpos, conditioning_dict=None):
        trajectory = self(image, qpos, conditioning_dict=conditioning_dict)
        if self.model.readiness_logits is None:
            raise RuntimeError('Load a readiness model with HDT_EXPERIMENT=readiness')
        return trajectory, self.model.readiness_logits.sigmoid()

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
