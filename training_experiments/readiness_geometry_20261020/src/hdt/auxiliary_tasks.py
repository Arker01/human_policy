"""Isolated readiness and scene-frame fingertip objectives.

Readiness labels describe the interval BEFORE a demonstrated transition, never
closed/open recognition. Uncertain boundaries and unobserved phases are masked.
The prediction head only sees current visual tokens (no future CVAE posterior).
"""
import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

VARIANT = os.environ.get('HDT_EXPERIMENT', 'baseline')


def transition_labels(states, fps):
    n = len(states)
    targets = np.zeros((n, 4), np.float32)
    masks = np.zeros_like(targets)
    weights = np.ones((n, 2), np.float32)
    events = []
    # Five fingertips follow a zero palm-origin keypoint.
    for hand, offset in enumerate((10, 40)):
        tips = np.asarray(states[:, offset:offset + 18]).reshape(n, 6, 3)[:, 1:]
        spread = np.linalg.norm(tips[:, 1:] - tips[:, :1], axis=-1).mean(-1)
        smooth = max(3, int(round(fps * .1)) | 1)
        spread = np.convolve(np.pad(spread, smooth//2, mode='edge'),
                             np.ones(smooth)/smooth, mode='valid')
        low, high = np.percentile(spread, [10, 90])
        if high - low < .015:
            continue
        level = (spread-low)/(high-low)
        stable = np.where(level >= .75, 1, np.where(level <= .25, 0, -1))
        hold = max(2, int(round(.1*fps)))
        phase, phase_start = None, 0
        for t in range(hold-1, n):
            destination = int(stable[t])
            if destination < 0 or not np.all(stable[t-hold+1:t+1] == destination):
                continue
            if phase is None:
                phase, phase_start = destination, t
                continue
            if destination == phase:
                continue
            # Find the last frame still at the source plateau: onset must follow it.
            source = np.flatnonzero(stable[phase_start:t-hold+2] == phase)
            if not len(source):
                phase, phase_start = destination, t
                continue
            onset = phase_start + int(source[-1]) + 1
            col = 2*hand + (0 if phase == 1 else 1)
            pos_start = max(phase_start, onset-max(2, int(round(.2*fps))))
            neg_end = max(phase_start, onset-int(round(.6*fps)))
            targets[pos_start:onset, col] = 1
            masks[pos_start:onset, col] = 1
            masks[phase_start:neg_end, col] = 1
            weights[max(0,onset-int(.3*fps)):min(n,t+int(.3*fps)+1),hand] = 5
            events.append((hand, 'close' if phase else 'open', onset, t))
            phase, phase_start = destination, t
    return targets, masks, weights, events


class ReadinessHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(4, dim)*.02)
        self.attn = nn.MultiheadAttention(dim, 8, batch_first=True, dropout=0.)
        self.classifier = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim,128),
                                        nn.GELU(), nn.Linear(128,1))

    def forward(self, src, pos):
        values = src.flatten(2).transpose(1,2)
        keys = values + pos.flatten(2).transpose(1,2)
        q = self.queries[None].expand(len(src),-1,-1)
        pooled, _ = self.attn(q, keys, values, need_weights=False)
        return self.classifier(pooled + q).squeeze(-1)


def balanced_readiness_loss(logits, target, mask):
    element = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    # Average positives and negatives separately per head. Missing classes
    # contribute zero, with no batch-dependent huge positive weights.
    pos, neg = mask*target, mask*(1-target)
    pp, nn_ = pos.sum(0), neg.sum(0)
    present = (pp>0).float() + (nn_>0).float()
    per_head = ((element*pos).sum(0)/pp.clamp_min(1)
              +(element*neg).sum(0)/nn_.clamp_min(1))/present.clamp_min(1)
    return per_head.sum()/(present>0).sum().clamp_min(1)


def rotation_6d_rows(d):
    a,b = d[..., :3], d[...,3:]
    x = F.normalize(a,dim=-1)
    y = F.normalize(b-(x*b).sum(-1,keepdim=True)*x,dim=-1)
    return torch.stack((x,y,torch.cross(x,y,dim=-1)),dim=-2)


def scene_tips(action):
    hands=[]
    for base, fingers in ((80,10),(30,40)):
        p = action[...,base:base+3]
        r = rotation_6d_rows(action[...,base+3:base+9])
        local = action[...,fingers:fingers+18].reshape(*action.shape[:-1],6,3)[...,1:,:]
        hands.append(torch.einsum('...ij,...kj->...ki',r,local)+p[...,None,:])
    return torch.stack(hands,dim=-3)


def geometry_loss(pred, target, mean, std, weights, is_pad):
    pred = scene_tips(pred*(std[:,None]+1e-6)+mean[:,None])
    target = scene_tips(target*(std[:,None]+1e-6)+mean[:,None])
    distance = torch.linalg.vector_norm(pred-target,dim=-1).mean(-1)
    weight = weights * (~is_pad)[...,None]
    loss = F.smooth_l1_loss(pred,target,reduction='none',beta=.01).mean((-1,-2))/.01
    return (loss*weight).sum()/weight.sum().clamp_min(1), (distance*weight).sum()/weight.sum().clamp_min(1)*100


def dataset_aux(dataset,index,start_ts):
    if not hasattr(dataset,'_aux_cache'):
        dataset._aux_cache={}
    if index not in dataset._aux_cache:
        import h5py
        with h5py.File(dataset.dataset_paths[index],'r') as h:
            states = h['observation.state'][:]
        dataset._aux_cache[index] = transition_labels(states,dataset.source_fps_list[index])[:3]
    target,mask,weight = dataset._aux_cache[index]
    raw = dataset._raw_index(index,start_ts)
    embodiment = dataset.source_embodiment_list[index]
    # Exactly align event weights with original action-mode interpolation.
    if 'human' in embodiment.lower():
        available = min(dataset.chunk_size,len(weight)-raw)
        positions = np.linspace(raw,raw+available-1,available*max(int(dataset.slow_down_factor),1))[:dataset.chunk_size]
        indices = np.floor(positions).astype(int)
    else:
        indices = dataset.episode_frame_indices[index][start_ts:start_ts+dataset.chunk_size]
    ww = np.ones((dataset.chunk_size,2),np.float32)
    ww[:len(indices)] = weight[indices]
    stats=dataset.norm_stats[embodiment]
    return dict(readiness_target=torch.from_numpy(target[raw].copy()),
                readiness_mask=torch.from_numpy(mask[raw].copy()),
                hand_event_weights=torch.from_numpy(ww),
                action_mean=torch.from_numpy(stats['action_mean']).float(),
                action_std=torch.from_numpy(stats['action_std']).float())


def gate_hand_commands(raw_action_chunk, raw_current_state, readiness_prob, threshold=.5, deadband_m=.003):
    """Optional deployment gate on raw 128-D commands; wrist motion continues.

    Inputs may be batched, with chunk [B,T,128], current [B,128], prob [B,4].
    Current hand geometry is held only when a proposed aperture change lacks the
    corresponding pre-action readiness. No robot I/O is performed here.
    """
    result=raw_action_chunk.clone()
    for hand,offset in enumerate((10,40)):
        current=raw_current_state[:,offset:offset+18].reshape(-1,6,3)[:,1:]
        planned=raw_action_chunk[:,:,offset:offset+18].reshape(*raw_action_chunk.shape[:2],6,3)[:,:,1:]
        current_ap=torch.linalg.vector_norm(current[:,1:]-current[:,:1],dim=-1).mean(-1)
        planned_ap=torch.linalg.vector_norm(planned[:,:,1:]-planned[:,:,:1],dim=-1).mean(-1)
        delta=planned_ap-current_ap[:,None]
        blocked=((delta < -deadband_m)&(readiness_prob[:,2*hand,None]<threshold))|((delta>deadband_m)&(readiness_prob[:,2*hand+1,None]<threshold))
        result[:,:,offset:offset+18]=torch.where(blocked[:,:,None],raw_current_state[:,None,offset:offset+18],result[:,:,offset:offset+18])
    return result
