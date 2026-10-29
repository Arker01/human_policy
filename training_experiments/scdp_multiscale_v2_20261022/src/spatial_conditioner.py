"""One-pass trajectory-conditioned FiLM. No camera parameters are inferred here."""
import torch
from torch import nn
from torch.nn import functional as F


def resize_crop_affine(raw_hw, resized_hw, crop_tlhw=None, output_hw=(224, 304)):
    """Raw pixel centres -> encoder pixels, matching align_corners=False resize.

    Crop is applied AFTER the first resize. Pass the actual sampled crop, never
    reconstruct a RandomResizedCrop window from its output image.
    """
    rh, rw = raw_hw
    ih, iw = resized_hw
    top, left, ch, cw = crop_tlhw or (0, 0, ih, iw)
    oh, ow = output_hw
    if min(rh, rw, ih, iw, ch, cw, oh, ow) <= 0:
        raise ValueError('Image and crop sizes must be positive')
    sx, sy = iw / rw, ih / rh
    ax, ay = ow / cw, oh / ch
    return torch.tensor([[sx * ax, 0., ((sx - 1) / 2 - left + .5) * ax - .5],
                         [0., sy * ay, ((sy - 1) / 2 - top + .5) * ay - .5],
                         [0., 0., 1.]])


def project_wrist(xyz, camera, image_hw=(224, 304)):
    """xyz [B,A,2,3] in physical metres; T maps its KP frame into RGB camera.

    Camera contract: K [B,3,3], optional OpenCV rational D [B,8], T_cam_from_kp [B,4,4],
    pixel_from_raw [B,3,3], raw_hw [B,2], valid [B]. T MUST refer to the
    CURRENT observation, including its dynamic head pose and KP frame convention.
    """
    shape = xyz.shape[:-1]
    empty = torch.zeros(shape, dtype=torch.bool, device=xyz.device)
    if camera is None:
        return xyz.new_zeros(*shape, 2), empty
    required = ('K', 'T_cam_from_kp', 'pixel_from_raw', 'raw_hw', 'valid')
    if any(k not in camera for k in required):
        return xyz.new_zeros(*shape, 2), empty
    K, T, A, raw_hw, available = [camera[k].to(xyz.device) for k in required]
    K, T, A = K.float(), T.float(), A.float()
    p = torch.einsum('bij,bahj->bahi', T[:, :3, :3], xyz.float()) + T[:, None, None, :3, 3]
    z = p[..., 2:3]
    normalized = p[..., :2] / torch.where(z.abs() > 1e-8, z, torch.ones_like(z))
    D = camera.get('D', torch.zeros(K.shape[0], 8, device=xyz.device)).to(xyz.device).float()
    if D.shape != (K.shape[0], 8):
        raise ValueError('D must use OpenCV [k1,k2,p1,p2,k3,k4,k5,k6] order')
    k1,k2,p1,p2,k3,k4,k5,k6 = D[:,None,None,:].unbind(-1)
    x,y = normalized.unbind(-1)
    r2 = x.square()+y.square(); r4=r2.square(); r6=r4*r2
    numerator=1+k1*r2+k2*r4+k3*r6
    denominator=1+k4*r2+k5*r4+k6*r6
    safe_denominator=torch.where(denominator.abs()>1e-8,denominator,torch.ones_like(denominator))
    radial=numerator/safe_denominator
    distorted=torch.stack((x*radial+2*p1*x*y+p2*(r2+2*x.square()),
                           y*radial+p1*(r2+2*y.square())+2*p2*x*y),-1)
    uvz=torch.einsum('bij,bahj->bahi',K,torch.cat((distorted,torch.ones_like(z)),-1))
    raw_uv=uvz[..., :2]/uvz[..., 2:].clamp_min(1e-8)
    pixels = torch.cat((raw_uv, torch.ones_like(raw_uv[..., :1])), -1)
    mapped = torch.einsum('bij,bahj->bahi', A, pixels)
    uv = mapped[..., :2] / mapped[..., 2:].clamp_min(1e-8)
    h, w = image_hw
    valid = available.bool()[:, None, None].expand(shape).clone()
    valid &= torch.isfinite(xyz).all(-1) & torch.isfinite(p).all(-1) & (p[..., 2] > 1e-6)
    valid &= torch.isfinite(K).flatten(1).all(1)[:, None, None]
    valid &= torch.isfinite(D).all(1)[:,None,None] & torch.isfinite(denominator) & (denominator.abs()>1e-8)
    valid &= (K[:,0,0]>0)[:,None,None] & (K[:,1,1]>0)[:,None,None]
    hand_valid=camera.get('hand_valid',torch.ones(K.shape[0],2,dtype=torch.bool,device=xyz.device))
    valid &= hand_valid.to(xyz.device).bool()[:,None,:]
    valid &= torch.isfinite(T).flatten(1).all(1)[:, None, None]
    valid &= torch.isfinite(A).flatten(1).all(1)[:, None, None]
    valid &= torch.isfinite(raw_hw).all(1)[:, None, None] & (raw_hw > 0).all(1)[:, None, None]
    valid &= (raw_uv[..., 0] >= 0) & (raw_uv[..., 0] < raw_hw[:, None, None, 1])
    valid &= (raw_uv[..., 1] >= 0) & (raw_uv[..., 1] < raw_hw[:, None, None, 0])
    valid &= torch.isfinite(uv).all(-1) & (mapped[..., 2] > 0)
    valid &= (uv[..., 0] >= 0) & (uv[..., 0] < w) & (uv[..., 1] >= 0) & (uv[..., 1] < h)
    grid = 2 * (uv + .5) / uv.new_tensor([w, h]) - 1
    return torch.where(valid[..., None], grid, torch.zeros_like(grid)), valid


def sample_local(feature_map, grid, valid):
    # align_corners=False maps encoder pixel centres to native patch centres.
    local = F.grid_sample(feature_map, grid.to(feature_map.dtype), mode='bilinear',
                          padding_mode='zeros', align_corners=False).permute(0, 2, 3, 1)
    return torch.where(valid[..., None], local, torch.zeros_like(local))


class SpatialTrajectoryConditioner(nn.Module):
    def __init__(self, feature_dim=768, hidden_dim=512, film_dim=256, anchors=8,
                 wrist_indices=((80, 81, 82), (30, 31, 32)), enabled=False):
        super().__init__()
        if anchors < 1:
            raise ValueError('anchors must be positive')
        self.enabled, self.anchors = enabled, anchors
        self.register_buffer('wrist_indices', torch.tensor(wrist_indices), persistent=False)
        self.norm = nn.LayerNorm(hidden_dim)
        self.film = nn.Sequential(nn.Linear(2 * feature_dim, film_dim), nn.GELU(),
                                  nn.Linear(film_dim, 2 * hidden_dim))
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)

    def forward(self, feature_map, coarse, hidden, camera=None, action_mean=None,
                action_std=None):
        B, T, _ = hidden.shape
        ids = torch.linspace(0, T - 1, min(self.anchors, T), device=hidden.device).round().long()
        valid = torch.zeros(B, len(ids), 2, dtype=torch.bool, device=hidden.device)
        debug = {'anchor_indices': ids, 'valid': valid}
        if not self.enabled:
            return hidden, debug
        # Missing normalization is as unsafe as missing calibration.
        if action_mean is None or action_std is None:
            camera = None
            physical = coarse.detach()
        else:
            action_mean = action_mean.to(device=coarse.device, dtype=torch.float32)
            action_std = action_std.to(device=coarse.device, dtype=torch.float32)
            physical = coarse.detach().float() * (action_std[:, None] + 1e-6) + action_mean[:, None]
        xyz = physical[:, ids][:, :, self.wrist_indices]
        grid, valid = project_wrist(xyz, camera)
        context = self.aggregate(feature_map, grid, valid)
        gamma, beta = self.film(context.flatten(1).to(hidden.dtype)).chunk(2, -1)
        fused = hidden + gamma[:, None] * self.norm(hidden) + beta[:, None]
        fused = torch.where(valid.any((1, 2))[:, None, None], fused, hidden)
        debug.update(valid=valid, grid=grid, wrist_xyz_m=xyz,
                     valid_fraction=valid.float().mean())
        return fused, debug

    def aggregate(self, feature_map, grid, valid):
        local = sample_local(feature_map, grid, valid)
        return local.sum(1) / valid.sum(1).clamp_min(1)[..., None]
