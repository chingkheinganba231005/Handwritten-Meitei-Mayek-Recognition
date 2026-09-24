"""Training augmentation and test-time views, both on the GPU.

No flips: a mirrored Meitei Mayek character is another character or none.
"""

import math

import torch
import torch.nn.functional as F

AUG = dict(rot=10, shear=0.2, scale=0.12, shift=0.05,   # degrees, tan(angle), +-fraction, +-fraction
           elastic=0.035, p_elastic=0.5,                 # smooth local warps
           p_thick=0.2, p_thin=0.1,                      # stroke width +-1 px
           gamma=0.35, contrast=0.25, p_blur=0.15, noise=0.03, p_noise=0.2,
           p_erase=0.1)                                  # small blank-out, kept rare on purpose

# Ablation settings. Disabled parts keep drawing their random numbers, so the
# remaining parts see the same random stream as in the full recipe.
AUG_PRESETS = {
    "full": AUG,
    "geometric": dict(AUG, p_thick=0.0, p_thin=0.0, gamma=0.0, contrast=0.0, p_blur=0.0, p_noise=0.0, p_erase=0.0),
    "none": None,
}

TTA_VIEWS = ["id", "s0.94", "s1.06", "dx2", "dx-2"]  # zoom in/out 6%, shift +-2 px, zero padding

_GAUSS = {}


def _gauss(device):
    if device not in _GAUSS:
        k = torch.tensor([1.0, 2.0, 1.0])
        _GAUSS[device] = (k[:, None] * k[None, :] / 16).view(1, 1, 3, 3).to(device)
    return _GAUSS[device]


def augment(x, aug=AUG):
    """Random affine + elastic warp on all channels, stroke/ink changes on channel 0."""
    if aug is None:
        return x
    B, C, H, W = x.shape
    dev = x.device

    def u(lo, hi):
        return torch.empty(B, device=dev).uniform_(lo, hi)

    def chance(p):
        return (torch.rand(B, device=dev) < p).float().view(B, 1, 1, 1)

    ang = u(-aug["rot"], aug["rot"]) * math.pi / 180
    sh = u(-aug["shear"], aug["shear"])
    sx, sy = u(1 - aug["scale"], 1 + aug["scale"]), u(1 - aug["scale"], 1 + aug["scale"])
    tx, ty = u(-aug["shift"], aug["shift"]) * 2, u(-aug["shift"], aug["shift"]) * 2
    cos, sin = ang.cos(), ang.sin()
    theta = torch.stack([torch.stack([cos * sx, (cos * sh - sin) * sy, tx], 1),
                         torch.stack([sin * sx, (sin * sh + cos) * sy, ty], 1)], 1)
    grid = F.affine_grid(theta, (B, C, H, W), align_corners=False)
    warp = torch.randn(B, 2, 5, 5, device=dev) * aug["elastic"] * chance(aug["p_elastic"])
    grid = grid + F.interpolate(warp, size=(H, W), mode="bicubic", align_corners=False).permute(0, 2, 3, 1)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    g = x[:, :1]
    pick = torch.rand(B, device=dev).view(B, 1, 1, 1)
    thick, thin = F.max_pool2d(g, 3, 1, 1), -F.max_pool2d(-g, 3, 1, 1)
    g = torch.where(pick < aug["p_thick"], thick, torch.where(pick > 1 - aug["p_thin"], thin, g))
    gamma = torch.exp(u(-aug["gamma"], aug["gamma"])).view(B, 1, 1, 1)
    g = g.clamp_min(1e-4) ** gamma * u(1 - aug["contrast"], 1.0).view(B, 1, 1, 1)
    blur = chance(aug["p_blur"])
    g = blur * F.conv2d(F.pad(g, (1, 1, 1, 1), mode="replicate"), _gauss(dev)) + (1 - blur) * g
    g = g + torch.randn_like(g) * aug["noise"] * chance(aug["p_noise"])
    x = torch.cat([g.clamp(0, 1), x[:, 1:]], 1)

    # small blank rectangle, rarely
    area, ratio = u(0.02, 0.06), torch.exp(u(-0.7, 0.7))
    eh, ew = (area * ratio).sqrt() * H, (area / ratio).sqrt() * W
    cy, cx = u(0, H), u(0, W)
    yy = torch.arange(H, device=dev).view(1, H, 1)
    xx = torch.arange(W, device=dev).view(1, 1, W)
    box = ((yy - cy.view(B, 1, 1)).abs() < eh.view(B, 1, 1) / 2) & ((xx - cx.view(B, 1, 1)).abs() < ew.view(B, 1, 1) / 2)
    return x * (1 - box.unsqueeze(1).float() * chance(aug["p_erase"]))


def view(x, v):
    """Deterministic test-time view: 'id', 's<scale>', 'dx<px>' or 'dy<px>'."""
    if v == "id":
        return x
    B, C, H, W = x.shape
    s, tx, ty = 1.0, 0.0, 0.0
    if v.startswith("s"):
        s = float(v[1:])
    elif v.startswith("dx"):
        tx = -float(v[2:]) * 2 / W
    elif v.startswith("dy"):
        ty = -float(v[2:]) * 2 / H
    theta = torch.tensor([[s, 0, tx], [0, s, ty]], device=x.device, dtype=x.dtype).expand(B, 2, 3)
    grid = F.affine_grid(theta, (B, C, H, W), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
