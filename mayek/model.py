"""The networks: an unmodified timm backbone with a new 55-way linear head.

The size-aware variants also feed five size numbers through a small MLP and
concatenate its output with the pooled image features before the head.
"""

import timm
import torch
import torch.nn as nn


CHANNELS = {"gray": 1, "topo": 3}  # "topo" = [ink, skeleton, distance transform]

# The three members of the ensemble, as trained for the paper.
MEMBERS = {
    "convnext_t": dict(arch="convnext_tiny.fb_in22k_ft_in1k", channels="gray",
                       epochs=60, lr=2e-4, wd=0.05, drop_path=0.1),
    "effv2_s": dict(arch="tf_efficientnetv2_s.in21k_ft_in1k", channels="gray",
                    epochs=60, lr=3e-4, wd=0.05, drop_path=0.1),
    "resnet50d_topo": dict(arch="resnet50d.ra2_in1k", channels="topo",
                           epochs=60, lr=5e-4, wd=0.05, drop_path=0.05),
}
META_DIM = 5


class Net(nn.Module):
    def __init__(self, arch, in_chans, num_classes, drop_path, meta_dim=0, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(arch, pretrained=pretrained, num_classes=0,
                                          in_chans=in_chans, drop_path_rate=drop_path)
        width = self.backbone.num_features
        self.meta = None
        if meta_dim:
            self.meta = nn.Sequential(nn.Linear(meta_dim, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU())
            width += 64
        self.head = nn.Linear(width, num_classes)

    def forward(self, x, meta=None):
        f = self.backbone(x)
        if self.meta is not None:
            f = torch.cat([f, self.meta(meta)], 1)
        return self.head(f)


def make_model(cfg, num_classes=55, pretrained=None, device="cpu"):
    """Build the network described by a member config (see MEMBERS)."""
    if pretrained is None:
        pretrained = cfg.get("pretrained", True)
    model = Net(cfg["arch"], CHANNELS[cfg["channels"]], num_classes, cfg["drop_path"],
                meta_dim=META_DIM if cfg.get("meta") else 0, pretrained=pretrained)
    return model.to(device).to(memory_format=torch.channels_last)
