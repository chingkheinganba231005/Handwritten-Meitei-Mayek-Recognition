import numpy as np
import torch

from mayek.augment import AUG_PRESETS, TTA_VIEWS, augment, view
from mayek.ensemble import apply_specialists, combine, confusions, fit_weights, wilson
from mayek.model import make_model

TINY = dict(arch="resnet18", channels="gray", drop_path=0.0, pretrained=False)


def test_forward_gray_topo_meta():
    for cfg, c in ((TINY, 1), (dict(TINY, channels="topo"), 3), (dict(TINY, meta=True), 1)):
        model = make_model(cfg, 55).eval()
        x = torch.rand(2, c, 64, 64)
        meta = torch.zeros(2, 5) if cfg.get("meta") else None
        assert model(x, meta).shape == (2, 55)


def test_augment_keeps_shape_and_range():
    torch.manual_seed(0)
    x = torch.rand(8, 3, 32, 32)
    for preset in AUG_PRESETS.values():
        y = augment(x, preset)
        assert y.shape == x.shape and y.min() >= 0 and y.max() <= 1


def test_geometric_preset_draws_the_same_geometry():
    x = torch.rand(4, 1, 32, 32)
    torch.manual_seed(1)
    full = augment(x, AUG_PRESETS["full"])
    torch.manual_seed(1)
    geo = augment(x, AUG_PRESETS["geometric"])
    assert full.shape == geo.shape and not torch.equal(full, geo)


def test_views():
    x = torch.rand(2, 1, 32, 32)
    assert torch.equal(view(x, "id"), x)
    for v in TTA_VIEWS:
        assert view(x, v).shape == x.shape


def test_ensemble_helpers():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 200)
    good = np.eye(5)[y] * 0.8 + 0.04
    bad = rng.dirichlet(np.ones(5), 200)
    w = fit_weights([good, bad], y)
    assert np.isclose(w.sum(), 1) and w[0] > w[1]
    assert np.allclose(combine([good, good]), good)
    assert np.allclose(apply_specialists(good, {(0, 1): np.full(200, 0.5)}, 0), good)
    lo, hi = wilson(95, 100)
    assert lo < 0.95 < hi
    assert confusions(good, y) == []
