import json

import numpy as np
import onnxruntime as ort
import torch

from mayek.model import make_model
from mayek.web import ORT_CDN, build_site, export_onnx

STATS = {"gray": {"mean": [0.2], "std": [0.3]}, "topo": {"mean": [0.2, 0.1, 0.1], "std": [0.3, 0.2, 0.2]},
         "meta": {"mean": [3.1, 3.1, 2.9, 2.9, 0.2], "std": [0.1, 0.1, 0.2, 0.2, 0.05]}}


def calibrated(cfg):
    """A random network with batch-norm statistics set, so it is numerically well behaved."""
    torch.manual_seed(0)
    model = make_model(cfg, 55, pretrained=False).train()
    with torch.no_grad():
        for _ in range(5):
            model(torch.rand(16, 1, 64, 64), torch.randn(16, 5) if cfg.get("meta") else None)
    return model.eval()


def test_onnx_matches_torch(tmp_path):
    for cfg in (dict(arch="resnet18", channels="gray", drop_path=0.0),
                dict(arch="resnet18", channels="gray", drop_path=0.0, meta=True)):
        model = calibrated(cfg)
        x, meta = torch.rand(3, 1, 64, 64), torch.randn(3, 5)
        with torch.no_grad():
            ref = model(x, meta if cfg.get("meta") else None).numpy()
        path = export_onnx(model, cfg, tmp_path / "m.onnx", img=64)
        feeds = {"image": x.numpy()}
        if cfg.get("meta"):
            feeds["meta"] = meta.numpy()
        out = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, feeds)[0]
        assert np.abs(out - ref).max() < 1e-2 and (out.argmax(1) == ref.argmax(1)).all()


def test_build_site(tmp_path):
    cfg = dict(arch="resnet18", channels="gray", drop_path=0.0)
    site = build_site(tmp_path / "site", cfg, STATS, ["id", "dx2"], 0.08, {"model_name": "x"}, img=64)
    config = json.loads((site / "config.json").read_text())
    assert config["ort_base"] == ORT_CDN and config["img"] == 64 and len(config["classes"]) == 55
    assert {"index.html", "app.js", "pipeline.js", "style.css"} <= {p.name for p in site.iterdir()}
