"""The browser demo: one network exported to ONNX plus the static page in ``web/``.

Everything runs in the visitor's browser with ONNX Runtime Web, so the demo
can be hosted for free as a static site (a static Hugging Face Space, GitHub
Pages, ...). The page reimplements the preprocessing of ``preprocess.py`` in
JavaScript.
"""

import io
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import torch

from .charset import CLASSES
from .files import must_write
from .model import CHANNELS

ORT_VERSION = "1.30.0"
ORT_CDN = f"https://cdn.jsdelivr.net/npm/onnxruntime-web@{ORT_VERSION}/dist/"
ORT_FILES = ("ort.wasm.min.js", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm")
WEB_SRC = Path(__file__).resolve().parents[1] / "web"


class _Forward(torch.nn.Module):
    """Wraps Net so the exported graph has plain positional inputs."""

    def __init__(self, net, meta):
        super().__init__()
        self.net, self.use_meta = net, meta

    def forward(self, image, meta=None):
        return self.net(image, meta if self.use_meta else None)


def export_onnx(model, cfg, path, img=128, half_weights=True):
    """Export a trained network; weights are stored as float16 and cast back to float32 on load."""
    path = Path(path)
    net = _Forward(model.eval().float().cpu(), bool(cfg.get("meta")))
    x = torch.zeros(2, CHANNELS[cfg["channels"]], img, img)
    args, names = (x,), ["image"]
    if cfg.get("meta"):
        args, names = (x, torch.zeros(2, 5)), ["image", "meta"]
    kwargs = dict(input_names=names, output_names=["logits"], opset_version=17,
                  dynamic_axes={n: {0: "batch"} for n in names + ["logits"]})

    def write(f):
        try:
            torch.onnx.export(net, args, f, dynamo=False, **kwargs)
        except TypeError:  # older torch without the dynamo switch
            torch.onnx.export(net, args, f, **kwargs)
        if half_weights:
            _store_half(f)

    return must_write(write, path)


def _store_half(path):
    """Halve the download: large float32 initializers become float16 + a Cast node."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    m = onnx.load(str(path))
    g = m.graph
    casts, halves = [], []
    for init in list(g.initializer):
        if init.data_type == TensorProto.FLOAT and int(np.prod(init.dims)) >= 256:
            half = numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16), init.name + "__fp16")
            halves.append(half)
            casts.append(helper.make_node("Cast", [half.name], [init.name], to=TensorProto.FLOAT))
            g.initializer.remove(init)
    g.initializer.extend(halves)
    nodes = casts + list(g.node)
    del g.node[:]
    g.node.extend(nodes)
    onnx.checker.check_model(m)
    onnx.save(m, str(path))


def fetch_ort(dest, tarball=None):
    """Copy the ONNX Runtime Web files the page needs (from npm, or a local tarball)."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if tarball is None:
        url = f"https://registry.npmjs.org/onnxruntime-web/-/onnxruntime-web-{ORT_VERSION}.tgz"
        data = urllib.request.urlopen(url, timeout=300).read()
    else:
        data = Path(tarball).read_bytes()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for name in ORT_FILES:
            (dest / name).write_bytes(tar.extractfile(f"package/dist/{name}").read())


def build_site(out_dir, cfg, stats, views, stroke_ratio, info, onnx_path=None, model_url="model.onnx",
               ort_tarball=None, img=128, model_mb=None):
    """Assemble the static site: page, config, and optionally the model and the runtime.

    stats: Store.export_stats(); info: shown on the page (model name, accuracy, links).
    model_url: where the page downloads the ONNX file; with onnx_path the file is
    copied next to the page. ort_tarball: bundle ONNX Runtime Web from this npm
    tarball instead of loading it from the jsDelivr CDN.
    """
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(WEB_SRC, out_dir)
    if onnx_path is not None:
        shutil.copy(onnx_path, out_dir / "model.onnx")
    ort_base = ORT_CDN
    if ort_tarball is not None:
        fetch_ort(out_dir / "ort", ort_tarball)
        ort_base = "ort/"
    ch = cfg["channels"]
    if ch != "gray":
        raise ValueError("the browser demo implements the ink channel only")
    config = {
        "img": img, "views": list(views), "stroke_ratio": float(stroke_ratio), "model_mb": model_mb,
        "mean": stats[ch]["mean"][0], "std": stats[ch]["std"][0],
        "meta": bool(cfg.get("meta")), "meta_mean": stats["meta"]["mean"], "meta_std": stats["meta"]["std"],
        "model_url": model_url, "ort_base": ort_base,
        "classes": [{"id": c.id, "char": c.char, "name": c.name} for c in CLASSES],
        **info,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=1, ensure_ascii=False))
    return out_dir
