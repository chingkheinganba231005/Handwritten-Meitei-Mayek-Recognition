import json
import cv2
import numpy as np
import pandas as pd
import torch

from mayek.data import Store, build_cache, load_index
from mayek.model import make_model
from mayek.recognizer import Recognizer, export
from mayek.split import make_split

from helpers import fake_zip, glyph


def test_split_is_deterministic(tmp_path):
    fake_zip(tmp_path / "t.zip", per_class=20)
    s1, summary = make_split(tmp_path / "t.zip", tmp_path / "a")
    s2, _ = make_split(tmp_path / "t.zip", tmp_path / "b")
    assert summary == {"classes": 3, "train": 51, "val": 9, "test": 6}
    assert pd.read_csv(s1 / "val.csv").path.tolist() == pd.read_csv(s2 / "val.csv").path.tolist()


def test_export_and_recognise(tmp_path):
    fake_zip(tmp_path / "t.zip")
    split_dir, _ = make_split(tmp_path / "t.zip", tmp_path / "data")
    index, labels, idx = load_index(split_dir)
    store = Store(build_cache(index.file.tolist(), 32, tmp_path / "cache", workers=1), "cpu", idx["train"])

    members = []
    for name, cfg in (("a", dict(arch="resnet18", channels="gray", drop_path=0.0, pretrained=False)),
                      ("b", dict(arch="resnet18", channels="topo", drop_path=0.0, pretrained=False, meta=True))):
        run = tmp_path / "runs" / name / "full"
        run.mkdir(parents=True)
        torch.save({"model": make_model(cfg, 55).state_dict(), "cfg": cfg}, run / "final.pt")
        members.append({"name": name, "cfg": cfg, "views": ["id", "dx2"], "weight": 0.5})
    out = export(members, tmp_path / "runs", tmp_path / "models", store.export_stats(), img=32)
    assert json.loads((out / "config.json").read_text())["members"][1]["cfg"]["meta"] is True

    rec = Recognizer(out)
    canvas = np.full((400, 400), 255, np.uint8)
    cv2.circle(canvas, (200, 200), 120, 0, 8)
    preds, seen = rec.predict(canvas, source="canvas", topk=5)
    assert len(preds) == 5 and seen.shape == (32, 32)
    assert all(0 <= p.prob <= 1 for p in preds) and preds[0].prob >= preds[-1].prob
    probs = rec.probs([glyph(0), glyph(1)])
    assert probs.shape == (2, 55) and np.allclose(probs.sum(1), 1, atol=1e-4)


def test_download_skips_the_certificate_check_only_when_it_fails(tmp_path, monkeypatch):
    import io
    import ssl
    import urllib.error
    import zipfile

    import pytest
    from mayek import split

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("a.txt", "x")
    contexts = []

    def bad_certificate(req, context=None, timeout=None):
        contexts.append(context)
        if context is None:
            raise urllib.error.URLError(ssl.SSLError("certificate verify failed"))
        return io.BytesIO(buf.getvalue())

    monkeypatch.setattr(split.urllib.request, "urlopen", bad_certificate)
    split.download("https://example.org/t.zip", tmp_path / "t.zip")
    assert zipfile.is_zipfile(tmp_path / "t.zip")
    assert contexts[0] is None and contexts[1] is not None

    def refused(req, context=None, timeout=None):
        raise urllib.error.URLError(ConnectionRefusedError())

    monkeypatch.setattr(split.urllib.request, "urlopen", refused)
    with pytest.raises(urllib.error.URLError):
        split.download("https://example.org/t.zip", tmp_path / "u.zip")
