import pytest
import torch

from helpers import fake_zip
from mayek import train as T
from mayek.data import Store, build_cache, load_index
from mayek.split import make_split


class Stop(Exception):
    pass


def test_training_resumes_after_a_crash(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "LOCAL", tmp_path / "local")
    monkeypatch.setattr(T, "DRIVE_EVERY", 1)
    fake_zip(tmp_path / "t.zip")
    split_dir, _ = make_split(tmp_path / "t.zip", tmp_path / "data")
    index, labels, idx = load_index(split_dir)
    store = Store(build_cache(index.file.tolist(), 32, tmp_path / "cache", workers=1), "cpu", idx["train"])
    cfg = dict(arch="resnet18", channels="gray", drop_path=0.0, pretrained=False, epochs=3, lr=1e-3, wd=0.0, batch=8)

    def crash_in_epoch_2(msg):
        if "epoch 2" in msg:
            raise Stop

    with pytest.raises(Stop):
        T.train("m", cfg, idx["train"], "dev", store, labels, tmp_path / "work", log=crash_in_epoch_2)
    assert (tmp_path / "work" / "runs" / "m" / "dev" / "last.pt").exists()

    logged = []
    model = T.train("m", cfg, idx["train"], "dev", store, labels, tmp_path / "work", log=logged.append)
    assert logged[0].endswith("resuming at epoch 2")
    ck = torch.load(tmp_path / "work" / "runs" / "m" / "dev" / "final.pt", weights_only=False)
    assert [h["epoch"] for h in ck["history"]] == [1, 2, 3]
    assert not (tmp_path / "work" / "runs" / "m" / "dev" / "last.pt").exists()
    assert model is not None


def test_safe_save_reports_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(T.time, "sleep", lambda s: None)
    blocked = tmp_path / "file"
    blocked.write_text("not a folder")
    assert T.safe_save({"a": 1}, blocked / "x.pt", tries=2) is not None
    assert T.safe_save({"a": 1}, tmp_path / "ok.pt") is None and (tmp_path / "ok.pt").exists()
