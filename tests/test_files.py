import os
import shutil

import numpy as np
import pytest

from mayek import files as F


@pytest.fixture(autouse=True)
def no_wait(monkeypatch):
    monkeypatch.setattr(F.time, "sleep", lambda s: None)


def test_copy_is_retried(tmp_path, monkeypatch):
    real, calls = shutil.copyfile, []

    def flaky(src, dst):
        calls.append(dst)
        if len(calls) == 1:
            raise OSError(95, "Operation not supported")
        return real(src, dst)

    monkeypatch.setattr(F.shutil, "copyfile", flaky)
    assert F.write_file(lambda f: np.save(f, np.arange(3)), tmp_path / "a.npy") is None
    assert np.load(tmp_path / "a.npy").tolist() == [0, 1, 2]
    assert len(calls) == 2 and not (tmp_path / "a.npy.tmp").exists()


def test_last_try_skips_the_rename(tmp_path, monkeypatch):
    def no_rename(src, dst):
        raise OSError(95, "Operation not supported")

    monkeypatch.setattr(F.os, "replace", no_rename)
    assert F.write_file(lambda f: open(f, "w").write("x"), tmp_path / "b.txt", tries=3) is None
    assert (tmp_path / "b.txt").read_text() == "x"


def test_failure_is_reported_and_nothing_is_left_behind(tmp_path):
    blocked = tmp_path / "file"
    blocked.write_text("not a folder")
    before = set(os.listdir(F.tempfile.gettempdir()))
    assert F.write_file(lambda f: open(f, "w").write("x"), blocked / "c.txt", tries=2) is not None
    assert set(os.listdir(F.tempfile.gettempdir())) <= before
    with pytest.raises(RuntimeError, match="Could not write"):
        F.must_write(lambda f: open(f, "w").write("x"), blocked / "c.txt", tries=2)
