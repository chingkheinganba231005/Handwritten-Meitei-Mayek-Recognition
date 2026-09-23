"""Writing files to a mounted cloud drive (Google Drive in Colab).

Such mounts refuse a write now and then ("Operation not supported"), mostly
when a library opens the file itself with low-level calls. So every file is
made on local disk first and then copied over as plain bytes, retried a few
times, through a temporary name so a crash never leaves a half-written file.
"""

import os
import shutil
import tempfile
import time
from pathlib import Path


def write_file(write, path, tries=4, wait=5):
    """Create ``path`` with ``write(local_path)``. Returns the error if every copy failed."""
    path = Path(path)
    fd, local = tempfile.mkstemp(suffix=path.suffix)
    os.close(fd)
    try:
        write(local)
        err = None
        for k in range(tries):
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                if k == tries - 1 and tries > 1:  # last try: no rename, in case the mount rejects those
                    shutil.copyfile(local, path)
                else:
                    tmp = path.with_name(path.name + ".tmp")
                    shutil.copyfile(local, tmp)
                    os.replace(tmp, path)
                return None
            except OSError as e:
                err = e
                if k < tries - 1:
                    time.sleep(wait * (k + 1))
        return err
    finally:
        os.unlink(local)


def must_write(write, path, tries=6):
    """write_file that raises if the file could not be written."""
    err = write_file(write, path, tries)
    if err is not None:
        raise RuntimeError(f"Could not write {path} ({err}). If the drive is full, free some space "
                           "(and empty its bin), then run the cell again.") from err
    return Path(path)
