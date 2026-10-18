import gzip

import numpy as np
import pytest

from kiseki.datasets.idx import read_idx


def test_reads_idx_images(tmp_path):
    images = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    header = bytes([0, 0, 0x08, 3]) + b"".join(n.to_bytes(4, "big") for n in images.shape)
    path = tmp_path / "x.gz"
    path.write_bytes(gzip.compress(header + images.tobytes()))
    assert np.array_equal(read_idx(path), images)


def test_rejects_other_idx_types(tmp_path):
    path = tmp_path / "x.gz"
    path.write_bytes(gzip.compress(bytes([0, 0, 0x0D, 1, 0, 0, 0, 1]) + b"\0\0\0\0"))
    with pytest.raises(ValueError):
        read_idx(path)
