"""Tests run JAX on the CPU (set KISEKI_TEST_GPU=1 to use the GPU) against a scratch
kiseki home and config; datasets come from the real cache, and tests that need one skip
when it is missing (fetch with `kiseki datasets --fetch mnist`)."""

import os

if not os.environ.get("KISEKI_TEST_GPU"):
    os.environ["JAX_PLATFORMS"] = "cpu"

import pytest

from kiseki import datasets


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("KISEKI_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("KISEKI_CONFIG", str(tmp_path / "config"))
    return tmp_path / "home"


@pytest.fixture
def mnist():
    if not datasets.is_cached("mnist"):
        pytest.skip("MNIST is not cached")
    return "mnist"
