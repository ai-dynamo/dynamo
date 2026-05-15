import pytest

from dynamo.nixl_connect import DeviceKind


def test_host_nixl_mem_type():
    assert DeviceKind.HOST.nixl_mem_type == "DRAM"


def test_cuda_nixl_mem_type():
    assert DeviceKind.CUDA.nixl_mem_type == "VRAM"


def test_str_unchanged():
    assert str(DeviceKind.HOST) == "cpu"
    assert str(DeviceKind.CUDA) == "cuda"


def test_unspecified_raises():
    with pytest.raises(ValueError):
        _ = DeviceKind.UNSPECIFIED.nixl_mem_type
