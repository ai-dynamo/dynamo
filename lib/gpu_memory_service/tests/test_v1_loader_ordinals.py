# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from gpu_memory_service.v1.snapshot import loader

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def test_loader_separates_local_device_artifact_and_socket(monkeypatch):
    calls = []
    monkeypatch.setattr(loader, "init_vmm", lambda _: None)
    monkeypatch.setattr(
        loader, "get_socket_path", lambda device, domain: f"/{device}/{domain}"
    )
    monkeypatch.setattr(
        loader, "load_weights", lambda *args, **kw: calls.append((args, kw))
    )
    loader.main(
        [
            "--checkpoint-dir",
            "/capture",
            "--device",
            "0",
            "--socket-device",
            "7",
            "--artifact-device",
            "3",
        ]
    )
    assert calls[0][0] == ("/capture/device-3", "/7/weights", 0)


@pytest.mark.parametrize("flag", ["--device", "--socket-device", "--artifact-device"])
def test_negative_ordinal_rejected_before_cuda(flag):
    with pytest.raises(SystemExit) as error:
        loader.main(["--checkpoint-dir", "/capture", flag, "-1"])
    assert error.value.code == 2
