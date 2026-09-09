# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from tests.utils import port_utils

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def test_reserved_ports_releases_ports_after_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    released: list[list[int]] = []
    monkeypatch.setattr(port_utils, "allocate_ports", lambda count, start_port: [12000])
    monkeypatch.setattr(port_utils, "deallocate_ports", released.append)

    with pytest.raises(RuntimeError, match="startup failed"):
        with port_utils.reserved_ports(count=1, start_port=12000):
            raise RuntimeError("startup failed")

    assert released == [[12000]]
