# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.llm import KvRouter, MooncakeClient


@pytest.fixture(scope="module", autouse=True)
def nats_and_etcd():
    """Client construction needs no discovery services."""
    yield


@pytest.mark.asyncio
async def test_policy_commands_are_exposed_only_on_router():
    control = MooncakeClient("http://localhost:9003")
    assert not hasattr(control, "pin")
    assert not hasattr(control, "unpin")
    for action in ("pin", "unpin"):
        for selection in ("blocks", "session", "session_frontier"):
            assert callable(getattr(KvRouter, f"{action}_{selection}"))
    with pytest.raises(Exception, match="expected HTTP"):
        MooncakeClient("file:///store")
