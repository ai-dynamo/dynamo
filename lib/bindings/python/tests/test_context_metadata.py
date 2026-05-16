# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo._core import Context

pytestmark = [
    pytest.mark.parallel,
    pytest.mark.pre_merge,
    pytest.mark.unit,
]


def test_context_metadata_pop_mutates_live_mapping():
    ctx = Context(id="req-1", metadata={"tenant": "alpha", "region": "us-west"})

    popped = ctx.metadata.pop("tenant", None)

    assert popped == "alpha"
    assert "tenant" not in ctx.metadata
    assert dict(ctx.metadata.items()) == {"region": "us-west"}


def test_context_metadata_item_assignment_persists():
    ctx = Context(id="req-1", metadata={"tenant": "alpha"})

    ctx.metadata["tenant"] = "beta"
    ctx.metadata["region"] = "us-west"

    assert ctx.metadata["tenant"] == "beta"
    assert ctx.metadata.get("region") == "us-west"
    assert dict(ctx.metadata.items()) == {"region": "us-west", "tenant": "beta"}


def test_context_metadata_setter_replaces_mapping():
    ctx = Context(id="req-1", metadata={"tenant": "alpha"})

    ctx.metadata = {"region": "us-east"}

    assert "tenant" not in ctx.metadata
    assert dict(ctx.metadata.items()) == {"region": "us-east"}
