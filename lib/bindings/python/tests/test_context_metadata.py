# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo._core import Context

pytestmark = [
    pytest.mark.parallel,
    pytest.mark.gpu_0,
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


def test_context_detached_uses_new_id_and_metadata_snapshot():
    ctx = Context(id="prefill", metadata={"tenant": "alpha", "region": "us-west"})
    ctx.stop_generating()

    detached = ctx.detached("decode")

    assert detached.id() == "decode"
    assert not detached.is_stopped()
    assert not detached.is_killed()
    assert detached.trace_headers() == ctx.trace_headers()
    assert dict(detached.metadata.items()) == {
        "region": "us-west",
        "tenant": "alpha",
    }

    ctx.metadata["tenant"] = "beta"
    detached.metadata["detached"] = "true"

    assert dict(ctx.metadata.items()) == {"region": "us-west", "tenant": "beta"}
    assert dict(detached.metadata.items()) == {
        "detached": "true",
        "region": "us-west",
        "tenant": "alpha",
    }
    detached.notify_first_token()
