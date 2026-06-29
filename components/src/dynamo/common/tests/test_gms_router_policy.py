# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo import gms_router_policy
from dynamo.gms_router_policy import (
    DynamoGmsPlacementPublisher,
    maybe_fetch_gms_placement,
)

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def test_router_policy_exposes_no_gms_control_endpoint():
    assert not hasattr(gms_router_policy, "GMS_CONTROL_ENDPOINT")
    assert not hasattr(gms_router_policy, "make_gms_control_handler")


@pytest.mark.asyncio
async def test_missing_local_gms_socket_leaves_request_on_vanilla_path():
    request = {
        "prompt": "hello",
        "gms_placement": {
            "source_nixl_agent_name": "source",
            "source_nixl_agent_metadata_hex": "00",
            "hashes": ["00"],
            "descriptors": [None],
        },
    }

    result = await maybe_fetch_gms_placement(request, None)

    assert result is None
    assert request["prompt"] == "hello"
    assert "gms_placement" in request


def test_router_gms_decode_transfer_arg_defaults_off():
    from dynamo.router.args import parse_args

    config = parse_args(["--endpoint", "ns.comp.generate"])

    assert config.router_gms_decode_transfer is False
    assert config.kv_router_kwargs()["router_gms_decode_transfer"] is False


def test_router_gms_decode_transfer_arg_can_opt_in():
    from dynamo.router.args import parse_args

    config = parse_args(
        ["--endpoint", "ns.comp.generate", "--router-gms-decode-transfer"]
    )

    assert config.router_gms_decode_transfer is True
    assert config.kv_router_kwargs()["router_gms_decode_transfer"] is True


class _FakeKvPublisher:
    def __init__(self):
        self.stored = []
        self.removed = []

    def publish_gms_placement_stored(self, *args, **kwargs):
        self.stored.append((args, kwargs))

    def publish_gms_placement_removed(self, *args, **kwargs):
        self.removed.append((args, kwargs))


def test_dynamo_gms_placement_publisher_normalizes_descriptor():
    fake = _FakeKvPublisher()
    publisher = DynamoGmsPlacementPublisher(fake, dp_rank=2)

    publisher.publish_stored(
        bytes.fromhex("aa" * 32),
        "host_pinned",
        64,
        metadata={
            "source_nixl_agent_name": "agent-a",
            "source_nixl_agent_metadata_hex": "abcd",
            "source_nixl_ip": "10.0.0.1",
            "source_nixl_listen_port": "5555",
            "gms_descriptor": {
                "ptr": "1234",
                "size": "64",
                "tier": "host",
                "generation": "7",
                "ranges": [
                    {
                        "ptr": "1234",
                        "size": "64",
                        "tier": "host",
                        "layer": "3",
                        "offset": "8",
                    }
                ],
            },
        },
    )

    assert len(fake.stored) == 1
    args, kwargs = fake.stored[0]
    assert args[0] == "agent-a"
    assert args[1] == "abcd"
    assert args[2][0]["content_hash_hex"] == "aa" * 32
    descriptor = args[2][0]["descriptor"]
    assert descriptor["remote_ptr"] == 1234
    assert descriptor["generation"] == 7
    assert descriptor["ranges"][0]["layer"] == 3
    assert kwargs == {
        "source_nixl_ip": "10.0.0.1",
        "source_nixl_listen_port": 5555,
        "dp_rank": 2,
    }


def test_dynamo_gms_placement_publisher_drops_missing_metadata():
    fake = _FakeKvPublisher()
    publisher = DynamoGmsPlacementPublisher(fake)

    publisher.publish_stored(
        bytes.fromhex("bb" * 32),
        "host_pinned",
        64,
        metadata={"gms_descriptor": {"remote_ptr": 1, "size": 64}},
    )

    assert fake.stored == []


def test_dynamo_gms_placement_publisher_removes_hash():
    fake = _FakeKvPublisher()
    publisher = DynamoGmsPlacementPublisher(fake, dp_rank=1)

    publisher.publish_removed(bytes.fromhex("cc" * 32), "host_pinned")

    assert fake.removed == [((["cc" * 32],), {"dp_rank": 1})]
