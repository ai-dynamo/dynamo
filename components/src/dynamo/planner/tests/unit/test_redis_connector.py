# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import AsyncMock, patch

import pytest

from dynamo.planner.config.defaults import SubComponentType, TargetReplica
from dynamo.planner.connectors.redis_connector import RedisConnector
from dynamo.planner.errors import EmptyTargetReplicasError

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


@pytest.fixture
def mock_redis_client():
    client = AsyncMock()
    client.hgetall = AsyncMock(return_value={})
    client.hset = AsyncMock()
    return client


@pytest.fixture
def connector(mock_redis_client):
    with patch(
        "dynamo.planner.connectors.redis_connector.redis_asyncio.from_url",
        return_value=mock_redis_client,
    ):
        return RedisConnector(
            "test-namespace",
            model_name="test-model",
            redis_url="redis://localhost:6379",
        )


def test_requires_model_name():
    with pytest.raises(ValueError, match="Model name is required"):
        RedisConnector(
            "test-namespace", model_name=None, redis_url="redis://localhost:6379"
        )


def test_requires_redis_url(monkeypatch):
    monkeypatch.delenv("DYN_REDIS_URL", raising=False)
    with pytest.raises(ValueError, match="redis_url is required"):
        RedisConnector("test-namespace", model_name="test-model", redis_url=None)


def test_redis_url_falls_back_to_env(mock_redis_client, monkeypatch):
    monkeypatch.setenv("DYN_REDIS_URL", "redis://from-env:6379")
    with patch(
        "dynamo.planner.connectors.redis_connector.redis_asyncio.from_url",
        return_value=mock_redis_client,
    ) as mock_from_url:
        RedisConnector("test-namespace", model_name="test-model")
        mock_from_url.assert_called_once_with(
            "redis://from-env:6379", decode_responses=True
        )


def test_key_uses_hash_tag_on_namespace_and_model_name(connector):
    assert connector._key == "dynamo:planner:target:{test-namespace:test-model}"


def test_key_isolates_same_model_name_across_namespaces(mock_redis_client):
    with patch(
        "dynamo.planner.connectors.redis_connector.redis_asyncio.from_url",
        return_value=mock_redis_client,
    ):
        a = RedisConnector(
            "namespace-a", model_name="shared-model", redis_url="redis://localhost:6379"
        )
        b = RedisConnector(
            "namespace-b", model_name="shared-model", redis_url="redis://localhost:6379"
        )
    assert a._key != b._key


def test_get_model_name_is_sync(connector):
    assert connector.get_model_name() == "test-model"


def test_model_name_case_is_preserved(mock_redis_client):
    """Unlike VirtualConnector/KubernetesConnector, this connector never
    matches model_name against MDC entries, so there's no reason to fold
    case -- and doing so risks merging two distinct, differently-cased
    model names onto the same Redis key."""
    with patch(
        "dynamo.planner.connectors.redis_connector.redis_asyncio.from_url",
        return_value=mock_redis_client,
    ):
        connector = RedisConnector(
            "test-namespace",
            model_name="Some-Mixed-Case-Model",
            redis_url="redis://localhost:6379",
        )
    assert connector.model_name == "Some-Mixed-Case-Model"
    assert connector.get_model_name() == "Some-Mixed-Case-Model"
    assert "{test-namespace:Some-Mixed-Case-Model}" in connector._key


def test_get_gpu_counts_returns_none_none(connector):
    assert connector.get_gpu_counts() == (None, None)


def test_get_worker_info_uses_defaults(connector):
    info = connector.get_worker_info(SubComponentType.PREFILL, backend="vllm")
    assert info.model_name == "test-model"


@pytest.mark.asyncio
async def test_set_component_replicas_disagg_writes_both_roles(
    connector, mock_redis_client
):
    await connector.set_component_replicas(
        [
            TargetReplica(
                sub_component_type=SubComponentType.PREFILL, desired_replicas=3
            ),
            TargetReplica(
                sub_component_type=SubComponentType.DECODE, desired_replicas=5
            ),
        ]
    )
    mock_redis_client.hset.assert_called_once()
    args, kwargs = mock_redis_client.hset.call_args
    assert args[0] == "dynamo:planner:target:{test-namespace:test-model}"
    mapping = kwargs["mapping"]
    assert mapping["prefill"] == 3
    assert mapping["decode"] == 5
    assert "updated_at" in mapping


@pytest.mark.asyncio
async def test_set_component_replicas_single_role_omits_the_other(
    connector, mock_redis_client
):
    """A dedicated PrefillPlanner only ever sends its own role -- the write
    must not clobber whatever a sibling DecodePlanner last wrote for
    "decode" under the same model_name key."""
    await connector.set_component_replicas(
        [TargetReplica(sub_component_type=SubComponentType.PREFILL, desired_replicas=4)]
    )
    mapping = mock_redis_client.hset.call_args.kwargs["mapping"]
    assert mapping["prefill"] == 4
    assert "decode" not in mapping


@pytest.mark.asyncio
async def test_set_component_replicas_empty_raises(connector):
    with pytest.raises(EmptyTargetReplicasError):
        await connector.set_component_replicas([])


@pytest.mark.asyncio
async def test_set_component_replicas_negative_raises(connector, mock_redis_client):
    with pytest.raises(ValueError, match="must not be negative"):
        await connector.set_component_replicas(
            [
                TargetReplica(
                    sub_component_type=SubComponentType.PREFILL, desired_replicas=-1
                )
            ]
        )
    mock_redis_client.hset.assert_not_called()


@pytest.mark.asyncio
async def test_add_component_increments_from_current(connector, mock_redis_client):
    mock_redis_client.hgetall.return_value = {"prefill": "2", "decode": "1"}
    await connector.add_component(SubComponentType.PREFILL)
    mapping = mock_redis_client.hset.call_args.kwargs["mapping"]
    assert mapping["prefill"] == 3


@pytest.mark.asyncio
async def test_remove_component_floors_at_zero(connector, mock_redis_client):
    mock_redis_client.hgetall.return_value = {"prefill": "0", "decode": "0"}
    await connector.remove_component(SubComponentType.DECODE)
    mapping = mock_redis_client.hset.call_args.kwargs["mapping"]
    assert mapping["decode"] == 0


@pytest.mark.asyncio
async def test_validate_deployment_and_wait_are_no_ops(connector):
    await connector.validate_deployment()
    await connector.wait_for_deployment_ready()


@pytest.mark.asyncio
async def test_read_desired_counts_negative_raises(connector, mock_redis_client):
    mock_redis_client.hgetall.return_value = {"prefill": "-1", "decode": "0"}
    with pytest.raises(ValueError, match="'prefill'.*must not be negative"):
        await connector.add_component(SubComponentType.PREFILL)


@pytest.mark.asyncio
async def test_read_desired_counts_invalid_raises(connector, mock_redis_client):
    mock_redis_client.hgetall.return_value = {"prefill": "not-a-number", "decode": "0"}
    with pytest.raises(ValueError, match="'prefill'.*not a valid integer"):
        await connector.add_component(SubComponentType.PREFILL)


class TestShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_closes_the_client(self, connector, mock_redis_client):
        mock_redis_client.aclose = AsyncMock()
        await connector.shutdown()
        mock_redis_client.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_shutdown_is_idempotent(self, connector, mock_redis_client):
        mock_redis_client.aclose = AsyncMock()
        await connector.shutdown()
        await connector.shutdown()
        mock_redis_client.aclose.assert_awaited_once()


class TestGetActualWorkerCounts:
    """get_actual_worker_counts reads observed state a companion process
    (not part of this repo) writes back into the same hash. If it hasn't
    published anything yet -- or never does -- missing fields must default
    to inactive/unstable, and a component whose name arg is None must
    report 0 and not count against the returned stable flag.
    """

    @pytest.mark.asyncio
    async def test_no_observed_fields_defaults_to_unstable(
        self, connector, mock_redis_client
    ):
        mock_redis_client.hgetall.return_value = {"prefill": "3", "decode": "5"}
        prefill, decode, stable = await connector.get_actual_worker_counts(
            prefill_component_name="prefill-worker",
            decode_component_name="decode-worker",
        )
        assert (prefill, decode, stable) == (0, 0, False)

    @pytest.mark.asyncio
    async def test_reads_observed_fields_when_present(
        self, connector, mock_redis_client
    ):
        mock_redis_client.hgetall.return_value = {
            "prefill_active": "3",
            "decode_active": "5",
            "prefill_stable": "true",
            "decode_stable": "true",
        }
        prefill, decode, stable = await connector.get_actual_worker_counts(
            prefill_component_name="prefill-worker",
            decode_component_name="decode-worker",
        )
        assert (prefill, decode, stable) == (3, 5, True)

    @pytest.mark.asyncio
    async def test_either_role_unstable_makes_the_whole_result_unstable(
        self, connector, mock_redis_client
    ):
        mock_redis_client.hgetall.return_value = {
            "prefill_active": "3",
            "decode_active": "5",
            "prefill_stable": "true",
            "decode_stable": "false",
        }
        _, _, stable = await connector.get_actual_worker_counts(
            prefill_component_name="prefill-worker",
            decode_component_name="decode-worker",
        )
        assert stable is False

    @pytest.mark.asyncio
    async def test_component_name_none_reports_zero_and_excluded_from_stable(
        self, connector, mock_redis_client
    ):
        """Decode isn't required by this planner mode (name arg is None):
        its observed values must not appear in the count or gate stability,
        even though the hash happens to carry stale/irrelevant decode data.
        """
        mock_redis_client.hgetall.return_value = {
            "prefill_active": "3",
            "prefill_stable": "true",
            "decode_active": "99",
            "decode_stable": "false",
        }
        prefill, decode, stable = await connector.get_actual_worker_counts(
            prefill_component_name="prefill-worker",
            decode_component_name=None,
        )
        assert (prefill, decode, stable) == (3, 0, True)

    @pytest.mark.asyncio
    async def test_negative_observed_value_raises(self, connector, mock_redis_client):
        mock_redis_client.hgetall.return_value = {
            "prefill_active": "-2",
            "prefill_stable": "true",
        }
        with pytest.raises(ValueError, match="'prefill_active'.*must not be negative"):
            await connector.get_actual_worker_counts(
                prefill_component_name="prefill-worker",
                decode_component_name=None,
            )
