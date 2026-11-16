# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Media loaders must pay NIXL agent creation costs before their first request."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from dynamo import nixl_connect
from dynamo.common.multimodal import audio_loader, image_loader, video_loader

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


@pytest.fixture
def agent_factory(monkeypatch):
    api = MagicMock()
    monkeypatch.setattr(nixl_connect, "nixl_api", api)
    return api.nixl_agent


@pytest.fixture(
    params=[audio_loader, image_loader, video_loader], ids=["audio", "image", "video"]
)
def loader_factory(request, monkeypatch):
    module = request.param
    connector = nixl_connect.Connector()
    factory = MagicMock(return_value=connector)
    monkeypatch.setattr(module, "_create_nixl_connector", factory)
    loader_type = {
        audio_loader: audio_loader.AudioLoader,
        image_loader: image_loader.ImageLoader,
        video_loader: video_loader.VideoLoader,
    }[module]
    return loader_type, connector, factory


def test_loader_warms_connection_in_sync_context(loader_factory, agent_factory):
    loader_type, connector, _ = loader_factory
    loader_type(enable_frontend_decoding=True)
    agent_factory.assert_called_once()
    assert connector._shared_connection is not None


@pytest.mark.asyncio
async def test_loader_warms_connection_before_first_transfer(
    loader_factory, agent_factory
):
    # Loader construction uses run_async's temporary thread/loop in this context.
    loader_type, connector, _ = loader_factory
    loader_type(enable_frontend_decoding=True)
    agent_factory.assert_called_once()
    connection = connector._shared_connection
    assert connection is not None
    descriptor = nixl_connect.Descriptor(b"warmup")
    operation = await connector.create_readable(descriptor)
    assert operation.connection is connection
    await connector.warmup()
    agent_factory.assert_called_once()


def test_loader_without_frontend_decoding_keeps_nixl_lazy(
    loader_factory, agent_factory
):
    loader_type, _, factory = loader_factory
    loader_type(enable_frontend_decoding=False)
    factory.assert_not_called()
    agent_factory.assert_not_called()


def test_loader_fails_startup_if_warmup_fails(loader_factory, agent_factory):
    loader_type, connector, _ = loader_factory
    agent_factory.side_effect = RuntimeError("NIXL setup failed")
    with pytest.raises(RuntimeError, match="NIXL setup failed"):
        loader_type(enable_frontend_decoding=True)
    assert connector._shared_connection is None


@pytest.mark.asyncio
async def test_concurrent_warmup_and_transfer_share_one_connection(
    monkeypatch, agent_factory
):
    connector = nixl_connect.Connector()
    initialize = nixl_connect.Connection.initialize
    init = AsyncMock()

    async def yielding_initialize(connection):
        await asyncio.sleep(0)
        await init()
        await initialize(connection)

    monkeypatch.setattr(nixl_connect.Connection, "initialize", yielding_initialize)
    descriptor = nixl_connect.Descriptor(b"warmup")
    _, _, operation = await asyncio.gather(
        connector.warmup(), connector.warmup(), connector.create_readable(descriptor)
    )
    agent_factory.assert_called_once()
    assert operation.connection is connector._shared_connection
    init.assert_awaited_once()


@pytest.mark.asyncio
async def test_connector_remains_lazy_until_warmup(agent_factory):
    connector = nixl_connect.Connector()
    await connector.initialize()  # Preserve the deprecated no-op's behavior.
    agent_factory.assert_not_called()
    await connector.warmup()
    await connector.warmup()
    agent_factory.assert_called_once()


@pytest.mark.asyncio
async def test_failed_warmup_can_be_retried(agent_factory):
    connector = nixl_connect.Connector()
    agent = MagicMock()
    agent_factory.side_effect = [RuntimeError("NIXL setup failed"), agent]
    with pytest.raises(RuntimeError, match="NIXL setup failed"):
        await connector.warmup()
    assert connector._shared_connection is None
    await connector.warmup()
    assert connector._shared_connection._nixl is agent
    operation = await connector.create_readable(nixl_connect.Descriptor(b"warmup"))
    assert operation.connection is connector._shared_connection
    assert agent_factory.call_count == 2
