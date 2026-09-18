# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import sys
import types
from pathlib import Path

import pytest
from dynamo.common.lora.runtime import (
    ResolveContext,
    ResolvedLoRA,
    RuntimeLoRAConfigurationError,
    RuntimeLoRANotFoundError,
    RuntimeLoRAPluginError,
    RuntimeLoRAResolverChain,
    load_runtime_lora_resolver,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


class Resolver:
    protocol_version = 2
    schemes = frozenset({"wandb-artifact"})

    def __init__(self, result: ResolvedLoRA | None = None):
        self.result = result
        self.calls: list[str] = []

    async def resolve(
        self, *, source_uri: str, context: ResolveContext
    ) -> ResolvedLoRA | None:
        self.calls.append(source_uri)
        return self.result


@pytest.fixture
def resolver_module(monkeypatch):
    module = types.ModuleType("test_runtime_lora_plugin")
    instance = Resolver()
    module.instance = instance
    module.Resolver = Resolver
    module.factory = lambda: Resolver()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module, instance


def test_loads_instance_class_and_factory_with_both_import_syntaxes(resolver_module):
    module, instance = resolver_module

    assert load_runtime_lora_resolver(f"{module.__name__}:instance") is instance
    assert isinstance(
        load_runtime_lora_resolver(f"{module.__name__}.Resolver"), Resolver
    )
    assert isinstance(
        load_runtime_lora_resolver(f"{module.__name__}:factory"), Resolver
    )


@pytest.mark.parametrize(
    "value,match",
    [
        (object(), "resolve"),
        (
            types.SimpleNamespace(
                protocol_version=1,
                schemes=frozenset({"wandb-artifact"}),
                resolve=Resolver.resolve,
            ),
            "protocol_version",
        ),
        (
            types.SimpleNamespace(
                protocol_version=2,
                schemes=frozenset({"WandB"}),
                resolve=Resolver.resolve,
            ),
            "scheme",
        ),
    ],
)
def test_invalid_resolver_contract_fails_startup(
    monkeypatch, resolver_module, value, match
):
    module, _ = resolver_module
    module.invalid = value
    with pytest.raises(RuntimeLoRAConfigurationError, match=match):
        load_runtime_lora_resolver(f"{module.__name__}:invalid")


@pytest.mark.asyncio
async def test_resolver_chain_falls_through_only_on_none(tmp_path):
    snapshot = tmp_path / "adapter"
    snapshot.mkdir()
    expected = ResolvedLoRA(snapshot, "sha256:immutable")
    first = Resolver()
    second = Resolver(expected)
    chain = RuntimeLoRAResolverChain([first, second], {"wandb-artifact"})

    deadline = asyncio.get_running_loop().time() + 1
    actual = await chain.resolve(
        source_uri="wandb-artifact:///entity/project/adapter:v1",
        context=ResolveContext(
            adapter_key="dyn-lora-0123456789abcdef0123456789abcdef",
            base_model_name="base",
            cache_root=tmp_path,
            deadline_monotonic=deadline,
            max_download_bytes=1024,
            request_id="request-id",
        ),
    )

    assert actual == expected
    assert first.calls == ["wandb-artifact:///entity/project/adapter:v1"]
    assert second.calls == ["wandb-artifact:///entity/project/adapter:v1"]


@pytest.mark.asyncio
async def test_resolver_chain_raises_typed_not_found(tmp_path):
    chain = RuntimeLoRAResolverChain([Resolver()], {"wandb-artifact"})
    deadline = asyncio.get_running_loop().time() + 1
    with pytest.raises(RuntimeLoRANotFoundError):
        await chain.resolve(
            source_uri="wandb-artifact:///missing:v1",
            context=ResolveContext(
                adapter_key="dyn-lora-0123456789abcdef0123456789abcdef",
                base_model_name="base",
                cache_root=tmp_path,
                deadline_monotonic=deadline,
                max_download_bytes=1024,
                request_id="request-id",
            ),
        )


@pytest.mark.asyncio
async def test_timeout_cancels_resolver(tmp_path):
    cancelled = asyncio.Event()

    class BlockingResolver(Resolver):
        async def resolve(self, *, source_uri, context):
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

    chain = RuntimeLoRAResolverChain([BlockingResolver()], {"wandb-artifact"})
    context = ResolveContext(
        adapter_key="dyn-lora-0123456789abcdef0123456789abcdef",
        base_model_name="base",
        cache_root=Path(tmp_path),
        deadline_monotonic=asyncio.get_running_loop().time() + 0.01,
        max_download_bytes=1024,
        request_id="request-id",
    )

    with pytest.raises(TimeoutError):
        await chain.resolve(
            source_uri="wandb-artifact:///entity/project/adapter:v1",
            context=context,
        )
    await asyncio.wait_for(cancelled.wait(), timeout=1)


@pytest.mark.asyncio
async def test_provider_exception_details_are_not_exposed(tmp_path):
    class FailingResolver(Resolver):
        async def resolve(self, *, source_uri, context):
            raise OSError("credential-for-private-provider")

    chain = RuntimeLoRAResolverChain([FailingResolver()], {"wandb-artifact"})
    context = ResolveContext(
        adapter_key="dyn-lora-0123456789abcdef0123456789abcdef",
        base_model_name="base",
        cache_root=tmp_path,
        deadline_monotonic=asyncio.get_running_loop().time() + 1,
        max_download_bytes=1024,
        request_id="request-id",
    )

    with pytest.raises(RuntimeLoRAPluginError) as exc_info:
        await chain.resolve(
            source_uri="wandb-artifact:///entity/project/adapter:v1",
            context=context,
        )

    assert str(exc_info.value) == "runtime LoRA resolver failed"


def test_allowlist_must_intersect_declared_schemes():
    with pytest.raises(RuntimeLoRAConfigurationError, match="allowed scheme"):
        RuntimeLoRAResolverChain([Resolver()], {"s3"})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result",
    [
        ResolvedLoRA(Path("adapter"), ""),
        ResolvedLoRA(Path("adapter"), "revision\nsecret"),
        ResolvedLoRA(Path("adapter"), "immutable", size_bytes=-1),
    ],
)
async def test_invalid_resolver_result_is_rejected(result, tmp_path):
    chain = RuntimeLoRAResolverChain(
        [Resolver(result)],
        {"wandb-artifact"},
    )
    context = ResolveContext(
        adapter_key="dyn-lora-0123456789abcdef0123456789abcdef",
        base_model_name="base",
        cache_root=tmp_path,
        deadline_monotonic=asyncio.get_running_loop().time() + 1,
        max_download_bytes=1024,
        request_id="request-id",
    )

    with pytest.raises(RuntimeLoRAPluginError):
        await chain.resolve(
            source_uri="wandb-artifact:///entity/project/adapter:v1",
            context=context,
        )
