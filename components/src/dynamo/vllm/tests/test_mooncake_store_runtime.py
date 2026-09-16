# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import copy
import hashlib
import json
import sys
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from dynamo.vllm import mooncake_store_runtime as runtime

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.router,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.fixture
def resolved_store(monkeypatch):
    @dataclass
    class FullAttentionSpec:
        block_size: int = 16
        sliding_window: int | None = None
        attention_chunk_size: int | None = None
        non_causal: bool = False

    @dataclass
    class SlidingWindowSpec:
        block_size: int = 16
        sliding_window: int = 32
        extra_retained_tokens: int = 0

    @dataclass
    class MambaSpec:
        block_size: int = 16
        mamba_cache_mode: str = "align"
        num_speculative_blocks: int = 0

    @dataclass
    class UniformTypeKVCacheSpecs:
        block_size: int = 16
        kv_cache_specs: dict = field(default_factory=dict)

    class MooncakeStoreConnector:
        pass

    class MooncakeStoreWorker:
        pass

    class MultiConnector:
        def __init__(self, connectors):
            self.sub_connectors = connectors

    class MooncakeStoreCoordinator:
        pass

    class ChunkedTokenDatabase:
        pass

    class FullAttentionManager:
        pass

    class SlidingWindowManager:
        pass

    class MambaManager:
        pass

    specs = SimpleNamespace(
        FullAttentionSpec=FullAttentionSpec,
        SlidingWindowSpec=SlidingWindowSpec,
        MambaSpec=MambaSpec,
        UniformTypeKVCacheSpecs=UniformTypeKVCacheSpecs,
    )
    managers = SimpleNamespace(
        FullAttentionManager=FullAttentionManager,
        SlidingWindowManager=SlidingWindowManager,
        MambaManager=MambaManager,
    )
    modules = {
        runtime._SPEC_MODULE: specs,
        runtime._MANAGER_MODULE: managers,
        runtime._STORE_MODULE + ".connector": SimpleNamespace(
            MooncakeStoreConnector=MooncakeStoreConnector
        ),
        runtime._STORE_MODULE + ".worker": SimpleNamespace(
            MooncakeStoreWorker=MooncakeStoreWorker
        ),
        runtime._STORE_MODULE + ".coordinator": SimpleNamespace(
            MooncakeStoreCoordinator=MooncakeStoreCoordinator
        ),
        runtime._STORE_MODULE + ".data": SimpleNamespace(
            ChunkedTokenDatabase=ChunkedTokenDatabase
        ),
        "vllm.distributed.kv_transfer.kv_connector.v1.multi_connector": SimpleNamespace(
            MultiConnector=MultiConnector
        ),
        "vllm.envs": SimpleNamespace(VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=True),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            is_multimodal_model=False,
            enable_prompt_embeds=False,
            is_encoder_decoder=False,
            runner_type="generate",
        ),
        cache_config=SimpleNamespace(
            enable_prefix_caching=True, prefix_caching_hash_algo="sha256"
        ),
        kv_events_config=SimpleNamespace(enable_kv_cache_events=True),
        scheduler_config=SimpleNamespace(scheduler_cls=None),
        speculative_config=None,
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            worker_extension_cls=runtime.WORKER_EXTENSION,
            world_size=2,
            rank=0,
            data_parallel_index=0,
        ),
    )

    def build(group_specs=None, *, spans=None, group_ids=None):
        if group_specs is None:
            group_specs = [FullAttentionSpec(), SlidingWindowSpec()]
        if spans is None:
            spans = [16] * len(group_specs)
        if group_ids is None:
            group_ids = tuple(range(len(group_specs)))
        worker = MooncakeStoreWorker()
        worker._capacity_only = False
        worker.pcp_size = worker.dcp_size = 1
        worker.hash_block_size = min(spans)
        worker.block_size = max(spans)
        groups = [
            SimpleNamespace(kv_cache_spec=spec, is_eagle_group=False)
            for spec in group_specs
        ]
        original = [SimpleNamespace(kv_cache_spec=None)] * (max(group_ids) + 1)
        for idx, group in zip(group_ids, groups):
            original[idx] = group
        worker._kv_cache_config = SimpleNamespace(
            kv_cache_groups=original, prefix_cacheable_group_ids=group_ids
        )
        worker._kv_cache_groups = groups
        worker.coord = MooncakeStoreCoordinator()
        coord = worker.coord
        coord.kv_cache_groups = groups
        coord.use_eagle = False
        coord.eagle_group_ids = set()
        coord.hash_block_size = worker.hash_block_size
        coord.lcm_block_size = worker.block_size
        coord.enable_partial_hash_hits = False
        coord.attention_groups = []
        worker.token_dbs = []
        worker._lookup_key_prefixes = []
        for idx, (spec, span) in enumerate(zip(group_specs, spans)):
            concrete = (
                next(iter(spec.kv_cache_specs.values()))
                if isinstance(spec, UniformTypeKVCacheSpecs)
                else spec
            )
            manager_name = type(concrete).__name__.replace("Spec", "Manager")
            coord.attention_groups.append(
                SimpleNamespace(
                    spec=concrete,
                    group_ids=[idx],
                    manager_cls=getattr(managers, manager_name),
                    use_eagle=False,
                )
            )
            db = ChunkedTokenDatabase()
            db.block_size = span
            db.hash_block_size = worker.hash_block_size
            db.metadata = SimpleNamespace(group_id=idx, cache_prefix="deployment")
            worker.token_dbs.append(db)
            worker._lookup_key_prefixes.append(
                tuple(f"deployment@model@tp_rank:{rank}@group:{idx}" for rank in (1, 0))
            )
        connector = MooncakeStoreConnector()
        connector.connector_worker = worker
        connector._kv_transfer_config = SimpleNamespace(kv_connector_extra_config={})
        return connector

    return SimpleNamespace(
        build=build, config=config, specs=specs, managers=managers, multi=MultiConnector
    )


def test_direct_and_multi_export_exact_resolved_prefixes(resolved_store):
    connector = resolved_store.build(group_ids=(1, 2))
    direct = runtime._export_descriptor(connector, resolved_store.config)
    multi = runtime._export_descriptor(
        resolved_store.multi([object(), connector]), resolved_store.config
    )
    assert direct == multi
    assert direct["main_event_group"] == 1
    assert direct["gpu_to_store_group"] == [None, 0, 1]
    assert direct["groups"][0]["key_prefixes"] == sorted(
        connector.connector_worker._lookup_key_prefixes[0]
    )
    assert direct["groups"][1]["manager"] == "SlidingWindowManager"
    assert json.loads(json.dumps(direct)) == direct


def test_parallel_mamba_export(resolved_store):
    connector = resolved_store.build(
        [resolved_store.specs.FullAttentionSpec(), resolved_store.specs.MambaSpec()]
    )
    descriptor = runtime._export_descriptor(connector, resolved_store.config)
    assert descriptor["groups"][1]["kind"] == "mamba"
    assert descriptor["groups"][1]["mamba_cache_mode"] == "align"
    assert len(descriptor["groups"][1]["key_prefixes"]) == 2


def test_uniform_group_checks_every_inner_spec(resolved_store):
    specs = resolved_store.specs
    uniform = specs.UniformTypeKVCacheSpecs(
        kv_cache_specs={"a": specs.FullAttentionSpec(), "b": specs.FullAttentionSpec()}
    )
    connector = resolved_store.build([uniform])
    assert (
        runtime._export_descriptor(connector, resolved_store.config)["groups"][0][
            "spec"
        ]
        == "FullAttentionSpec"
    )
    uniform.kv_cache_specs["b"] = specs.MambaSpec()
    with pytest.raises(runtime.UnsupportedContract, match="mixed uniform"):
        runtime._export_descriptor(connector, resolved_store.config)


@pytest.mark.parametrize(
    "mutation",
    [
        "no_prefix",
        "duplicate_prefix",
        "hash_span",
        "manager",
        "eagle",
        "capacity",
        "lookup_off",
        "missing_shape",
    ],
)
def test_unsupported_worker_shapes(resolved_store, mutation):
    connector = resolved_store.build()
    worker = connector.connector_worker
    if mutation == "no_prefix":
        worker.token_dbs[0].metadata.cache_prefix = ""
    elif mutation == "duplicate_prefix":
        worker._lookup_key_prefixes[1] = worker._lookup_key_prefixes[0]
    elif mutation == "hash_span":
        worker.token_dbs[1].hash_block_size = 8
    elif mutation == "manager":
        worker.coord.attention_groups[1].manager_cls = object
    elif mutation == "eagle":
        worker.coord.use_eagle = True
    elif mutation == "capacity":
        worker._capacity_only = True
    elif mutation == "lookup_off":
        connector._kv_transfer_config.kv_connector_extra_config["enable_lookup"] = False
    else:
        del worker._lookup_key_prefixes
    with pytest.raises((runtime.UnsupportedContract, AttributeError)):
        runtime._export_descriptor(connector, resolved_store.config)


def test_ambiguous_store_connector_is_not_selected(resolved_store):
    with pytest.raises(runtime.UnsupportedContract, match="exactly one"):
        runtime._export_descriptor(
            resolved_store.multi([resolved_store.build(), resolved_store.build()]),
            resolved_store.config,
        )


@pytest.mark.parametrize(
    ("span", "window", "supported"), [(8, 8, True), (8, 16, False), (32, 64, True)]
)
def test_observable_sliding_window_geometry(resolved_store, span, window, supported):
    specs = resolved_store.specs
    connector = resolved_store.build(
        [
            specs.FullAttentionSpec(block_size=32),
            specs.SlidingWindowSpec(block_size=span, sliding_window=window),
        ],
        spans=[32, span],
    )
    if supported:
        assert runtime._export_descriptor(connector, resolved_store.config)
    else:
        with pytest.raises(runtime.UnsupportedContract, match="unobservable"):
            runtime._export_descriptor(connector, resolved_store.config)


@pytest.mark.parametrize(
    ("section", "field_name", "value"),
    [
        ("model_config", "is_multimodal_model", True),
        ("model_config", "enable_prompt_embeds", True),
        ("model_config", "is_encoder_decoder", True),
        ("cache_config", "prefix_caching_hash_algo", "sha256_cbor"),
        ("cache_config", "enable_prefix_caching", False),
        ("kv_events_config", "enable_kv_cache_events", False),
        ("parallel_config", "decode_context_parallel_size", 2),
        ("scheduler_config", "scheduler_cls", "unknown.Scheduler"),
    ],
)
def test_input_config_fails_closed(resolved_store, section, field_name, value):
    setattr(getattr(resolved_store.config, section), field_name, value)
    with pytest.raises(runtime.UnsupportedContract):
        runtime._validate_input_config(resolved_store.config)


def test_seed_and_speculative_policy(resolved_store, monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "9")
    with pytest.raises(runtime.UnsupportedContract, match="PYTHONHASHSEED"):
        runtime._validate_input_config(resolved_store.config)
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    resolved_store.config.speculative_config = SimpleNamespace()
    with pytest.raises(runtime.UnsupportedContract, match="speculative"):
        runtime._validate_input_config(resolved_store.config)


def test_named_rpc_preserves_serving_on_unrecognized_revision(
    resolved_store, monkeypatch
):
    monkeypatch.setattr(
        runtime,
        "_verify_pinned_sources",
        Mock(side_effect=runtime.UnsupportedContract("unknown revision")),
    )
    extension = runtime.MooncakeStoreWorkerExtension()
    extension.vllm_config = resolved_store.config
    assert extension.dynamo_mooncake_store_descriptor() == {
        "unsupported": "unknown revision"
    }


@pytest.fixture
def publication(resolved_store, monkeypatch):
    monkeypatch.setattr(runtime, "_verify_pinned_sources", lambda: None)
    descriptor = runtime._export_descriptor(
        resolved_store.build(), resolved_store.config
    )
    replies = [
        {"rank": rank, "dp_rank": 0, "descriptor": copy.deepcopy(descriptor)}
        for rank in (0, 1)
    ]
    engine = SimpleNamespace(collective_rpc=AsyncMock(return_value=replies))
    metadata = SimpleNamespace(set_engine_specific=Mock())

    def publish(**kwargs):
        options = {"dp_range": (0, 1), "event_span": 16}
        options.update(kwargs)
        return asyncio.run(
            runtime.publish_mooncake_store_runtime(
                metadata, engine, resolved_store.config, **options
            )
        )

    return SimpleNamespace(
        descriptor=descriptor,
        replies=replies,
        engine=engine,
        metadata=metadata,
        publish=publish,
        config=resolved_store.config,
    )


def test_publish_agrees_all_ranks_and_uses_named_bounded_rpc(publication):
    publication.replies.reverse()
    assert publication.publish()
    publication.engine.collective_rpc.assert_awaited_once_with(
        runtime.RPC_METHOD, timeout=runtime.RPC_TIMEOUT_SECONDS
    )
    key, value = publication.metadata.set_engine_specific.call_args.args
    assert key == runtime.RUNTIME_KEY
    assert json.loads(value) == publication.descriptor


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate",
        "wrong_dp",
        "different",
        "unsupported",
        "missing_field",
        "span",
        "multi_dp",
        "timeout",
    ],
)
def test_publication_withholds_incomplete_contract(publication, mutation):
    options = {}
    if mutation == "missing":
        publication.replies.pop()
    elif mutation == "duplicate":
        publication.replies[1]["rank"] = 0
    elif mutation == "wrong_dp":
        publication.replies[1]["dp_rank"] = 1
    elif mutation == "different":
        publication.replies[1]["descriptor"]["groups"][0]["key_prefixes"][0] += "other"
    elif mutation == "unsupported":
        publication.replies[1] = {"unsupported": "shape"}
    elif mutation == "missing_field":
        del publication.replies[1]["descriptor"]["hash"]
    elif mutation == "span":
        options["event_span"] = 32
    elif mutation == "multi_dp":
        options["dp_range"] = (0, 2)
    else:
        publication.engine.collective_rpc.side_effect = TimeoutError
    assert publication.publish(**options) is False
    publication.metadata.set_engine_specific.assert_not_called()


def test_extension_is_not_auto_enabled(publication):
    publication.config.parallel_config.worker_extension_cls = ""
    assert publication.publish() is False
    publication.engine.collective_rpc.assert_not_called()


def test_outer_rpc_timeout_cancels_wait_and_preserves_serving(publication, monkeypatch):
    cancelled = []

    async def blocked_rpc(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)

    publication.engine.collective_rpc.side_effect = blocked_rpc
    monkeypatch.setattr(runtime, "RPC_TIMEOUT_SECONDS", 0.01)
    assert publication.publish() is False
    assert cancelled == [True]
    publication.metadata.set_engine_specific.assert_not_called()


def test_publication_does_not_swallow_shutdown_cancellation(publication):
    publication.engine.collective_rpc.side_effect = asyncio.CancelledError
    with pytest.raises(asyncio.CancelledError):
        publication.publish()
    publication.metadata.set_engine_specific.assert_not_called()


def test_missing_optional_extension_attribute_preserves_older_engine(publication):
    del publication.config.parallel_config.worker_extension_cls
    assert publication.publish() is False
    publication.engine.collective_rpc.assert_not_called()


def test_pinned_revision_and_private_source_fingerprints(tmp_path, monkeypatch):
    content = b"pinned private source\n"
    (tmp_path / "private.py").write_bytes(content)
    monkeypatch.setattr(
        runtime, "_PINNED_FILES", (("private.py", hashlib.sha256(content).hexdigest()),)
    )
    distribution = SimpleNamespace(
        version="0.29.0.dev1+g1085b644", read_text=lambda _: None
    )
    monkeypatch.setattr(
        runtime.importlib.metadata, "distribution", lambda _: distribution
    )
    monkeypatch.setitem(
        sys.modules, "vllm", SimpleNamespace(__file__=str(tmp_path / "__init__.py"))
    )
    runtime._verify_pinned_sources()
    (tmp_path / "private.py").write_bytes(b"patched source\n")
    with pytest.raises(runtime.UnsupportedContract, match="sources differ"):
        runtime._verify_pinned_sources()
    distribution.version = "0.29.0"
    with pytest.raises(runtime.UnsupportedContract, match="pinned revision"):
        runtime._verify_pinned_sources()
