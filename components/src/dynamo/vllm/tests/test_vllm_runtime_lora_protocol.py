# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from types import SimpleNamespace

import pytest

from dynamo.llm import WorkerType
from dynamo.vllm import runtime_lora_protocol as protocol_mod
from dynamo.vllm.runtime_lora_protocol import publish_runtime_lora_capability

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.mark.parametrize(
    "request_payload",
    [
        {},
        {"routing": None},
        {"routing": []},
        {"routing": {"lora_name": "known-adapter"}},
    ],
)
def test_explicit_requests_do_not_use_runtime_protocol(request_payload):
    assert protocol_mod._parse_request(request_payload) is None


@pytest.mark.parametrize(
    "source",
    [
        "",
        "custom:",
        "1custom:adapter",
        "custom",
        "custom:\x00bad",
        "custom:\x7fbad",
        "custom:\ud800",
        "custom:" + "x" * 3072,
    ],
)
def test_invalid_source_envelopes_are_rejected(source):
    with pytest.raises(protocol_mod.HttpError, match="invalid_lora_model_id"):
        protocol_mod._validate_source_uri(source)


@pytest.mark.parametrize(
    "field,value",
    [
        ("lora_name", "wrong"),
        ("base_model_name", None),
        ("base_model_name", ""),
        ("base_model_name", "b" * 513),
        ("lora_source_uri", None),
        ("lora_resolution_version", True),
        ("lora_resolution_version", 1),
    ],
)
def test_incomplete_or_invalid_runtime_metadata_is_rejected(field, value):
    request = _request("custom:adapter")
    request = {**request, "routing": {**request["routing"], field: value}}
    with pytest.raises(protocol_mod.HttpError, match="invalid_lora_model_id"):
        protocol_mod._parse_request(request)


def test_runtime_metadata_must_match_model_and_identity():
    request = _request("custom:adapter")
    with pytest.raises(protocol_mod.HttpError, match="invalid_lora_model_id"):
        protocol_mod._parse_request({**request, "model": "different-base"})
    with pytest.raises(protocol_mod.HttpError, match="runtime_lora_identity_mismatch"):
        protocol_mod._parse_request(
            {
                **request,
                "routing": {**request["routing"], "lora_name": "dyn-lora-" + "0" * 32},
            }
        )


@pytest.mark.parametrize("value", ["bad", "0", "-1"])
def test_runtime_settings_reject_invalid_positive_limits(monkeypatch, value):
    monkeypatch.setenv("DYN_LORA_MAX_PENDING_ADMISSIONS", value)
    with pytest.raises(
        protocol_mod.RuntimeLoRAConfigurationError,
        match="DYN_LORA_MAX_PENDING_ADMISSIONS",
    ):
        protocol_mod.RuntimeLoRASettings.from_engine_args(SimpleNamespace())


@pytest.mark.parametrize(
    "engine_args",
    [
        SimpleNamespace(max_loras=0),
        SimpleNamespace(max_loras=-1),
        SimpleNamespace(max_loras=None),
        SimpleNamespace(max_cpu_loras=-1),
    ],
)
def test_runtime_settings_reject_invalid_engine_capacity(engine_args):
    with pytest.raises(protocol_mod.RuntimeLoRAConfigurationError):
        protocol_mod.RuntimeLoRASettings.from_engine_args(engine_args)


def test_runtime_download_budget_cannot_exceed_cache(monkeypatch):
    monkeypatch.setenv("DYN_LORA_MAX_DOWNLOAD_BYTES", "20")
    monkeypatch.setenv("DYN_LORA_MAX_CACHE_BYTES", "10")
    with pytest.raises(
        protocol_mod.RuntimeLoRAConfigurationError, match="cannot exceed"
    ):
        protocol_mod.RuntimeLoRASettings.from_engine_args(SimpleNamespace())


def test_disabled_capability_does_not_load_plugin(monkeypatch):
    monkeypatch.setenv("DYN_LORA_RUNTIME_LOAD_ENABLED", "false")

    def reject_plugin_load(**_kwargs):
        pytest.fail("disabled capability must not initialize the plugin")

    monkeypatch.setattr(protocol_mod, "get_lora_manager", reject_plugin_load)
    publish_runtime_lora_capability(
        SimpleNamespace(), SimpleNamespace(), WorkerType.Aggregated
    )


@pytest.mark.parametrize(
    "lora_enabled,engine_enabled", [("false", True), ("true", False)]
)
def test_capability_requires_both_lora_switches(
    monkeypatch, lora_enabled, engine_enabled
):
    monkeypatch.setenv("DYN_LORA_RUNTIME_LOAD_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_ENABLED", lora_enabled)
    with pytest.raises(
        protocol_mod.RuntimeLoRAConfigurationError, match="requires vLLM LoRA support"
    ):
        publish_runtime_lora_capability(
            SimpleNamespace(),
            SimpleNamespace(engine_args=SimpleNamespace(enable_lora=engine_enabled)),
            WorkerType.Aggregated,
        )


def test_capability_rejects_disaggregated_worker(monkeypatch):
    monkeypatch.setenv("DYN_LORA_RUNTIME_LOAD_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_ENABLED", "true")
    with pytest.raises(
        protocol_mod.RuntimeLoRAConfigurationError, match="aggregated workers only"
    ):
        publish_runtime_lora_capability(
            SimpleNamespace(),
            SimpleNamespace(engine_args=SimpleNamespace(enable_lora=True)),
            WorkerType.Prefill,
        )


@pytest.mark.parametrize(
    "manager_kind", ["none", "empty_schemes", "missing_root", "file_root"]
)
def test_capability_requires_usable_resolver_cache(monkeypatch, tmp_path, manager_kind):
    monkeypatch.setenv("DYN_LORA_RUNTIME_LOAD_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_ENABLED", "true")
    file_root = tmp_path / "file"
    file_root.write_text("not a directory")
    managers = {
        "none": None,
        "empty_schemes": SimpleNamespace(runtime_lora_schemes=frozenset()),
        "missing_root": SimpleNamespace(
            runtime_lora_schemes=frozenset({"custom"}), cache_root=tmp_path / "absent"
        ),
        "file_root": SimpleNamespace(
            runtime_lora_schemes=frozenset({"custom"}), cache_root=file_root
        ),
    }
    monkeypatch.setattr(
        protocol_mod, "get_lora_manager", lambda **_kwargs: managers[manager_kind]
    )
    with pytest.raises(protocol_mod.RuntimeLoRAConfigurationError):
        publish_runtime_lora_capability(
            SimpleNamespace(),
            SimpleNamespace(engine_args=SimpleNamespace(enable_lora=True)),
            WorkerType.Aggregated,
        )


def _identity(base_model: str, source_uri: str) -> str:
    digest = hashlib.sha256(
        b"dynamo-runtime-lora-v1\0" + base_model.encode() + b"\0" + source_uri.encode()
    ).hexdigest()
    return f"dyn-lora-{digest[:32]}"


def _request(source_uri: str, *, adapter_key: str | None = None):
    key = adapter_key or _identity("base", source_uri)
    return {
        "model": "base",
        "routing": {
            "lora_name": key,
            "base_model_name": "base",
            "lora_source_uri": source_uri,
            "lora_resolution_version": 2,
        },
    }


@pytest.mark.parametrize(
    ("base_model", "source_uri", "expected"),
    [
        (
            "meta-llama/Llama-3.1-8B-Instruct",
            "wandb-artifact:///team/project/adapter:v7",
            "dyn-lora-c7246b154263a336c006517e4bc6d7c8",
        ),
        (
            "base",
            "custom://adapter@sha256:abc",
            "dyn-lora-e3cfcd02f2e62cb70f45999324e7d4de",
        ),
        (
            "base",
            "wandb-artifact:///entity/project/artifact:v1",
            "dyn-lora-27056eb300024fbfc91334895bba1e26",
        ),
        (
            "base",
            "wandb-artifact:///a|b:v1",
            "dyn-lora-74bbe88c562d9d177d098e3ba118851a",
        ),
    ],
)
def test_adapter_key_matches_rust_golden_vectors(base_model, source_uri, expected):
    assert (
        protocol_mod._adapter_key(protocol_mod._identity_digest(base_model, source_uri))
        == expected
    )


def test_runtime_capability_is_published_only_for_valid_aggregated_worker(
    monkeypatch,
    tmp_path,
):
    class RuntimeConfig:
        def __init__(self):
            self.values = {}
            self.taints = {"existing"}

        def set_engine_specific(self, key, value):
            self.values[key] = json.loads(value)

    manager = SimpleNamespace(
        cache_root=tmp_path,
        runtime_lora_schemes=frozenset({"wandb-artifact"}),
    )
    monkeypatch.setenv("DYN_LORA_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_RUNTIME_LOAD_ENABLED", "true")
    monkeypatch.setattr(protocol_mod, "get_lora_manager", lambda **_kwargs: manager)
    runtime_config = RuntimeConfig()

    publish_runtime_lora_capability(
        runtime_config,
        SimpleNamespace(engine_args=SimpleNamespace(enable_lora=True)),
        WorkerType.Aggregated,
    )

    assert runtime_config.values["supports_runtime_lora_resolution"] is True
    assert runtime_config.values["runtime_lora_protocol_versions"] == [2]
    assert runtime_config.values["runtime_lora_schemes"] == ["wandb-artifact"]
    assert runtime_config.taints == {
        "existing",
        "dynamo.runtime-lora/v2",
    }


def test_runtime_capability_rejects_invalid_resource_bounds(monkeypatch, tmp_path):
    manager = SimpleNamespace(
        cache_root=tmp_path,
        runtime_lora_schemes=frozenset({"wandb-artifact"}),
    )
    monkeypatch.setenv("DYN_LORA_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_RUNTIME_LOAD_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_MAX_RESIDENT_RUNTIME_LORAS", "5")
    monkeypatch.setattr(protocol_mod, "get_lora_manager", lambda **_kwargs: manager)

    with pytest.raises(
        protocol_mod.RuntimeLoRAConfigurationError,
        match="cannot exceed",
    ):
        publish_runtime_lora_capability(
            SimpleNamespace(),
            SimpleNamespace(engine_args=SimpleNamespace(enable_lora=True, max_loras=4)),
            WorkerType.Aggregated,
        )


def test_runtime_settings_parse_pending_admission_timeout(monkeypatch):
    monkeypatch.setenv("DYN_LORA_PENDING_ADMISSION_TIMEOUT_SECONDS", "17")

    settings = protocol_mod.RuntimeLoRASettings.from_engine_args(
        SimpleNamespace(max_loras=4, max_num_seqs=32)
    )

    assert settings.pending_admission_timeout_seconds == 17


def test_runtime_capability_rejects_vllm_tokenizer_mode(monkeypatch, tmp_path):
    manager = SimpleNamespace(
        cache_root=tmp_path,
        runtime_lora_schemes=frozenset({"wandb-artifact"}),
    )
    monkeypatch.setenv("DYN_LORA_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_RUNTIME_LOAD_ENABLED", "true")
    monkeypatch.setattr(protocol_mod, "get_lora_manager", lambda **_kwargs: manager)

    with pytest.raises(
        protocol_mod.RuntimeLoRAConfigurationError,
        match="tokenized request path",
    ):
        publish_runtime_lora_capability(
            SimpleNamespace(),
            SimpleNamespace(
                use_vllm_tokenizer=True,
                engine_args=SimpleNamespace(enable_lora=True, max_loras=4),
            ),
            WorkerType.Aggregated,
        )


def test_opaque_source_is_preserved_for_plugin_policy():
    source_uri = "custom:opaque?api_key=provider-owned#fragment"
    metadata = protocol_mod._parse_request(
        _request(source_uri, adapter_key=_identity("base", source_uri))
    )
    assert metadata is not None
    assert metadata.source_uri == source_uri


def test_runtime_capability_is_published_only_for_valid_aggregated_worker_with_cpu_capacity(
    monkeypatch,
    tmp_path,
):
    class RuntimeConfig:
        def __init__(self):
            self.values = {}
            self.taints = {"existing"}

        def set_engine_specific(self, key, value):
            self.values[key] = json.loads(value)

    manager = SimpleNamespace(
        cache_root=tmp_path,
        runtime_lora_schemes=frozenset({"wandb-artifact"}),
    )
    monkeypatch.setenv("DYN_LORA_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_RUNTIME_LOAD_ENABLED", "true")
    monkeypatch.setattr(protocol_mod, "get_lora_manager", lambda **_kwargs: manager)
    runtime_config = RuntimeConfig()
    handler_config = SimpleNamespace(
        engine_args=SimpleNamespace(enable_lora=True, max_loras=4, max_cpu_loras=8)
    )

    publish_runtime_lora_capability(
        runtime_config,
        handler_config,
        WorkerType.Aggregated,
    )

    assert runtime_config.values["supports_runtime_lora_resolution"] is True
    assert runtime_config.values["runtime_lora_protocol_versions"] == [2]
    assert runtime_config.values["runtime_lora_schemes"] == ["wandb-artifact"]
    assert runtime_config.taints == {
        "existing",
        "dynamo.runtime-lora/v2",
    }
    assert handler_config.runtime_lora_settings.max_registered_loras == 8
    assert handler_config.runtime_lora_settings.max_resident_runtime_loras == 8


def test_runtime_capability_rejects_invalid_resource_bounds_with_cpu_capacity(
    monkeypatch, tmp_path
):
    manager = SimpleNamespace(
        cache_root=tmp_path,
        runtime_lora_schemes=frozenset({"wandb-artifact"}),
    )
    monkeypatch.setenv("DYN_LORA_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_RUNTIME_LOAD_ENABLED", "true")
    monkeypatch.setenv("DYN_LORA_MAX_RESIDENT_RUNTIME_LORAS", "5")
    monkeypatch.setattr(protocol_mod, "get_lora_manager", lambda **_kwargs: manager)

    with pytest.raises(
        protocol_mod.RuntimeLoRAConfigurationError,
        match="cannot exceed",
    ):
        publish_runtime_lora_capability(
            SimpleNamespace(),
            SimpleNamespace(engine_args=SimpleNamespace(enable_lora=True, max_loras=4)),
            WorkerType.Aggregated,
        )
