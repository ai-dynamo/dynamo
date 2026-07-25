# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.router,
]


def stub_module(name: str, **attributes: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    return module


class _ConfigBase:
    @classmethod
    def from_cli_args(cls, args):
        obj = cls.__new__(cls)
        for name, value in vars(args).items():
            setattr(obj, name, value)
        return obj


class _KvRouterConfigBase(_ConfigBase):
    router_prefill_load_model = "kv"
    router_track_prefill_tokens = False
    use_remote_indexer = False

    def apply_load_aware_preset(self) -> None:
        pass

    def kv_router_kwargs(self) -> dict:
        return {}


class _AicPerfConfigBase(_ConfigBase):
    aic_backend = None
    aic_model_path = None
    aic_system = None

    def aic_perf_kwargs(self) -> dict:
        return {}


class _ArgGroup:
    pass


class _NoopArgGroup:
    def add_arguments(self, parser) -> None:
        pass


def _add_argument(parser, **kwargs) -> None:
    parser.add_argument(kwargs["flag_name"], default=kwargs.get("default"))


def _add_negatable_bool_argument(parser, **kwargs) -> None:
    parser.add_argument(kwargs["flag_name"], action="store_true")


def _get_worker_namespace(namespace: str | None = None) -> str:
    if not namespace:
        namespace = os.environ.get("DYN_NAMESPACE", "dynamo")

    suffix = os.environ.get("DYN_NAMESPACE_WORKER_SUFFIX")
    if suffix:
        namespace = f"{namespace}-{suffix}"
    return namespace


def load_router_args_module():
    placeholder_config = type("PlaceholderConfig", (), {})
    stubs = {
        "dynamo": stub_module("dynamo"),
        "dynamo.common": stub_module("dynamo.common"),
        "dynamo.common.configuration": stub_module("dynamo.common.configuration"),
        "dynamo.common.configuration.arg_group": stub_module(
            "dynamo.common.configuration.arg_group",
            ArgGroup=_ArgGroup,
        ),
        "dynamo.common.configuration.groups": stub_module(
            "dynamo.common.configuration.groups"
        ),
        "dynamo.common.configuration.groups.aic_perf_args": stub_module(
            "dynamo.common.configuration.groups.aic_perf_args",
            AicPerfArgGroup=_NoopArgGroup,
            AicPerfConfigBase=_AicPerfConfigBase,
        ),
        "dynamo.common.configuration.groups.kv_router_args": stub_module(
            "dynamo.common.configuration.groups.kv_router_args",
            KvRouterArgGroup=_NoopArgGroup,
            KvRouterConfigBase=_KvRouterConfigBase,
        ),
        "dynamo.common.configuration.utils": stub_module(
            "dynamo.common.configuration.utils",
            add_argument=_add_argument,
            add_negatable_bool_argument=_add_negatable_bool_argument,
        ),
        "dynamo.common.utils": stub_module("dynamo.common.utils"),
        "dynamo.common.utils.namespace": stub_module(
            "dynamo.common.utils.namespace",
            get_worker_namespace=_get_worker_namespace,
        ),
        "dynamo.llm": stub_module(
            "dynamo.llm",
            AicPerfConfig=placeholder_config,
            KvRouterConfig=placeholder_config,
        ),
    }
    previous = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        module_path = Path(__file__).parents[1] / "args.py"
        spec = importlib.util.spec_from_file_location(
            "router_args_under_test", module_path
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous_module in previous.items():
            if previous_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous_module


def _validated_router_config(module, endpoint: str):
    config = module.DynamoRouterConfig.__new__(module.DynamoRouterConfig)
    config.endpoint = endpoint
    config.serve_indexer = False
    config.use_remote_indexer = False
    config.router_prefill_load_model = "kv"
    config.router_track_prefill_tokens = False
    config.aic_backend = None
    config.aic_model_path = None
    config.aic_system = None

    config.validate()

    return config


def test_validate_preserves_cli_namespace_without_kubernetes_env(monkeypatch) -> None:
    monkeypatch.delenv("DYN_NAMESPACE", raising=False)
    monkeypatch.delenv("DYN_NAMESPACE_WORKER_SUFFIX", raising=False)
    module = load_router_args_module()

    config = _validated_router_config(module, "dynamo.backend.generate")

    assert config.namespace == "dynamo"
    assert config.endpoint == "dynamo.backend.generate"


def test_validate_applies_base_dynamo_namespace_to_worker_endpoint(
    monkeypatch, caplog
) -> None:
    monkeypatch.setenv("DYN_NAMESPACE", "dynamo-system-thunderagent-demo")
    monkeypatch.delenv("DYN_NAMESPACE_WORKER_SUFFIX", raising=False)
    module = load_router_args_module()

    with caplog.at_level("INFO", logger=module.__name__):
        config = _validated_router_config(module, "dynamo.backend.generate")

    assert config.namespace == "dynamo-system-thunderagent-demo"
    assert config.endpoint == "dynamo-system-thunderagent-demo.backend.generate"
    assert "Resolved router worker endpoint namespace" in caplog.text


def test_validate_uses_worker_suffix_only_for_worker_endpoint(
    monkeypatch, caplog
) -> None:
    monkeypatch.setenv("DYN_NAMESPACE", "dynamo-system-thunderagent-demo")
    monkeypatch.setenv("DYN_NAMESPACE_WORKER_SUFFIX", "03302af8")
    module = load_router_args_module()

    with caplog.at_level("INFO", logger=module.__name__):
        config = _validated_router_config(module, "dynamo.backend.generate")

    assert config.namespace == "dynamo-system-thunderagent-demo"
    assert (
        config.endpoint == "dynamo-system-thunderagent-demo-03302af8.backend.generate"
    )
    assert "Resolved router worker endpoint namespace" in caplog.text
