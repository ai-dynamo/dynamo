# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SGLang backend components."""

import argparse
import asyncio
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST

import dynamo.sglang._compat as sglang_compat
from dynamo.sglang._compat import (
    ensure_sglang_top_level_exports,
    filter_supported_async_generate_kwargs,
)
from dynamo.sglang.args import parse_args
from dynamo.sglang.backend_args import DynamoSGLangArgGroup, DynamoSGLangConfig
from dynamo.sglang.health_check import (
    SglangDisaggHealthCheckPayload,
    SglangPrefillHealthCheckPayload,
)
from dynamo.sglang.request_handlers.llm.decode_handler import DecodeWorkerHandler
from dynamo.sglang.tests.conftest import make_cli_args_fixture
from dynamo.sglang.warmup import (
    warmup_decode_handler,
    warmup_generation_engine,
    warmup_runtime_endpoint,
)

# Get path relative to this test file
REPO_ROOT = Path(__file__).resolve().parents[5]
TEST_DIR = REPO_ROOT / "tests"
# Now construct the full path to the shared test fixture
JINJA_TEMPLATE_PATH = str(
    REPO_ROOT / "tests" / "serve" / "fixtures" / "custom_template.jinja"
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_1,  # needs sglang & GPU packages installed but does not actually use GPU
    pytest.mark.profiled_vram_gib(0),  # These unit tests do not actually use GPU VRAM
    pytest.mark.pre_merge,
]


def _expected_decode_warmup_request():
    return {
        "token_ids": list(range(18)),
        "stop_conditions": {
            "max_tokens": 8,
            "stop": [],
            "stop_token_ids": [],
            "min_tokens": 0,
            "ignore_eos": False,
        },
        "sampling_options": {
            "n": 1,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "repetition_penalty": 1.0,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0.0,
            "seed": None,
            "guided_decoding": None,
        },
        "output_options": {
            "logprobs": None,
            "prompt_logprobs": None,
            "skip_special_tokens": True,
            "return_tokens_as_token_ids": None,
        },
        "eos_token_ids": [],
        "annotations": [],
        "routing": None,
    }


# Create SGLang-specific CLI args fixture
# This will use monkeypatch to write to argv
mock_sglang_cli = make_cli_args_fixture("dynamo.sglang")


def test_compat_restores_sglang_top_level_exports():
    """Dynamo supports SGLang builds that omit top-level Engine/ServerArgs."""
    import sglang as sgl
    from sglang.srt.entrypoints.engine import Engine
    from sglang.srt.server_args import ServerArgs

    missing = object()
    original_engine = getattr(sgl, "Engine", missing)
    original_server_args = getattr(sgl, "ServerArgs", missing)

    try:
        if hasattr(sgl, "Engine"):
            delattr(sgl, "Engine")
        if hasattr(sgl, "ServerArgs"):
            delattr(sgl, "ServerArgs")

        ensure_sglang_top_level_exports()

        assert sgl.Engine is Engine
        assert sgl.ServerArgs is ServerArgs
    finally:
        if original_engine is missing:
            if hasattr(sgl, "Engine"):
                delattr(sgl, "Engine")
        else:
            sgl.Engine = original_engine

        if original_server_args is missing:
            if hasattr(sgl, "ServerArgs"):
                delattr(sgl, "ServerArgs")
        else:
            sgl.ServerArgs = original_server_args


def test_compat_filters_async_generate_kwargs_for_older_engines():
    class OldEngine:
        async def async_generate(self, input_ids=None, sampling_params=None):
            return None

    kwargs = {
        "input_ids": [1, 2, 3],
        "return_routed_experts": True,
    }

    assert filter_supported_async_generate_kwargs(OldEngine(), kwargs) == {
        "input_ids": [1, 2, 3]
    }


def test_compat_keeps_async_generate_kwargs_for_newer_engines():
    class NewEngine:
        async def async_generate(self, return_routed_experts=False):
            return None

    kwargs = {"return_routed_experts": True}

    assert filter_supported_async_generate_kwargs(NewEngine(), kwargs) == kwargs


def test_compat_keeps_async_generate_kwargs_for_variadic_engines():
    class VariadicEngine:
        async def async_generate(self, **kwargs):
            return None

    kwargs = {"return_routed_experts": True}

    assert filter_supported_async_generate_kwargs(VariadicEngine(), kwargs) == kwargs


def test_routed_experts_kwarg_omitted_when_flag_off():
    """Default config (no enable_return_routed_experts) → empty dict."""

    class NewEngine:
        async def async_generate(self, return_routed_experts=False):
            return None

    server_args = SimpleNamespace()  # flag absent → treated as False

    assert (
        DecodeWorkerHandler._resolve_routed_experts_kwargs(NewEngine(), server_args)
        == {}
    )


def test_routed_experts_kwarg_dropped_on_deepseek_v4_engine():
    """Opt-in + sglang deepseek_v4-shaped engine (no kwarg, no **kwargs) → empty dict.

    Mirrors the deepseek_v4 branch of sglang/srt/entrypoints/engine.py:
    async_generate has explicit named params and no return_routed_experts.
    The compat layer must drop the kwarg even when the user opted in.
    """

    class DeepSeekV4Engine:
        async def async_generate(
            self,
            prompt=None,
            sampling_params=None,
            input_ids=None,
            stream=False,
            bootstrap_host=None,
            bootstrap_port=None,
            bootstrap_room=None,
            data_parallel_rank=None,
            external_trace_header=None,
            rid=None,
        ):
            return None

    server_args = SimpleNamespace(enable_return_routed_experts=True)

    assert (
        DecodeWorkerHandler._resolve_routed_experts_kwargs(
            DeepSeekV4Engine(), server_args
        )
        == {}
    )


def test_routed_experts_kwarg_forwarded_when_flag_on_and_supported():
    """Opt-in + engine with kwarg in signature → kwarg forwarded as True."""

    class NewEngine:
        async def async_generate(self, return_routed_experts=False):
            return None

    server_args = SimpleNamespace(enable_return_routed_experts=True)

    assert DecodeWorkerHandler._resolve_routed_experts_kwargs(
        NewEngine(), server_args
    ) == {"return_routed_experts": True}


def test_compat_caches_async_generate_signature_inspection(monkeypatch):
    class CachedEngine:
        async def async_generate(self, return_routed_experts=False):
            return None

    sglang_compat._get_async_generate_supported_kwarg_names.cache_clear()
    calls = 0
    original_signature = sglang_compat.inspect.signature

    def counting_signature(obj):
        nonlocal calls
        calls += 1
        return original_signature(obj)

    monkeypatch.setattr(sglang_compat.inspect, "signature", counting_signature)

    kwargs = {"return_routed_experts": True}
    assert filter_supported_async_generate_kwargs(CachedEngine(), kwargs) == kwargs
    assert filter_supported_async_generate_kwargs(CachedEngine(), kwargs) == kwargs
    assert calls == 1

    sglang_compat._get_async_generate_supported_kwarg_names.cache_clear()


@pytest.mark.asyncio
async def test_custom_jinja_template_invalid_path(mock_sglang_cli):
    """Test that invalid file path raises FileNotFoundError."""
    invalid_path = "/nonexistent/path/to/template.jinja"
    mock_sglang_cli(
        "--model", "Qwen/Qwen3-0.6B", "--custom-jinja-template", invalid_path
    )

    with pytest.raises(
        FileNotFoundError,
        match=re.escape(f"Custom Jinja template file not found: {invalid_path}"),
    ):
        await parse_args(sys.argv[1:])


@pytest.mark.asyncio
async def test_custom_jinja_template_valid_path(mock_sglang_cli):
    """Test that valid absolute path is stored correctly."""
    mock_sglang_cli(model="Qwen/Qwen3-0.6B", custom_jinja_template=JINJA_TEMPLATE_PATH)

    config = await parse_args(sys.argv[1:])

    assert config.dynamo_args.custom_jinja_template == JINJA_TEMPLATE_PATH, (
        f"Expected custom_jinja_template value to be {JINJA_TEMPLATE_PATH}, "
        f"got {config.dynamo_args.custom_jinja_template}"
    )


@pytest.mark.asyncio
async def test_custom_jinja_template_env_var_expansion(monkeypatch, mock_sglang_cli):
    """Test that environment variables in paths are expanded by Python code."""
    jinja_dir = str(TEST_DIR / "serve" / "fixtures")
    monkeypatch.setenv("JINJA_DIR", jinja_dir)

    cli_path = "$JINJA_DIR/custom_template.jinja"
    mock_sglang_cli(model="Qwen/Qwen3-0.6B", custom_jinja_template=cli_path)

    config = await parse_args(sys.argv[1:])

    assert "$JINJA_DIR" not in config.dynamo_args.custom_jinja_template
    assert config.dynamo_args.custom_jinja_template == JINJA_TEMPLATE_PATH, (
        f"Expected custom_jinja_template value to be {JINJA_TEMPLATE_PATH}, "
        f"got {config.dynamo_args.custom_jinja_template}"
    )


# --- Tool Call Parser Validation Tests ---


@pytest.mark.asyncio
async def test_tool_call_parser_valid_with_dynamo_tokenizer(mock_sglang_cli):
    """Valid parser name works when using Dynamo's tokenizer."""
    mock_sglang_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--dyn-tool-call-parser",
        "hermes",  # supported by Dynamo
    )

    config = await parse_args(sys.argv[1:])

    assert config.dynamo_args.dyn_tool_call_parser == "hermes"


@pytest.mark.asyncio
async def test_tool_call_parser_invalid_with_dynamo_tokenizer(mock_sglang_cli):
    """Invalid parser name exits when using Dynamo's tokenizer."""
    mock_sglang_cli(
        "--model", "Qwen/Qwen3-0.6B", "--dyn-tool-call-parser", "nonexistent_parser"
    )

    with pytest.raises(SystemExit):
        await parse_args(sys.argv[1:])


@pytest.mark.asyncio
async def test_tool_call_parser_both_flags_error(mock_sglang_cli):
    """Setting both --dyn-tool-call-parser and --tool-call-parser exits with error."""
    mock_sglang_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--dyn-tool-call-parser",
        "hermes",
        "--tool-call-parser",
        "qwen25",
    )

    with pytest.raises(SystemExit):
        await parse_args(sys.argv[1:])


@pytest.mark.asyncio
async def test_namespace_flag_drives_default_endpoint_namespace(mock_sglang_cli):
    """CLI namespace should be used for auto-derived endpoint."""
    mock_sglang_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--namespace",
        "custom-ns",
    )

    config = await parse_args(sys.argv[1:])
    assert config.dynamo_args.namespace == "custom-ns"


@pytest.mark.asyncio
async def test_forward_pass_metrics_enabled_from_env(monkeypatch, mock_sglang_cli):
    """Dynamo should enable FPM when DYN_FORWARDPASS_METRIC_PORT is set."""
    monkeypatch.setenv("DYN_FORWARDPASS_METRIC_PORT", "1")
    mock_sglang_cli("--model", "Qwen/Qwen3-0.6B")

    config = await parse_args(sys.argv[1:])
    assert config.server_args.enable_forward_pass_metrics is True


@pytest.mark.asyncio
async def test_obsolete_dyn_endpoint_types_flag_is_supported(mock_sglang_cli):
    """Obsolete --dyn-endpoint-types alias should map to endpoint_types."""
    mock_sglang_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--dyn-endpoint-types",
        "completions",
    )

    config = await parse_args(sys.argv[1:])
    assert config.dynamo_args.endpoint_types == "completions"


def test_warmup_decode_engine_flag():
    parser = argparse.ArgumentParser()
    DynamoSGLangArgGroup().add_arguments(parser)

    config = DynamoSGLangConfig.from_cli_args(
        parser.parse_args(["--warmup-decode-engine"])
    )

    assert config.warmup_decode_engine is True


def test_generation_warmup_drains_engine_stream():
    class FakeEngine:
        def __init__(self):
            self.calls = []
            self.freeze_gc_calls = 0
            self.tokenizer_manager = self

        async def async_generate(self, **kwargs):
            self.calls.append(kwargs)

            async def stream():
                yield {"text": "warm"}
                yield {"text": "done"}

            return stream()

        async def freeze_gc(self):
            self.freeze_gc_calls += 1

    engine = FakeEngine()

    asyncio.run(
        warmup_generation_engine(
            engine,
            SimpleNamespace(disaggregation_mode="null"),
            timeout=1,
        )
    )

    assert engine.calls == [
        {
            "input_ids": list(range(18)),
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": 8,
                "ignore_eos": True,
            },
            "stream": True,
        }
    ]
    assert engine.freeze_gc_calls == 1


def test_generation_warmup_skips_disaggregated_decode():
    class FakeEngine:
        async def async_generate(self, **kwargs):
            raise AssertionError("warmup should not run")

    asyncio.run(
        warmup_generation_engine(
            FakeEngine(),
            SimpleNamespace(disaggregation_mode="decode"),
            timeout=1,
        )
    )


def test_handler_warmup_drains_decode_handler_stream():
    class FakeHandler:
        def __init__(self):
            self.calls = []
            self.engine = SimpleNamespace(tokenizer_manager=FakeTokenizerManager())

        async def generate(self, request, context):
            self.calls.append(
                {
                    "request": request,
                    "context_id": context.id(),
                    "trace_id": context.trace_id,
                    "span_id": context.span_id,
                    "is_stopped": context.is_stopped(),
                    "cancel_future_done": context.async_killed_or_stopped().done(),
                }
            )
            yield {"token_ids": [1]}
            yield {"finish_reason": "length", "token_ids": []}

    handler = FakeHandler()

    asyncio.run(
        warmup_decode_handler(
            handler,
            SimpleNamespace(disaggregation_mode="null"),
            timeout=1,
        )
    )

    assert handler.calls == [
        {
            "request": _expected_decode_warmup_request(),
            "context_id": "dynamo-sglang-warmup",
            "trace_id": "dynamo-sglang-warmup",
            "span_id": None,
            "is_stopped": False,
            "cancel_future_done": False,
        }
    ]
    assert handler.engine.tokenizer_manager.freeze_gc_calls == 1


class FakeTokenizerManager:
    def __init__(self):
        self.freeze_gc_calls = 0

    async def freeze_gc(self):
        self.freeze_gc_calls += 1


def test_runtime_endpoint_warmup_calls_own_instance():
    class FakeStream:
        def __init__(self):
            self.items = iter([{"token_ids": [1]}, {"token_ids": [2]}])

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self.items)
            except StopIteration as e:
                raise StopAsyncIteration from e

    class FakeClient:
        def __init__(self):
            self.wait_calls = 0
            self.direct_calls = []
            self.instances = [1234]

        async def wait_for_instances(self):
            self.wait_calls += 1
            return self.instances

        def instance_ids(self):
            return self.instances

        async def direct(self, request, instance_id, annotated=True):
            self.direct_calls.append(
                {
                    "request": request,
                    "instance_id": instance_id,
                    "annotated": annotated,
                }
            )
            return FakeStream()

    class FakeEndpoint:
        def __init__(self):
            self.fake_client = FakeClient()

        async def client(self):
            return self.fake_client

        def connection_id(self):
            return 1234

    endpoint = FakeEndpoint()
    engine = SimpleNamespace(tokenizer_manager=FakeTokenizerManager())

    asyncio.run(
        warmup_runtime_endpoint(
            endpoint,
            engine,
            SimpleNamespace(disaggregation_mode="null"),
            timeout=1,
        )
    )

    assert endpoint.fake_client.wait_calls == 1
    assert endpoint.fake_client.direct_calls == [
        {
            "request": _expected_decode_warmup_request(),
            "instance_id": 1234,
            "annotated": False,
        }
    ]
    assert engine.tokenizer_manager.freeze_gc_calls == 1


def test_runtime_endpoint_warmup_skips_disaggregated_decode():
    class FakeEndpoint:
        async def client(self):
            raise AssertionError("runtime endpoint warmup should not run")

    asyncio.run(
        warmup_runtime_endpoint(
            FakeEndpoint(),
            SimpleNamespace(tokenizer_manager=FakeTokenizerManager()),
            SimpleNamespace(disaggregation_mode="decode"),
            timeout=1,
        )
    )


def test_handler_warmup_skips_disaggregated_decode():
    class FakeHandler:
        async def generate(self, request, context):
            raise AssertionError("handler warmup should not run")

    asyncio.run(
        warmup_decode_handler(
            FakeHandler(),
            SimpleNamespace(disaggregation_mode="decode"),
            timeout=1,
        )
    )


@pytest.mark.asyncio
async def test_disagg_config_requires_disagg_config_key(mock_sglang_cli):
    """--disagg-config and --disagg-config-key must be provided together."""
    mock_sglang_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--disagg-config",
        "/tmp/nonexistent.yaml",
    )

    with pytest.raises(ValueError, match="disagg_config.*disagg_config_key.*together"):
        await parse_args(sys.argv[1:])


@pytest.mark.asyncio
async def test_disagg_config_key_requires_disagg_config(mock_sglang_cli):
    """--disagg-config-key alone should fail."""
    mock_sglang_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--disagg-config-key",
        "prefill",
    )

    with pytest.raises(ValueError, match="disagg_config.*disagg_config_key.*together"):
        await parse_args(sys.argv[1:])


@pytest.mark.asyncio
async def test_disagg_config_key_not_found_error(tmp_path, mock_sglang_cli):
    """Missing disagg section key should raise a clear ValueError."""
    config_path = tmp_path / "disagg.yaml"
    config_path.write_text(
        yaml.safe_dump({"prefill": {"tensor_parallel_size": 1}}), encoding="utf-8"
    )

    mock_sglang_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--disagg-config",
        str(config_path),
        "--disagg-config-key",
        "decode",
    )

    with pytest.raises(ValueError, match="Disagg config key 'decode' not found"):
        await parse_args(sys.argv[1:])


@pytest.mark.asyncio
async def test_disagg_config_section_must_be_dict(tmp_path, mock_sglang_cli):
    """Selected disagg section must be a dictionary."""
    config_path = tmp_path / "disagg.yaml"
    config_path.write_text(yaml.safe_dump({"prefill": "not-a-dict"}), encoding="utf-8")

    mock_sglang_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--disagg-config",
        str(config_path),
        "--disagg-config-key",
        "prefill",
    )

    with pytest.raises(
        ValueError, match="Disagg config section 'prefill' must be a dictionary"
    ):
        await parse_args(sys.argv[1:])


@pytest.mark.asyncio
async def test_disagg_config_preserves_bootstrap_port(tmp_path, mock_sglang_cli):
    """Bootstrap port from disagg section should not be overridden by auto-port logic."""
    config_path = tmp_path / "disagg.yaml"
    config_path.write_text(
        yaml.safe_dump({"prefill": {"disaggregation-bootstrap-port": 42345}}),
        encoding="utf-8",
    )

    mock_sglang_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--disagg-config",
        str(config_path),
        "--disagg-config-key",
        "prefill",
    )

    config = await parse_args(sys.argv[1:])
    assert config.server_args.disaggregation_bootstrap_port == 42345


@pytest.mark.asyncio
async def test_disagg_config_rejects_dynamo_keys(tmp_path, mock_sglang_cli, capfd):
    """Disagg config should only accept SGLang-native keys."""
    config_path = tmp_path / "disagg.yaml"
    config_path.write_text(
        yaml.safe_dump({"prefill": {"store-kv": "mem"}}), encoding="utf-8"
    )

    mock_sglang_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--disagg-config",
        str(config_path),
        "--disagg-config-key",
        "prefill",
    )

    with pytest.raises(SystemExit):
        await parse_args(sys.argv[1:])

    out, err = capfd.readouterr()
    assert "unrecognized arguments: --store-kv mem" in err


def test_disagg_health_check_payload_includes_bootstrap_info():
    payload = SglangDisaggHealthCheckPayload().to_dict()

    assert payload["bootstrap_info"]["bootstrap_host"] == FAKE_BOOTSTRAP_HOST
    assert payload["bootstrap_info"]["bootstrap_port"] == 0
    assert payload["bootstrap_info"]["bootstrap_room"] == 0
    assert payload["token_ids"] == [1]


def test_prefill_health_check_payload_is_disagg_compatible_alias():
    payload = SglangPrefillHealthCheckPayload().to_dict()

    assert "request" not in payload
    assert payload["bootstrap_info"]["bootstrap_host"] == FAKE_BOOTSTRAP_HOST
    assert payload["stop_conditions"]["max_tokens"] == 1


# ---------------------------------------------------------------------------
# LoRA registration model_type gate
# ---------------------------------------------------------------------------
# Pins the serving_mode → model_type selection in LoraMixin.load_lora so a
# refactor that flips prefill back to Chat|Completions cannot silently land:
# it would re-introduce the disagg hang where the frontend routes
# /v1/chat/completions directly to the prefill worker.


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "serving_mode, endpoint_types, expected_model_type_str",
    [
        ("prefill", "chat,completions", "prefill"),
        ("decode", "chat,completions", "chat,completions"),
        ("agg", "chat,completions", "chat,completions"),
        ("decode", "completions", "completions"),
        ("agg", "chat", "chat"),
    ],
)
async def test_lora_registration_model_type_gate(
    monkeypatch, serving_mode, endpoint_types, expected_model_type_str
):
    """LoraMixin.load_lora must select model_type based on serving_mode.

    PREFILL → ModelType.Prefill (so the prefill router activates and the frontend
    does not route chat completions directly to prefill).
    Otherwise → parse_endpoint_types(endpoint_types) (mirrors base-model
    registration so --endpoint-types overrides are honored).
    """
    from unittest.mock import AsyncMock, MagicMock

    from dynamo.common.constants import DisaggregationMode
    from dynamo.sglang.request_handlers import handler_base
    from dynamo.sglang.request_handlers.handler_base import LoraMixin

    # Capture the kwargs passed to register_llm.
    captured: dict = {}

    async def fake_register_llm(**kw):
        captured.update(kw)

    # Fake LoRA manager that returns a successful download.
    fake_lora_manager = MagicMock()
    fake_lora_manager.download_lora = AsyncMock(
        return_value={"status": "success", "local_path": "/tmp/fake_lora"}
    )

    monkeypatch.setattr(handler_base, "register_llm", fake_register_llm)
    monkeypatch.setattr(handler_base, "get_lora_manager", lambda: fake_lora_manager)
    monkeypatch.setattr(handler_base, "lora_name_to_id", lambda name: 12345)

    # Fake SGLang engine — only the LoRA load path is exercised.
    fake_load_result = SimpleNamespace(success=True, error_message=None)
    fake_engine = MagicMock()
    fake_engine.tokenizer_manager = MagicMock()
    fake_engine.tokenizer_manager.load_lora_adapter = AsyncMock(
        return_value=fake_load_result
    )

    # Exercise the mixin in isolation — avoids needing a concrete subclass
    # with abstract methods, real publisher, runtime, etc. The mixin only
    # touches engine, config, generate_endpoint, and its own LoRA tracking.
    class _Host(LoraMixin):
        pass

    handler = _Host()
    handler.engine = fake_engine
    handler.generate_endpoint = MagicMock()

    config = MagicMock()
    config.serving_mode = DisaggregationMode(serving_mode)
    config.server_args.model_path = "/models/base"
    config.server_args.page_size = 16
    config.dynamo_args.endpoint_types = endpoint_types
    handler.config = config

    handler._init_lora_tracking()

    # Drain the async generator.
    results = [
        chunk
        async for chunk in handler.load_lora(
            {"lora_name": "test_lora", "source": {"uri": "s3://x/y"}}
        )
    ]

    assert results and results[-1]["status"] == "success", results
    assert captured, "register_llm was not invoked"
    assert (
        str(captured["model_type"]) == expected_model_type_str
    ), f"model_type {captured['model_type']} != expected {expected_model_type_str}"
    assert captured["lora_name"] == "test_lora"
