# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SGLang backend components."""

import json
import logging
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST

import dynamo.sglang._compat as sglang_compat
import dynamo.sglang.llm_engine as sglang_llm_engine
from dynamo.common.constants import DisaggregationMode, EmbeddingTransferMode
from dynamo.common.snapshot.constants import SNAPSHOT_CONTROL_DIR_ENV
from dynamo.sglang._compat import (
    ensure_sglang_external_mm_hash_updates_pad_value,
    ensure_sglang_frontend_decoded_video_support,
    ensure_sglang_tensor_image_size,
    ensure_sglang_top_level_exports,
    filter_supported_async_generate_kwargs,
    require_reasoning_kwargs,
    start_profile_compat,
)
from dynamo.sglang.args import (
    _forward_pass_metrics_source,
    _normalize_multimodal_disaggregation_args,
    parse_args,
    should_fetch_model,
    use_modelexpress_remote_instance,
)
from dynamo.sglang.backend_args import DynamoSGLangConfig
from dynamo.sglang.health_check import (
    SglangDisaggHealthCheckPayload,
    SglangPrefillHealthCheckPayload,
)
from dynamo.sglang.request_handlers.llm.decode_handler import DecodeWorkerHandler
from dynamo.sglang.tests.conftest import make_cli_args_fixture

try:
    from dynamo.sglang import register as sglang_register
except ImportError:
    sglang_register = None

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
    pytest.mark.core,
    pytest.mark.gpu_1,  # needs sglang & GPU packages installed but does not actually use GPU
    pytest.mark.profiled_vram_gib(0),  # These unit tests do not actually use GPU VRAM
    pytest.mark.pre_merge,
]
# Create SGLang-specific CLI args fixture
# This will use monkeypatch to write to argv
mock_sglang_cli = make_cli_args_fixture("dynamo.sglang")


def _make_sglang_config(**overrides):
    config = DynamoSGLangConfig()
    config.use_sglang_tokenizer = False
    config.multimodal_encode_worker = False
    config.multimodal_worker = False
    config.enable_multimodal = False
    config.dedicated_mm_encoder = False
    config.embedding_transfer_mode = EmbeddingTransferMode.NIXL_WRITE
    config.embedding_worker = False
    config.image_diffusion_worker = False
    config.video_generation_worker = False
    config.enable_rl = False
    config.frontend_decoding = False
    config.sglang_trace_level = 2
    config.fpm_trace = False
    config.disagg_config = None
    config.disagg_config_key = None
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


@pytest.mark.parametrize(
    ("guided_decoding", "expected"),
    [
        (
            {"json": {"type": "object", "required": ["city"]}},
            {"json_schema": json.dumps({"type": "object", "required": ["city"]})},
        ),
        (
            {"json": '{"type":"object","required":["city"]}'},
            {"json_schema": '{"type":"object","required":["city"]}'},
        ),
        ({"regex": r"[A-Z]{3}-\d{4}"}, {"regex": r"[A-Z]{3}-\d{4}"}),
        (
            {"grammar": 'root ::= "yes" | "no"'},
            {"ebnf": 'root ::= "yes" | "no"'},
        ),
        ({"choice": ["yes", "no"]}, {"regex": "(yes|no)"}),
    ],
    ids=["json-object", "json-string", "regex", "grammar", "choice"],
)
def test_unified_guided_decoding_maps_sglang_constraints(guided_decoding, expected):
    assert (
        sglang_llm_engine.SglangLLMEngine._get_guided_decoding_params(guided_decoding)
        == expected
    )


def test_unified_guided_decoding_escapes_choice_regex_metacharacters():
    params = sglang_llm_engine.SglangLLMEngine._get_guided_decoding_params(
        {"choice": ["a+b", "answer (A)", "[done]?"]}
    )

    assert params == {"regex": r"(a\+b|answer\ \(A\)|\[done\]\?)"}
    for choice in ("a+b", "answer (A)", "[done]?"):
        assert re.fullmatch(params["regex"], choice)


def test_unified_guided_decoding_ignores_empty_choice():
    assert (
        sglang_llm_engine.SglangLLMEngine._get_guided_decoding_params({"choice": []})
        == {}
    )


def test_unified_guided_decoding_preserves_structural_tag():
    structural_tag = {"begin": "<tool>", "schema": {"type": "object"}}

    assert sglang_llm_engine.SglangLLMEngine._get_guided_decoding_params(
        {"structural_tag": structural_tag}
    ) == {"structural_tag": json.dumps(structural_tag)}


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


def test_compat_supports_tensor_image_sizes_and_is_idempotent(caplog, monkeypatch):
    from sglang.srt.multimodal.processors.base_processor import (
        BaseMultimodalProcessor,
        BaseMultiModalProcessorOutput,
        MultimodalSpecialTokens,
    )

    class Processor:
        image_sizes = None

        def _get_num_multimodal_tokens(self, *, image_sizes):
            self.image_sizes = image_sizes
            return SimpleNamespace(num_image_tokens=[4])

    class ConcreteMultimodalProcessor(BaseMultimodalProcessor):
        async def process_mm_data_async(self, *args, **kwargs):
            raise NotImplementedError

    original = BaseMultimodalProcessor.resolve_image_token_counts
    try:
        ensure_sglang_tensor_image_size()
        installed = BaseMultimodalProcessor.resolve_image_token_counts
        ensure_sglang_tensor_image_size()

        processor = object.__new__(ConcreteMultimodalProcessor)
        processor._processor = Processor()
        image_token_id = 99
        processor._process_and_collect_mm_items = lambda **kwargs: (
            [],
            torch.tensor(
                [20, image_token_id, image_token_id, image_token_id, image_token_id, 21]
            ),
            {},
        )
        base_output = BaseMultiModalProcessorOutput(
            input_text="decoded prompt",
            input_ids=[10, image_token_id, 11],
            images=[torch.empty((3, 48, 80), dtype=torch.uint8)],
        )
        mm_tokens = MultimodalSpecialTokens(image_token_id=image_token_id)
        # SGLang defaults this on to preserve caller token IDs and expand only
        # image placeholders instead of decoding and retokenizing the prompt.
        monkeypatch.setenv("SGLANG_MM_AVOID_RETOKENIZE", "1")

        with caplog.at_level(
            logging.WARNING,
            logger="sglang.srt.multimodal.processors.base_processor",
        ):
            _, input_ids, _ = processor.process_and_combine_mm_data(
                base_output, mm_tokens
            )

        assert installed is BaseMultimodalProcessor.resolve_image_token_counts
        assert processor._processor.image_sizes == [(48, 80)]
        assert input_ids.tolist() == [
            10,
            image_token_id,
            image_token_id,
            image_token_id,
            image_token_id,
            11,
        ]
        assert not any(
            "falling back to decode+retokenize" in record.message
            for record in caplog.records
        )
    finally:
        BaseMultimodalProcessor.resolve_image_token_counts = original


def test_compat_refreshes_stale_pad_value_after_external_hash():
    from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem

    from dynamo.common.multimodal.routing_utils import pad_value_for_mm_hash

    original = MultimodalDataItem.set_pad_value
    try:
        item = MultimodalDataItem(
            modality=Modality.VIDEO,
            precomputed_embeddings=torch.arange(4, dtype=torch.float32),
        )
        item.set_pad_value()
        stale_pad_value = item.pad_value

        external_hash = 0x0123_4567_89AB_CDEF
        item.hash = external_hash
        ensure_sglang_external_mm_hash_updates_pad_value()
        installed = MultimodalDataItem.set_pad_value
        ensure_sglang_external_mm_hash_updates_pad_value()
        item.set_pad_value()

        assert installed is MultimodalDataItem.set_pad_value
        assert item.pad_value == pad_value_for_mm_hash(external_hash)
        assert item.pad_value != stale_pad_value
    finally:
        MultimodalDataItem.set_pad_value = original


@pytest.mark.asyncio
async def test_compat_preserves_frontend_sampled_video_frames_and_timestamps():
    from sglang.srt.disaggregation import encode_server
    from sglang.srt.multimodal.processors import qwen_vl

    from dynamo.sglang.request_handlers.multimodal.video_input import (
        FrontendDecodedVideo,
    )

    original_encode_server = encode_server.preprocess_video
    original_qwen = qwen_vl.preprocess_video
    try:
        ensure_sglang_frontend_decoded_video_support()
        installed = encode_server.preprocess_video
        ensure_sglang_frontend_decoded_video_support()

        assert installed is qwen_vl.preprocess_video

        frames = np.arange(4 * 3 * 5 * 3, dtype=np.uint8).reshape(4, 3, 5, 3)
        video = FrontendDecodedVideo(
            frames,
            {
                "fps": 24.0,
                "duration": 10.0,
                "total_num_frames": 240,
                "frames_indices": [0, 80, 160, 239],
            },
        )

        processor_frames, metadata = await encode_server.preprocess_video(
            video,
            video_config={"max_pixels": 1},
        )

        assert installed is encode_server.preprocess_video
        torch.testing.assert_close(
            processor_frames,
            torch.from_numpy(frames).permute(0, 3, 1, 2),
        )
        assert metadata == {
            "fps": 24.0,
            "duration": 10.0,
            "total_num_frames": 240,
            "frames_indices": [0, 80, 160, 239],
            "video_backend": "dynamo_frontend",
        }
    finally:
        encode_server.preprocess_video = original_encode_server
        qwen_vl.preprocess_video = original_qwen


@pytest.mark.asyncio
@pytest.mark.parametrize("is_multimodal", [False, True])
async def test_tensor_image_size_compat_uses_resolved_model_capability(
    monkeypatch, mock_sglang_cli, is_multimodal
):
    server_args = SimpleNamespace(
        disaggregation_mode="null",
        dllm_algorithm=None,
        kv_events_config=None,
        get_model_config=lambda: SimpleNamespace(is_multimodal=is_multimodal),
    )
    install_calls = []
    monkeypatch.setattr(
        "dynamo.sglang.args.ServerArgs.from_cli_args", lambda _: server_args
    )
    monkeypatch.setattr(
        "dynamo.sglang.args.ensure_sglang_tensor_image_size",
        lambda: install_calls.append(True),
    )
    mock_sglang_cli(model="/tmp")

    await parse_args(sys.argv[1:])

    assert install_calls == ([True] if is_multimodal else [])


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


@pytest.mark.parametrize(
    ("request_data", "expected"),
    [
        ({"require_reasoning": True}, {"require_reasoning": True}),
        ({"require_reasoning": False}, {"require_reasoning": False}),
        ({}, {"require_reasoning": False}),
    ],
)
def test_require_reasoning_kwarg_preserves_request_intent(request_data, expected):
    """The internal reasoning intent reaches SGLang, including explicit false."""

    class ReasoningEngine:
        async def async_generate(self, require_reasoning=False):
            return None

    assert require_reasoning_kwargs(ReasoningEngine(), request_data) == expected


def test_require_reasoning_kwarg_warns_once_when_dropped(caplog):
    """A dropped true reasoning requirement is visible without log spam."""

    class OldEngine:
        async def async_generate(self, input_ids=None, sampling_params=None):
            return None

    sglang_compat._warn_require_reasoning_unsupported.cache_clear()
    with caplog.at_level(logging.WARNING):
        assert require_reasoning_kwargs(OldEngine(), {"require_reasoning": True}) == {}
        assert require_reasoning_kwargs(OldEngine(), {"require_reasoning": True}) == {}

    assert caplog.text.count("Dropping require_reasoning=true") == 1
    sglang_compat._warn_require_reasoning_unsupported.cache_clear()


def test_require_reasoning_kwarg_silently_drops_false(caplog):
    """A dropped false value preserves the quiet compatibility fallback."""

    class OldEngine:
        async def async_generate(self, input_ids=None, sampling_params=None):
            return None

    sglang_compat._warn_require_reasoning_unsupported.cache_clear()
    with caplog.at_level(logging.WARNING):
        assert require_reasoning_kwargs(OldEngine(), {"require_reasoning": False}) == {}

    assert "Dropping require_reasoning=true" not in caplog.text


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
async def test_compat_starts_profile_with_legacy_kwargs():
    class LegacyTokenizerManager:
        received = None

        async def start_profile(self, output_dir=None, start_step=None, num_steps=None):
            self.received = {
                "output_dir": output_dir,
                "start_step": start_step,
                "num_steps": num_steps,
            }

    manager = LegacyTokenizerManager()
    body = {"output_dir": "/tmp/profile", "start_step": 10, "num_steps": 5}

    await start_profile_compat(manager, body)

    assert manager.received == body


@pytest.mark.asyncio
async def test_compat_starts_profile_with_request_object(monkeypatch):
    class RequestTokenizerManager:
        received = None

        async def start_profile(self, req=None):
            self.received = req

    request = SimpleNamespace(output_dir="/tmp/profile", start_step=10, num_steps=5)
    monkeypatch.setattr(sglang_compat, "_build_profile_request", lambda body: request)
    manager = RequestTokenizerManager()

    await start_profile_compat(
        manager,
        {"output_dir": "/tmp/profile", "start_step": 10, "num_steps": 5},
    )

    assert manager.received is request


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


@pytest.mark.asyncio
async def test_multiple_served_model_names_register_primary_and_aliases(
    mock_sglang_cli,
):
    """SGLang packed served names split into primary + Dynamo aliases."""
    mock_sglang_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--served-model-name",
        "primary,alias-one alias-two",
    )

    config = await parse_args(sys.argv[1:])

    assert config.server_args.served_model_name == "primary"
    assert config.dynamo_args.served_model_aliases == ["alias-one", "alias-two"]


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
async def test_reasoning_parser_both_flags_are_allowed(mock_sglang_cli):
    """Native gating and Dynamo response parsing may use separate reasoners."""
    mock_sglang_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--dyn-reasoning-parser",
        "qwen3",
        "--reasoning-parser",
        "qwen3",
    )

    config = await parse_args(sys.argv[1:])

    assert config.dynamo_args.dyn_reasoning_parser == "qwen3"
    assert config.server_args.reasoning_parser == "qwen3"


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


@pytest.mark.parametrize(
    (
        "mode",
        "expected_encode_worker",
        "expected_mm_worker",
        "expected_args",
    ),
    [
        ("encode", True, False, []),
        ("prefill", False, False, ["--disaggregation-mode", "prefill"]),
        ("decode", False, False, ["--disaggregation-mode", "decode"]),
        ("agg", False, False, ["--disaggregation-mode", "null"]),
        ("pd", False, False, ["--disaggregation-mode", "null"]),
    ],
)
def test_enable_multimodal_disaggregation_mode_maps_sglang_roles(
    mode,
    expected_encode_worker,
    expected_mm_worker,
    expected_args,
):
    """Canonical multimodal roles map to SGLang's current worker flags."""
    config = _make_sglang_config(enable_multimodal=True)

    normalized = _normalize_multimodal_disaggregation_args(
        ["--disaggregation-mode", mode], config
    )

    assert normalized == expected_args
    assert config.multimodal_encode_worker is expected_encode_worker
    assert config.multimodal_worker is expected_mm_worker


@pytest.mark.parametrize(
    ("mode", "expected_args"),
    [
        ("prefill", ["--disaggregation-mode", "prefill"]),
        ("decode", ["--disaggregation-mode", "decode"]),
        ("pd", ["--disaggregation-mode", "null"]),
    ],
)
def test_dedicated_mm_encoder_selects_internal_multimodal_worker(mode, expected_args):
    """Dedicated-mm-encoder selects internal E/PD or E/P/D workers."""
    config = _make_sglang_config(enable_multimodal=True, dedicated_mm_encoder=True)

    normalized = _normalize_multimodal_disaggregation_args(
        ["--disaggregation-mode", mode], config
    )

    assert normalized == expected_args
    assert config.multimodal_encode_worker is False
    assert config.multimodal_worker is True


def test_dedicated_mm_encoder_rejects_standalone_modes():
    """The dedicated encoder flag must not silently turn agg/encode into a different topology."""
    config = _make_sglang_config(enable_multimodal=True, dedicated_mm_encoder=True)

    with pytest.raises(ValueError, match="--dedicated-mm-encoder only applies"):
        _normalize_multimodal_disaggregation_args(
            ["--disaggregation-mode", "agg"], config
        )


def test_dedicated_mm_encoder_rejects_encode_worker_mode():
    config = _make_sglang_config(enable_multimodal=True, dedicated_mm_encoder=True)

    with pytest.raises(ValueError, match="Do not combine"):
        _normalize_multimodal_disaggregation_args(
            ["--disaggregation-mode", "encode"], config
        )


def test_dedicated_mm_encoder_requires_explicit_worker_role():
    config = _make_sglang_config(enable_multimodal=True, dedicated_mm_encoder=True)

    with pytest.raises(ValueError, match="requires --disaggregation-mode"):
        _normalize_multimodal_disaggregation_args([], config)


def test_multimodal_disaggregation_mode_uses_last_cli_value_with_dedicated_mm_encoder():
    """Config-merged args precede CLI args, so the last explicit value must win."""
    config = _make_sglang_config(enable_multimodal=True, dedicated_mm_encoder=True)

    normalized = _normalize_multimodal_disaggregation_args(
        ["--disaggregation-mode", "prefill", "--disaggregation-mode", "pd"],
        config,
    )

    assert normalized == ["--disaggregation-mode", "null"]
    assert config.multimodal_worker is True


def test_enable_multimodal_without_role_keeps_standalone_worker():
    """Capability-only SGLang serving should not select the internal EPD worker."""
    config = _make_sglang_config(enable_multimodal=True)

    normalized = _normalize_multimodal_disaggregation_args([], config)

    assert normalized == []
    assert config.enable_multimodal is True
    assert config.multimodal_worker is False
    assert config.multimodal_encode_worker is False


def test_legacy_multimodal_worker_sets_enable_multimodal():
    """Legacy multimodal role stays accepted while enabling the canonical flag."""
    config = _make_sglang_config(multimodal_worker=True)

    with pytest.warns(DeprecationWarning, match="--multimodal-worker"):
        config.validate()

    assert config.enable_multimodal is True
    assert config.multimodal_worker is True


def test_dedicated_mm_encoder_requires_enable_multimodal():
    """Dedicated-mm-encoder is a topology modifier, not a multimodal enable switch."""
    config = _make_sglang_config(dedicated_mm_encoder=True)

    with pytest.raises(ValueError, match="requires --enable-multimodal"):
        config.validate()


@pytest.mark.asyncio
async def test_forward_pass_metrics_enabled_from_env(monkeypatch, mock_sglang_cli):
    """Dynamo should enable FPM when DYN_FORWARDPASS_METRIC_PORT is set."""
    monkeypatch.setenv("DYN_FORWARDPASS_METRIC_PORT", "23456")
    mock_sglang_cli("--model", "Qwen/Qwen3-0.6B")

    config = await parse_args(sys.argv[1:])
    assert config.server_args.enable_forward_pass_metrics is True


@pytest.mark.asyncio
async def test_explicit_fpm_port_takes_precedence_over_trace(
    monkeypatch, mock_sglang_cli, caplog
):
    """The legacy explicit port remains authoritative even if trace is invalid."""
    monkeypatch.setenv("DYN_FORWARDPASS_METRIC_PORT", "23456")
    monkeypatch.setenv("DYN_FPM_TRACE", "sometimes")
    mock_sglang_cli("--model", "Qwen/Qwen3-0.6B")

    with caplog.at_level(logging.INFO):
        config = await parse_args(sys.argv[1:])

    assert config.server_args.enable_forward_pass_metrics is True
    assert (
        "Enabled forward_pass_metrics from DYN_FORWARDPASS_METRIC_PORT" in caplog.text
    )
    assert "Invalid DYN_FPM_TRACE value" not in caplog.text


@pytest.mark.asyncio
async def test_forward_pass_metrics_enabled_from_trace(monkeypatch, mock_sglang_cli):
    """DYN_FPM_TRACE should enable SGLang's existing FPM publisher."""
    monkeypatch.delenv("DYN_FORWARDPASS_METRIC_PORT", raising=False)
    monkeypatch.setenv("DYN_FPM_TRACE", "on")
    mock_sglang_cli("--model", "Qwen/Qwen3-0.6B")

    config = await parse_args(sys.argv[1:])
    assert config.server_args.enable_forward_pass_metrics is True


@pytest.mark.asyncio
async def test_forward_pass_metrics_enabled_from_cli_flag(monkeypatch, mock_sglang_cli):
    """The shared CLI flag should enable both Python and Rust trace handling."""
    monkeypatch.delenv("DYN_FORWARDPASS_METRIC_PORT", raising=False)
    monkeypatch.delenv("DYN_FPM_TRACE", raising=False)
    mock_sglang_cli("--fpm-trace", "--model", "Qwen/Qwen3-0.6B")

    config = await parse_args(sys.argv[1:])

    assert config.dynamo_args.fpm_trace is True
    assert config.server_args.enable_forward_pass_metrics is True
    assert os.environ["DYN_FPM_TRACE"] == "1"


@pytest.mark.asyncio
async def test_false_fpm_trace_does_not_enable_metrics(monkeypatch, mock_sglang_cli):
    monkeypatch.delenv("DYN_FORWARDPASS_METRIC_PORT", raising=False)
    monkeypatch.setenv("DYN_FPM_TRACE", "off")
    mock_sglang_cli("--model", "Qwen/Qwen3-0.6B")

    config = await parse_args(sys.argv[1:])
    assert not config.server_args.enable_forward_pass_metrics


@pytest.mark.asyncio
async def test_invalid_fpm_trace_is_disabled_by_arg_parser(
    monkeypatch, mock_sglang_cli
):
    monkeypatch.delenv("DYN_FORWARDPASS_METRIC_PORT", raising=False)
    monkeypatch.setenv("DYN_FPM_TRACE", "sometimes")
    mock_sglang_cli("--model", "Qwen/Qwen3-0.6B")

    config = await parse_args(sys.argv[1:])

    assert not config.server_args.enable_forward_pass_metrics
    assert os.environ["DYN_FPM_TRACE"] == "0"


@pytest.mark.parametrize(
    ("overrides", "role", "fpm_trace_relay_supported"),
    [
        ({}, "unified backend", False),
        ({"embedding_worker": True}, "embedding", True),
        ({"multimodal_encode_worker": True}, "dedicated multimodal", True),
        ({"multimodal_worker": True}, "dedicated multimodal", True),
        ({"image_diffusion_worker": True}, "image diffusion", True),
        ({"video_generation_worker": True}, "video generation", True),
    ],
)
def test_trace_does_not_activate_fpm_without_relay(
    monkeypatch, caplog, overrides, role, fpm_trace_relay_supported
):
    monkeypatch.delenv("DYN_FORWARDPASS_METRIC_PORT", raising=False)
    dynamo_config = _make_sglang_config(fpm_trace=True, **overrides)

    with caplog.at_level(logging.WARNING):
        source = _forward_pass_metrics_source(
            dynamo_config,
            fpm_trace_relay_supported=fpm_trace_relay_supported,
        )

    assert source is None
    assert f"SGLang {role} workers do not create a Dynamo FPM relay" in caplog.text


def test_explicit_port_preserves_legacy_activation_without_relay(monkeypatch, caplog):
    monkeypatch.setenv("DYN_FORWARDPASS_METRIC_PORT", "23456")
    dynamo_config = _make_sglang_config(embedding_worker=True, fpm_trace=True)

    with caplog.at_level(logging.WARNING):
        source = _forward_pass_metrics_source(
            dynamo_config,
            fpm_trace_relay_supported=False,
        )

    assert source == "DYN_FORWARDPASS_METRIC_PORT"
    assert "do not create a Dynamo FPM relay" not in caplog.text


def test_trace_does_not_activate_fpm_during_snapshot_startup(
    monkeypatch, caplog, tmp_path
):
    monkeypatch.delenv("DYN_FORWARDPASS_METRIC_PORT", raising=False)
    monkeypatch.setenv(SNAPSHOT_CONTROL_DIR_ENV, str(tmp_path))

    with caplog.at_level(logging.WARNING):
        source = _forward_pass_metrics_source(_make_sglang_config(fpm_trace=True))

    assert source is None
    assert "SGLang snapshot workers do not create a Dynamo FPM relay" in caplog.text


@pytest.mark.asyncio
async def test_unified_from_args_marks_fpm_relay_unsupported(monkeypatch):
    server_args = SimpleNamespace(
        skip_tokenizer_init=True,
        model_path="Qwen/Qwen3-0.6B",
        served_model_name="Qwen/Qwen3-0.6B",
    )
    dynamo_args = SimpleNamespace(use_sglang_tokenizer=False)
    config = SimpleNamespace(
        server_args=server_args,
        dynamo_args=dynamo_args,
        serving_mode=DisaggregationMode.AGGREGATED,
    )
    worker_config = object()
    parse_options = {}

    async def fake_parse_args(argv, *, fpm_trace_relay_supported):
        parse_options["fpm_trace_relay_supported"] = fpm_trace_relay_supported
        return config

    monkeypatch.delenv("DYN_ENABLE_TEST_LOGITS_PROCESSOR", raising=False)
    monkeypatch.setattr(sglang_llm_engine, "parse_args", fake_parse_args)
    monkeypatch.setattr(
        sglang_llm_engine.WorkerConfig,
        "from_runtime_config",
        lambda *args, **kwargs: worker_config,
    )

    engine, result_worker_config = await sglang_llm_engine.SglangLLMEngine.from_args(
        ["--model-path", "Qwen/Qwen3-0.6B"]
    )

    assert engine.server_args is server_args
    assert result_worker_config is worker_config
    assert parse_options["fpm_trace_relay_supported"] is False


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


def test_use_modelexpress_remote_instance_for_sglang_remote_instance():
    args = SimpleNamespace(
        load_format="remote_instance",
        remote_instance_weight_loader_backend="modelexpress",
    )

    assert use_modelexpress_remote_instance(args) is True


@pytest.mark.parametrize(
    "load_format, backend",
    [
        ("auto", "modelexpress"),
        ("remote_instance", "nccl"),
        ("remote_instance", None),
    ],
)
def test_use_modelexpress_remote_instance_rejects_other_load_paths(
    load_format, backend
):
    args = SimpleNamespace(
        load_format=load_format,
        remote_instance_weight_loader_backend=backend,
    )

    assert use_modelexpress_remote_instance(args) is False


def test_should_fetch_model_skips_sglang_modelexpress_remote_instance():
    args = SimpleNamespace(
        load_format="remote_instance",
        remote_instance_weight_loader_backend="modelexpress",
    )

    assert should_fetch_model(args, "Qwen/Qwen3-0.6B") is False


@pytest.mark.parametrize(
    "model_path",
    [
        "s3://bucket/model/",
        "gs://bucket/model/",
        "az://container/model/",
    ],
)
def test_should_fetch_model_skips_object_storage_paths(model_path):
    args = SimpleNamespace(load_format="runai_streamer")

    assert should_fetch_model(args, model_path) is False


def test_should_fetch_model_keeps_default_non_local_fetch():
    args = SimpleNamespace(load_format="auto")

    assert should_fetch_model(args, "Qwen/Qwen3-0.6B") is True


@pytest.mark.asyncio
async def test_register_model_uses_metadata_only_for_sglang_modelexpress(monkeypatch):
    if sglang_register is None:
        pytest.skip("dynamo.sglang.register is unavailable")

    captured: dict = {}

    async def fake_get_runtime_config(engine, server_args, dynamo_args):
        return None

    async def fake_register_model(*args, **kwargs):
        captured["kwargs"] = kwargs

    monkeypatch.setattr(sglang_register, "_get_runtime_config", fake_get_runtime_config)
    monkeypatch.setattr(sglang_register, "register_model", fake_register_model)

    server_args = SimpleNamespace(
        model_path="Qwen/Qwen3-0.6B",
        served_model_name="Qwen/Qwen3-0.6B",
        context_length=4096,
        page_size=1,
        load_format="remote_instance",
        remote_instance_weight_loader_backend="modelexpress",
    )
    dynamo_args = SimpleNamespace(
        use_sglang_tokenizer=False,
        frontend_decoding=False,
        custom_jinja_template=None,
    )

    result = await sglang_register._register_model_with_runtime_config(
        engine=SimpleNamespace(),
        endpoint=SimpleNamespace(),
        server_args=server_args,
        dynamo_args=dynamo_args,
        worker_type=sglang_register.WorkerType.Aggregated,
    )

    assert result is True
    assert captured["kwargs"]["ignore_weights"] is True


@pytest.mark.asyncio
async def test_register_model_uses_engine_managed_path_for_runai_object_storage(
    monkeypatch,
):
    if sglang_register is None:
        pytest.skip("dynamo.sglang.register is unavailable")

    captured: dict = {}

    async def fake_get_runtime_config(engine, server_args, dynamo_args):
        return None

    async def fake_register_model(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        if args[3].startswith("s3://"):
            raise AssertionError("object-storage path used as a normal model_path")

    monkeypatch.setattr(sglang_register, "_get_runtime_config", fake_get_runtime_config)
    monkeypatch.setattr(sglang_register, "register_model", fake_register_model)

    server_args = SimpleNamespace(
        model_path="s3://bucket/model",
        served_model_name="bucket-model",
        context_length=4096,
        page_size=1,
        load_format="runai_streamer",
        remote_instance_weight_loader_backend=None,
    )
    dynamo_args = SimpleNamespace(
        use_sglang_tokenizer=False,
        frontend_decoding=False,
        custom_jinja_template=None,
    )
    engine = SimpleNamespace(
        tokenizer_manager=SimpleNamespace(
            model_config=SimpleNamespace(
                model_weights="s3://bucket/model",
                model_path="/tmp/runai-model-metadata",
            )
        )
    )

    result = await sglang_register._register_model_with_runtime_config(
        engine=engine,
        endpoint=SimpleNamespace(),
        server_args=server_args,
        dynamo_args=dynamo_args,
        worker_type=sglang_register.WorkerType.Aggregated,
    )

    assert result is True
    assert captured["args"][3] == "/tmp/runai-model-metadata"
    assert "skip_model_path_fetch" not in captured["kwargs"]
    assert captured["kwargs"]["ignore_weights"] is False


# ---------------------------------------------------------------------------
# LoRA registration model_type + worker_type gate
# ---------------------------------------------------------------------------
# Pins the serving_mode → (model_type, worker_type) selection in
# LoraMixin.load_lora. A refactor that flips prefill back to Chat|Completions
# (or drops the explicit worker_type) cannot silently land: it would
# re-introduce the disagg hang where the frontend routes
# /v1/chat/completions directly to the prefill worker.


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "serving_mode, endpoint_types, expected_model_type_str, expected_worker_type",
    [
        # Prefill workers register the legacy ModelType.Prefill marker bit
        # (no OpenAI surface, dual-emitted for old-frontend compat) and
        # worker_type=Prefill.
        ("prefill", "chat,completions", "prefill", "prefill"),
        ("decode", "chat,completions", "chat,completions", "decode"),
        ("agg", "chat,completions", "chat,completions", "aggregated"),
        ("decode", "completions", "completions", "decode"),
        ("agg", "chat", "chat", "aggregated"),
    ],
)
async def test_lora_registration_model_type_gate(
    monkeypatch,
    serving_mode,
    endpoint_types,
    expected_model_type_str,
    expected_worker_type,
):
    """LoraMixin.load_lora must select (model_type, worker_type) based on serving_mode.

    PREFILL → (ModelType.Prefill, WorkerType.Prefill). The prefill router
    activates off worker_type; ModelType carries the legacy Prefill marker bit
    (no OpenAI surface, dual-emitted for old-frontend compat). Otherwise
    model_type follows parse_endpoint_types and worker_type follows the serving
    mode.
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
    assert (
        str(captured["worker_type"]) == expected_worker_type
    ), f"worker_type {captured['worker_type']} != expected {expected_worker_type}"
    assert captured["lora_name"] == "test_lora"
