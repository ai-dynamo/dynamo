# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for OmniConfig validation and omni argument parsing."""

import contextlib
import dataclasses
import sys
from types import SimpleNamespace

import pytest

try:
    from dynamo.vllm.omni.args import (
        OmniConfig,
        OmniDiffusionKwargs,
        OmniParallelKwargs,
        parse_omni_args,
    )
except ImportError:
    pytest.skip("vLLM omni dependencies not available", allow_module_level=True)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    # Building the vLLM argument parser resolves a device; on an accelerator-less
    # host that raises unless a platform is pinned first.
    pytest.mark.usefixtures("vllm_cpu_platform_when_no_accelerator"),
    pytest.mark.xpu_1,
    pytest.mark.pre_merge,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.timeout(180),  # 0-GiB unit tests, floor 180s
]

_DIFFUSION_FIELDS = {f.name for f in dataclasses.fields(OmniDiffusionKwargs)}
_PARALLEL_FIELDS = {f.name for f in dataclasses.fields(OmniParallelKwargs)}


def _make_omni_config(**overrides) -> OmniConfig:
    """Build a minimal OmniConfig with valid defaults, applying overrides.

    Overrides for diffusion fields (e.g. boundary_ratio) and parallel fields
    (e.g. ulysses_degree) are automatically routed to the correct nested struct.
    """
    diffusion_overrides = {k: v for k, v in overrides.items() if k in _DIFFUSION_FIELDS}
    parallel_overrides = {k: v for k, v in overrides.items() if k in _PARALLEL_FIELDS}
    flat_overrides = {
        k: v
        for k, v in overrides.items()
        if k not in _DIFFUSION_FIELDS and k not in _PARALLEL_FIELDS
    }

    flat_defaults: dict = {
        "namespace": "dynamo",
        "component": "backend",
        "endpoint": None,
        "discovery_backend": "etcd",
        "request_plane": "tcp",
        "event_plane": "nats",
        "connector": [],
        "enable_local_indexer": True,
        "dyn_tool_call_parser": None,
        "dyn_reasoning_parser": None,
        "custom_jinja_template": None,
        "endpoint_types": "chat,completions",
        "dump_config_to": None,
        "multimodal_embedding_cache_capacity_gb": 0,
        "output_modalities": None,
        "media_output_fs_url": "file:///tmp/dynamo_media",
        "media_output_http_url": None,
        "model": "test-model",
        "served_model_name": None,
        "engine_args": SimpleNamespace(),
        "stage_configs_path": None,
        "default_video_fps": 16,
        "tts_max_instructions_length": 500,
        "tts_max_new_tokens_min": 1,
        "tts_max_new_tokens_max": 4096,
        "tts_ref_audio_timeout": 15,
        "tts_ref_audio_max_bytes": 50 * 1024 * 1024,
        "stage_id": None,
        "omni_router": False,
    }
    flat_defaults.update(flat_overrides)

    obj = OmniConfig.__new__(OmniConfig)
    for k, v in flat_defaults.items():
        setattr(obj, k, v)
    obj.diffusion = dataclasses.replace(OmniDiffusionKwargs(), **diffusion_overrides)
    obj.parallel = dataclasses.replace(OmniParallelKwargs(), **parallel_overrides)
    return obj


def test_omni_config_valid_defaults():
    config = _make_omni_config()
    config.validate()


@pytest.mark.parametrize("fps", [0, -1, -100])
def test_omni_config_invalid_video_fps(fps):
    config = _make_omni_config(default_video_fps=fps)
    with pytest.raises(ValueError, match="--default-video-fps must be > 0"):
        config.validate()


@pytest.mark.parametrize(
    ("field", "flag"),
    [
        ("ulysses_degree", "--ulysses-degree"),
        ("ring_degree", "--ring-degree"),
        ("text_encoder_tp_size", "--text-encoder-tp-size"),
    ],
)
@pytest.mark.parametrize("degree", [0, -1])
def test_omni_config_invalid_parallel_degree(field, flag, degree):
    config = _make_omni_config(**{field: degree})
    with pytest.raises(ValueError, match=rf"{flag} must be > 0"):
        config.validate()


@pytest.mark.parametrize("ratio", [0, -0.1, 1.01, 2.0])
def test_omni_config_invalid_boundary_ratio(ratio):
    config = _make_omni_config(boundary_ratio=ratio)
    with pytest.raises(ValueError, match=r"--boundary-ratio must be in \(0, 1\]"):
        config.validate()


@pytest.mark.parametrize("ratio", [0.001, 0.5, 0.875, 1.0])
def test_omni_config_valid_boundary_ratio(ratio):
    config = _make_omni_config(boundary_ratio=ratio)
    config.validate()


def test_negative_stage_id_rejected():
    config = _make_omni_config(stage_id=-1, stage_configs_path="/fake/path.yaml")
    with pytest.raises(ValueError, match="--stage-id must be >= 0"):
        config.validate()


def test_stage_id_requires_stage_configs_path():
    config = _make_omni_config(stage_id=0, stage_configs_path=None)
    with pytest.raises(ValueError, match="--stage-id requires"):
        config.validate()


def test_omni_router_requires_stage_configs_path():
    config = _make_omni_config(omni_router=True, stage_configs_path=None)
    with pytest.raises(ValueError, match="--omni-router requires"):
        config.validate()


def test_stage_id_and_omni_router_mutually_exclusive(tmp_path):
    config = _make_omni_config(
        stage_id=0, omni_router=True, stage_configs_path=str(tmp_path / "stages.yaml")
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        config.validate()


def test_stage_id_with_stage_configs_path_valid(tmp_path):
    config = _make_omni_config(
        stage_id=0, stage_configs_path=str(tmp_path / "stages.yaml")
    )
    config.validate()


def test_omni_router_with_stage_configs_path_valid(tmp_path):
    config = _make_omni_config(
        omni_router=True, stage_configs_path=str(tmp_path / "stages.yaml")
    )
    config.validate()


# --- parse_omni_args() on a host with no accelerator ---

_PLATFORM_UNSET = object()


@contextlib.contextmanager
def _no_accelerator():
    """Pin the platform a host with no accelerator resolves to.

    Every builtin plugin declines there and vLLM falls back to
    ``UnspecifiedPlatform``, whose ``device_type`` is the empty string -- the
    state ``DeviceConfig.__post_init__`` raises on. Restores exactly the way
    ``vllm_cpu_platform_when_no_accelerator`` does: *delete* the module-dict
    entry when there was none, so the PEP 562 lazy ``__getattr__`` in
    ``vllm.platforms`` is re-armed for later tests on this worker.
    """
    import vllm.platforms as vllm_platforms
    from vllm.platforms.interface import UnspecifiedPlatform

    previous = vllm_platforms.__dict__.get("current_platform", _PLATFORM_UNSET)
    vllm_platforms.current_platform = UnspecifiedPlatform()
    try:
        yield
    finally:
        if previous is _PLATFORM_UNSET:
            del vllm_platforms.current_platform
        else:
            vllm_platforms.current_platform = previous


def _router_argv(tmp_path, *extra):
    return [
        "dynamo.vllm.omni",
        "--stage-configs-path",
        str(tmp_path / "stages.yaml"),
        "--model",
        "test-model",
        *extra,
    ]


def test_stage_router_parses_without_an_accelerator(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", _router_argv(tmp_path, "--omni-router"))

    with _no_accelerator():
        config = parse_omni_args()

    assert config.omni_router is True
    assert config.model == "test-model"
    assert config.engine_args.model == "test-model"
    assert config.engine_args.trust_remote_code is False


def test_stage_router_selected_by_environment_parses_without_an_accelerator(
    monkeypatch, tmp_path
):
    # The containerized deployment shape: the role comes from DYN_OMNI_ROUTER
    # with no --omni-router token on the command line at all.
    monkeypatch.setenv("DYN_OMNI_ROUTER", "true")
    monkeypatch.setattr(sys, "argv", _router_argv(tmp_path))

    with _no_accelerator():
        config = parse_omni_args()

    assert config.omni_router is True
    assert config.model == "test-model"


def test_stage_router_ignores_engine_option_passthrough(monkeypatch, tmp_path):
    # The launch scripts forward their EXTRA_ARGS to every omni process, so
    # engine-only flags reach the router and must not abort it.
    monkeypatch.setattr(
        sys,
        "argv",
        _router_argv(
            tmp_path,
            "--omni-router",
            "--gpu-memory-utilization",
            "0.9",
            "--enable-lora",
        ),
    )

    with _no_accelerator():
        config = parse_omni_args()

    assert config.omni_router is True
    assert config.model == "test-model"


def test_stage_router_honors_negated_flag_over_environment(monkeypatch, tmp_path):
    # --no-omni-router must win over a truthy DYN_OMNI_ROUTER, the way argparse
    # itself resolves a flag against its env-derived default. Losing that would
    # route an engine-building worker onto the reduced parser.
    monkeypatch.setenv("DYN_OMNI_ROUTER", "true")
    monkeypatch.setattr(sys, "argv", _router_argv(tmp_path, "--no-omni-router"))

    with _no_accelerator(), pytest.raises(RuntimeError, match="Failed to infer device"):
        parse_omni_args()


def test_stage_worker_still_requires_an_accelerator(monkeypatch, tmp_path):
    # Negative control: --stage-id builds an engine, so it must keep failing
    # loudly here rather than being swept up by the router's reduced parser.
    monkeypatch.setattr(sys, "argv", _router_argv(tmp_path, "--stage-id", "0"))

    with _no_accelerator(), pytest.raises(RuntimeError, match="Failed to infer device"):
        parse_omni_args()


def test_parse_rejects_stage_id_combined_with_omni_router(monkeypatch, tmp_path):
    # The role pre-scan must not swallow the existing mutual-exclusion check.
    # Runs under the module fixture's resolvable platform, not _no_accelerator,
    # because --stage-id keeps this argv on the full engine parser.
    monkeypatch.setattr(
        sys, "argv", _router_argv(tmp_path, "--stage-id", "0", "--omni-router")
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        parse_omni_args()


# --- vllm_omni API compatibility guards ---


def test_omni_engine_args_importable():
    from vllm_omni.engine.arg_utils import OmniEngineArgs

    assert hasattr(OmniEngineArgs, "add_cli_args")
    assert hasattr(OmniEngineArgs, "from_cli_args")


def test_omni_engine_args_add_cli_args_no_extra_params():
    from vllm_omni.engine.arg_utils import OmniEngineArgs

    try:
        from vllm.utils import FlexibleArgumentParser
    except ImportError:
        from vllm.utils.argparse_utils import FlexibleArgumentParser
    parser = FlexibleArgumentParser(add_help=False)
    OmniEngineArgs.add_cli_args(parser)


def test_omni_config_imports_cleanly():
    from dynamo.vllm.omni.args import OmniConfig, parse_omni_args

    assert OmniConfig is not None
    assert callable(parse_omni_args)
