# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for OmniConfig validation."""

import dataclasses
from types import SimpleNamespace

import pytest

try:
    from dynamo.vllm.omni.args import (
        OmniConfig,
        OmniDiffusionKwargs,
        OmniParallelKwargs,
    )
except ImportError:
    pytest.skip("vLLM omni dependencies not available", allow_module_level=True)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_1,
    pytest.mark.xpu_1,
    pytest.mark.pre_merge,
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
        "durable_kv_events": False,
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


@pytest.mark.parametrize("degree", [0, -1])
def test_omni_config_invalid_ulysses_degree(degree):
    config = _make_omni_config(ulysses_degree=degree)
    with pytest.raises(ValueError, match="--ulysses-degree must be > 0"):
        config.validate()


@pytest.mark.parametrize("degree", [0, -1])
def test_omni_config_invalid_ring_degree(degree):
    config = _make_omni_config(ring_degree=degree)
    with pytest.raises(ValueError, match="--ring-degree must be > 0"):
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
