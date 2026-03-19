# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for OmniConfig validation."""

from types import SimpleNamespace

import pytest

try:
    from dynamo.vllm.omni.args import OmniConfig
except ImportError:
    pytest.skip("vLLM omni dependencies not available", allow_module_level=True)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_1,
    pytest.mark.pre_merge,
]


def _make_omni_config(**overrides) -> OmniConfig:
    """Build a minimal OmniConfig with valid defaults, applying overrides."""
    defaults = {
        # DynamoRuntimeConfig fields
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
        # OmniConfig fields
        "model": "test-model",
        "served_model_name": None,
        "engine_args": SimpleNamespace(),
        "stage_configs_path": None,
        "default_video_fps": 16,
        "enable_layerwise_offload": False,
        "layerwise_num_gpu_layers": 1,
        "vae_use_slicing": False,
        "vae_use_tiling": False,
        "boundary_ratio": 0.875,
        "flow_shift": None,
        "cache_backend": None,
        "cache_config": None,
        "enable_cache_dit_summary": False,
        "enable_cpu_offload": False,
        "enforce_eager": False,
        "ulysses_degree": 1,
        "ring_degree": 1,
        "cfg_parallel_size": 1,
    }
    defaults.update(overrides)
    obj = OmniConfig.__new__(OmniConfig)
    for k, v in defaults.items():
        setattr(obj, k, v)
    return obj


def test_omni_config_valid_defaults():
    """Config with valid defaults passes validation."""
    config = _make_omni_config()
    config.validate()  # should not raise


@pytest.mark.parametrize("fps", [0, -1, -100])
def test_omni_config_invalid_video_fps(fps):
    """Non-positive FPS must be rejected."""
    config = _make_omni_config(default_video_fps=fps)
    with pytest.raises(ValueError, match="--default-video-fps must be > 0"):
        config.validate()


@pytest.mark.parametrize("degree", [0, -1])
def test_omni_config_invalid_ulysses_degree(degree):
    """Non-positive ulysses_degree must be rejected."""
    config = _make_omni_config(ulysses_degree=degree)
    with pytest.raises(ValueError, match="--ulysses-degree must be > 0"):
        config.validate()


@pytest.mark.parametrize("degree", [0, -1])
def test_omni_config_invalid_ring_degree(degree):
    """Non-positive ring_degree must be rejected."""
    config = _make_omni_config(ring_degree=degree)
    with pytest.raises(ValueError, match="--ring-degree must be > 0"):
        config.validate()


@pytest.mark.parametrize("ratio", [0, -0.1, 1.01, 2.0])
def test_omni_config_invalid_boundary_ratio(ratio):
    """boundary_ratio outside (0, 1] must be rejected."""
    config = _make_omni_config(boundary_ratio=ratio)
    with pytest.raises(ValueError, match=r"--boundary-ratio must be in \(0, 1\]"):
        config.validate()


@pytest.mark.parametrize("ratio", [0.001, 0.5, 0.875, 1.0])
def test_omni_config_valid_boundary_ratio(ratio):
    """boundary_ratio within (0, 1] should pass."""
    config = _make_omni_config(boundary_ratio=ratio)
    config.validate()  # should not raise
