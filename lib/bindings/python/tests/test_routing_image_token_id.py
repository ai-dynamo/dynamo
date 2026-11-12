# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the worker's real Rust resolver without model weights or services."""

import json
import shutil
from pathlib import Path

import pytest

from dynamo import _core

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.multimodal,
    pytest.mark.parallel,
    pytest.mark.skipif(
        not hasattr(_core, "resolve_routing_image_token_id"),
        reason="requires a wheel built with mm-routing",
    ),
]


@pytest.fixture(scope="module", autouse=True)
def nats_and_etcd():
    """Override the binding suite's service fixture for this pure resolver."""
    yield


@pytest.fixture
def glm_model_dir(tmp_path):
    fixture = Path(__file__).resolve().parents[3] / "llm/tests/data/glm5-next-routing"
    for name in ("config.json", "processor_config.json"):
        shutil.copyfile(fixture / name, tmp_path / name)
    return tmp_path


@pytest.mark.parametrize("flat", [False, True], ids=["nested", "flat"])
def test_glm_resolves_checkpoint_placeholder(glm_model_dir, flat):
    processor_path = glm_model_dir / "processor_config.json"
    if flat:
        processor = json.loads(processor_path.read_text())["image_processor"]
        (glm_model_dir / "preprocessor_config.json").write_text(json.dumps(processor))
        processor_path.unlink()

    # A custom checkpoint name must work without tokenizer.json or weights.
    token_id = _core.resolve_routing_image_token_id(
        "custom-checkpoint", str(glm_model_dir)
    )
    assert type(token_id) is int
    assert token_id == 154854


@pytest.mark.parametrize(
    ("file", "path", "value"),
    [
        ("config.json", ("image_token_id",), -1),
        ("config.json", ("image_token_id",), 2**31),
        ("config.json", ("vision_config", "patch_size"), 16),
        ("config.json", ("model_type",), "unsupported_model"),
        ("processor_config.json", ("image_processor", "do_resize"), False),
        ("processor_config.json", ("image_processor", "max_image_tokens"), None),
        ("processor_config.json", ("image_processor", "unknown_geometry_knob"), 1),
    ],
)
def test_glm_unsupported_config_returns_none(glm_model_dir, file, path, value):
    config_path = glm_model_dir / file
    config = json.loads(config_path.read_text())
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    config_path.write_text(json.dumps(config))

    assert (
        _core.resolve_routing_image_token_id("custom-checkpoint", str(glm_model_dir))
        is None
    )


@pytest.mark.parametrize("invalid", [False, True], ids=["missing", "malformed"])
def test_glm_unreadable_processor_returns_none(glm_model_dir, invalid):
    processor_path = glm_model_dir / "processor_config.json"
    if invalid:
        processor_path.write_text("{")
    else:
        processor_path.unlink()
    assert (
        _core.resolve_routing_image_token_id("custom-checkpoint", str(glm_model_dir))
        is None
    )
