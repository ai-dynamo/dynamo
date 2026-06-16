# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""User-config preservation guardrail for the TRT-LLM worker (issue #9288).

The Dynamo->TRT-LLM ``arg_map`` is mutated by Dynamo's own code *after* the
user's config (YAML / --extra-engine-args / --override-engine-args) has been
merged in -- the publish-events block, ``skip_tokenizer_init``, ... Those
post-merge mutations have repeatedly overwritten user-supplied values silently.

This guardrail lives entirely in test code -- it adds NOTHING to production.
It captures two ``arg_map`` snapshots out of a real ``init_llm_worker`` run:

  * ``before``: taken by wrapping the existing ``_sync_config_from_engine_args``
    call, which sits exactly at the seam -- right after the user-config merge,
    just before Dynamo's post-merge mutations.
  * ``after``:  the final ``arg_map`` handed to the engine (via the existing
    ``get_llm_engine`` mock).

Any leaf key present in BOTH whose value changed is a clobber. Keys merely
*added* by Dynamo are intentional defaults for fields the user did not set and
are ignored. ``DYNAMO_POST_MERGE_OVERRIDES`` is the only hand-maintained
surface: a near-empty allowlist of fields Dynamo is deliberately allowed to
overwrite. A new engine config is covered automatically; a new post-merge
override trips this test until consciously allowlisted.
"""

import asyncio
import copy
from collections.abc import Mapping
from unittest import mock

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        "Skipping to avoid errors during collection with '-m gpu_0'. "
        "CUDA/GPU not available, but tensorrt_llm import and the test require GPU.",
        allow_module_level=True,
    )

import dynamo.trtllm.workers.llm_worker as worker_mod
from dynamo.trtllm.args import parse_args
from dynamo.trtllm.workers.llm_worker import init_llm_worker

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.gpu_1,
    pytest.mark.pre_merge,
    pytest.mark.profiled_vram_gib(0),
]

# Leaf paths (dotted) Dynamo is intentionally allowed to overwrite after the
# user-config merge. Keep this near-empty: today Dynamo overwrites NO user-set
# engine arg post-merge (it only *adds* keys the user omitted). Add an entry
# ONLY when deliberately introducing a post-merge override of a user-settable
# field -- that is the conscious, reviewed decision this test forces.
DYNAMO_POST_MERGE_OVERRIDES: frozenset[str] = frozenset()


# --------------------------------------------------------------------------- #
# Diff detector (test-only)
# --------------------------------------------------------------------------- #


def _normalize(value):
    """Recursively dump pydantic models / config objects to plain values.

    The pipeline legitimately converts a ``KvCacheConfig`` object into a dict
    mid-flight; dumping both snapshots the same way keeps that benign type
    change from reading as a wholesale change of every sub-key.
    """
    if hasattr(value, "model_dump"):
        try:
            value = value.model_dump(exclude_none=True)
        except Exception:
            return value
    if isinstance(value, Mapping):
        return {k: _normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    return value


def _flatten(value, prefix: str = "") -> dict:
    """Flatten a normalized structure into ``{dotted_path: leaf_value}``."""
    leaves: dict = {}
    if isinstance(value, Mapping):
        for key, sub in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            leaves.update(_flatten(sub, child))
    else:
        leaves[prefix] = value
    return leaves


def clobbered_user_args(
    before: Mapping, after: Mapping, allow: frozenset = DYNAMO_POST_MERGE_OVERRIDES
) -> dict:
    """Return ``{path: (old, new)}`` for user values Dynamo overwrote post-merge.

    Only leaf keys present in BOTH snapshots whose value changed are reported;
    keys merely added by Dynamo (defaults the user omitted) are not. Paths in
    ``allow`` are deliberate overrides and are filtered out.
    """
    before_leaves = _flatten(_normalize(dict(before)))
    after_leaves = _flatten(_normalize(dict(after)))
    return {
        path: (old, after_leaves[path])
        for path, old in before_leaves.items()
        if path not in allow and path in after_leaves and after_leaves[path] != old
    }


# --------------------------------------------------------------------------- #
# Unit tests for the detector
# --------------------------------------------------------------------------- #


def test_preserved_value_is_not_flagged():
    snap = {"kv_cache_config": {"event_buffer_max_size": 777}}
    assert clobbered_user_args(snap, copy.deepcopy(snap)) == {}


def test_clobbered_nested_value_is_flagged_with_dotted_path():
    # Mirrors the actual #9284 regression: user 777 overwritten by the default.
    before = {"kv_cache_config": {"event_buffer_max_size": 777}}
    after = {"kv_cache_config": {"event_buffer_max_size": 100_000}}
    assert clobbered_user_args(before, after) == {
        "kv_cache_config.event_buffer_max_size": (777, 100_000)
    }


def test_added_key_is_not_a_clobber():
    # Dynamo adding a default the user didn't set is fine, not a regression.
    before = {"kv_cache_config": {"free_gpu_memory_fraction": 0.42}}
    after = {
        "kv_cache_config": {
            "free_gpu_memory_fraction": 0.42,
            "event_buffer_max_size": 100_000,
        }
    }
    assert clobbered_user_args(before, after) == {}


def test_object_to_dict_conversion_is_not_a_false_positive():
    """A KvCacheConfig object dumped to a dict mid-pipeline must not look changed."""

    class FakeKvConfig:
        def model_dump(self, exclude_none=True):
            return {"free_gpu_memory_fraction": 0.42, "event_buffer_max_size": 777}

    before = {"kv_cache_config": FakeKvConfig()}
    after = {
        "kv_cache_config": {
            "free_gpu_memory_fraction": 0.42,
            "event_buffer_max_size": 777,
        }
    }
    assert clobbered_user_args(before, after) == {}


def test_allowlist_suppresses_intentional_override():
    before = {"backend": "trt"}
    after = {"backend": "pytorch"}
    assert clobbered_user_args(before, after) == {"backend": ("trt", "pytorch")}
    assert clobbered_user_args(before, after, allow=frozenset({"backend"})) == {}


# --------------------------------------------------------------------------- #
# Integration: capture before/after from a real init_llm_worker run
# --------------------------------------------------------------------------- #


class _EngineArgsCaptured(Exception):
    def __init__(self, engine_args):
        self.engine_args = engine_args


async def _capture_arg_map_snapshots(config):
    """Run init_llm_worker (engine mocked) and return (before, after) arg_maps.

    'before' is grabbed by wrapping the real _sync_config_from_engine_args (the
    seam right after the user-config merge); 'after' is the final arg_map.
    Production code is untouched -- only test-side mocks are used.
    """
    snapshots: dict = {}
    real_sync = worker_mod._sync_config_from_engine_args

    def capturing_sync(cfg, arg_map):
        snapshots["before"] = copy.deepcopy(arg_map)
        real_sync(cfg, arg_map)

    def capturing_engine(engine_args, *args, **kwargs):
        # engine_args is the final arg_map; nothing mutates it after this.
        snapshots["after"] = engine_args
        raise _EngineArgsCaptured(engine_args)

    with (
        mock.patch("dynamo.trtllm.workers.llm_worker.tokenizer_factory"),
        mock.patch("dynamo.trtllm.workers.llm_worker.nixl_connect.Connector"),
        mock.patch("dynamo.trtllm.workers.llm_worker.dump_config"),
        mock.patch("dynamo.trtllm.workers.llm_worker.LLMBackendMetrics"),
        mock.patch(
            "dynamo.trtllm.workers.llm_worker._sync_config_from_engine_args",
            side_effect=capturing_sync,
        ),
        mock.patch(
            "dynamo.trtllm.workers.llm_worker.get_llm_engine",
            side_effect=capturing_engine,
        ),
    ):
        with pytest.raises(_EngineArgsCaptured):
            await init_llm_worker(
                runtime=mock.MagicMock(),
                config=config,
                shutdown_event=asyncio.Event(),
            )

    return snapshots["before"], snapshots["after"]


@pytest.mark.asyncio
async def test_pipeline_preserves_user_config(tmp_path, monkeypatch):
    """The real pipeline must not clobber any user-supplied engine arg."""
    monkeypatch.delenv("DYN_ENABLE_TEST_LOGITS_PROCESSOR", raising=False)

    yaml_file = tmp_path / "engine.yaml"
    yaml_file.write_text(
        "kv_cache_config:\n"
        "  free_gpu_memory_fraction: 0.42\n"
        "  event_buffer_max_size: 777\n"
    )
    config = parse_args(
        [
            "--model",
            "fake-model",
            "--publish-kv-events",
            "--extra-engine-args",
            str(yaml_file),
        ]
    )

    before, after = await _capture_arg_map_snapshots(config)
    clobbered = clobbered_user_args(before, after)
    assert clobbered == {}, f"Dynamo clobbered user engine args: {clobbered}"


@pytest.mark.asyncio
async def test_pipeline_clobber_is_detected(tmp_path, monkeypatch):
    """A real post-merge mutation overwriting a user value must be caught.

    The test-logits path (DYN_ENABLE_TEST_LOGITS_PROCESSOR=1) forces
    skip_tokenizer_init=False after the user-config merge; if the user set
    skip_tokenizer_init=True in YAML that is exactly the issue #9288 clobber
    class, and the detector must flag it.
    """
    monkeypatch.setenv("DYN_ENABLE_TEST_LOGITS_PROCESSOR", "1")

    yaml_file = tmp_path / "engine.yaml"
    yaml_file.write_text("skip_tokenizer_init: true\n")
    config = parse_args(
        ["--model", "fake-model", "--extra-engine-args", str(yaml_file)]
    )

    before, after = await _capture_arg_map_snapshots(config)
    clobbered = clobbered_user_args(before, after)
    assert "skip_tokenizer_init" in clobbered, clobbered
    assert clobbered["skip_tokenizer_init"] == (True, False)
