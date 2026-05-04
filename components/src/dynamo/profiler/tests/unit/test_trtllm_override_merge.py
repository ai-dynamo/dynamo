# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for TRT-LLM --override-engine-args / --trtllm.* conflict resolution.

Validates that _merge_overrides_into_args correctly merges profiler overrides
into an existing --override-engine-args JSON blob instead of appending
mutually-exclusive --trtllm.* flags (GitHub issue #8659).
"""

import json

import pytest

from dynamo.profiler.utils.config_modifiers.trtllm import (
    _deep_merge,
    _dotted_to_nested,
    _merge_overrides_into_args,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.planner,
    pytest.mark.parallel,
]


# ---------------------------------------------------------------------------
# _dotted_to_nested
# ---------------------------------------------------------------------------


class TestDottedToNested:
    def test_single_level_key(self):
        assert _dotted_to_nested({"foo": 1}) == {"foo": 1}

    def test_nested_key(self):
        result = _dotted_to_nested({"kv_cache_config.enable_block_reuse": False})
        assert result == {"kv_cache_config": {"enable_block_reuse": False}}

    def test_multiple_keys_same_prefix(self):
        result = _dotted_to_nested(
            {
                "kv_cache_config.enable_block_reuse": True,
                "kv_cache_config.tokens_per_block": 32,
            }
        )
        assert result == {
            "kv_cache_config": {"enable_block_reuse": True, "tokens_per_block": 32}
        }

    def test_none_value(self):
        result = _dotted_to_nested({"cache_transceiver_config": None})
        assert result == {"cache_transceiver_config": None}


# ---------------------------------------------------------------------------
# _deep_merge
# ---------------------------------------------------------------------------


class TestDeepMerge:
    def test_override_replaces_scalar(self):
        base = {"disable_overlap_scheduler": True}
        overrides = {"disable_overlap_scheduler": False}
        assert _deep_merge(base, overrides) == {"disable_overlap_scheduler": False}

    def test_nested_merge(self):
        base = {"kv_cache_config": {"tokens_per_block": 32, "enable_block_reuse": True}}
        overrides = {"kv_cache_config": {"enable_block_reuse": False}}
        result = _deep_merge(base, overrides)
        assert result == {
            "kv_cache_config": {"tokens_per_block": 32, "enable_block_reuse": False}
        }

    def test_override_replaces_dict_with_none(self):
        base = {"cache_transceiver_config": {"backend": "DEFAULT"}}
        overrides = {"cache_transceiver_config": None}
        assert _deep_merge(base, overrides) == {"cache_transceiver_config": None}

    def test_new_keys_added(self):
        base = {"a": 1}
        overrides = {"b": 2}
        assert _deep_merge(base, overrides) == {"a": 1, "b": 2}

    def test_does_not_mutate_base(self):
        base = {"kv_cache_config": {"tokens_per_block": 32}}
        overrides = {"kv_cache_config": {"tokens_per_block": 64}}
        _deep_merge(base, overrides)
        assert base["kv_cache_config"]["tokens_per_block"] == 32


# ---------------------------------------------------------------------------
# _merge_overrides_into_args  (the key fix for #8659)
# ---------------------------------------------------------------------------


class TestMergeOverridesIntoArgs:
    def test_no_existing_override_uses_trtllm_flags(self):
        """When no --override-engine-args exists, --trtllm.* flags are appended."""
        args = ["--model-path", "my-model"]
        result = _merge_overrides_into_args(
            args, {"kv_cache_config.enable_block_reuse": False}
        )
        assert "--trtllm.kv_cache_config.enable_block_reuse" in result
        assert "--override-engine-args" not in result

    def test_existing_override_merges_into_json(self):
        """When --override-engine-args exists, overrides merge into the JSON."""
        existing_json = json.dumps(
            {
                "cache_transceiver_config": {"backend": "DEFAULT"},
                "disable_overlap_scheduler": True,
                "kv_cache_config": {"tokens_per_block": 32},
            }
        )
        args = ["--model-path", "my-model", "--override-engine-args", existing_json]

        result = _merge_overrides_into_args(
            args,
            {
                "kv_cache_config.enable_block_reuse": False,
                "disable_overlap_scheduler": False,
                "cache_transceiver_config": None,
            },
        )

        # Must NOT have any --trtllm.* flags
        assert not any(a.startswith("--trtllm.") for a in result)

        # Must have a single --override-engine-args
        idx = result.index("--override-engine-args")
        merged = json.loads(result[idx + 1])

        # Profiler overrides take precedence
        assert merged["disable_overlap_scheduler"] is False
        assert merged["cache_transceiver_config"] is None
        assert merged["kv_cache_config"]["enable_block_reuse"] is False
        # Existing values that were not overridden are preserved
        assert merged["kv_cache_config"]["tokens_per_block"] == 32

    def test_existing_override_with_tp_size(self):
        """TP size override merges cleanly into existing JSON."""
        existing_json = json.dumps({"kv_cache_config": {"tokens_per_block": 32}})
        args = ["--override-engine-args", existing_json]

        result = _merge_overrides_into_args(args, {"tensor_parallel_size": 4})

        idx = result.index("--override-engine-args")
        merged = json.loads(result[idx + 1])
        assert merged["tensor_parallel_size"] == 4
        assert merged["kv_cache_config"]["tokens_per_block"] == 32

    def test_reproduces_issue_8659_scenario(self):
        """Reproduces the exact scenario from the bug report."""
        existing_json = json.dumps(
            {
                "cache_transceiver_config": {
                    "backend": "DEFAULT",
                },
                "disable_overlap_scheduler": True,
                "kv_cache_config": {"tokens_per_block": 32},
            }
        )
        args = [
            "--model-path",
            "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic",
            "--override-engine-args",
            existing_json,
        ]

        result = _merge_overrides_into_args(
            args,
            {
                "kv_cache_config.enable_block_reuse": False,
                "disable_overlap_scheduler": False,
                "cache_transceiver_config": None,
            },
        )

        # The exact failure case: both flags must NOT coexist
        has_override = "--override-engine-args" in result
        has_dynamic = any(a.startswith("--trtllm.") for a in result)
        assert not (has_override and has_dynamic), (
            "Both --override-engine-args and --trtllm.* flags present; "
            "TRT-LLM will reject this combination"
        )

    def test_args_with_pipe_redirect_preserved(self):
        """Args containing pipe/redirect operators are preserved at the end."""
        args = ["--model-path", "m", "|", "2>&1"]
        result = _merge_overrides_into_args(args, {"disable_overlap_scheduler": False})
        assert result[-2:] == ["|", "2>&1"]
        assert "--trtllm.disable_overlap_scheduler" in result
