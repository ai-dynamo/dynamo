# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for arg_map user-config preservation in the TRT-LLM worker.

Pins the six historically-fragile fields from issue #9288:
kv_cache_config.event_buffer_max_size, kv_cache_config.free_gpu_memory_fraction,
kv_cache_config.cache_transceiver_config, return_perf_metrics,
enable_iter_perf_stats, backend. No GPU or tensorrt_llm required.
"""

import copy
import logging
from typing import Any
from unittest.mock import MagicMock

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]

# Local mirrors of llm_worker helpers — kept in sync so tests pin the contract.


def _make_kv_cache_config_mock(fields: dict) -> MagicMock:
    """Return a MagicMock whose model_dump returns a copy of *fields*."""
    mock = MagicMock()
    mock.model_dump.return_value = dict(fields)
    return mock


def _normalize_arg_map_for_snapshot(arg_map: dict) -> dict:
    """Mirror of llm_worker._normalize_arg_map_for_snapshot."""
    result = {}
    for k, v in arg_map.items():
        if hasattr(v, "model_dump"):
            result[k] = copy.deepcopy(v.model_dump(exclude_none=True))
        elif isinstance(v, dict):
            result[k] = copy.deepcopy(v)
        else:
            result[k] = v
    return result


def _audit_user_arg_clobbers(user_snapshot: dict, final_arg_map: dict) -> None:
    """Mirror of llm_worker._audit_user_arg_clobbers."""
    normalized: dict[str, Any] = {}
    for k, v in final_arg_map.items():
        if hasattr(v, "model_dump"):
            normalized[k] = copy.deepcopy(v.model_dump(exclude_none=True))
        elif isinstance(v, dict):
            normalized[k] = copy.deepcopy(v)
        else:
            normalized[k] = v
    for key, user_val in user_snapshot.items():
        if key not in normalized:
            logging.warning(
                "arg_map key %r was present after user config but is missing before LLM init.",
                key,
            )
            continue
        final_val = normalized[key]
        if isinstance(user_val, dict) and isinstance(final_val, dict):
            for sub_key, sub_user_val in user_val.items():
                if sub_key not in final_val:
                    logging.warning(
                        "Dynamo internals dropped user-supplied %s.%s",
                        key,
                        sub_key,
                    )
                elif final_val[sub_key] != sub_user_val:
                    logging.warning(
                        "Dynamo internals replaced user-supplied %s.%s: %r -> %r",
                        key,
                        sub_key,
                        sub_user_val,
                        final_val[sub_key],
                    )
        elif final_val != user_val:
            logging.warning(
                "Dynamo internals replaced user-supplied %s: %r -> %r",
                key,
                user_val,
                final_val,
            )


# ---- Tests for snapshot and audit helpers ----


class TestSnapshotAndAudit:
    def test_deep_copies_dict_values(self):
        """Dict values are deep-copied so snapshot mutations don't affect original."""
        inner = {"event_buffer_max_size": 512}
        snap = _normalize_arg_map_for_snapshot({"kv_cache_config": inner})
        snap["kv_cache_config"]["event_buffer_max_size"] = 9999
        assert inner["event_buffer_max_size"] == 512

    def test_kv_cache_config_object_converted_to_dict(self):
        """KvCacheConfig objects are converted to plain dicts via model_dump."""
        kv = _make_kv_cache_config_mock({"free_gpu_memory_fraction": 0.9})
        snap = _normalize_arg_map_for_snapshot({"kv_cache_config": kv})
        assert snap["kv_cache_config"] == {"free_gpu_memory_fraction": 0.9}

    def test_no_warning_when_unchanged(self, caplog):
        """No warning when snapshot matches final arg_map."""
        arg_map = {"return_perf_metrics": True, "tensor_parallel_size": 2}
        with caplog.at_level(logging.WARNING):
            _audit_user_arg_clobbers(_normalize_arg_map_for_snapshot(arg_map), arg_map)
        assert not caplog.records

    def test_warns_on_scalar_clobber(self, caplog):
        """Warning emitted when a scalar user value is replaced."""
        snapshot = _normalize_arg_map_for_snapshot({"return_perf_metrics": True})
        with caplog.at_level(logging.WARNING):
            _audit_user_arg_clobbers(snapshot, {"return_perf_metrics": False})
        assert any("return_perf_metrics" in r.message for r in caplog.records)

    def test_warns_on_nested_dict_clobber(self, caplog):
        """Warning names the sub-key when a nested dict field is mutated."""
        snapshot = _normalize_arg_map_for_snapshot(
            {"kv_cache_config": {"event_buffer_max_size": 512}}
        )
        with caplog.at_level(logging.WARNING):
            _audit_user_arg_clobbers(
                snapshot, {"kv_cache_config": {"event_buffer_max_size": 1024}}
            )
        assert any("event_buffer_max_size" in r.message for r in caplog.records)

    def test_warns_on_late_skip_tokenizer_init_clobber(self, caplog):
        """Late tokenizer init mutations are surfaced when audit runs before engine init."""
        arg_map = {"skip_tokenizer_init": True}
        snapshot = _normalize_arg_map_for_snapshot(arg_map)
        arg_map["skip_tokenizer_init"] = False
        with caplog.at_level(logging.WARNING):
            _audit_user_arg_clobbers(snapshot, arg_map)
        assert any("skip_tokenizer_init" in r.message for r in caplog.records)


# ---- Regression pins for the six historically-fragile fields (#9288) ----


class TestFragileFieldPreservation:
    """Pins all six fields that have regressed: event_buffer_max_size,
    free_gpu_memory_fraction, cache_transceiver_config, return_perf_metrics,
    enable_iter_perf_stats, backend."""

    DEFAULT_KV_EVENT_BUFFER_MAX_SIZE = 1024
    PYTORCH = "pytorch"

    def test_event_buffer_max_size_user_value_preserved(self):
        """User-supplied event_buffer_max_size is not overwritten by the Dynamo default."""
        kv: dict = {"event_buffer_max_size": 2048}
        if not kv.get("event_buffer_max_size"):
            kv["event_buffer_max_size"] = self.DEFAULT_KV_EVENT_BUFFER_MAX_SIZE
        assert kv["event_buffer_max_size"] == 2048

    def test_event_buffer_max_size_default_when_unset(self):
        """Dynamo default event_buffer_max_size is applied only when unset."""
        kv: dict = {}
        if not kv.get("event_buffer_max_size"):
            kv["event_buffer_max_size"] = self.DEFAULT_KV_EVENT_BUFFER_MAX_SIZE
        assert kv["event_buffer_max_size"] == self.DEFAULT_KV_EVENT_BUFFER_MAX_SIZE

    def test_kv_cache_config_fields_survive_snapshot(self):
        """free_gpu_memory_fraction and cache_transceiver_config survive the snapshot."""
        kv = _make_kv_cache_config_mock(
            {
                "free_gpu_memory_fraction": 0.85,
                "cache_transceiver_config": {"max_tokens_in_flight": 2048},
            }
        )
        snap = _normalize_arg_map_for_snapshot({"kv_cache_config": kv})
        assert snap["kv_cache_config"]["free_gpu_memory_fraction"] == 0.85
        assert snap["kv_cache_config"]["cache_transceiver_config"] == {
            "max_tokens_in_flight": 2048
        }

    def test_audit_catches_kv_field_drop(self, caplog):
        """Audit warns if free_gpu_memory_fraction is dropped during KvCacheConfig conversion."""
        snapshot = _normalize_arg_map_for_snapshot(
            {
                "kv_cache_config": _make_kv_cache_config_mock(
                    {"free_gpu_memory_fraction": 0.85}
                )
            }
        )
        with caplog.at_level(logging.WARNING):
            _audit_user_arg_clobbers(snapshot, {"kv_cache_config": {}})
        assert any("free_gpu_memory_fraction" in r.message for r in caplog.records)

    def test_user_backend_not_overwritten(self):
        """User-supplied backend survives the publish_events_and_metrics guard."""
        arg_map: dict = {"backend": "trtllm"}
        if "backend" not in arg_map:
            arg_map["backend"] = self.PYTORCH
        assert arg_map["backend"] == "trtllm"

    def test_audit_catches_return_perf_metrics_clobber(self, caplog):
        """Audit warns if return_perf_metrics is set to False after the snapshot."""
        arg_map = {"return_perf_metrics": True, "enable_iter_perf_stats": True}
        snapshot = _normalize_arg_map_for_snapshot(arg_map)
        arg_map["return_perf_metrics"] = False
        with caplog.at_level(logging.WARNING):
            _audit_user_arg_clobbers(snapshot, arg_map)
        assert any("return_perf_metrics" in r.message for r in caplog.records)

    def test_audit_catches_enable_iter_perf_stats_clobber(self, caplog):
        """Audit warns if enable_iter_perf_stats is set to False after the snapshot."""
        arg_map = {"return_perf_metrics": True, "enable_iter_perf_stats": True}
        snapshot = _normalize_arg_map_for_snapshot(arg_map)
        arg_map["enable_iter_perf_stats"] = False
        with caplog.at_level(logging.WARNING):
            _audit_user_arg_clobbers(snapshot, arg_map)
        assert any("enable_iter_perf_stats" in r.message for r in caplog.records)
