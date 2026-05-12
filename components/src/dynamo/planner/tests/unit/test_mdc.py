# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the shared MDC helpers.

These cover the pure transform shared by KubernetesConnector and
VirtualConnector (see PR #8042 review): parsing ``card_json`` /
``runtime_config`` into ``WorkerInfo``, the prefill/decode heuristic,
and the LoRA-card filter.
"""

import pytest

from dynamo.planner.config.defaults import SubComponentType
from dynamo.planner.connectors.mdc import (
    MdcEntry,
    is_model_card,
    is_prefill_card,
    select_entry,
    worker_info_from_mdc,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


# ── is_model_card ──────────────────────────────────────────────────


class TestIsModelCard:
    def test_model_wrapper_accepted(self):
        assert is_model_card({"type": "Model"})

    def test_lora_wrapper_rejected(self):
        assert not is_model_card({"type": "LoRA"})

    def test_missing_type_rejected(self):
        assert not is_model_card({})


# ── is_prefill_card ────────────────────────────────────────────────


class TestIsPrefillCard:
    def test_int_prefill_bit_set(self):
        # ModelType::Prefill = 1 << 4 = 16
        assert is_prefill_card({"model_type": 16})

    def test_int_prefill_bit_unset(self):
        # Chat | Completions = 1 | 2 = 3
        assert not is_prefill_card({"model_type": 3})

    def test_int_prefill_combined_with_other_bits(self):
        # Prefill | Chat = 16 | 1 = 17
        assert is_prefill_card({"model_type": 17})

    def test_bitflags_dict_with_prefill(self):
        assert is_prefill_card({"model_type": {"bits": 16}})

    def test_bitflags_dict_without_prefill(self):
        assert not is_prefill_card({"model_type": {"bits": 2}})

    def test_string_prefill(self):
        assert is_prefill_card({"model_type": "Prefill"})

    def test_string_prefill_combined(self):
        assert is_prefill_card({"model_type": "Prefill|Chat"})

    def test_string_no_prefill(self):
        assert not is_prefill_card({"model_type": "Chat|Completions"})

    def test_missing_model_type_defaults_to_decode(self):
        assert not is_prefill_card({})

    def test_unparseable_model_type_defaults_to_decode(self):
        assert not is_prefill_card({"model_type": "not-a-thing"})
        assert not is_prefill_card({"model_type": None})


# ── worker_info_from_mdc ──────────────────────────────────────────


def _card(**runtime_config_overrides) -> dict:
    """Build a minimal realistic card_json payload."""
    return {
        "display_name": "meta-llama/Llama-3.1-8B",
        "model_type": 2,  # Completions (not prefill)
        "kv_cache_block_size": 16,
        "context_length": 8192,
        "runtime_config": {
            "total_kv_blocks": 1024,
            "max_num_seqs": 256,
            "max_num_batched_tokens": 8192,
            **runtime_config_overrides,
        },
    }


class TestWorkerInfoFromMdc:
    def test_happy_path_populates_all_fields(self):
        entry = MdcEntry(
            card_json=_card(),
            component="backend",
            endpoint="generate",
        )
        info = worker_info_from_mdc(entry, SubComponentType.DECODE, backend="vllm")
        assert info.model_name == "meta-llama/Llama-3.1-8B"
        assert info.component_name == "backend"
        assert info.endpoint == "generate"
        assert info.total_kv_blocks == 1024
        assert info.max_num_seqs == 256
        assert info.max_num_batched_tokens == 8192
        assert info.kv_cache_block_size == 16
        assert info.context_length == 8192

    def test_missing_wrapper_fields_fall_back_to_defaults(self):
        entry = MdcEntry(card_json=_card())
        info = worker_info_from_mdc(entry, SubComponentType.DECODE, backend="vllm")
        # From VllmComponentName
        assert info.component_name == "backend"
        assert info.endpoint == "generate"
        assert info.k8s_name == "VllmDecodeWorker"

    def test_prefill_defaults(self):
        entry = MdcEntry(card_json=_card())
        info = worker_info_from_mdc(entry, SubComponentType.PREFILL, backend="vllm")
        assert info.component_name == "prefill"
        assert info.k8s_name == "VllmPrefillWorker"

    def test_model_name_fallback_invoked_when_card_missing(self):
        card = _card()
        del card["display_name"]
        entry = MdcEntry(card_json=card)

        info = worker_info_from_mdc(
            entry,
            SubComponentType.DECODE,
            backend="vllm",
            model_name_fallback=lambda: "from-dgd-args",
        )
        assert info.model_name == "from-dgd-args"

    def test_model_name_fallback_not_invoked_when_card_has_it(self):
        called = []
        entry = MdcEntry(card_json=_card())

        info = worker_info_from_mdc(
            entry,
            SubComponentType.DECODE,
            backend="vllm",
            model_name_fallback=lambda: (called.append(1), "other")[1],
        )
        assert info.model_name == "meta-llama/Llama-3.1-8B"
        assert not called

    def test_k8s_name_override(self):
        entry = MdcEntry(card_json=_card())
        info = worker_info_from_mdc(
            entry,
            SubComponentType.DECODE,
            backend="vllm",
            k8s_name_override="custom-decode-svc",
        )
        assert info.k8s_name == "custom-decode-svc"

    def test_partial_runtime_config_populates_available_fields(self):
        card = _card()
        del card["runtime_config"]["max_num_seqs"]
        entry = MdcEntry(card_json=card)
        info = worker_info_from_mdc(entry, SubComponentType.DECODE, backend="vllm")
        assert info.max_num_batched_tokens == 8192
        assert info.max_num_seqs is None

    def test_missing_runtime_config(self):
        card = _card()
        del card["runtime_config"]
        entry = MdcEntry(card_json=card)
        info = worker_info_from_mdc(entry, SubComponentType.DECODE, backend="vllm")
        assert info.max_num_batched_tokens is None
        assert info.total_kv_blocks is None
        # Non-runtime_config fields still populate
        assert info.kv_cache_block_size == 16
        assert info.context_length == 8192

    def test_non_dict_runtime_config_treated_as_missing(self):
        card = _card()
        card["runtime_config"] = "not-a-dict"
        entry = MdcEntry(card_json=card)
        info = worker_info_from_mdc(entry, SubComponentType.DECODE, backend="vllm")
        assert info.max_num_batched_tokens is None
        assert info.total_kv_blocks is None
        assert info.max_num_seqs is None

    def test_empty_card_json(self):
        entry = MdcEntry(card_json={})
        info = worker_info_from_mdc(entry, SubComponentType.DECODE, backend="vllm")
        # Component/endpoint/k8s_name from defaults
        assert info.component_name == "backend"
        # Runtime fields all None
        assert info.max_num_batched_tokens is None
        assert info.model_name is None

    def test_fallback_exception_produces_none_model_name(self):
        card = _card()
        del card["display_name"]
        entry = MdcEntry(card_json=card)

        def _raise():
            raise RuntimeError("boom")

        info = worker_info_from_mdc(
            entry,
            SubComponentType.DECODE,
            backend="vllm",
            model_name_fallback=_raise,
        )
        assert info.model_name is None


# ── select_entry ───────────────────────────────────────────────────


class TestSelectEntry:
    def test_selects_prefill_entry(self):
        entries = [
            MdcEntry(card_json={**_card(), "model_type": 2}, component="backend"),
            MdcEntry(card_json={**_card(), "model_type": 16}, component="prefill"),
        ]
        hit = select_entry(entries, SubComponentType.PREFILL)
        assert hit is not None
        assert hit.component == "prefill"

    def test_selects_decode_entry(self):
        entries = [
            MdcEntry(card_json={**_card(), "model_type": 2}, component="backend"),
            MdcEntry(card_json={**_card(), "model_type": 16}, component="prefill"),
        ]
        hit = select_entry(entries, SubComponentType.DECODE)
        assert hit is not None
        assert hit.component == "backend"

    def test_component_filter_skips_mismatched(self):
        # Simulates a LoRA wrapper that slipped past is_model_card but points
        # at a different component — should be skipped when we know the
        # expected component.
        entries = [
            MdcEntry(card_json={**_card(), "model_type": 2}, component="lora-adapter"),
            MdcEntry(card_json={**_card(), "model_type": 2}, component="backend"),
        ]
        hit = select_entry(
            entries, SubComponentType.DECODE, expected_component="backend"
        )
        assert hit is not None
        assert hit.component == "backend"

    def test_returns_none_when_no_match(self):
        entries = [
            MdcEntry(card_json={**_card(), "model_type": 2}, component="backend"),
        ]
        assert select_entry(entries, SubComponentType.PREFILL) is None
