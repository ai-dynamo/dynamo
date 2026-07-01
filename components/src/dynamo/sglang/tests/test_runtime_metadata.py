# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from dynamo.sglang.capacity import get_spec_decode_runtime_data

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def test_spec_decode_runtime_data_uses_speculative_num_steps():
    server_args = SimpleNamespace(
        speculative_num_steps="5",
        speculative_algorithm="EAGLE",
    )

    assert get_spec_decode_runtime_data(server_args) == {
        "nextn": 5,
        "method": "EAGLE",
        "source": "backend_config",
    }


@pytest.mark.parametrize(
    "speculative_num_steps",
    [None, 0, "bad"],
)
def test_spec_decode_runtime_data_ignores_invalid_nextn(speculative_num_steps):
    server_args = SimpleNamespace(
        speculative_num_steps=speculative_num_steps,
        speculative_algorithm="EAGLE",
    )

    assert get_spec_decode_runtime_data(server_args) is None


@pytest.mark.parametrize(
    "speculative_algorithm, expected",
    [
        ("EAGLE", True),
        ("EAGLE3", True),
        ("FROZEN_KV_MTP", True),
        ("DFLASH", False),
        ("NGRAM", False),
        ("STANDALONE", False),
        ("NONE", False),
        (None, False),
        (
            "some_unregistered_algo",
            False,
        ),  # from_string raises -> guarded to False, no crash
    ],
)
def test_eagle_enabled_for_speculative_algorithm(speculative_algorithm, expected):
    # enable_eagle must equal sglang's SpeculativeAlgorithm.is_eagle() -- the SAME predicate the
    # radix cache uses to bigram-key its KV events -- so the KV-router frontend's block-hash window
    # matches the worker's events. EAGLE3 + FROZEN_KV_MTP were previously omitted -> cache-blind.
    # (NEXTN/EAGLE are normalized to EAGLE/FROZEN_KV_MTP in ServerArgs before register sees them.)
    # NOTE: import lazily. register.py does `from sglang.srt.environ import envs`, which is absent in
    # the lint/collection env of the `pytest-marker-report` pre-commit hook (unlike the sglang-free
    # `capacity` module imported at top), so a module-level import breaks that hook's collection.
    from dynamo.sglang.register import _eagle_enabled_for

    assert _eagle_enabled_for(speculative_algorithm) is expected
