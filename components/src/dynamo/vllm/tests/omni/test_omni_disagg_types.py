# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for omni/types.py Protocol definitions.

No GPU, no vllm_omni — pure structural typing checks.
"""

import pytest

from dynamo.vllm.omni.types import StageConnector, StageEngine

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_1,
    pytest.mark.pre_merge,
]


class _MockEngine:
    def generate(self, prompt, request_id="", *, sampling_params_list=None):
        async def _gen():
            yield {}

        return _gen()


class _MockConnector:
    def put(self, from_stage, to_stage, put_key, data):
        return True, 0, {}

    def cleanup(self, request_id):
        pass


def test_stage_engine_protocol_satisfied():
    assert isinstance(_MockEngine(), StageEngine)


def test_stage_connector_protocol_satisfied():
    assert isinstance(_MockConnector(), StageConnector)


def test_missing_generate_not_stage_engine():
    assert not isinstance(object(), StageEngine)


def test_missing_put_not_stage_connector():
    class NoCleanup:
        def put(self, f, t, r, p):
            return True, 0, {}

    assert not isinstance(NoCleanup(), StageConnector)


def test_missing_cleanup_not_stage_connector():
    class NoPut:
        def cleanup(self, rid):
            pass

    assert not isinstance(NoPut(), StageConnector)
