# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, call

import pytest

from tests.gpu_memory_service.common.runtime import SGLangWithGMSProcess

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.fault_tolerance,
    pytest.mark.parallel,
]


def test_sglang_pause_resume_use_native_engine_routes():
    process = object.__new__(SGLangWithGMSProcess)
    process._post_engine = Mock(
        side_effect=[
            {"status": "ok"},
            None,
            None,
            {"status": "ok"},
        ]
    )

    assert process.pause() == {"status": "ok"}
    assert process.resume(timeout=45) == {"status": "ok"}
    assert process._post_engine.call_args_list == [
        call("pause_generation", {}, 30, "pause generation"),
        call("release_memory_occupation", {}, 30, "release memory occupation"),
        call("resume_memory_occupation", {}, 45, "resume memory occupation"),
        call("continue_generation", {}, 45, "continue generation"),
    ]
