# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# TEMPORARY: validates that pytest jobs killed by the timeout wrapper still
# produce an allure report. Revert this file (and the cpu_only_test_timeout_minutes
# override in pr.yaml) once the wrapper is confirmed working in CI.

import time

import pytest


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.vllm
@pytest.mark.gpu_0
def test_allure_timeout_quick_a():
    assert True


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.vllm
@pytest.mark.gpu_0
def test_allure_timeout_quick_b():
    assert True


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.vllm
@pytest.mark.gpu_0
def test_allure_timeout_slow_should_be_killed():
    # Sleeps long enough that the `timeout --signal=TERM` wrapper fires.
    # Pytest should receive SIGTERM, finalize whatever allure JSONs are
    # already written, and exit with code 124.
    time.sleep(600)
