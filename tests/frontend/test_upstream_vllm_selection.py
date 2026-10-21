# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep infrastructure errors outside the known-failure allowance."""

import xml.etree.ElementTree as ET

import pytest

from tests.frontend.test_upstream_vllm import UpstreamTestFailure, check_result

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.core,
]


@pytest.mark.parametrize("tag", ["error", "skipped", "failure"])
def test_upstream_result_failure_types(tag):
    """Only test-body failures may match the expected-failure marker."""
    case = ET.fromstring(f'<testcase><{tag} message="detail"/></testcase>')
    error = UpstreamTestFailure if tag == "failure" else RuntimeError
    with pytest.raises(error, match="detail"):
        check_result(case)


def test_teardown_error_overrides_test_failure():
    """A known assertion failure must not hide broken server cleanup."""
    case = ET.fromstring(
        "<testcase><failure>known</failure><error>cleanup failed</error></testcase>"
    )
    with pytest.raises(RuntimeError, match="cleanup failed"):
        check_result(case)


def test_upstream_pass():
    """A successful upstream case needs no exception or special handling."""
    check_result(ET.fromstring("<testcase/>"))
