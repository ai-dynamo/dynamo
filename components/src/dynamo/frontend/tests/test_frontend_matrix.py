#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Guard for the FRONTEND.* parity-matrix generator (frontend_matrix.py).

Pure CPU (no vllm/sglang import, no GPU): locks the JUnit -> case-status
conventions (pass / xfail / FAIL) the dashboard renders.
"""

import frontend_matrix as fm
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


def test_junit_status_conventions(tmp_path):
    xml = (
        "<testsuite>"
        '<testcase name="T::test_assembly[a-20]"/>'
        '<testcase name="T::test_assembly[b-20]">'
        '<skipped type="pytest.xfail" message="known gap"/></testcase>'
        '<testcase name="T::test_assembly[c-20]"><failure message="boom"/></testcase>'
        "</testsuite>"
    )
    junit = tmp_path / "j.xml"
    junit.write_text(xml)
    assert fm.parse_junit(junit) == {"a-20": "pass", "b-20": "xfail", "c-20": "FAIL"}
