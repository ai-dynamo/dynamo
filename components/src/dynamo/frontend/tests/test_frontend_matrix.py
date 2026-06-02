#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Guards for the FRONTEND.* parity-matrix generator (frontend_matrix.py).

Pure CPU: scans annotation text and parses a JUnit string -- no vllm/sglang
import, no GPU. Ratchets coverage (cells may improve, never silently regress)
and locks the case-matrix status conventions (pass / xfail / FAIL).
"""

from pathlib import Path

import frontend_matrix as fm
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]

TESTS_DIR = Path(__file__).parent

# Known, tracked coverage gaps. Everything else must stay covered/shared/n-a.
# Add an entry here (with a reason) only when a stage is intentionally left
# uncovered; remove it the moment its tests land. Currently none -- every
# non-n/a cell is covered or shared.
ALLOWED_GAPS: set[tuple[str, str]] = set()


def test_na_cells_are_exactly_declared():
    cov = fm.scan_coverage(TESTS_DIR)
    for stage, backend in fm.NA_CELLS:
        assert fm.coverage_cell(cov, stage, backend) == "n/a"


def test_coverage_does_not_regress():
    cov = fm.scan_coverage(TESTS_DIR)
    for stage in fm.STAGES:
        for backend in fm.BACKENDS:
            if (stage, backend) in fm.NA_CELLS or (stage, backend) in ALLOWED_GAPS:
                continue
            cell = fm.coverage_cell(cov, stage, backend)
            assert cell in ("covered", "shared"), (
                f"FRONTEND.{stage}/{backend} regressed to {cell!r}; "
                f"add coverage or record it in ALLOWED_GAPS with a reason"
            )


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
