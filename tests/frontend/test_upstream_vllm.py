# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run selected upstream vLLM assertions against Dynamo and report each result."""

import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from tests.utils.constants import DynamoPortRange
from tests.utils.managed_process import ManagedProcess
from tests.utils.output_paths import resolve_test_output_path
from tests.utils.port_utils import reserved_ports

SUPPORT = Path(__file__).parents[1] / "utils/upstream_vllm"
SELECTION = json.loads((SUPPORT / "selection.json").read_text())
pytestmark = [
    pytest.mark.nightly,
    pytest.mark.e2e,
    pytest.mark.gpu_1,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.upstream_vllm,
    pytest.mark.timeout(900),
]


class UpstreamTestFailure(AssertionError):
    """An upstream test body failed; setup, teardown and skips use other errors."""


def check_result(case):
    """Raise for an upstream JUnit failure, keeping infrastructure errors distinct."""
    for tag in ("error", "skipped", "failure"):
        problem = case.find(tag)
        if problem is not None:
            error = UpstreamTestFailure if tag == "failure" else RuntimeError
            raise error(problem.text or problem.get("message", tag))


@pytest.fixture(scope="module")
def upstream_results():
    """Run upstream pytest once in isolation; retain its log and JUnit report.

    VLLM_TESTS_DIR must contain the pinned tests-only tree prepared by CI.
    A separate process avoids the two repositories' conflicting tests packages.
    """
    upstream = os.environ.get("VLLM_TESTS_DIR")
    if not upstream:
        pytest.skip("Run upstream-vllm-chat or set VLLM_TESTS_DIR")
    output = Path(resolve_test_output_path("upstream-vllm-chat"))
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env["DYNAMO_CHAT_LAUNCHER"] = str(
        Path(__file__).parents[2] / "examples/backends/vllm/launch/agg.sh"
    )
    # Use the image's installed runtime rather than CI's source-path overrides.
    env.pop("PYTHONPATH", None)
    with reserved_ports(1, DynamoPortRange.FRONTEND) as ports, ManagedProcess(
        command=[
            os.environ.get("VLLM_TEST_PYTHON", sys.executable),
            str(SUPPORT / "run.py"),
            upstream,
            str(output),
        ],
        env={**env, "DYN_SYSTEM_PORT": str(ports[0])},
        log_dir=str(output),
        display_name="upstream-vllm",
        terminate_all_matching_process_names=False,
    ) as process:
        returncode = process.proc.wait(timeout=800)
    junit = output / "junit.xml"
    assert (
        junit.exists()
    ), f"Upstream setup failed ({returncode}); see {process.log_path}"
    cases = ET.parse(junit).findall(".//testcase")
    results = {case.attrib["name"]: case for case in cases}
    assert len(cases) == len(SELECTION) and set(results) == {
        node.split("::")[-1] for node in SELECTION
    }, f"Incomplete upstream execution; see {junit}"
    assert returncode in (0, 1), f"Upstream harness error: exit {returncode}"
    return results


@pytest.mark.parametrize(
    "nodeid",
    [
        pytest.param(
            node,
            id=node.split("::")[-1],
            marks=pytest.mark.xfail(
                reason=reason, raises=UpstreamTestFailure, strict=False
            )
            if reason
            else (),
        )
        for node, reason in SELECTION.items()
    ],
)
def test_upstream_chat(nodeid, upstream_results):
    """Require the upstream assertion; tolerate only listed test-body failures."""
    check_result(upstream_results[nodeid.split("::")[-1]])
