# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Discover upstream chat tests and report Dynamo and vLLM-only results separately."""

import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from tests.utils.managed_process import ManagedProcess
from tests.utils.output_paths import resolve_test_output_path

SUPPORT = Path(__file__).parents[1] / "utils/upstream_vllm"
ALLOW_FAILURES = json.loads((SUPPORT / "allow_failures.json").read_text())
pytestmark = [
    pytest.mark.nightly,
    pytest.mark.e2e,
    pytest.mark.gpu_1,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.upstream_vllm,
    pytest.mark.timeout(1600),
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


def run_upstream(phase, *options):
    """Collect or execute the full upstream directory in an isolated process.

    Requires VLLM_TESTS_DIR and a fresh output directory. Returns its output path
    and pytest exit code; timeouts and process cleanup errors propagate.
    """
    upstream = os.environ["VLLM_TESTS_DIR"]
    output = Path(resolve_test_output_path("upstream-vllm-chat")) / phase
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env["DYNAMO_CHAT_LAUNCHER"] = str(
        Path(__file__).parents[2] / "examples/backends/vllm/launch/agg.sh"
    )
    # Use the image's installed runtime rather than CI's source-path overrides.
    env.pop("PYTHONPATH", None)
    with ManagedProcess(
        command=[
            os.environ.get("VLLM_TEST_PYTHON", sys.executable),
            str(SUPPORT / "run.py"),
            upstream,
            str(output),
            *options,
        ],
        env=env,
        log_dir=str(output),
        display_name=phase,
        terminate_all_matching_process_names=False,
    ) as process:
        returncode = process.proc.wait(timeout=120 if phase == "collection" else 1500)
    return output, returncode


def pytest_generate_tests(metafunc):
    """Parametrize from upstream pytest discovery; the allowance list never selects tests."""
    if "nodeid" not in metafunc.fixturenames:
        return
    if not os.environ.get("VLLM_TESTS_DIR"):
        metafunc.parametrize(
            "nodeid",
            [
                pytest.param(
                    None,
                    marks=pytest.mark.skip(
                        reason="Run upstream-vllm-chat or set VLLM_TESTS_DIR"
                    ),
                )
            ],
        )
        return
    output, code = run_upstream("collection", "--collect-only")
    if code != 0:
        raise RuntimeError(f"Upstream collection failed (exit {code}); see {output}")
    nodes = json.loads((output / "collection.json").read_text())
    assert nodes and len(nodes) == len(set(nodes)), "Empty or duplicate collection"
    assert not ALLOW_FAILURES.keys() - set(nodes), "Stale allow-failure entries"
    metafunc.parametrize(
        "nodeid",
        [
            pytest.param(
                node,
                id=node,
                marks=pytest.mark.xfail(
                    reason=ALLOW_FAILURES[node],
                    raises=UpstreamTestFailure,
                    strict=False,
                )
                if node in ALLOW_FAILURES
                else (),
            )
            for node in nodes
        ],
    )


@pytest.fixture(scope="module")
def upstream_results():
    """Execute all discovered tests once and require one result per full upstream ID."""
    output, code = run_upstream("execution")
    junit = output / "junit.xml"
    assert junit.exists(), f"Upstream setup failed ({code}); see {output}"
    cases = ET.parse(junit).findall(".//testcase")
    results = {
        case.find("properties/property[@name='upstream_nodeid']").get("value"): case
        for case in cases
    }
    nodes = json.loads((output.parent / "collection/collection.json").read_text())
    assert len(cases) == len(nodes) and set(results) == set(
        nodes
    ), f"Incomplete upstream execution; see {junit}"
    assert code in (0, 1), f"Upstream harness error: exit {code}"
    return results


def test_upstream_chat(nodeid, upstream_results, request):
    """Check each actual outcome; only listed test-body failures may be allowed."""
    case = upstream_results[nodeid]
    target = case.find("properties/property[@name='upstream_target']")
    request.node.user_properties.append(
        (
            "upstream_target",
            target.get("value") if target is not None else "not-executed",
        )
    )
    check_result(case)
