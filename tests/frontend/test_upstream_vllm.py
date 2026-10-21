# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Expose unchanged upstream assertions as individually reported Dynamo tests."""

import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from huggingface_hub import snapshot_download

from tests.utils.constants import DynamoPortRange
from tests.utils.managed_process import ManagedProcess
from tests.utils.output_paths import resolve_test_output_path
from tests.utils.port_utils import reserved_ports

SUPPORT = Path(__file__).parents[1] / "utils/upstream_vllm"
SELECTION = json.loads((SUPPORT / "selection.json").read_text())
NODES = [case["nodeid"] for case in SELECTION["cases"]]
pytestmark = [
    pytest.mark.nightly,
    pytest.mark.e2e,
    pytest.mark.gpu_1,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.upstream_vllm,
    pytest.mark.timeout(1500),
]


@pytest.fixture(scope="module")
def upstream_results(tmp_path_factory):
    """Run each endpoint once, preserving upstream module-scoped server fixtures.

    Requires the dedicated lane's tests-only checkout and client environment.
    Nested JUnit and HTTP evidence remain in the standard Dynamo test-output root.
    """
    upstream = os.environ.get("VLLM_TESTS_DIR")
    if not upstream:
        pytest.skip("Run the dedicated upstream-vllm-chat job or set VLLM_TESTS_DIR")
    output = Path(resolve_test_output_path("upstream-vllm-chat"))
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["DYNAMO_CHAT_LAUNCHER"] = str(
        Path(__file__).parents[2] / "examples/backends/vllm/launch/agg.sh"
    )
    # Use the installed wheel in both child servers, not source-path overrides from CI.
    env.pop("PYTHONPATH", None)
    # Upstream's vocabulary helper loads "main"; pin that alias in a private view.
    cache = tmp_path_factory.mktemp("upstream-model-cache")
    for profile in SELECTION["models"].values():
        snapshot = snapshot_download(
            profile["id"],
            revision=profile["revision"],
            allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors", "*.jinja"],
        )
        model_cache = cache / ("models--" + profile["id"].replace("/", "--"))
        (model_cache / "snapshots").mkdir(parents=True)
        (model_cache / "snapshots" / profile["revision"]).symlink_to(snapshot)
        (model_cache / "refs").mkdir()
        (model_cache / "refs/main").write_text(profile["revision"])
    env.update(HF_HUB_CACHE=str(cache), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    results = {}
    for backend in ("reference", "dynamo"):
        directory = output / backend
        with reserved_ports(1, DynamoPortRange.FRONTEND) as ports, ManagedProcess(
            command=[
                os.environ.get("VLLM_TEST_PYTHON", sys.executable),
                str(SUPPORT / "run.py"),
                "--upstream",
                upstream,
                "--backend",
                backend,
                "--output",
                str(directory),
            ],
            env={**env, "DYN_SYSTEM_PORT": str(ports[0])},
            log_dir=str(output),
            display_name=backend,
            terminate_all_matching_process_names=False,
        ) as process:
            returncode = process.proc.wait(timeout=700)
        junit = directory / "junit.xml"
        assert (
            junit.exists()
        ), f"{backend} setup failed (exit {returncode}); see {process.log_path}"
        cases = ET.parse(junit).findall(".//testcase")
        assert len(cases) == len(NODES), f"Incomplete {backend} execution; see {junit}"
        results[backend] = {case.attrib["name"]: case for case in cases}
        assert set(results[backend]) == {n.split("::")[-1] for n in NODES}
        assert returncode in (0, 1), f"{backend} harness error: exit {returncode}"
    return results


@pytest.mark.parametrize("nodeid", NODES, ids=[node.split("::")[-1] for node in NODES])
def test_upstream_chat(nodeid, upstream_results):
    """Require the reference and Dynamo to satisfy the same unchanged upstream test."""
    name = nodeid.split("::")[-1]
    for backend in ("reference", "dynamo"):
        case = upstream_results[backend][name]
        problems = [
            *case.findall("failure"),
            *case.findall("error"),
            *case.findall("skipped"),
        ]
        assert not problems, f"{backend}: {nodeid}\n" + "\n".join(
            p.text or p.attrib.get("message", "") for p in problems
        )
