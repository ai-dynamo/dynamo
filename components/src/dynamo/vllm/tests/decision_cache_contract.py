# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate cache-statistics provenance projections from exact released source."""

import argparse
import ast
import json
import subprocess
from pathlib import Path

from dynamo.vllm.decision_cache_provenance import (
    CAUSAL_CACHE_SOURCE_BLOBS,
    CAUSAL_RELEASE_COMMITS,
    git_blob_id,
)

CONTRACT_PATH = Path(__file__).parent / "fixtures/causal-cache-provenance-v1.json"


def definition(tree, name):
    node = tree
    for part in name.split("."):
        node = next(
            child for child in node.body if getattr(child, "name", None) == part
        )
    return node


def only(nodes):
    rows = list(nodes)
    assert len(rows) == 1
    return rows[0]


def project_release(read_source):
    scheduler = ast.parse(read_source("vllm/v1/core/sched/scheduler.py"))
    schedule = definition(scheduler, "Scheduler.schedule")
    update = definition(scheduler, "Scheduler.update_from_output")
    capture = only(
        node
        for node in ast.walk(schedule)
        if isinstance(node, ast.If)
        and ast.unparse(node.test)
        == "request.prefill_stats and request.num_preemptions <= 0"
    )
    emit = only(
        node
        for node in ast.walk(update)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "should_emit_output"
        and any(
            isinstance(child, ast.Call)
            and ast.unparse(child.func) == "request.take_prefill_stats"
            for child in ast.walk(node)
        )
    )
    should_emit = only(
        node
        for node in ast.walk(update)
        if isinstance(node, ast.Assign)
        and any(ast.unparse(target) == "should_emit_output" for target in node.targets)
    )
    outputs = [
        node
        for node in ast.walk(update)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "EngineCoreOutput"
    ]
    success = only(
        node
        for node in outputs
        if any(key.arg == "prefill_stats" for key in node.keywords)
    )
    assert (
        ast.unparse(
            only(key.value for key in success.keywords if key.arg == "prefill_stats")
        )
        == "prefill_stats"
    )
    owner = only(
        node
        for node in ast.walk(update)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "should_emit_output"
        and success in list(ast.walk(node))
    )
    assert owner.lineno > emit.end_lineno
    async_tree = ast.parse(read_source("vllm/v1/core/sched/async_scheduler.py"))
    async_class = definition(async_tree, "AsyncScheduler")
    assert [ast.unparse(base) for base in async_class.bases] == ["Scheduler"]
    assert not {"schedule", "update_from_output"} & {
        getattr(node, "name", "") for node in async_class.body
    }
    request = ast.parse(read_source("vllm/v1/request.py"))
    initial = only(
        node
        for node in ast.walk(definition(request, "Request.__init__"))
        if isinstance(node, ast.AnnAssign)
        and ast.unparse(node.target) == "self.prefill_stats"
    )
    processor = ast.parse(read_source("vllm/v1/engine/output_processor.py"))
    projection = only(
        node
        for node in ast.walk(definition(processor, "OutputProcessor.process_outputs"))
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "req_state.is_prefilling"
    )
    stats = ast.parse(read_source("vllm/v1/metrics/stats.py"))
    return {
        "first_capture": ast.unparse(capture),
        "emit_condition": ast.unparse(should_emit.value),
        "emit_statistics": ast.unparse(emit),
        "initial_statistics": ast.unparse(initial),
        "take_statistics": ast.unparse(
            definition(request, "Request.take_prefill_stats")
        ),
        "cache_arithmetic": ast.unparse(definition(stats, "PrefillStats")),
        "project_statistics": ast.unparse(projection),
    }


def project_contract(repository):
    releases = {}
    for release, blobs in CAUSAL_CACHE_SOURCE_BLOBS.items():
        revision = CAUSAL_RELEASE_COMMITS[release]
        sources = {}
        for path, expected in blobs:
            data = subprocess.check_output(
                ["git", "-C", str(repository), "show", f"{revision}:{path}"], timeout=10
            )
            assert git_blob_id(data) == expected
            sources[path] = data.decode()
        releases[release] = {
            "commit": revision,
            "blobs": dict(blobs),
            "projection": project_release(sources.__getitem__),
        }
    return {
        "_generated": "Do not edit; regenerate with python -m dynamo.vllm.tests.decision_cache_contract --source-repo <repo>",
        "scope": "Cache-statistics source equivalence only; not full runtime or GPU/model qualification.",
        "releases": releases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    contract = project_contract(arguments.source_repo)
    if arguments.check:
        assert (
            json.loads(CONTRACT_PATH.read_bytes()) == contract
        ), "Cache provenance projection is stale"
    else:
        CONTRACT_PATH.write_text(json.dumps(contract, indent=2) + "\n")


if __name__ == "__main__":
    main()
