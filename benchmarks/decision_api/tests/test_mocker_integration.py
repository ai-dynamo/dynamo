# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in actual AIPerf transport over CPU mocker, not an inference benchmark.

Run in the serving environment with DECISION_PERF_MOCKER_INTEGRATION=1,
DECISION_PERF_TOKENIZER pointing to the pinned local snapshot, and
DECISION_PERF_AIPERF pointing to the isolated, plugin-enabled AIPerf executable.
"""

import json
import os
from pathlib import Path

import pytest
from dynamo_decision_perf.runner import RunSpec, build_command
from dynamo_decision_perf.serving import ServingConfig, serve
from dynamo_decision_perf.workloads import Shape, generate_workloads

pytestmark = [
    pytest.mark.skipif(
        os.getenv("DECISION_PERF_MOCKER_INTEGRATION") != "1",
        reason="explicit local CPU instrument qualification only",
    ),
    pytest.mark.gpu_0,
    pytest.mark.integration,
    pytest.mark.post_merge,
    pytest.mark.timeout(420),
]


def test_real_aiperf_mocker_contracts(tmp_path):
    from transformers import AutoTokenizer

    from tests.utils.managed_process import ManagedProcess

    tokenizer_path = Path(os.environ["DECISION_PERF_TOKENIZER"]).resolve(strict=True)
    executable = Path(os.environ["DECISION_PERF_AIPERF"]).resolve(strict=True)
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), local_files_only=True, trust_remote_code=False
    )
    model = "Qwen/Qwen3.8-27B"
    workloads = tmp_path / "workloads"
    manifest = generate_workloads(
        workloads, tokenizer, model, count=4, shapes=(Shape("base"),)
    )
    assert not manifest["unsupported"]
    config = ServingConfig(
        model_path=tokenizer_path, model=model, log_dir=tmp_path / "serving"
    )
    with serve(config) as server:
        for dialect in ("oai", "sglang_native", "systemone"):
            artifacts = tmp_path / dialect
            spec = RunSpec(
                dialect=dialect,
                url=server.base_url,
                model=model,
                requests=4,
                duration=30,
                timeout=15,
            )
            command = build_command(
                spec, workloads / f"base.{dialect}.json", artifacts, str(executable)
            )
            with ManagedProcess(
                command=command,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                log_dir=str(tmp_path / "load-logs"),
                display_name=f"aiperf-{dialect}",
                terminate_all_matching_process_names=False,
            ) as process:
                assert process.proc.wait(timeout=120) == 0, process.read_logs()
            report = json.loads((artifacts / "profile_export_aiperf.json").read_text())
            assert report["request_count"]["avg"] == 4
            assert report.get("error_request_count", {}).get("avg", 0) == 0
