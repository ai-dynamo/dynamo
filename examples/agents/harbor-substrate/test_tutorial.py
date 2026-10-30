# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Credential isolation, task fidelity, and recipe-preservation regressions."""

import copy
import hashlib
import json
import logging
import sys
import tomllib
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from add_tracing import main as add_tracing_main
from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.models.agent.context import AgentContext
from harbor.models.job.config import JobConfig
from harbor.models.task.config import EnvironmentConfig, TaskConfig
from harbor.models.trial.paths import TrialPaths
from prepare_workload import prepare
from substrate_environment import SubstrateEnvironment
from tutorial_pi import TutorialPi

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0]


@pytest.mark.asyncio
@pytest.mark.parametrize("return_code", [0, 23], ids=["success", "failed-command"])
async def test_model_key_reaches_guest_without_entering_exec_logs(
    tmp_path, monkeypatch, caplog, return_code
):
    secret = "synthetic-model-credential-for-test"
    monkeypatch.setenv("MODEL_API_KEY", secret)
    caplog.set_level(logging.DEBUG)
    policy = tmp_path / "egress.yaml"
    policy.write_text("rules: []\n")
    environment = SubstrateEnvironment(
        endpoint="unused.invalid",
        template="test-template",
        task_name="test-task",
        kube_context="unused",
        egress_policy=str(policy),
        environment_dir=tmp_path,
        environment_name="test-task",
        session_id="test-environment",
        trial_paths=TrialPaths(tmp_path / "trial"),
        task_env_config=EnvironmentConfig(),
    )
    process_calls = []

    class RemoteProcess:
        async def output(self, follow):
            yield SimpleNamespace(
                stdout=b"",
                stderr=b"",
                exit=SimpleNamespace(exit_code=return_code),
            )

    class RemoteEnvironment:
        async def start_process(self, command, **kwargs):
            process_calls.append((command, kwargs))
            return RemoteProcess()

    # Replace only the remote transport; run the actual agent, Harbor logging,
    # scoped environment merge, and Substrate command adapter.
    environment._env = RemoteEnvironment()
    agent = TutorialPi(
        logs_dir=tmp_path / "agent",
        model_name="dynamo/test-model",
        version="0.72.1",
        preinstalled=True,
        extra_env={
            "DYNAMO_BASE_URL": "https://model.example/v1",
            "DYN_AGENT_SESSION_FINAL": "0",
            "DYNAMO_CONTEXT_WINDOW": "32768",
            "DYNAMO_MAX_TOKENS": "4096",
        },
    )
    agent.session_id = "test-agent-session"
    # Harbor Trial scopes agent.env around the run; these settings must reach
    # Pi even though DynamoPi._dynamo_env does not enumerate model limits.
    with environment.scoped_exec_env(agent.extra_env):
        if return_code:
            with pytest.raises(NonZeroAgentExitCodeError):
                await agent.run("Inspect the repository.", environment, AgentContext())
        else:
            await agent.run("Inspect the repository.", environment, AgentContext())
        assert "DYNAMO_API_KEY" not in environment._merge_env({})
    assert environment._merge_env({}) is None
    assert len(process_calls) == 1
    command, kwargs = process_calls[0]
    assert kwargs["env"]["DYNAMO_API_KEY"] == secret
    assert kwargs["env"]["DYNAMO_CONTEXT_WINDOW"] == "32768"
    assert kwargs["env"]["DYNAMO_MAX_TOKENS"] == "4096"
    assert kwargs["env"]["DYN_AGENT_SESSION_ID"] == agent.session_id
    assert secret not in repr(command)
    command_records = [record for record in caplog.records if "env" in vars(record)]
    assert command_records, "The test must exercise Harbor's explicit env logging"
    assert all("DYNAMO_API_KEY" not in record.env for record in command_records)
    assert all(secret not in repr(vars(record)) for record in caplog.records)
    assert secret not in repr(agent.extra_env)


@pytest.mark.parametrize(
    "golden,endpoint_kind,trace", [(False, "external", False), (True, "dynamo", True)]
)
def test_preparation_preserves_task_and_freezes_image_template_mapping(
    tmp_path, monkeypatch, golden, endpoint_kind, trace
):
    monkeypatch.setenv("MODEL_API_KEY", "synthetic-must-not-enter-job")
    source = tmp_path / "dataset" / "project__case-1"
    (source / "tests").mkdir(parents=True)
    (source / "environment").mkdir()
    (source / "task.toml").write_text(
        'schema_version = "1.3"\n'
        "[agent]\ntimeout_sec = 731\n"
        "[verifier]\ntimeout_sec = 419\n"
        '[[verifier.collect]]\ncommand = "true"\nservice = "main"\n'
        '[environment]\nworkdir = "/testbed"\n'
    )
    (source / "instruction.md").write_text("Fix the reported behavior.\n")
    (source / "tests" / "test.sh").write_text("#!/bin/bash\nexit 0\n")
    (source / "tests" / "test.sh").chmod(0o755)
    (source / "tests" / "config.json").write_text(
        json.dumps({"base_commit": "a" * 40, "unchanged_verifier_setting": True})
    )
    (source / "environment" / "Dockerfile").write_text("FROM scratch\n")
    original = {
        path.relative_to(source): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }
    prepared_image = "registry.example/task@sha256:" + "1" * 64
    guest_image = "registry.example/guest@sha256:" + "2" * 64
    images = tmp_path / "images.json"
    images.write_text(json.dumps({source.name: prepared_image}))
    output = tmp_path / "prepared"
    prepare(
        Namespace(
            dataset=source.parent,
            images=images,
            guest_image=guest_image,
            output=output,
            base_url="https://model.example/v1",
            model="test-model",
            context_window=32768,
            max_tokens=4096,
            endpoint_kind=endpoint_kind,
            concurrency=2,
            job_name="test-job",
            golden=golden,
            trace=trace,
        )
    )
    copied = output / "tasks" / source.name
    for relative, content in original.items():
        assert (source / relative).read_bytes() == content
        if relative != Path("task.toml"):
            assert (copied / relative).read_bytes() == content
    assert (copied / "tests/test.sh").stat().st_mode & 0o777 == 0o755
    original_config = tomllib.loads(original[Path("task.toml")].decode())
    copied_config = tomllib.loads((copied / "task.toml").read_text())
    task = TaskConfig.model_validate(copy.deepcopy(copied_config))
    assert task.agent.timeout_sec == 731
    assert task.verifier.timeout_sec == 419
    capture = copied_config["verifier"]["collect"].pop()
    assert copied_config == original_config
    assert "a" * 40 in capture["command"]
    assert "/logs/artifacts/candidate" in capture["command"]

    job_text = (output / "job.json").read_text()
    job = JobConfig.model_validate_json(job_text)
    assert "synthetic-must-not-enter-job" not in job_text
    assert "API_KEY" not in job_text
    assert job.n_concurrent_trials == 2
    assert job.agents[0].env["DYN_AGENT_SESSION_FINAL"] == (
        "1" if endpoint_kind == "dynamo" else "0"
    )
    assert job.agents[0].env["DYNAMO_CONTEXT_WINDOW"] == "32768"
    assert job.agents[0].env["DYNAMO_MAX_TOKENS"] == "4096"
    assert [entry.path for entry in job.tasks] == [Path("/work/tasks") / source.name]
    records = json.loads((output / "task-lock.json").read_text())
    assert len(records) == 1
    record = records[0]
    assert record["image"] == prepared_image
    assert job.environment.kwargs["template_map"] == {source.name: record["template"]}
    template = yaml.safe_load(
        (output / "templates" / f"{record['template']}.yaml").read_text()
    )
    assert template["containers"][0]["image"] == prepared_image
    assert template["volumes"][0]["image"]["reference"] == guest_image
    assert template["snapshotConfig"]["goldenSnapshotConfig"]["mode"] == (
        "GOLDEN_SNAPSHOT_MODE_ENABLED" if golden else "GOLDEN_SNAPSHOT_MODE_DISABLED"
    )
    assert record["sha256"] == {
        str(path.relative_to(copied)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in copied.rglob("*")
        if path.is_file()
    }


def test_tracing_preserves_recipe_workers_and_frontend_serving_configuration(
    tmp_path, monkeypatch
):
    recipe = (
        Path(__file__).resolve().parents[3]
        / "recipes/glm-5.3-flash/vllm/agg-h200-agentic/deploy.yaml"
    )
    originals = list(yaml.safe_load_all(recipe.read_text()))
    output = tmp_path / "traced.yaml"
    monkeypatch.setattr(sys, "argv", ["add_tracing.py", str(recipe), str(output)])
    add_tracing_main()
    traced_documents = list(yaml.safe_load_all(output.read_text()))
    assert len(traced_documents) == len(originals)
    for before, after in zip(originals, traced_documents, strict=True):
        assert after["kind"] == before["kind"]
        if before["kind"] != "DynamoGraphDeployment":
            assert after == before
    original = next(
        item for item in originals if item["kind"] == "DynamoGraphDeployment"
    )
    traced = next(
        item for item in traced_documents if item["kind"] == "DynamoGraphDeployment"
    )
    assert traced["apiVersion"] == original["apiVersion"]
    assert traced["metadata"] == original["metadata"]
    assert {
        key: value for key, value in traced["spec"].items() if key != "components"
    } == {key: value for key, value in original["spec"].items() if key != "components"}
    workers_before = [
        component
        for component in original["spec"]["components"]
        if component["type"] != "frontend"
    ]
    workers_after = [
        component
        for component in traced["spec"]["components"]
        if component["type"] != "frontend"
    ]
    assert workers_after == workers_before
    frontend_before = next(
        component
        for component in original["spec"]["components"]
        if component["type"] == "frontend"
    )
    frontend_after = next(
        component
        for component in traced["spec"]["components"]
        if component["type"] == "frontend"
    )
    pod_before = frontend_before["podTemplate"]["spec"]
    pod_after = frontend_after["podTemplate"]["spec"]
    main_before = next(
        container
        for container in pod_before["containers"]
        if container["name"] == "main"
    )
    main_after = next(
        container
        for container in pod_after["containers"]
        if container["name"] == "main"
    )
    for key, value in main_before.items():
        if key not in {"env", "volumeMounts"}:
            assert main_after[key] == value
    assert all(entry in main_after["env"] for entry in main_before.get("env", []))
    assert all(
        mount in main_after["volumeMounts"]
        for mount in main_before.get("volumeMounts", [])
    )
    trace_env = {entry["name"]: entry.get("value") for entry in main_after["env"]}
    assert trace_env["DYN_REQUEST_TRACE"] == "1"
    assert trace_env["DYN_REQUEST_TRACE_OUTPUT_PATH"].startswith("/traces/")
    assert trace_env["DYN_REQUEST_TRACE_FILE_FORMAT"] == "jsonl"
    sidecar = next(
        container
        for container in pod_after["containers"]
        if container["name"] == "tool-events-tls"
    )
    assert any(
        mount["mountPath"] == "/tls" and mount["readOnly"]
        for mount in sidecar["volumeMounts"]
    )
    assert any(
        volume.get("secret", {}).get("secretName") == "harbor-tool-trace-tls"
        for volume in pod_after["volumes"]
    )
