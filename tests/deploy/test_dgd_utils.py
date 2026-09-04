# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for schema-aware DynamoGraphDeployment helpers."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml

from tests.deploy.dgd_utils import DeploymentSpec, ManagedDeployment

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


def test_logging_config_reads_existing_v1beta1_env(tmp_path) -> None:
    """Recognize JSONL logging already declared in a v1beta1 manifest."""
    manifest = {
        "apiVersion": "nvidia.com/v1beta1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": "logging-test"},
        "spec": {
            "components": [],
            "env": [{"name": "DYN_LOGGING_JSONL", "value": "1"}],
        },
    }
    manifest_path = tmp_path / "deploy.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest))

    deployment_spec = DeploymentSpec(str(manifest_path))

    assert deployment_spec.get_logging_config()["jsonl_enabled"] is True


async def test_in_flight_restart_preserves_bounded_previous_log(tmp_path) -> None:
    """Keep a bounded previous-instance log before Kubernetes rotates again."""
    deployment = ManagedDeployment(
        log_dir=str(tmp_path),
        deployment_spec=SimpleNamespace(name="test-dgd"),
        namespace="default",
    )
    terminated = SimpleNamespace(reason="Error", exit_code=1)
    container_status = SimpleNamespace(
        name="main",
        restart_count=1,
        last_state=SimpleNamespace(terminated=terminated),
    )
    pod = SimpleNamespace(
        metadata=SimpleNamespace(name="worker-0"),
        status=SimpleNamespace(container_statuses=[container_status]),
    )
    deployment._core_api = SimpleNamespace(
        list_namespaced_pod=AsyncMock(return_value=SimpleNamespace(items=[pod])),
        read_namespaced_pod_log=AsyncMock(
            return_value="first line\nsecond line\nthird line\n"
        ),
    )

    warnings = await deployment._dump_in_flight_restart_logs(prev_log_tail_lines=2)

    assert len(warnings) == 1
    assert "first line" not in warnings[0]
    assert "second line" in warnings[0]
    assert "third line" in warnings[0]
    preserved = tmp_path / "restarts" / "worker-0.main.restart-1.previous.log"
    assert preserved.read_text() == "first line\nsecond line\nthird line\n"
    deployment._core_api.read_namespaced_pod_log.assert_awaited_once_with(
        name="worker-0",
        namespace="default",
        container="main",
        previous=True,
        tail_lines=50000,
    )


def _shell_style_manifest(tmp_path, command, script, name="lc-test"):
    """A v1beta1 DGD whose worker is launched through a shell."""
    manifest = {
        "apiVersion": "nvidia.com/v1beta1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": name},
        "spec": {
            "components": [
                {
                    "name": "Worker",
                    "podTemplate": {
                        "spec": {
                            "containers": [
                                {
                                    "name": "main",
                                    "command": command,
                                    "args": [script],
                                }
                            ]
                        }
                    },
                }
            ]
        },
    }
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(manifest))
    return DeploymentSpec(str(path))


def _worker_container(spec):
    component = spec._deployment_spec["spec"]["components"][0]
    return component["podTemplate"]["spec"]["containers"][0]


@pytest.mark.parametrize(
    "flag, expected",
    [
        ("-c", True),
        ("-lc", True),
        ("-ec", True),
        ("-euxc", True),
        ("--config", False),
        ("-l", False),
        ("-", False),
    ],
)
def test_shell_command_flag_matches_clusters_ending_in_c(flag, expected) -> None:
    """``-lc`` invokes a shell exactly as ``-c`` does; ``--config`` does not."""
    from tests.deploy.dgd_utils import ServiceSpec

    assert ServiceSpec._is_shell_command_flag(flag) is expected


def test_login_shell_worker_reports_its_model(tmp_path) -> None:
    """A ``-lc`` worker's ``--model`` is readable, not hidden in one string.

    Treating ``sh -lc`` as argv-style left the whole command as a single token,
    so scanning for ``--model`` by equality found nothing and ``.model``
    reported ``None`` for a worker that plainly declares one.
    """
    spec = _shell_style_manifest(
        tmp_path,
        ["/bin/bash", "-lc"],
        "python3 -m dynamo.vllm --model Qwen/Qwen3-0.6B --tp 1",
    )

    assert spec["Worker"].model == "Qwen/Qwen3-0.6B"


def test_login_shell_worker_model_is_actually_rewritten(tmp_path) -> None:
    """``set_model`` on a ``-lc`` worker changes what the pod will serve."""
    spec = _shell_style_manifest(
        tmp_path,
        ["/bin/bash", "-lc"],
        "python3 -m dynamo.vllm --model Qwen/Qwen3-0.6B --tp 1",
    )

    spec.set_model("meta-llama/Llama-3.1-8B")

    assert spec["Worker"].model == "meta-llama/Llama-3.1-8B"
    args = _worker_container(spec)["args"]
    assert len(args) == 1, f"shell contract broken, args must stay one string: {args}"
    assert "Qwen/Qwen3-0.6B" not in args[0]


def test_added_flag_stays_inside_the_login_shell_command(tmp_path) -> None:
    """An added flag must land in the command, not in the shell's ``$0``/``$1``.

    Writing argv tokens back as extra list entries yields
    ``args: ["<command>", "--max-model-len", "1024"]``; a shell binds those to
    positional parameters and the worker never sees the flag.
    """
    spec = _shell_style_manifest(
        tmp_path,
        ["/bin/bash", "-lc"],
        "python3 -m dynamo.vllm --model Qwen/Qwen3-0.6B",
    )

    spec.add_arg_to_service("Worker", "--max-model-len", "1024")

    args = _worker_container(spec)["args"]
    assert len(args) == 1, f"flag escaped the shell command string: {args}"
    assert "--max-model-len 1024" in args[0]


@pytest.mark.parametrize("shell_flag", ["-c", "-lc"])
def test_shell_operators_survive_a_rewrite_unquoted(tmp_path, shell_flag) -> None:
    """``&&`` must stay an operator, not become a literal argument.

    Re-joining tokens through a quoter that treats ``&&`` as an ordinary word
    emits ``'&&'``, collapsing ``ulimit -l unlimited && exec python3 …`` into a
    single ``ulimit`` call whose third argument is the string ``&&``. The pod
    then never starts the worker at all. This affected ``-c`` containers
    already; extending the predicate to ``-lc`` would have widened it.
    """
    spec = _shell_style_manifest(
        tmp_path,
        ["/bin/bash", shell_flag],
        "ulimit -l unlimited && exec python3 -m dynamo.sglang "
        "--model-path Qwen/Qwen3-0.6B",
        name=f"ops{shell_flag.strip('-')}",
    )

    spec.set_model("deepseek-ai/DeepSeek-V3")

    rewritten = _worker_container(spec)["args"][0]
    # The rewrite must have happened -- otherwise this passes by doing nothing.
    assert spec["Worker"].model == "deepseek-ai/DeepSeek-V3"
    assert "Qwen/Qwen3-0.6B" not in rewritten
    assert "'&&'" not in rewritten, rewritten
    assert rewritten.startswith("ulimit -l unlimited && exec python3"), rewritten


def test_comment_lines_do_not_break_tokenisation(tmp_path) -> None:
    """An apostrophe inside a ``#`` comment must not read as an open quote.

    ``shlex`` has no notion of comments, so ``# Dynamo's adapter`` opened a
    quote that never closed and tokenisation raised ``No closing quotation``.
    """
    spec = _shell_style_manifest(
        tmp_path,
        ["/bin/bash", "-lc"],
        "# Dynamo's metrics adapter is stale for this image\n"
        "exec python3 -m dynamo.vllm --model Qwen/Qwen3-0.6B",
    )

    assert spec["Worker"].model == "Qwen/Qwen3-0.6B"


def test_argv_style_worker_is_untouched(tmp_path) -> None:
    """Widening the shell predicate must not reclassify argv-style workers."""
    manifest = {
        "apiVersion": "nvidia.com/v1beta1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": "argv-test"},
        "spec": {
            "components": [
                {
                    "name": "Worker",
                    "podTemplate": {
                        "spec": {
                            "containers": [
                                {
                                    "name": "main",
                                    "command": ["python3", "-m", "dynamo.vllm"],
                                    "args": ["--model", "Qwen/Qwen3-0.6B"],
                                }
                            ]
                        }
                    },
                }
            ]
        },
    }
    path = tmp_path / "argv.yaml"
    path.write_text(yaml.safe_dump(manifest))
    spec = DeploymentSpec(str(path))

    spec.add_arg_to_service("Worker", "--max-model-len", "1024")

    assert _worker_container(spec)["args"] == [
        "--model",
        "Qwen/Qwen3-0.6B",
        "--max-model-len",
        "1024",
    ]
