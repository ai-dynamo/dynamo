# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for schema-aware DynamoGraphDeployment helpers."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import httpx
import kr8s
import pytest
import requests
import yaml

from tests.deploy.dgd_utils import DeploymentSpec, ManagedDeployment

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


def managed_deployment(tmp_path) -> ManagedDeployment:
    spec = SimpleNamespace(name="test-deployment", services=[], api_version="v1beta1")
    return ManagedDeployment(
        log_dir=str(tmp_path),
        deployment_spec=spec,  # type: ignore[arg-type]
        namespace="default",
    )


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


@pytest.mark.parametrize(
    "connection_error",
    [
        httpx.RemoteProtocolError("tunnel disconnected"),
        kr8s.APITimeoutError("vCluster request timed out"),
    ],
)
def test_get_pods_retries_transient_vcluster_disconnect(
    monkeypatch, tmp_path, connection_error
) -> None:
    deployment = managed_deployment(tmp_path)
    pod = MagicMock()
    get_pods = MagicMock(side_effect=[connection_error, [pod]])
    sleep = MagicMock()
    monkeypatch.setattr("tests.deploy.dgd_utils.kr8s.get", get_pods)
    monkeypatch.setattr("tests.deploy.vcluster_utils.time.sleep", sleep)

    result = deployment.get_pods(["Frontend"])

    assert result == {"Frontend": [pod]}
    assert get_pods.call_count == 2
    sleep.assert_called_once_with(5)


async def test_delete_deployment_retries_vcluster_connection_failure(
    monkeypatch, tmp_path
) -> None:
    deployment = managed_deployment(tmp_path)
    connection_key = MagicMock(host="127.0.0.1", port=8443, ssl=True)
    deployment._custom_api = MagicMock()
    deployment._custom_api.delete_namespaced_custom_object = AsyncMock(
        side_effect=[
            aiohttp.ClientConnectorError(
                connection_key,
                ConnectionRefusedError(111, "vCluster tunnel unavailable"),
            ),
            None,
        ]
    )
    sleep = AsyncMock()
    monkeypatch.setattr("tests.deploy.vcluster_utils.asyncio.sleep", sleep)

    await deployment._delete_deployment()

    assert deployment._custom_api.delete_namespaced_custom_object.await_count == 2
    sleep.assert_awaited_once_with(5)


@pytest.mark.parametrize(
    "cleanup_error",
    [
        httpx.ConnectError("tunnel unavailable"),
        kr8s.APITimeoutError("vCluster request timed out"),
    ],
)
async def test_context_exit_preserves_original_error_when_cleanup_fails(
    tmp_path, cleanup_error
) -> None:
    deployment = managed_deployment(tmp_path)
    deployment._cleanup = AsyncMock(side_effect=cleanup_error)

    result = await deployment.__aexit__(ValueError, ValueError("test failed"), None)

    assert result is False


async def test_context_enter_preserves_setup_error_when_cleanup_times_out(
    tmp_path,
) -> None:
    deployment = managed_deployment(tmp_path)
    deployment._init_kubernetes = AsyncMock(side_effect=ValueError("setup failed"))
    deployment._cleanup = AsyncMock(
        side_effect=kr8s.APITimeoutError("vCluster request timed out")
    )

    with pytest.raises(ValueError, match="setup failed"):
        await deployment.__aenter__()


async def test_context_exit_reraises_unexpected_cleanup_error(tmp_path) -> None:
    deployment = managed_deployment(tmp_path)
    deployment._cleanup = AsyncMock(side_effect=RuntimeError("cleanup defect"))

    with pytest.raises(RuntimeError, match="cleanup defect"):
        await deployment.__aexit__(ValueError, ValueError("test failed"), None)


@pytest.mark.parametrize(
    "transport_error",
    [
        requests.ConnectionError("forward dropped"),
        requests.Timeout("forward stalled"),
    ],
)
def test_request_rebuilds_port_forward_after_transport_failure(
    monkeypatch, tmp_path, transport_error
) -> None:
    deployment = managed_deployment(tmp_path)
    original_port_forward = MagicMock(local_port=31001)
    replacement_port_forward = MagicMock(local_port=31002)
    deployment.port_forward = MagicMock(return_value=replacement_port_forward)
    response = MagicMock(spec=requests.Response)
    request_sender = MagicMock(side_effect=[transport_error, response])
    sleep = MagicMock()
    monkeypatch.setattr("tests.deploy.dgd_utils.time.sleep", sleep)

    result = deployment.send_request_with_port_forward_retry(
        pod=MagicMock(),
        remote_port=8000,
        endpoint="/v1/chat/completions",
        payload={"model": "test"},
        timeout=120,
        port_forward=original_port_forward,
        request_sender=request_sender,
    )

    assert result is response
    assert (
        request_sender.call_args_list[0].args[0].startswith("http://localhost:31001/")
    )
    assert (
        request_sender.call_args_list[1].args[0].startswith("http://localhost:31002/")
    )
    original_port_forward.stop.assert_called_once_with()
    deployment.port_forward.assert_called_once()
    sleep.assert_called_once_with(5)


def test_request_reraises_unexpected_port_forward_stop_error(tmp_path) -> None:
    deployment = managed_deployment(tmp_path)
    original_port_forward = MagicMock(local_port=31001)
    original_port_forward.stop.side_effect = ValueError("stop defect")
    request_sender = MagicMock(side_effect=requests.ConnectionError("forward dropped"))

    with pytest.raises(ValueError, match="stop defect"):
        deployment.send_request_with_port_forward_retry(
            pod=MagicMock(),
            remote_port=8000,
            endpoint="/v1/chat/completions",
            payload={"model": "test"},
            timeout=120,
            port_forward=original_port_forward,
            request_sender=request_sender,
        )


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
