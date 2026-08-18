# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import inspect
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

# The fixture runner is a sibling script rather than deployable Agent code.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_k8s_hardware_qualification as fixture  # noqa: E402
from run_k8s_hardware_qualification import build_manifests  # noqa: E402


def _load_entrypoint(monkeypatch):
    module_name = "_p2_8_hardware_qualification_entrypoint_test"
    path = Path(__file__).with_name("e2e_k8s_hardware_qualification_entrypoint.py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setitem(sys.modules, "pynvml", types.ModuleType("pynvml"))
    spec.loader.exec_module(module)
    return module


def test_manifest_requests_exclusive_exact_product_gpu_pool():
    config_map, job = build_manifests(
        name="p2q-test",
        run_token="token-123",
        namespace="qualification",
        gpu_product="NVIDIA-GB200",
        gpu_count=4,
        architecture="arm64",
        actuator_mode="nvml-dcgm-parity",
        dcgm_image="dcgm:test",
        runtime_image="python:test",
        hostengine_host="nvidia-dcgm.gpu-operator.svc.cluster.local",
        hostengine_port=5555,
        expected_hostengine_version="4.4.2",
        active_deadline_seconds=1170,
        qualification_timeout_seconds=900,
        source_data={"actuator.py": "source"},
    )

    assert config_map["data"] == {"actuator.py": "source"}
    assert (
        config_map["metadata"]["labels"]["phase2.dynamo.nvidia.com/run-token"]
        == "token-123"
    )
    assert job["spec"]["activeDeadlineSeconds"] == 1170
    pod = job["spec"]["template"]["spec"]
    assert pod["nodeSelector"] == {
        "kubernetes.io/arch": "arm64",
        "nvidia.com/gpu.product": "NVIDIA-GB200",
    }
    qualification = pod["containers"][0]
    assert qualification["resources"]["requests"] == {"nvidia.com/gpu": "4"}
    assert qualification["resources"]["limits"] == {"nvidia.com/gpu": "4"}
    assert qualification["securityContext"]["privileged"] is True
    assert qualification["command"] == [
        "python3",
        "/app/e2e_k8s_hardware_qualification_entrypoint.py",
    ]
    assert qualification["args"][-2:] == [
        "--timeout-seconds",
        "900",
    ]
    assert qualification["args"][-4:-2] == ["--verified-hostengine-version", "4.4.2"]
    assert qualification["args"][4:6] == [
        "--actuator-mode",
        "nvml-dcgm-parity",
    ]
    dcgm_vendor = pod["initContainers"][0]
    assert dcgm_vendor["env"] == [
        {
            "name": "DCGM_HOST",
            "value": "nvidia-dcgm.gpu-operator.svc.cluster.local",
        },
        {"name": "DCGM_PORT", "value": "5555"},
    ]
    assert (
        'dcgmi -v --host "${DCGM_HOST}:${DCGM_PORT}" ' "> /vendor/dcgmi-version.txt"
    ) in dcgm_vendor["args"][0]


def test_nvml_manifest_has_no_dcgm_hostengine_dependency():
    _, job = build_manifests(
        name="p2q-a100",
        run_token="token-a100",
        namespace="qualification",
        gpu_product="NVIDIA-A100-SXM4-80GB",
        gpu_count=8,
        architecture="amd64",
        actuator_mode="nvml",
        dcgm_image=None,
        runtime_image="python:test",
        hostengine_host=None,
        hostengine_port=5555,
        expected_hostengine_version=None,
        active_deadline_seconds=1170,
        qualification_timeout_seconds=900,
        source_data={"actuator.py": "source"},
    )

    pod = job["spec"]["template"]["spec"]
    assert [container["name"] for container in pod["initContainers"]] == [
        "python-dependencies"
    ]
    assert [volume["name"] for volume in pod["volumes"]] == [
        "source",
        "python-deps",
    ]
    qualification = pod["containers"][0]
    assert qualification["args"] == [
        "--expected-gpu-count",
        "8",
        "--gpu-product",
        "NVIDIA-A100-SXM4-80GB",
        "--actuator-mode",
        "nvml",
        "--timeout-seconds",
        "900",
    ]
    assert qualification["env"] == [
        {"name": "PYTHONPATH", "value": "/app:/deps"},
        {"name": "PYTHONUNBUFFERED", "value": "1"},
    ]
    assert {mount["name"] for mount in qualification["volumeMounts"]} == {
        "source",
        "python-deps",
    }


def test_parity_manifest_requires_hostengine_inputs():
    with pytest.raises(ValueError, match="parity mode requires"):
        build_manifests(
            name="p2q-test",
            run_token="token-123",
            namespace="qualification",
            gpu_product="NVIDIA-GB200",
            gpu_count=4,
            architecture="arm64",
            actuator_mode="nvml-dcgm-parity",
            dcgm_image=None,
            runtime_image="python:test",
            hostengine_host=None,
            hostengine_port=5555,
            expected_hostengine_version=None,
            active_deadline_seconds=1170,
            qualification_timeout_seconds=900,
            source_data={"actuator.py": "source"},
        )


def test_entrypoint_nvml_mode_drives_nvml_probe_without_hostengine(monkeypatch):
    entrypoint = _load_entrypoint(monkeypatch)
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(entrypoint.subprocess, "run", fake_run)

    entrypoint._run_parity(
        "/app/e2e_actuator_parity.py",
        "nvml",
        60,
        "--test-watts",
        "300",
    )

    assert calls == [
        (
            [
                sys.executable,
                "/app/e2e_actuator_parity.py",
                "--skip-dcgm",
                "--test-watts",
                "300",
            ],
            {"check": True, "timeout": 60},
        )
    ]


def test_entrypoint_parity_mode_requires_hostengine(monkeypatch):
    entrypoint = _load_entrypoint(monkeypatch)
    with pytest.raises(ValueError, match="requires a DCGM hostengine"):
        entrypoint._run_parity(
            "/app/e2e_actuator_parity.py",
            "nvml-dcgm-parity",
            60,
        )


def test_entrypoint_parses_dcgm_client_and_hostengine_versions(monkeypatch):
    entrypoint = _load_entrypoint(monkeypatch)
    output = """Local build info:
Version : 4.5.3

Hostengine build info:
Version : 4.4.2
"""

    assert entrypoint._parse_dcgm_versions(output) == {
        "client": "4.5.3",
        "hostengine": "4.4.2",
    }


def test_fixture_uses_argv_subprocess_without_shell_execution():
    source = inspect.getsource(fixture)
    assert "shell=True" not in source
    assert "subprocess.run(" in source


def test_source_payload_includes_gate_and_backend_probe_within_configmap_limit():
    power_agent_dir = Path(__file__).resolve().parent.parent
    data, hashes = fixture._source_payload(power_agent_dir)

    assert "gate.py" in data
    assert "e2e_gate_entrypoint.py" in data
    assert "e2e_gate_backend_probe.py" in data
    assert set(hashes) == set(fixture.SOURCE_FILES)
    assert sum(len(content.encode("utf-8")) for content in data.values()) < 900_000


def test_hostengine_version_parser_uses_hostengine_section():
    output = """Local build info:
Version : 4.5.0

Hostengine build info:
Version : 4.4.2
Build ID : 15378
"""
    assert fixture._parse_hostengine_version(output) == "4.4.2"


def test_hostengine_preflight_binds_every_ready_service_endpoint(monkeypatch):
    responses = {
        ("get", "service", "nvidia-dcgm", "-o", "json"): {
            "metadata": {"uid": "service-uid", "resourceVersion": "41"},
            "spec": {
                "type": "ClusterIP",
                "clusterIP": "10.0.0.1",
                "internalTrafficPolicy": "Local",
                "ports": [{"port": 5555, "targetPort": 5555}],
            },
        },
        ("get", "endpoints", "nvidia-dcgm", "-o", "json"): {
            "metadata": {"resourceVersion": "42"},
            "subsets": [
                {
                    "addresses": [
                        {
                            "nodeName": "node-a",
                            "targetRef": {
                                "kind": "Pod",
                                "name": "dcgm-a",
                                "uid": "uid-a",
                            },
                        }
                    ]
                }
            ],
        },
        ("get", "pod", "dcgm-a", "-o", "json"): {
            "metadata": {"uid": "uid-a"},
            "spec": {"nodeName": "node-a"},
            "status": {
                "containerStatuses": [
                    {
                        "name": "nvidia-dcgm-ctr",
                        "ready": True,
                        "image": "dcgm:4.4.2",
                        "imageID": "dcgm@sha256:" + ("a" * 64),
                    }
                ]
            },
        },
    }

    def fake_kubectl(*args, **kwargs):
        command = tuple(args[3:])
        if command[:1] == ("exec",):
            output = "Hostengine build info:\nVersion : 4.4.2\n"
            return subprocess.CompletedProcess(
                args=[], returncode=0, stdout=output, stderr=""
            )
        return subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(responses[command]), stderr=""
        )

    monkeypatch.setattr(fixture, "_kubectl", fake_kubectl)

    result = fixture._verify_hostengine_endpoints(
        "kubectl",
        "context",
        "gpu-operator",
        "nvidia-dcgm",
        "nvidia-dcgm-ctr",
        5555,
        "4.4.2",
        "nvidia-dcgm.gpu-operator.svc.cluster.local",
    )

    assert result["status"] == "PASS"
    assert result["readyEndpointCount"] == 1
    assert result["endpoints"][0] == {
        "pod": "dcgm-a",
        "podUID": "uid-a",
        "node": "node-a",
        "container": "nvidia-dcgm-ctr",
        "image": "dcgm:4.4.2",
        "imageID": "dcgm@sha256:" + ("a" * 64),
        "version": "4.4.2",
    }


def test_hostengine_preflight_rejects_an_unbound_host():
    with pytest.raises(RuntimeError, match="must be the verified Service"):
        fixture._verify_hostengine_endpoints(
            "kubectl",
            "context",
            "gpu-operator",
            "nvidia-dcgm",
            "nvidia-dcgm-ctr",
            5555,
            "4.4.2",
            "different.example",
        )


def test_hostengine_preflight_rejects_a_headless_service(monkeypatch):
    service = {
        "metadata": {"uid": "service-uid"},
        "spec": {
            "type": "ClusterIP",
            "clusterIP": "None",
            "internalTrafficPolicy": "Local",
            "ports": [{"port": 5555, "targetPort": 5555}],
        },
    }
    monkeypatch.setattr(
        fixture,
        "_kubectl",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(service), stderr=""
        ),
    )

    with pytest.raises(RuntimeError, match="non-headless ClusterIP"):
        fixture._verify_hostengine_endpoints(
            "kubectl",
            "context",
            "gpu-operator",
            "nvidia-dcgm",
            "nvidia-dcgm-ctr",
            5555,
            "4.4.2",
            "nvidia-dcgm.gpu-operator.svc.cluster.local",
        )


def test_hostengine_postflight_rejects_endpoint_roll():
    preflight = {
        "serviceUID": "service-uid",
        "endpoints": [
            {
                "podUID": "old-uid",
                "node": "node-a",
                "version": "4.4.2",
                "imageID": "dcgm@sha256:" + ("a" * 64),
            }
        ],
    }
    postflight = {
        "serviceUID": "service-uid",
        "endpoints": [
            {
                "podUID": "new-uid",
                "node": "node-a",
                "version": "4.4.2",
                "imageID": "dcgm@sha256:" + ("a" * 64),
            }
        ],
    }

    with pytest.raises(RuntimeError, match="endpoint identity"):
        fixture._bind_hostengine_postflight(preflight, postflight, "node-a")


def test_kubectl_timeout_is_bounded_and_reported(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", timeout)

    result = fixture._kubectl(
        "kubectl", "context", "namespace", "get", "pods", check=False
    )

    assert result.returncode == 124
    assert "kubectl timed out" in result.stderr


def test_cleanup_failure_is_recorded(monkeypatch):
    monkeypatch.setattr(
        fixture,
        "_kubectl",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="delete denied"
        ),
    )

    result = fixture._cleanup_resources(
        "kubectl", "context", "namespace", "run-token", ["job"]
    )

    assert result == {
        "status": "FAIL",
        "runToken": "run-token",
        "resources": [
            {
                "kind": "job",
                "selector": "phase2.dynamo.nvidia.com/run-token=run-token",
                "returnCode": 1,
                "stderr": "delete denied",
                "verificationReturnCode": 1,
                "verificationStderr": "delete denied",
            }
        ],
    }


def test_cleanup_stops_job_controller_before_verifying_pods(monkeypatch):
    commands = []

    def fake_kubectl(*args, **kwargs):
        commands.append(args[3:])
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(fixture, "_kubectl", fake_kubectl)

    result = fixture._cleanup_resources(
        "kubectl",
        "context",
        "namespace",
        "run-token",
        ["configmap", "job", "pod"],
    )

    delete_commands = [command for command in commands if command[0] == "delete"]
    assert [command[1] for command in delete_commands] == [
        "job",
        "pod",
        "configmap",
    ]
    assert "--cascade=foreground" in delete_commands[0]
    assert result["status"] == "PASS"


def test_pass_evidence_is_published_only_with_cleanup(tmp_path, capsys):
    payload = fixture._publish_successful_evidence(
        tmp_path,
        {"qualification": {"status": "PASS"}},
        {"status": "PASS", "runToken": "token"},
    )

    assert payload["evidenceStatus"] == "PASS"
    assert payload["cleanup"]["status"] == "PASS"
    assert json.loads((tmp_path / "summary.json").read_text()) == payload
    assert capsys.readouterr().out.startswith(fixture.EVIDENCE_PREFIX)


def test_failed_cleanup_cannot_publish_pass_evidence(tmp_path, capsys):
    with pytest.raises(RuntimeError, match="before cleanup succeeds"):
        fixture._publish_successful_evidence(
            tmp_path,
            {"qualification": {"status": "PASS"}},
            {"status": "FAIL", "runToken": "token"},
        )

    assert not (tmp_path / "summary.json").exists()
    assert capsys.readouterr().out == ""
