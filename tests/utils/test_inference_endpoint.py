# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the deployment-agnostic addressing layer."""

import ast
import logging
import subprocess
import sys
from pathlib import Path

import pytest

from tests.utils.inference_endpoint import (
    DEFAULT_SETTLE_SECONDS,
    DeploymentEndpoints,
    InferenceEndpoint,
    NotServingError,
    wait_until_serving,
)
from tests.utils.payload_builder import deployment_smoke_chat_payload
from tests.utils.payloads import ChatPayload

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.parallel,
]


class _FakeResponse:
    def __init__(self, status_code: int, text: str = "") -> None:
        self.status_code = status_code
        self.text = text


def test_url_joins_without_double_slash():
    ep = InferenceEndpoint(base_url="http://localhost:8000")
    assert ep.url("/v1/models") == "http://localhost:8000/v1/models"
    assert ep.url("v1/models") == "http://localhost:8000/v1/models"


def test_trailing_slash_is_normalized_away():
    assert (
        InferenceEndpoint(base_url="https://dynamo.example.com/").base_url
        == "https://dynamo.example.com"
    )


def test_from_port_builds_localhost_url():
    ep = InferenceEndpoint.from_port(31337, model="m")
    assert ep.base_url == "http://localhost:31337"
    assert ep.model == "m"


def test_non_http_base_url_is_supported():
    ep = InferenceEndpoint(base_url="https://frontend.example.com")
    assert ep.url("v1/chat/completions") == (
        "https://frontend.example.com/v1/chat/completions"
    )


@pytest.mark.parametrize("bad", ["", "localhost:8000", "/v1"])
def test_rejects_addresses_without_a_scheme(bad):
    with pytest.raises(ValueError):
        InferenceEndpoint(base_url=bad)


def test_with_headers_merges_and_does_not_mutate():
    base = InferenceEndpoint(base_url="http://x:1", headers={"A": "1"})
    derived = base.with_headers({"B": "2"})
    assert derived.headers == {"A": "1", "B": "2"}
    assert base.headers == {"A": "1"}


def test_worker_index_error_names_the_available_count():
    endpoints = DeploymentEndpoints(
        frontend=InferenceEndpoint(base_url="http://x:1"),
        workers=(InferenceEndpoint(base_url="http://x:2"),),
    )
    assert endpoints.worker(0).base_url == "http://x:2"
    with pytest.raises(IndexError, match="1 worker endpoint"):
        endpoints.worker(1)


# --- payload binding ---------------------------------------------------------


def test_payload_url_defaults_to_host_port_when_unbound():
    payload = ChatPayload(body={}, expected_response=[], expected_log=[], port=1234)
    assert payload.url() == "http://localhost:1234/v1/chat/completions"


def test_bind_targets_the_endpoint_and_leaves_the_original_alone():
    payload = ChatPayload(body={}, expected_response=[], expected_log=[], port=1234)
    endpoint = InferenceEndpoint(
        base_url="https://dynamo.example.com", model="m", headers={"Host": "h"}
    )

    bound = payload.bind(endpoint)

    assert bound.url() == "https://dynamo.example.com/v1/chat/completions"
    assert bound.headers == {"Host": "h"}
    assert bound.body["model"] == "m"
    # The shared/parametrized instance must be reusable across cases.
    assert payload.base_url is None
    assert payload.url() == "http://localhost:1234/v1/chat/completions"


def test_bind_keeps_an_explicit_model_over_the_endpoint_default():
    payload = ChatPayload(
        body={"model": "explicit"}, expected_response=[], expected_log=[]
    )
    bound = payload.bind(
        InferenceEndpoint(base_url="http://x:1", model="from-endpoint")
    )
    assert bound.body["model"] == "explicit"


def test_bind_leaves_worker_system_ports_untouched():
    """System ports are topology-dependent; a frontend URL cannot replace them."""
    payload = ChatPayload(body={}, expected_response=[], expected_log=[])
    payload.system_ports = [8081, 8082]
    bound = payload.bind(InferenceEndpoint(base_url="http://x:1"))
    assert bound.system_ports == [8081, 8082]


# --- readiness ---------------------------------------------------------------


def test_wait_until_serving_returns_once_a_request_succeeds(monkeypatch):
    calls = []

    def fake_send_request(url, payload, **kwargs):
        calls.append(url)
        return _FakeResponse(503 if len(calls) < 3 else 200)

    monkeypatch.setattr(
        "tests.utils.inference_endpoint.send_request", fake_send_request
    )
    monkeypatch.setattr("tests.utils.inference_endpoint.time.sleep", lambda _s: None)

    wait_until_serving(
        InferenceEndpoint(base_url="http://x:1", model="m"),
        timeout=30,
        poll_interval=0,
    )
    assert len(calls) == 3
    assert calls[0] == "http://x:1/v1/chat/completions"


def test_wait_until_serving_settles_before_returning(monkeypatch):
    """The first 200 can land while the rest of the graph is still coming up."""
    slept = []
    monkeypatch.setattr(
        "tests.utils.inference_endpoint.send_request",
        lambda url, payload, **kwargs: _FakeResponse(200),
    )
    monkeypatch.setattr(
        "tests.utils.inference_endpoint.time.sleep", lambda s: slept.append(s)
    )

    endpoint = InferenceEndpoint(base_url="http://x:1", model="m")
    wait_until_serving(endpoint, timeout=30)
    assert slept == [DEFAULT_SETTLE_SECONDS]

    slept.clear()
    wait_until_serving(endpoint, timeout=30, settle_seconds=0)
    assert slept == [], "settle_seconds=0 opts out"


def test_wait_until_serving_raises_with_the_last_failure(monkeypatch):
    monkeypatch.setattr(
        "tests.utils.inference_endpoint.send_request",
        lambda url, payload, **kwargs: _FakeResponse(503, "worker not ready"),
    )
    monkeypatch.setattr("tests.utils.inference_endpoint.time.sleep", lambda _s: None)

    # Third call onward reports the deadline as passed so the loop terminates
    # without real sleeping.
    times = iter([0.0] + [0.0, 0.5, 99.0, 99.0, 99.0] * 10)
    monkeypatch.setattr(
        "tests.utils.inference_endpoint.time.monotonic", lambda: next(times)
    )

    with pytest.raises(NotServingError, match="worker not ready"):
        wait_until_serving(
            InferenceEndpoint(base_url="http://x:1", model="m"),
            timeout=10,
            poll_interval=0,
        )


def test_wait_until_serving_surfaces_transport_errors(monkeypatch):
    def boom(url, payload, **kwargs):
        raise ConnectionError("connection refused")

    monkeypatch.setattr("tests.utils.inference_endpoint.send_request", boom)
    monkeypatch.setattr("tests.utils.inference_endpoint.time.sleep", lambda _s: None)
    times = iter([0.0] + [0.0, 99.0, 99.0, 99.0] * 10)
    monkeypatch.setattr(
        "tests.utils.inference_endpoint.time.monotonic", lambda: next(times)
    )

    with pytest.raises(NotServingError, match="connection refused"):
        wait_until_serving(
            InferenceEndpoint(base_url="http://x:1", model="m"),
            timeout=1,
            poll_interval=0,
        )


def test_probe_needs_a_model(monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    with pytest.raises(ValueError, match="model"):
        wait_until_serving(InferenceEndpoint(base_url="http://x:1"), timeout=1)


# --- the deploy-runner import constraint -------------------------------------

# The Kubernetes deploy job runs pytest on a bare GitHub runner that installs
# container/deps/requirements.test.txt and NOT the ai-dynamo wheel
# (.github/actions/dynamo-deploy-test/action.yml). Any module in the shared
# verification layer that imports `dynamo` at module scope would make the deploy
# tests uncollectable there. Checked statically so the failure shows up here
# rather than as an ImportError in a cluster job.
_MUST_NOT_IMPORT_DYNAMO = (
    "tests/utils/client.py",
    "tests/utils/inference_endpoint.py",
    "tests/utils/payload_builder.py",
    "tests/utils/payloads.py",
    "tests/utils/prometheus.py",
    "tests/utils/router_nvext.py",
    "tests/utils/verification.py",
    "tests/deploy/conftest.py",
    "tests/deploy/test_dgd.py",
    "tests/deploy/test_dynamocheckpoint.py",
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


class _ImportsAtModuleScope(ast.NodeVisitor):
    """Collect root module names imported when the module is first executed.

    Descends into class bodies (those run at import) but not into function
    bodies -- a deferred import inside a function is exactly the escape hatch
    this rule allows.
    """

    def __init__(self) -> None:
        self.names: set[str] = set()
        self.lines: dict[str, int] = {}

    def _record(self, name: str, lineno: int) -> None:
        root = name.split(".")[0]
        self.names.add(root)
        self.lines.setdefault(root, lineno)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._record(alias.name, node.lineno)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module and node.level == 0:
            self._record(node.module, node.lineno)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return


@pytest.mark.parametrize("rel_path", _MUST_NOT_IMPORT_DYNAMO)
def test_shared_verification_layer_does_not_import_dynamo(rel_path):
    path = _REPO_ROOT / rel_path
    assert path.exists(), f"{rel_path} moved; update this list"

    visitor = _ImportsAtModuleScope()
    visitor.visit(ast.parse(path.read_text()))

    assert "dynamo" not in visitor.names, (
        f"{rel_path}:{visitor.lines.get('dynamo')} imports `dynamo` at module "
        "scope. The Kubernetes deploy job runs pytest without the ai-dynamo "
        "wheel installed, so this breaks collection there. Move the import "
        "inside the function that needs it."
    )


def test_deploy_test_modules_import_without_the_dynamo_package(tmp_path):
    """Authoritative version of the check above: import them for real.

    The static check points at the offending line; this one proves the whole
    transitive graph is clean, including any import the AST scan cannot see.
    """
    blocker = tmp_path / "block_dynamo.py"
    blocker.write_text(
        "import sys\n"
        "from importlib.abc import MetaPathFinder\n"
        "class _Blocker(MetaPathFinder):\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'dynamo' or name.startswith('dynamo.'):\n"
        "            raise ImportError('BLOCKED: ' + name)\n"
        "        return None\n"
        "sys.meta_path.insert(0, _Blocker())\n"
    )
    script = (
        f"import sys; sys.path.insert(0, {str(tmp_path)!r}); import block_dynamo\n"
        f"sys.path.insert(0, {str(_REPO_ROOT)!r})\n"
        "import tests.deploy.conftest\n"
        "import tests.deploy.test_dgd\n"
        "import tests.deploy.test_dynamocheckpoint\n"
        "import tests.utils.verification\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )
    assert result.returncode == 0, (
        "The deploy test import chain requires the ai-dynamo package, which the "
        "Kubernetes deploy runner does not install:\n" + result.stderr[-3000:]
    )


def test_deployment_smoke_payload_asserts_model_role_and_length():
    payload = deployment_smoke_chat_payload(model="Qwen/Qwen3-0.6B")
    assert payload.expected_model == "Qwen/Qwen3-0.6B"
    assert payload.expected_role == "assistant"
    assert payload.min_content_length == 100
    assert payload.body["model"] == "Qwen/Qwen3-0.6B"
    assert payload.body["stream"] is False
