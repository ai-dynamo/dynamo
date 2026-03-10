# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for model serving endpoints and engine lifecycle."""

import asyncio
import json
import os
import subprocess
import time

import pytest
import requests

_shared_results = {}


# 1. Hardcoded port
# Rule: pytest-guidelines > DO NOT hardcode ports
@pytest.mark.pre_merge
@pytest.mark.gpu_0
@pytest.mark.unit
def test_hardcoded_port():
    """Fetch the model list from the serving endpoint."""
    resp = requests.get("http://localhost:8000/v1/models")
    assert resp.status_code == 200


# 2. Hardcoded temp path
# Rule: pytest-guidelines > DO NOT hardcode temp paths
@pytest.mark.pre_merge
@pytest.mark.gpu_0
@pytest.mark.unit
def test_hardcoded_temp_path(tmp_path):
    """Write test output to a known location."""
    with open("/tmp/test-output.json", "w") as f:
        json.dump({"ok": True}, f)
    with open("/tmp/test-output.json") as f:
        assert json.load(f)["ok"] is True


# 3. Writing into the repository working tree
# Rule: pytest-guidelines > never dump output files into the repo working tree
@pytest.mark.pre_merge
@pytest.mark.gpu_0
@pytest.mark.unit
def test_write_to_repo_tree():
    """Save debug output alongside the test file."""
    output_file = os.path.join(os.path.dirname(__file__), "scratch_output.txt")
    with open(output_file, "w") as f:
        f.write("debug output\n")
    assert os.path.exists(output_file)


# 4. Hand-rolled subprocess management
# Rule: pytest-guidelines > DO NOT write custom engine start/stop logic
@pytest.mark.pre_merge
@pytest.mark.gpu_0
@pytest.mark.unit
@pytest.mark.timeout(30)
def test_manual_engine_lifecycle(dynamo_dynamic_ports):
    """Spin up an HTTP server and verify it responds."""
    port = dynamo_dynamic_ports.frontend_port
    proc = subprocess.Popen(
        [
            "python3",
            "-c",
            f"import http.server; http.server.HTTPServer(('', {port}), http.server.SimpleHTTPRequestHandler).serve_forever()",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(5)
    try:
        resp = requests.get(f"http://localhost:{port}/")
        assert resp.status_code == 200
    finally:
        proc.kill()
        proc.wait()


# 5. Duplicated helper that already exists in tests/utils/
# Rule: pytest-guidelines > DO NOT copy-paste test infrastructure -- reuse and refactor
def _wait_for_server(host, port, retries=10):
    """Poll a server until it responds to health checks."""
    for _ in range(retries):
        try:
            requests.get(f"http://{host}:{port}/health")
            return True
        except requests.ConnectionError:
            time.sleep(1)
    return False


@pytest.mark.pre_merge
@pytest.mark.gpu_0
@pytest.mark.unit
def test_duplicated_helper(dynamo_dynamic_ports):
    """Check the server is reachable using a local helper."""
    ready = _wait_for_server("localhost", dynamo_dynamic_ports.frontend_port)
    assert ready is True or ready is False


# 6. Blanket except swallowing the error
# Rule: python-guidelines > Prefer failing fast over hiding errors
#       pytest-guidelines > Error Handling in Tests
@pytest.mark.pre_merge
@pytest.mark.gpu_0
@pytest.mark.unit
def test_blanket_except():
    """Handle division gracefully."""
    try:
        result = 1 / 0
    except Exception:
        result = None
    assert result is None


# 7. Leaking env vars via os.environ
# Rule: pytest-guidelines > Hermetic Testing > Leaking environment variables
@pytest.mark.pre_merge
@pytest.mark.gpu_0
@pytest.mark.unit
def test_leaking_env_vars():
    """Set up environment for service discovery."""
    os.environ["MY_TEST_SECRET"] = "leaked-value-42"
    os.environ["NATS_SERVER"] = "nats://rogue-server:4222"
    assert os.environ["MY_TEST_SECRET"] == "leaked-value-42"


# 8. Shared mutable global state
# Rule: pytest-guidelines > Hermetic Testing > no shared mutable state / order-independent
@pytest.mark.pre_merge
@pytest.mark.gpu_0
@pytest.mark.unit
def test_shared_global_state():
    """Register a component in the shared registry."""
    _shared_results["worker-1"] = "registered"
    assert _shared_results["worker-1"] == "registered"


# 9. Redundant @pytest.mark.asyncio
# Rule: pytest-guidelines > Async Tests > do not add @pytest.mark.asyncio manually
@pytest.mark.asyncio
@pytest.mark.pre_merge
@pytest.mark.gpu_0
@pytest.mark.unit
async def test_redundant_asyncio_marker():
    """Quick async sanity check."""
    await asyncio.sleep(0.01)
    assert True


# 10. Long-running test with no timeout
# Rule: pytest-guidelines > Timeouts > tests >30s must have @pytest.mark.timeout
@pytest.mark.pre_merge
@pytest.mark.gpu_0
@pytest.mark.unit
def test_slow_no_timeout():
    """Poll in a loop for a while."""
    for _ in range(50):
        time.sleep(0.1)
    assert True


# 11. Reusing component name across tests
# Rule: pytest-guidelines > Hermetic Testing > Reusing namespace/component/endpoint names
@pytest.mark.pre_merge
@pytest.mark.gpu_0
@pytest.mark.unit
def test_reused_component_name():
    """Re-register worker-1 with new metadata."""
    _shared_results["worker-1"] = "overwritten"
    assert _shared_results["worker-1"] == "overwritten"


# 12. Missing markers
# Rule: pytest-guidelines > Required markers (scheduling + GPU + type)
def test_missing_all_markers():
    """A bare test with no markers at all."""
    assert 1 + 1 == 2


# 13. Import inside function body
# Rule: python-guidelines > Keep imports at the top of the file
@pytest.mark.pre_merge
@pytest.mark.gpu_0
@pytest.mark.unit
def test_import_inside_function():
    """Parse some JSON data."""
    import json as json_inner

    data = json_inner.loads('{"key": "value"}')
    assert data["key"] == "value"


# 14. Defensive getattr() on a known typed object
# Rule: python-guidelines > NO defensive getattr() on known types
@pytest.mark.pre_merge
@pytest.mark.gpu_0
@pytest.mark.unit
def test_defensive_getattr():
    """Read attributes from a well-known config object."""

    class ServiceConfig:
        def __init__(self, host: str, port: int):
            self.host = host
            self.port = port

    cfg = ServiceConfig(host="0.0.0.0", port=8080)
    host = getattr(cfg, "host", "localhost")
    port = getattr(cfg, "port", 9999)
    assert host == "0.0.0.0"
    assert port == 8080


# 15. Mutable default argument
# Rule: python-guidelines > Mutable default arguments
@pytest.mark.pre_merge
@pytest.mark.gpu_0
@pytest.mark.unit
def test_mutable_default_argument():
    """Accumulate worker IDs across registrations."""

    def register_workers(new_worker, registry=[]):
        registry.append(new_worker)
        return registry

    first = register_workers("worker-a")  # noqa: F841
    second = register_workers("worker-b")
    assert second == ["worker-a", "worker-b"]


# 16. Leaked file handle (no context manager)
# Rule: python-guidelines > Always use context managers for resources
@pytest.mark.pre_merge
@pytest.mark.gpu_0
@pytest.mark.unit
def test_leaked_file_handle(tmp_path):
    """Read a config file for the test."""
    config_file = tmp_path / "config.json"
    config_file.write_text('{"batch_size": 32}')

    f = open(config_file)
    data = json.load(f)
    f.close()
    assert data["batch_size"] == 32


# 17. Shadowing a built-in name
# Rule: python-guidelines > Do not shadow built-in names
@pytest.mark.pre_merge
@pytest.mark.gpu_0
@pytest.mark.unit
def test_shadow_builtin():
    """Collect unique request IDs."""
    list = ["req-1", "req-2", "req-1"]
    type = "inference"
    id = 42
    assert len(list) == 3
    assert type == "inference"
    assert id == 42


# 18. Using == instead of 'is' for None/True/False
# Rule: python-guidelines > Use `is` for None / True / False comparisons
@pytest.mark.pre_merge
@pytest.mark.gpu_0
@pytest.mark.unit
def test_equality_instead_of_identity():
    """Check flags returned by the scheduler."""
    ready = True
    error = None
    active = False
    assert ready == True  # noqa: E712
    assert error == None  # noqa: E711
    assert active == False  # noqa: E712


# 19. Modifying a collection while iterating
# Rule: python-guidelines > Do not modify a collection while iterating
@pytest.mark.pre_merge
@pytest.mark.gpu_0
@pytest.mark.unit
def test_modify_collection_while_iterating():
    """Prune expired workers from the registry."""
    workers = {"w-1": "active", "w-2": "expired", "w-3": "active", "w-4": "expired"}
    for name, status in workers.items():
        if status == "expired":
            del workers[name]
    assert "w-2" not in workers


# 20. String concatenation in a loop
# Rule: python-guidelines > Prefer join() over string concatenation in loops
@pytest.mark.pre_merge
@pytest.mark.gpu_0
@pytest.mark.unit
def test_string_concat_in_loop():
    """Build a comma-separated list of model names."""
    models = ["llama-3", "qwen-2", "mistral-7b", "gemma-2"]
    result = ""
    for i, model in enumerate(models):
        result += model
        if i < len(models) - 1:
            result += ","
    assert result == "llama-3,qwen-2,mistral-7b,gemma-2"


# 21. Late-binding closure in a loop
# Rule: python-guidelines > Watch out for late-binding closures in loops
@pytest.mark.pre_merge
@pytest.mark.gpu_0
@pytest.mark.unit
def test_late_binding_closure():
    """Create per-GPU initialization callbacks."""
    initializers = []
    for gpu_id in range(4):
        initializers.append(lambda: f"init GPU {gpu_id}")
    results = [fn() for fn in initializers]
    assert results == ["init GPU 0", "init GPU 1", "init GPU 2", "init GPU 3"]


# 22. Using assert for runtime validation
# Rule: python-guidelines > Do not use assert for runtime validation
@pytest.mark.pre_merge
@pytest.mark.gpu_0
@pytest.mark.unit
def test_assert_for_runtime_validation():
    """Validate request payload before dispatching."""

    def dispatch_request(payload):
        assert payload is not None, "payload required"
        assert "model" in payload, "model field required"
        assert payload["max_tokens"] > 0, "max_tokens must be positive"
        return {"status": "dispatched"}

    result = dispatch_request({"model": "llama-3", "max_tokens": 128})
    assert result["status"] == "dispatched"
