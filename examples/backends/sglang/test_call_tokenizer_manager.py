# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Smoke test for the /engine/call_tokenizer_manager generic passthrough route.

Starts a Dynamo frontend + SGLang backend, then exercises the generic route
with the same calls Slime makes during RL training weight sync.

Usage:
    python test_call_tokenizer_manager.py [--model MODEL] [--system-port PORT]

Requires a GPU with enough VRAM for the model (~2 GB for Qwen3-0.6B).
"""

import argparse
import os
import signal
import subprocess
import sys
import time

import requests

# Defaults
MODEL = "Qwen/Qwen3-0.6B"
HOST = "127.0.0.1"
FRONTEND_PORT = 30000
SYSTEM_PORT = 9090


def wait_for_health(url, label, timeout=120, poll=2):
    """Poll a health endpoint until it returns 200."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=1)
            if r.status_code == 200:
                print(f"  [ok] {label} ready")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(poll)
    print(f"  [FAIL] {label} not ready after {timeout}s")
    return False


def _subprocess_env():
    """Build env dict that inherits PYTHONPATH so subprocesses use the same code."""
    env = os.environ.copy()
    # Propagate PYTHONPATH so subprocesses pick up the same component overrides.
    if "PYTHONPATH" in os.environ:
        env["PYTHONPATH"] = os.environ["PYTHONPATH"]
    return env


def start_frontend():
    proc = subprocess.Popen(
        [sys.executable, "-m", "dynamo.frontend", "--http-port", str(FRONTEND_PORT)],
        env=_subprocess_env(),
    )
    if not wait_for_health(
        f"http://{HOST}:{FRONTEND_PORT}/health", "frontend", timeout=30, poll=1
    ):
        proc.kill()
        sys.exit(1)
    return proc


def start_backend(model):
    env = _subprocess_env()
    env["DYN_SYSTEM_PORT"] = str(SYSTEM_PORT)
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "dynamo.sglang",
            "--model-path",
            model,
            "--tp",
            "1",
            "--mem-fraction-static",
            "0.8",
        ],
        env=env,
    )
    if not wait_for_health(
        f"http://{HOST}:{SYSTEM_PORT}/health", "backend", timeout=120, poll=2
    ):
        proc.kill()
        sys.exit(1)
    return proc


def engine_post(route, body=None):
    """POST to /engine/<route> and return (status_code, json)."""
    url = f"http://{HOST}:{SYSTEM_PORT}/engine/{route}"
    r = requests.post(url, json=body or {})
    try:
        data = r.json()
    except Exception:
        data = r.text
    return r.status_code, data


def call_tm(method, args=None, kwargs=None):
    """Shortcut: POST to call_tokenizer_manager with method/args/kwargs."""
    body = {"method": method}
    if args is not None:
        body["args"] = args
    if kwargs is not None:
        body["kwargs"] = kwargs
    return engine_post("call_tokenizer_manager", body)


# ── individual tests ────────────────────────────────────────────────────


def test_get_weight_version():
    """Dedicated route — should always work."""
    code, data = engine_post("get_weight_version")
    assert code == 200, f"get_weight_version: {code} {data}"
    assert "weight_version" in data, f"unexpected response: {data}"
    print(f"  weight_version = {data['weight_version']}")


def test_flush_cache():
    """flush_cache() — no args, returns a dataclass."""
    code, data = call_tm("flush_cache")
    assert code == 200, f"flush_cache: {code} {data}"
    print(f"  flush_cache -> {data}")


def test_pause_continue_generation():
    """pause_generation(req) then continue_generation(req)."""
    code, data = call_tm(
        "pause_generation",
        args=[{"io_struct.PauseGenerationReqInput": {}}],
    )
    assert code == 200, f"pause_generation: {code} {data}"
    print(f"  pause_generation -> {data}")

    code, data = call_tm(
        "continue_generation",
        args=[{"io_struct.ContinueGenerationReqInput": {}}],
    )
    assert code == 200, f"continue_generation: {code} {data}"
    print(f"  continue_generation -> {data}")


def test_nonexistent_route():
    """Unknown engine route should 404."""
    code, data = engine_post("does_not_exist")
    assert code == 404, f"expected 404 for unknown route, got {code} {data}"
    print("  unknown route -> 404 (correct)")


def test_bad_method():
    """Calling a method that doesn't exist on tokenizer_manager should error."""
    code, data = call_tm("this_method_does_not_exist_xyz")
    # The handler does getattr which will raise AttributeError → 500
    assert code in (400, 500), f"expected error for bad method, got {code} {data}"
    print(f"  bad method -> {code} (correct)")


# ── main ────────────────────────────────────────────────────────────────


def run_tests():
    tests = [
        ("get_weight_version (dedicated route)", test_get_weight_version),
        ("flush_cache via call_tokenizer_manager", test_flush_cache),
        ("pause + continue generation", test_pause_continue_generation),
        ("unknown engine route (expect 404)", test_nonexistent_route),
        ("bad tokenizer_manager method (expect error)", test_bad_method),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"\n--- {name} ---")
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print("=" * 60)
    return failed == 0


def main():
    global SYSTEM_PORT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default=MODEL, help=f"Model to load (default: {MODEL})"
    )
    parser.add_argument("--system-port", type=int, default=SYSTEM_PORT)
    args = parser.parse_args()

    SYSTEM_PORT = args.system_port

    frontend_proc = None
    backend_proc = None
    try:
        print("Starting frontend...")
        frontend_proc = start_frontend()

        print("Starting backend...")
        backend_proc = start_backend(args.model)

        time.sleep(2)
        ok = run_tests()
        sys.exit(0 if ok else 1)

    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"\nFatal: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        for label, proc in [("backend", backend_proc), ("frontend", frontend_proc)]:
            if proc is None:
                continue
            print(f"Stopping {label}...")
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        print("Done.")


if __name__ == "__main__":
    main()
