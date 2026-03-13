# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the hello_world example in examples/custom_backend/hello_world
"""

import asyncio
import os
import subprocess
import sys
import uuid

import pytest

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
]


async def wait_for_output(log_name, process, expected_text, timeout=10):
    """Read process output until the expected text appears."""
    if process.stdout is None:
        raise RuntimeError("Process stdout was not captured")

    output = ""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout

    while True:
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise TimeoutError(
                f"Timed out waiting for '{expected_text}'. Output so far:\n{output}"
            )

        line = await asyncio.wait_for(
            asyncio.to_thread(process.stdout.readline),
            timeout=remaining,
        )
        # Doesn't show when the tests succeed, very helpful if they fail
        print(f"{log_name}: {line.strip()}")

        if not line:
            if process.poll() is not None:
                raise RuntimeError(
                    f"Process exited before '{expected_text}' appeared. Output:\n{output}"
                )
            continue

        output += line
        if expected_text in output:
            return output


@pytest.fixture(scope="module")
def example_dir():
    """Path to the hello_world example directory"""
    # Get the directory of this test file
    test_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate to the hello_world example directory relative to this test
    return os.path.normpath(
        os.path.join(test_dir, "../../../../examples/custom_backend/hello_world")
    )


@pytest.fixture(scope="module")
def example_env():
    """Environment for isolated hello_world example subprocesses."""
    env = os.environ.copy()
    env["DYN_TEST_HELLO_WORLD_NAMESPACE"] = f"hello_world_test_{uuid.uuid4().hex}"
    env["DYN_TEST_HELLO_WORLD_WORD_DELAY_SEC"] = "0.2"
    return env


def stop_process(process):
    """Terminate a process and collect any remaining output."""
    process.terminate()
    try:
        stdout, _ = process.communicate(timeout=1)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, _ = process.communicate(timeout=1)
    return stdout


@pytest.fixture(scope="module")
async def server_process(example_dir, example_env):
    """Start the hello_world server and clean up after test"""
    server_proc = subprocess.Popen(
        [sys.executable, "-u", "hello_world.py"],
        cwd=example_dir,
        env=example_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        await wait_for_output("SERVER", server_proc, "Successfully registered")
    except Exception:
        stop_process(server_proc)
        raise

    yield server_proc

    # Cleanup
    stop_process(server_proc)


async def run_client(example_dir, example_env):
    """Run the client for a specified duration and capture its output"""

    # -u means unbuffered mode
    client_proc = subprocess.Popen(
        [sys.executable, "-u", "client.py"],
        cwd=example_dir,
        env=example_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    output = await wait_for_output("CLIENT", client_proc, "Hello star!")

    stdout = stop_process(client_proc)

    return output + stdout


@pytest.mark.asyncio
async def test_hello_world(example_dir, example_env, server_process):
    """Test that hello_world starts and its client produces the expected output sequence"""
    # Run the client for 5 seconds
    client_output = await run_client(example_dir, example_env)

    # Split output into lines and strip whitespace, filter out empty lines
    lines = [line.strip() for line in client_output.split("\n") if line.strip()]

    # Under the test-only fast path, one iteration completes within the 2s client window.
    # Check that all 4 expected lines appear in the output
    expected_lines = ["Hello world!", "Hello sun!", "Hello moon!", "Hello star!"]
    for expected_line in expected_lines:
        assert expected_line in lines, (
            f"Expected line '{expected_line}' not found in output.\n" f"Lines: {lines}"
        )
