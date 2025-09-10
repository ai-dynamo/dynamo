# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the cancellation example in examples/custom_backend/cancellation
"""

import asyncio
import os
import subprocess

import pytest

pytestmark = pytest.mark.pre_merge


@pytest.fixture(scope="module")
def example_dir():
    """Path to the cancellation example directory"""
    # Get the directory of this test file
    test_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate to the cancellation example directory relative to this test
    return os.path.normpath(
        os.path.join(test_dir, "../../../../../examples/custom_backend/cancellation")
    )


@pytest.fixture(scope="function")
async def server_process(example_dir):
    """Start the backend server and clean up after test"""
    server_proc = subprocess.Popen(
        ["python3", "-u", "server.py"],
        cwd=example_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Wait for server to start
    await asyncio.sleep(1)

    yield server_proc

    # Cleanup
    server_proc.terminate()
    server_proc.wait(timeout=1)


@pytest.fixture(scope="function")
async def middle_server_process(example_dir, server_process):
    """Start the middle server (depends on backend server) and clean up after test"""
    middle_proc = subprocess.Popen(
        ["python3", "-u", "middle_server.py"],
        cwd=example_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Wait for middle server to start
    await asyncio.sleep(1)

    yield middle_proc

    # Cleanup
    middle_proc.terminate()
    middle_proc.wait(timeout=1)


def run_client(example_dir, use_middle=False):
    """Run the client and capture its output"""
    cmd = ["python3", "client.py"]
    if use_middle:
        cmd.append("--middle")

    client_proc = subprocess.Popen(
        cmd,
        cwd=example_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Wait for client to complete
    stdout, _ = client_proc.communicate(timeout=1)

    if client_proc.returncode != 0:
        pytest.fail(
            f"Client failed with return code {client_proc.returncode}. Output: {stdout}"
        )

    return stdout


def stop_process(process):
    """Stop a running process and capture its output"""
    process.terminate()
    stdout, _ = process.communicate(timeout=1)
    return stdout


@pytest.mark.asyncio
async def test_direct_connection_cancellation(example_dir, server_process):
    """Test cancellation with direct client-server connection"""
    # Run the client (direct connection)
    client_output = run_client(example_dir, use_middle=False)

    # Wait for server to print cancellation message
    await asyncio.sleep(1)

    # Capture server output
    server_output = stop_process(server_process)

    # Assert expected messages
    assert (
        "Client: Cancelling after 3 responses..." in client_output
    ), f"Client output: {client_output}"
    assert (
        "Server: Cancelled at iteration" in server_output
    ), f"Server output: {server_output}"


@pytest.mark.asyncio
async def test_middle_server_cancellation(
    example_dir, server_process, middle_server_process
):
    """Test cancellation with middle server proxy"""
    # Run the client (through middle server)
    client_output = run_client(example_dir, use_middle=True)

    # Wait for server to print cancellation message
    await asyncio.sleep(1)

    # Capture output from all processes
    server_output = stop_process(server_process)
    middle_output = stop_process(middle_server_process)

    # Assert expected messages
    assert (
        "Client: Cancelling after 3 responses..." in client_output
    ), f"Client output: {client_output}"
    assert (
        "Server: Cancelled at iteration" in server_output
    ), f"Server output: {server_output}"
    assert (
        "Middle server: Backend stream ended early due to cancellation" in middle_output
    ), f"Middle server output: {middle_output}"
