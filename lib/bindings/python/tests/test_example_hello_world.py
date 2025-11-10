# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the hello_world example in examples/custom_backend/hello_world
"""

import asyncio
import os
import subprocess

import pytest

pytestmark = pytest.mark.pre_merge


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
async def server_process(example_dir):
    """Start the hello_world server and clean up after test"""
    server_proc = subprocess.Popen(
        ["python3", "hello_world.py"],
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


async def run_client(example_dir):
    """Run the client for a specified duration and capture its output"""
    client_proc = subprocess.Popen(
        ["python3", "-u", "client.py"],
        cwd=example_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Let it run for 5 seconds
    await asyncio.sleep(5)

    # Terminate the client
    client_proc.terminate()
    stdout, _ = client_proc.communicate(timeout=1)

    return stdout


@pytest.mark.asyncio
async def test_hello_world(example_dir, server_process):
    """Test that hello_world starts and its client produces the expected output sequence"""
    # Run the client for 5 seconds
    client_output = await run_client(example_dir)

    # Split output into lines and strip whitespace, filter out empty lines
    lines = [line.strip() for line in client_output.split("\n") if line.strip()]

    # Each client iteration produces 4 lines in about 4 seconds
    # The client ran for 5 seconds so the first iteration is expected to be completed
    # Check that all 4 expected lines appear in the output
    expected_lines = ["Hello world!", "Hello sun!", "Hello moon!", "Hello star!"]
    for expected_line in expected_lines:
        assert expected_line in lines, (
            f"Expected line '{expected_line}' not found in output.\n" f"Lines: {lines}"
        )
