# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import shutil
import socket
import subprocess
import time
from contextlib import contextmanager
from typing import Optional, Union, List

import requests


@contextmanager
def managed_process(
    command: List[str],
    env: Optional[dict] = None,
    check_ports: Optional[List[int]] = None,
    timeout: int = 300,
    cwd: Optional[str] = None,
    output: bool = False,
):
    """Run a process and manage its lifecycle, checking for ports to become available."""
    check_ports = check_ports or []

    print(f"Running command: {' '.join(command)} in {cwd or os.getcwd()}")

    stdin: Union[int, None] = subprocess.DEVNULL
    stdout: Union[int, None] = subprocess.DEVNULL
    stderr: Union[int, None] = subprocess.DEVNULL
    if output:
        stdin = None
        stdout = None
        stderr = None

    proc = subprocess.Popen(
        command,
        env=env or os.environ.copy(),
        cwd=cwd,
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
    )
    start_time = time.time()

    # Wait for ports to become available
    if check_ports:
        print(f"Waiting for ports: {check_ports}")
        while time.time() - start_time < timeout:
            if all(is_port_open(p) for p in check_ports):
                print(f"All ports {check_ports} are ready")
                break
            time.sleep(0.1)
        else:
            proc.terminate()
            raise TimeoutError(f"Ports {check_ports} not ready in time")

    try:
        yield proc
    finally:
        print(f"Terminating process: {command[0]}")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("Process did not terminate gracefully, killing it")
            proc.kill()
            proc.wait()


def is_port_open(port: int) -> bool:
    """Check if a port is open on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def find_free_port() -> int:
    """Find and return a free port on the system."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def cleanup_directory(path: str) -> None:
    """Safely clean up a directory."""
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception as e:
        print(f"Warning: Failed to clean up directory {path}: {e}")


def check_service_health(url: str, timeout: int = 5, max_retries: int = 3, retry_interval: float = 1.0) -> bool:
    """Check if a service is healthy by making HTTP requests."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                # If this is the models endpoint, validate that models exist
                if url.endswith("/v1/models"):
                    data = response.json()
                    if data.get("data") and len(data["data"]) > 0:
                        print(f"Models found: {data['data']}")
                        return True
                    # If no models yet, continue retrying
                    print("No models found yet, continuing to wait...")
                else:
                    return True
        except requests.RequestException as e:
            print(f"Health check failed: {e}")
        
        time.sleep(retry_interval)
    
    return False


def wait_for_service_health(url: str, timeout: int = 60, check_interval: float = 1.0) -> bool:
    """Wait for a service to become healthy."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if check_service_health(url, max_retries=1, retry_interval=0):
            return True
        time.sleep(check_interval)
    return False
