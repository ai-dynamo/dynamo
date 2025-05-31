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
import json
import os
import pty
import subprocess
import threading
import time
from io import StringIO
from typing import Optional, Any, Dict, Union, Tuple

import requests
from tests.utils import find_free_port, check_service_health
import pytest


class DynamoRunProcess:
    """
    Manages a dynamo-run process with various input/output options.
    """
    def __init__(self, model, backend, port=None, input_type="text", timeout=300, preloaded_model=None):
        self.port = port or find_free_port()
        self.model = model
        self.backend = backend
        self.process = None
        self.input_type = input_type
        self.url = f"http://localhost:{self.port}"
        self.output_buffer = StringIO()
        self._output_thread = None
        self._prompt_ready = threading.Event()
        self.parent_fd: Optional[int] = None
        self.child_fd: Optional[int] = None
        self._timeout = timeout
        self._ready = threading.Event()
        self.preloaded_model = preloaded_model  # Store the preloaded model instance

    def start(self):
        # If using HTTP and a preloaded model, start HTTP server using the model directly
        if self.input_type == "http" and self.preloaded_model is not None:
            self._start_with_preloaded_model()
        else:
            # Otherwise, use command-line approach
            self._start_with_cli()
        
        # Wait for process to be ready
        if self.input_type == "http":
            # Use check_service_health from utils
            def _models_ready(response):
                try:
                    data = response.json()
                    return "data" in data and len(data["data"]) > 0
                except Exception:
                    return False
            ready = check_service_health(
                f"{self.url}/v1/models",
                max_retries=int(self._timeout * 2),  # retry every 0.5s
                retry_interval=0.5,
                timeout=10,
                callback=_models_ready,
            )
            if ready:
                self._ready.set()
            else:
                self.stop()
                raise TimeoutError(f"DynamoRun process failed to start within {self._timeout}s (HTTP health check)")
        else:
            # Wait for prompt in text mode
            if not self._wait_for_ready(self._timeout):
                self.stop()
                raise TimeoutError(f"DynamoRun process failed to start within {self._timeout}s")
    
    def _start_with_preloaded_model(self):
        """Start using a preloaded model instead of launching a new process"""
        try:
            print(f"[DYNAMO_RUN] Starting HTTP server with preloaded model on port {self.port}")
            
            # For now, we'll fall back to CLI approach since the vLLM serve function
            # doesn't exist in the current version. The model is already cached,
            # so the CLI approach will be faster due to cached model files.
            print(f"[DYNAMO_RUN] Using CLI approach with cached model for compatibility")
            self._start_with_cli()
            
        except Exception as e:
            print(f"[DYNAMO_RUN] Failed to start with preloaded model: {e}")
            # Fall back to CLI approach if server fails
            self._start_with_cli()
    
    def _start_with_cli(self):
        """Start using the CLI approach (original implementation)"""
        cmd = [
            "dynamo-run",
            f"in={self.input_type}",
            f"out={self.backend}",
            str(self.model),
        ]

        if self.input_type == "http":
            cmd += ["--http-port", str(self.port)]

        # Create pseudo-terminal
        self.parent_fd, self.child_fd = pty.openpty()

        print(" ".join(cmd))

        self.process = subprocess.Popen(
            cmd,
            stdin=self.child_fd,
            stdout=self.child_fd,
            stderr=self.child_fd,
            close_fds=True,
        )

        # Close slave FD in parent process
        os.close(self.child_fd)
        self.child_fd = None

        # Start output capturing threads
        self._output_thread = threading.Thread(
            target=self._capture_pty_output,
        )
        self._output_thread.daemon = True
        self._output_thread.start()

    def _capture_pty_output(self):
        assert self.parent_fd
        assert self.process
        
        while True:
            try:
                data = os.read(self.parent_fd, 1024).decode(errors="replace")
                if not data and self.process.poll() is not None:
                    break
                print(f"'{data}'")
                self.output_buffer.write(data)
                if r"[33m?[0m [1mUser[0m [38;5;8m›[0m " in data:
                    self._prompt_ready.set()
                    self._ready.set()  # Also signal process is ready
            except OSError:
                break

    def send_input(self, text):
        if self.input_type != "text":
            raise RuntimeError("send_input() only available in text mode")
        assert self.parent_fd
        os.write(self.parent_fd, f"{text}\n".encode())

    def _wait_for_ready(self, timeout=100):
        """Wait until the process is ready to handle requests"""
        return self._ready.wait(timeout)

    def stop(self):
        if hasattr(self, '_server_thread') and self._server_thread.is_alive():
            # Stop server if it's running with preloaded model
            # Note: This is a simplified approach - in a real implementation,
            # you might need a more graceful way to stop the server
            pass  # The thread is daemon, so it will be terminated when the process exits
        
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)  # Reduced from 3 seconds
            except subprocess.TimeoutExpired:
                self.process.kill()
            if self.parent_fd:
                os.close(self.parent_fd)
                self.parent_fd = None
            if self._output_thread:
                self._output_thread.join(timeout=1)  # Reduced from 3 seconds

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    @property
    def output(self):
        return self.output_buffer.getvalue()


def send_chat_completion_request(url, model, prompt, stream=False, max_tokens=100):
    """
    Send a chat completion request to a dynamo-run HTTP endpoint.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": stream,
        "seed": 0,
    }

    # Define a retry mechanism for the request
    max_retries = 5
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            if stream:
                with requests.post(f"{url}/v1/chat/completions", json=payload, stream=True) as response:
                    if response.status_code != 200:
                        raise RuntimeError(f"Request failed with status {response.status_code}: {response.text}")
                        
                    chunks = []
                    for line in response.iter_lines():
                        if line and b"DONE" not in line:
                            try:
                                completion = json.loads(line.replace(b"data:", b""))
                                if "choices" in completion and len(completion["choices"]) > 0:
                                    if "delta" in completion["choices"][0] and "content" in completion["choices"][0]["delta"]:
                                        chunks.append(completion["choices"][0]["delta"]["content"])
                            except json.JSONDecodeError:
                                pass  # Skip malformed lines

                    if None in chunks:
                        chunks.remove(None)
                    return "".join(chunks)
            else:
                response = requests.post(f"{url}/v1/chat/completions", json=payload)
                if response.status_code != 200:
                    raise RuntimeError(f"Request failed with status {response.status_code}: {response.text}")
                    
                completion = response.json()
                if "choices" in completion and len(completion["choices"]) > 0:
                    if "message" in completion["choices"][0] and "content" in completion["choices"][0]["message"]:
                        return completion["choices"][0]["message"]["content"]
                
                return None
        except Exception as e:
            if attempt == max_retries - 1:
                raise  # Re-raise on last attempt
            print(f"Request attempt {attempt+1} failed: {e}. Retrying...")
            time.sleep(retry_delay)
            retry_delay = int(retry_delay * 1.5)  # Exponential backoff, converted to int 