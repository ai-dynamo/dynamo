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
from typing import Optional

# test_dynamo_run.py
import pytest
import requests


class DynamoRunProcess:
    def __init__(self, model, backend, port, input_type="text", timeout=300):
        self.port = port
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

    def start(self):
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

        # Start output capturing threads
        self._output_thread = threading.Thread(
            target=self._capture_pty_output,
        )
        self._output_thread.start()

        self._wait_for_ready(self._timeout)

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
                if r"[33m?[0m [1mUser[0m [38;5;8m›[0m " in data:
                    self._prompt_ready.set()
            except OSError:
                break

    def send_input(self, text):
        if self.input_type != "text":
            raise RuntimeError("send_input() only available in text mode")
        assert self.parent_fd
        os.write(self.parent_fd, f"{text}\n".encode())

    def _wait_for_ready(self, timeout=100):
        if self.input_type == "http":
            # HTTP readiness check remains the same
            start_time = time.time()
            while True:
                try:
                    response = requests.get(f"{self.url}/v1/models", timeout=1)
                    model_list = response.json()
                    print(model_list)
                    if response.status_code == 200:
                        return
                except Exception:
                    pass
                if time.time() - start_time > timeout:
                    raise TimeoutError("HTTP server failed to start")
                time.sleep(0.5)
        else:
            # Wait for CLI prompt in text mode
            if not self._prompt_ready.wait(timeout):
                raise TimeoutError("Text interface prompt not detected")
            self._prompt_ready.clear()

    def stop(self):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
            if self.parent_fd:
                os.close(self.parent_fd)
            if self._output_thread:
                self._output_thread.join(timeout=3)

    @property
    def output(self):
        return self.output_buffer.getvalue()


@pytest.fixture(scope="function", autouse=True)
def kill_dynamo_processes():
    yield
    subprocess.run(["pkill", "-f", ".*dynamo-run.*|.*multiprocessing.*"], check=False)


@pytest.fixture()
def dynamo_run(tmp_path_factory, backend, model, input_type, timeout):
    port = 8888  # In real usage should use random port

    process = DynamoRunProcess(
        model=model, backend=backend, port=port, input_type=input_type, timeout=timeout
    )
    process.start()
    yield process
    process.stop()


@pytest.mark.parametrize(
    ["backend", "timeout"],
    [("vllm", 100), ("mistralrs", 600)],
    ids=["vllm", "mistralrs"],
)
@pytest.mark.parametrize("model", ["deepseek-ai/DeepSeek-R1-Distill-Llama-8B"])
@pytest.mark.parametrize(
    ["input_type", "stream"],
    [("text", False), ("http", True), ("http", False)],
    ids=["text", "http-streaming", "http"],
)
@pytest.mark.parametrize("prompt", [("Hello!", "Hello! How can I assist you today?")])
def test_run(dynamo_run, backend, model, input_type, prompt, stream, timeout):
    if input_type == "text":
        dynamo_run.send_input(f"{prompt[0]}")
        if not dynamo_run._prompt_ready.wait(timeout):
            raise TimeoutError("Text interface prompt not detected")
        content = dynamo_run.output
    else:
        # Test OpenAI-compatible endpoint
        url = f"{dynamo_run.url}/v1/chat/completions"
        payload = {
            "model": f"{model}",
            "messages": [{"role": "user", "content": f"{prompt[0]}"}],
            "max_tokens": 200,
            "temperature": 0,
            "stream": stream,
        }

        content = None

        if stream:
            with requests.post(url, json=payload, stream=True) as response:
                chunks = []
                for line in response.iter_lines():
                    if line and b"DONE" not in line:
                        completion = json.loads(line.replace(b"data:", b""))
                        assert "choices" in completion
                        assert len(completion["choices"]) > 0
                        assert "delta" in completion["choices"][0]
                        assert "content" in completion["choices"][0]["delta"]
                        chunks.append(completion["choices"][0]["delta"]["content"])

            if None in chunks:
                chunks.remove(None)
            content = "".join(chunks)

        else:
            response = requests.post(url, json=payload)
            assert response.status_code == 200
            # Verify response structure
            completion = response.json()
            assert "choices" in completion
            assert len(completion["choices"]) > 0
            assert "message" in completion["choices"][0]
            assert "content" in completion["choices"][0]["message"]
            content = completion["choices"][0]["message"]["content"]
    assert prompt[1] in content
