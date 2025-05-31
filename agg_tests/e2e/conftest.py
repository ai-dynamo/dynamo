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
import pytest
from tests.utils import find_free_port
from tests.e2e.dynamo_client import DynamoRunProcess




import os
import shutil
import subprocess
import time

import pytest

from tests.utils import managed_process, find_free_port


def _ensure_killed(pattern: str):
    subprocess.run(["pkill", "-9", "-f", pattern], check=False)
    time.sleep(0.1)


@pytest.fixture(scope="session")
def etcd_server():
    _ensure_killed("etcd")
    client_port = 2379 # find_free_port()
    peer_port   = 2380 # find_free_port()

    client_url = f"http://127.0.0.1:{client_port}"
    peer_url   = f"http://127.0.0.1:{peer_port}"

    etcd_env = os.environ.copy()
    etcd_env["ALLOW_NONE_AUTHENTICATION"] = "yes"
    shutil.rmtree("/tmp/etcd-test-data", ignore_errors=True)

    cmd = [
        "etcd",
        "--listen-client-urls", client_url,
        "--advertise-client-urls", client_url,
        "--listen-peer-urls", peer_url,
        "--initial-advertise-peer-urls", peer_url,
        "--initial-cluster",              f"default={peer_url}",
        "--initial-cluster-state",        "new",
        "--data-dir",                     "/tmp/etcd-test-data",
    ]

    with managed_process(cmd, env=etcd_env, check_ports=[client_port], output=True):
        yield client_url


@pytest.fixture(scope="session")
def nats_server():
    _ensure_killed("nats-server")
    client_port = 4222 # find_free_port()
    http_port   = 8222 # find_free_port()

    shutil.rmtree("/tmp/nats/jetstream", ignore_errors=True)
    os.makedirs("/tmp/nats/jetstream", exist_ok=True)

    cmd = [
        "nats-server",
        "-js",
        "--trace",
        "--store_dir", "/tmp/nats/jetstream",
        "--port",        str(client_port),
        "--http_port",   str(http_port),
    ]

    with managed_process(cmd, check_ports=[client_port, http_port], output=True):
        time.sleep(0.2)
        yield {
            "client_url":   f"nats://127.0.0.1:{client_port}",
            "http_monitor": f"http://127.0.0.1:{http_port}",
        }


def pytest_sessionfinish(session, exitstatus):
    """Final cleanup: kill stray daemons and reap zombies."""
    subprocess.run(["pkill", "-9", "-f", "etcd|nats-server"], check=False)
    import os
    from time import sleep
    sleep(0.1)
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
            if pid == 0:
                break
        except ChildProcessError:
            break


# pytest fixture for DynamoRunProcess
@pytest.fixture()
def dynamo_run(backend, model, input_type, timeout):
    """
    Create and start a DynamoRunProcess for testing.
    """
    port = 8000 # find_free_port()
    with DynamoRunProcess(
        model=model, backend=backend, port=port, input_type=input_type, timeout=timeout
    ) as process:
        yield process 



@pytest.fixture(scope="session")
def model_cache():
    """
    Pre-download and cache the model used in tests to avoid download delays during test execution.
    This fixture runs once per test session and ensures the model is available locally.
    """
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    print(f"\n=== CHECKING MODEL CACHE FOR {model_name} ===")
    
    # Check if model is already cached
    import os
    from pathlib import Path
    
    # Common HuggingFace cache locations
    hf_cache_dirs = [
        os.path.expanduser("~/.cache/huggingface/hub"),
        os.path.expanduser("~/.cache/huggingface/transformers"),
        "/tmp/huggingface_cache",
    ]
    
    # Check if model exists in any cache directory
    model_cached = False
    for cache_dir in hf_cache_dirs:
        if os.path.exists(cache_dir):
            # Look for model directories that contain the model name
            for item in os.listdir(cache_dir):
                if "deepseek" in item.lower() and "r1" in item.lower() and "distill" in item.lower():
                    model_path = os.path.join(cache_dir, item)
                    if os.path.isdir(model_path):
                        print(f"Found cached model at: {model_path}")
                        model_cached = True
                        break
            if model_cached:
                break
    
    if model_cached:
        print("Model is already cached, skipping download")
    else:
        print("Model not found in cache, downloading...")
        
        # Use Python to download the model via transformers/huggingface_hub
        download_script = f'''
import os
from huggingface_hub import snapshot_download
import torch

print("Starting model download...")
model_name = "{model_name}"

try:
    # Download model files
    cache_dir = snapshot_download(
        repo_id=model_name,
        cache_dir=os.path.expanduser("~/.cache/huggingface/hub"),
        resume_download=True,
        local_files_only=False
    )
    print(f"Model downloaded successfully to: {{cache_dir}}")
    
    # Also try to load tokenizer to ensure all files are downloaded
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Tokenizer loaded successfully")
    
except Exception as e:
    print(f"Error downloading model: {{e}}")
    exit(1)

print("Model download completed successfully!")
'''
        try:
            print("Running model download script...")
            result = subprocess.run(
                ["python", "-c", download_script],
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes timeout for download
            )
            
            if result.returncode == 0:
                print("Model download completed successfully!")
                print("Download output:", result.stdout)
            else:
                print(f"Model download failed with exit code {result.returncode}")
                print("Error output:", result.stderr)
                # Don't fail the test session, just warn
                print("WARNING: Model download failed, tests may be slower due to download during execution")
                
        except subprocess.TimeoutExpired:
            print("Model download timed out after 30 minutes")
            print("WARNING: Model download timed out, tests may be slower due to download during execution")
        except Exception as e:
            print(f"Error during model download: {e}")
            print("WARNING: Model download failed, tests may be slower due to download during execution")
    
    print("=== MODEL CACHE CHECK COMPLETED ===")
    yield model_name 