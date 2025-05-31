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
import subprocess
import time
from typing import Dict, Any, Optional

import pytest

from tests.utils import managed_process, find_free_port


# Dictionary to hold cached model instances
_MODEL_CACHE = {}

# Centralized port configuration for all services
SERVICE_PORTS = {
    "etcd_client": 2379,
    "etcd_peer": 2380,
    "nats_client": 4222,
    "nats_http": 8222,
    "dynamo_serve_base": 8000,  # Base port, actual port will be dynamic
}


def _ensure_killed(pattern: str):
    subprocess.run(["pkill", "-9", "-f", pattern], check=False)
    time.sleep(0.1)


@pytest.fixture(scope="session")
def service_ports():
    """
    Provides centralized port configuration for all services.
    Returns a dictionary with service names and their assigned ports.
    """
    return SERVICE_PORTS.copy()


@pytest.fixture(scope="session")
def etcd_server(service_ports):
    _ensure_killed("etcd")
    
    client_port = service_ports["etcd_client"]
    peer_port = service_ports["etcd_peer"]

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
def nats_server(service_ports):
    _ensure_killed("nats-server")
    
    client_port = service_ports["nats_client"]
    http_port = service_ports["nats_http"]

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


@pytest.fixture(scope="session")
def model_cache_dir():
    """
    Provides a dedicated cache directory for model downloads.
    """
    cache_dir = os.path.expanduser("~/.cache/dynamo_tests/models")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


@pytest.fixture(scope="session")
def model_loader(model_cache_dir):
    """
    Session-scoped fixture for loading and caching models.
    
    This fixture provides a function that loads models only once per test session,
    then returns the cached instances for subsequent requests, improving test performance.
    Also supports pre-downloading models to cache.
    """
    def _load_model(model_name: str, backend: str = "vllm", preload_only: bool = False, **kwargs) -> Any:
        """
        Load a model or return a cached instance if available.
        
        Args:
            model_name: Name or path of the model to load
            backend: Backend to use for model loading (vllm, transformers, etc.)
            preload_only: If True, only download/cache the model without loading it
            **kwargs: Additional arguments to pass to the model loader
            
        Returns:
            The loaded model instance, or None if preload_only=True
        """
        cache_key = f"{model_name}_{backend}_{sorted(kwargs.items())}"
        
        # Set cache directory for model downloads
        if "download_dir" not in kwargs:
            kwargs["download_dir"] = model_cache_dir
        
        if cache_key in _MODEL_CACHE and not preload_only:
            print(f"Using cached model: {model_name}")
            return _MODEL_CACHE[cache_key]
        
        print(f"{'Pre-downloading' if preload_only else 'Loading'} model: {model_name} with backend {backend}")
        
        # Implement actual model loading logic based on backend
        if backend == "vllm":
            try:
                # Import here to avoid dependency issues if not testing with vllm
                from vllm import LLM
                
                if preload_only:
                    # For preload, we just need to trigger the download
                    # This can be done by initializing with minimal resources
                    print(f"Pre-downloading model {model_name} to cache...")
                    # Use a minimal configuration for download
                    temp_kwargs = kwargs.copy()
                    temp_kwargs.update({
                        "max_model_len": 512,  # Minimal context length
                        "gpu_memory_utilization": 0.1,  # Minimal GPU usage
                        "enforce_eager": True,  # Avoid compilation overhead
                    })
                    try:
                        temp_model = LLM(model=model_name, **temp_kwargs)
                        # Clean up immediately after download
                        del temp_model
                        print(f"Successfully pre-downloaded model: {model_name}")
                        return None
                    except Exception as e:
                        print(f"Pre-download failed, will download during actual loading: {e}")
                        return None
                else:
                    # Normal loading
                    model = LLM(model=model_name, **kwargs)
                    _MODEL_CACHE[cache_key] = model
                    return model
                    
            except ImportError:
                if not preload_only:
                    pytest.skip("vllm not installed, skipping test")
                return None
                
        elif backend == "transformers":
            try:
                # Import here to avoid dependency issues
                from transformers import AutoModelForCausalLM, AutoTokenizer
                
                if preload_only:
                    # For preload, just download the model files
                    print(f"Pre-downloading transformers model {model_name} to cache...")
                    AutoTokenizer.from_pretrained(model_name, cache_dir=model_cache_dir)
                    AutoModelForCausalLM.from_pretrained(
                        model_name, 
                        cache_dir=model_cache_dir,
                        torch_dtype="auto",
                        device_map="cpu",  # Keep on CPU for preload
                        **kwargs
                    )
                    print(f"Successfully pre-downloaded transformers model: {model_name}")
                    return None
                else:
                    # Normal loading
                    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_cache_dir, **kwargs)
                    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=model_cache_dir, **kwargs)
                    _MODEL_CACHE[cache_key] = (model, tokenizer)
                    return model, tokenizer
                    
            except ImportError:
                if not preload_only:
                    pytest.skip("transformers not installed, skipping test")
                return None
        
        # Add support for other backends as needed
        if not preload_only:
            raise ValueError(f"Unsupported backend: {backend}")
        return None
    
    return _load_model


@pytest.fixture(scope="session", autouse=True)
def preload_common_models(model_loader):
    """
    Automatically preload commonly used models at the start of the test session.
    This runs once per session and downloads models to cache for faster test execution.
    """
    common_models = [
        ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "vllm"),
        ("llava-hf/llava-1.5-7b-hf", "vllm"),
    ]
    
    print("\n[PRELOAD] Starting model pre-download...")
    for model_name, backend in common_models:
        try:
            print(f"[PRELOAD] Pre-downloading {model_name} with {backend}...")
            model_loader(model_name, backend, preload_only=True)
        except Exception as e:
            print(f"[PRELOAD] Failed to pre-download {model_name}: {e}")
    
    print("[PRELOAD] Model pre-download completed")


def pytest_sessionfinish(session, exitstatus):
    """Final cleanup: kill stray daemons and reap zombies."""
    # Clear model cache
    for key, model in list(_MODEL_CACHE.items()):
        try:
            # Attempt to clean up model resources if possible
            if hasattr(model, 'unload') and callable(model.unload):
                model.unload()
            elif hasattr(model, '__del__'):
                del model
            # Remove from cache
            del _MODEL_CACHE[key]
        except Exception as e:
            print(f"Error cleaning up model {key}: {e}")
    
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
