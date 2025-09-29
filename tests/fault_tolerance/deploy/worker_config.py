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

"""Central configuration for worker naming and initialization patterns."""

import re
from typing import Dict, Pattern

# Worker name mapping for different backends
WORKER_MAP = {
    "vllm": {
        "decode": "VllmDecodeWorker",
        "prefill": "VllmPrefillWorker",
    },
    "sglang": {
        "decode": "decode",
        "prefill": "prefill",
    },
}

# Process ready patterns for recovery detection
WORKER_READY_PATTERNS: Dict[str, Pattern] = {
    # Frontend
    "Frontend": re.compile(r"added model"),
    
    # vLLM workers
    "VllmDecodeWorker": re.compile(
        r"VllmWorker for (?P<model_name>.*?) has been initialized"
    ),
    "VllmPrefillWorker": re.compile(
        r"VllmWorker for (?P<model_name>.*?) has been initialized"
    ),
    
    # SGLang workers - look for their specific initialization messages
    "decode": re.compile(
        r"Model registration succeeded|Decode worker handler initialized|worker handler initialized"
    ),
    "prefill": re.compile(
        r"Model registration succeeded|Prefill worker handler initialized"
    ),
}

def get_all_worker_types() -> list[str]:
    """Get all worker type names for both vLLM and SGLang."""
    worker_types = ["Frontend"]
    for backend in WORKER_MAP.values():
        worker_types.extend(backend.values())
    # Remove duplicates while preserving order
    seen = set()
    return [x for x in worker_types if not (x in seen or seen.add(x))]

def get_worker_ready_pattern(worker_name: str) -> Pattern:
    """Get the ready pattern for a specific worker type."""
    return WORKER_READY_PATTERNS.get(worker_name)

def get_backend_workers(backend: str) -> Dict[str, str]:
    """Get worker mapping for a specific backend."""
    return WORKER_MAP.get(backend, {})
