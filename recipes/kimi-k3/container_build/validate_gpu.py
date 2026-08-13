# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Validate the pinned Dynamo/vLLM runtime and launch a CUDA operation."""

import importlib.metadata as metadata
import json
import os
import platform

import dynamo
import dynamo._core
import dynamo.frontend
import dynamo.vllm
import torch
import vllm
import vllm._C_stable_libtorch


assert platform.machine() == "aarch64"
assert torch.cuda.is_available()
assert metadata.version("vllm") == "0.26.1rc1.dev602+g65b7662d3"
assert metadata.version("ai-dynamo") == "1.4.0"
assert os.environ["DYNAMO_COMMIT_SHA"] == "bf7542fd26613495cc2a59ded28848861e1fee3c"

x = torch.arange(16, dtype=torch.float32, device="cuda").reshape(4, 4)
y = x @ x
torch.cuda.synchronize()
assert tuple(y.shape) == (4, 4)

print(
    json.dumps(
        {
            "ai_dynamo": metadata.version("ai-dynamo"),
            "architecture": platform.machine(),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "cuda_kernel_launch": "passed",
            "dynamo_core": dynamo._core.__file__,
            "dynamo_frontend": dynamo.frontend.__file__,
            "dynamo_vllm": dynamo.vllm.__file__,
            "gpu": torch.cuda.get_device_name(0),
            "torch_cuda": torch.version.cuda,
            "vllm": metadata.version("vllm"),
            "vllm_module": vllm.__file__,
            "vllm_extension": vllm._C_stable_libtorch.__file__,
        },
        indent=2,
        sort_keys=True,
    )
)
