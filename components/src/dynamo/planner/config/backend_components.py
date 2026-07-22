# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


class ComponentName:
    """Base class for backend component name configurations."""

    prefill_worker_k8s_name: str = ""
    prefill_worker_component_name: str = ""
    prefill_worker_endpoint: str = ""
    decode_worker_k8s_name: str = ""
    decode_worker_component_name: str = ""
    decode_worker_endpoint: str = ""


class VllmComponentName(ComponentName):
    prefill_worker_k8s_name = "VllmPrefillWorker"
    prefill_worker_component_name = "prefill"
    prefill_worker_endpoint = "generate"
    decode_worker_k8s_name = "VllmDecodeWorker"
    decode_worker_component_name = "backend"
    decode_worker_endpoint = "generate"


class SGLangComponentName(ComponentName):
    prefill_worker_k8s_name = (
        "prefill"  # use short name to stay within k8s limits with grove
    )
    prefill_worker_component_name = "prefill"
    prefill_worker_endpoint = "generate"
    decode_worker_k8s_name = (
        "decode"  # use short name to stay within k8s limits with grove
    )
    decode_worker_component_name = "backend"
    decode_worker_endpoint = "generate"


class TrtllmComponentName(ComponentName):
    # Unified frontend architecture (consistent with vLLM/SGLang):
    # - Prefill workers use "prefill" component
    # - Decode workers use "backend" component
    # Use short k8s names to stay within Grove's 45-char resource name limit
    prefill_worker_k8s_name = "prefill"
    prefill_worker_component_name = "prefill"
    prefill_worker_endpoint = "generate"
    decode_worker_k8s_name = "decode"
    decode_worker_component_name = "backend"
    decode_worker_endpoint = "generate"


class MockerComponentName(ComponentName):
    # Mocker backend for testing/simulation purposes
    prefill_worker_k8s_name = "prefill"
    prefill_worker_component_name = "prefill"
    prefill_worker_endpoint = "generate"
    decode_worker_k8s_name = "decode"
    decode_worker_component_name = "backend"
    decode_worker_endpoint = "generate"


WORKER_COMPONENT_NAMES: dict[str, type[ComponentName]] = {
    "vllm": VllmComponentName,
    "sglang": SGLangComponentName,
    "trtllm": TrtllmComponentName,
    "mocker": MockerComponentName,
}


def get_planner_k8s_component_names(backend: str) -> tuple["str | None", "str | None"]:
    """Return the backend-default DGD/K8s component names.

    The returned names are only explicit-name fallbacks. The DGD is still the
    source of truth: aggregated deployments often declare a single generic
    ``type: worker`` component whose actual name varies by manifest. The
    component resolver handles that by selecting the unique generic worker from
    the DGD and returning its actual name.

    Returns:
        (prefill_k8s_name, decode_k8s_name), either of which may be None if
        the backend is unknown.
    """
    defaults = WORKER_COMPONENT_NAMES.get(backend)
    if defaults is None:
        return None, None
    prefill = defaults.prefill_worker_k8s_name or None
    decode = defaults.decode_worker_k8s_name or None
    return prefill, decode
