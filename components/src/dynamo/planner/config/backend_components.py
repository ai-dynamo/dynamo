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
    # The full PCSG-backed pod name
    # <pcs>-<pcs-index>-<pcsg>-<pcsg-index>-<pclq>-<suffix> must fit
    # Kubernetes' 63-character generated-name limit. Grove therefore limits the
    # combined PCS + PCSG + PCLQ name segments to 45 characters:
    # https://github.com/ai-dynamo/grove/blob/2b586991fc4b7faa9e5e0cefa2956d89bda70965/operator/internal/webhook/admission/pcs/validation/podcliqueset.go#L1012-L1041
    #
    # Note:
    # - PCS: DGD metadata.name, truncated with a hash by PCSNameForDGD if needed.
    # - PCSG: lowercase component name for components rendered through a PCSG.
    # - PCLQ: lowercase component/role name, e.g. prefill-ldr or prefill-wkr.
    # - Pod: the full generated name shown above; standalone PodCliques omit the
    #   PCSG name and index.
    # - Overhead: five separators and the five-character suffix are fixed; the
    #   two decimal index widths vary. At the 45-character segment cap, eight
    #   characters remain for the indices.
    #
    # Dynamo enforces the 45-character budget as
    # MaxCombinedGroveResourceNameLength. PCSNameForDGD reserves room for the
    # component-derived PCSG/PCLQ segments and truncates only the PCS. Short
    # component names therefore preserve more PCS budget; admission rejects only
    # when too little room remains even after truncation.
    prefill_worker_k8s_name = "prefill"
    prefill_worker_component_name = "prefill"
    prefill_worker_endpoint = "generate"
    decode_worker_k8s_name = "decode"
    decode_worker_component_name = "backend"
    decode_worker_endpoint = "generate"
    agg_worker_k8s_name = "worker"


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
