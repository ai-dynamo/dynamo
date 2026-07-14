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
    # Aggregated mode emits a single worker; name matches VllmWorker
    # log identifier in dynamo.vllm.main.
    agg_worker_k8s_name = "VllmWorker"


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
    # Aggregated mode: the single worker component is also named "decode"
    # (matches decode_worker_k8s_name, but made explicit to keep agg logic
    # consistent across all backends and prevent future drift).
    agg_worker_k8s_name = "decode"


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
    # Aggregated mode: the single worker component is named "TRTLLMWorker" in
    # all v1beta1 agg DGD examples (agg.yaml, agg-with-config.yaml,
    # agg_router.yaml).  Without this the sweep falls back to "decode" which
    # does not match, so no pods are found and no annotations are written.
    agg_worker_k8s_name = "TRTLLMWorker"


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
