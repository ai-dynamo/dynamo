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

import logging
from dataclasses import dataclass
from typing import Any, Optional

from dynamo.planner.config.backend_components import WORKER_COMPONENT_NAMES
from dynamo.planner.config.defaults import SubComponentType

logger = logging.getLogger(__name__)


@dataclass
class WorkerInfo:
    """Consolidated worker metadata for the planner.

    Populated from MDC (DynamoWorkerMetadata CRs) in Kubernetes mode,
    with fallback to DGD container-arg parsing, then hard-coded defaults.
    """

    # Component / endpoint names used for scaling and runtime client creation
    k8s_name: Optional[str] = None
    component_name: Optional[str] = None
    endpoint: Optional[str] = None

    # Runtime configuration from MDC
    model_name: Optional[str] = None
    total_kv_blocks: Optional[int] = None
    kv_cache_block_size: Optional[int] = None
    max_num_seqs: Optional[int] = None
    max_num_batched_tokens: Optional[int] = None
    context_length: Optional[int] = None

    @property
    def max_kv_tokens(self) -> Optional[int]:
        if self.total_kv_blocks is not None and self.kv_cache_block_size is not None:
            return self.total_kv_blocks * self.kv_cache_block_size
        return None

    def summary(self) -> str:
        parts = [f"k8s_name={self.k8s_name}"]
        parts.append(f"component={self.component_name}")
        parts.append(f"endpoint={self.endpoint}")
        if self.model_name is not None:
            parts.append(f"model={self.model_name}")
        if self.max_kv_tokens is not None:
            parts.append(f"max_kv_tokens={self.max_kv_tokens}")
        if self.max_num_seqs is not None:
            parts.append(f"max_num_seqs(max_bs)={self.max_num_seqs}")
        if self.max_num_batched_tokens is not None:
            parts.append(f"max_num_batched_tokens={self.max_num_batched_tokens}")
        if self.context_length is not None:
            parts.append(f"context_length={self.context_length}")
        return ", ".join(parts)


def build_worker_info_from_defaults(
    backend: str, sub_component_type: SubComponentType
) -> WorkerInfo:
    """Build a WorkerInfo populated only from hard-coded backend defaults."""
    names = WORKER_COMPONENT_NAMES.get(backend)
    if names is None:
        return WorkerInfo()
    if sub_component_type == SubComponentType.PREFILL:
        return WorkerInfo(
            k8s_name=names.prefill_worker_k8s_name,
            component_name=names.prefill_worker_component_name,
            endpoint=names.prefill_worker_endpoint,
        )
    else:
        return WorkerInfo(
            k8s_name=names.decode_worker_k8s_name,
            component_name=names.decode_worker_component_name,
            endpoint=names.decode_worker_endpoint,
        )


def resolve_worker_info(
    backend: str,
    require_prefill: bool,
    require_decode: bool,
    connector: Any = None,
    config_model_name: str = "",
    no_operation: bool = False,
) -> tuple[WorkerInfo, WorkerInfo]:
    """Build WorkerInfo for prefill/decode and resolve model name.

    If the connector has a ``get_worker_info`` method (KubernetesConnector),
    MDC is queried first with fallback to DGD container-arg parsing, then
    hard-coded defaults.  Otherwise hard-coded defaults are used directly.

    The resolved model name is written into both WorkerInfo objects so callers
    can read it from either ``prefill_info.model_name`` or
    ``decode_info.model_name``.

    Returns:
        (prefill_worker_info, decode_worker_info)
    """
    can_query_mdc = connector is not None and hasattr(connector, "get_worker_info")

    # --- Build WorkerInfo ---
    prefill_info = WorkerInfo()
    decode_info = WorkerInfo()

    if can_query_mdc:
        if require_prefill:
            prefill_info = connector.get_worker_info(SubComponentType.PREFILL, backend)
        if require_decode:
            decode_info = connector.get_worker_info(SubComponentType.DECODE, backend)
    else:
        if require_prefill:
            prefill_info = build_worker_info_from_defaults(
                backend, SubComponentType.PREFILL
            )
        if require_decode:
            decode_info = build_worker_info_from_defaults(
                backend, SubComponentType.DECODE
            )

    if require_prefill:
        logger.info(f"Prefill WorkerInfo: {prefill_info.summary()}")
    if require_decode:
        logger.info(f"Decode WorkerInfo: {decode_info.summary()}")

    # Cross-validate model names
    p_model = prefill_info.model_name
    d_model = decode_info.model_name
    if (
        require_prefill
        and require_decode
        and p_model
        and d_model
        and p_model != d_model
    ):
        logger.warning(
            f"Model name mismatch between prefill ({p_model}) and "
            f"decode ({d_model}) WorkerInfo"
        )

    # --- Resolve model name and write back into both WorkerInfo ---
    if no_operation:
        if not config_model_name:
            raise ValueError(
                "Model name is required in no-operation mode. "
                "Please set model_name in the config."
            )
        model_name = config_model_name
    else:
        mdc_model = decode_info.model_name or prefill_info.model_name
        if mdc_model:
            model_name = mdc_model
            logger.info(f"Using model name from MDC: {model_name}")
        elif can_query_mdc:
            model_name = connector.get_model_name(
                require_prefill=require_prefill,
                require_decode=require_decode,
            )
            logger.info(f"Detected model name from DGD container args: {model_name}")
        elif config_model_name:
            model_name = config_model_name
            logger.info(f"Using model name from config: {model_name}")
        else:
            raise ValueError(
                "Could not determine model name. "
                "Please set model_name in the config."
            )

    prefill_info.model_name = model_name
    decode_info.model_name = model_name

    return prefill_info, decode_info
