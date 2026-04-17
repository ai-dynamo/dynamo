# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
TRT-LLM KV-connector worker wired to the Rust ConnectorWorker.

Adapts TRT-LLM's ``KvCacheConnectorWorker`` ABC to the Rust
``ConnectorWorker`` via ``KvbmConnectorWorkerBase``.

Key differences from the vLLM worker:
- KV cache is a single stacked tensor ``[num_blocks, num_layers, 2, flat_block]``
  rather than a per-layer dict. Requires ``FullyContiguous`` layout on the Rust
  side (Stream B).
- Layer indices are integers (no name→index mapping needed).
- ``get_finished`` signature differs: TRT-LLM passes request ID lists and
  expects ``(List[int], List[int])`` return; the Rust worker returns
  ``(Optional[set[str]], Optional[set[str]])``.
- ``register_forward_pass_callable`` returns ``None`` in stage 1 (eager mode).
  CUDA graph support is deferred to stage 2.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import kvbm
import torch
from kvbm.v2.connector.base import KvbmConnectorWorkerBase, VeloPeerMetadata

if kvbm.v2.is_available():
    KvbmRuntime = kvbm.v2.KvbmRuntime
else:
    KvbmRuntime = None

from tensorrt_llm._torch.pyexecutor.kv_cache_connector import KvCacheConnectorWorker
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

logger = logging.getLogger(__name__)


class TrtllmConnectorWorker(KvCacheConnectorWorker, KvbmConnectorWorkerBase):
    """
    TRT-LLM KV-connector worker backed by the Rust ConnectorWorker.
    """

    def __init__(self, llm_args: TorchLlmArgs):
        KvCacheConnectorWorker.__init__(self, llm_args)
        KvbmConnectorWorkerBase.__init__(self)

        if not kvbm.v2.is_available():
            raise ImportError(
                "TrtllmConnectorWorker requires the 'v2' feature. "
                "Rebuild kvbm with: maturin develop --features v2"
            )

        runtime = KvbmRuntime.build_worker(None)
        self._init_worker(runtime)
        self._num_layers: int = 0

        instance_id = self._handshake_metadata.instance_id
        logger.info(
            "TrtllmConnectorWorker initialized with Velo instance: %s",
            instance_id.hex()[:8],
        )

    # ------------------------------------------------------------------
    # KvCacheConnectorWorker ABC implementations
    # ------------------------------------------------------------------

    def register_kv_caches(self, kv_cache_tensor: torch.Tensor) -> None:
        """
        Register the stacked KV cache tensor with NIXL.

        TRT-LLM allocates KV cache as a single contiguous tensor with shape
        ``[num_blocks, num_layers, 2, flat_block]`` where
        ``flat_block = tokens_per_block * num_kv_heads * head_dim``.

        This requires ``FullyContiguous`` layout on the Rust side (Stream B)
        rather than the ``LayerSeparate`` layout used by vLLM.
        """
        num_blocks = kv_cache_tensor.shape[0]
        num_layers = kv_cache_tensor.shape[1]
        page_size = self._llm_args.kv_cache_config.tokens_per_block
        dtype_width = kv_cache_tensor.dtype.itemsize

        logger.info(
            "Registering KV caches: num_blocks=%d, num_layers=%d, "
            "page_size=%d, dtype_width=%d, shape=%s",
            num_blocks,
            num_layers,
            page_size,
            dtype_width,
            list(kv_cache_tensor.shape),
        )

        # Pass as single-element list — Rust auto-detects FullyContiguous
        # layout when tensors.len() == 1 and infers num_layers from shape[1].
        self._worker.register_kv_caches(
            [kv_cache_tensor],
            num_blocks,
            page_size,
            dtype_width,
        )

        self._num_layers = num_layers
        logger.info("KV caches registered (deferred mode)")

    def start_load_kv(self, stream: torch.cuda.Stream) -> None:
        self._worker.start_load_kv()

    def wait_for_layer_load(self, layer_idx: int, stream: torch.cuda.Stream) -> None:
        self._worker.wait_for_layer_load(layer_idx, stream.cuda_stream)

    def save_kv_layer(self, layer_idx: int, stream: torch.cuda.Stream) -> None:
        self._worker.save_kv_layer(layer_idx, stream.cuda_stream)

    def wait_for_save(self, stream: torch.cuda.Stream) -> None:
        pass  # No-op: forward-pass completion signalled via Velo in save_kv_layer

    def get_finished(
        self,
        finished_gen_req_ids: List[int],
        started_loading_req_ids: List[int],
    ) -> Tuple[List[int], List[int]]:
        # Rust worker tracks state internally; input args are ignored.
        finished_sending, finished_recving = self._worker.get_finished()
        sending = [int(req_id) for req_id in (finished_sending or set())]
        recving = [int(req_id) for req_id in (finished_recving or set())]
        return sending, recving

    def get_block_ids_with_load_errors(self) -> List[int]:
        return list(self._worker.get_failed_onboarding())

    def get_handshake_metadata(self) -> VeloPeerMetadata:
        return self._handshake_metadata

    # ------------------------------------------------------------------
    # Connector metadata
    # ------------------------------------------------------------------

    def bind_connector_meta(self, metadata: object) -> None:
        super().bind_connector_meta(metadata)
        self._worker.bind_connector_metadata(metadata)

    def _clear_connector_meta(self) -> None:
        super()._clear_connector_meta()
        self._worker.clear_connector_metadata()
