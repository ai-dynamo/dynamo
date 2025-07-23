from typing import TYPE_CHECKING, Any, Optional

import torch
from pydantic.dataclasses import dataclass
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.worker.cache_engine import CacheEngine

from dynamo._core import _vllm_connector_integration
from dynamo.llm import BlockManager, KvbmLeader, KvbmWorker

KvbmConnectorLeader = _vllm_connector_integration.KvbmConnectorLeader

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request


@dataclass
class DynamoKvbmConnectorMetadata(KVConnectorMetadata):
    ...


class DynamoKvbmConnector(KVConnectorBase_V1):
    def __init__(self, vllm_config: "VllmConfig", role: "KVConnectorRole"):
        super().__init__(vllm_config, role)
        # We can immediately initialize our leader here.
        if role == KVConnectorRole.SCHEDULER:
            world_size = self._vllm_config.parallel_config.world_size

            bytes_per_block = CacheEngine.get_cache_block_size(
                self._vllm_config.cache_config,
                self._vllm_config.model_config,
                self._vllm_config.parallel_config,
            )

            total_bytes = bytes_per_block * world_size

            leader = KvbmLeader(total_bytes, world_size)

            block_manager = BlockManager(
                0,
                leader,
                self._vllm_config.cache_config.block_size,
            )

            self.leader = KvbmConnectorLeader(block_manager)

    @property
    def role(self) -> "KVConnectorRole":
        return self._role

    # ==============================
    # Worker-side methods
    # ==============================

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert (
            self.role == KVConnectorRole.WORKER
        ), "Only worker role can register KV caches"

        cache_config = self._vllm_config.cache_config

        shape = list(kv_caches.values())[0].shape

        if not all(t.shape == shape for t in kv_caches.values()):
            raise NotImplementedError(
                "Hybrid models with different KV cache shapes are not supported yet."
            )

        # TODO: Assume the block dimension is within the first 2. This will break if you're doing something weird like having 1 or 2 device blocks.
        num_device_blocks = max(shape[0], shape[1])
        page_size = cache_config.block_size
        tensors = list(kv_caches.values())

        if cache_config.cache_dtype == "auto":
            kv_cache_dtype = self._vllm_config.model_config.dtype
        else:
            kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        device_id = tensors[0].device.index

        # TODO: We can actually just initialize our connection to the leader from the kv transfer params port argument.
        self.worker = KvbmWorker(
            num_device_blocks,
            page_size,
            tensors,
            device_id=device_id,
            worker_id=device_id,
            dtype_width_bytes=kv_cache_dtype.itemsize,
        )

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        """
        Start loading the KV cache from the connector to vLLM's paged
        KV buffer. This is called from the forward context before the
        forward pass to enable async loading during model execution.
        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation
        Note:
            The number of elements in kv_caches and layer_names should be
            the same.

        """
        pass

    def wait_for_layer_load(self, layer_name: str) -> None:
        """
        Block until the KV for a specific layer is loaded into vLLM's
        paged buffer. This is called from within attention layer to ensure
        async copying from start_load_kv is complete.

        This interface will be useful for layer-by-layer pipelining.
        Args:
            layer_name: the name of that layer
        """
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:  # type: ignore
        """
        Start saving a layer of KV cache from vLLM's paged buffer
        to the connector. This is called from within attention layer to
        enable async copying during execution.
        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """
        print(attn_metadata)
        pass

    def wait_for_save(self):
        """
        Block until all the save operations is done. This is called
        as the forward context exits to ensure that the async saving
        from save_kv_layer is complete before finishing the forward.
        This prevents overwrites of paged KV buffer before saving done.
        """
        pass

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens on the worker.
        The scheduler process (via the MultiprocExecutor) will use this output
        to track which workers are done.
        Returns:
            ids of requests that have finished asynchronous transfer
            (requests that previously returned True from request_finished()),
            tuple of (sending/saving ids, recving/loading ids).
            The finished saves/sends req ids must belong to a set provided in a
            call to this method (this call or a prior one).
        """
        return None, None

    # ==============================
    # Scheduler-side methods
    # ==============================

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request
        Returns:
            A tuple with the following elements:
                - The number of tokens that can be loaded from the
                  external KV cache beyond what is already computed.
                - `True` if external KV cache tokens will be loaded
                  asynchronously (between scheduler steps). Must be
                  'False' if the first element is 0.
        """
        _tokens = request.all_token_ids
        return 0, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        """
        Update KVConnector state after block allocation.
        If get_num_new_matched_tokens previously returned True for a
        request, this function may be called twice for that same request -
        first when blocks are allocated for the connector tokens to be
        asynchronously loaded into, and second when any additional blocks
        are allocated, after the load/transfer is complete.
        Args:
            request (Request): the request object.
            blocks (KVCacheBlocks): the blocks allocated for the request.
            num_external_tokens (int): the number of tokens that will be
                loaded from the external KV cache.
        """
        pass

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> "KVConnectorMetadata":
        """
        Build the connector metadata for this step.
        This function should NOT modify fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.
        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        return DynamoKvbmConnectorMetadata()

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Called when a request has finished, before its blocks are freed.
        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """
        return False, None
