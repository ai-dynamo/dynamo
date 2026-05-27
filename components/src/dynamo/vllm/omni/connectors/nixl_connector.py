# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NixlConnector adapter for vllm-omni OmniConnector interface.

Bridges Dynamo's nixl_connect module to vllm-omni's connector registry,
enabling real NIXL RDMA-based inter-stage AR->DIT disaggregation.

Transfer model (PULL / READ, TENSOR-ONLY):
    All payloads must be torch.Tensor or list[torch.Tensor].

    1. Sender (put):
    - Convert tensors to contiguous tensors (preserve source device)
       - Wrap each in nixl_connect.Descriptor
       - Await connector.create_readable(descriptor_or_list)
       - Return small RdmaMetadata + tensor specs in metadata
       - Keep tensors pinned in _pending until RDMA completes

    2. Receiver (get):
       - Decode tensor specs from metadata and allocate matching local tensors
       - Wrap in nixl_connect.Descriptor
       - Await connector.begin_read(rdma_metadata, local_descriptor)
       - Await read_op.wait_for_completion()
       - Return tensors directly
"""

import asyncio
import logging
import time
from typing import Any

import torch

from dynamo.vllm.omni.utils import is_tensor_payload

logger = logging.getLogger(__name__)

_METADATA_SCHEMA_VERSION = 1
_TENSOR_PAYLOAD_KIND = "tensor_list"

try:
    from dynamo import nixl_connect as _nixl_connect

    _NIXL_AVAILABLE = True
except ImportError:
    _nixl_connect = None  # type: ignore[assignment]
    _NIXL_AVAILABLE = False
    logger.warning(
        "[DynamoOmniNixlConnector] dynamo.nixl_connect not available; "
        "NIXL RDMA transfers will be disabled."
    )


class DynamoOmniNixlConnector:
    """Real NIXL RDMA connector adapting dynamo.nixl_connect to vllm-omni interface.

    put()  - registers local tensor(s) as RDMA-readable via create_readable(),
             returns only small RdmaMetadata (pointer + agent info) in the
             metadata dict that travels over gRPC.
    get()  - allocates local tensor(s), calls begin_read() with the received
             RdmaMetadata, awaits RDMA completion, returns tensor payload.

    Both put() and get() are async coroutines, callers must await them.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self._connector: Any = _nixl_connect.Connector() if _NIXL_AVAILABLE else None
        self._receive_device = _resolve_configured_receive_device(self.config)
        self._pending_timeout_s = float(self.config.get("pending_timeout_s", 300.0))
        self._cleanup_interval_s = float(self.config.get("cleanup_interval_s", 5.0))
        # Keeps pending ReadableOperation + payload tensor(s) alive until remote read completes.
        # { request_id: (token, ReadableOperation, Any, asyncio.Task, deadline_monotonic) }
        self._pending: dict[str, tuple[str, Any, Any, asyncio.Task[Any], float]] = {}
        self._cleanup_task: asyncio.Task[Any] | None = None
        self._closed = False
        logger.info(
            "[DynamoOmniNixlConnector] Initialized (nixl_available=%s timeout=%ss interval=%ss receive_device=%s)",
            _NIXL_AVAILABLE,
            self._pending_timeout_s,
            self._cleanup_interval_s,
            self._receive_device,
        )

    async def _cleanup_pending_loop(self) -> None:
        try:
            while not self._closed:
                await asyncio.sleep(max(0.1, self._cleanup_interval_s))
                now = time.monotonic()
                expired = [
                    request_id
                    for request_id, (_, _, _, _, deadline) in self._pending.items()
                    if now >= deadline
                ]
                for request_id in expired:
                    self._cancel_pending(
                        request_id,
                        "pending timeout exceeded before remote completion",
                    )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("[DynamoOmniNixlConnector] pending cleanup loop failed")
            raise

    def _ensure_cleanup_task(self) -> None:
        if self._closed:
            raise RuntimeError("Connector is closed")
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_pending_loop())

    def _cancel_pending(self, request_id: str, reason: str) -> None:
        pending = self._pending.pop(request_id, None)
        if pending is None:
            return
        _, _, _, task, _ = pending
        if not task.done():
            task.cancel()
        logger.warning(
            "[DynamoOmniNixlConnector] cleaned pending request req=%s reason=%s",
            request_id,
            reason,
        )

    def _maybe_pop_pending(self, request_id: str, token: str) -> None:
        pending = self._pending.get(request_id)
        if pending is None:
            return
        current_token = pending[0]
        if current_token == token:
            self._pending.pop(request_id, None)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        for request_id in list(self._pending.keys()):
            self._cancel_pending(request_id, "connector closed")

        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            finally:
                self._cleanup_task = None

    # ------------------------------------------------------------------
    # Sender side
    # ------------------------------------------------------------------

    async def put(
        self,
        from_stage: str,
        to_stage: str,
        request_id: str,
        payload: Any,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[bool, Any, dict[str, Any]]:
        """Register tensor payload as NIXL-readable and return RdmaMetadata.

        Only the small RdmaMetadata dict (pointer + agent info) travels over
        gRPC.  The actual tensor data is transferred directly via NIXL RDMA
        when the receiver calls get().

        Returns:
            (success, error, out_metadata) where out_metadata contains:
                - "rdma_metadata": RdmaMetadata.model_dump() (small)
                - "tensor_specs": list of tensor shape/dtype/device/size
                - "size": total tensor data size in bytes
        """
        if not _NIXL_AVAILABLE or self._connector is None:
            return False, "dynamo.nixl_connect not available", {}
        if self._closed:
            return False, "connector is closed", {}
        del metadata

        try:
            self._ensure_cleanup_task()

            if is_tensor_payload(payload):
                tensors, tensor_specs = _normalize_tensor_payload(payload)
                descriptors = [_nixl_connect.Descriptor(tensor) for tensor in tensors]
                readable_op = await self._connector.create_readable(
                    _descriptor_or_list(descriptors)
                )

                if request_id in self._pending:
                    self._cancel_pending(request_id, "duplicate request id replaced")

                token = str(id(readable_op))
                deadline = time.monotonic() + max(0.0, self._pending_timeout_s)

                async def _wait_release(req_id: str = request_id) -> None:
                    try:
                        await readable_op.wait_for_completion()
                        logger.debug(
                            "[DynamoOmniNixlConnector.put] RDMA complete req=%s", req_id
                        )
                    except Exception as exc:
                        logger.warning(
                            "[DynamoOmniNixlConnector.put] wait_for_completion req=%s: %s",
                            req_id,
                            exc,
                        )
                    finally:
                        self._maybe_pop_pending(req_id, token)

                task = asyncio.create_task(_wait_release())
                self._pending[request_id] = (
                    token,
                    readable_op,
                    tensors,
                    task,
                    deadline,
                )

                out_metadata: dict[str, Any] = {
                    "schema_version": _METADATA_SCHEMA_VERSION,
                    "kind": _TENSOR_PAYLOAD_KIND,
                    "rdma_metadata": readable_op.metadata().model_dump(),
                    "tensor_specs": tensor_specs,
                    "size": sum(spec["size"] for spec in tensor_specs),
                }
                logger.debug(
                    "[DynamoOmniNixlConnector.put] edge=%s->%s req=%s tensor_count=%d",
                    from_stage,
                    to_stage,
                    request_id,
                    len(tensors),
                )
                return True, None, out_metadata

            raise RuntimeError(
                f"[DynamoOmniNixlConnector.put] Unsupported payload type for req={request_id}: "
                f"expected torch.Tensor or list[torch.Tensor], got {type(payload)}"
            )

        except Exception as exc:
            logger.error(
                "[DynamoOmniNixlConnector.put] Failed req=%s: %s",
                request_id,
                exc,
                exc_info=True,
            )
            return False, str(exc), {}

    # ------------------------------------------------------------------
    # Receiver side
    # ------------------------------------------------------------------

    async def get(
        self,
        from_stage: str,
        to_stage: str,
        request_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """RDMA-pull tensor data from sender and return tensors.

        Uses the RdmaMetadata from metadata["rdma_metadata"] to issue a
        NIXL READ directly into locally allocated tensors - zero gRPC payload transfer.

        Returns:
            torch.Tensor or list[torch.Tensor]: The received tensor payload.

        Raises:
            RuntimeError: if NIXL is unavailable or metadata is missing.
        """
        if not _NIXL_AVAILABLE or self._connector is None:
            raise RuntimeError("dynamo.nixl_connect not available for get()")
        if self._closed:
            raise RuntimeError("connector is closed")

        if metadata is None or "rdma_metadata" not in metadata:
            raise RuntimeError(
                f"[DynamoOmniNixlConnector.get] Missing 'rdma_metadata' in "
                f"metadata for req={request_id}"
            )

        try:
            self._ensure_cleanup_task()

            schema_version = metadata.get("schema_version", _METADATA_SCHEMA_VERSION)
            if schema_version != _METADATA_SCHEMA_VERSION:
                raise RuntimeError(
                    "[DynamoOmniNixlConnector.get] Unsupported metadata schema "
                    f"version={schema_version} expected={_METADATA_SCHEMA_VERSION} "
                    f"for req={request_id}"
                )

            # Reconstruct RdmaMetadata from the serialized dict (travels over gRPC).
            rdma_meta = _nixl_connect.RdmaMetadata.model_validate(
                metadata["rdma_metadata"]
            )

            if metadata.get("kind") == _TENSOR_PAYLOAD_KIND:
                tensor_specs = metadata.get("tensor_specs")
                if not isinstance(tensor_specs, list) or not tensor_specs:
                    raise RuntimeError(
                        f"[DynamoOmniNixlConnector.get] Invalid tensor_specs for req={request_id}"
                    )

                local_tensors: list[torch.Tensor] = []
                local_descriptors: list[Any] = []
                for spec in tensor_specs:
                    if not isinstance(spec, dict):
                        raise RuntimeError(
                            f"[DynamoOmniNixlConnector.get] Invalid tensor spec for req={request_id}: {spec!r}"
                        )
                    tensor = _allocate_tensor_from_spec(
                        spec,
                        receive_device=self._receive_device,
                    )
                    local_tensors.append(tensor)
                    local_descriptors.append(_nixl_connect.Descriptor(tensor))

                read_op = await self._connector.begin_read(
                    rdma_meta,
                    _descriptor_or_list(local_descriptors),
                )
                await read_op.wait_for_completion()

                logger.debug(
                    "[DynamoOmniNixlConnector.get] edge=%s->%s req=%s received_tensor_count=%d",
                    from_stage,
                    to_stage,
                    request_id,
                    len(local_tensors),
                )
                return local_tensors[0] if len(local_tensors) == 1 else local_tensors

            raise RuntimeError(
                f"[DynamoOmniNixlConnector.get] Unsupported payload kind for req={request_id}: "
                f"expected 'tensor_list', got {metadata.get('kind')!r}"
            )

        except Exception as exc:
            logger.error(
                "[DynamoOmniNixlConnector.get] Failed edge=%s->%s req=%s: %s",
                from_stage,
                to_stage,
                request_id,
                exc,
                exc_info=True,
            )
            raise


def _normalize_tensor_payload(
    payload: Any,
) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
    tensors = [payload] if isinstance(payload, torch.Tensor) else list(payload)
    normalized: list[torch.Tensor] = []
    tensor_specs: list[dict[str, Any]] = []
    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Tensor payload must contain only torch.Tensor values")
        contiguous = tensor.detach().contiguous()
        normalized.append(contiguous)
        tensor_specs.append(
            {
                "shape": list(contiguous.shape),
                "dtype": str(contiguous.dtype),
                "device": str(contiguous.device),
                "size": contiguous.numel() * contiguous.element_size(),
            }
        )
    return normalized, tensor_specs


def _allocate_tensor_from_spec(
    spec: dict[str, Any],
    receive_device: torch.device | None = None,
) -> torch.Tensor:
    shape = spec.get("shape")
    dtype = _resolve_torch_dtype(spec.get("dtype"))
    if not isinstance(shape, list):
        raise RuntimeError(f"Invalid tensor spec shape: {shape!r}")

    shape_tuple = tuple(int(dim) for dim in shape)
    source_device = _parse_torch_device(spec.get("device"))

    last_error: Exception | None = None
    for device in _candidate_receive_devices(source_device, receive_device):
        try:
            return torch.empty(shape_tuple, dtype=dtype, device=device)
        except Exception as exc:
            last_error = exc

    try:
        return torch.empty(shape_tuple, dtype=dtype)
    except Exception as exc:
        message = (
            f"Failed to allocate tensor for spec={spec!r} "
            f"with receive_device={receive_device!r}"
        )
        if last_error is not None:
            raise RuntimeError(message) from last_error
        raise RuntimeError(message) from exc


def _resolve_torch_dtype(dtype_name: Any) -> torch.dtype:
    if not isinstance(dtype_name, str):
        raise RuntimeError(f"Invalid tensor dtype: {dtype_name!r}")
    normalized = dtype_name.removeprefix("torch.")
    dtype = getattr(torch, normalized, None)
    if dtype is None:
        raise RuntimeError(f"Unsupported tensor dtype: {dtype_name!r}")
    return dtype


def _resolve_configured_receive_device(config: dict[str, Any]) -> torch.device | None:
    configured = config.get("receive_device", config.get("target_device"))
    return _parse_torch_device(configured)


def _parse_torch_device(device_like: Any) -> torch.device | None:
    if device_like is None:
        return None
    try:
        return torch.device(device_like)
    except Exception as exc:  # pragma: no cover - defensive path
        raise RuntimeError(f"Invalid device spec: {device_like!r}") from exc


def _candidate_receive_devices(
    source_device: torch.device | None,
    configured_device: torch.device | None,
) -> list[torch.device]:
    if configured_device is None:
        return [source_device] if source_device is not None else []
    if source_device is None or source_device == configured_device:
        return [configured_device]
    return [configured_device, source_device]


def _descriptor_or_list(descriptors: list[Any]) -> Any:
    return descriptors[0] if len(descriptors) == 1 else descriptors


def create_dynamoomni_nixl_connector(
    config_dict: dict[str, Any],
) -> DynamoOmniNixlConnector:
    """Factory function for vllm-omni connector registration."""
    return DynamoOmniNixlConnector(config=config_dict.get("extra", {}))


def register_dynamoomni_nixl_connector() -> None:
    """Register DynamoOmniNixlConnector with vllm-omni's OmniConnectorFactory.

    Call this during initialization to enable NixlConnector support.
    """
    try:
        from vllm_omni.distributed.omni_connectors.factory import OmniConnectorFactory

        OmniConnectorFactory.register_connector(
            "NixlConnector",
            create_dynamoomni_nixl_connector,
        )
        logger.info(
            "[DynamoOmniNixlConnector] Successfully registered with OmniConnectorFactory"
        )
    except ImportError as exc:
        logger.error(
            "[DynamoOmniNixlConnector] Failed to import OmniConnectorFactory: %s", exc
        )
        raise
    except Exception as exc:
        logger.error(
            "[DynamoOmniNixlConnector] Registration failed: %s", exc, exc_info=True
        )
        raise
