# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NixlConnector adapter for vllm-omni OmniConnector interface.

Bridges Dynamo's nixl_connect module to vllm-omni's connector registry,
enabling real NIXL RDMA-based inter-stage AR->DIT disaggregation.

Transfer model (PULL / READ):
    - Tensor payloads: torch.Tensor or list[torch.Tensor] (zero-copy tensor path)
    - Generic Python payloads: serialized to bytes via OmniConnectorBase
        and transported as a uint8 tensor.

    1. Sender (put)
        - Convert tensors to detached contiguous tensors (preserve source device)
        - Wrap each tensor in nixl_connect.Descriptor
        - Await connector.create_readable(descriptor_or_list)
        - Return serialized RdmaMetadata + tensor specs in metadata
        - Keep tensors pinned in _pending until RDMA completes or cleanup occurs

    2. Receiver (get)
        - Decode tensor specs from metadata and allocate matching local tensors
        - Wrap tensors in nixl_connect.Descriptor
        - Await connector.begin_read(rdma_metadata, local_descriptor)
        - Await read_op.wait_for_completion()
        - Return tensors (or deserialize object payload)
"""

import asyncio
import logging
import threading
import time
from typing import Any

import torch
from vllm_omni.distributed.omni_connectors.connectors.base import OmniConnectorBase

logger = logging.getLogger(__name__)

_METADATA_SCHEMA_VERSION = 1
_TENSOR_PAYLOAD_KIND = "tensor_list"
_SERIALIZED_PAYLOAD_KIND = "serialized_obj"
_DEFAULT_PENDING_TIMEOUT_S = 300.0
_DEFAULT_CLEANUP_INTERVAL_S = 5.0

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

try:
    from vllm_omni.distributed.omni_connectors.factory import OmniConnectorFactory

    _OMNI_FACTORY_IMPORT_ERROR: Exception | None = None
except ImportError as exc:
    OmniConnectorFactory = None  # type: ignore[assignment]
    _OMNI_FACTORY_IMPORT_ERROR = exc


class DynamoOmniNixlConnector(OmniConnectorBase):
    """NIXL RDMA connector adapting dynamo.nixl_connect to vllm-omni OmniConnectorBase.

    Provides zero-copy tensor transfer and serialized Python object support:
    - put(): Registers payload as RDMA-readable, returns metadata with RdmaMetadata
    - get(): Allocates local tensors, pulls data via RDMA, deserializes if needed

    All operations are synchronous (blocking) but backed by async RDMA on a dedicated loop.
    """

    supports_raw_data: bool = True

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._connector: Any = _nixl_connect.Connector() if _NIXL_AVAILABLE else None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._loop_ready = threading.Event()
        self._receive_device = _resolve_configured_receive_device(self.config)
        self._pending_timeout_s = float(
            self.config.get("pending_timeout_s", _DEFAULT_PENDING_TIMEOUT_S)
        )
        self._cleanup_interval_s = float(
            self.config.get("cleanup_interval_s", _DEFAULT_CLEANUP_INTERVAL_S)
        )
        # Keeps pending ReadableOperation + payload tensor(s) alive until
        # remote read completion, timeout, explicit cleanup, or connector close.
        # { request_id: (token, ReadableOperation, Any, asyncio.Task, deadline_monotonic) }
        self._pending: dict[str, tuple[str, Any, Any, asyncio.Task[Any], float]] = {}
        self._cleanup_task: asyncio.Task[Any] | None = None
        self._closed = False
        self._metrics: dict[str, int] = {"puts": 0, "gets": 0, "errors": 0}
        self._start_loop_thread()
        logger.info(
            "[DynamoOmniNixlConnector] Initialized (nixl_available=%s timeout=%ss interval=%ss receive_device=%s)",
            _NIXL_AVAILABLE,
            self._pending_timeout_s,
            self._cleanup_interval_s,
            self._receive_device,
        )

    def _start_loop_thread(self) -> None:
        if self._loop_thread is not None and self._loop_thread.is_alive():
            return

        self._loop_ready.clear()

        def _loop_main() -> None:
            loop = asyncio.new_event_loop()
            self._loop = loop
            asyncio.set_event_loop(loop)
            self._loop_ready.set()
            loop.run_forever()
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            loop.close()

        self._loop_thread = threading.Thread(
            target=_loop_main,
            name="dynamo-omni-nixl-connector-loop",
            daemon=True,
        )
        self._loop_thread.start()
        if not self._loop_ready.wait(timeout=5.0):
            raise RuntimeError("Failed to start connector event loop thread")

    def _run_in_loop(self, coro: Any) -> Any:
        if (
            self._loop is None
            or self._loop_thread is None
            or not self._loop_thread.is_alive()
        ):
            raise RuntimeError("Connector event loop is not running")
        if threading.current_thread() is self._loop_thread:
            raise RuntimeError("Cannot block waiting for loop result from loop thread")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

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
        if (pending := self._pending.get(request_id)) and pending[0] == token:
            self._pending.pop(request_id, None)

    async def _close_async(self) -> None:
        for request_id in tuple(self._pending):
            self._cancel_pending(request_id, "connector closed")
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            self._cleanup_task = None

    def close(self) -> None:
        """Release all resources; safe to call multiple times."""
        if self._closed:
            return
        self._closed = True

        try:
            if self._loop is not None and self._loop.is_running():
                self._run_in_loop(self._close_async())
        except Exception:
            logger.exception("[DynamoOmniNixlConnector] close cleanup failed")

        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread is not None and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=5.0)
        self._loop = None
        self._loop_thread = None
        self._loop_ready.clear()

    # ------------------------------------------------------------------
    # Sender side
    # ------------------------------------------------------------------

    async def _put_async(
        self,
        from_stage: str,
        to_stage: str,
        put_key: str,
        data: Any,
    ) -> tuple[bool, int, dict[str, Any] | None]:
        """Register payload as NIXL-readable and return metadata for remote RDMA fetch.

        Handles two paths:
        - Tensor payloads: Direct RDMA descriptor registration
        - Non-tensor payloads: Serialize via OmniConnectorBase, wrap in uint8 tensor

        Returns (success, bytes_transferred, metadata_dict).
        """
        if not _NIXL_AVAILABLE or self._connector is None:
            logger.error(
                "[DynamoOmniNixlConnector.put] nixl_connect unavailable req=%s", put_key
            )
            return False, 0, None
        if self._closed:
            logger.error(
                "[DynamoOmniNixlConnector.put] connector is closed req=%s", put_key
            )
            return False, 0, None

        try:
            self._ensure_cleanup_task()

            kind = _TENSOR_PAYLOAD_KIND
            if _is_tensor_payload(data):
                tensors, tensor_specs = _normalize_tensor_payload(data)
            else:
                serialized_data = self.serialize_obj(data)
                payload_tensor = _bytes_to_uint8_tensor(
                    serialized_data,
                    device=torch.device("cpu"),
                )
                tensors, tensor_specs = _normalize_tensor_payload(payload_tensor)
                kind = _SERIALIZED_PAYLOAD_KIND

            descriptors = [_nixl_connect.Descriptor(t) for t in tensors]
            readable_op = await self._connector.create_readable(
                descriptors[0] if len(descriptors) == 1 else descriptors
            )

            if put_key in self._pending:
                self._cancel_pending(put_key, "duplicate request id replaced")

            token = str(id(readable_op))
            deadline = time.monotonic() + max(0.0, self._pending_timeout_s)

            async def _wait_release(req_id: str = put_key) -> None:
                try:
                    await readable_op.wait_for_completion()
                    logger.debug(
                        "[DynamoOmniNixlConnector.put] RDMA complete req=%s", req_id
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning(
                        "[DynamoOmniNixlConnector.put] wait_for_completion req=%s: %s",
                        req_id,
                        exc,
                    )
                finally:
                    self._maybe_pop_pending(req_id, token)

            task = asyncio.create_task(_wait_release())
            self._pending[put_key] = (token, readable_op, tensors, task, deadline)

            total_size = sum(spec["size"] for spec in tensor_specs)
            out_metadata: dict[str, Any] = {
                "schema_version": _METADATA_SCHEMA_VERSION,
                "kind": kind,
                "rdma_metadata": readable_op.metadata().model_dump(),
                "tensor_specs": tensor_specs,
                "size": total_size,
            }
            self._metrics["puts"] += 1
            logger.debug(
                "[DynamoOmniNixlConnector.put] edge=%s->%s req=%s kind=%s tensors=%d size=%d",
                from_stage,
                to_stage,
                put_key,
                kind,
                len(tensors),
                total_size,
            )
            return True, total_size, out_metadata

        except Exception as exc:
            self._metrics["errors"] += 1
            logger.error(
                "[DynamoOmniNixlConnector.put] Failed req=%s: %s",
                put_key,
                exc,
                exc_info=True,
            )
            return False, 0, None

    # ------------------------------------------------------------------
    # Receiver side
    # ------------------------------------------------------------------

    async def _get_async(
        self,
        from_stage: str,
        to_stage: str,
        get_key: str,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Any, int] | None:
        """RDMA-pull payload from sender using RdmaMetadata.

        Allocates local tensors, performs RDMA read, and deserializes if needed.
        Payload bytes do not travel through gRPC; only metadata + RDMA offsets.

        Returns (payload, bytes_transferred) on success, None on failure.
        """
        if not _NIXL_AVAILABLE or self._connector is None:
            logger.error(
                "[DynamoOmniNixlConnector.get] nixl_connect unavailable req=%s", get_key
            )
            return None
        if self._closed:
            logger.error(
                "[DynamoOmniNixlConnector.get] connector is closed req=%s", get_key
            )
            return None
        if metadata is None or "rdma_metadata" not in metadata:
            logger.error(
                "[DynamoOmniNixlConnector.get] Missing 'rdma_metadata' in metadata req=%s",
                get_key,
            )
            return None

        try:
            self._ensure_cleanup_task()

            schema_version = metadata.get("schema_version", _METADATA_SCHEMA_VERSION)
            if schema_version != _METADATA_SCHEMA_VERSION:
                raise RuntimeError(
                    f"Unsupported metadata schema version={schema_version} "
                    f"expected={_METADATA_SCHEMA_VERSION}"
                )

            # Reconstruct RdmaMetadata from the serialized dict (travels over gRPC).
            rdma_meta = _nixl_connect.RdmaMetadata.model_validate(
                metadata["rdma_metadata"]
            )

            payload_kind = metadata.get("kind", _TENSOR_PAYLOAD_KIND)
            if payload_kind not in (_TENSOR_PAYLOAD_KIND, _SERIALIZED_PAYLOAD_KIND):
                raise RuntimeError(
                    "Unsupported payload kind: expected one of "
                    f"{(_TENSOR_PAYLOAD_KIND, _SERIALIZED_PAYLOAD_KIND)!r}, "
                    f"got {payload_kind!r}"
                )

            tensor_specs = metadata.get("tensor_specs")
            if not isinstance(tensor_specs, list) or not tensor_specs:
                raise RuntimeError(f"Invalid tensor_specs for req={get_key}")

            local_tensors: list[torch.Tensor] = []
            local_descriptors: list[Any] = []
            for spec in tensor_specs:
                if not isinstance(spec, dict):
                    raise RuntimeError(f"Invalid tensor spec: {spec!r}")
                tensor = _allocate_tensor_from_spec(
                    spec, receive_device=self._receive_device
                )
                local_tensors.append(tensor)
                local_descriptors.append(_nixl_connect.Descriptor(tensor))

            read_op = await self._connector.begin_read(
                rdma_meta,
                local_descriptors[0]
                if len(local_descriptors) == 1
                else local_descriptors,
            )
            await read_op.wait_for_completion()

            total_size = sum(spec.get("size", 0) for spec in tensor_specs)
            if payload_kind == _SERIALIZED_PAYLOAD_KIND:
                if len(local_tensors) != 1:
                    raise RuntimeError(
                        "Serialized payload requires exactly one tensor; "
                        f"got {len(local_tensors)}"
                    )
                raw_bytes = _tensor_uint8_to_bytes(local_tensors[0])
                result = self.deserialize_obj(raw_bytes)
            else:
                result = local_tensors[0] if len(local_tensors) == 1 else local_tensors

            self._metrics["gets"] += 1
            logger.debug(
                "[DynamoOmniNixlConnector.get] edge=%s->%s req=%s kind=%s tensors=%d size=%d",
                from_stage,
                to_stage,
                get_key,
                payload_kind,
                len(local_tensors),
                total_size,
            )
            return result, total_size

        except Exception as exc:
            self._metrics["errors"] += 1
            logger.error(
                "[DynamoOmniNixlConnector.get] Failed edge=%s->%s req=%s: %s",
                from_stage,
                to_stage,
                get_key,
                exc,
                exc_info=True,
            )
            return None

    def put(
        self,
        from_stage: str,
        to_stage: str,
        put_key: str,
        data: Any,
    ) -> tuple[bool, int, dict[str, Any] | None]:
        """Sync wrapper for NIXL put executed on the connector loop thread."""
        return self._run_in_loop(self._put_async(from_stage, to_stage, put_key, data))

    def get(
        self,
        from_stage: str,
        to_stage: str,
        get_key: str,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Any, int] | None:
        """Sync wrapper for NIXL get executed on the connector loop thread."""
        return self._run_in_loop(
            self._get_async(from_stage, to_stage, get_key, metadata)
        )

    def cleanup(self, request_id: str) -> None:
        """Cancel and discard any pinned tensors/operations for *request_id*."""
        if self._closed:
            return

        async def _cleanup_request() -> None:
            self._cancel_pending(request_id, "explicit cleanup")

        self._run_in_loop(_cleanup_request())

    def health(self) -> dict[str, Any]:
        """Return connector health status and transfer metrics."""
        return {
            "status": "unhealthy" if self._closed or not _NIXL_AVAILABLE else "healthy",
            "nixl_available": _NIXL_AVAILABLE,
            "pending_requests": len(self._pending),
            **self._metrics,
        }


def _is_tensor_payload(payload: Any) -> bool:
    if isinstance(payload, torch.Tensor):
        return True
    return (
        isinstance(payload, (list, tuple))
        and bool(payload)
        and all(isinstance(item, torch.Tensor) for item in payload)
    )


def _tensor_uint8_to_bytes(tensor: torch.Tensor) -> bytes:
    flat = tensor.detach().cpu().contiguous().view(-1)
    if flat.dtype != torch.uint8:
        flat = flat.to(dtype=torch.uint8)
    return flat.numpy().tobytes()


def _bytes_to_uint8_tensor(value: bytes, device: torch.device) -> torch.Tensor:
    return (
        torch.frombuffer(value, dtype=torch.uint8)
        .clone()
        .to(device=device)
        .contiguous()
    )


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


def create_dynamoomni_nixl_connector(config: dict[str, Any]) -> DynamoOmniNixlConnector:
    """Factory function for vllm-omni connector registration."""
    return DynamoOmniNixlConnector(config)


def register_dynamoomni_nixl_connector() -> None:
    """Register DynamoOmniNixlConnector with vllm-omni's OmniConnectorFactory."""
    try:
        if OmniConnectorFactory is None:
            raise ImportError(str(_OMNI_FACTORY_IMPORT_ERROR))

        connector_name = "NixlConnector"
        if connector_name in OmniConnectorFactory.list_registered_connectors():
            logger.info(
                "[DynamoOmniNixlConnector] %s already registered; skipping",
                connector_name,
            )
            return

        OmniConnectorFactory.register_connector(
            connector_name,
            create_dynamoomni_nixl_connector,
        )
        logger.info(
            "[DynamoOmniNixlConnector] Registered %s with OmniConnectorFactory",
            connector_name,
        )
    except ImportError as exc:
        logger.error(
            "[DynamoOmniNixlConnector] Failed to import OmniConnectorFactory: %s",
            exc,
            exc_info=True,
        )
        raise
    except Exception as exc:
        logger.error(
            "[DynamoOmniNixlConnector] Registration failed: %s",
            exc,
            exc_info=True,
        )
        raise
