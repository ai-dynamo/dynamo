# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Implementation of vLLM KV cache manager protocol.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Optional

import torch

logger = logging.getLogger("kvbm.diag")

_tracer = None


def _get_tracer():
    global _tracer
    if _tracer is not None:
        return _tracer
    if not os.environ.get("OTEL_EXPORT_ENABLED", "").lower() in ("1", "true", "on", "yes"):
        return None
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource

        provider = trace.get_tracer_provider()
        if not isinstance(provider, TracerProvider):
            endpoint = os.environ.get(
                "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://localhost:4317"
            )
            resource = Resource.create({
                "service.name": os.environ.get("OTEL_SERVICE_NAME", "dynamo") + ".python",
            })
            provider = TracerProvider(resource=resource)
            provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True))
            )
            trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer("kvbm.connector", "0.1.0")
    except Exception:
        _tracer = None
    return _tracer


@contextmanager
def _otel_span(name: str, attributes: dict | None = None, traceparent: str | None = None):
    tracer = _get_tracer()
    if tracer is None:
        yield None
        return
    ctx = None
    if traceparent:
        try:
            from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
            ctx = TraceContextTextMapPropagator().extract({"traceparent": traceparent})
        except Exception:
            pass
    with tracer.start_as_current_span(name, context=ctx, attributes=attributes or {}) as span:
        yield span


def _request_traceparent(request) -> str | None:
    """Extract traceparent from a vLLM Request's trace_headers."""
    headers = getattr(request, "trace_headers", None)
    if headers:
        return headers.get("traceparent")
    return None


from typing_extensions import override
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


# from kvbm.vllm_integration.kv_cache_utils import KvbmCacheBlocks
from kvbm.utils import nvtx_annotate
from kvbm.vllm_integration.connector_leader import KvConnectorLeader
from kvbm.vllm_integration.connector_worker import KvConnectorWorker

EngineId = str


class DynamoConnectorMetadata(KVConnectorMetadata):
    def __init__(self, metadata: bytes):
        assert isinstance(metadata, bytes)
        self.metadata = metadata


class DynamoConnector(KVConnectorBase_V1):
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ):
        super().__init__(
            vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config
        )

        assert vllm_config.kv_transfer_config is not None
        assert vllm_config.kv_transfer_config.engine_id is not None
        self.engine_id: EngineId = vllm_config.kv_transfer_config.engine_id

        if role == KVConnectorRole.SCHEDULER:
            self._scheduler = KvConnectorLeader(
                vllm_config=vllm_config, engine_id=self.engine_id
            )
            self._worker = None
            try:
                import kvbm.vllm_tracing_patch  # noqa: F401
            except Exception:
                pass
        elif role == KVConnectorRole.WORKER:
            self._worker = KvConnectorWorker(
                vllm_config=vllm_config, engine_id=self.engine_id
            )
            self._scheduler = None

    # Scheduler/Leader

    @nvtx_annotate(category="scheduler")
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[Optional[int], bool]:
        # Ensure _create_slot runs first so request.trace_headers is
        # populated before the span reads it for trace correlation.
        self._scheduler._create_slot(request)
        with _otel_span("kvbm.get_matched_tokens", {
            "request_id": request.request_id,
            "num_computed_tokens": num_computed_tokens,
        }, traceparent=_request_traceparent(request)) as span:
            result = self._scheduler.get_num_new_matched_tokens(request, num_computed_tokens)
            num_matched, needs_onboard = result
            if span is not None and num_matched is not None:
                span.set_attribute("num_matched", num_matched)
                span.set_attribute("needs_onboard", needs_onboard)
            if num_matched is not None and num_matched > 0:
                logger.info(
                    "[KVBM-DIAG] get_num_new_matched_tokens: req=%s "
                    "num_computed_tokens=%d num_matched=%s needs_onboard=%s "
                    "total_tokens=%d",
                    request.request_id, num_computed_tokens,
                    num_matched, needs_onboard, request.num_tokens,
                )
            elif num_matched is None:
                logger.debug(
                    "[KVBM-DIAG] get_num_new_matched_tokens: req=%s "
                    "lookup_pending (None returned), num_computed_tokens=%d",
                    request.request_id, num_computed_tokens,
                )
            elif num_matched == 0:
                logger.warning(
                    "[KVBM-DIAG] get_num_new_matched_tokens: req=%s "
                    "ZERO matched (full prefill), num_computed_tokens=%d",
                    request.request_id, num_computed_tokens,
                )
            return result

    @nvtx_annotate(category="scheduler")
    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        with _otel_span("kvbm.update_state_after_alloc", {
            "request_id": request.request_id,
            "num_external_tokens": num_external_tokens,
        }, traceparent=_request_traceparent(request)) as span:
            if num_external_tokens > 0:
                block_ids = blocks.get_block_ids()[0]
                if span is not None:
                    span.set_attribute("num_blocks", len(block_ids))
                    ids_str = ",".join(str(b) for b in block_ids[:64])
                    if len(block_ids) > 64:
                        ids_str += f"...+{len(block_ids)-64}"
                    span.set_attribute("device_block_ids", ids_str)
                logger.info(
                    "[KVBM-DIAG] update_state_after_alloc: req=%s "
                    "num_external_tokens=%d num_blocks=%d",
                    request.request_id, num_external_tokens, len(block_ids),
                )
            self._scheduler.update_state_after_alloc(request, blocks, num_external_tokens)

    @nvtx_annotate(category="scheduler")
    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        with _otel_span("kvbm.build_connector_meta", {
            "num_new_reqs": len(scheduler_output.scheduled_new_reqs),
        }):
            for req in scheduler_output.scheduled_new_reqs:
                scheduled = scheduler_output.num_scheduled_tokens.get(req.req_id, -1)
                if req.num_computed_tokens > 0 or scheduled != req.num_computed_tokens:
                    logger.info(
                        "[KVBM-DIAG] build_connector_meta NEW: req=%s "
                        "num_computed_tokens=%d num_scheduled_tokens=%d "
                        "total_prompt_tokens=%d",
                        req.req_id, req.num_computed_tokens, scheduled,
                        len(req.prompt_token_ids),
                    )
            data = self._scheduler.build_connector_meta(scheduler_output)
            return DynamoConnectorMetadata(data)

    @nvtx_annotate(category="scheduler")
    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        with _otel_span("kvbm.request_finished", {
            "request_id": request.request_id,
            "num_blocks": len(block_ids),
        }, traceparent=_request_traceparent(request)) as span:
            if span is not None and block_ids:
                ids_str = ",".join(str(b) for b in block_ids[:64])
                if len(block_ids) > 64:
                    ids_str += f"...+{len(block_ids)-64}"
                span.set_attribute("block_ids", ids_str)
            return self._scheduler.request_finished(request, block_ids)

    # Worker

    @nvtx_annotate(category="worker")
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        self._worker.register_kv_caches(kv_caches)

    @nvtx_annotate(category="worker")
    @override
    def bind_connector_metadata(
        self, connector_metadata: DynamoConnectorMetadata
    ) -> None:
        # Must call super() to set _connector_metadata so has_connector_metadata() returns True
        # This is required for save_kv_layer to be called during the forward pass
        super().bind_connector_metadata(connector_metadata)
        assert isinstance(connector_metadata.metadata, bytes)
        self._worker.bind_connector_metadata(connector_metadata.metadata)

    @nvtx_annotate(category="worker")
    @override
    def clear_connector_metadata(self) -> None:
        super().clear_connector_metadata()
        self._worker.clear_connector_metadata()

    @nvtx_annotate(category="worker")
    @override
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        with _otel_span("kvbm.start_load_kv"):
            self._worker.start_load_kv(forward_context, **kwargs)

    @nvtx_annotate(category="worker")
    @override
    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    @nvtx_annotate(category="worker")
    @override
    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        with _otel_span("kvbm.save_kv_layer", {"layer_name": layer_name}):
            self._worker.save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)

    @nvtx_annotate(category="worker")
    @override
    def wait_for_save(self):
        pass

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        return self._worker.get_finished(finished_req_ids)

    @override
    def get_block_ids_with_load_errors(self) -> set[int]:
        """Return block IDs that failed to load asynchronously."""
        if self._worker is None:
            return set()
        return self._worker.get_block_ids_with_load_errors()

    # Management API

    def clear_pool(self, pool: str) -> None:
        """Clear (wipe) all KV cache entries from a specific pool.

        Requires KVBM_DEV_MODE=TRUE environment variable.

        This is a destructive operation that drops all in-flight slots
        and resets the target pool, returning every block to the empty state.

        Args:
            pool: One of "gpu"/"device", "cpu"/"host", or "disk".

        Raises:
            RuntimeError: If not in scheduler role, KVBM_DEV_MODE is not
                enabled, or the pool name is invalid.
        """
        if self._scheduler is None:
            raise RuntimeError(
                "clear_pool is only available on the scheduler (leader) side"
            )
        self._scheduler.clear_pool(pool)

    @override
    def shutdown(self):
        """
        Shutdown the connector and cleanup resources.

        Called when the worker process is shutting down to ensure
        all async operations complete and resources are released.
        """
        # TODO: Implement proper cleanup in Rust layer
        # if self._worker:
        #     self._worker.shutdown()
        # if self._scheduler:
        #     self._scheduler.shutdown()
        pass
