# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prototype Dynamo coordinator for transactional decode-to-decode migration.

The standard Dynamo frontend dispatches to ``dynamo.backend.generate``. This
coordinator starts the request on ``dynamo.fast.generate``, prepares source KV
through the fast worker's migration endpoint, and resumes generation through
``dynamo.slow.generate``. The worker components intentionally have one instance
each, which makes routing deterministic for the prototype.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import logging
import uuid
from typing import Any, AsyncGenerator, Dict, Optional

import uvloop

from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

logger = logging.getLogger(__name__)


def _response_data(response: Any) -> Dict[str, Any]:
    data = response.data() if hasattr(response, "data") else response
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected a dict response, got {type(data)!r}")
    return data


def _trim_output_prefix(
    output: Dict[str, Any], count: int
) -> tuple[Dict[str, Any], int]:
    """Drop up to ``count`` already-forwarded token positions from one chunk."""
    if count <= 0:
        return output, 0
    output = dict(output)
    token_ids = list(output.get("token_ids") or [])
    dropped = min(count, len(token_ids))
    output["token_ids"] = token_ids[dropped:]
    for key in ("log_probs", "top_logprobs"):
        values = output.get(key)
        if isinstance(values, list):
            output[key] = values[dropped:]
    return output, dropped


def _migrated_request(
    request: Dict[str, Any], prepare: Dict[str, Any]
) -> Dict[str, Any]:
    migrated = copy.deepcopy(request)
    migrated["token_ids"] = list(prepare["committed_input_ids"])
    migrated.pop("prompt_embeds", None)
    migrated.pop("multi_modal_data", None)
    migrated.pop("prefill_result", None)
    migrated["bootstrap_info"] = {
        "bootstrap_host": prepare["bootstrap_host"],
        "bootstrap_port": prepare["bootstrap_port"],
        "bootstrap_room": prepare["bootstrap_room"],
    }
    migrated["_decode_migration_source_dp_rank"] = int(prepare.get("source_dp_rank", 0))

    committed_output_tokens = max(
        0, int(prepare["committed_len"]) - int(prepare["prompt_len"])
    )
    stop_conditions = dict(migrated.get("stop_conditions") or {})
    max_tokens = stop_conditions.get("max_tokens")
    if max_tokens is not None:
        stop_conditions["max_tokens"] = max(
            1, int(max_tokens) - committed_output_tokens
        )
    min_tokens = stop_conditions.get("min_tokens")
    if min_tokens is not None:
        stop_conditions["min_tokens"] = max(
            0, int(min_tokens) - committed_output_tokens
        )
    migrated["stop_conditions"] = stop_conditions
    return migrated


class DecodeMigrationCoordinator:
    def __init__(
        self,
        runtime: DistributedRuntime,
        migrate_after_tokens: int,
        migrate_on_token_ids: Optional[set[int]] = None,
        namespace: str = "dynamo",
        component: str = "backend",
        worker_namespace: Optional[str] = None,
        destination_start_delay_ms: int = 0,
        force_destination_failure: bool = False,
    ) -> None:
        self.runtime = runtime
        self.namespace = namespace
        self.component = component
        self.worker_namespace = worker_namespace or namespace
        self.migrate_after_tokens = migrate_after_tokens
        self.migrate_on_token_ids = migrate_on_token_ids or set()
        self.destination_start_delay_ms = destination_start_delay_ms
        self.force_destination_failure = force_destination_failure
        self.fast_client = None
        self.slow_client = None
        self.destination_prepare_client = None
        self.source_sync_client = None
        self.source_finalize_client = None
        self.destination_finalize_client = None

    async def initialize(self) -> None:
        endpoints = {
            "fast_client": f"{self.worker_namespace}.fast.generate",
            "slow_client": f"{self.worker_namespace}.slow.generate",
            "destination_prepare_client": (
                f"{self.worker_namespace}.slow.migration_prepare"
            ),
            "source_sync_client": f"{self.worker_namespace}.fast.migration_sync",
            "source_finalize_client": (
                f"{self.worker_namespace}.fast.migration_finalize"
            ),
            "destination_finalize_client": (
                f"{self.worker_namespace}.slow.migration_finalize"
            ),
        }
        for attr, endpoint_name in endpoints.items():
            client = await self.runtime.endpoint(endpoint_name).client()
            await client.wait_for_instances()
            setattr(self, attr, client)
            logger.info("Connected %s to %s", attr, endpoint_name)

    async def _one(self, client, request: Dict[str, Any]) -> Dict[str, Any]:
        stream = await client.generate(request)
        async for response in stream:
            return _response_data(response)
        raise RuntimeError("Control endpoint returned an empty stream")

    async def _finalize_source(
        self, rid: str, migration_id: str, action: str
    ) -> Dict[str, Any]:
        assert self.source_finalize_client is not None
        request = {"rid": rid, "migration_id": migration_id, "action": action}
        deadline = asyncio.get_running_loop().time() + 10.0
        while True:
            result = await self._one(self.source_finalize_client, request)
            if result.get("success") or action != "commit":
                return result
            if result.get("transfer_status") not in (
                "bootstrapping",
                "transferring",
            ):
                return result
            if asyncio.get_running_loop().time() >= deadline:
                return result
            await asyncio.sleep(0.01)

    async def _finalize_destination(
        self, rid: str, migration_id: str, action: str
    ) -> Dict[str, Any]:
        assert self.destination_finalize_client is not None
        return await self._one(
            self.destination_finalize_client,
            {
                "rid": rid,
                "migration_id": migration_id,
                "side": "destination",
                "action": action,
            },
        )

    async def generate(
        self, request: Dict[str, Any], context
    ) -> AsyncGenerator[Dict[str, Any], None]:
        assert self.fast_client is not None
        assert self.slow_client is not None
        assert self.destination_prepare_client is not None
        assert self.source_sync_client is not None

        rid = context.trace_id or uuid.uuid4().hex
        migration_id = uuid.uuid4().hex
        migration_active = False
        migration_committed = False
        migration_attempted = False
        forwarded_tokens = 0
        source_skip_tokens = 0
        migration_enabled = self.migrate_after_tokens > 0 or bool(
            self.migrate_on_token_ids
        )
        migration_trigger_seen = False

        source_request = copy.deepcopy(request)
        source_request["_decode_migration_rid"] = rid
        source_stream = await self.fast_client.generate(source_request, context=context)
        try:
            async for source_response in source_stream:
                source_output = _response_data(source_response)
                source_output, dropped = _trim_output_prefix(
                    source_output, source_skip_tokens
                )
                source_skip_tokens -= dropped

                token_ids = list(source_output.get("token_ids") or [])
                if token_ids or source_output.get("finish_reason") is not None:
                    if not context.is_stopped():
                        yield source_output
                    forwarded_tokens += len(token_ids)
                    matched_trigger_ids = [
                        token_id
                        for token_id in token_ids
                        if token_id in self.migrate_on_token_ids
                    ]
                    if matched_trigger_ids and not migration_trigger_seen:
                        migration_trigger_seen = True
                        logger.info(
                            "Observed migration token boundary request_id=%s "
                            "token_ids=%s forwarded_tokens=%d",
                            rid,
                            matched_trigger_ids,
                            forwarded_tokens,
                        )

                if (
                    context.is_stopped()
                    or source_output.get("finish_reason") is not None
                ):
                    return
                if self.migrate_on_token_ids:
                    if not migration_trigger_seen:
                        continue
                elif forwarded_tokens < self.migrate_after_tokens:
                    continue
                if not migration_enabled:
                    continue
                if migration_attempted:
                    continue
                migration_attempted = True

                source = await self._one(
                    self.source_sync_client,
                    {
                        "rid": rid,
                        "migration_id": migration_id,
                        "phase": "describe",
                    },
                )
                if not source.get("success"):
                    logger.warning("Source describe failed rid=%s: %s", rid, source)
                    continue

                prompt_len = len(request.get("token_ids") or [])
                max_tokens = int(
                    (request.get("stop_conditions") or {}).get("max_tokens") or 0
                )
                reserve = await self._one(
                    self.destination_prepare_client,
                    {
                        "rid": rid,
                        "migration_id": migration_id,
                        "source": {
                            "bootstrap_host": source["bootstrap_host"],
                            "bootstrap_port": source["bootstrap_port"],
                            "dp_rank": source.get("source_dp_rank", 0),
                        },
                        "reserve_tokens": prompt_len + max_tokens,
                    },
                )
                if not reserve.get("success"):
                    logger.warning(
                        "Destination reservation declined rid=%s: %s", rid, reserve
                    )
                    continue

                prepare = await self._one(
                    self.source_sync_client,
                    {
                        "rid": rid,
                        "migration_id": migration_id,
                        "phase": "quiesce",
                        "bootstrap_room": reserve["bootstrap_room"],
                        "output_tokens_seen": forwarded_tokens,
                    },
                )
                if prepare.get("status") == "finished":
                    await self._finalize_destination(rid, migration_id, "abort")
                    continue
                if not prepare.get("success"):
                    await self._finalize_destination(rid, migration_id, "abort")
                    logger.warning(
                        "Migration prepare declined rid=%s status=%s error=%s",
                        rid,
                        prepare.get("status"),
                        prepare.get("error"),
                    )
                    continue

                migration_active = True
                unforwarded = list(
                    prepare.get("unforwarded_committed_output_ids") or []
                )
                if unforwarded:
                    synthetic = {"token_ids": unforwarded, "index": 0}
                    if not context.is_stopped():
                        yield synthetic
                    forwarded_tokens += len(unforwarded)

                committed_output_tokens = max(
                    0, int(prepare["committed_len"]) - int(prepare["prompt_len"])
                )
                destination_duplicate_tokens = max(
                    0, forwarded_tokens - committed_output_tokens
                )
                destination_request = _migrated_request(request, prepare)
                destination_request["_decode_migration_id"] = migration_id
                destination_request["_decode_migration_rid"] = rid

                arm = await self._one(
                    self.destination_prepare_client,
                    {
                        "rid": rid,
                        "migration_id": migration_id,
                        "source_state": {
                            "committed_input_ids": prepare["committed_input_ids"],
                            "pending_input_ids": prepare["pending_input_ids"],
                            "committed_len": prepare["committed_len"],
                            "logical_len": prepare["logical_len"],
                        },
                        "destination_request": destination_request,
                    },
                )
                if not arm.get("success") or arm.get("status") != "ready":
                    await self._finalize_source(rid, migration_id, "resume")
                    await self._finalize_destination(rid, migration_id, "abort")
                    logger.warning("Destination arm failed rid=%s: %s", rid, arm)
                    continue

                try:
                    if self.destination_start_delay_ms > 0:
                        await asyncio.sleep(self.destination_start_delay_ms / 1000)
                        if context.is_stopped():
                            return
                    if self.force_destination_failure:
                        raise RuntimeError("Injected destination startup failure")
                    destination_stream = await self.slow_client.generate(
                        destination_request, context=context
                    )
                    destination_ready = False
                    async for destination_response in destination_stream:
                        destination_output = _response_data(destination_response)
                        if context.is_stopped():
                            return
                        destination_output, dropped = _trim_output_prefix(
                            destination_output, destination_duplicate_tokens
                        )
                        destination_duplicate_tokens -= dropped

                        if not destination_ready:
                            activate = await self._finalize_destination(
                                rid, migration_id, "activate"
                            )
                            if not activate.get("success"):
                                raise RuntimeError(
                                    "Destination activation failed: "
                                    f"{activate.get('error') or activate}"
                                )
                            commit = await self._finalize_source(
                                rid, migration_id, "commit"
                            )
                            if not commit.get("success"):
                                raise RuntimeError(
                                    "Source commit failed: "
                                    f"{commit.get('error') or commit}"
                                )
                            destination_ready = True
                            migration_committed = True
                            migration_active = False

                        if (
                            destination_output.get("token_ids")
                            or destination_output.get("finish_reason") is not None
                        ) and not context.is_stopped():
                            yield destination_output
                    return
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "Destination migration failed; resuming source rid=%s", rid
                    )
                    await self._finalize_destination(rid, migration_id, "abort")
                    result = await self._finalize_source(rid, migration_id, "resume")
                    migration_active = False
                    if not result.get("success"):
                        raise RuntimeError(
                            f"Failed to resume source after migration: {result}"
                        )
                    # The synthetic committed tail may still be queued on the
                    # original worker stream. Skip exactly that tail on resume.
                    source_skip_tokens += len(unforwarded)
                    continue
        finally:
            if migration_active and not migration_committed:
                try:
                    await self._finalize_destination(rid, migration_id, "abort")
                    await self._finalize_source(rid, migration_id, "cancel")
                except Exception:
                    logger.exception(
                        "Failed to cancel source migration rid=%s migration_id=%s",
                        rid,
                        migration_id,
                    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", default="dynamo")
    parser.add_argument("--component", default="backend")
    parser.add_argument("--worker-namespace", default=None)
    parser.add_argument("--migrate-after-tokens", type=int, default=8)
    parser.add_argument(
        "--migrate-on-token-id",
        action="append",
        type=int,
        default=[],
        help="Migrate after any listed output token is forwarded.",
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--reasoning-parser", default=None)
    parser.add_argument("--stream-interval", type=int, default=1)
    parser.add_argument("--destination-start-delay-ms", type=int, default=0)
    parser.add_argument("--force-destination-failure", action="store_true")
    return parser.parse_args()


async def run(args: argparse.Namespace) -> None:
    loop = asyncio.get_running_loop()
    runtime = DistributedRuntime(loop)
    coordinator = DecodeMigrationCoordinator(
        runtime,
        migrate_after_tokens=args.migrate_after_tokens,
        migrate_on_token_ids=set(args.migrate_on_token_id),
        namespace=args.namespace,
        component=args.component,
        worker_namespace=args.worker_namespace,
        destination_start_delay_ms=args.destination_start_delay_ms,
        force_destination_failure=args.force_destination_failure,
    )
    await coordinator.initialize()
    endpoint = runtime.endpoint(f"{args.namespace}.{args.component}.generate")
    await endpoint.serve_endpoint(coordinator.generate)


@dynamo_worker()
async def worker(runtime: DistributedRuntime):
    args = parse_args()
    coordinator = DecodeMigrationCoordinator(
        runtime,
        migrate_after_tokens=args.migrate_after_tokens,
        migrate_on_token_ids=set(args.migrate_on_token_id),
        namespace=args.namespace,
        component=args.component,
        worker_namespace=args.worker_namespace,
        destination_start_delay_ms=args.destination_start_delay_ms,
        force_destination_failure=args.force_destination_failure,
    )
    await coordinator.initialize()
    endpoint = runtime.endpoint(f"{args.namespace}.{args.component}.generate")
    from dynamo.llm import (
        ModelInput,
        ModelRuntimeConfig,
        ModelType,
        WorkerType,
        register_model,
    )

    runtime_config = ModelRuntimeConfig()
    runtime_config.reasoning_parser = args.reasoning_parser
    runtime_config.set_engine_specific("stream_interval", str(args.stream_interval))

    await register_model(
        ModelInput.Tokens,
        ModelType.Chat | ModelType.Completions,
        endpoint,
        args.model_path,
        args.served_model_name,
        kv_cache_block_size=args.page_size,
        runtime_config=runtime_config,
        worker_type=WorkerType.Aggregated,
    )
    await endpoint.serve_endpoint(coordinator.generate)


if __name__ == "__main__":
    configure_dynamo_logging(service_name="decode-migration-coordinator")
    uvloop.install()
    asyncio.run(worker())
