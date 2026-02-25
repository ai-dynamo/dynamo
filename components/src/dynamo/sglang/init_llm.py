# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os
import time
from typing import Awaitable, Callable, Optional

import sglang as sgl

from dynamo.common.utils.endpoint_types import parse_endpoint_types
from dynamo.llm import ModelInput, ModelType
from dynamo.runtime import DistributedRuntime
from dynamo.sglang.args import Config
from dynamo.sglang.health_check import (
    SglangHealthCheckPayload,
    SglangPrefillHealthCheckPayload,
)
from dynamo.sglang.publisher import handle_non_leader_node, setup_sgl_metrics
from dynamo.sglang.register import register_model_with_readiness_gate
from dynamo.sglang.request_handlers import DecodeWorkerHandler, PrefillWorkerHandler


async def _warmup_prefill_engine(engine: sgl.Engine, server_args) -> None:
    """Perform warmup request for prefill engine to reduce initial TTFT."""
    logging.info("Start of prefill disaggregation warmup ...")
    try:
        from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
        from sglang.srt.sampling.sampling_params import SamplingParams

        sampling_params = SamplingParams(
            temperature=0.0,
            max_new_tokens=8,
            ignore_eos=True,
        )

        async def _do_warmup():
            results = await engine.async_generate(
                input_ids=[0, 1, 2, 3],
                sampling_params=sampling_params,
                stream=True,
                bootstrap_host=FAKE_BOOTSTRAP_HOST,
                bootstrap_port=server_args.disaggregation_bootstrap_port,
                bootstrap_room=999999,
            )
            async for _ in results:
                pass

        await asyncio.wait_for(_do_warmup(), timeout=1800)
        logging.info("Prefill warmup completed")
    except asyncio.TimeoutError:
        logging.warning("Prefill warmup timed out after 1800s")
    except Exception as e:
        logging.warning(f"Prefill warmup failed: {e}")


async def init_decode(
    runtime: DistributedRuntime,
    config: Config,
    shutdown_event: asyncio.Event,
    shutdown_endpoints: list,
    run_deferred_handlers: Callable[[], Awaitable[None]] | None = None,
    pre_created_engine: Optional[sgl.Engine] = None,
):
    server_args, dynamo_args = config.server_args, config.dynamo_args

    if server_args.node_rank >= 1:
        os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"

    # Use pre-created engine if provided (checkpoint/restore mode)
    if pre_created_engine is not None:
        engine = pre_created_engine
        load_time = 0.0
    else:
        start_time = time.time()
        engine = sgl.Engine(server_args=server_args)
        load_time = time.time() - start_time

    generate_endpoint = runtime.endpoint(
        f"{dynamo_args.namespace}.{dynamo_args.component}.{dynamo_args.endpoint}"
    )

    shutdown_endpoints[:] = [generate_endpoint]

    publisher, metrics_task, metrics_labels = await setup_sgl_metrics(
        engine, config, generate_endpoint
    )

    publisher.component_gauges.set_model_load_time(load_time)
    logging.debug(f"SGLang model load time: {load_time:.2f}s")

    if server_args.node_rank >= 1:
        await handle_non_leader_node(engine, publisher, metrics_task)
        return

    ready_event = asyncio.Event()

    handler = DecodeWorkerHandler(
        engine, config, publisher, generate_endpoint, shutdown_event
    )
    handler.register_engine_routes(runtime)

    health_check_payload = SglangHealthCheckPayload(
        engine, use_text_input=dynamo_args.use_sglang_tokenizer
    ).to_dict()

    logging.info(f"Registering model with endpoint types: {dynamo_args.endpoint_types}")
    if dynamo_args.custom_jinja_template and "chat" not in dynamo_args.endpoint_types:
        logging.warning(
            "Custom Jinja template provided (--custom-jinja-template) but 'chat' not in --dyn-endpoint-types. "
            "The chat template will be loaded but the /v1/chat/completions endpoint will not be available."
        )

    try:
        await asyncio.gather(
            generate_endpoint.serve_endpoint(
                handler.generate,
                graceful_shutdown=True,
                metrics_labels=metrics_labels,
                health_check_payload=health_check_payload,
            ),
            register_model_with_readiness_gate(
                engine,
                generate_endpoint,
                server_args,
                dynamo_args,
                output_type=parse_endpoint_types(dynamo_args.endpoint_types),
                readiness_gate=ready_event,
            ),
        )
    except Exception as e:
        logging.error(f"Failed to serve endpoints: {e}")
        raise
    finally:
        metrics_task.cancel()
        try:
            await metrics_task
        except asyncio.CancelledError:
            logging.info("Metrics task successfully cancelled")
            pass
        handler.cleanup()
        if run_deferred_handlers is not None:
            logging.info("Running deferred handlers")
            await run_deferred_handlers()


async def init_prefill(
    runtime: DistributedRuntime,
    config: Config,
    shutdown_event: asyncio.Event,
    shutdown_endpoints: list,
    run_deferred_handlers: Callable[[], Awaitable[None]] | None = None,
    pre_created_engine: Optional[sgl.Engine] = None,
):
    server_args, dynamo_args = config.server_args, config.dynamo_args

    if server_args.node_rank >= 1:
        os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"

    # Use pre-created engine if provided (checkpoint/restore mode)
    if pre_created_engine is not None:
        engine = pre_created_engine
    else:
        engine = sgl.Engine(server_args=server_args)

    generate_endpoint = runtime.endpoint(
        f"{dynamo_args.namespace}.{dynamo_args.component}.{dynamo_args.endpoint}"
    )

    shutdown_endpoints[:] = [generate_endpoint]

    publisher, metrics_task, metrics_labels = await setup_sgl_metrics(
        engine, config, generate_endpoint
    )

    if server_args.node_rank >= 1:
        await handle_non_leader_node(engine, publisher, metrics_task)
        return

    await _warmup_prefill_engine(engine, server_args)

    handler = PrefillWorkerHandler(
        engine, config, publisher, generate_endpoint, shutdown_event
    )
    handler.register_engine_routes(runtime)

    health_check_payload = SglangPrefillHealthCheckPayload(engine).to_dict()

    ready_event = asyncio.Event()

    try:
        await asyncio.gather(
            generate_endpoint.serve_endpoint(
                handler.generate,
                graceful_shutdown=True,
                metrics_labels=metrics_labels,
                health_check_payload=health_check_payload,
            ),
            register_model_with_readiness_gate(
                engine,
                generate_endpoint,
                server_args,
                dynamo_args,
                input_type=ModelInput.Tokens,
                output_type=ModelType.Prefill,
                readiness_gate=ready_event,
            ),
        )
    except Exception as e:
        logging.error(f"Failed to serve endpoints: {e}")
        raise
    finally:
        metrics_task.cancel()
        try:
            await metrics_task
        except asyncio.CancelledError:
            logging.info("Metrics task successfully cancelled")
            pass
        handler.cleanup()
        if run_deferred_handlers is not None:
            logging.info("Running deferred handlers")
            await run_deferred_handlers()
