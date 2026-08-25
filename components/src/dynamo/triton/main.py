# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import tritonclient.grpc.model_config_pb2 as mc
import uvloop
from google.protobuf import text_format
from tritonserver import Server as TritonServer

from dynamo.common.utils.graceful_shutdown import install_signal_handlers
from dynamo.common.utils.runtime import create_runtime
from dynamo.llm import ModelInput, ModelType, WorkerType, register_model
from dynamo.runtime import DistributedRuntime
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.triton.args import Config
from dynamo.triton.backend_args import DynamoTritonConfig, parse_args
from dynamo.triton.handlers import RequestHandler
from dynamo.triton.health_check import TritonHealthCheckPayload
from dynamo.triton.metrics import (
    TritonMetricsBridge,
    _register_triton_metrics_bridge,
    _stop_triton_server,
)
from dynamo.triton.util import create_triton_log_callback, endpoint_slug

logger = logging.getLogger(__name__)
configure_dynamo_logging()

# Use NVIDIA_TRITON_SERVER_VERSION in the QA workflow.
# .github/workflows/shared-triton-test.yml exports only NVIDIA_TRITON_SERVER_VERSION.
# For Triton 26.08 or newer, the worker reads an empty version and disables log_callback.
TRITON_VERSION = os.environ.get("NVIDIA_TRITON_SERVER_VERSION", "")

# tritonserver.Options gained the log_callback option in 26.08; older public
# release containers reject it, so log forwarding is gated on the container
# version (parsed as a YY.MM tuple; unparseable versions are treated as older).
_LOG_CALLBACK_MIN_VERSION = (26, 8)


def _triton_supports_log_callback() -> bool:
    try:
        year, month = (int(p) for p in TRITON_VERSION.split(".")[:2])
    except (ValueError, AttributeError):
        return False
    return (year, month) >= _LOG_CALLBACK_MIN_VERSION


async def _register_and_serve(
    runtime: DistributedRuntime,
    config: DynamoTritonConfig,
    server: TritonServer,
    model_repository: str,
    model_name: str,
    shutdown_endpoints: Optional[list] = None,
) -> None:
    # The endpoint name is slugified (Triton model names may contain characters
    # illegal in a Dynamo endpoint identifier) and suffixed with a content hash
    # so no endpoint name is a textual prefix of another, which would trip the
    # discovery store's boundary-less prefix scan. The model still registers
    # under its real name (model_name), so frontend routing is unaffected.
    endpoint_path = f"{config.namespace}.{config.server_id}.{endpoint_slug(model_name)}"
    endpoint = runtime.endpoint(endpoint_path)
    # Track the endpoint so the shutdown handler can unregister it from discovery
    # before tearing the runtime down (routers stop targeting this worker).
    if shutdown_endpoints is not None:
        shutdown_endpoints.append(endpoint)
    logger.info(
        f"✓ Created endpoint '{endpoint_path.replace('.', '/')}' for model '{model_name}'"
    )

    model = server.model(model_name)  # type: TritonModel
    logger.info(f"✓ Model '{model_name}' loaded")

    # Read Triton model config from config.pbtxt
    config_path = f"{model_repository}/{model_name}/config.pbtxt"
    with open(config_path, "r") as f:
        triton_model_config = text_format.Parse(f.read(), mc.ModelConfig())

    logger.info(f"Loaded model config from {config_path}")

    # Model metadata for the KServe frontend. register_model reads the tensor
    # protocol layout for tensor-based models from tensor_model_config.
    model_config = {
        "name": "",
        "inputs": [],
        "outputs": [],
        "triton_model_config": triton_model_config.SerializeToString(),
    }

    logger.info(f"Attempting to register model '{model_name}' with Dynamo runtime...")
    # register_model for tensor-based models skips HuggingFace downloads.
    await register_model(
        ModelInput.Tensor,
        ModelType.TensorBased,
        endpoint,
        model_name,  # model_path (used as display name for tensor-based models)
        worker_type=WorkerType.Aggregated,
        tensor_model_config=model_config,
    )
    logger.info(
        f"✓ Successfully registered model '{model_name}' with endpoint "
        f"{endpoint_path.replace('.', '/')}"
    )

    handler = RequestHandler(server, model)
    health_check_payload = TritonHealthCheckPayload(model_name).to_dict()
    logger.info(f"Serving endpoint for model '{model_name}'...")
    await endpoint.serve_endpoint(
        handler.generate,
        health_check_payload=health_check_payload,
    )


@dataclass
class WorkerState:
    """Shared worker lifecycle state populated by init_worker and consumed by
    the shutdown handler."""

    endpoints: list = field(default_factory=list)
    server: Optional[TritonServer] = None
    metrics_bridge: Optional[TritonMetricsBridge] = None


async def init_worker(
    runtime: DistributedRuntime,
    config: Config,
    worker_state: Optional[WorkerState] = None,
):
    logger.info("Starting Triton Runtime for Dynamo")

    if worker_state is None:
        worker_state = WorkerState()

    model_repository = config.model_repository

    server_options = config.to_server_options()

    # Forward Triton server logs to Dynamo's logging pipeline so the worker
    # produces a single, consistently formatted log stream instead of Triton's
    # separate stdout/stderr outputs. Only supported on Triton 26.08+.
    if _triton_supports_log_callback():
        server_options["log_callback"] = create_triton_log_callback()
    else:
        logger.warning(
            "Triton %s predates the log_callback API (26.08+); Triton Runtime "
            "logs will not be routed through Dynamo's logging pipeline.",
            TRITON_VERSION or "version unknown",
        )

    logger.info(
        f"Initializing Triton Runtime with model_repository={model_repository}, "
        f"backend_directory={server_options.get('backend_directory')}"
    )
    logger.debug(f"Triton Runtime options: {server_options}")

    server = TritonServer(**server_options)
    server.start(wait_until_ready=True)
    logger.info("✓ Triton Runtime started")

    # Expose the started server so the shutdown handler's cleanup callback can
    # stop it (unload models, release GPU memory) before the runtime tears down.
    worker_state.server = server

    # Bridge Triton's native metrics into Dynamo's /metrics whenever Triton
    # metrics collection is enabled.
    if config.metrics is not False:
        worker_state.metrics_bridge = _register_triton_metrics_bridge(
            runtime, config, server
        )

    model_names = sorted(
        {name for name, _version in server.models(exclude_not_ready=True)}
    )
    if not model_names:
        raise RuntimeError(f"No ready models found in repository '{model_repository}'.")

    logger.info(f"Auto-discovered {len(model_names)} model(s): {model_names}")

    logger.info(f"Serving {len(model_names)} model(s): {model_names}")

    # Register and serve every model concurrently. Each model gets its own
    # endpoint URI (<namespace>.<server_id>.<model_name>) and its own handler bound
    # to that model, so requests are routed by model name via Dynamo's frontend.
    await asyncio.gather(
        *[
            _register_and_serve(
                runtime, config, server, model_repository, name, worker_state.endpoints
            )
            for name in model_names
        ]
    )


async def worker() -> None:
    config = parse_args(sys.argv[1:])
    runtime, loop = create_runtime(
        discovery_backend=config.discovery_backend,
        request_plane=config.request_plane,
    )

    # Graceful shutdown: on SIGTERM/SIGINT, unregister the model endpoints from
    # discovery, wait out the grace period, then stop the Triton Server before
    # the runtime tears down. init_worker fills in `worker_state` once it has
    # created the endpoints and started the server.
    worker_state = WorkerState()

    async def _stop_server() -> None:
        if worker_state.server is None:
            return

        # Server.stop() is blocking (unloads models, frees GPU memory); run
        # it off the event loop so the shutdown coroutine isn't blocked.
        if worker_state.metrics_bridge is not None:
            # Wait for an active scrape and disable future Triton callbacks
            # before tearing down the native server.
            await asyncio.to_thread(
                _stop_triton_server,
                worker_state.server,
                worker_state.metrics_bridge,
            )
        else:
            # Metrics collection disabled: just stop the server.
            await asyncio.to_thread(worker_state.server.stop)

    install_signal_handlers(
        loop,
        runtime,
        worker_state.endpoints,
        cleanup_callback=_stop_server,
    )

    await init_worker(runtime, config, worker_state)


def main():
    uvloop.run(worker())


if __name__ == "__main__":
    main()
