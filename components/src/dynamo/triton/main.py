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
from tritonserver import Model as TritonModel
from tritonserver import Server as TritonServer

from dynamo.common.utils.graceful_shutdown import install_signal_handlers
from dynamo.common.utils.runtime import create_runtime
from dynamo.llm import ModelInput, ModelType, WorkerType, register_model
from dynamo.runtime import DistributedRuntime
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.triton.backend_args import DynamoTritonConfig, parse_args
from dynamo.triton.handlers import RequestHandler
from dynamo.triton.health_check import TritonHealthCheckPayload
from dynamo.triton.metrics import (
    TritonMetricsBridge,
    _register_triton_metrics_bridge,
    _stop_triton_server,
)
from dynamo.triton.pooling_handlers import ClassifyWorkerHandler
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


def _read_model_config(
    model: TritonModel, model_name: str, repository_path: str
) -> bytes:
    model_config = None if model is None else model.config()
    if model_config is None or len(model_config) == 0:
        logger.debug("Failed to read model config from Triton.")
        # Read Triton model config from config.pbtxt
        config_path = f"{repository_path}/{model_name}/config.pbtxt"
        with open(config_path, "r") as f:
            model_config = text_format.Parse(f.read(), mc.ModelConfig())
            serialized_config = model_config.SerializeToString()
            logger.info(f"Loaded model config from {config_path}")
            logger.debug(serialized_config)
            return serialized_config
    else:
        # model.config() returns dict[str, Any]; parse it back into the ModelConfig
        # protobuf and serialize to bytes, matching the config.pbtxt branch above.
        from google.protobuf import json_format

        model_config_pb = json_format.ParseDict(model_config, mc.ModelConfig())
        serialized_config = model_config_pb.SerializeToString()
        logger.info("Read model config from Triton.")
        logger.debug(serialized_config)
        return serialized_config


def _collect_classify_dependency_models(
    server: TritonServer,
    model_names: list[str],
    repository_path: str,
) -> set[str]:
    """Return names of models referenced as a step inside another model's ensemble.

    Invoked only on the ``--task classify`` path. A typical classify
    ensemble in Triton pairs one ``ensemble`` model with a Python
    tokenizer dependency and a numeric classifier dependency; Triton
    loads all three as ready, but only the ensemble carries the
    STRING-in / FP32-out contract the OpenAI ``/v1/classify`` adapter
    needs. Constructing a ``ClassifyWorkerHandler`` for the tokenizer or
    numeric stage raises, which cancels the entire TaskGroup and aborts
    the valid ensemble with it.

    Dependency models remain loaded in Triton so their ensembles can
    call them, but are not exposed as Dynamo endpoints.
    """
    deps: set[str] = set()
    for name in model_names:
        try:
            model = server.model(name)
            cfg = mc.ModelConfig.FromString(
                _read_model_config(model, name, repository_path)
            )
        except (OSError, text_format.ParseError, mc.DecodeError) as exc:
            # A config we cannot read also cannot declare dependencies; let
            # the model fall through to registration, where the same read
            # path raises with the full error context.
            logger.warning(
                "Could not read config for model %r while scanning for "
                "ensemble dependencies: %s",
                name,
                exc,
            )
            continue
        if cfg.HasField("ensemble_scheduling"):
            for step in cfg.ensemble_scheduling.step:
                if step.model_name:
                    deps.add(step.model_name)
    return deps


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

    model = server.model(model_name)
    logger.info(f"✓ Model '{model_name}' loaded")

    triton_model_config = _read_model_config(model, model_name, model_repository)

    # Model metadata for the KServe frontend. register_model reads the tensor
    # protocol layout for tensor-based models from tensor_model_config.
    tensor_model_config = {
        "name": "",
        "inputs": [],
        "outputs": [],
        "triton_model_config": triton_model_config,
    }

    if config.task == "classify":
        model_input = ModelInput.Text
        model_type = ModelType.Classify
    else:
        model_input = ModelInput.Tensor
        model_type = ModelType.TensorBased

    logger.info(
        f"Attempting to register model '{model_name}' with Dynamo runtime "
        f"(task={config.task}, model_type={model_type})..."
    )
    await register_model(
        model_input,
        model_type,
        endpoint,
        model_name,  # model_path (used as display name for tensor-based models)
        worker_type=WorkerType.Aggregated,
        tensor_model_config=tensor_model_config,
    )
    logger.info(
        f"✓ Successfully registered model '{model_name}' with endpoint "
        f"{endpoint_path.replace('.', '/')}"
    )

    handler = _build_handler(config, server, model, triton_model_config)
    health_check_payload = TritonHealthCheckPayload(model_name).to_dict()
    logger.info(f"Serving endpoint for model '{model_name}'...")
    await endpoint.serve_endpoint(
        handler.generate,
        health_check_payload=health_check_payload,
    )


def _build_handler(
    config: DynamoTritonConfig,
    server: TritonServer,
    model: TritonModel,
    triton_model_config_bytes: bytes,
):
    # Parsed once so both handlers share the same source of truth.
    parsed_config = mc.ModelConfig.FromString(triton_model_config_bytes)
    if config.task == "classify":
        return ClassifyWorkerHandler(
            server,
            model,
            parsed_config,
            classify_input_name=config.classify_input_name,
            classify_output_name=config.classify_output_name,
        )
    return RequestHandler(server, model, parsed_config)


@dataclass
class WorkerState:
    """Shared worker lifecycle state populated by init_worker and consumed by
    the shutdown handler."""

    endpoints: list = field(default_factory=list)
    server: Optional[TritonServer] = None
    metrics_bridge: Optional[TritonMetricsBridge] = None


async def init_worker(
    runtime: DistributedRuntime,
    config: DynamoTritonConfig,
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

    # Only user-facing classify models get a Dynamo endpoint. In a typical
    # Triton classify ensemble (ensemble + Python tokenizer + numeric stage),
    # the tokenizer has no FP32 output and the numeric stage has no STRING
    # input, so constructing a ClassifyWorkerHandler for either raises and
    # cancels every sibling task in the TaskGroup below. Dependency models
    # stay loaded in Triton to serve their ensembles but are not exposed as
    # Dynamo endpoints.
    #
    # The tensor path registers everything: a dependency model may still be
    # useful to call directly over KServe gRPC for debugging.
    if config.task == "classify":
        deps = _collect_classify_dependency_models(
            server, model_names, model_repository
        )
        skipped = sorted(set(model_names) & deps)
        exposed = [n for n in model_names if n not in deps]
        if skipped:
            logger.info(
                "Skipping %d ensemble dependency model(s) from "
                "/v1/classify registration (still loaded in Triton for use "
                "by their ensembles): %s",
                len(skipped),
                skipped,
            )
        if not exposed:
            raise RuntimeError(
                "No user-facing classify models found in "
                f"'{model_repository}'. Every ready model is referenced "
                "as an ensemble step of another model. Add a user-facing "
                "ensemble (STRING in, FP32 out) that wires these "
                "together, or run with --task tensor."
            )
        model_names = exposed

    logger.info(f"Serving {len(model_names)} model(s): {model_names}")

    # Register and serve every model concurrently. Each model gets its own
    # endpoint URI (<namespace>.<server_id>.<model_name>) and its own handler bound
    # to that model, so requests are routed by model name via Dynamo's frontend.
    async with asyncio.TaskGroup() as aio_tasks:
        for name in model_names:
            aio_tasks.create_task(
                _register_and_serve(
                    runtime,
                    config,
                    server,
                    model_repository,
                    name,
                    worker_state.endpoints,
                ),
                name=f"dynamo.triton/model={name}",
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

        # Use a local copy of server and set work_state.server to None
        # to make this call idempotent.
        # Allowing the server to be "stopped" multiple times leads to
        # unspecified outcomes.
        server = worker_state.server
        worker_state.server = None

        # Server.stop() is blocking (unloads models, frees GPU memory); run
        # it off the event loop so the shutdown coroutine isn't blocked.
        if worker_state.metrics_bridge is not None:
            # Wait for an active scrape and disable future Triton callbacks
            # before tearing down the native server.
            await asyncio.to_thread(
                _stop_triton_server,
                server,
                worker_state.metrics_bridge,
            )
        else:
            # Metrics collection disabled: just stop the server.
            await asyncio.to_thread(server.stop)

    install_signal_handlers(
        loop,
        runtime,
        worker_state.endpoints,
        cleanup_callback=_stop_server,
    )

    try:
        await init_worker(runtime, config, worker_state)
    except Exception:
        await _stop_server()
        raise


def main():
    uvloop.run(worker())


if __name__ == "__main__":
    main()
