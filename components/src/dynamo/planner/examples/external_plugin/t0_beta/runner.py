# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a local t0-beta model as a statically registered external PREDICT plugin."""

import argparse
import asyncio
import logging
import signal

import grpc
from t0 import T0Forecaster

from dynamo.planner.examples.external_plugin.t0_beta.model import T0Forecast
from dynamo.planner.examples.external_plugin.t0_beta.predictor import T0Predictor
from dynamo.planner.examples.external_plugin.t0_beta.server import T0Service
from dynamo.planner.plugins.proto.v1 import plugin_pb2_grpc as pbg

logger = logging.getLogger(__name__)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen", default="127.0.0.1:9099")
    parser.add_argument("--model", default="theforecastingcompany/t0-beta")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--interval-seconds", type=float, default=60.0)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--min-history", type=int, default=32)
    parser.add_argument("--quantile", type=float, default=0.9)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    model = T0Forecaster.from_pretrained(args.model).to(args.device).eval()
    predictor = T0Predictor(
        T0Forecast(model),
        interval_seconds=args.interval_seconds,
        context_length=args.context_length,
        min_history=args.min_history,
        quantile=args.quantile,
    )
    server = grpc.aio.server()
    pbg.add_PredictPluginServicer_to_server(T0Service(predictor), server)
    if server.add_insecure_port(args.listen) == 0:
        raise RuntimeError(f"Could not bind plugin server to {args.listen}")
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    try:
        await server.start()
        logger.info("t0-beta predictor listening on %s", args.listen)
        await stop.wait()
    finally:
        await server.stop(grace=1.0)
        await predictor.close()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.remove_signal_handler(sig)


if __name__ == "__main__":
    asyncio.run(main())
