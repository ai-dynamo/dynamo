# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""gRPC boundary for the t0 predictor."""

import logging

import grpc

from dynamo.planner.examples.external_plugin.t0_beta.predictor import T0Predictor
from dynamo.planner.plugins.proto.v1 import plugin_pb2 as pb
from dynamo.planner.plugins.proto.v1 import plugin_pb2_grpc as pbg

logger = logging.getLogger(__name__)


class T0Service(pbg.PredictPluginServicer):
    def __init__(self, predictor: T0Predictor) -> None:
        self._predictor = predictor

    async def Predict(
        self,
        request: pb.PredictStageRequest,
        context: grpc.aio.ServicerContext,
    ) -> pb.PredictStageResponse:
        try:
            return await self._predictor.predict(request)
        except (RuntimeError, ValueError) as error:
            logger.exception(
                "t0 prediction failed; planner should use builtin fallback"
            )
            await context.abort(grpc.StatusCode.UNAVAILABLE, str(error))
            raise  # abort raises; retain the original error if an implementation returns.
