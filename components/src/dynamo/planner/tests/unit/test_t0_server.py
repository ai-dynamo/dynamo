# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the external plugin's actual protobuf boundary over localhost gRPC."""

from collections.abc import Sequence

import grpc
import pytest

from dynamo.planner.examples.external_plugin.t0_beta.predictor import (
    T0Predictor,
    Traffic,
)
from dynamo.planner.examples.external_plugin.t0_beta.server import T0Service
from dynamo.planner.plugins.proto.v1 import plugin_pb2 as pb
from dynamo.planner.plugins.proto.v1 import plugin_pb2_grpc as pbg

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


@pytest.mark.timeout(10)
@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_predict_rpc_returns_forecast_or_visible_failure(fail: bool) -> None:
    now: list[float] = [0.0]

    def forecast(history: Sequence[Traffic], quantile: float) -> Traffic:
        if fail:
            raise RuntimeError("inference unavailable")
        return (80.0, 200.0, 50.0)

    predictor = T0Predictor(forecast, min_history=2, clock=lambda: now[0])
    server = grpc.aio.server()
    pbg.add_PredictPluginServicer_to_server(T0Service(predictor), server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    try:
        async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = pbg.PredictPluginStub(channel)
            req = pb.PredictStageRequest(
                context=pb.PipelineContext(
                    observations=pb.ObservationData(
                        traffic=pb.TrafficMetrics(
                            duration_s=60, num_req=10, isl=100, osl=20
                        )
                    )
                )
            )
            assert not (await stub.Predict(req, timeout=2)).HasField("predictions")
            now[0] = 60
            if fail:
                with pytest.raises(grpc.aio.AioRpcError) as error:
                    await stub.Predict(req, timeout=2)
                assert error.value.code() == grpc.StatusCode.UNAVAILABLE
            else:
                result = await stub.Predict(req, timeout=2)
                assert result.predictions.predicted_num_req == 80
                assert result.predictions.predicted_isl == 200
                assert result.predictions.predicted_osl == 50
                assert not result.final
    finally:
        await server.stop(grace=0)
        await predictor.close()
