# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Serve private stages or their manually coded public orchestrator."""

import argparse
import asyncio
from collections.abc import AsyncIterator, Mapping
from typing import Any

import uvloop

from dynamo.llm import ModelInput, ModelType, WorkerType, register_model
from dynamo.runtime import Context, DistributedRuntime, dynamo_worker
from examples.custom_backend.workflow_hello_world.common import (
    HelloStage,
    MergeStage,
    WorldStage,
)

MODEL_NAME = "hello-world"
ORCHESTRATOR_ENDPOINT = "workflow-hello-world.manual.generate"
STAGE_ENDPOINTS = {
    "hello": "workflow-hello-world.manual-hello.generate",
    "world": "workflow-hello-world.manual-world.generate",
}
STAGE_TYPES = {
    "hello": HelloStage,
    "world": WorldStage,
}


class ManualOrchestrator:
    """Own fan-out, cancellation, validation, merge, and output shaping."""

    def __init__(self, stage_clients: Mapping[str, Any]) -> None:
        self._stage_clients = dict(stage_clients)
        self._merge = MergeStage()

    async def _call_stage(
        self, stage_name: str, request: Mapping[str, Any], context: Context
    ) -> str:
        child_context = context.detached(f"{context.id()}:{stage_name}")
        stream = await self._stage_clients[stage_name].round_robin(
            request,
            annotated=False,
            context=child_context,
        )
        responses = []
        try:
            async for response in stream:
                responses.append(response)
                if len(responses) > 1:
                    raise RuntimeError(
                        f"stage {stage_name!r} returned multiple responses"
                    )
        except BaseException:
            child_context.stop_generating()
            raise

        if not responses:
            raise RuntimeError(f"stage {stage_name!r} returned no response")
        response = responses[0]
        if not isinstance(response, Mapping) or not isinstance(
            response.get("text"), str
        ):
            raise RuntimeError(f"stage {stage_name!r} returned an invalid response")
        return response["text"]

    async def _fan_out(
        self, request: Mapping[str, Any], context: Context
    ) -> tuple[str, str]:
        tasks = {
            stage_name: asyncio.create_task(
                self._call_stage(stage_name, request, context),
                name=f"dynamo-manual:{stage_name}",
            )
            for stage_name in STAGE_ENDPOINTS
        }
        try:
            hello, world = await asyncio.gather(tasks["hello"], tasks["world"])
        except BaseException:
            for task in tasks.values():
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks.values(), return_exceptions=True)
            raise
        return hello, world

    async def generate(
        self, request: Mapping[str, Any], context: Context
    ) -> AsyncIterator[dict[str, Any]]:
        fan_out = asyncio.create_task(
            self._fan_out(request, context),
            name=f"dynamo-manual:{context.id()}",
        )
        cancelled = asyncio.ensure_future(context.async_killed_or_stopped())
        try:
            done, _ = await asyncio.wait(
                {fan_out, cancelled}, return_when=asyncio.FIRST_COMPLETED
            )
            if fan_out not in done:
                fan_out.cancel()
                await asyncio.gather(fan_out, return_exceptions=True)
                return
            hello, world = fan_out.result()
        finally:
            if not cancelled.done():
                cancelled.cancel()
            await asyncio.gather(cancelled, return_exceptions=True)

        text = await self._merge.run(hello, world)
        yield {
            "token_ids": [],
            "text": text,
            "index": 0,
            "finish_reason": "stop",
        }


class StageServer:
    """Adapt one shared stage implementation to a private Dynamo endpoint."""

    def __init__(self, stage: Any) -> None:
        self._stage = stage

    async def generate(
        self, request: Mapping[str, Any], context: Context
    ) -> AsyncIterator[dict[str, str]]:
        del context
        yield {"text": await self._stage.run(request)}


async def _serve_orchestrator(runtime: DistributedRuntime, model_path: str) -> None:
    stage_clients = {
        stage_name: await runtime.endpoint(endpoint_id).client()
        for stage_name, endpoint_id in STAGE_ENDPOINTS.items()
    }
    await asyncio.gather(
        *(client.wait_for_instances() for client in stage_clients.values())
    )

    endpoint = runtime.endpoint(ORCHESTRATOR_ENDPOINT)
    await register_model(
        model_input=ModelInput.Tokens,
        model_type=ModelType.Chat | ModelType.Completions,
        endpoint=endpoint,
        model_path=model_path,
        model_name=MODEL_NAME,
        worker_type=WorkerType.Aggregated,
        needs=[],
        ignore_weights=True,
    )
    await endpoint.serve_endpoint(ManualOrchestrator(stage_clients).generate)


async def _serve_stage(runtime: DistributedRuntime, stage_name: str) -> None:
    endpoint = runtime.endpoint(STAGE_ENDPOINTS[stage_name])
    stage = STAGE_TYPES[stage_name]()
    await endpoint.serve_endpoint(StageServer(stage).generate)


@dynamo_worker()
async def worker(runtime: DistributedRuntime, role: str, model_path: str) -> None:
    if role == "orchestrator":
        await _serve_orchestrator(runtime, model_path)
    else:
        await _serve_stage(runtime, role)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("orchestrator", *STAGE_ENDPOINTS))
    parser.add_argument("--model-path", default="Qwen/Qwen3-0.6B")
    args = parser.parse_args()
    uvloop.run(worker(args.role, args.model_path))


if __name__ == "__main__":
    main()
