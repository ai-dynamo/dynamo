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

"""Serve a manually orchestrated graph or one of its Dynamo stages."""

import argparse
import asyncio
from collections.abc import AsyncIterator, Mapping
from typing import Any

from dynamo.runtime import Context, DistributedRuntime, dynamo_worker
from examples.custom_backend.workflow_hello_world.common import (
    HelloStage,
    MergeStage,
    WorldStage,
)

ORCHESTRATOR_ENDPOINT = "workflow-hello-world.manual.generate"
STAGE_ENDPOINTS = {
    "hello": "workflow-hello-world.manual-hello.generate",
    "world": "workflow-hello-world.manual-world.generate",
    "merge": "workflow-hello-world.manual-merge.generate",
}
STAGE_TYPES = {
    "hello": HelloStage,
    "world": WorldStage,
    "merge": MergeStage,
}


class ManualOrchestrator:
    """Express the fixed graph directly as Python control flow."""

    def __init__(self, stage_clients: Mapping[str, Any]) -> None:
        self._stage_clients = dict(stage_clients)

    async def _call_stage(
        self, stage_name: str, request: Mapping[str, Any], context: Context
    ) -> Mapping[str, Any]:
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

        if not responses or not isinstance(responses[0], Mapping):
            raise RuntimeError(f"stage {stage_name!r} returned an invalid response")
        return responses[0]

    async def _fan_out(
        self, request: Mapping[str, Any], context: Context
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        tasks = {
            stage_name: asyncio.create_task(
                self._call_stage(stage_name, request, context),
                name=f"dynamo-manual:{stage_name}",
            )
            for stage_name in ("hello", "world")
        }
        try:
            return await asyncio.gather(tasks["hello"], tasks["world"])
        except BaseException:
            for task in tasks.values():
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks.values(), return_exceptions=True)
            raise

    async def generate(
        self, request: Mapping[str, Any], context: Context
    ) -> AsyncIterator[dict[str, Any]]:
        hello, world = await self._fan_out(request, context)
        result = await self._call_stage(
            "merge",
            {"hello": hello["text"], "world": world["text"]},
            context,
        )
        chunk = result["chunk"]
        if not isinstance(chunk, Mapping):
            raise RuntimeError("merge stage returned an invalid chunk")
        yield dict(chunk)


class StageServer:
    """Expose one shared stage implementation through a Dynamo endpoint."""

    def __init__(self, stage_name: str) -> None:
        self._stage_name = stage_name
        self._stage = STAGE_TYPES[stage_name]()

    async def generate(
        self, request: Mapping[str, Any], context: Context
    ) -> AsyncIterator[dict[str, Any]]:
        del context
        if self._stage_name == "merge":
            text = await self._stage.run(request["hello"], request["world"])
            yield {
                "chunk": {
                    "token_ids": [],
                    "text": text,
                    "index": 0,
                    "finish_reason": "stop",
                }
            }
            return
        yield {"text": await self._stage.run(request)}


async def _serve_orchestrator(runtime: DistributedRuntime) -> None:
    stage_clients = {
        stage_name: await runtime.endpoint(endpoint_id).client()
        for stage_name, endpoint_id in STAGE_ENDPOINTS.items()
    }
    await asyncio.gather(
        *(client.wait_for_instances() for client in stage_clients.values())
    )
    endpoint = runtime.endpoint(ORCHESTRATOR_ENDPOINT)
    await endpoint.serve_endpoint(ManualOrchestrator(stage_clients).generate)


async def _serve_stage(runtime: DistributedRuntime, stage_name: str) -> None:
    endpoint = runtime.endpoint(STAGE_ENDPOINTS[stage_name])
    await endpoint.serve_endpoint(StageServer(stage_name).generate)


@dynamo_worker()
async def worker(runtime: DistributedRuntime, role: str) -> None:
    if role == "orchestrator":
        await _serve_orchestrator(runtime)
    else:
        await _serve_stage(runtime, role)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("orchestrator", *STAGE_ENDPOINTS))
    args = parser.parse_args()
    asyncio.run(worker(args.role))


if __name__ == "__main__":
    main()
