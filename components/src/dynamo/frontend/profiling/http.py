# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP adapter for the callable profiling coordinator."""

import msgspec
from aiohttp import web

from dynamo.profiling import Coordinator, CoordinatorError, Plan


@web.middleware
async def coordinator_errors(request, handler):
    try:
        return await handler(request)
    except CoordinatorError as exc:
        return web.Response(status=exc.status, text=str(exc))


def application(coordinator: Coordinator) -> web.Application:
    app = web.Application(client_max_size=65536, middlewares=[coordinator_errors])

    async def targets(request: web.Request) -> web.Response:
        return web.json_response({"targets": coordinator.targets()})

    async def read_plan(request: web.Request) -> Plan:
        try:
            return msgspec.json.decode(await request.read(), type=Plan)
        except msgspec.DecodeError as exc:
            raise web.HTTPBadRequest(text=str(exc)) from exc

    async def validate(request: web.Request) -> web.Response:
        plan = await read_plan(request)
        coordinator.validate(plan)
        return web.json_response(
            {
                "valid": True,
                "frontend_collector": "linux_perf"
                if coordinator.config.frontend_perf_enabled
                and plan.mode == "torch_profile"
                else None,
                "worker_collector": "torch_profiler"
                if plan.mode == "torch_profile"
                else plan.mode,
            }
        )

    async def submit(request: web.Request) -> web.Response:
        session = coordinator.submit(await read_plan(request))
        return web.json_response(coordinator.manifest(session), status=202)

    def lookup(request: web.Request):
        return coordinator.lookup(request.match_info["session_id"])

    async def status(request: web.Request) -> web.Response:
        return web.json_response(coordinator.manifest(lookup(request)))

    async def stop(request: web.Request) -> web.Response:
        session = coordinator.stop(request.match_info["session_id"])
        return web.json_response(coordinator.manifest(session), status=202)

    prefix = "/admin/profiling/v1"
    app.add_routes(
        [
            web.get(prefix + "/targets", targets),
            web.post(prefix + "/plans/validate", validate),
            web.post(prefix + "/sessions", submit),
            web.get(prefix + "/sessions/{session_id}", status),
            web.post(prefix + "/sessions/{session_id}/stop", stop),
            web.get(prefix + "/sessions/{session_id}/artifacts", status),
        ]
    )
    return app
