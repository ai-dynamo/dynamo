# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Lightweight round-robin reverse proxy for load-balancing across multiple
# Dynamo frontend replicas.  Streams SSE responses back to the client.
#
# Usage:
#   python lb_proxy.py --listen-port 9000 --backends 8000,8001,8002,...

from __future__ import annotations

import argparse
import itertools
import threading

from aiohttp import ClientSession, ClientTimeout, web


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Round-robin HTTP reverse proxy")
    parser.add_argument("--listen-port", type=int, default=9000)
    parser.add_argument(
        "--backends",
        type=str,
        required=True,
        help="Comma-separated backend ports (e.g. 8000,8001,8002)",
    )
    return parser.parse_args()


_lock = threading.Lock()


def _make_backend_cycle(ports: list[int]) -> itertools.cycle:
    urls = [f"http://localhost:{p}" for p in ports]
    return itertools.cycle(urls)


async def _proxy_handler(
    request: web.Request,
) -> web.StreamResponse:
    backend = next(request.app["backend_cycle"])
    target_url = f"{backend}{request.path_qs}"

    session: ClientSession = request.app["client_session"]
    body = await request.read()
    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in ("host", "transfer-encoding")
    }

    async with session.request(
        request.method,
        target_url,
        headers=headers,
        data=body,
    ) as resp:
        response = web.StreamResponse(
            status=resp.status,
            headers={
                k: v
                for k, v in resp.headers.items()
                if k.lower() not in ("transfer-encoding", "content-length")
            },
        )
        response.content_type = resp.content_type
        await response.prepare(request)

        async for chunk in resp.content.iter_any():
            await response.write(chunk)

        await response.write_eof()
        return response


async def _on_startup(app: web.Application) -> None:
    app["client_session"] = ClientSession(
        timeout=ClientTimeout(total=600),
    )


async def _on_cleanup(app: web.Application) -> None:
    await app["client_session"].close()


def main() -> None:
    args = parse_args()
    ports = [int(p.strip()) for p in args.backends.split(",")]
    print(f"LB proxy on :{args.listen_port} -> backends {ports}")

    app = web.Application()
    app["backend_cycle"] = _make_backend_cycle(ports)
    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)
    app.router.add_route("*", "/{path_info:.*}", _proxy_handler)

    web.run_app(app, port=args.listen_port, print=lambda msg: print(f"  {msg}"))


if __name__ == "__main__":
    main()
