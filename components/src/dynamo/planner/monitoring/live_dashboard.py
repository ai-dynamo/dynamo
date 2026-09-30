# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight HTTP server for live planner diagnostics.

Serves the current accumulated snapshot data as an interactive Plotly HTML
report on demand, without clearing snapshots or writing to disk.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from aiohttp import web

if TYPE_CHECKING:
    from dynamo.planner.monitoring.diagnostics_recorder import DiagnosticsRecorder

logger = logging.getLogger(__name__)

_NO_DATA_HTML = (
    "<html><body><h3>Planner Live Dashboard</h3>"
    "<p>No snapshot data yet. Waiting for planner ticks...</p>"
    "<meta http-equiv='refresh' content='5'>"
    "</body></html>"
)


_AUTO_REFRESH_TAG = '<meta http-equiv="refresh" content="30">'


def _build_app(recorder: DiagnosticsRecorder) -> web.Application:
    async def handle_live(request: web.Request) -> web.Response:
        html = recorder.render_live_html()
        if html is None:
            return web.Response(text=_NO_DATA_HTML, content_type="text/html")
        # Inject auto-refresh into the Plotly HTML so the dashboard
        # updates every 30 seconds without manual browser refresh.
        html = html.replace("<head>", f"<head>{_AUTO_REFRESH_TAG}", 1)
        return web.Response(text=html, content_type="text/html")

    app = web.Application()
    app.router.add_get("/", handle_live)
    return app


async def start_live_dashboard(
    recorder: DiagnosticsRecorder, port: int
) -> web.AppRunner:
    """Start the live dashboard HTTP server.

    Returns the ``AppRunner`` so the caller can clean it up on shutdown.
    """
    app = _build_app(recorder)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    logger.info(f"Live diagnostics dashboard running on http://0.0.0.0:{port}/")
    return runner
