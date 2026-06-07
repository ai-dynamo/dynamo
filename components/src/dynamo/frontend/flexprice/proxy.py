"""Dynamo auth proxy with optional async FlexPrice usage metering.

Request lifecycle (proxy is active whenever DYN_AUTH_ENABLED=true):

  1. Auth  — validate ``Authorization: Bearer <jwt>``, decode ``org_uuid``
             from claims, enforce DYN_AUTH_VALID_ORGS allowlist if set.
             Return 401 on any failure.

  2. Forward to the internal Dynamo Rust HTTP service.

  3. Usage metering  (only when DYN_FLEXPRICE_ENABLED=true)
             — extract token usage from the response (SSE stream or JSON),
             fire-and-forget enqueue to FlexPrice (zero request latency impact).
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, Optional

from aiohttp import ClientSession, ClientTimeout, TCPConnector, web

from .auth import AuthError, authenticate
from .client import FlexPriceClient
from .config import FlexPriceConfig

logger = logging.getLogger(__name__)

# Endpoints where token usage should be captured and metered
_BILLED_PATHS = frozenset(
    ["/v1/chat/completions", "/v1/completions", "/v1/embeddings"]
)

# Hop-by-hop headers that must not be forwarded
_HOP_BY_HOP = frozenset(
    [
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    ]
)

_JSON_CT = "application/json"


def _json_error(status: int, message: str) -> web.Response:
    return web.Response(
        status=status,
        content_type=_JSON_CT,
        text=json.dumps({"statusCode": status, "message": message}),
    )


class DynamoProxy:
    """Lightweight aiohttp reverse proxy providing JWT auth and optional FlexPrice metering.

    FlexPrice usage events are enqueued fire-and-forget after the response is
    written — billing adds zero latency to the request path.
    """

    def __init__(
        self,
        backend_url: str,
        config: FlexPriceConfig,
        flexprice_client: Optional[FlexPriceClient] = None,
        model_name: str = "",
    ) -> None:
        self._backend = backend_url.rstrip("/")
        self._config = config
        self._client = flexprice_client  # None when DYN_FLEXPRICE_ENABLED=false
        self._model_name = model_name
        self._session: Optional[ClientSession] = None

    async def start(self) -> None:
        self._session = ClientSession(
            connector=TCPConnector(ssl=False, limit=1000),
            # No total timeout — requests may stream for an extended period.
            timeout=ClientTimeout(total=None, connect=10),
        )

    async def stop(self) -> None:
        if self._session:
            await self._session.close()

    # ------------------------------------------------------------------
    # Main handler
    # ------------------------------------------------------------------

    async def handle(self, request: web.Request) -> web.StreamResponse:
        # ---- 1. Authentication (always required) ----------------------
        auth_header = request.headers.get("Authorization", "")
        try:
            auth_ctx = authenticate(
                auth_header,
                self._config.auth_secret_keys,
                self._config.auth_valid_orgs or None,
            )
        except AuthError as exc:
            logger.warning("Auth failed: %s", exc)
            return _json_error(exc.status, str(exc))

        org_id = auth_ctx.org_uuid

        # ---- 2. Forward to Dynamo Rust service ------------------------
        path = request.path
        qs = request.query_string
        url = f"{self._backend}{path}{'?' + qs if qs else ''}"

        # Only capture usage when FlexPrice metering is enabled
        is_metered = self._client is not None and path in _BILLED_PATHS
        body = await request.read()

        model_name = self._model_name
        is_streaming_req = False
        if is_metered and body:
            try:
                req_json = json.loads(body)
                model_name = req_json.get("model") or model_name
                is_streaming_req = bool(req_json.get("stream", False))
            except Exception:
                pass

        fwd_headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in _HOP_BY_HOP and k.lower() != "host"
        }

        start = time.monotonic()

        try:
            async with self._session.request(  # type: ignore[union-attr]
                method=request.method,
                url=url,
                headers=fwd_headers,
                data=body,
                allow_redirects=False,
            ) as backend_resp:
                resp_headers = {
                    k: v
                    for k, v in backend_resp.headers.items()
                    if k.lower()
                    not in (_HOP_BY_HOP | {"content-encoding", "content-length"})
                }
                is_sse = "text/event-stream" in backend_resp.headers.get(
                    "content-type", ""
                )

                if is_sse or is_streaming_req:
                    return await self._handle_stream(
                        request,
                        backend_resp,
                        resp_headers,
                        is_metered=is_metered,
                        org_id=org_id,
                        model_name=model_name,
                        start=start,
                    )
                else:
                    return await self._handle_buffered(
                        backend_resp,
                        resp_headers,
                        is_metered=is_metered,
                        org_id=org_id,
                        model_name=model_name,
                        start=start,
                    )
        except Exception as exc:
            logger.warning("Proxy error on %s: %s", path, exc)
            return _json_error(502, "Bad Gateway")

    # ------------------------------------------------------------------
    # Streaming response
    # ------------------------------------------------------------------

    async def _handle_stream(
        self,
        request: web.Request,
        backend_resp: Any,
        resp_headers: Dict[str, str],
        *,
        is_metered: bool,
        org_id: str,
        model_name: str,
        start: float,
    ) -> web.StreamResponse:
        response = web.StreamResponse(
            status=backend_resp.status, headers=resp_headers
        )
        await response.prepare(request)

        usage: Optional[Dict[str, Any]] = None
        buf = b""

        async for chunk in backend_resp.content.iter_any():
            await response.write(chunk)
            if is_metered:
                buf += chunk
                while b"\n" in buf:
                    line_bytes, buf = buf.split(b"\n", 1)
                    line = line_bytes.decode("utf-8", errors="ignore").rstrip("\r")
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str and data_str != "[DONE]":
                            usage = _parse_usage_from_sse(data_str, usage)

        await response.write_eof()

        # Fire-and-forget after response is fully written to the client
        if is_metered and usage:
            self._emit_usage(
                org_id=org_id,
                model_name=model_name,
                usage=usage,
                elapsed=time.monotonic() - start,
                streaming=True,
            )

        return response

    # ------------------------------------------------------------------
    # Buffered (non-streaming) response
    # ------------------------------------------------------------------

    async def _handle_buffered(
        self,
        backend_resp: Any,
        resp_headers: Dict[str, str],
        *,
        is_metered: bool,
        org_id: str,
        model_name: str,
        start: float,
    ) -> web.Response:
        body_bytes = await backend_resp.read()

        response = web.Response(
            status=backend_resp.status,
            headers=resp_headers,
            body=body_bytes,
        )

        # Fire-and-forget after response body is ready
        if is_metered:
            usage = _extract_usage_from_json(body_bytes)
            if usage:
                self._emit_usage(
                    org_id=org_id,
                    model_name=model_name,
                    usage=usage,
                    elapsed=time.monotonic() - start,
                    streaming=False,
                )

        return response

    # ------------------------------------------------------------------
    # Usage emission (enqueued — never blocks the request path)
    # ------------------------------------------------------------------

    def _emit_usage(
        self,
        *,
        org_id: str,
        model_name: str,
        usage: Dict[str, Any],
        elapsed: float,
        streaming: bool,
    ) -> None:
        assert self._client is not None
        event_name = self._config.resolve_event_name(model_name)
        source = self._config.resolve_source_name(model_name)

        properties: Dict[str, Any] = {
            "model_id": model_name,
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "time_taken": round(elapsed, 4),
            "streaming": streaming,
            "status": "success",
        }

        self._client.enqueue(
            event_name=event_name,
            external_customer_id=org_id,
            properties=properties,
            source=source,
        )
        logger.debug(
            "FlexPrice usage enqueued: org=%s model=%s in=%s out=%s",
            org_id,
            model_name,
            properties["input_tokens"],
            properties["output_tokens"],
        )


# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------


def _extract_usage_from_json(body: bytes) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(body).get("usage")
    except Exception:
        return None


def _parse_usage_from_sse(
    data_str: str, current: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Merge ``usage`` from an SSE data payload into *current*."""
    try:
        obj = json.loads(data_str)
        usage = obj.get("usage")
        if usage and isinstance(usage, dict):
            if current:
                merged = dict(current)
                for k, v in usage.items():
                    if isinstance(v, (int, float)):
                        merged[k] = merged.get(k, 0) + v
                    else:
                        merged[k] = v
                return merged
            return usage
    except Exception:
        pass
    return current


# ------------------------------------------------------------------
# Server entrypoint
# ------------------------------------------------------------------


async def run_proxy(
    host: str,
    port: int,
    backend_url: str,
    config: FlexPriceConfig,
    model_name: str = "",
) -> None:
    """Start the Dynamo auth proxy and block until cancelled.

    A FlexPriceClient is created only when DYN_FLEXPRICE_ENABLED=true so that
    auth-only mode has zero FlexPrice overhead.
    """
    flexprice_client: Optional[FlexPriceClient] = None
    if config.enabled:
        flexprice_client = FlexPriceClient(
            api_host=config.api_host,
            api_key=config.api_key,
        )
        await flexprice_client.start()

    proxy = DynamoProxy(
        backend_url=backend_url,
        config=config,
        flexprice_client=flexprice_client,
        model_name=model_name,
    )
    await proxy.start()

    app = web.Application()
    app.router.add_route("*", "/{path_info:.*}", proxy.handle)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host=host, port=port)
    await site.start()

    logger.info(
        "Dynamo proxy listening on %s:%d → %s  (auth=%s flexprice=%s)",
        host,
        port,
        backend_url,
        config.auth_enabled,
        config.enabled,
    )

    try:
        await asyncio.Future()  # run forever
    except asyncio.CancelledError:
        pass
    finally:
        await runner.cleanup()
        await proxy.stop()
        if flexprice_client:
            await flexprice_client.stop()
