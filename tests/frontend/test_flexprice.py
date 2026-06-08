"""Unit tests for the flexprice auth + billing module.

Covers:
  - auth.py      : JWT validation, key rotation, expiry, allowlist
  - config.py    : env var parsing, validate() error paths
  - client.py    : FlexPriceClient async event emission
  - proxy.py     : DynamoProxy end-to-end with a mock backend

No external services (Dynamo, FlexPrice) are required — all network calls
are intercepted with aiohttp test utilities or asyncio mocks.
"""

import asyncio
import base64
import hashlib
import hmac
import json
import time
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from dynamo.frontend.flexprice.auth import AuthCtx, AuthError, authenticate
from dynamo.frontend.flexprice.client import FlexPriceClient
from dynamo.frontend.flexprice.config import FlexPriceConfig
from dynamo.frontend.flexprice.proxy import (
    DynamoProxy,
    _extract_usage_from_json,
    _parse_usage_from_sse,
)

# Helpers

_SECRET = "test-secret-key"
_ORG_UUID = "org-abc-123"
_USER_UUID = "user-xyz-456"


def _make_jwt(
    org_uuid: str = _ORG_UUID,
    user_uuid: str = _USER_UUID,
    exp_offset: int = 3600,
    secret: str = _SECRET,
    alg: str = "HS256",
) -> str:
    """Build a minimal signed JWT for testing."""
    hash_fns = {"HS256": hashlib.sha256, "HS384": hashlib.sha384, "HS512": hashlib.sha512}

    def b64(data: Any) -> str:
        return base64.urlsafe_b64encode(json.dumps(data).encode()).rstrip(b"=").decode()

    header = b64({"alg": alg, "typ": "JWT"})
    payload = b64(
        {"org_uuid": org_uuid, "uuid": user_uuid, "exp": int(time.time()) + exp_offset}
    )
    sig_input = f"{header}.{payload}".encode()
    sig = hmac.new(secret.encode(), sig_input, hash_fns[alg]).digest()
    sig_b64 = base64.urlsafe_b64encode(sig).rstrip(b"=").decode()
    return f"{header}.{payload}.{sig_b64}"


def _make_config(
    auth_enabled: bool = True,
    flexprice_enabled: bool = False,
    secret: str = _SECRET,
    valid_orgs: Optional[list] = None,
) -> FlexPriceConfig:
    return FlexPriceConfig(
        auth_enabled=auth_enabled,
        auth_secret_keys=[secret],
        auth_valid_orgs=valid_orgs or [],
        enabled=flexprice_enabled,
        api_key="fp-key" if flexprice_enabled else "",
        api_host="api.flexprice.io" if flexprice_enabled else "",
        event_name="",
        source_name="",
        internal_port_offset=1,
    )


# auth.py tests


class TestAuthenticate:
    def test_valid_token(self):
        token = _make_jwt()
        ctx = authenticate(f"Bearer {token}", [_SECRET])
        assert ctx.org_uuid == _ORG_UUID
        assert ctx.user_uuid == _USER_UUID

    def test_missing_header(self):
        with pytest.raises(AuthError) as exc:
            authenticate("", [_SECRET])
        assert exc.value.status == 401

    def test_wrong_scheme(self):
        with pytest.raises(AuthError):
            authenticate("Basic dXNlcjpwYXNz", [_SECRET])

    def test_wrong_secret(self):
        token = _make_jwt(secret="other-secret")
        with pytest.raises(AuthError) as exc:
            authenticate(f"Bearer {token}", [_SECRET])
        assert "signature" in str(exc.value).lower()

    def test_key_rotation(self):
        """Second key in the list should still validate."""
        token = _make_jwt(secret="new-secret")
        ctx = authenticate(f"Bearer {token}", ["old-secret", "new-secret"])
        assert ctx.org_uuid == _ORG_UUID

    def test_expired_token(self):
        token = _make_jwt(exp_offset=-10)
        with pytest.raises(AuthError) as exc:
            authenticate(f"Bearer {token}", [_SECRET])
        assert "expired" in str(exc.value).lower()

    def test_missing_org_uuid_claim(self):
        def _b64(d):
            return base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()

        header = _b64({"alg": "HS256", "typ": "JWT"})
        payload = _b64({"uuid": _USER_UUID, "exp": int(time.time()) + 3600})
        sig_input = f"{header}.{payload}".encode()
        sig = hmac.new(_SECRET.encode(), sig_input, hashlib.sha256).digest()
        sig_b64 = base64.urlsafe_b64encode(sig).rstrip(b"=").decode()
        token = f"{header}.{payload}.{sig_b64}"

        with pytest.raises(AuthError) as exc:
            authenticate(f"Bearer {token}", [_SECRET])
        assert "org_uuid" in str(exc.value)

    def test_org_allowlist_passes(self):
        token = _make_jwt()
        ctx = authenticate(f"Bearer {token}", [_SECRET], valid_orgs=[_ORG_UUID])
        assert ctx.org_uuid == _ORG_UUID

    def test_org_allowlist_blocks(self):
        token = _make_jwt()
        with pytest.raises(AuthError) as exc:
            authenticate(f"Bearer {token}", [_SECRET], valid_orgs=["other-org"])
        assert exc.value.status == 401

    def test_hs512_algorithm(self):
        token = _make_jwt(alg="HS512")
        ctx = authenticate(f"Bearer {token}", [_SECRET])
        assert ctx.org_uuid == _ORG_UUID

    def test_unsupported_algorithm(self):
        def _b64(d):
            return base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()

        header = _b64({"alg": "RS256", "typ": "JWT"})
        payload = _b64({"org_uuid": _ORG_UUID, "uuid": _USER_UUID})
        token = f"{header}.{payload}.fakesig"
        with pytest.raises(AuthError) as exc:
            authenticate(f"Bearer {token}", [_SECRET])
        assert "unsupported" in str(exc.value).lower()


# config.py tests


class TestFlexPriceConfig:
    def test_from_env_defaults(self, monkeypatch):
        for key in [
            "DYN_AUTH_ENABLED", "DYN_FLEXPRICE_ENABLED", "DYN_AUTH_SECRET_KEY",
            "DYN_AUTH_VALID_ORGS", "DYN_FLEXPRICE_API_KEY", "DYN_FLEXPRICE_API_HOST",
        ]:
            monkeypatch.delenv(key, raising=False)
        cfg = FlexPriceConfig.from_env()
        assert cfg.auth_enabled is False
        assert cfg.enabled is False
        assert cfg.auth_secret_keys == []

    def test_from_env_auth_only(self, monkeypatch):
        monkeypatch.setenv("DYN_AUTH_ENABLED", "true")
        monkeypatch.setenv("DYN_AUTH_SECRET_KEY", "s1,s2")
        monkeypatch.delenv("DYN_FLEXPRICE_ENABLED", raising=False)
        cfg = FlexPriceConfig.from_env()
        assert cfg.auth_enabled is True
        assert cfg.auth_secret_keys == ["s1", "s2"]
        assert cfg.enabled is False

    def test_from_env_flexprice_full(self, monkeypatch):
        monkeypatch.setenv("DYN_AUTH_ENABLED", "true")
        monkeypatch.setenv("DYN_AUTH_SECRET_KEY", "secret")
        monkeypatch.setenv("DYN_FLEXPRICE_ENABLED", "true")
        monkeypatch.setenv("DYN_FLEXPRICE_API_KEY", "fp-key")
        monkeypatch.setenv("DYN_FLEXPRICE_API_HOST", "api.flexprice.io")
        cfg = FlexPriceConfig.from_env()
        assert cfg.enabled is True
        assert cfg.api_host == "api.flexprice.io"

    def test_validate_flexprice_without_auth_raises(self):
        cfg = _make_config(auth_enabled=False, flexprice_enabled=True)
        cfg.auth_secret_keys = []
        with pytest.raises(ValueError, match="DYN_AUTH_ENABLED"):
            cfg.validate()

    def test_validate_auth_without_secret_raises(self):
        cfg = _make_config(auth_enabled=True)
        cfg.auth_secret_keys = []
        with pytest.raises(ValueError, match="DYN_AUTH_SECRET_KEY"):
            cfg.validate()

    def test_validate_flexprice_missing_api_key_raises(self):
        cfg = _make_config(auth_enabled=True, flexprice_enabled=True)
        cfg.api_key = ""
        with pytest.raises(ValueError, match="DYN_FLEXPRICE_API_KEY"):
            cfg.validate()

    def test_proxy_required_reflects_auth(self):
        assert _make_config(auth_enabled=True).proxy_required is True
        assert _make_config(auth_enabled=False).proxy_required is False

    def test_resolve_event_name_default(self):
        cfg = _make_config()
        assert cfg.resolve_event_name("llama3") == "llama3-llm-usage"
        assert cfg.resolve_event_name("") == "dynamo-llm-usage"

    def test_resolve_event_name_override(self):
        cfg = _make_config()
        cfg.event_name = "my-event"
        assert cfg.resolve_event_name("any-model") == "my-event"


# client.py tests


class TestFlexPriceClient:
    async def test_enqueue_and_send(self):
        """Events enqueued must be POSTed to the FlexPrice events endpoint."""
        sent: list[Dict] = []

        def fake_post(_url, json=None, **_kwargs):
            sent.append(json)
            resp = MagicMock()
            resp.status = 200
            resp.__aenter__ = AsyncMock(return_value=resp)
            resp.__aexit__ = AsyncMock(return_value=False)
            return resp

        client = FlexPriceClient(api_host="api.flexprice.io", api_key="key")
        await client.start()
        client._session.post = fake_post  # type: ignore[union-attr]

        client.enqueue(
            event_name="llama3-llm-usage",
            external_customer_id=_ORG_UUID,
            properties={"input_tokens": 10, "output_tokens": 20},
            source="llama3",
        )

        # Give the worker time to drain the queue
        await asyncio.sleep(0.05)
        await client.stop()

        assert len(sent) == 1
        assert sent[0]["event_name"] == "llama3-llm-usage"
        assert sent[0]["external_customer_id"] == _ORG_UUID
        assert sent[0]["properties"]["input_tokens"] == "10"  # values stringified

    async def test_queue_full_drops_silently(self):
        client = FlexPriceClient(api_host="api.flexprice.io", api_key="key")
        # Fill the queue manually without starting the worker
        for _ in range(client._queue.maxsize):
            client._queue.put_nowait({"dummy": True})

        # This must not raise
        client.enqueue("ev", _ORG_UUID, {})

    async def test_stop_drains_queue(self):
        """stop() must flush remaining events before closing."""
        sent: list = []

        def fake_post(_url, json=None, **_kwargs):
            sent.append(json)
            resp = MagicMock()
            resp.status = 200
            resp.__aenter__ = AsyncMock(return_value=resp)
            resp.__aexit__ = AsyncMock(return_value=False)
            return resp

        client = FlexPriceClient(api_host="api.flexprice.io", api_key="key")
        await client.start()
        client._session.post = fake_post  # type: ignore[union-attr]

        for i in range(5):
            client.enqueue(f"event-{i}", _ORG_UUID, {})

        await client.stop()
        assert len(sent) == 5


# Utility function tests


class TestUtilityFunctions:
    def test_extract_usage_from_json(self):
        body = json.dumps({
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        }).encode()
        usage = _extract_usage_from_json(body)
        assert usage == {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}

    def test_extract_usage_from_json_missing(self):
        assert _extract_usage_from_json(b'{"choices": []}') is None

    def test_extract_usage_from_json_invalid(self):
        assert _extract_usage_from_json(b"not json") is None

    def test_parse_usage_from_sse_first_chunk(self):
        data = json.dumps({"usage": {"prompt_tokens": 5, "completion_tokens": 10}})
        result = _parse_usage_from_sse(data, None)
        assert result == {"prompt_tokens": 5, "completion_tokens": 10}

    def test_parse_usage_from_sse_merges(self):
        current = {"prompt_tokens": 5, "completion_tokens": 10}
        data = json.dumps({"usage": {"completion_tokens": 5}})
        result = _parse_usage_from_sse(data, current)
        assert result["completion_tokens"] == 15  # merged/summed
        assert result["prompt_tokens"] == 5

    def test_parse_usage_from_sse_no_usage(self):
        data = json.dumps({"choices": []})
        result = _parse_usage_from_sse(data, {"prompt_tokens": 1})
        assert result == {"prompt_tokens": 1}  # unchanged

    def test_parse_usage_from_sse_invalid_json(self):
        result = _parse_usage_from_sse("not-json", {"prompt_tokens": 1})
        assert result == {"prompt_tokens": 1}  # unchanged


# proxy.py tests (DynamoProxy with a mock aiohttp backend)


async def _make_mock_backend(response_body: dict, status: int = 200) -> TestServer:
    """Start a tiny aiohttp server that returns a fixed JSON response."""

    async def handler(request: web.Request) -> web.Response:
        return web.Response(
            status=status,
            content_type="application/json",
            text=json.dumps(response_body),
        )

    app = web.Application()
    app.router.add_route("*", "/{path_info:.*}", handler)
    server = TestServer(app)
    await server.start_server()
    return server


async def _make_sse_backend(chunks: list[str]) -> TestServer:
    """Start a backend that streams SSE chunks."""

    async def handler(request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse()
        resp.content_type = "text/event-stream"
        await resp.prepare(request)
        for chunk in chunks:
            await resp.write(chunk.encode())
        await resp.write_eof()
        return resp

    app = web.Application()
    app.router.add_route("*", "/{path_info:.*}", handler)
    server = TestServer(app)
    await server.start_server()
    return server


class TestDynamoProxy:
    async def _make_proxy_client(
        self,
        backend_server: TestServer,
        config: Optional[FlexPriceConfig] = None,
        flexprice_client: Optional[FlexPriceClient] = None,
    ) -> TestClient:
        if config is None:
            config = _make_config()
        proxy = DynamoProxy(
            backend_url=str(backend_server.make_url("/")),
            config=config,
            flexprice_client=flexprice_client,
        )
        await proxy.start()

        app = web.Application()
        app.router.add_route("*", "/{path_info:.*}", proxy.handle)
        client = TestClient(TestServer(app))
        await client.start_server()
        return client

    async def test_valid_token_forwards_request(self):
        backend = await _make_mock_backend({"choices": [], "usage": None})
        try:
            client = await self._make_proxy_client(backend)
            try:
                token = _make_jwt()
                resp = await client.get("/v1/models", headers={"Authorization": f"Bearer {token}"})
                assert resp.status == 200
            finally:
                await client.close()
        finally:
            await backend.close()

    async def test_missing_token_returns_401(self):
        backend = await _make_mock_backend({})
        try:
            client = await self._make_proxy_client(backend)
            try:
                resp = await client.get("/v1/models")
                assert resp.status == 401
                body = await resp.json()
                assert body["statusCode"] == 401
            finally:
                await client.close()
        finally:
            await backend.close()

    async def test_wrong_token_returns_401(self):
        backend = await _make_mock_backend({})
        try:
            client = await self._make_proxy_client(backend)
            try:
                resp = await client.get(
                    "/v1/models", headers={"Authorization": "Bearer bad.token.sig"}
                )
                assert resp.status == 401
            finally:
                await client.close()
        finally:
            await backend.close()

    async def test_expired_token_returns_401(self):
        backend = await _make_mock_backend({})
        try:
            client = await self._make_proxy_client(backend)
            try:
                token = _make_jwt(exp_offset=-60)
                resp = await client.get(
                    "/v1/models", headers={"Authorization": f"Bearer {token}"}
                )
                assert resp.status == 401
            finally:
                await client.close()
        finally:
            await backend.close()

    async def test_org_allowlist_blocks(self):
        backend = await _make_mock_backend({})
        try:
            cfg = _make_config(valid_orgs=["other-org"])
            client = await self._make_proxy_client(backend, config=cfg)
            try:
                token = _make_jwt()
                resp = await client.get(
                    "/v1/models", headers={"Authorization": f"Bearer {token}"}
                )
                assert resp.status == 401
            finally:
                await client.close()
        finally:
            await backend.close()

    async def test_usage_emitted_for_billed_path(self):
        """FlexPrice client must receive an event after a /v1/chat/completions call."""
        usage_body = {
            "choices": [{"message": {"content": "hi"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
        backend = await _make_mock_backend(usage_body)
        try:
            enqueued: list = []

            fp_client = MagicMock(spec=FlexPriceClient)
            fp_client.enqueue = lambda **kwargs: enqueued.append(kwargs)

            cfg = _make_config(flexprice_enabled=True)
            cfg.api_key = "fp-key"
            cfg.api_host = "api.flexprice.io"
            proxy_client = await self._make_proxy_client(
                backend, config=cfg, flexprice_client=fp_client
            )
            try:
                token = _make_jwt()
                body = json.dumps({"model": "llama3", "messages": []})
                resp = await proxy_client.post(
                    "/v1/chat/completions",
                    data=body,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                )
                assert resp.status == 200
                assert len(enqueued) == 1
                assert enqueued[0]["external_customer_id"] == _ORG_UUID
                assert enqueued[0]["properties"]["input_tokens"] == 10
            finally:
                await proxy_client.close()
        finally:
            await backend.close()

    async def test_no_emission_for_non_billed_path(self):
        """GET /v1/models must not trigger FlexPrice emission."""
        backend = await _make_mock_backend({"data": []})
        try:
            enqueued: list = []
            fp_client = MagicMock(spec=FlexPriceClient)
            fp_client.enqueue = lambda **kwargs: enqueued.append(kwargs)

            cfg = _make_config(flexprice_enabled=True)
            proxy_client = await self._make_proxy_client(
                backend, config=cfg, flexprice_client=fp_client
            )
            try:
                token = _make_jwt()
                resp = await proxy_client.get(
                    "/v1/models", headers={"Authorization": f"Bearer {token}"}
                )
                assert resp.status == 200
                assert enqueued == []
            finally:
                await proxy_client.close()
        finally:
            await backend.close()

    async def test_no_emission_when_flexprice_disabled(self):
        """When FlexPrice is disabled (no client), no events are emitted."""
        usage_body = {
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}
        }
        backend = await _make_mock_backend(usage_body)
        try:
            # No flexprice_client — auth-only mode
            cfg = _make_config(flexprice_enabled=False)
            proxy_client = await self._make_proxy_client(backend, config=cfg)
            try:
                token = _make_jwt()
                resp = await proxy_client.post(
                    "/v1/chat/completions",
                    data=json.dumps({"model": "m", "messages": []}),
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                )
                assert resp.status == 200  # still forwards fine
            finally:
                await proxy_client.close()
        finally:
            await backend.close()

    async def test_sse_stream_usage_extracted(self):
        """Usage from the last SSE chunk must be emitted after stream ends."""
        chunks = [
            'data: {"choices":[{"delta":{"content":"hello"}}],"usage":null}\n\n',
            'data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":10,"total_tokens":15}}\n\n',
            "data: [DONE]\n\n",
        ]
        backend = await _make_sse_backend(chunks)
        try:
            enqueued: list = []
            fp_client = MagicMock(spec=FlexPriceClient)
            fp_client.enqueue = lambda **kwargs: enqueued.append(kwargs)

            cfg = _make_config(flexprice_enabled=True)
            proxy_client = await self._make_proxy_client(
                backend, config=cfg, flexprice_client=fp_client
            )
            try:
                token = _make_jwt()
                body = json.dumps({"model": "llama3", "messages": [], "stream": True})
                resp = await proxy_client.post(
                    "/v1/chat/completions",
                    data=body,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                )
                assert resp.status == 200
                await resp.read()  # consume the stream
                assert len(enqueued) == 1
                assert enqueued[0]["properties"]["input_tokens"] == 5
                assert enqueued[0]["properties"]["output_tokens"] == 10
            finally:
                await proxy_client.close()
        finally:
            await backend.close()
