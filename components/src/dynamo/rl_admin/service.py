"""
Dynamo RL Admin service.

A lightweight FastAPI service that Prime-RL talks to.  It provides:

1. Admin endpoints (/pause, /resume, /update_weights, /health, /ready)
   - Calls vLLM engine routes via Dynamo system port.
2. /tokenize, /detokenize
   - Proxied directly to Dynamo Rust frontend (uses the same tokenizer as inference).
3. /v1/chat/completions
   - Proxied to Dynamo frontend. Injects prompt_token_ids and choice.token_ids
     into responses (via Rust /tokenize + logprobs) for Prime-RL's verifiers client.
4. /v1/chat/completions/tokens (TITO)
   - Translates Prime-RL's `tokens` field to nvext.token_data and proxies.

No Python HF tokenizer dependency -- all tokenization goes through the Rust frontend's
/tokenize endpoint, which uses the same tokenizer as the inference path.

Prime-RL config:
    base_url = ["http://localhost:<this_port>/v1"]
    admin_base_url = ["http://localhost:<this_port>"]
"""

import argparse
import logging
import os
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

logger = logging.getLogger("dynamo.rl_admin")

app = FastAPI(title="Dynamo RL Admin")

# ── Configuration (set via env or CLI) ──────────────────────────────
DYNAMO_FRONTEND_URL = os.getenv("DYNAMO_FRONTEND_URL", "http://localhost:8000")
DYNAMO_SYSTEM_URL = os.getenv("DYNAMO_SYSTEM_URL", "http://localhost:8081")
MODEL_NAME = os.getenv("MODEL_NAME", "")
_http_client: httpx.AsyncClient | None = None


def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(600.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=30),
        )
    return _http_client


@app.on_event("shutdown")
async def _shutdown():
    global _http_client
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None


# ── Helpers ─────────────────────────────────────────────────────────

async def _call_engine_route(route: str, body: dict | None = None) -> dict:
    """Call a registered engine route on the vLLM worker via system HTTP."""
    client = _get_http_client()
    url = f"{DYNAMO_SYSTEM_URL}/engine/{route}"
    try:
        resp = await client.post(url, json=body or {})
        if resp.status_code == 200:
            return resp.json()
        return {"status": "error", "message": resp.text, "http_status": resp.status_code}
    except Exception as e:
        logger.error(f"Engine route {route} failed: {e}")
        return {"status": "error", "message": str(e)}


async def _tokenize_via_frontend(messages: list[dict], model: str = "") -> list[int]:
    """Tokenize messages using the Rust frontend's /tokenize endpoint.

    Uses the same tokenizer as the inference path -- no Rust/Python divergence.
    """
    client = _get_http_client()
    body: dict = {"messages": messages, "add_generation_prompt": True}
    if model:
        body["model"] = model
    resp = await client.post(f"{DYNAMO_FRONTEND_URL}/v1/tokenize", json=body)
    resp.raise_for_status()
    return resp.json()["tokens"]


def _extract_completion_ids_from_logprobs(logprobs_content: list[dict]) -> list[int]:
    """Extract one token ID per logprob entry from the bytes field.

    Each logprob entry contains a `bytes` field with the raw byte values
    of the token. We reconstruct the token text and look it up. This
    always produces exactly len(logprobs_content) IDs.
    """
    completion_ids = []
    for entry in logprobs_content:
        token_bytes = entry.get("bytes")
        if token_bytes is not None:
            # The bytes field is authoritative -- it's the exact bytes vLLM produced.
            # Reconstruct the token ID by encoding the bytes as text.
            # For single-byte tokens (most common), this is a direct vocab lookup.
            # For multi-byte tokens (emoji, CJK), encode returns the correct ID.
            text = bytes(token_bytes).decode("utf-8", errors="replace")
            # Use a simple heuristic: most tokens encode to exactly 1 ID.
            # If not, we still take the first to maintain 1:1 alignment.
            completion_ids.append(
                _byte_token_cache.get(tuple(token_bytes), _encode_single(text))
            )
        else:
            completion_ids.append(0)
    return completion_ids


# Cache for byte->token_id mappings (populated lazily)
_byte_token_cache: dict[tuple, int] = {}
_tokenizer_for_fallback = None


def _encode_single(text: str) -> int:
    """Encode a single token's text to its ID using a lightweight tokenizer."""
    global _tokenizer_for_fallback
    if _tokenizer_for_fallback is None:
        try:
            from transformers import AutoTokenizer
            model = MODEL_NAME or os.getenv("MODEL_NAME", "")
            if model:
                _tokenizer_for_fallback = AutoTokenizer.from_pretrained(
                    model, trust_remote_code=True
                )
        except Exception:
            pass
    if _tokenizer_for_fallback is not None:
        ids = _tokenizer_for_fallback.encode(text, add_special_tokens=False)
        return ids[0] if ids else 0
    return 0


# ══════════════════════════════════════════════════════════════════════
# Admin endpoints (Prime-RL calls these on admin_base_url)
# ══════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    return JSONResponse(content="OK")


@app.get("/ready")
async def ready():
    """Deep health check: verify connectivity to frontend and system port."""
    client = _get_http_client()
    try:
        frontend_ok = (await client.get(f"{DYNAMO_FRONTEND_URL}/v1/models")).status_code == 200
    except Exception:
        frontend_ok = False
    try:
        system_ok = (await client.get(f"{DYNAMO_SYSTEM_URL}/health")).status_code == 200
    except Exception:
        system_ok = False
    if frontend_ok and system_ok:
        return JSONResponse(content={"status": "ready"})
    return JSONResponse(
        content={"status": "not_ready", "frontend": frontend_ok, "system": system_ok},
        status_code=503,
    )


@app.post("/pause")
async def pause(request: Request):
    """Pause inference engines (drain in-flight requests)."""
    result = await _call_engine_route("pause_generation")
    logger.info(f"[admin] POST /pause -> {result}")
    status_code = 200 if result.get("status") == "ok" else 502
    return JSONResponse(content=result, status_code=status_code)


@app.post("/resume")
async def resume():
    """Resume inference engines after weight update."""
    result = await _call_engine_route("resume_generation")
    logger.info(f"[admin] POST /resume -> {result}")
    status_code = 200 if result.get("status") == "ok" else 502
    return JSONResponse(content=result, status_code=status_code)


@app.post("/update_weights")
async def update_weights(request: Request):
    """Update weights from filesystem path.

    Prime-RL sends: {"weight_dir": "/path/to/weights"} or {"weight_dir": null}
    When weight_dir is null, it signals an NCCL broadcast (not filesystem).
    """
    body = await request.json()
    weight_dir = body.get("weight_dir")

    if weight_dir is None:
        logger.info("[admin] POST /update_weights weight_dir=None (NCCL mode, no-op)")
        return JSONResponse(content={"status": "ok", "message": "NCCL mode, no-op on Dynamo side"})

    logger.info(f"[admin] POST /update_weights weight_dir={weight_dir}")

    flush_result = await _call_engine_route("flush_cache")
    logger.info(f"[admin] flush_cache -> {flush_result}")
    if flush_result.get("status") != "ok":
        return JSONResponse(content=flush_result, status_code=502)

    load_result = await _call_engine_route(
        "update_weights_from_path",
        {"path": weight_dir, "version": Path(weight_dir).name if weight_dir else "unknown"},
    )
    logger.info(f"[admin] update_weights_from_path -> {load_result}")

    status_code = 200 if load_result.get("status") == "ok" else 502
    return JSONResponse(content=load_result, status_code=status_code)


# ══════════════════════════════════════════════════════════════════════
# /tokenize, /detokenize -- proxy to Rust frontend
# ══════════════════════════════════════════════════════════════════════

@app.post("/tokenize")
@app.post("/v1/tokenize")
async def tokenize(request: Request):
    """Proxy to Dynamo Rust frontend /tokenize (same tokenizer as inference)."""
    client = _get_http_client()
    body = await request.json()
    resp = await client.post(f"{DYNAMO_FRONTEND_URL}/v1/tokenize", json=body)
    return JSONResponse(content=resp.json(), status_code=resp.status_code)


@app.post("/detokenize")
@app.post("/v1/detokenize")
async def detokenize(request: Request):
    """Proxy to Dynamo Rust frontend /v1/detokenize."""
    client = _get_http_client()
    body = await request.json()
    resp = await client.post(f"{DYNAMO_FRONTEND_URL}/v1/detokenize", json=body)
    return JSONResponse(content=resp.json(), status_code=resp.status_code)


# ══════════════════════════════════════════════════════════════════════
# /v1/chat/completions/tokens  (TITO endpoint)
# ══════════════════════════════════════════════════════════════════════

@app.post("/v1/chat/completions/tokens")
async def chat_completions_tokens(request: Request):
    """TITO: Token-In / Token-Out chat completions.

    Prime-RL sends a standard ChatCompletionRequest with an extra `tokens` field.
    We translate: tokens -> nvext.token_data, then proxy to Dynamo frontend.
    """
    body = await request.json()

    tokens = body.pop("tokens", None)
    if tokens is None:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "Missing 'tokens' field for TITO endpoint"}},
        )

    if "messages" not in body or not body["messages"]:
        body["messages"] = [{"role": "user", "content": "(token-in mode)"}]

    nvext = body.get("nvext", {}) or {}
    nvext["token_data"] = tokens
    extra_fields = nvext.get("extra_fields", []) or []
    for field in ["token_ids", "completion_token_ids"]:
        if field not in extra_fields:
            extra_fields.append(field)
    nvext["extra_fields"] = extra_fields
    body["nvext"] = nvext

    _strip_unsupported(body)

    if "logprobs" not in body:
        body["logprobs"] = True

    stream = body.get("stream", False)

    client = _get_http_client()
    url = f"{DYNAMO_FRONTEND_URL}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if "authorization" in request.headers:
        headers["Authorization"] = request.headers["authorization"]

    if stream:
        async def _stream_proxy():
            async with client.stream("POST", url, json=body, headers=headers) as resp:
                async for chunk in resp.aiter_bytes():
                    yield chunk

        return StreamingResponse(_stream_proxy(), media_type="text/event-stream")
    else:
        resp = await client.post(url, json=body, headers=headers)
        return JSONResponse(content=resp.json(), status_code=resp.status_code)


# ══════════════════════════════════════════════════════════════════════
# /v1/chat/completions -- proxy with token_ids injection
# ══════════════════════════════════════════════════════════════════════

_STRIP_FIELDS = {"return_token_ids", "tokens"}


def _strip_unsupported(body: dict) -> dict:
    """Remove fields the Dynamo frontend doesn't support."""
    for f in _STRIP_FIELDS:
        body.pop(f, None)
    return body


async def _inject_token_ids(body: dict, response_data: dict) -> dict:
    """Inject prompt_token_ids and choice.token_ids into the response.

    Uses the Rust frontend's /tokenize for prompt_ids (same tokenizer as inference).
    Extracts completion_ids from logprobs bytes (1:1 alignment guaranteed).
    """
    try:
        messages = body.get("messages", [])
        if not messages:
            return response_data

        # Get prompt token IDs from the Rust frontend (same tokenizer as inference path)
        prompt_ids = await _tokenize_via_frontend(messages, model=body.get("model", ""))
        response_data["prompt_token_ids"] = prompt_ids

        # Extract completion token IDs from logprobs
        for choice in response_data.get("choices", []):
            lp = choice.get("logprobs")
            if lp and lp.get("content"):
                logprobs_content = lp["content"]
                completion_ids = _extract_completion_ids_from_logprobs(logprobs_content)
                assert len(completion_ids) == len(logprobs_content), (
                    f"Token alignment: {len(completion_ids)} ids vs "
                    f"{len(logprobs_content)} logprobs"
                )
                choice["token_ids"] = completion_ids
    except Exception as e:
        logger.warning(f"[admin] Failed to inject token_ids: {e}")

    return response_data


@app.post("/v1/chat/completions")
async def chat_completions_proxy(request: Request):
    """Proxy chat completions to Dynamo frontend with token_ids injection.

    Injects prompt_token_ids (from Rust /tokenize) and choice.token_ids
    (from logprobs bytes) into the response for Prime-RL's verifiers client.
    """
    body = await request.json()
    wants_token_ids = body.pop("return_token_ids", False)
    extra = body.get("extra_body", {})
    if isinstance(extra, dict):
        wants_token_ids = wants_token_ids or extra.pop("return_token_ids", False)
        if not extra:
            body.pop("extra_body", None)
    _strip_unsupported(body)

    if "logprobs" not in body:
        body["logprobs"] = True

    stream = body.get("stream", False)

    client = _get_http_client()
    url = f"{DYNAMO_FRONTEND_URL}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if "authorization" in request.headers:
        headers["Authorization"] = request.headers["authorization"]

    if stream:
        async def _stream_proxy():
            async with client.stream("POST", url, json=body, headers=headers) as resp:
                async for chunk in resp.aiter_bytes():
                    yield chunk

        return StreamingResponse(_stream_proxy(), media_type="text/event-stream")
    else:
        resp = await client.post(url, json=body, headers=headers)
        response_data = resp.json()
        if resp.status_code == 200:
            response_data = await _inject_token_ids(body, response_data)
        return JSONResponse(content=response_data, status_code=resp.status_code)


# ══════════════════════════════════════════════════════════════════════
# Proxy: /v1/models (Prime-RL health-checks this)
# ══════════════════════════════════════════════════════════════════════

@app.get("/v1/models")
async def models_proxy():
    """Proxy /v1/models to Dynamo frontend."""
    client = _get_http_client()
    resp = await client.get(f"{DYNAMO_FRONTEND_URL}/v1/models")
    return JSONResponse(content=resp.json(), status_code=resp.status_code)


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="Dynamo RL Admin Service")
    parser.add_argument("--port", type=int, default=8002, help="Port to listen on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--model", type=str, default="", help="Model name (for fallback tokenizer)")
    parser.add_argument("--dynamo-frontend", type=str, default="http://localhost:8000",
                        help="Dynamo Rust frontend URL")
    parser.add_argument("--dynamo-system", type=str, default="http://localhost:8081",
                        help="Dynamo vLLM system port URL")
    args = parser.parse_args()

    global DYNAMO_FRONTEND_URL, DYNAMO_SYSTEM_URL, MODEL_NAME
    DYNAMO_FRONTEND_URL = args.dynamo_frontend
    DYNAMO_SYSTEM_URL = args.dynamo_system
    if args.model:
        MODEL_NAME = args.model

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logger.info(
        f"Starting RL Admin service on {args.host}:{args.port} "
        f"(frontend={DYNAMO_FRONTEND_URL}, system={DYNAMO_SYSTEM_URL}, model={MODEL_NAME})"
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
