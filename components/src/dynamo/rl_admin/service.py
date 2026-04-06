"""
Dynamo RL Admin + TITO + Tokenize service.

A standalone FastAPI service that Prime-RL talks to.  It provides:

1. Admin endpoints (/pause, /resume, /update_weights, /health)
   - Calls vLLM engine routes via Dynamo system port.
2. /tokenize endpoint
   - Uses HuggingFace tokenizer locally.
3. /v1/chat/completions/tokens  (TITO)
   - Translates Prime-RL's `tokens` field to nvext.token_data and proxies
     to the Dynamo Rust frontend on /v1/chat/completions.

Prime-RL config:
    base_url = ["http://localhost:<this_port>/v1"]
    admin_base_url = ["http://localhost:<this_port>"]
"""

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

logger = logging.getLogger("dynamo.rl_admin")

app = FastAPI(title="Dynamo RL Admin + TITO Proxy")

# ── Configuration (set via env or CLI) ──────────────────────────────
DYNAMO_FRONTEND_URL = os.getenv("DYNAMO_FRONTEND_URL", "http://localhost:8000")
DYNAMO_SYSTEM_URL = os.getenv("DYNAMO_SYSTEM_URL", "http://localhost:8081")
MODEL_NAME = os.getenv("MODEL_NAME", "")
_tokenizer = None
_http_client: httpx.AsyncClient | None = None


def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=httpx.Timeout(600.0))
    return _http_client


# ── Tokenizer (lazy init) ──────────────────────────────────────────
def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer

        model = MODEL_NAME or os.getenv("MODEL_NAME", "")
        if not model:
            raise RuntimeError("MODEL_NAME not set -- cannot initialize tokenizer")
        logger.info(f"Loading tokenizer for {model}")
        _tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    return _tokenizer


# ── Helper: call engine route via Dynamo system port ────────────────
async def _call_engine_route(route: str, body: dict | None = None) -> dict:
    """Call a registered engine route on the vLLM worker via system HTTP."""
    client = _get_http_client()
    url = f"{DYNAMO_SYSTEM_URL}/engine_route/{route}"
    try:
        resp = await client.post(url, json=body or {})
        if resp.status_code == 200:
            return resp.json()
        # Engine routes may return plain text on error
        return {"status": "error", "message": resp.text, "http_status": resp.status_code}
    except Exception as e:
        logger.error(f"Engine route {route} failed: {e}")
        return {"status": "error", "message": str(e)}


# ══════════════════════════════════════════════════════════════════════
# Admin endpoints (Prime-RL calls these on admin_base_url)
# ══════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    return JSONResponse(content="OK")


@app.post("/pause")
async def pause(request: Request):
    """Pause inference engines (drain in-flight requests)."""
    result = await _call_engine_route("pause_generation")
    logger.info(f"[admin] POST /pause -> {result}")
    return JSONResponse(content=result)


@app.post("/resume")
async def resume():
    """Resume inference engines after weight update."""
    result = await _call_engine_route("resume_generation")
    logger.info(f"[admin] POST /resume -> {result}")
    return JSONResponse(content=result)


@app.post("/update_weights")
async def update_weights(request: Request):
    """Update weights from filesystem path.

    Prime-RL sends: {"weight_dir": "/path/to/weights"} or {"weight_dir": null}
    When weight_dir is null, it signals an NCCL broadcast (not filesystem).
    For filesystem mode, we do: pause -> flush -> load -> resume.
    """
    body = await request.json()
    weight_dir = body.get("weight_dir")

    if weight_dir is None:
        # NCCL mode: Prime-RL handles the broadcast itself.
        # The caller (Prime-RL) already paused/resumed around this call.
        logger.info("[admin] POST /update_weights weight_dir=None (NCCL mode, no-op)")
        return JSONResponse(content={"status": "ok", "message": "NCCL mode, no-op on Dynamo side"})

    logger.info(f"[admin] POST /update_weights weight_dir={weight_dir}")

    # Filesystem mode: load weights from path
    # Note: Prime-RL already calls /pause and /resume around /update_weights.
    # So we just do the load + cache flush here.
    flush_result = await _call_engine_route("flush_cache")
    logger.info(f"[admin] flush_cache -> {flush_result}")

    load_result = await _call_engine_route(
        "update_weights_from_path",
        {"path": weight_dir, "version": Path(weight_dir).name if weight_dir else "unknown"},
    )
    logger.info(f"[admin] update_weights_from_path -> {load_result}")

    return JSONResponse(content=load_result)


@app.post("/load_lora_adapter")
async def load_lora_adapter(request: Request):
    """Load a LoRA adapter. Stub for now."""
    body = await request.json()
    lora_name = body.get("lora_name", "")
    lora_path = body.get("lora_path", "")
    logger.info(f"[admin] POST /load_lora_adapter name={lora_name} path={lora_path}")
    # TODO: implement via engine route when LoRA support is added
    return JSONResponse(content={"status": "ok", "message": f"LoRA {lora_name} load stub"})


# ══════════════════════════════════════════════════════════════════════
# /tokenize endpoint (Prime-RL calls this for multi-turn prefix stitching)
# ══════════════════════════════════════════════════════════════════════

@app.post("/tokenize")
@app.post("/v1/tokenize")
async def tokenize(request: Request):
    """Tokenize messages using the model's chat template.

    Accepts: {"model": "...", "messages": [...], "add_generation_prompt": true}
    Returns: {"tokens": [int, ...], "count": int, "max_model_len": int}
    """
    body = await request.json()
    messages = body.get("messages", [])
    tools = body.get("tools")
    add_generation_prompt = body.get("add_generation_prompt", True)

    tok = _get_tokenizer()

    # Build kwargs for apply_chat_template
    kwargs: dict = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
    }
    if tools:
        kwargs["tools"] = tools

    token_ids = tok.apply_chat_template(messages, **kwargs)

    max_model_len = getattr(tok, "model_max_length", 32768)

    return JSONResponse(content={
        "tokens": token_ids,
        "count": len(token_ids),
        "max_model_len": min(max_model_len, 1_000_000),  # cap absurd defaults
    })


# ══════════════════════════════════════════════════════════════════════
# /v1/chat/completions/tokens  (TITO endpoint)
# Translates Prime-RL's `tokens` field -> nvext.token_data, proxies
# to the Dynamo Rust frontend at /v1/chat/completions.
# ══════════════════════════════════════════════════════════════════════

@app.post("/v1/chat/completions/tokens")
async def chat_completions_tokens(request: Request):
    """TITO: Token-In / Token-Out chat completions.

    Prime-RL sends a standard ChatCompletionRequest with an extra `tokens` field.
    We translate: tokens -> nvext.token_data, then proxy to Dynamo frontend.
    """
    body = await request.json()

    # Extract the tokens field (Prime-RL's TITO input)
    tokens = body.pop("tokens", None)
    if tokens is None:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "Missing 'tokens' field for TITO endpoint"}},
        )

    # Inject into nvext
    nvext = body.get("nvext", {}) or {}
    nvext["token_data"] = tokens
    # Request completion token IDs and prompt echo in response
    extra_fields = nvext.get("extra_fields", []) or []
    for field in ["token_ids", "completion_token_ids"]:
        if field not in extra_fields:
            extra_fields.append(field)
    nvext["extra_fields"] = extra_fields
    body["nvext"] = nvext

    # Ensure logprobs are requested (RL always needs them)
    if "logprobs" not in body:
        body["logprobs"] = True

    # Determine if streaming
    stream = body.get("stream", False)

    # Forward to Dynamo frontend
    client = _get_http_client()
    url = f"{DYNAMO_FRONTEND_URL}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    # Copy auth headers from original request
    if "authorization" in request.headers:
        headers["Authorization"] = request.headers["authorization"]

    if stream:
        # Stream the response back
        async def _stream_proxy():
            async with client.stream("POST", url, json=body, headers=headers) as resp:
                async for chunk in resp.aiter_bytes():
                    yield chunk

        return StreamingResponse(
            _stream_proxy(),
            media_type="text/event-stream",
        )
    else:
        resp = await client.post(url, json=body, headers=headers)
        return JSONResponse(content=resp.json(), status_code=resp.status_code)


# ══════════════════════════════════════════════════════════════════════
# Proxy: /v1/chat/completions (forward to Dynamo frontend)
# This allows Prime-RL to use a single base_url for everything.
# ══════════════════════════════════════════════════════════════════════

@app.post("/v1/chat/completions")
async def chat_completions_proxy(request: Request):
    """Proxy standard chat completions to Dynamo frontend."""
    body = await request.json()
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

        return StreamingResponse(
            _stream_proxy(),
            media_type="text/event-stream",
        )
    else:
        resp = await client.post(url, json=body, headers=headers)
        return JSONResponse(content=resp.json(), status_code=resp.status_code)


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

    parser = argparse.ArgumentParser(description="Dynamo RL Admin + TITO + Tokenize Service")
    parser.add_argument("--port", type=int, default=8002, help="Port to listen on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--model", type=str, default="", help="Model name for tokenizer")
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
