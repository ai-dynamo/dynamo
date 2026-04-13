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
    """Call a registered engine route on the vLLM worker via system HTTP.

    Engine routes are exposed at /engine/<route_name> on the system port.
    """
    client = _get_http_client()
    url = f"{DYNAMO_SYSTEM_URL}/engine/{route}"
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
    if flush_result.get("status") != "ok":
        return JSONResponse(content=flush_result, status_code=502)

    load_result = await _call_engine_route(
        "update_weights_from_path",
        {"path": weight_dir, "version": Path(weight_dir).name if weight_dir else "unknown"},
    )
    logger.info(f"[admin] update_weights_from_path -> {load_result}")

    status_code = 200 if load_result.get("status") == "ok" else 502
    return JSONResponse(content=load_result, status_code=status_code)


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

    # Dynamo Rust frontend requires `messages` to be non-empty.
    # When Prime-RL sends tokens, it usually also sends messages (for logging).
    # If messages are missing, inject a placeholder so the frontend doesn't reject.
    if "messages" not in body or not body["messages"]:
        body["messages"] = [{"role": "user", "content": "(token-in mode)"}]

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

    # Strip fields Dynamo frontend doesn't know about
    _strip_unsupported(body)

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

# Fields that Prime-RL sends but Dynamo Rust frontend doesn't recognize.
# Strip them before proxying to avoid 400 validation errors.
_STRIP_FIELDS = {"return_token_ids", "tokens"}


def _strip_unsupported(body: dict) -> dict:
    """Remove fields the Dynamo frontend doesn't support."""
    for f in _STRIP_FIELDS:
        body.pop(f, None)
    return body


def _inject_token_ids(body: dict, response_data: dict) -> dict:
    """Inject prompt_token_ids and choice.token_ids into the response.

    Prime-RL's verifiers client requires these vLLM-specific fields to
    extract per-token logprobs.  Dynamo doesn't support return_token_ids
    natively, so we reconstruct the fields here using the HF tokenizer.
    """
    try:
        tok = _get_tokenizer()
        messages = body.get("messages", [])
        if not messages:
            return response_data

        # Tokenize the prompt (same way the orchestrator will re-tokenize)
        prompt_ids = tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
        )
        if isinstance(prompt_ids, dict):
            prompt_ids = prompt_ids["input_ids"]
        response_data["prompt_token_ids"] = prompt_ids

        # Extract completion token IDs from logprobs content.
        # Each logprob entry is exactly one generated token.  We must produce
        # exactly one token ID per entry to keep lengths aligned.
        #
        # Strategy: try convert_tokens_to_ids (works for special tokens like
        # <think>).  For byte-level tokens (e.g. "\n" -> None because the vocab
        # entry is "Ċ"), encode the raw bytes and take the first ID.
        for choice in response_data.get("choices", []):
            lp = choice.get("logprobs")
            if lp and lp.get("content"):
                logprobs_content = lp["content"]
                completion_ids = []
                for entry in logprobs_content:
                    token_str = entry.get("token", "")
                    tid = tok.convert_tokens_to_ids(token_str)
                    if isinstance(tid, int):
                        completion_ids.append(tid)
                    else:
                        # Byte-level fallback: decode bytes, encode, take first ID.
                        # Always append exactly 1 ID to keep alignment with logprobs.
                        token_bytes = entry.get("bytes")
                        if token_bytes is not None:
                            text = bytes(token_bytes).decode("utf-8", errors="replace")
                            ids = tok.encode(text, add_special_tokens=False)
                            completion_ids.append(ids[0] if ids else 0)
                        else:
                            completion_ids.append(0)
                choice["token_ids"] = completion_ids
    except Exception as e:
        logger.warning(f"[admin] Failed to inject token_ids: {e}")

    return response_data


def _pretokenize_request(body: dict) -> list[int] | None:
    """Pre-tokenize the prompt with the Python HF tokenizer.

    Returns the prompt token IDs, or None if tokenization is not possible.
    Also injects the token IDs into the request body via nvext.token_data
    so the Rust frontend bypasses its own tokenizer and uses ours.
    This ensures the model sees exactly the same tokens that Prime-RL's
    trainer will use for KL computation -- eliminating the Rust/Python
    tokenizer mismatch that causes is_masked noise.
    """
    try:
        tok = _get_tokenizer()
        messages = body.get("messages")
        if not messages:
            return None

        prompt_ids = tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
        )
        if isinstance(prompt_ids, dict):
            prompt_ids = prompt_ids["input_ids"]

        # Inject into nvext.token_data so the Rust frontend uses our tokens
        nvext = body.get("nvext") or {}
        nvext["token_data"] = prompt_ids
        body["nvext"] = nvext

        return prompt_ids
    except Exception as e:
        logger.warning(f"[admin] Pre-tokenization failed, falling back to Rust tokenizer: {e}")
        return None


@app.post("/v1/chat/completions")
async def chat_completions_proxy(request: Request):
    """Proxy standard chat completions to Dynamo frontend.

    Pre-tokenizes the prompt with the Python HF tokenizer and injects via
    nvext.token_data so the Rust frontend uses the same tokens as the trainer.
    On the response side, injects prompt_token_ids and choice.token_ids
    for Prime-RL's verifiers client.
    """
    body = await request.json()
    wants_token_ids = body.pop("return_token_ids", False)
    # Also check extra_body
    extra = body.get("extra_body", {})
    if isinstance(extra, dict):
        wants_token_ids = wants_token_ids or extra.pop("return_token_ids", False)
        if not extra:
            body.pop("extra_body", None)
    _strip_unsupported(body)

    # Always ensure logprobs for RL (needed for token_ids injection + training signal)
    if "logprobs" not in body:
        body["logprobs"] = True

    # Pre-tokenize prompt with Python HF tokenizer -> inject via nvext.token_data
    # This bypasses the Rust frontend's tokenizer, ensuring the model sees
    # the same tokens that Prime-RL's trainer uses for reference logprobs.
    prompt_ids = _pretokenize_request(body)

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
        response_data = resp.json()
        if resp.status_code == 200:
            response_data = _inject_token_ids(body, response_data)
            # If we pre-tokenized, use our prompt_ids (they match the model's input exactly)
            if prompt_ids is not None:
                response_data["prompt_token_ids"] = prompt_ids
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
