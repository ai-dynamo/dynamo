# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the pinned vLLM recipe's raw schema and explicit SSE adapter.

Run as a standalone script in the wheel environment documented in
lib/llm/docs/dynamo-vllm-response-assessment.md. Only the standard library is
needed beyond vLLM's dependencies; no comparator or inference engine is started.
The reviewed profile owns source identity and SSE semantics, not this helper.
"""

import argparse
import base64
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import sys
from pathlib import Path


def verify_installation(profile: dict):
    """Return the installed distribution matching the profile's URL and version.

    Reject editable/index installs and shadowing source checkouts. This checks
    installer metadata, not archive authenticity: install with the profile's
    SHA256 first. Missing metadata or a mismatch raises ValueError.
    """
    dist = importlib.metadata.distribution("vllm")
    origin = json.loads(dist.read_text("direct_url.json") or "{}")
    spec = importlib.util.find_spec("vllm")
    if (
        dist.version != profile["wheel"]["version"]
        or origin.get("url") != profile["wheel"]["url"]
        or "archive_info" not in origin
        or spec is None
        or spec.origin is None
        or Path(spec.origin).resolve()
        != Path(dist.locate_file("vllm/__init__.py")).resolve()
    ):
        raise ValueError("Install the profile's exact wheel URL with its SHA256")
    return dist


def verify_loaded_sources(dist) -> None:
    """Reject imported vLLM Python files missing from or changed since RECORD.

    Checks installed-source consistency, not native execution or the truth of
    wheel metadata. Missing files/hashes and source overlays raise ValueError.
    """
    recorded = {Path(p.locate()).resolve(): p.hash for p in dist.files or ()}
    for name, module in list(sys.modules.items()):
        if name == "vllm" or name.startswith("vllm."):
            filename = getattr(module, "__file__", None)
            if filename and filename.endswith(".py"):
                path = Path(filename).resolve()
                entry = recorded.get(path)
                digest = (
                    base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest())
                    .decode()
                    .rstrip("=")
                )
                if entry is None or entry.mode != "sha256" or entry.value != digest:
                    raise ValueError(
                        f"Imported source differs from wheel RECORD: {name}"
                    )


def capture() -> bytes:
    """GET production-app OpenAPI without lifespan; return unaltered HTTP bytes.

    Call in a fresh process, before importing vLLM. Clear inherited VLLM_* settings
    in this process, disable GPU visibility and plugins, and enable offline model
    access. Never enter TestClient's context manager: that starts lifespan.
    HTTP errors or CUDA initialization fail.
    """
    # Inherited VLLM_* switches can add routes (dev mode, LoRA, Cohere).
    for name in [name for name in os.environ if name.startswith("VLLM_")]:
        del os.environ[name]
    os.environ.update(
        CUDA_VISIBLE_DEVICES="",
        VLLM_PLUGINS="",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    import torch
    from fastapi.testclient import TestClient
    from vllm.entrypoints.launchers.app import build_app

    args = argparse.Namespace(
        disable_fastapi_docs=False,
        enable_offline_docs=False,
        root_path="",
        allowed_origins=["*"],
        allow_credentials=False,
        allowed_methods=["*"],
        allowed_headers=["*"],
        api_key=None,
        enable_request_id_headers=False,
        enable_fault_tolerance=False,
        middleware=[],
        log_error_stack=False,
    )
    client = TestClient(build_app(args, supported_tasks=("generate",)))
    try:
        response = client.get("/openapi.json")
        response.raise_for_status()
        if torch.cuda.is_initialized():
            raise ValueError("Schema capture initialized CUDA")
        return response.content
    finally:
        client.close()


def resolve(document: dict, pointer: str):
    """Resolve a local JSON Pointer; reject external or unresolved references."""
    if not isinstance(pointer, str) or not pointer.startswith("#/"):
        raise ValueError(f"Expected a local JSON Pointer: {pointer}")
    value = document
    for part in pointer[2:].split("/"):
        key = part.replace("~1", "/").replace("~0", "~")
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def make_adapter(raw: bytes, profile: dict) -> dict:
    """Check this recipe's declarations and bind reviewed selectors to raw bytes.

    Require both unary bodies and parsed-event envelopes, and resolve every
    local $ref without rewriting the document. Unexpected dialects, selectors,
    payload shapes or broken/external refs raise ValueError/KeyError/IndexError.
    These are capture sanity checks, not the comparator's coverage analysis.
    """
    document = json.loads(raw)
    contract = profile["response_contract"]
    if document.get("openapi") != contract["openapi"]:
        raise ValueError("Unexpected OpenAPI dialect")
    # Map keys are user-defined names (including "$ref" or "default").
    maps = {
        "properties",
        "patternProperties",
        "$defs",
        "schemas",
        "paths",
        "responses",
        "content",
        "headers",
        "callbacks",
        "securitySchemes",
    }
    literals = {"default", "const", "enum", "example", "examples"}
    pending = [(document, False)]
    while pending:
        node, is_map = pending.pop()
        if isinstance(node, dict):
            if not is_map and "$ref" in node:
                resolve(document, node["$ref"])
            pending.extend(
                (value, not is_map and key in maps)
                for key, value in node.items()
                if is_map or (key not in literals and not key.startswith("x-"))
            )
        elif isinstance(node, list):
            pending.extend((value, False) for value in node)
    expected = {
        f"#/paths/{path}/post/responses/200/content/text~1event-stream/schema"
        for path in ("~1v1~1chat~1completions", "~1v1~1completions")
    }
    streams = contract["streams"]
    if len(streams) != 2 or {s["pointer"] for s in streams} != expected:
        raise ValueError("Profile must select both completion SSE responses")
    for selection in streams:
        if selection["interpretation"] != "event-schema":
            raise ValueError("This capture recipe requires event-schema")
        location = selection["pointer"]
        unary = resolve(
            document, location.replace("text~1event-stream", "application~1json")
        )
        event = resolve(document, location)
        properties = event.get("properties", {})
        data = properties.get("data", {})
        alternatives = data.get("anyOf", [])
        if (
            "$ref" not in unary
            or event.get("type") != "object"
            or "data" not in event.get("required", [])
            or not {"event", "id", "retry"} <= properties.keys()
            or data.get("type") != "string"
            or not any(a.get("const") == "[DONE]" for a in alternatives)
            or sum(
                a.get("contentMediaType") == "application/json"
                and "$ref" in a.get("contentSchema", {})
                for a in alternatives
            )
            != 2
        ):
            raise ValueError(f"Unexpected unary/SSE declaration: {location}")
    return {
        "version": 1,
        "sources": {
            "framework": {
                "sha256": hashlib.sha256(raw).hexdigest(),
                "source": profile["source"]["repository"],
                "revision": profile["source"]["revision"],
                "streams": streams,
            }
        },
    }


def main() -> None:
    """Write exactly two inputs to a new directory; failures exit nonzero.

    Existing output directories are never reused. Validation precedes output
    creation; an I/O failure may leave a partial directory, which must not be
    assessed. Only a successful exit qualifies the pair for comparison.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    profile = json.loads(args.profile.read_text())
    dist = verify_installation(profile)
    raw = capture()
    verify_loaded_sources(dist)
    adapter = make_adapter(raw, profile)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "vllm-openapi.json").write_bytes(raw)
    (args.output_dir / "response-adapters.json").write_text(
        json.dumps(adapter, indent=2) + "\n"
    )
    print(f"Captured schema and adapter in {args.output_dir}; no lifespan or inference")


if __name__ == "__main__":
    main()
