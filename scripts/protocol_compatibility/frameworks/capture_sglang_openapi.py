# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture pinned SGLang production OpenAPI through ASGI without lifespan.

Run in the recorded SGLang dependency environment. The profile owns the source
revision and SSE selectors. This helper preserves HTTP bytes, verifies imported
SGLang Python sources, and writes acquisition evidence; it never starts inference.
"""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path


def git(source: Path, *arguments: str) -> str:
    """Return stripped Git output from source; propagate command failures."""
    return subprocess.check_output(
        ["git", "-C", str(source), *arguments], text=True
    ).strip()


def verify_source(source: Path, profile: dict) -> None:
    """Require the exact clean source revision, including no untracked files.

    Missing repositories, a different head, and worktree changes fail before
    import. Loaded-file checks separately reject ignored or shadowing modules.
    """
    if git(source, "rev-parse", "HEAD") != profile["source"]["revision"]:
        raise ValueError("SGLang checkout does not match the profile revision")
    if git(source, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("SGLang checkout must be clean")


def verify_loaded_sources(source: Path) -> dict:
    """Return SHA256s of loaded SGLang Python files after checking Git blobs.

    Reject imports outside source/python and modified or untracked Python files.
    These checks identify Python source, not native-library build provenance.
    """
    hashes = {}
    for name, module in list(sys.modules.items()):
        if name != "sglang" and not name.startswith("sglang."):
            continue
        filename = getattr(module, "__file__", None)
        if not filename or not filename.endswith(".py"):
            continue
        path = Path(filename).resolve()
        relative = path.relative_to(source.resolve())
        if relative.parts[0] != "python":
            raise ValueError(f"Unexpected SGLang module location: {name}")
        if git(source, "hash-object", str(path)) != git(
            source, "rev-parse", f"HEAD:{relative.as_posix()}"
        ):
            raise ValueError(f"Imported SGLang source changed: {name}")
        hashes[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    if "python/sglang/srt/entrypoints/http_server.py" not in hashes:
        raise ValueError("Production SGLang HTTP app was not imported")
    return hashes


def resolve(document: dict, pointer: str):
    """Resolve a local JSON pointer, raising on external or unresolved references."""
    if not isinstance(pointer, str) or not pointer.startswith("#/"):
        raise ValueError(f"Expected local reference: {pointer}")
    value = document
    for part in pointer[2:].split("/"):
        key = part.replace("~1", "/").replace("~0", "~")
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def make_adapter(raw: bytes, profile: dict) -> dict:
    """Bind reviewed SGLang event selectors to raw bytes after sanity checks.

    Require native unary/chunk refs and SGLang's inline nested error declaration.
    Resolve schema refs without rewriting the document; invalid dialects, shapes,
    selectors or references raise. This is not a runtime or completeness check.
    """
    document = json.loads(raw)
    if document.get("openapi") != "3.1.0":
        raise ValueError("Expected SGLang OpenAPI 3.1.0")
    # Map keys may themselves be '$ref'; examples/defaults/extensions are data.
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
    streams = profile["streams"]
    expected = {
        f"#/paths/{path}/post/responses/200/content/text~1event-stream/schema": model
        for path, model in (
            ("~1v1~1chat~1completions", "ChatCompletion"),
            ("~1v1~1completions", "Completion"),
        )
    }
    if len(streams) != 2 or {s["pointer"] for s in streams} != expected.keys():
        raise ValueError("Profile must select both completion endpoints")
    for selection in streams:
        location = selection["pointer"]
        prefix = expected[location]
        unary = resolve(
            document, location.replace("text~1event-stream", "application~1json")
        )
        event = resolve(document, location)
        alternatives = event.get("properties", {}).get("data", {}).get("anyOf", [])
        payloads = [
            a.get("contentSchema")
            for a in alternatives
            if a.get("contentMediaType") == "application/json"
        ]
        error = {
            "type": "object",
            "required": ["error"],
            "properties": {"error": {"$ref": "#/components/schemas/ErrorResponse"}},
        }
        if (
            selection["interpretation"] != "event-schema"
            or unary.get("$ref") != f"#/components/schemas/{prefix}Response"
            or event.get("type") != "object"
            or "data" not in event.get("required", [])
            or event["properties"]["data"].get("type") != "string"
            or {"$ref": f"#/components/schemas/{prefix}StreamResponse"} not in payloads
            or error not in payloads
            or not any(a.get("const") == "[DONE]" for a in alternatives)
        ):
            raise ValueError(f"Unexpected SGLang unary/SSE declaration: {location}")
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


async def capture() -> tuple[bytes, dict]:
    """Return raw production-app HTTP bytes and headers, without ASGI lifespan.

    Call in a fresh process with GPU visibility disabled before importing SGLang.
    HTTP errors and CUDA initialization fail; no engine or model is constructed.
    """
    import httpx
    import torch
    from sglang.srt.entrypoints.http_server import app

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://schema.invalid"
    ) as client:
        response = await client.get("/openapi.json")
    response.raise_for_status()
    if torch.cuda.is_initialized():
        raise ValueError("Schema capture initialized CUDA")
    return response.content, {
        "status": response.status_code,
        "headers": dict(response.headers),
    }


def main() -> None:
    """Write raw schema, adapter and provenance to a new output directory.

    Require a clean pinned checkout and fresh process. Validation precedes writes;
    an I/O failure can leave partial output, which must not be assessed. Existing
    output directories are never reused. Dependencies are recorded, not installed.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    profile = json.loads(args.profile.read_text())
    verify_source(args.source, profile)
    if any(n == "sglang" or n.startswith("sglang.") for n in sys.modules):
        raise ValueError("Capture requires a fresh process")
    settings = dict(
        CUDA_VISIBLE_DEVICES="",
        SGLANG_USE_CPU_ENGINE="1",
        DISABLE_OPENAPI_DOC="0",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    os.environ.update(settings)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.source.resolve() / "python"))
    raw, http = asyncio.run(capture())
    loaded = verify_loaded_sources(args.source)
    verify_source(args.source, profile)
    adapter = make_adapter(raw, profile)
    evidence = {
        "kind": "production_app_asgi_no_lifespan",
        "source": profile["source"],
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "http": http,
        "configuration": settings,
        "python": sys.version,
        "loaded_source_sha256": loaded,
        "dependencies": dict(
            sorted(
                (d.metadata["Name"], d.version)
                for d in importlib.metadata.distributions()
                if d.metadata["Name"]
            )
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "sglang-openapi.json").write_bytes(raw)
    for filename, value in (
        ("response-adapters.json", adapter),
        ("acquisition.json", evidence),
    ):
        (args.output_dir / filename).write_text(json.dumps(value, indent=2) + "\n")


if __name__ == "__main__":
    main()
