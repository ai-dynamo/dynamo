# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Archive raw token evidence from an already-running reference HTTP server."""

from __future__ import annotations

import argparse
import base64
import hashlib
import http.client
import ipaddress
import math
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .contract import (
    SCHEMA,
    ReplayError,
    dumps,
    file_digest,
    loads,
    read_request_seeds,
    validate_profile,
    write_json,
)

CAPTURE_SCHEMA = "dynamo-frontend-replay-capture/v1"
RECORD_SCHEMA = "dynamo-frontend-replay-capture-record/v1"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _reference_url(value: str, authenticated: bool) -> str:
    try:
        parsed = urllib.parse.urlsplit(value)
        valid = (
            parsed.scheme in {"http", "https"}
            and bool(parsed.hostname)
            and parsed.username is None
            and parsed.password is None
            and not parsed.query
            and not parsed.fragment
        )
        # Accessing port also rejects malformed ports before any HTTP request.
        parsed.port
    except ValueError as exc:
        raise ReplayError("invalid reference URL") from exc
    if not valid:
        raise ReplayError(
            "reference URL requires HTTP(S), without credentials or query"
        )
    host = parsed.hostname or ""
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host.lower().rstrip(".") == "localhost"
    if authenticated and parsed.scheme != "https" and not loopback:
        raise ReplayError("reference credentials require HTTPS except on loopback")
    return value.rstrip("/")


def _seeds(payload: bytes, model_name: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for seed in read_request_seeds(payload, model_name):
        identifier = seed["id"]
        request = seed["request"]
        if not isinstance(request.get("messages"), list) or not request["messages"]:
            raise ReplayError("capture requires nonempty Chat Completions messages")
        if "stream" in request and request["stream"] is not False:
            raise ReplayError("capture accepts only absent or false stream")
        if "stream_options" in request:
            raise ReplayError("stream_options is unsupported for unary capture")
        if "n" in request and (type(request["n"]) is not int or request["n"] != 1):
            raise ReplayError("capture supports only one choice")
        if "return_token_ids" in request and request["return_token_ids"] is not True:
            raise ReplayError("capture requires return_token_ids to be true when set")
        if {"return_prompt_token_ids", "return_meta_info"} & request.keys():
            raise ReplayError("unsupported capture-only request selector")
        original = {key: seed[key] for key in ("id", "source", "request")}
        if "reference" in seed:
            original["reference"] = seed["reference"]
        wire = {
            key: value
            for key, value in request.items()
            if key not in {"stream", "return_token_ids"}
        }
        wire["return_token_ids"] = True
        wire["stream"] = False
        try:
            filename = _sha256(identifier.encode("utf-8")) + ".json"
            wire_text = dumps(wire)
            wire_text.encode("utf-8")
            dumps(original).encode("utf-8")
        except (ValueError, UnicodeError, RecursionError) as exc:
            raise ReplayError("seed cannot be serialized as UTF-8 JSON") from exc
        result.append({"seed": original, "wire": wire_text, "filename": filename})
    if not result:
        raise ReplayError("capture requires at least one seed")
    return result


def _fetch(
    opener: urllib.request.OpenerDirector,
    url: str,
    headers: dict[str, str],
    timeout: float,
    data: bytes | None = None,
) -> tuple[int | None, bytes, dict[str, str] | None]:
    status = None
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        try:
            response = opener.open(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            response = exc
        with response:
            status = response.status
            return status, response.read(), None
    except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
        partial = exc.partial if isinstance(exc, http.client.IncompleteRead) else b""
        return status, partial, {"kind": "transport", "message": type(exc).__name__}


def _parse_response(payload: bytes) -> Any:
    return loads(payload.decode("utf-8", errors="strict"))


def _server_info(
    opener: urllib.request.OpenerDirector,
    url: str,
    headers: dict[str, str],
    timeout: float,
    model_name: str,
) -> dict[str, Any]:
    info = {}
    for path in ("/version", "/v1/models"):
        status, payload, error = _fetch(opener, url + path, headers, timeout)
        if error or status != 200:
            raise ReplayError("reference server metadata request failed")
        try:
            info[path] = _parse_response(payload)
        except (ValueError, UnicodeError, RecursionError) as exc:
            raise ReplayError("reference server metadata is not UTF-8 JSON") from exc
    version = (
        info["/version"].get("version") if isinstance(info["/version"], dict) else None
    )
    models = (
        info["/v1/models"].get("data") if isinstance(info["/v1/models"], dict) else None
    )
    if (
        not isinstance(version, str)
        or not version
        or not isinstance(models, list)
        or not all(
            isinstance(item, dict) and isinstance(item.get("id"), str)
            for item in models
        )
    ):
        raise ReplayError("reference server metadata has an unsupported shape")
    identifiers = [item["id"] for item in models]
    if model_name not in identifiers:
        raise ReplayError("reference server does not advertise profile.model_name")
    # Do not retain arbitrary server fields, local model paths, or HTTP headers.
    return {"url": url, "version": version, "models": identifiers}


def _token_evidence(response: Any) -> None:
    if not isinstance(response, dict):
        raise ReplayError("reference response must be an object")
    choices = response.get("choices")
    if (
        not isinstance(choices, list)
        or len(choices) != 1
        or not isinstance(choices[0], dict)
    ):
        raise ReplayError("reference response must contain exactly one choice")
    choice = choices[0]
    if "index" in choice and (type(choice["index"]) is not int or choice["index"] != 0):
        raise ReplayError("reference response choice index must be zero")
    for value in (response.get("prompt_token_ids"), choice.get("token_ids")):
        if (
            not isinstance(value, list)
            or not value
            or any(type(i) is not int or not 0 <= i <= 0xFFFFFFFF for i in value)
        ):
            raise ReplayError("reference response requires nonempty uint32 token IDs")
    if not isinstance(choice.get("finish_reason"), str) or choice[
        "finish_reason"
    ] not in {"stop", "length", "tool_calls"}:
        raise ReplayError("reference response requires a terminal API finish reason")


def validate_target(
    url: str,
    timeout: float,
    concurrency: int,
    api_key_env: str | None,
) -> tuple[str, str]:
    """Validate capture transport settings before downloads or HTTP requests."""
    if (
        type(timeout) not in {int, float}
        or not math.isfinite(timeout)
        or timeout <= 0
        or type(concurrency) is not int
        or concurrency <= 0
    ):
        raise ReplayError("capture timeout and concurrency must be positive")
    token = os.environ.get(api_key_env, "") if api_key_env else ""
    if api_key_env and not token:
        raise ReplayError("the requested API-key environment variable is empty")
    if token and any(not 0x21 <= ord(character) <= 0x7E for character in token):
        raise ReplayError("the API key must be a visible ASCII header value")
    return _reference_url(url, bool(token)), token


def capture_requests(
    requests_path: Path,
    profile_path: Path,
    url: str,
    output_dir: Path,
    *,
    timeout: float = 600,
    concurrency: int = 2,
    api_key_env: str | None = None,
) -> dict[str, Any]:
    """Capture prepared seeds; no inference engine or model is started here."""
    url, token = validate_target(url, timeout, concurrency, api_key_env)
    profile_bytes = profile_path.read_bytes()
    document = loads(profile_bytes)
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise ReplayError("profile document must use the replay schema")
    profile = validate_profile(document.get("profile"))
    if not profile["model_name"]:
        raise ReplayError("capture requires a nonempty profile.model_name")
    payload = requests_path.read_bytes()
    seeds = _seeds(payload, profile["model_name"])
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ReplayError("capture output directory must be empty")
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Exclusive ownership also prevents two fresh captures racing in an empty directory.
    lock = output_dir / ".capture-in-progress"
    with lock.open("x", encoding="utf-8"):
        pass
    fd = os.open(
        output_dir / "requests.jsonl", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    # urllib has no implicit netrc handling; an explicit empty proxy map ignores all
    # proxy environment variables. Redirects never receive the bearer token.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    reference = _server_info(opener, url, headers, timeout, profile["model_name"])

    def capture(seed: dict[str, Any]) -> dict[str, Any]:
        original = seed["seed"]
        request_bytes = seed["wire"].encode("utf-8")
        status, raw, error = _fetch(
            opener, url + "/v1/chat/completions", headers, timeout, request_bytes
        )
        response = None
        try:
            response = _parse_response(raw)
        except (ValueError, UnicodeError, RecursionError) as exc:
            error = error or {
                "kind": "response_encoding",
                "message": type(exc).__name__,
            }
        if status != 200:
            error = error or {
                "kind": "http",
                "message": "reference returned a non-200 HTTP status",
            }
        if error is None:
            try:
                _token_evidence(response)
            except ReplayError as exc:
                error = {"kind": "token_evidence", "message": str(exc)}
        record = {
            "schema": RECORD_SCHEMA,
            "id": original["id"],
            "source": original["source"],
            "original_request": original["request"],
            "wire_request_bytes": seed["wire"],
            "request_sha256": _sha256(request_bytes),
            "http_status": status,
            "response_bytes_base64": base64.b64encode(raw).decode("ascii"),
            "response_sha256": _sha256(raw),
            "response": response,
            "error": error,
        }
        if "reference" in original:
            record["reference"] = original["reference"]
        relative = "records/" + seed["filename"]
        destination = output_dir / relative
        write_json(destination, record)
        return {
            "id": original["id"],
            "path": relative,
            "sha256": file_digest(destination),
            "status": "captured" if error is None else "error",
        }

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        entries = list(executor.map(capture, seeds))
    successful = sum(item["status"] == "captured" for item in entries)
    manifest = {
        "schema": CAPTURE_SCHEMA,
        "complete": successful == len(seeds),
        "count": len(seeds),
        "successful_count": successful,
        "profile": profile,
        "profile_sha256": _sha256(profile_bytes),
        "requests_path": "requests.jsonl",
        "requests_sha256": _sha256(payload),
        "reference": reference,
        "records": entries,
    }
    # The manifest is the commit point; interrupted runs remain unpreparable.
    write_json(output_dir / "capture.json", manifest)
    lock.unlink()
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--api-key-env")
    args = parser.parse_args(argv)
    try:
        result = capture_requests(
            args.requests,
            args.profile,
            args.url,
            args.output,
            timeout=args.timeout,
            concurrency=args.concurrency,
            api_key_env=args.api_key_env,
        )
    except (ReplayError, OSError) as exc:
        print(dumps({"complete": False, "error": str(exc)}))
        return 2
    print(
        dumps({key: result[key] for key in ("complete", "count", "successful_count")})
    )
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
