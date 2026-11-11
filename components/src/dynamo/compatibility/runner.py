# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare observed unary text-chat prompt tokens, without substituting counts."""

import argparse
import copy
import hashlib
import html
import http.client
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

MAX_BODY = 4 * 1024 * 1024
REQUEST_FIELDS = {
    "messages",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "chat_template_kwargs",
    "chat_template_args",
    "reasoning_effort",
    "thinking",
    "continue_final_message",
    "add_generation_prompt",
}
SCOPE = "Unary text Chat Completions prompt-token parity only"
LIMITATIONS = (
    "Output parsing, streaming, generation controls and media are NOT TESTED. "
    "Checkpoint, template, processor and build identities are caller-declared, "
    "not verified by model-list responses. A PASS is scoped observed input parity, "
    "not full model or API certification."
)


def unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def decode_json(content: bytes) -> Any:
    try:
        return json.loads(
            content, object_pairs_hook=unique_object, parse_constant=reject_constant
        )
    except RecursionError as error:
        raise ValueError("JSON nesting exceeds the decoder limit") from error


@dataclass(frozen=True)
class Endpoint:
    base_url: str
    kind: str
    timeout: float
    api_key: str

    def request(self, route: str, payload: dict | None = None) -> dict:
        if any(
            ord(character) < 32 or ord(character) == 127 or ord(character) > 255
            for character in self.api_key
        ):
            raise ValueError("API key contains invalid header characters")
        address = urlsplit(self.base_url)
        connection_class = (
            http.client.HTTPSConnection
            if address.scheme == "https"
            else http.client.HTTPConnection
        )
        connection = connection_class(
            address.hostname, address.port, timeout=self.timeout
        )
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = json.dumps(payload, ensure_ascii=False).encode() if payload else None
        try:
            connection.request(
                "POST" if payload is not None else "GET", route, body, headers
            )
            response = connection.getresponse()
            content = response.read(MAX_BODY + 1)
            if len(content) > MAX_BODY:
                raise ValueError("response exceeds the 4 MiB evidence limit")
            if response.status != 200:
                detail = content[:1024].decode("utf-8", errors="replace")
                if self.api_key:
                    for secret in (self.api_key, json.dumps(self.api_key)[1:-1]):
                        detail = detail.replace(secret, "[REDACTED]")
                raise ValueError(f"HTTP {response.status} on {route}: {detail}")
            data = decode_json(content)
            if not isinstance(data, dict):
                raise ValueError("response must be a JSON object")
            return data
        finally:
            connection.close()


def validate_manifest(data: Any) -> None:
    if not isinstance(data, dict) or set(data) != {
        "schema_version",
        "profile",
        "model",
        "provenance",
        "required_cases",
        "cases",
    }:
        raise ValueError(
            "manifest fields must be schema_version, profile, model, provenance, required_cases, cases"
        )
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("unsupported manifest schema_version")
    for key in ("profile", "model"):
        if not isinstance(data[key], str) or not data[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    if not isinstance(data["provenance"], dict) or not data["provenance"]:
        raise ValueError(
            "provenance must declare model_revision, dynamo_build, sglang_build and processor"
        )
    for key in ("model_revision", "dynamo_build", "sglang_build", "processor"):
        if (
            not isinstance(data["provenance"].get(key), str)
            or not data["provenance"][key].strip()
        ):
            raise ValueError(f"provenance.{key} must be a nonempty declaration")
        if data["provenance"][key].startswith("SET_TO_"):
            raise ValueError(
                f"replace the provenance.{key} example placeholder before execution"
            )
    required = data["required_cases"]
    if not isinstance(required, list) or not required or len(required) > 128:
        raise ValueError("required_cases must contain 1..128 unique IDs")
    if any(not isinstance(item, str) or not item.strip() for item in required):
        raise ValueError("required case IDs must be nonempty strings")
    if len(set(required)) != len(required):
        raise ValueError("duplicate required case ID")
    if not isinstance(data["cases"], list) or len(data["cases"]) > 128:
        raise ValueError("cases must be a list of at most 128 entries")
    seen = set()
    for case in data["cases"]:
        if not isinstance(case, dict) or set(case) != {"id", "request"}:
            raise ValueError("each case must contain only id and request")
        case_id = case["id"]
        if not isinstance(case_id, str) or case_id not in required or case_id in seen:
            raise ValueError("case IDs must be unique and declared in required_cases")
        seen.add(case_id)
        request = case["request"]
        if not isinstance(request, dict) or set(request) - REQUEST_FIELDS:
            raise ValueError(
                f"{case_id}: unsupported request fields for the text-input profile"
            )
        messages = request.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{case_id}: nonempty messages required")
        for message in messages:
            if (
                not isinstance(message, dict)
                or not isinstance(message.get("role"), str)
                or message["role"]
                not in {"system", "developer", "user", "assistant", "tool"}
            ):
                raise ValueError(f"{case_id}: invalid message")
            if not isinstance(message.get("content"), (str, type(None))):
                raise ValueError(
                    f"{case_id}: this profile accepts text message content only"
                )


def validate_url(value: str) -> str:
    address = urlsplit(value)
    if (
        address.scheme not in {"http", "https"}
        or not address.hostname
        or address.username is not None
        or address.password is not None
        or address.path not in {"", "/"}
        or address.query
        or address.fragment
    ):
        raise ValueError(
            "endpoint URLs must be HTTP(S) origins without credentials, paths or queries"
        )
    address.port  # Validate a malformed or out-of-range port before connecting.
    return value.rstrip("/")


def observe(endpoint: Endpoint, model: str, request: dict) -> dict:
    payload = copy.deepcopy(request)
    payload.update(model=model, stream=False, n=1, max_tokens=1, temperature=0)
    if endpoint.kind == "dynamo":
        payload["nvext"] = {"extra_fields": ["prompt_token_ids"]}
    else:
        payload["return_prompt_token_ids"] = True
    response = endpoint.request("/v1/chat/completions", payload)
    if response.get("model") != model:
        raise ValueError("response model does not match the requested model")
    choices = response.get("choices")
    if (
        not isinstance(choices, list)
        or len(choices) != 1
        or not isinstance(choices[0], dict)
    ):
        raise ValueError("expected exactly one response choice")
    if type(choices[0].get("index")) is not int or choices[0]["index"] != 0:
        raise ValueError("expected choice index 0")
    if endpoint.kind == "dynamo":
        extension = response.get("nvext")
        tokens = (
            extension.get("prompt_token_ids") if isinstance(extension, dict) else None
        )
    else:
        tokens = choices[0].get("prompt_token_ids")
    if (
        not isinstance(tokens, list)
        or not tokens
        or any(type(token) is not int or not 0 <= token < 2**32 for token in tokens)
    ):
        raise ValueError(
            "missing or invalid prompt token IDs; counts are not a substitute"
        )
    return {"token_ids": tokens, "response_model": response["model"]}


def compare_tokens(left: list[int], right: list[int]) -> dict | None:
    for index in range(max(len(left), len(right))):
        left_id = left[index] if index < len(left) else None
        right_id = right[index] if index < len(right) else None
        if left_id != right_id:
            return {
                "index": index,
                "dynamo_id": left_id,
                "sglang_id": right_id,
                "dynamo_length": len(left),
                "sglang_length": len(right),
                "dynamo_context": left[max(0, index - 4) : index + 5],
                "sglang_context": right[max(0, index - 4) : index + 5],
            }
    return None


def evaluate(manifest: dict, endpoints: list[Endpoint]) -> tuple[list[dict], dict]:
    available = {case["id"]: case for case in manifest["cases"]}
    if any(case_id not in available for case_id in manifest["required_cases"]):
        return [
            {
                "id": case_id,
                "status": "MISSING" if case_id not in available else "BLOCKED",
                "observations": {},
                "errors": {
                    "coverage": "required coverage is incomplete; no requests executed"
                },
            }
            for case_id in manifest["required_cases"]
        ], {}
    models = {}
    errors = {}
    for endpoint in endpoints:
        try:
            observed = endpoint.request("/v1/models")
            entries = observed.get("data")
            if not isinstance(entries, list) or not any(
                isinstance(item, dict) and item.get("id") == manifest["model"]
                for item in entries
            ):
                raise ValueError("requested model is not advertised by the endpoint")
            models[endpoint.kind] = observed
        except (OSError, ValueError, http.client.HTTPException) as error:
            errors[endpoint.kind] = str(error)
    rows = []
    for case_id in manifest["required_cases"]:
        row = {"id": case_id, "status": "PASS", "observations": {}, "errors": {}}
        if case_id not in available:
            row.update(
                status="MISSING", errors={"coverage": "required case has no request"}
            )
        elif errors:
            row.update(status="BLOCKED", errors=dict(errors))
        else:
            for endpoint in endpoints:
                try:
                    row["observations"][endpoint.kind] = observe(
                        endpoint, manifest["model"], available[case_id]["request"]
                    )
                except (OSError, ValueError, http.client.HTTPException) as error:
                    row["status"] = "BLOCKED"
                    row["errors"][endpoint.kind] = str(error)
            if row["status"] == "PASS":
                difference = compare_tokens(
                    row["observations"]["dynamo"]["token_ids"],
                    row["observations"]["sglang"]["token_ids"],
                )
                if difference is not None:
                    row.update(status="FAIL", first_difference=difference)
        rows.append(row)
    return rows, models


def markdown_cell(value: Any) -> str:
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\r", " ")
        .replace("\n", " ")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def write_reports(output: Path, report: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    matrix = [
        "# Prompt-token compatibility matrix",
        "",
        SCOPE,
        "",
        LIMITATIONS,
        "",
        f"Run: {report['started_at']}; manifest SHA-256: {report.get('manifest_sha256', 'unavailable')}",
        "",
        f"Verdict: **{report['status']}**",
        "",
        "| Model | Profile | Case | Input token parity | Details |",
        "| --- | --- | --- | --- | --- |",
    ]
    html_rows = []
    for row in report["cases"]:
        detail = json.dumps(
            row.get("first_difference") or row["errors"], ensure_ascii=False
        )
        cells = [
            report.get("model", "unavailable"),
            report.get("profile", "unavailable"),
            row["id"],
            row["status"],
            detail,
        ]
        matrix.append("| " + " | ".join(markdown_cell(cell) for cell in cells) + " |")
        html_rows.append(
            "<tr>"
            + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in cells)
            + "</tr>"
        )
    matrix.extend(
        [
            "",
            "## Declared provenance (not verified)",
            "",
            "```json",
            json.dumps(
                report.get("declared_provenance", {}), indent=2, ensure_ascii=False
            ),
            "```",
        ]
    )
    if report.get("error"):
        matrix.extend(["", "Run error: " + markdown_cell(report["error"])])
    (output / "compatibility-matrix.md").write_text(
        "\n".join(matrix) + "\n", encoding="utf-8"
    )
    document = (
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Prompt-token compatibility</title>'
        "<style>body{font:16px system-ui;margin:2rem}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:.6rem;vertical-align:top}pre{white-space:pre-wrap}</style>"
        f"<h1>{html.escape(SCOPE)}</h1><p>{html.escape(LIMITATIONS)}</p>"
        f"<p>Verdict: <strong>{report['status']}</strong></p>"
        "<table><thead><tr><th>Model</th><th>Profile</th><th>Case</th><th>Input parity</th><th>Details</th></tr></thead><tbody>"
        + "".join(html_rows)
        + "</tbody></table><h2>Run evidence</h2><pre>"
        + html.escape(
            json.dumps(
                {key: value for key, value in report.items() if key != "cases"},
                indent=2,
                ensure_ascii=False,
            )
        )
        + "</pre></html>"
    )
    (output / "compatibility-matrix.html").write_text(document, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=SCOPE)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dynamo-url", required=True)
    parser.add_argument("--sglang-url", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--timeout",
        type=float,
        default=30,
        help="Per-socket-operation timeout in seconds",
    )
    args = parser.parse_args(argv)
    report = {
        "schema_version": 1,
        "scope": SCOPE,
        "limitations": LIMITATIONS,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "BLOCKED",
        "cases": [],
    }
    try:
        if not math.isfinite(args.timeout) or not 0 < args.timeout <= 300:
            raise ValueError("timeout must be finite and in (0, 300]")
        with args.manifest.open("rb") as stream:
            raw = stream.read(MAX_BODY + 1)
        if len(raw) > MAX_BODY:
            raise ValueError("manifest exceeds the 4 MiB input limit")
        report["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
        manifest = decode_json(raw)
        validate_manifest(manifest)
        report.update(
            profile=manifest["profile"],
            model=manifest["model"],
            manifest=manifest,
            declared_provenance=manifest["provenance"],
        )
        endpoints = [
            Endpoint(
                validate_url(args.dynamo_url),
                "dynamo",
                args.timeout,
                os.environ.get("DYNAMO_COMPAT_DYNAMO_API_KEY", ""),
            ),
            Endpoint(
                validate_url(args.sglang_url),
                "sglang",
                args.timeout,
                os.environ.get("DYNAMO_COMPAT_SGLANG_API_KEY", ""),
            ),
        ]
        if endpoints[0].base_url == endpoints[1].base_url:
            raise ValueError("Dynamo and SGLang must be distinct endpoints")
        report["endpoints"] = {
            endpoint.kind: endpoint.base_url for endpoint in endpoints
        }
        rows, observed = evaluate(manifest, endpoints)
        report.update(cases=rows, observed_models=observed)
        if rows and all(row["status"] == "PASS" for row in rows):
            report["status"] = "PASS"
        elif any(row["status"] in {"FAIL", "MISSING"} for row in rows):
            report["status"] = "FAIL"
    except (OSError, ValueError, http.client.HTTPException) as error:
        report["error"] = str(error)
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    try:
        write_reports(args.output, report)
    except OSError as error:
        print(f"Cannot write compatibility evidence: {error}", file=sys.stderr)
        return 2
    print(f"{report['status']}: {SCOPE}; evidence: {args.output / 'results.json'}")
    return 0 if report["status"] == "PASS" else 1
