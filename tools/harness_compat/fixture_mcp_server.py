#!/usr/bin/env python3
"""Minimal stdio MCP fixture for native coding-harness compatibility probes.

The server intentionally exposes one no-argument tool and never logs request
content. It is started only from an artifact-local client configuration.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any


_trace_path = Path(os.environ["DYNAMO_COMPAT_FIXTURE_MCP_TRACE"]) if "DYNAMO_COMPAT_FIXTURE_MCP_TRACE" in os.environ else None
_trace: dict[str, Any] = {}


def _record_trace(**fields: Any) -> None:
    """Persist only MCP framing metadata for a fixture-debug artifact."""
    if _trace_path is None:
        return
    _trace.update(fields)
    _trace_path.write_text(json.dumps(_trace, sort_keys=True) + "\n", encoding="utf-8")


def _send(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _result(request_id: int | str, result: dict[str, Any]) -> None:
    _send({"jsonrpc": "2.0", "id": request_id, "result": result})


def _error(request_id: int | str, code: int, message: str) -> None:
    _send({"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}})


def _elicit() -> bool:
    """Ask the MCP client for one fixed form value without logging its response."""
    elicitation_id = "fixture-elicitation-1"
    mode = "openai/form" if os.environ.get("DYNAMO_COMPAT_FIXTURE_MCP_OPENAI_FORM") == "1" else "form"
    _send(
        {
            "jsonrpc": "2.0",
            "id": elicitation_id,
            "method": "elicitation/create",
            "params": {
                "mode": mode,
                "message": "Compatibility fixture confirmation.",
                "requestedSchema": {
                    "type": "object",
                    "properties": {"choice": {"type": "string", "enum": ["CONTINUE"]}},
                    "required": ["choice"],
                },
            },
        }
    )
    _record_trace(elicitation_mode=mode, elicitation_sent=True)
    for line in sys.stdin:
        try:
            response = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(response, dict) or response.get("id") != elicitation_id:
            continue
        result = response.get("result")
        _record_trace(
            elicitation_response_action=result.get("action") if isinstance(result, dict) else None,
            elicitation_response_error_code=(
                response.get("error", {}).get("code") if isinstance(response.get("error"), dict) else None
            ),
            elicitation_response_result_keys=sorted(result) if isinstance(result, dict) else [],
        )
        return isinstance(result, dict) and result.get("action") == "accept"
    return False


def _progress(params: Any) -> None:
    metadata = params.get("_meta") if isinstance(params, dict) else None
    token = metadata.get("progressToken") if isinstance(metadata, dict) else None
    _record_trace(progress_token_present=token is not None)
    if token is not None:
        _send(
            {
                "jsonrpc": "2.0",
                "method": "notifications/progress",
                "params": {"progressToken": token, "progress": 1, "total": 1},
            }
        )
        _record_trace(progress_sent=True)
        time.sleep(0.05)


def main() -> int:
    failure_mode = os.environ.get("DYNAMO_COMPAT_FIXTURE_MCP_FAIL") == "1"
    elicitation_mode = os.environ.get("DYNAMO_COMPAT_FIXTURE_MCP_ELICIT") == "1"
    progress_mode = os.environ.get("DYNAMO_COMPAT_FIXTURE_MCP_PROGRESS") == "1"
    tool_name = "fixture_failure" if failure_mode else "fixture_elicitation" if elicitation_mode else "fixture_answer"
    tool_description = (
        "Return the fixed compatibility fixture error."
        if failure_mode
        else "Elicit one fixed form response and return the compatibility fixture answer."
        if elicitation_mode
        else "Return the fixed compatibility fixture answer."
    )
    for line in sys.stdin:
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(request, dict):
            continue
        method = request.get("method")
        request_id = request.get("id")
        if not isinstance(method, str):
            continue
        if method == "notifications/initialized":
            continue
        if not isinstance(request_id, (int, str)) or isinstance(request_id, bool):
            continue
        if method == "initialize":
            params = request.get("params")
            capabilities = params.get("capabilities") if isinstance(params, dict) else None
            elicitation_capabilities = capabilities.get("elicitation") if isinstance(capabilities, dict) else None
            _record_trace(
                initialize_capability_keys=sorted(capabilities) if isinstance(capabilities, dict) else [],
                initialize_elicitation_keys=(
                    sorted(elicitation_capabilities) if isinstance(elicitation_capabilities, dict) else []
                ),
                initialize_protocol_version=params.get("protocolVersion") if isinstance(params, dict) else None,
            )
            negotiated_protocol = (
                params.get("protocolVersion")
                if elicitation_mode and isinstance(params, dict) and isinstance(params.get("protocolVersion"), str)
                else "2025-03-26"
            )
            _result(
                request_id,
                {
                    "protocolVersion": negotiated_protocol,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "dynamo-compat-fixture", "version": "1"},
                },
            )
        elif method == "tools/list":
            _result(
                request_id,
                {
                    "tools": [
                        {
                            "name": tool_name,
                            "description": tool_description,
                            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
                        }
                    ]
                },
            )
        elif method == "tools/call":
            params = request.get("params")
            _record_trace(tool_call_received=True)
            if isinstance(params, dict) and params.get("name") == tool_name:
                if progress_mode:
                    _progress(params)
                if failure_mode:
                    _result(
                        request_id,
                        {"content": [{"type": "text", "text": "fixture unavailable"}], "isError": True},
                    )
                elif elicitation_mode and not _elicit():
                    _result(
                        request_id,
                        {"content": [{"type": "text", "text": "fixture elicitation declined"}], "isError": True},
                    )
                else:
                    _result(request_id, {"content": [{"type": "text", "text": "42"}], "isError": False})
            else:
                _result(
                    request_id,
                    {"content": [{"type": "text", "text": "fixture tool unavailable"}], "isError": True},
                )
        elif method == "ping":
            _result(request_id, {})
        else:
            _error(request_id, -32601, "method not found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
