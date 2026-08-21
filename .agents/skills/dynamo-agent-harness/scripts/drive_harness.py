#!/usr/bin/env -S uv run --script
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["agent-client-protocol==0.12.0"]
# ///
"""Drive a persistent coding-agent harness through a Dynamo endpoint over ACP."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest
from urllib.parse import urlsplit

from acp import PROTOCOL_VERSION, spawn_agent_process, text_block

COMMANDS = {
    "claude": ("npx", "-y", "@agentclientprotocol/claude-agent-acp@0.66.0"),
    "codex": ("npx", "-y", "@agentclientprotocol/codex-acp@1.1.14"),
    "opencode": ("opencode", "acp", "--pure"),
}

MODES = {
    ("claude", "verify"): "plan",
    ("claude", "act"): "bypassPermissions",
    ("codex", "verify"): "read-only",
    ("codex", "act"): "agent",
    ("opencode", "verify"): "plan",
    ("opencode", "act"): "build",
}


@dataclass(frozen=True)
class HarnessConfig:
    command: tuple[str, ...]
    environment: dict[str, str]
    gateway_url: str | None
    openai_url: str
    api_key: str
    mode: str
    model: str
    session_model: str


class HarnessClient:
    """Collect text responses and enforce the requested capability boundary."""

    def __init__(self, capability: str):
        self.capability = capability
        self.parts: list[str] = []

    async def request_permission(
        self,
        session_id: str,
        tool_call: Any,
        options: list[Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        del session_id, tool_call, kwargs
        if self.capability == "act":
            for option in options:
                if option.kind == "allow_once":
                    return {
                        "outcome": {
                            "outcome": "selected",
                            "optionId": option.option_id,
                        }
                    }
        return {"outcome": {"outcome": "cancelled"}}

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        del session_id, kwargs
        if update.session_update != "agent_message_chunk":
            return
        if update.content.type == "text":
            self.parts.append(update.content.text)

    def start_turn(self) -> None:
        self.parts = []

    def response(self) -> str:
        return "".join(self.parts).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", choices=sorted(COMMANDS), required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--add-dir", action="append", default=[], type=Path)
    parser.add_argument("--capability", choices=("verify", "act"), default="verify")
    parser.add_argument("--api-key-env", default="DYNAMO_API_KEY")
    parser.add_argument(
        "--session-final",
        action="store_true",
        help="send a terminal ThunderAgent lifecycle signal when a Codex session closes",
    )
    parser.add_argument("--session-final-timeout", type=float, default=5.0)
    return parser.parse_args()


def normalize_base_url(value: str) -> tuple[str, str]:
    root = value.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    parsed = urlsplit(root)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("--base-url must be an absolute HTTP(S) URL")
    return root, f"{root}/v1"


def validate_paths(cwd: Path, additional: list[Path]) -> tuple[Path, list[Path]]:
    resolved_cwd = cwd.resolve()
    if not resolved_cwd.is_dir():
        raise ValueError(f"--cwd is not a directory: {resolved_cwd}")
    resolved_additional = [path.resolve() for path in additional]
    missing = next((path for path in resolved_additional if not path.is_dir()), None)
    if missing is not None:
        raise ValueError(f"--add-dir is not a directory: {missing}")
    return resolved_cwd, resolved_additional


def build_config(args: argparse.Namespace) -> HarnessConfig:
    if not args.model.strip():
        raise ValueError("--model must not be empty")
    root_url, openai_url = normalize_base_url(args.base_url)
    environment = dict(os.environ)
    api_key = environment.get(args.api_key_env) or "dummy"
    gateway_url = None
    session_model = args.model

    if args.harness == "claude":
        gateway_url = root_url
    elif args.harness == "codex":
        gateway_url = openai_url
        codex_config_raw = environment.get("CODEX_CONFIG", "{}")
        codex_config = json.loads(codex_config_raw)
        if not isinstance(codex_config, dict):
            raise ValueError("CODEX_CONFIG must contain a JSON object")
        codex_config.update(
            model=args.model,
            model_reasoning_effort="medium",
        )
        environment["CODEX_CONFIG"] = json.dumps(codex_config)
        environment["NO_BROWSER"] = "1"
    else:
        provider = "dynamo-acp"
        session_model = f"{provider}/{args.model}"
        environment["OPENCODE_CONFIG_CONTENT"] = json.dumps(
            {
                "model": session_model,
                "enabled_providers": [provider],
                "provider": {
                    provider: {
                        "npm": "@ai-sdk/openai-compatible",
                        "name": "Dynamo",
                        "options": {
                            "baseURL": openai_url,
                            "apiKey": api_key,
                        },
                        "models": {
                            args.model: {
                                "name": args.model,
                            }
                        },
                    }
                },
            }
        )

    return HarnessConfig(
        command=COMMANDS[args.harness],
        environment=environment,
        gateway_url=gateway_url,
        openai_url=openai_url,
        api_key=api_key,
        mode=MODES[(args.harness, args.capability)],
        model=args.model,
        session_model=session_model,
    )


async def relay_stderr(process: Any) -> None:
    if process.stderr is None:
        return
    while line := await process.stderr.readline():
        sys.stderr.write(line.decode(errors="replace"))
        sys.stderr.flush()


async def configure_mode(conn: Any, session: Any, mode: str) -> None:
    available_modes = {
        entry.id for entry in (session.modes.available_modes if session.modes else [])
    }
    if mode in available_modes:
        await conn.set_session_mode(session_id=session.session_id, mode_id=mode)
        return

    mode_option = next(
        (option for option in (session.config_options or []) if option.id == "mode"),
        None,
    )
    mode_values = {
        option.value for option in (mode_option.options if mode_option else [])
    }
    if mode not in mode_values:
        raise RuntimeError(
            f"ACP agent does not expose mode {mode!r}; available: "
            f"{sorted(available_modes | mode_values)}"
        )
    await conn.set_config_option(
        session_id=session.session_id,
        config_id="mode",
        value=mode,
    )


def validate_model(harness: str, session: Any, expected: str) -> None:
    # Claude accepts custom gateway models through claudeCode.options but reports
    # its ACP model config as "default". The request trace is authoritative.
    if harness == "claude":
        return
    model_option = next(
        (option for option in (session.config_options or []) if option.id == "model"),
        None,
    )
    if model_option is not None and model_option.current_value != expected:
        raise RuntimeError(
            f"ACP agent selected {model_option.current_value!r}, expected {expected!r}"
        )


async def prompt(conn: Any, client: HarnessClient, session_id: str, text: str) -> None:
    client.start_turn()
    result = await conn.prompt(session_id=session_id, prompt=[text_block(text)])
    response = client.response()
    if not response:
        emit(
            {
                "type": "error",
                "session_id": session_id,
                "ok": False,
                "error": "agent returned no text response",
            }
        )
        return
    output: dict[str, Any] = {
        "type": "response",
        "session_id": session_id,
        "ok": True,
        "response": response,
        "stop_reason": result.stop_reason,
    }
    if result.usage is not None:
        output["usage"] = result.usage.model_dump(by_alias=True, exclude_none=True)
    emit(output)


def emit(value: Any) -> None:
    print(json.dumps(value, separators=(",", ":")), flush=True)


def post_session_final(
    config: HarnessConfig,
    session_id: str,
    timeout: float,
) -> None:
    """Terminate one ThunderAgent program without forwarding work to the engine."""
    payload = json.dumps(
        {
            "model": config.model,
            "messages": [{"role": "user", "content": "."}],
            "max_tokens": 1,
            "stream": False,
        }
    ).encode()
    request = urlrequest.Request(
        f"{config.openai_url}/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
            "x-dynamo-session-id": session_id,
            "x-dynamo-session-final": "true",
        },
        method="POST",
    )
    try:
        with urlrequest.urlopen(request, timeout=timeout) as response:
            status = response.status
            response.read()
    except urlerror.URLError as error:
        raise RuntimeError(f"session-final request failed: {error.reason}") from error
    if not 200 <= status < 300:
        raise RuntimeError(f"session-final request returned HTTP {status}")


async def send_session_final(
    config: HarnessConfig,
    session_id: str,
    timeout: float,
) -> None:
    try:
        await asyncio.to_thread(post_session_final, config, session_id, timeout)
    except Exception as error:
        emit(
            {
                "type": "session_final",
                "session_id": session_id,
                "ok": False,
                "error": str(error),
            }
        )
        raise
    emit({"type": "session_final", "session_id": session_id, "ok": True})


async def run(args: argparse.Namespace) -> None:
    cwd, additional = validate_paths(args.cwd, args.add_dir)
    if args.session_final and args.harness != "codex":
        raise ValueError(
            "--session-final is currently verified only with --harness codex"
        )
    if args.session_final_timeout <= 0:
        raise ValueError("--session-final-timeout must be greater than zero")
    config = build_config(args)
    if shutil.which(config.command[0]) is None:
        raise FileNotFoundError(f"executable not found: {config.command[0]}")

    client = HarnessClient(args.capability)
    async with spawn_agent_process(
        client,
        *config.command,
        cwd=str(cwd),
        env=config.environment,
    ) as (conn, process):
        stderr_task = asyncio.create_task(relay_stderr(process))
        try:
            capabilities = (
                {"auth": {"_meta": {"gateway": True}}} if config.gateway_url else None
            )
            initialized = await conn.initialize(
                protocol_version=PROTOCOL_VERSION,
                client_capabilities=capabilities,
            )
            if config.gateway_url:
                api_key = config.environment.get(args.api_key_env) or "dummy"
                headers = (
                    {"x-api-key": api_key}
                    if args.harness == "claude"
                    else {"Authorization": f"Bearer {api_key}"}
                )
                gateway = {
                    "baseUrl": config.gateway_url,
                    "headers": headers,
                }
                if args.harness == "codex":
                    gateway["providerName"] = "Dynamo"
                await conn.authenticate(
                    method_id="gateway",
                    gateway=gateway,
                )

            session_metadata = (
                {"claudeCode": {"options": {"model": config.model}}}
                if args.harness == "claude"
                else {}
            )
            session = await conn.new_session(
                cwd=str(cwd),
                additional_directories=[str(path) for path in additional],
                mcp_servers=[],
                **session_metadata,
            )
            await configure_mode(conn, session, config.mode)
            validate_model(args.harness, session, config.session_model)
            agent_name = (
                initialized.agent_info.name
                if initialized.agent_info is not None
                else args.harness
            )
            session_id = str(session.session_id)
            emit(
                {
                    "type": "ready",
                    "harness": args.harness,
                    "agent": agent_name,
                    "session_id": session_id,
                    "mode": config.mode,
                    "model": config.model,
                }
            )

            while line := await asyncio.to_thread(sys.stdin.readline):
                try:
                    request = json.loads(line)
                    if not isinstance(request, dict):
                        raise ValueError("expected one JSON object per line")
                    if request.get("close") is True:
                        break
                    prompt_text = request.get("prompt")
                    if not isinstance(prompt_text, str) or not prompt_text.strip():
                        raise ValueError('expected {"prompt":"..."} or {"close":true}')
                except (json.JSONDecodeError, ValueError) as error:
                    emit({"type": "error", "ok": False, "error": str(error)})
                    continue
                await prompt(conn, client, session_id, prompt_text)
            if args.session_final:
                await send_session_final(
                    config,
                    session_id,
                    args.session_final_timeout,
                )
        finally:
            stderr_task.cancel()


def main() -> int:
    args = parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
