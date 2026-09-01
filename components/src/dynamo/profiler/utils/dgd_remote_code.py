# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apply the profiler's remote-code trust policy to a DGD."""

from __future__ import annotations

import re
from collections.abc import Iterator

from dynamo.profiler.utils.config import get_main_container_dict
from dynamo.profiler.utils.model_info import (
    model_has_auto_map,
    model_ref_allows_implicit_trust_remote_code,
)

_TRUST_REMOTE_CODE_BACKENDS = frozenset({"vllm", "sglang"})
_TRUST_REMOTE_CODE_FLAG = "--trust-remote-code"
_TRUST_REMOTE_CODE_PATTERN = re.compile(
    rf"(?<!\S){re.escape(_TRUST_REMOTE_CODE_FLAG)}(?=\s|$)"
)
_WORKER_COMPONENT_TYPES = frozenset({"worker", "prefill", "decode"})
_BACKEND_SHELL_MARKERS = {
    "vllm": ("dynamo.vllm", "vllm.entrypoints"),
    "sglang": ("dynamo.sglang", "sglang.launch_server"),
}


def apply_remote_code_policy(
    config: dict,
    runtime_backend: str | None,
    model_name_or_path: str | None,
) -> dict:
    """Enable trusted model code when required and safe to infer."""
    if (
        runtime_backend not in _TRUST_REMOTE_CODE_BACKENDS
        or not model_name_or_path
        or not model_has_auto_map(model_name_or_path)
    ):
        return config

    if _all_workers_already_have_trust_flag(config, runtime_backend):
        return config
    if not model_ref_allows_implicit_trust_remote_code(model_name_or_path):
        raise RuntimeError(
            "Refusing to auto-inject --trust-remote-code for mutable remote "
            f"model ref {model_name_or_path!r}. Set --trust-remote-code "
            "explicitly via overrides if this ref is intended."
        )

    _inject_trust_remote_code_flag(config, runtime_backend)
    return config


def _all_workers_already_have_trust_flag(config: dict, runtime_backend: str) -> bool:
    """Return True when every real worker carries --trust-remote-code."""
    workers_seen = False
    for main_container in _worker_main_containers(config):
        workers_seen = True
        args = main_container.get("args") or []
        command = main_container.get("command") or []
        if _is_mocker_container(command, args):
            continue
        if _is_shell_command(command, args):
            if not _shell_backend_has_trust_flag(args[0], runtime_backend):
                return False
        elif _TRUST_REMOTE_CODE_FLAG not in args:
            return False
    return workers_seen


def _inject_trust_remote_code_flag(config: dict, runtime_backend: str) -> None:
    """Append --trust-remote-code to real workers without changing CLI form."""
    for main_container in _worker_main_containers(config):
        args = main_container.get("args") or []
        command = main_container.get("command") or []
        if _is_mocker_container(command, args):
            continue
        if _is_shell_command(command, args):
            if not _shell_backend_has_trust_flag(args[0], runtime_backend):
                main_container["args"] = [
                    _insert_shell_backend_trust_flag(args[0], runtime_backend)
                ]
        elif _TRUST_REMOTE_CODE_FLAG not in args:
            main_container["args"] = [*args, _TRUST_REMOTE_CODE_FLAG]


def _shell_backend_segment(command: str, runtime_backend: str) -> tuple[int, int]:
    """Return the backend command's offsets within a shell script."""
    marker_indexes = [
        command.find(marker)
        for marker in _BACKEND_SHELL_MARKERS[runtime_backend]
        if marker in command
    ]
    if not marker_indexes:
        return 0, len(command.rstrip())

    start = min(marker_indexes)
    index = start
    quote: str | None = None
    while index < len(command):
        char = command[index]
        if quote is not None:
            if char == quote:
                quote = None
                index += 1
            elif quote == '"' and char == "\\" and index + 1 < len(command):
                index += 2
            else:
                index += 1
            continue

        if char in ("'", '"'):
            quote = char
            index += 1
        elif char == "\\" and index + 1 < len(command):
            index += 2
        elif char == "&" and index > 0 and command[index - 1] in "<>":
            index += 1
        elif char in ";|&\n":
            return start, index
        else:
            index += 1

    return start, len(command.rstrip())


def _shell_backend_has_trust_flag(command: str, runtime_backend: str) -> bool:
    """Report whether the backend command segment carries the trust flag."""
    start, end = _shell_backend_segment(command, runtime_backend)
    return _TRUST_REMOTE_CODE_PATTERN.search(command[start:end]) is not None


def _insert_shell_backend_trust_flag(command: str, runtime_backend: str) -> str:
    """Insert the trust flag before the backend command's shell operator."""
    _, end = _shell_backend_segment(command, runtime_backend)
    prefix = command[:end].rstrip()
    suffix = command[end:]
    trailing_space = " " if suffix and not suffix[0].isspace() else ""
    return f"{prefix} {_TRUST_REMOTE_CODE_FLAG}{trailing_space}{suffix}"


def _worker_main_containers(config: dict) -> Iterator[dict]:
    components = config.get("spec", {}).get("components", [])
    if not isinstance(components, list):
        return
    for component in components:
        if (
            not isinstance(component, dict)
            or component.get("type") not in _WORKER_COMPONENT_TYPES
        ):
            continue
        main_container = get_main_container_dict(component)
        if main_container is not None:
            yield main_container


def _is_mocker_container(command: object, args: object) -> bool:
    if not isinstance(command, list) or not isinstance(args, list):
        return False
    return "dynamo.mocker" in " ".join(str(token) for token in [*command, *args])


def _is_shell_command(command: object, args: object) -> bool:
    return (
        isinstance(command, list)
        and len(command) >= 2
        and command[0] in ("/bin/sh", "sh")
        and command[1] == "-c"
        and isinstance(args, list)
        and len(args) == 1
        and isinstance(args[0], str)
    )
