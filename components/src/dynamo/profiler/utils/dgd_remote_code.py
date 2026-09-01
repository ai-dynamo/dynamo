# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apply the profiler's remote-code trust policy to a DGD."""

from __future__ import annotations

from collections.abc import Iterator

from dynamo.profiler.utils.config import get_main_container_dict
from dynamo.profiler.utils.model_info import (
    model_has_auto_map,
    model_ref_allows_implicit_trust_remote_code,
)

_TRUST_REMOTE_CODE_BACKENDS = frozenset({"vllm", "sglang"})
_TRUST_REMOTE_CODE_FLAG = "--trust-remote-code"
_WORKER_COMPONENT_TYPES = frozenset({"worker", "prefill", "decode"})


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

    if _all_workers_already_have_trust_flag(config):
        return config
    if not model_ref_allows_implicit_trust_remote_code(model_name_or_path):
        raise RuntimeError(
            "Refusing to auto-inject --trust-remote-code for mutable remote "
            f"model ref {model_name_or_path!r}. Set --trust-remote-code "
            "explicitly via overrides if this ref is intended."
        )

    _inject_trust_remote_code_flag(config)
    return config


def _all_workers_already_have_trust_flag(config: dict) -> bool:
    """Return True when every real worker carries --trust-remote-code."""
    workers_seen = False
    for main_container in _worker_main_containers(config):
        workers_seen = True
        args = main_container.get("args") or []
        command = main_container.get("command") or []
        if _is_mocker_container(command, args):
            continue
        if _is_shell_command(command, args):
            if _TRUST_REMOTE_CODE_FLAG not in args[0]:
                return False
        elif _TRUST_REMOTE_CODE_FLAG not in args:
            return False
    return workers_seen


def _inject_trust_remote_code_flag(config: dict) -> None:
    """Append --trust-remote-code to real workers without changing CLI form."""
    for main_container in _worker_main_containers(config):
        args = main_container.get("args") or []
        command = main_container.get("command") or []
        if _is_mocker_container(command, args):
            continue
        if _is_shell_command(command, args):
            if _TRUST_REMOTE_CODE_FLAG not in args[0]:
                main_container["args"] = [f"{args[0]} {_TRUST_REMOTE_CODE_FLAG}"]
        elif _TRUST_REMOTE_CODE_FLAG not in args:
            main_container["args"] = [*args, _TRUST_REMOTE_CODE_FLAG]


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
