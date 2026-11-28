# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Temporary checks of explicit launch flags, not effective configuration.

Retune's binding supplies startup settings. Until the frontend validates its
effective configuration, retain checks of other literal command/args options.
Never inspect env, expand references, split shell scripts or open policy files.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from typing import Any, NoReturn

from dynamo.common.configuration.groups.kv_router_args import (
    _KV_ROUTER_FIELDS,
    KvRouterArgGroup,
)
from dynamo.common.configuration.groups.router_args import (
    RouterArgGroup,
    add_worker_router_arguments,
)
from dynamo.common.configuration.utils import ArgEnv

# These settings come from the operator binding, not authored launch syntax.
_INJECTED = frozenset(
    {
        "router_mode",
        "router_replica_sync",
        "overlap_score_credit",
        "prefill_load_scale",
        "router_temperature",
    }
)
_TOLERATED = frozenset(
    {
        "router_event_threads",
        "router_tracking_hash",
        "router_tracking_key_file",
        "router_tracking_key_id",
        "serve_indexer",
        "enable_session_prefix_index",
        "skip_initial_worker_wait",
    }
)
_ADMISSION = (
    "session_affinity_ttl_secs",
    "active_decode_blocks_threshold",
    "active_prefill_tokens_threshold",
    "active_prefill_tokens_threshold_frac",
)
_OTHER = ("load_aware", "conditional_disagg_config", "tls_cert_path", "tls_key_path")


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise ValueError(message)


def launch_flag_errors(components: Sequence[Mapping[str, Any]]) -> list[str]:
    """Reject visible unsupported flags without deriving process settings."""

    parser = _Parser(add_help=False)
    empty_env = ArgEnv(environ={})
    RouterArgGroup(
        default_router_mode=None, include_frontend_only=True, env=empty_env
    ).add_arguments(parser)
    KvRouterArgGroup(env=empty_env).add_arguments(parser)
    parser.add_argument("--tls-cert-path")
    parser.add_argument("--tls-key-path")
    worker_parser = _Parser(add_help=False)
    add_worker_router_arguments(worker_parser, env=empty_env)
    defaults = vars(parser.parse_args([]))
    checked = (
        ((set(_KV_ROUTER_FIELDS) | set(_ADMISSION) | set(_OTHER)) & defaults.keys())
        - _INJECTED
        - _TOLERATED
    )
    errors: list[str] = []

    for component in components:
        component_type = str(component.get("type") or "")
        if component_type not in {"frontend", "worker", "prefill", "decode", ""}:
            continue
        templates = _templates(component)

        # A native runtime sidecar, not its engine main, advertises the worker.
        if component_type != "frontend" and any(
            container.get("name") == "runtime"
            for role, pod in templates
            if role in {"", "leader"}
            for container in pod.get("initContainers") or []
            if isinstance(container, Mapping)
        ):
            continue
        for role, pod in templates:
            main = next(
                (
                    container
                    for container in pod.get("containers") or []
                    if isinstance(container, Mapping)
                    and container.get("name") == "main"
                ),
                {},
            )
            tokens = [*(main.get("command") or []), *(main.get("args") or [])]
            label = str(component.get("name", ""))
            if role:
                label += f" (role {role})"
            if component_type != "frontend":
                if any(
                    (action := _known_option(token.split("=", 1)[0], worker_parser))
                    is not None
                    and action.dest == "router_mode"
                    for token in tokens
                    if isinstance(token, str)
                ):
                    errors.append(
                        f"P4: worker component {label} sets --router-mode; its router "
                        "configuration replaces the frontend's for its workers"
                    )
                continue
            try:
                parsed = vars(
                    parser.parse_args(_literal_flags(tokens, parser, checked))
                )
            except ValueError as error:
                errors.append(
                    f"P4: frontend component {label} has invalid explicit router flags: {error}"
                )
                continue
            for field in sorted(checked):
                if parsed[field] != defaults[field]:
                    errors.append(
                        f"P4: frontend component {label} sets {field}={parsed[field]!r}; "
                        "live retune requires its default for replay"
                    )
    return errors


def _literal_flags(
    tokens: Sequence[Any], parser: argparse.ArgumentParser, checked: set[str]
) -> list[str]:
    """Select literal known options; references remain uninspected strings."""

    result: list[str] = []
    for index, token in enumerate(tokens):
        if not isinstance(token, str):
            continue
        flag, equal, value = token.partition("=")
        action = _known_option(flag, parser)
        if action is None or action.dest not in checked:
            continue
        if action.nargs == 0:
            result.append(token)
            continue
        if not equal:
            value = str(tokens[index + 1]) if index + 1 < len(tokens) else ""
        if "$" in value:
            continue
        result.append(token)
        if not equal and index + 1 < len(tokens) and not value.startswith("--"):
            result.append(value)
    return result


def _known_option(flag: str, parser: argparse.ArgumentParser) -> argparse.Action | None:
    action = parser._option_string_actions.get(flag)
    if action is not None or not flag.startswith("--"):
        return action
    matches = [
        action
        for option, action in parser._option_string_actions.items()
        if option.startswith(flag)
    ]
    return matches[0] if len(matches) == 1 else None


def _templates(component: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    template = component.get("podTemplate")
    if isinstance(template, Mapping):
        return [("", template.get("spec") or {})]
    return [
        (str(role.get("name") or ""), template.get("spec") or {})
        for role in component.get("roles") or []
        if isinstance(role, Mapping)
        and isinstance(template := role.get("podTemplate"), Mapping)
    ]
