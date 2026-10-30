# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""First-increment FE/EPP effective-config contract for issue #16592.

Inventory is fail-closed and knowingly incomplete: only the three live
profiles have explicit decisions. Discovered CLI/env/schema names without a
row fail the completeness guard. Do not treat that failure as full-issue
completion.

The EPP observer reads the post-parse INFO event from
``log_env_config``. It does not observe later Runtime EPP instance overrides
(``skip_initial_worker_wait=true``,
``prefill_config.router_track_active_blocks=false``). The nine scalar
profile fields are not those overrides.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

# Compiled KvRouterConfig and FrontendArgGroup stay function-local so this
# helper can be imported during pytest collection without pulling native or
# frontend runtime. That is optional-runtime test selection, not a hidden
# missing-dependency fallback: a selected function still imports directly.

REPO_ROOT = Path(__file__).resolve().parents[2]
ROUTER_ARGS = (
    REPO_ROOT / "components/src/dynamo/common/configuration/groups/router_args.py"
)
KV_ROUTER_ARGS = (
    REPO_ROOT / "components/src/dynamo/common/configuration/groups/kv_router_args.py"
)
FRONTEND_ARGS = REPO_ROOT / "components/src/dynamo/frontend/frontend_args.py"
RUST_CONFIG = REPO_ROOT / "lib/kv-router/src/scheduling/config.rs"
RUST_TREES = (
    REPO_ROOT / "lib/kv-router/src",
    REPO_ROOT / "lib/llm/src",
)
FRONTEND_ROUTING_DESTS = frozenset({"serve_indexer", "kv_cache_block_size"})
INIT_EVENT = "KvRouterConfig initialized (DYN_* env overrides applied)"
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
FIELD_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=(\S+)")
ENV_LITERAL_RE = re.compile(
    r"DYN_(?:ROUTER_[A-Z0-9_]+|ACTIVE_[A-Z0-9_]+|SHARED_CACHE_[A-Z0-9_]+|"
    r"OVERLAP_SCORE_WEIGHT|USE_KV_EVENTS|USE_REMOTE_INDEXER|"
    r"ENABLE_SESSION_PREFIX_INDEX|ENFORCE_DISAGG|ADMISSION_CONTROL|"
    r"SERVE_INDEXER|KV_CACHE_BLOCK_SIZE)"
)
WRAPPER_CLI = ("--router-mode", "kv", "--kv-cache-block-size", "16")
WRAPPER_MIN_WORKERS = "1"
PROFILE_NAMES = ("scalar_kv", "legacy_precedence", "policy_yaml_over_env")
IMPLEMENTED_OBSERVER = "fe_native_to_json+epp_init_event"
_POLICY_PARAMETER_ENVS = {
    "DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT": "overlap_score_credit",
    "DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT_DECAY": "overlap_score_credit_decay",
    "DYN_ROUTER_PREFILL_LOAD_SCALE": "prefill_load_scale",
    "DYN_ROUTER_DECODE_ACTIVE_REQUEST_WEIGHT": "decode_active_request_weight",
    "DYN_ROUTER_TEMPERATURE": "router_temperature",
    "DYN_ROUTER_KV_OVERLAP_SCORE_WEIGHT": "prefill_load_scale",
}
PROFILE_ASSERTED_FIELDS = {
    "scalar_kv": (
        "overlap_score_credit",
        "overlap_score_credit_decay",
        "prefill_load_scale",
        "decode_active_request_weight",
        "router_temperature",
        "host_cache_hit_weight",
        "disk_cache_hit_weight",
        "router_ttl_secs",
        "router_event_threads",
    ),
    "legacy_precedence": (
        "overlap_score_credit",
        "prefill_load_scale",
        "use_kv_events",
    ),
    "policy_yaml_over_env": (
        "use_kv_events",
        "router_event_threads",
        "router_ttl_secs",
    ),
}
PROFILE_DEPRECATED_ENVS = {
    "scalar_kv": (
        "DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT",
        "DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT_DECAY",
        "DYN_ROUTER_PREFILL_LOAD_SCALE",
        "DYN_ROUTER_DECODE_ACTIVE_REQUEST_WEIGHT",
        "DYN_ROUTER_TEMPERATURE",
    ),
    "legacy_precedence": (
        "DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT",
        "DYN_ROUTER_PREFILL_LOAD_SCALE",
        "DYN_ROUTER_KV_OVERLAP_SCORE_WEIGHT",
    ),
    "policy_yaml_over_env": (),
}

STATUS_PENDING = "pending"
STATUS_SUPPORTED = "supported_candidate"
STATUS_ALIAS = "alias"


@dataclass(frozen=True)
class SettingDecision:
    """Reviewed decision for one logical setting exercised or required here."""

    name: str
    flags: tuple[str, ...]
    env: tuple[str, ...]
    schema_fields: tuple[str, ...]
    status: str
    observer: str | None
    profiles: tuple[str, ...]
    affected_fields: tuple[str, ...]
    reason: str


# Explicit decisions for this increment only. Do not derive rows from discovery.
SETTING_DECISIONS: tuple[SettingDecision, ...] = (
    SettingDecision(
        "overlap_score_credit",
        ("--router-kv-overlap-score-credit",),
        ("DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT",),
        ("overlap_score_credit",),
        STATUS_SUPPORTED,
        IMPLEMENTED_OBSERVER,
        ("scalar_kv", "legacy_precedence"),
        ("overlap_score_credit",),
        "Candidate for FE to_json and the EPP init event on the named profiles.",
    ),
    SettingDecision(
        "overlap_score_credit_decay",
        ("--router-kv-overlap-score-credit-decay",),
        ("DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT_DECAY",),
        ("overlap_score_credit_decay",),
        STATUS_SUPPORTED,
        IMPLEMENTED_OBSERVER,
        ("scalar_kv",),
        ("overlap_score_credit_decay",),
        "Candidate for FE to_json and the EPP init event on scalar_kv.",
    ),
    SettingDecision(
        "prefill_load_scale",
        ("--router-prefill-load-scale",),
        ("DYN_ROUTER_PREFILL_LOAD_SCALE",),
        ("prefill_load_scale",),
        STATUS_SUPPORTED,
        IMPLEMENTED_OBSERVER,
        ("scalar_kv", "legacy_precedence"),
        ("prefill_load_scale",),
        "Candidate for both paths. Legacy overlap-weight aliases rewrite this field.",
    ),
    SettingDecision(
        "decode_active_request_weight",
        ("--router-decode-active-request-weight",),
        ("DYN_ROUTER_DECODE_ACTIVE_REQUEST_WEIGHT",),
        ("decode_active_request_weight",),
        STATUS_SUPPORTED,
        IMPLEMENTED_OBSERVER,
        ("scalar_kv",),
        ("decode_active_request_weight",),
        "Candidate non-default value for FE to_json and the EPP init event.",
    ),
    SettingDecision(
        "router_temperature",
        ("--router-temperature",),
        ("DYN_ROUTER_TEMPERATURE",),
        ("router_temperature",),
        STATUS_SUPPORTED,
        IMPLEMENTED_OBSERVER,
        ("scalar_kv",),
        ("router_temperature",),
        "Candidate for FE to_json and the EPP init event on scalar_kv.",
    ),
    SettingDecision(
        "host_cache_hit_weight",
        ("--router-host-cache-hit-weight",),
        ("DYN_ROUTER_HOST_CACHE_HIT_WEIGHT",),
        ("host_cache_hit_weight",),
        STATUS_SUPPORTED,
        IMPLEMENTED_OBSERVER,
        ("scalar_kv",),
        ("host_cache_hit_weight",),
        "Candidate for FE to_json and the EPP init event on scalar_kv.",
    ),
    SettingDecision(
        "disk_cache_hit_weight",
        ("--router-disk-cache-hit-weight",),
        ("DYN_ROUTER_DISK_CACHE_HIT_WEIGHT",),
        ("disk_cache_hit_weight",),
        STATUS_SUPPORTED,
        IMPLEMENTED_OBSERVER,
        ("scalar_kv",),
        ("disk_cache_hit_weight",),
        "Candidate for FE to_json and the EPP init event on scalar_kv.",
    ),
    SettingDecision(
        "router_ttl_secs",
        ("--router-ttl-secs",),
        ("DYN_ROUTER_TTL_SECS",),
        ("router_ttl_secs",),
        STATUS_SUPPORTED,
        IMPLEMENTED_OBSERVER,
        ("scalar_kv", "policy_yaml_over_env"),
        ("router_ttl_secs",),
        "Candidate scalar and YAML-over-env winner on the named profiles.",
    ),
    SettingDecision(
        "router_event_threads",
        ("--router-event-threads",),
        ("DYN_ROUTER_EVENT_THREADS",),
        ("router_event_threads",),
        STATUS_SUPPORTED,
        IMPLEMENTED_OBSERVER,
        ("scalar_kv", "policy_yaml_over_env"),
        ("router_event_threads",),
        "Candidate scalar and YAML-over-env winner on the named profiles.",
    ),
    SettingDecision(
        "overlap_score_weight",
        ("--router-kv-overlap-score-weight", "--kv-overlap-score-weight"),
        ("DYN_ROUTER_KV_OVERLAP_SCORE_WEIGHT", "DYN_OVERLAP_SCORE_WEIGHT"),
        ("overlap_score_weight",),
        STATUS_ALIAS,
        IMPLEMENTED_OBSERVER,
        ("legacy_precedence",),
        ("prefill_load_scale", "overlap_score_credit"),
        "Deprecated alias. Canonical env is expected to win over "
        "DYN_OVERLAP_SCORE_WEIGHT and rewrite prefill_load_scale.",
    ),
    SettingDecision(
        "use_kv_events",
        ("--router-kv-events", "--no-router-kv-events", "--kv-events", "--no-kv-events"),
        ("DYN_ROUTER_USE_KV_EVENTS",),
        ("use_kv_events",),
        STATUS_SUPPORTED,
        IMPLEMENTED_OBSERVER,
        ("legacy_precedence", "policy_yaml_over_env"),
        ("use_kv_events",),
        "Canonical FE/EPP name. YAML is expected to win over this env.",
    ),
    SettingDecision(
        "use_kv_events_legacy_env",
        (),
        ("DYN_USE_KV_EVENTS",),
        (),
        STATUS_ALIAS,
        IMPLEMENTED_OBSERVER,
        ("legacy_precedence",),
        ("use_kv_events",),
        "Rust/EPP-only alias. Canonical DYN_ROUTER_USE_KV_EVENTS=false is "
        "expected to win when both are set. FE is not claimed to honor the "
        "old name alone.",
    ),
    SettingDecision(
        "router_policy_config",
        ("--router-policy-config",),
        ("DYN_ROUTER_POLICY_CONFIG",),
        ("router_policy_config",),
        STATUS_SUPPORTED,
        IMPLEMENTED_OBSERVER,
        ("policy_yaml_over_env",),
        ("use_kv_events", "router_event_threads", "router_ttl_secs"),
        "YAML router settings are expected to win over conflicting env. "
        "The file path is a fixture artifact, not an asserted key.",
    ),
    SettingDecision(
        "serve_indexer",
        ("--serve-indexer", "--no-serve-indexer"),
        ("DYN_SERVE_INDEXER",),
        ("serve_indexer",),
        STATUS_PENDING,
        None,
        (),
        (),
        "Frontend routing flag. Native to_json may include it; the EPP init "
        "event does not. No live observer in this increment.",
    ),
    SettingDecision(
        "kv_cache_block_size",
        ("--kv-cache-block-size",),
        ("DYN_KV_CACHE_BLOCK_SIZE",),
        ("kv_cache_block_size",),
        STATUS_PENDING,
        None,
        (),
        (),
        "Frontend wrapper CLI sets this for KV mode. It is not a KvRouterConfig "
        "to_json/init-event field. Do not treat an env translation as EPP honor.",
    ),
)


@dataclass(frozen=True)
class ConfigProfile:
    name: str
    env: Mapping[str, str]
    fields: tuple[str, ...]
    expected: Mapping[str, Any]
    deprecated_envs: tuple[str, ...]


def required_source_paths() -> tuple[Path, ...]:
    return (ROUTER_ARGS, KV_ROUTER_ARGS, FRONTEND_ARGS, RUST_CONFIG, *RUST_TREES)


def assert_required_sources_exist() -> None:
    missing = [str(path) for path in required_source_paths() if not path.exists()]
    if missing:
        raise AssertionError("required config sources missing: " + ", ".join(missing))


def _register_routing_parser() -> argparse.ArgumentParser:
    from dynamo.common.configuration.groups.kv_router_args import KvRouterArgGroup
    from dynamo.common.configuration.groups.router_args import RouterArgGroup

    parser = argparse.ArgumentParser(add_help=False)
    RouterArgGroup(
        default_router_mode="round-robin", include_frontend_only=True
    ).add_arguments(parser)
    KvRouterArgGroup().add_arguments(parser)
    return parser


def _frontend_related_actions() -> list[argparse.Action]:
    from dynamo.frontend.frontend_args import FrontendArgGroup

    parser = argparse.ArgumentParser(add_help=False)
    FrontendArgGroup().add_arguments(parser)
    routing_dests = {action.dest for action in _register_routing_parser()._actions}
    found: list[argparse.Action] = []
    extras: list[str] = []
    seen_required = set()
    for action in parser._actions:
        dest = action.dest
        if dest in FRONTEND_ROUTING_DESTS:
            found.append(action)
            seen_required.add(dest)
            continue
        if dest in routing_dests:
            continue
        if dest.startswith("router_") or any(
            flag.startswith("--router-") for flag in action.option_strings
        ):
            extras.append(dest)
    missing = sorted(FRONTEND_ROUTING_DESTS - seen_required)
    if missing:
        raise AssertionError(
            "FrontendArgGroup lost required routing flags: " + ", ".join(missing)
        )
    if extras:
        raise AssertionError(
            "new Frontend routing dests have no decision: " + ", ".join(sorted(extras))
        )
    return found


def discover_cli_actions() -> list[argparse.Action]:
    actions = [
        action
        for action in _register_routing_parser()._actions
        if action.option_strings
    ]
    actions.extend(_frontend_related_actions())
    if not actions:
        raise AssertionError("routing CLI discovery produced no actions")
    return actions


def _python_env_literals(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            names.update(ENV_LITERAL_RE.findall(node.value))
    return names


def _strip_rust_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return "\n".join(
        line.split("//", 1)[0] if not line.strip().startswith("//") else ""
        for line in text.splitlines()
    )


def _drop_cfg_test_modules(text: str) -> str:
    """Drop `#[cfg(test)] mod name { ... }` bodies. Leave other production text."""
    pattern = re.compile(r"#\[cfg\(test\)\]\s*mod\s+\w+\s*\{", re.S)
    out: list[str] = []
    pos = 0
    for match in pattern.finditer(text):
        out.append(text[pos : match.start()])
        depth = 1
        i = match.end()
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        pos = i
    out.append(text[pos:])
    return "".join(out)


def _rust_env_literals(path: Path) -> set[str]:
    text = _drop_cfg_test_modules(_strip_rust_comments(path.read_text(encoding="utf-8")))
    return set(ENV_LITERAL_RE.findall(text))


def discover_env_names() -> set[str]:
    names = _python_env_literals(ROUTER_ARGS) | _python_env_literals(KV_ROUTER_ARGS)
    frontend_names = _python_env_literals(FRONTEND_ARGS)
    names.update(
        name
        for name in frontend_names
        if name in {"DYN_SERVE_INDEXER", "DYN_KV_CACHE_BLOCK_SIZE"}
        or name.startswith("DYN_ROUTER_")
    )
    if not RUST_CONFIG.is_file():
        raise AssertionError(f"missing Rust router config source: {RUST_CONFIG}")
    names.update(_rust_env_literals(RUST_CONFIG))
    for tree in RUST_TREES:
        if not tree.exists():
            raise AssertionError(f"missing Rust source tree: {tree}")
        for path in tree.rglob("*.rs"):
            if path.name == "tests.rs" or "tests" in path.parts:
                continue
            names.update(
                name
                for name in _rust_env_literals(path)
                if name.startswith("DYN_ROUTER_")
                or name
                in {
                    "DYN_OVERLAP_SCORE_WEIGHT",
                    "DYN_USE_KV_EVENTS",
                    "DYN_USE_REMOTE_INDEXER",
                    "DYN_ENABLE_SESSION_PREFIX_INDEX",
                    "DYN_SHARED_CACHE_TYPE",
                    "DYN_SHARED_CACHE_MULTIPLIER",
                }
            )
    if not names:
        raise AssertionError("env discovery produced an empty inventory")
    return names


def discovered_identifier_sets() -> tuple[set[str], set[str], set[str]]:
    from dynamo.common.configuration.groups.kv_router_args import (
        _CONDITIONAL_DISAGG_CONFIG_FIELDS,
        _KV_ROUTER_FIELDS,
        _ROUTER_SETTINGS,
    )
    from dynamo.common.configuration.groups.router_args import _ROUTER_FIELDS

    actions = discover_cli_actions()
    flags = {flag for action in actions for flag in action.option_strings}
    schema = {action.dest for action in actions if action.dest}
    schema.update(_KV_ROUTER_FIELDS)
    schema.update(_ROUTER_SETTINGS)
    schema.update(_ROUTER_FIELDS)
    schema.update(_CONDITIONAL_DISAGG_CONFIG_FIELDS)
    schema.update(_CONDITIONAL_DISAGG_CONFIG_FIELDS.values())
    return flags, discover_env_names(), schema


def decided_identifier_sets() -> tuple[set[str], set[str], set[str]]:
    flags: set[str] = set()
    env: set[str] = set()
    schema: set[str] = set()
    for row in SETTING_DECISIONS:
        flags.update(row.flags)
        env.update(row.env)
        schema.update(row.schema_fields)
    return flags, env, schema


def inventory_differences() -> dict[str, set[str]]:
    discovered = discovered_identifier_sets()
    decided = decided_identifier_sets()
    labels = ("cli", "env", "schema")
    diffs: dict[str, set[str]] = {}
    for label, got, expected in zip(labels, discovered, decided):
        missing = got - expected
        stale = expected - got
        if missing:
            diffs[f"undecided_{label}"] = missing
        if stale:
            diffs[f"stale_{label}"] = stale
    return diffs


def pending_decision_names() -> tuple[str, ...]:
    return tuple(row.name for row in SETTING_DECISIONS if row.status == STATUS_PENDING)


def assert_inventory_matches_manifest() -> None:
    assert_required_sources_exist()
    diffs = inventory_differences()
    if diffs:
        parts = [
            f"{label}={sorted(values)}" for label, values in sorted(diffs.items())
        ]
        raise AssertionError(
            "discovered identifiers lack explicit decisions or are stale: "
            + "; ".join(parts)
        )


def assert_coverage_complete() -> None:
    """Fail while any decided row is pending or any discovered name is undecided."""
    assert_inventory_matches_manifest()
    pending = pending_decision_names()
    if pending:
        raise AssertionError(
            "pending coverage remains; not full-issue completion: "
            + ", ".join(pending)
        )


def assert_manifest_bindings_match_collected_tests(nodeids: Iterable[str]) -> None:
    """Require observer/profile rows to name a collected parametrized test.

    Manifest metadata strings are not coverage. A supported or alias row
    must use the implemented observer and bind ``affected_fields`` to the
    live asserted fields of each collected
    ``test_effective_config_profile[<profile>]`` node.
    """
    collected: set[str] = set()
    for nodeid in nodeids:
        for name in PROFILE_NAMES:
            if f"test_effective_config_profile[{name}]" in nodeid:
                collected.add(name)
    missing_profiles = [name for name in PROFILE_NAMES if name not in collected]
    if missing_profiles:
        raise AssertionError(
            "profile names are not collected as tests: "
            + ", ".join(missing_profiles)
        )
    if set(PROFILE_ASSERTED_FIELDS) != set(PROFILE_NAMES):
        raise AssertionError(
            "PROFILE_ASSERTED_FIELDS keys must equal PROFILE_NAMES: "
            f"{sorted(PROFILE_ASSERTED_FIELDS)} vs {list(PROFILE_NAMES)}"
        )
    if set(PROFILE_DEPRECATED_ENVS) != set(PROFILE_NAMES):
        raise AssertionError(
            "PROFILE_DEPRECATED_ENVS keys must equal PROFILE_NAMES: "
            f"{sorted(PROFILE_DEPRECATED_ENVS)} vs {list(PROFILE_NAMES)}"
        )
    for name, envs in PROFILE_DEPRECATED_ENVS.items():
        unknown = [env for env in envs if env not in _POLICY_PARAMETER_ENVS]
        if unknown:
            raise AssertionError(
                f"{name} deprecated envs are not policy-parameter aliases: "
                + ", ".join(unknown)
            )
    for row in SETTING_DECISIONS:
        if row.status == STATUS_PENDING:
            continue
        unbound = [name for name in row.profiles if name not in collected]
        if row.observer != IMPLEMENTED_OBSERVER or not row.profiles or unbound:
            raise AssertionError(
                f"{row.name} claims observer={row.observer!r} profiles="
                f"{row.profiles} but collected tests are {sorted(collected)} "
                f"and the implemented observer is {IMPLEMENTED_OBSERVER!r}"
            )
        if not row.affected_fields:
            raise AssertionError(f"{row.name} has no affected_fields")
        for profile in row.profiles:
            live = PROFILE_ASSERTED_FIELDS[profile]
            missing = [field for field in row.affected_fields if field not in live]
            if missing:
                raise AssertionError(
                    f"{row.name} affected_fields {missing} are not in "
                    f"{profile} asserted fields {live}"
                )


def apply_profile_env(monkeypatch, env: Mapping[str, str]) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("RUST_LOG", "info")
    monkeypatch.setenv("DYN_ROUTER_MIN_INITIAL_WORKERS", WRAPPER_MIN_WORKERS)
    for name, value in env.items():
        monkeypatch.setenv(name, value)


def frontend_native_projection(fields: Iterable[str]) -> dict[str, Any]:
    """Parse the live process env the same way Frontend CLI construction does."""
    from dynamo.common.configuration.groups.router_args import build_router_config
    from dynamo.frontend.frontend_args import FrontendArgGroup, FrontendConfig

    parser = argparse.ArgumentParser(add_help=False)
    FrontendArgGroup().add_arguments(parser)
    config = FrontendConfig.from_cli_args(parser.parse_args(list(WRAPPER_CLI)))
    config.validate()
    router = build_router_config(config)
    if router is None or router.kv_router_config is None:
        raise AssertionError("build_router_config did not produce KvRouterConfig")
    payload = json.loads(router.kv_router_config.to_json())
    missing = [name for name in fields if name not in payload]
    if missing:
        raise AssertionError(
            "native KvRouterConfig.to_json omitted requested fields "
            f"{missing}; those stay pending/other-observer"
        )
    return {name: payload[name] for name in fields}


def native_default_projection(fields: Iterable[str]) -> dict[str, Any]:
    from dynamo.llm import KvRouterConfig

    config = KvRouterConfig()
    payload = json.loads(config.to_json())
    payload.update(config.policy_parameters("aggregated"))
    missing = [name for name in fields if name not in payload]
    if missing:
        raise AssertionError(
            "native default KvRouterConfig omitted sentinel fields "
            f"{missing}; do not invent Rust defaults"
        )
    return {name: payload[name] for name in fields}


def _decode_event_value(name: str, raw: str, expected: Any) -> Any:
    if raw in {"true", "false"}:
        value: Any = raw == "true"
    else:
        try:
            if isinstance(expected, bool):
                raise AssertionError(f"{name}={raw!r} is not a bool")
            if isinstance(expected, int) and not isinstance(expected, bool):
                if any(token in raw for token in (".", "e", "E")):
                    value = float(raw)
                else:
                    value = int(raw)
            else:
                value = float(raw)
        except (TypeError, ValueError) as exc:
            raise AssertionError(f"{name}={raw!r} failed typed decode") from exc
    return value


def parse_init_event(line: str, expected: Mapping[str, Any]) -> dict[str, Any]:
    cleaned = ANSI_RE.sub("", line)
    parsed = {key: value for key, value in FIELD_RE.findall(cleaned)}
    missing = [name for name in expected if name not in parsed]
    if missing:
        raise AssertionError(f"init event missing requested fields {missing}")
    return {
        name: _decode_event_value(name, parsed[name], expected[name])
        for name in expected
    }


def observe_epp_init_event(
    epp_process,
    *,
    node_name: str,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    """Read the owned process's unique log for exactly one post-parse init event."""
    if epp_process is None:
        raise AssertionError("EppFrontendParityStack.epp_process was not saved")
    proc = epp_process.proc
    if proc is None or proc.poll() is not None:
        raise AssertionError("owned EPP process is not live")
    log_path = epp_process.log_path
    if not log_path:
        raise AssertionError("owned EPP process has no log_path")
    log = Path(log_path)
    if node_name not in log.as_posix():
        raise AssertionError(
            f"EPP log_path {log} is not this profile node {node_name}"
        )
    if not log.is_file():
        raise AssertionError(f"EPP log is missing: {log}")
    lines = [
        line
        for line in log.read_text(encoding="utf-8", errors="ignore").splitlines()
        if INIT_EVENT in line
    ]
    if len(lines) != 1:
        raise AssertionError(
            f"expected exactly one {INIT_EVENT!r} in {log} pid={proc.pid}, "
            f"found {len(lines)}"
        )
    return parse_init_event(lines[0], expected)


def _values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return float(left) == float(right)
    return left == right


def compare_effective_fields(
    *,
    profile: str,
    frontend: Mapping[str, Any],
    epp: Mapping[str, Any],
    expected: Mapping[str, Any],
    defaults: Mapping[str, Any],
) -> None:
    for name, wanted in expected.items():
        if name not in frontend:
            raise AssertionError(f"{profile} FE projection missing {name}")
        if name not in epp:
            raise AssertionError(f"{profile} EPP event missing {name}")
        if not _values_equal(frontend[name], epp[name]):
            raise AssertionError(
                f"{profile} {name}: FE={frontend[name]!r} EPP={epp[name]!r}"
            )
        if not _values_equal(frontend[name], wanted):
            raise AssertionError(
                f"{profile} {name}: observed {frontend[name]!r} != {wanted!r}"
            )
        if name in defaults and _values_equal(frontend[name], defaults[name]):
            raise AssertionError(
                f"{profile} {name}={frontend[name]!r} equals the native default"
            )


def _expected_deprecation_messages(envs: Iterable[str]) -> tuple[str, ...]:
    from dynamo.common.configuration.groups.kv_router_args import _policy_parameter

    messages: list[str] = []
    unknown: list[str] = []
    for name in envs:
        parameter = _POLICY_PARAMETER_ENVS.get(name)
        if parameter is None:
            unknown.append(name)
            continue
        messages.append(_policy_parameter(parameter).message(name))
    if unknown:
        raise AssertionError(
            "deprecated envs have no authoritative policy-parameter message: "
            + ", ".join(unknown)
        )
    return tuple(messages)


def frontend_projection_with_expected_deprecations(
    profile: ConfigProfile,
) -> dict[str, Any]:
    """Run the real FE parse/construct path and assert its Deprecated set."""
    expected = _expected_deprecation_messages(profile.deprecated_envs)
    missing_env = [name for name in profile.deprecated_envs if name not in profile.env]
    if missing_env:
        raise AssertionError(
            f"{profile.name} expected deprecations are not in profile env: "
            + ", ".join(missing_env)
        )
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("error")
        warnings.simplefilter("always", FutureWarning)
        payload = frontend_native_projection(profile.fields)
    got = [
        str(item.message)
        for item in records
        if issubclass(item.category, FutureWarning)
    ]
    if set(got) != set(expected) or len(got) != len(expected):
        raise AssertionError(
            f"{profile.name} frontend deprecations expected {list(expected)} "
            f"got {got}"
        )
    return payload


def build_profile(name: str, tmp_path: Path) -> ConfigProfile:
    if name == "scalar_kv":
        expected = {
            "overlap_score_credit": 1.25,
            "overlap_score_credit_decay": 0.2,
            "prefill_load_scale": 1.5,
            "decode_active_request_weight": 0.4,
            "router_temperature": 0.5,
            "host_cache_hit_weight": 0.6,
            "disk_cache_hit_weight": 0.3,
            "router_ttl_secs": 180.0,
            "router_event_threads": 2,
        }
        return ConfigProfile(
            name,
            {
                "DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT": "1.25",
                "DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT_DECAY": "0.2",
                "DYN_ROUTER_PREFILL_LOAD_SCALE": "1.5",
                "DYN_ROUTER_DECODE_ACTIVE_REQUEST_WEIGHT": "0.4",
                "DYN_ROUTER_TEMPERATURE": "0.5",
                "DYN_ROUTER_HOST_CACHE_HIT_WEIGHT": "0.6",
                "DYN_ROUTER_DISK_CACHE_HIT_WEIGHT": "0.3",
                "DYN_ROUTER_TTL_SECS": "180",
                "DYN_ROUTER_EVENT_THREADS": "2",
            },
            PROFILE_ASSERTED_FIELDS[name],
            expected,
            PROFILE_DEPRECATED_ENVS[name],
        )
    if name == "legacy_precedence":
        expected = {
            "overlap_score_credit": 0.8,
            "prefill_load_scale": 3.0,
            "use_kv_events": False,
        }
        return ConfigProfile(
            name,
            {
                "DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT": "0.8",
                "DYN_ROUTER_PREFILL_LOAD_SCALE": "1.5",
                "DYN_ROUTER_KV_OVERLAP_SCORE_WEIGHT": "3",
                "DYN_OVERLAP_SCORE_WEIGHT": "4",
                "DYN_ROUTER_USE_KV_EVENTS": "false",
                "DYN_USE_KV_EVENTS": "true",
            },
            PROFILE_ASSERTED_FIELDS[name],
            expected,
            PROFILE_DEPRECATED_ENVS[name],
        )
    if name == "policy_yaml_over_env":
        policy_path = tmp_path / "router-policy.yaml"
        policy_path.write_text(
            "router:\n"
            "  use_kv_events: false\n"
            "  router_event_threads: 2\n"
            "  router_ttl_secs: 42\n",
            encoding="utf-8",
        )
        expected = {
            "use_kv_events": False,
            "router_event_threads": 2,
            "router_ttl_secs": 42.0,
        }
        return ConfigProfile(
            name,
            {
                "DYN_ROUTER_POLICY_CONFIG": str(policy_path),
                "DYN_ROUTER_USE_KV_EVENTS": "true",
                "DYN_ROUTER_EVENT_THREADS": "8",
                "DYN_ROUTER_TTL_SECS": "180",
            },
            PROFILE_ASSERTED_FIELDS[name],
            expected,
            PROFILE_DEPRECATED_ENVS[name],
        )
    raise AssertionError(f"unknown config profile {name!r}")


def run_effective_config_profile(
    request,
    tmp_path: Path,
    monkeypatch,
    model_name: str,
    profile_name: str,
    stack_cls,
) -> None:
    profile = build_profile(profile_name, tmp_path)
    if profile.fields != PROFILE_ASSERTED_FIELDS[profile.name]:
        raise AssertionError(
            f"{profile.name} fields {profile.fields} != asserted "
            f"{PROFILE_ASSERTED_FIELDS[profile.name]}"
        )
    if tuple(profile.expected) != profile.fields:
        raise AssertionError(
            f"{profile.name} expected keys {tuple(profile.expected)} != "
            f"{profile.fields}"
        )
    if profile.deprecated_envs != PROFILE_DEPRECATED_ENVS[profile.name]:
        raise AssertionError(
            f"{profile.name} deprecated_envs {profile.deprecated_envs} != "
            f"{PROFILE_DEPRECATED_ENVS[profile.name]}"
        )
    apply_profile_env(monkeypatch, profile.env)
    frontend = frontend_projection_with_expected_deprecations(profile)
    defaults = native_default_projection(profile.fields)
    with stack_cls(request, tmp_path, model_name) as stack:
        epp = observe_epp_init_event(
            stack.epp_process,
            node_name=request.node.name,
            expected=profile.expected,
        )
        compare_effective_fields(
            profile=profile.name,
            frontend=frontend,
            epp=epp,
            expected=profile.expected,
            defaults=defaults,
        )
