# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FE/EPP effective-config contract for issue #16592.

Inventory is fail-closed and knowingly incomplete. Discovered CLI/env/schema
names without a row, and explicit PENDING rows, fail the completeness guard.
Do not treat that failure as full-issue completion.

``load_aware_preset`` and ``conditional_json`` are measured strict gaps
(#16592): dedicated subclasses of ``FrontendEppConfigMismatch`` with
strict xfail. Setup, warning, FE sentinel, parse, and other mismatches
stay ordinary hard failures. Other profiles stay ordinary tests.

``admission_control`` and ``enforce_disagg`` are ignored-compatibility
parser contracts. The five conditional scalar envs and
``positional_search_mode`` are separate not-FE-honored rows.
``use_remote_indexer`` binds the existing first-selection query-counter
observer. ``session_affinity_ttl_secs`` binds TTL enable-presence only on
the existing session_id gap. ``shared_cache_multiplier`` is a
supported_candidate profile pending runtime evidence. Remaining rows stay
pending.

The EPP observer reads the post-parse INFO event from
``log_env_config``. It does not observe later Runtime EPP instance overrides
(``skip_initial_worker_wait=true``,
``prefill_config.router_track_active_blocks=false``).
``active_cache`` observes ``router_track_active_blocks`` only on the decode /
post-parse config. Runtime EPP later disables that flag in its prefill copy.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
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
RUST_PYO3_ENTRYPOINT = REPO_ROOT / "lib/bindings/python/rust/llm/entrypoint.rs"
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
PROFILE_NAMES = (
    "scalar_kv",
    "legacy_precedence",
    "policy_yaml_over_env",
    "tracking_queue",
    "active_cache",
    "load_aware_preset",
    "conditional_json",
    "shared_cache_multiplier",
)
IMPLEMENTED_OBSERVER = "fe_native_to_json+epp_init_event"
IGNORED_COMPAT_OBSERVER = "fe_parser_logger+router_kwargs+native_to_json"
NOT_FE_HONORED_OBSERVER = "fe_full_native_json_before_after"
ADMISSION_CONTROL_CASE_IDS = (
    "cli_token_capacity",
    "cli_none",
    "env_token_capacity",
    "env_none",
)
ENFORCE_DISAGG_CASE_IDS = (
    "cli_positive",
    "cli_negative",
    "env_true",
    "env_false",
)
CONDITIONAL_SCALAR_PROBES = (
    ("DYN_ROUTER_CONDITIONAL_DISAGG_POLICY", "isl_or_load"),
    ("DYN_ROUTER_CONDITIONAL_DISAGG_EFF_ISL_THRESHOLD", "1024"),
    ("DYN_ROUTER_CONDITIONAL_DISAGG_EFF_ISL_RATIO_THRESHOLD", "0.25"),
    ("DYN_ROUTER_CONDITIONAL_DISAGG_PREFILL_BUSY_THRESHOLD", "0.8"),
    ("DYN_ROUTER_CONDITIONAL_DISAGG_DECODE_BUSY_THRESHOLD", "0.7"),
)
CONDITIONAL_SCALAR_ENV_NAMES = tuple(name for name, _ in CONDITIONAL_SCALAR_PROBES)
NOT_FE_HONORED_ROW = "conditional_disagg_epp_scalar_envs"
NOT_FE_HONORED_TEST = "test_conditional_scalar_not_fe_honored"
REMOTE_INDEXER_ROW = "use_remote_indexer"
REMOTE_INDEXER_OBSERVER = "provider_query_counter+worker_rank_token_parity"
REMOTE_INDEXER_ITEM = (
    "tests/router/test_epp_frontend_routing_parity.py::"
    "test_epp_frontend_first_selection_parity"
)
SESSION_TTL_ROW = "session_affinity_ttl_secs"
SESSION_TTL_OBSERVER = "fe_session_ttl_enable+session_affinity_parity_mismatch"
SESSION_TTL_ITEM = (
    "tests/router/test_epp_frontend_inputs_parity.py::"
    "test_epp_frontend_inputs_parity[session_id]"
)
SESSION_TTL_SECS = 300
INPUTS_PARITY = REPO_ROOT / "tests/router/test_epp_frontend_inputs_parity.py"
POSITIONAL_ROW = "positional_search_mode"
POSITIONAL_ENV = "DYN_ROUTER_POSITIONAL_SEARCH_MODE"
POSITIONAL_SENTINEL = "binary"
POSITIONAL_TEST = "test_positional_search_not_fe_honored"
POSITIONAL_ITEM = (
    f"tests/router/test_epp_frontend_config_parity.py::{POSITIONAL_TEST}"
)
NOT_FE_HONORED_ROWS = frozenset({NOT_FE_HONORED_ROW, POSITIONAL_ROW})
KV_INDEXER_MOD = REPO_ROOT / "lib/kv-router/src/indexer/mod.rs"
POSITIONAL_RS = REPO_ROOT / "lib/kv-router/src/indexer/positional.rs"
KV_INDEXER_RS = REPO_ROOT / "lib/kv-router/src/indexer/kv_indexer.rs"
KV_ROUTER_LIB = REPO_ROOT / "lib/kv-router/src/lib.rs"
LLM_INDEXER_MOD = REPO_ROOT / "lib/llm/src/kv_router/indexer/mod.rs"
INDEXER_TESTS_RS = REPO_ROOT / "lib/kv-router/src/indexer/tests.rs"
INDEXER_DELEGATE_TESTS_RS = REPO_ROOT / "lib/kv-router/src/indexer/delegate_tests.rs"
POSITIONAL_LIB_EXPORTS = (
    "pub use indexer::positional as nested_map;",
    "pub use nested_map::PositionalIndexer;",
)
POSITIONAL_INDEXER_MOD = "pub mod positional;"
_POSITIONAL_REF_RE = re.compile(
    r"\bPositionalIndexer\b|"
    r"\bnested_map\b|"
    r"(?:pub\s+)?mod\s+positional\b|"
    r"indexer::positional\b|"
    r"::positional\b|"
    r"(?:\{|,)\s*positional\b|"
    r"\buse\s+positional\b"
)
_POLICY_PARAMETER_ENVS = {
    "DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT": "overlap_score_credit",
    "DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT_DECAY": "overlap_score_credit_decay",
    "DYN_ROUTER_PREFILL_LOAD_SCALE": "prefill_load_scale",
    "DYN_ROUTER_DECODE_ACTIVE_REQUEST_WEIGHT": "decode_active_request_weight",
    "DYN_ROUTER_TEMPERATURE": "router_temperature",
    "DYN_ROUTER_KV_OVERLAP_SCORE_WEIGHT": "prefill_load_scale",
    "DYN_SHARED_CACHE_MULTIPLIER": "shared_cache_multiplier",
}
# Replacement and remove_in copied from kv_router_args.py Deprecated(...).
_DIRECT_DEPRECATED_ENVS = {
    "DYN_ROUTER_QUEUE_THRESHOLD": (
        "policy_classes[].prefill_busy_threshold_frac in --router-policy-config",
        "v1.7",
    ),
    "DYN_ROUTER_QUEUE_POLICY": (
        "policy_classes[].queue_policy in --router-policy-config",
        "v1.7",
    ),
    "DYN_ROUTER_LOAD_AWARE": (
        "the load-only policy and router tracking settings in --router-policy-config",
        "v1.7",
    ),
}
_ENUM_EVENT_VALUES = {
    "fcfs": "fcfs",
    "Fcfs": "fcfs",
    "lcfs": "lcfs",
    "Lcfs": "lcfs",
    "wspt": "wspt",
    "Wspt": "wspt",
    "isl_bounding": "isl_bounding",
    "IslBounding": "isl_bounding",
    "prefill_load": "prefill_load",
    "PrefillLoad": "prefill_load",
    "isl_or_load": "isl_or_load",
    "IslOrLoad": "isl_or_load",
}
_SOME_RE = re.compile(r"^Some\((.+)\)$")
OMITTABLE_DEFAULT_NATIVE_FIELDS = frozenset(
    {
        "conditional_disagg_enabled",
        "conditional_disagg_policy",
        "conditional_disagg_eff_isl_threshold",
        "conditional_disagg_eff_isl_ratio_threshold",
        "conditional_disagg_prefill_busy_threshold",
        "conditional_disagg_decode_busy_threshold",
    }
)
OPTION_EVENT_FIELDS = frozenset(
    {
        "router_queue_threshold",
        "router_predicted_ttl_secs",
        "conditional_disagg_prefill_busy_threshold",
        "conditional_disagg_decode_busy_threshold",
        "shared_cache_multiplier",
    }
)
_CONDITIONAL_JSON = (
    '{"policy":"isl_or_load","eff_isl_threshold":1024,'
    '"eff_isl_ratio_threshold":0.25,"prefill_busy_threshold":0.8,'
    '"decode_busy_threshold":0.7}'
)
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
    "tracking_queue": (
        "router_replica_sync",
        "router_track_output_blocks",
        "router_assume_kv_reuse",
        "router_track_prefill_tokens",
        "router_queue_threshold",
        "router_queue_policy",
    ),
    "active_cache": (
        "router_track_active_blocks",
        "router_predicted_ttl_secs",
    ),
    "load_aware_preset": (
        "overlap_score_credit",
        "use_kv_events",
    ),
    "conditional_json": (
        "conditional_disagg_enabled",
        "conditional_disagg_policy",
        "conditional_disagg_eff_isl_threshold",
        "conditional_disagg_eff_isl_ratio_threshold",
        "conditional_disagg_prefill_busy_threshold",
        "conditional_disagg_decode_busy_threshold",
    ),
    "shared_cache_multiplier": ("shared_cache_multiplier",),
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
    "tracking_queue": (
        "DYN_ROUTER_QUEUE_THRESHOLD",
        "DYN_ROUTER_QUEUE_POLICY",
    ),
    "active_cache": (),
    "load_aware_preset": ("DYN_ROUTER_LOAD_AWARE",),
    "conditional_json": (),
    "shared_cache_multiplier": ("DYN_SHARED_CACHE_MULTIPLIER",),
}

STATUS_PENDING = "pending"
STATUS_SUPPORTED = "supported_candidate"
STATUS_ALIAS = "alias"
STATUS_KNOWN_GAP = "known_gap"
STATUS_IGNORED_COMPAT = "ignored_compatibility"
STATUS_NOT_FE_HONORED = "not_fe_honored"
KNOWN_STATUSES = frozenset(
    {
        STATUS_PENDING,
        STATUS_SUPPORTED,
        STATUS_ALIAS,
        STATUS_KNOWN_GAP,
        STATUS_IGNORED_COMPAT,
        STATUS_NOT_FE_HONORED,
    }
)
LOGGER = logging.getLogger(__name__)


class FrontendEppConfigMismatch(AssertionError):
    """FE and EPP both produced a value, and those values differ."""


class LoadAwarePresetConfigMismatch(FrontendEppConfigMismatch):
    """Measured FE load-aware preset vs EPP ignoring DYN_ROUTER_LOAD_AWARE."""


class ConditionalJsonConfigMismatch(FrontendEppConfigMismatch):
    """Measured FE conditional JSON vs EPP default scalar config."""


@dataclass(frozen=True)
class MeasuredGapSignature:
    """Exact 25a FE/EPP projections and mismatch set for one known gap."""

    profile: str
    exception: type[FrontendEppConfigMismatch]
    frontend: Mapping[str, Any]
    epp: Mapping[str, Any]
    mismatch_fields: frozenset[str]


MEASURED_GAP_SIGNATURES: dict[str, MeasuredGapSignature] = {
    "load_aware_preset": MeasuredGapSignature(
        "load_aware_preset",
        LoadAwarePresetConfigMismatch,
        {"overlap_score_credit": 0.0, "use_kv_events": False},
        {"overlap_score_credit": 1.0, "use_kv_events": True},
        frozenset({"overlap_score_credit", "use_kv_events"}),
    ),
    "conditional_json": MeasuredGapSignature(
        "conditional_json",
        ConditionalJsonConfigMismatch,
        {
            "conditional_disagg_enabled": True,
            "conditional_disagg_policy": "isl_or_load",
            "conditional_disagg_eff_isl_threshold": 1024,
            "conditional_disagg_eff_isl_ratio_threshold": 0.25,
            "conditional_disagg_prefill_busy_threshold": 0.8,
            "conditional_disagg_decode_busy_threshold": 0.7,
        },
        {
            "conditional_disagg_enabled": True,
            "conditional_disagg_policy": "isl_bounding",
            "conditional_disagg_eff_isl_threshold": 2048,
            "conditional_disagg_eff_isl_ratio_threshold": 0.7,
            "conditional_disagg_prefill_busy_threshold": None,
            "conditional_disagg_decode_busy_threshold": None,
        },
        frozenset(
            {
                "conditional_disagg_policy",
                "conditional_disagg_eff_isl_threshold",
                "conditional_disagg_eff_isl_ratio_threshold",
                "conditional_disagg_prefill_busy_threshold",
                "conditional_disagg_decode_busy_threshold",
            }
        ),
    ),
}


@dataclass(frozen=True)
class OmittedDefaultNativeField:
    """Recorded skip_serializing_if absence from default native JSON."""

    field: str
    source: str = "lib/kv-router/src/scheduling/config.rs"


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


@dataclass(frozen=True)
class IgnoredCompatContract:
    """Fixed ordinary case-ID and identifier contract for one ignored row."""

    test_name: str
    case_ids: tuple[str, ...]
    flags: tuple[str, ...]
    env: tuple[str, ...]
    schema_fields: tuple[str, ...]
    affected_fields: tuple[str, ...]


IGNORED_COMPAT_CONTRACTS = {
    "admission_control": IgnoredCompatContract(
        "test_ignored_admission_control",
        ADMISSION_CONTROL_CASE_IDS,
        ("--admission-control",),
        ("DYN_ADMISSION_CONTROL",),
        ("admission_control",),
        ("admission_control",),
    ),
    "enforce_disagg": IgnoredCompatContract(
        "test_ignored_enforce_disagg",
        ENFORCE_DISAGG_CASE_IDS,
        ("--enforce-disagg", "--no-enforce-disagg"),
        ("DYN_ENFORCE_DISAGG",),
        ("enforce_disagg",),
        ("enforce_disagg",),
    ),
}


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
        (
            "--router-kv-events",
            "--no-router-kv-events",
            "--kv-events",
            "--no-kv-events",
        ),
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
        "router_replica_sync",
        ("--router-replica-sync", "--no-router-replica-sync"),
        ("DYN_ROUTER_REPLICA_SYNC",),
        ("router_replica_sync",),
        STATUS_SUPPORTED,
        IMPLEMENTED_OBSERVER,
        ("tracking_queue",),
        ("router_replica_sync",),
        "Candidate for FE to_json and the EPP init event on tracking_queue.",
    ),
    SettingDecision(
        "router_track_output_blocks",
        (
            "--router-track-output-blocks",
            "--no-router-track-output-blocks",
            "--track-output-blocks",
            "--no-track-output-blocks",
        ),
        ("DYN_ROUTER_TRACK_OUTPUT_BLOCKS",),
        ("router_track_output_blocks",),
        STATUS_SUPPORTED,
        IMPLEMENTED_OBSERVER,
        ("tracking_queue",),
        ("router_track_output_blocks",),
        "Candidate for FE to_json and the EPP init event on tracking_queue.",
    ),
    SettingDecision(
        "router_assume_kv_reuse",
        (
            "--router-assume-kv-reuse",
            "--no-router-assume-kv-reuse",
            "--assume-kv-reuse",
            "--no-assume-kv-reuse",
        ),
        ("DYN_ROUTER_ASSUME_KV_REUSE",),
        ("router_assume_kv_reuse",),
        STATUS_SUPPORTED,
        IMPLEMENTED_OBSERVER,
        ("tracking_queue",),
        ("router_assume_kv_reuse",),
        "Candidate for FE to_json and the EPP init event on tracking_queue.",
    ),
    SettingDecision(
        "router_track_prefill_tokens",
        ("--router-track-prefill-tokens", "--no-router-track-prefill-tokens"),
        ("DYN_ROUTER_TRACK_PREFILL_TOKENS",),
        ("router_track_prefill_tokens",),
        STATUS_SUPPORTED,
        IMPLEMENTED_OBSERVER,
        ("tracking_queue",),
        ("router_track_prefill_tokens",),
        "Candidate for FE to_json and the EPP init event on tracking_queue.",
    ),
    SettingDecision(
        "router_queue_threshold",
        ("--router-queue-threshold",),
        ("DYN_ROUTER_QUEUE_THRESHOLD",),
        ("router_queue_threshold",),
        STATUS_SUPPORTED,
        IMPLEMENTED_OBSERVER,
        ("tracking_queue",),
        ("router_queue_threshold",),
        "Candidate Option value for FE to_json and the EPP Some(...) init event.",
    ),
    SettingDecision(
        "router_queue_policy",
        ("--router-queue-policy",),
        ("DYN_ROUTER_QUEUE_POLICY",),
        ("router_queue_policy",),
        STATUS_SUPPORTED,
        IMPLEMENTED_OBSERVER,
        ("tracking_queue",),
        ("router_queue_policy",),
        "Candidate enum for FE serde and the EPP Display init event.",
    ),
    SettingDecision(
        "router_track_active_blocks",
        (
            "--router-track-active-blocks",
            "--no-router-track-active-blocks",
            "--track-active-blocks",
            "--no-track-active-blocks",
        ),
        ("DYN_ROUTER_TRACK_ACTIVE_BLOCKS",),
        ("router_track_active_blocks",),
        STATUS_SUPPORTED,
        IMPLEMENTED_OBSERVER,
        ("active_cache",),
        ("router_track_active_blocks",),
        "Candidate non-default false on active_cache. Default true is a "
        "tracking_queue prerequisite, not sentinel coverage. Runtime EPP later "
        "forces prefill_config.router_track_active_blocks=false after this event.",
    ),
    SettingDecision(
        "router_predicted_ttl_secs",
        ("--router-predicted-ttl-secs",),
        ("DYN_ROUTER_PREDICTED_TTL_SECS",),
        ("router_predicted_ttl_secs",),
        STATUS_SUPPORTED,
        IMPLEMENTED_OBSERVER,
        ("active_cache",),
        ("router_predicted_ttl_secs",),
        "Candidate Option value. KV events stay the default true prerequisite.",
    ),
    SettingDecision(
        "load_aware",
        ("--load-aware", "--no-load-aware"),
        ("DYN_ROUTER_LOAD_AWARE",),
        ("load_aware",),
        STATUS_KNOWN_GAP,
        IMPLEMENTED_OBSERVER,
        ("load_aware_preset",),
        ("overlap_score_credit", "use_kv_events"),
        "Measured 25a gap: FE applies the load-aware preset; EPP ignores "
        "DYN_ROUTER_LOAD_AWARE. Same env on both paths.",
    ),
    SettingDecision(
        "conditional_disagg_enabled",
        ("--router-conditional-disagg", "--no-router-conditional-disagg"),
        ("DYN_ROUTER_CONDITIONAL_DISAGG",),
        ("conditional_disagg_enabled",),
        STATUS_SUPPORTED,
        IMPLEMENTED_OBSERVER,
        ("conditional_json",),
        ("conditional_disagg_enabled",),
        "Shared enable flag. Nested parameter values come from the JSON object.",
    ),
    SettingDecision(
        "conditional_disagg_json",
        ("--router-conditional-disagg-config",),
        ("DYN_ROUTER_CONDITIONAL_DISAGG_CONFIG",),
        (
            "conditional_disagg_config",
            "conditional_disagg_config.policy",
            "conditional_disagg_config.eff_isl_threshold",
            "conditional_disagg_config.eff_isl_ratio_threshold",
            "conditional_disagg_config.prefill_busy_threshold",
            "conditional_disagg_config.decode_busy_threshold",
        ),
        STATUS_KNOWN_GAP,
        IMPLEMENTED_OBSERVER,
        ("conditional_json",),
        (
            "conditional_disagg_policy",
            "conditional_disagg_eff_isl_threshold",
            "conditional_disagg_eff_isl_ratio_threshold",
            "conditional_disagg_prefill_busy_threshold",
            "conditional_disagg_decode_busy_threshold",
        ),
        "Measured 25a gap: FE nested JSON keys vs EPP default scalars. Same "
        "JSON env on both paths; no scalar shadow input.",
    ),
    SettingDecision(
        "conditional_disagg_policy",
        (),
        (),
        ("conditional_disagg_policy",),
        STATUS_KNOWN_GAP,
        IMPLEMENTED_OBSERVER,
        ("conditional_json",),
        ("conditional_disagg_policy",),
        "Measured 25a gap: FE flattened policy from JSON vs EPP isl_bounding.",
    ),
    SettingDecision(
        "conditional_disagg_eff_isl_threshold",
        (),
        (),
        ("conditional_disagg_eff_isl_threshold",),
        STATUS_KNOWN_GAP,
        IMPLEMENTED_OBSERVER,
        ("conditional_json",),
        ("conditional_disagg_eff_isl_threshold",),
        "Measured 25a gap: FE flattened eff_isl_threshold from JSON vs EPP 2048.",
    ),
    SettingDecision(
        "conditional_disagg_eff_isl_ratio_threshold",
        (),
        (),
        ("conditional_disagg_eff_isl_ratio_threshold",),
        STATUS_KNOWN_GAP,
        IMPLEMENTED_OBSERVER,
        ("conditional_json",),
        ("conditional_disagg_eff_isl_ratio_threshold",),
        "Measured 25a gap: FE flattened ratio from JSON vs EPP 0.7.",
    ),
    SettingDecision(
        "conditional_disagg_prefill_busy_threshold",
        (),
        (),
        ("conditional_disagg_prefill_busy_threshold",),
        STATUS_KNOWN_GAP,
        IMPLEMENTED_OBSERVER,
        ("conditional_json",),
        ("conditional_disagg_prefill_busy_threshold",),
        "Measured 25a gap: FE flattened prefill busy from JSON vs EPP None.",
    ),
    SettingDecision(
        "conditional_disagg_decode_busy_threshold",
        (),
        (),
        ("conditional_disagg_decode_busy_threshold",),
        STATUS_KNOWN_GAP,
        IMPLEMENTED_OBSERVER,
        ("conditional_json",),
        ("conditional_disagg_decode_busy_threshold",),
        "Measured 25a gap: FE flattened decode busy from JSON vs EPP None.",
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
    SettingDecision(
        "conditional_disagg_epp_scalar_envs",
        (),
        CONDITIONAL_SCALAR_ENV_NAMES,
        (),
        STATUS_NOT_FE_HONORED,
        NOT_FE_HONORED_OBSERVER,
        CONDITIONAL_SCALAR_ENV_NAMES,
        CONDITIONAL_SCALAR_ENV_NAMES,
        "EPP-only scalar env names. Isolated FE full-JSON probes must stay "
        "unchanged. Not support for the Frontend conditional JSON input.",
    ),
    SettingDecision(
        "active_decode_blocks_threshold",
        ("--active-decode-blocks-threshold",),
        ("DYN_ACTIVE_DECODE_BLOCKS_THRESHOLD",),
        ("active_decode_blocks_threshold",),
        STATUS_PENDING,
        None,
        (),
        (),
        "RouterConfig admission field. Not in KvRouterConfig to_json or the "
        "EPP init event. No live observer in this increment.",
    ),
    SettingDecision(
        "active_prefill_tokens_threshold",
        ("--active-prefill-tokens-threshold",),
        ("DYN_ACTIVE_PREFILL_TOKENS_THRESHOLD",),
        ("active_prefill_tokens_threshold",),
        STATUS_PENDING,
        None,
        (),
        (),
        "Deprecated RouterConfig admission field. Not in the EPP init event. "
        "No live observer in this increment.",
    ),
    SettingDecision(
        "active_prefill_tokens_threshold_frac",
        ("--active-prefill-tokens-threshold-frac",),
        ("DYN_ACTIVE_PREFILL_TOKENS_THRESHOLD_FRAC",),
        ("active_prefill_tokens_threshold_frac",),
        STATUS_PENDING,
        None,
        (),
        (),
        "RouterConfig admission field. Not in the EPP init event. No live "
        "observer in this increment.",
    ),
    SettingDecision(
        "admission_control",
        ("--admission-control",),
        ("DYN_ADMISSION_CONTROL",),
        ("admission_control",),
        STATUS_IGNORED_COMPAT,
        IGNORED_COMPAT_OBSERVER,
        ADMISSION_CONTROL_CASE_IDS,
        ("admission_control",),
        "Removed FE flag/env; accepted and ignored. Parser warning and "
        "router_kwargs/native JSON stay unchanged.",
    ),
    SettingDecision(
        "enforce_disagg",
        ("--enforce-disagg", "--no-enforce-disagg"),
        ("DYN_ENFORCE_DISAGG",),
        ("enforce_disagg",),
        STATUS_IGNORED_COMPAT,
        IGNORED_COMPAT_OBSERVER,
        ENFORCE_DISAGG_CASE_IDS,
        ("enforce_disagg",),
        "Deprecated ignored FE topology flag. Parser warning and "
        "router_kwargs/native JSON stay unchanged. Rust conversion forces "
        "enforce_disagg false.",
    ),
    SettingDecision(
        "enable_session_prefix_index",
        ("--enable-session-prefix-index", "--no-enable-session-prefix-index"),
        ("DYN_ENABLE_SESSION_PREFIX_INDEX",),
        ("enable_session_prefix_index",),
        STATUS_PENDING,
        None,
        (),
        (),
        "FE/EPP field exists. No profile exercises it. No live observer in "
        "this increment.",
    ),
    SettingDecision(
        "min_initial_workers",
        ("--router-min-initial-workers",),
        ("DYN_ROUTER_MIN_INITIAL_WORKERS",),
        ("min_initial_workers",),
        STATUS_PENDING,
        None,
        (),
        (),
        "Wrapper startup gate only. Not a KvRouterConfig to_json/init-event "
        "field. Do not claim the wrapper value as sentinel coverage.",
    ),
    SettingDecision(
        "router_mode",
        ("--router-mode",),
        ("DYN_ROUTER_MODE",),
        ("router_mode",),
        STATUS_PENDING,
        None,
        (),
        (),
        "Wrapper --router-mode kv selects KV construction. Not a "
        "KvRouterConfig to_json/init-event field.",
    ),
    SettingDecision(
        "router_approximate_cache_policy",
        ("--router-approximate-cache-policy",),
        ("DYN_ROUTER_APPROXIMATE_CACHE_POLICY",),
        ("router_approximate_cache_policy",),
        STATUS_PENDING,
        None,
        (),
        (),
        "Process-local; skipped from to_json. No live observer in this increment.",
    ),
    SettingDecision(
        "router_prefill_policy",
        ("--router-prefill-policy",),
        ("DYN_ROUTER_PREFILL_POLICY",),
        ("router_prefill_policy",),
        STATUS_PENDING,
        None,
        (),
        (),
        "Process-local; skipped from to_json. No live observer in this increment.",
    ),
    SettingDecision(
        "router_decode_policy",
        ("--router-decode-policy",),
        ("DYN_ROUTER_DECODE_POLICY",),
        ("router_decode_policy",),
        STATUS_PENDING,
        None,
        (),
        (),
        "Process-local; skipped from to_json. No live observer in this increment.",
    ),
    SettingDecision(
        "router_prefill_load_model",
        ("--router-prefill-load-model",),
        ("DYN_ROUTER_PREFILL_LOAD_MODEL",),
        ("router_prefill_load_model",),
        STATUS_PENDING,
        None,
        (),
        (),
        "FE/EPP field exists. No profile exercises it. No live observer in "
        "this increment.",
    ),
    SettingDecision(
        "session_affinity_ttl_secs",
        ("--router-session-affinity-ttl-secs",),
        ("DYN_ROUTER_SESSION_AFFINITY_TTL_SECS",),
        ("session_affinity_ttl_secs",),
        STATUS_KNOWN_GAP,
        SESSION_TTL_OBSERVER,
        (),
        ("session_affinity_ttl_secs",),
        "Existing session_id Item enables nondefault TTL 300 versus default "
        "None, then FE-A/EPP-B raises SessionAffinityParityMismatch. Not a "
        "300-second expiration sweep. session_affinity_mode stays pending.",
    ),
    SettingDecision(
        "session_affinity_mode",
        ("--router-session-affinity-mode",),
        ("DYN_ROUTER_SESSION_AFFINITY_MODE",),
        ("session_affinity_mode",),
        STATUS_PENDING,
        None,
        (),
        (),
        "RouterConfig session field. Not in KvRouterConfig to_json or the EPP "
        "init event. No live observer in this increment.",
    ),
    SettingDecision(
        "router_tracking_hash",
        ("--router-tracking-hash",),
        ("DYN_ROUTER_TRACKING_HASH",),
        ("router_tracking_hash",),
        STATUS_PENDING,
        None,
        (),
        (),
        "FE/EPP field exists. No profile exercises it. No live observer in "
        "this increment.",
    ),
    SettingDecision(
        "router_tracking_key_file",
        ("--router-tracking-key-file",),
        ("DYN_ROUTER_TRACKING_KEY_FILE",),
        ("router_tracking_key_file",),
        STATUS_PENDING,
        None,
        (),
        (),
        "FE/EPP Option field exists. No profile exercises it. No live "
        "observer in this increment.",
    ),
    SettingDecision(
        "router_tracking_key_id",
        ("--router-tracking-key-id",),
        ("DYN_ROUTER_TRACKING_KEY_ID",),
        ("router_tracking_key_id",),
        STATUS_PENDING,
        None,
        (),
        (),
        "FE/EPP Option field exists. No profile exercises it. No live "
        "observer in this increment.",
    ),
    SettingDecision(
        "shared_cache_multiplier",
        ("--shared-cache-multiplier",),
        ("DYN_SHARED_CACHE_MULTIPLIER",),
        ("shared_cache_multiplier",),
        STATUS_SUPPORTED,
        IMPLEMENTED_OBSERVER,
        ("shared_cache_multiplier",),
        ("shared_cache_multiplier",),
        "Candidate for FE to_json and the EPP init event on "
        "shared_cache_multiplier. Pending actual runtime evidence.",
    ),
    SettingDecision(
        "shared_cache_type",
        ("--shared-cache-type",),
        ("DYN_SHARED_CACHE_TYPE",),
        ("shared_cache_type",),
        STATUS_PENDING,
        None,
        (),
        (),
        "FE/EPP field exists. No profile exercises it. No live observer in "
        "this increment.",
    ),
    SettingDecision(
        "use_remote_indexer",
        ("--use-remote-indexer", "--no-use-remote-indexer"),
        ("DYN_USE_REMOTE_INDEXER",),
        ("use_remote_indexer",),
        STATUS_SUPPORTED,
        REMOTE_INDEXER_OBSERVER,
        (),
        ("use_remote_indexer",),
        "Existing first-selection Item observes provider kv_indexer_query via "
        "provider_query_count; each consumer arm requires a stable baseline, "
        "a fresh increment, and worker/rank/token parity.",
    ),
    SettingDecision(
        "active_request_expiry_secs",
        (),
        ("DYN_ROUTER_ACTIVE_REQUEST_EXPIRY_SECS",),
        (),
        STATUS_PENDING,
        None,
        (),
        (),
        "Rust env consumed by sequence expiry. No FE CLI dest. No live "
        "observer in this increment.",
    ),
    SettingDecision(
        "active_sequence_direct_zmq",
        (),
        ("DYN_ROUTER_ACTIVE_SEQUENCE_DIRECT_ZMQ",),
        (),
        STATUS_PENDING,
        None,
        (),
        (),
        "Rust env consumed by active-sequence transport. No FE CLI dest. No "
        "live observer in this increment.",
    ),
    SettingDecision(
        "active_sequence_zmq_rcvhwm",
        (),
        ("DYN_ROUTER_ACTIVE_SEQUENCE_ZMQ_RCVHWM",),
        (),
        STATUS_PENDING,
        None,
        (),
        (),
        "Rust env consumed by active-sequence HWM. No FE CLI dest. No live "
        "observer in this increment.",
    ),
    SettingDecision(
        "overlap_refresh_after_secs",
        (),
        ("DYN_ROUTER_OVERLAP_REFRESH_AFTER_SECS",),
        (),
        STATUS_PENDING,
        None,
        (),
        (),
        "Rust env consumed by overlap refresh. No FE CLI dest. No live "
        "observer in this increment.",
    ),
    SettingDecision(
        "positional_search_mode",
        (),
        (POSITIONAL_ENV,),
        (),
        STATUS_NOT_FE_HONORED,
        NOT_FE_HONORED_OBSERVER,
        (),
        (),
        "SearchMode::from_env is read only by PositionalIndexer::new. "
        "Production FE/EPP factory chooses ConcurrentRadixTreeCompressed or "
        "KvIndexer/RadixTree. Isolated FE full-JSON probe must stay unchanged.",
    ),
    SettingDecision(
        "worker_selection_policy",
        (),
        ("DYN_ROUTER_WORKER_SELECTION_POLICY",),
        (),
        STATUS_PENDING,
        None,
        (),
        (),
        "Rust env consumed by worker-selection instance choice. No FE CLI "
        "dest. No live observer in this increment.",
    ),
    SettingDecision(
        "zmq_endpoints_per_sub",
        (),
        ("DYN_ROUTER_ZMQ_ENDPOINTS_PER_SUB",),
        (),
        STATUS_PENDING,
        None,
        (),
        (),
        "Rust env consumed by subscriber pool sizing. No FE CLI dest. No live "
        "observer in this increment.",
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
    return (
        ROUTER_ARGS,
        KV_ROUTER_ARGS,
        FRONTEND_ARGS,
        RUST_CONFIG,
        RUST_PYO3_ENTRYPOINT,
        *RUST_TREES,
    )


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
    text = _drop_cfg_test_modules(
        _strip_rust_comments(path.read_text(encoding="utf-8"))
    )
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
    schema.update(
        f"conditional_disagg_config.{key}" for key in _CONDITIONAL_DISAGG_CONFIG_FIELDS
    )
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


def assert_decision_identifier_ownership() -> None:
    """Reject unknown statuses and duplicate name/flag/env/schema owners."""
    names: set[str] = set()
    flags: dict[str, str] = {}
    envs: dict[str, str] = {}
    schema: dict[str, str] = {}
    for row in SETTING_DECISIONS:
        if row.status not in KNOWN_STATUSES:
            raise AssertionError(
                f"{row.name} has unknown status {row.status!r}; "
                f"expected one of {sorted(KNOWN_STATUSES)}"
            )
        if (
            row.status == STATUS_IGNORED_COMPAT
            and row.name not in IGNORED_COMPAT_CONTRACTS
        ):
            raise AssertionError(
                f"{row.name} may not use {STATUS_IGNORED_COMPAT}; only "
                f"{sorted(IGNORED_COMPAT_CONTRACTS)} may"
            )
        if row.status == STATUS_NOT_FE_HONORED and row.name not in NOT_FE_HONORED_ROWS:
            raise AssertionError(
                f"{row.name} may not use {STATUS_NOT_FE_HONORED}; only "
                f"{sorted(NOT_FE_HONORED_ROWS)} may"
            )
        if row.observer == REMOTE_INDEXER_OBSERVER and row.name != REMOTE_INDEXER_ROW:
            raise AssertionError(
                f"{row.name} may not use {REMOTE_INDEXER_OBSERVER}; only "
                f"{REMOTE_INDEXER_ROW} may"
            )
        if row.observer == SESSION_TTL_OBSERVER and row.name != SESSION_TTL_ROW:
            raise AssertionError(
                f"{row.name} may not use {SESSION_TTL_OBSERVER}; only "
                f"{SESSION_TTL_ROW} may"
            )
        if row.name in names:
            raise AssertionError(f"duplicate decision name {row.name!r}")
        names.add(row.name)
        for flag in row.flags:
            owner = flags.get(flag)
            if owner is not None:
                raise AssertionError(
                    f"flag {flag!r} owned by {owner!r} and {row.name!r}"
                )
            flags[flag] = row.name
        for env in row.env:
            owner = envs.get(env)
            if owner is not None:
                raise AssertionError(f"env {env!r} owned by {owner!r} and {row.name!r}")
            envs[env] = row.name
        for field in row.schema_fields:
            owner = schema.get(field)
            if owner is not None:
                raise AssertionError(
                    f"schema {field!r} owned by {owner!r} and {row.name!r}"
                )
            schema[field] = row.name


def assert_inventory_matches_manifest() -> None:
    assert_required_sources_exist()
    assert_decision_identifier_ownership()
    diffs = inventory_differences()
    if diffs:
        parts = [f"{label}={sorted(values)}" for label, values in sorted(diffs.items())]
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
            "pending coverage remains; not full-issue completion: " + ", ".join(pending)
        )


def _require_collected_item(item: Any) -> Any:
    if isinstance(item, (str, bytes)) or not hasattr(item, "nodeid"):
        raise AssertionError(
            "manifest binding requires collected pytest Items, not "
            f"{type(item).__name__}"
        )
    if not callable(getattr(item, "iter_markers", None)):
        raise AssertionError(f"collected item {item.nodeid!r} has no iter_markers")
    return item


def _assert_measured_xfail_marker(name: str, mark: Any) -> None:
    spec = MEASURED_GAP_SIGNATURES[name]
    if mark.args:
        raise AssertionError(f"{name} xfail has positional condition {mark.args!r}")
    if "condition" in mark.kwargs:
        raise AssertionError(f"{name} xfail has condition={mark.kwargs['condition']!r}")
    if mark.kwargs.get("strict") is not True:
        raise AssertionError(
            f"{name} xfail strict={mark.kwargs.get('strict')!r} is not True"
        )
    if mark.kwargs.get("raises") is not spec.exception:
        raise AssertionError(
            f"{name} xfail raises={mark.kwargs.get('raises')!r} "
            f"!= {spec.exception!r}"
        )
    if mark.kwargs.get("run", True) is not True:
        raise AssertionError(f"{name} xfail run={mark.kwargs.get('run')!r} is disabled")


def _collect_parametrized_items(
    items: Iterable[Any], test_name: str, case_ids: tuple[str, ...]
) -> dict[str, Any]:
    collected: dict[str, Any] = {}
    matched: list[Any] = []
    for raw in items:
        item = _require_collected_item(raw)
        if f"{test_name}[" not in item.nodeid:
            continue
        matched.append(item)
        for case_id in case_ids:
            token = f"{test_name}[{case_id}]"
            if token not in item.nodeid:
                continue
            if case_id in collected:
                raise AssertionError(f"{token} collected more than once")
            collected[case_id] = item
    missing = [case_id for case_id in case_ids if case_id not in collected]
    if missing:
        raise AssertionError(
            f"{test_name} missing collected case IDs: " + ", ".join(missing)
        )
    if len(matched) != len(case_ids):
        raise AssertionError(
            f"{test_name} collected {len(matched)} items, expected "
            f"{len(case_ids)} fixed case IDs {list(case_ids)}"
        )
    return collected


def _assert_ordinary_collected_item(label: str, item: Any) -> None:
    xfails = list(item.iter_markers(name="xfail"))
    skips = list(item.iter_markers(name="skip")) + list(
        item.iter_markers(name="skipif")
    )
    if xfails or skips:
        raise AssertionError(
            f"{label} must remain an ordinary test, found "
            f"xfail={len(xfails)} skip={len(skips)}"
        )


def _assert_ignored_compat_binding(row: SettingDecision, items: Iterable[Any]) -> None:
    spec = IGNORED_COMPAT_CONTRACTS.get(row.name)
    if spec is None:
        raise AssertionError(
            f"{row.name} is not a fixed ignored-compatibility contract"
        )
    if row.observer != IGNORED_COMPAT_OBSERVER:
        raise AssertionError(
            f"{row.name} observer={row.observer!r} != {IGNORED_COMPAT_OBSERVER!r}"
        )
    if row.profiles != spec.case_ids:
        raise AssertionError(
            f"{row.name} profiles {row.profiles} != fixed case IDs " f"{spec.case_ids}"
        )
    if row.affected_fields != spec.affected_fields:
        raise AssertionError(
            f"{row.name} affected_fields {row.affected_fields} != "
            f"{spec.affected_fields}"
        )
    if row.flags != spec.flags:
        raise AssertionError(f"{row.name} flags {row.flags} != {spec.flags}")
    if row.env != spec.env:
        raise AssertionError(f"{row.name} env {row.env} != {spec.env}")
    if row.schema_fields != spec.schema_fields:
        raise AssertionError(
            f"{row.name} schema_fields {row.schema_fields} != {spec.schema_fields}"
        )
    collected = _collect_parametrized_items(items, spec.test_name, spec.case_ids)
    for case_id, item in collected.items():
        _assert_ordinary_collected_item(f"{spec.test_name}[{case_id}]", item)


def _canonical_item_nodeids(token: str) -> frozenset[str]:
    if not token.startswith("tests/router/") or token.count("::") != 1:
        raise AssertionError(
            f"canonical Item {token!r} must be tests/router/<file>::<id>"
        )
    rel_file, ident = token.split("::", 1)
    if (
        not rel_file.endswith(".py")
        or ".." in rel_file
        or "/" in ident
        or "\\" in ident
    ):
        raise AssertionError(
            f"canonical Item {token!r} is not tests/router/<file>::<id>"
        )
    abs_file = REPO_ROOT / rel_file
    names = {token, f"{abs_file.as_posix()}::{ident}"}
    native = str(abs_file)
    if native != abs_file.as_posix():
        names.add(f"{native}::{ident}")
    return frozenset(names)


def _require_unique_nodeid_item(items: Iterable[Any], token: str) -> Any:
    allowed = _canonical_item_nodeids(token)
    matched: list[Any] = []
    for raw in items:
        item = _require_collected_item(raw)
        if item.nodeid not in allowed:
            continue
        matched.append(item)
    if len(matched) != 1:
        raise AssertionError(
            f"{token} must collect exactly one Item, found {len(matched)}"
        )
    return matched[0]


def _assert_conditional_scalar_not_fe_binding(
    row: SettingDecision, items: Iterable[Any]
) -> None:
    if row.name != NOT_FE_HONORED_ROW:
        raise AssertionError(
            f"{row.name} is not the fixed {NOT_FE_HONORED_ROW} contract"
        )
    if row.observer != NOT_FE_HONORED_OBSERVER:
        raise AssertionError(
            f"{row.name} observer={row.observer!r} != {NOT_FE_HONORED_OBSERVER!r}"
        )
    if row.profiles != CONDITIONAL_SCALAR_ENV_NAMES:
        raise AssertionError(
            f"{row.name} profiles {row.profiles} != " f"{CONDITIONAL_SCALAR_ENV_NAMES}"
        )
    if row.env != CONDITIONAL_SCALAR_ENV_NAMES:
        raise AssertionError(
            f"{row.name} env {row.env} != {CONDITIONAL_SCALAR_ENV_NAMES}"
        )
    if row.affected_fields != CONDITIONAL_SCALAR_ENV_NAMES:
        raise AssertionError(
            f"{row.name} affected_fields {row.affected_fields} != "
            f"{CONDITIONAL_SCALAR_ENV_NAMES}"
        )
    if row.flags != ():
        raise AssertionError(f"{row.name} flags {row.flags} != ()")
    if row.schema_fields != ():
        raise AssertionError(f"{row.name} schema_fields {row.schema_fields} != ()")
    collected = _collect_parametrized_items(
        items, NOT_FE_HONORED_TEST, CONDITIONAL_SCALAR_ENV_NAMES
    )
    for case_id, item in collected.items():
        _assert_ordinary_collected_item(f"{NOT_FE_HONORED_TEST}[{case_id}]", item)


def _assert_positional_not_fe_binding(
    row: SettingDecision, items: Iterable[Any]
) -> None:
    if row.name != POSITIONAL_ROW:
        raise AssertionError(f"{row.name} is not the fixed {POSITIONAL_ROW} contract")
    if row.observer != NOT_FE_HONORED_OBSERVER:
        raise AssertionError(
            f"{row.name} observer={row.observer!r} != {NOT_FE_HONORED_OBSERVER!r}"
        )
    if row.env != (POSITIONAL_ENV,):
        raise AssertionError(f"{row.name} env {row.env} != {(POSITIONAL_ENV,)!r}")
    if row.flags != ():
        raise AssertionError(f"{row.name} flags {row.flags} != ()")
    if row.schema_fields != ():
        raise AssertionError(f"{row.name} schema_fields {row.schema_fields} != ()")
    if row.profiles != ():
        raise AssertionError(f"{row.name} profiles {row.profiles} != ()")
    if row.affected_fields != ():
        raise AssertionError(f"{row.name} affected_fields {row.affected_fields} != ()")
    item = _require_unique_nodeid_item(items, POSITIONAL_ITEM)
    _assert_ordinary_collected_item(POSITIONAL_ITEM, item)


def _assert_not_fe_honored_binding(row: SettingDecision, items: Iterable[Any]) -> None:
    if row.name == NOT_FE_HONORED_ROW:
        _assert_conditional_scalar_not_fe_binding(row, items)
        return
    if row.name == POSITIONAL_ROW:
        _assert_positional_not_fe_binding(row, items)
        return
    raise AssertionError(
        f"{row.name} is not an explicit not-FE-honored contract; "
        f"expected one of {sorted(NOT_FE_HONORED_ROWS)}"
    )


def _assert_remote_indexer_binding(row: SettingDecision, items: Iterable[Any]) -> None:
    if row.name != REMOTE_INDEXER_ROW:
        raise AssertionError(
            f"{row.name} is not the fixed {REMOTE_INDEXER_ROW} contract"
        )
    if row.status != STATUS_SUPPORTED:
        raise AssertionError(f"{row.name} status {row.status!r} != {STATUS_SUPPORTED!r}")
    if row.observer != REMOTE_INDEXER_OBSERVER:
        raise AssertionError(
            f"{row.name} observer={row.observer!r} != {REMOTE_INDEXER_OBSERVER!r}"
        )
    if row.flags != ("--use-remote-indexer", "--no-use-remote-indexer"):
        raise AssertionError(f"{row.name} flags {row.flags} are not the closed tuple")
    if row.env != ("DYN_USE_REMOTE_INDEXER",):
        raise AssertionError(f"{row.name} env {row.env} != ('DYN_USE_REMOTE_INDEXER',)")
    if row.schema_fields != ("use_remote_indexer",):
        raise AssertionError(
            f"{row.name} schema_fields {row.schema_fields} != ('use_remote_indexer',)"
        )
    if row.affected_fields != ("use_remote_indexer",):
        raise AssertionError(
            f"{row.name} affected_fields {row.affected_fields} != "
            "('use_remote_indexer',)"
        )
    if row.profiles != ():
        raise AssertionError(f"{row.name} profiles {row.profiles} != ()")
    item = _require_unique_nodeid_item(items, REMOTE_INDEXER_ITEM)
    _assert_ordinary_collected_item(REMOTE_INDEXER_ITEM, item)


def _assert_session_ttl_xfail_marker(mark: Any) -> None:
    from tests.router.common import SessionAffinityParityMismatch

    if mark.args:
        raise AssertionError(
            f"{SESSION_TTL_ROW} xfail has positional condition {mark.args!r}"
        )
    if "condition" in mark.kwargs:
        raise AssertionError(
            f"{SESSION_TTL_ROW} xfail has condition={mark.kwargs['condition']!r}"
        )
    if mark.kwargs.get("strict") is not True:
        raise AssertionError(
            f"{SESSION_TTL_ROW} xfail strict={mark.kwargs.get('strict')!r} "
            "is not True"
        )
    if mark.kwargs.get("raises") is not SessionAffinityParityMismatch:
        raise AssertionError(
            f"{SESSION_TTL_ROW} xfail raises={mark.kwargs.get('raises')!r} "
            f"!= {SessionAffinityParityMismatch!r}"
        )
    if mark.kwargs.get("run", True) is not True:
        raise AssertionError(
            f"{SESSION_TTL_ROW} xfail run={mark.kwargs.get('run')!r} is disabled"
        )


def _assert_session_ttl_binding(row: SettingDecision, items: Iterable[Any]) -> None:
    if row.name != SESSION_TTL_ROW:
        raise AssertionError(f"{row.name} is not the fixed {SESSION_TTL_ROW} contract")
    if row.status != STATUS_KNOWN_GAP:
        raise AssertionError(f"{row.name} status {row.status!r} != {STATUS_KNOWN_GAP!r}")
    if row.observer != SESSION_TTL_OBSERVER:
        raise AssertionError(
            f"{row.name} observer={row.observer!r} != {SESSION_TTL_OBSERVER!r}"
        )
    if row.flags != ("--router-session-affinity-ttl-secs",):
        raise AssertionError(f"{row.name} flags {row.flags} are not the closed tuple")
    if row.env != ("DYN_ROUTER_SESSION_AFFINITY_TTL_SECS",):
        raise AssertionError(
            f"{row.name} env {row.env} != ('DYN_ROUTER_SESSION_AFFINITY_TTL_SECS',)"
        )
    if row.schema_fields != ("session_affinity_ttl_secs",):
        raise AssertionError(
            f"{row.name} schema_fields {row.schema_fields} != "
            "('session_affinity_ttl_secs',)"
        )
    if row.affected_fields != ("session_affinity_ttl_secs",):
        raise AssertionError(
            f"{row.name} affected_fields {row.affected_fields} != "
            "('session_affinity_ttl_secs',)"
        )
    if row.profiles != ():
        raise AssertionError(f"{row.name} profiles {row.profiles} != ()")
    item = _require_unique_nodeid_item(items, SESSION_TTL_ITEM)
    skips = list(item.iter_markers(name="skip")) + list(
        item.iter_markers(name="skipif")
    )
    marks = list(item.iter_markers(name="xfail"))
    if skips or len(marks) != 1:
        raise AssertionError(
            f"{SESSION_TTL_ITEM} must collect exactly one xfail and no skip, "
            f"found xfail={len(marks)} skip={len(skips)}"
        )
    _assert_session_ttl_xfail_marker(marks[0])
    assert_session_inputs_stack_uses_ttl_300()


def assert_manifest_bindings_match_collected_tests(items: Iterable[Any]) -> None:
    """Require observer/profile rows to name a collected parametrized test.

    Manifest metadata strings are not coverage. Collection entries must be
    real pytest Items. A supported, alias, or known_gap row must use the
    implemented observer and bind ``affected_fields`` to the live asserted
    fields of each collected ``test_effective_config_profile[<profile>]``
    node. Ordinary profiles must carry exactly zero xfail markers. The two
    measured profiles must carry exactly one unconditional strict xfail
    whose ``raises`` is the dedicated subclass. known_gap affected fields
    must be in that profile's measured ``mismatch_fields``.

    ignored_compatibility and not_fe_honored rows bind only their fixed
    ordinary case-ID sets and verbatim identifier tuples. Those tests must
    stay free of xfail/skip. use_remote_indexer and session_affinity_ttl_secs
    bind only their exact existing collected Items.
    """
    item_list = [_require_collected_item(item) for item in items]
    collected: dict[str, list[Any]] = {}
    for item in item_list:
        nodeid = item.nodeid
        for name in PROFILE_NAMES:
            if f"test_effective_config_profile[{name}]" not in nodeid:
                continue
            marks = list(item.iter_markers(name="xfail"))
            if name in collected:
                raise AssertionError(f"profile {name} collected more than once")
            collected[name] = marks
    missing_profiles = [name for name in PROFILE_NAMES if name not in collected]
    if missing_profiles:
        raise AssertionError(
            "profile names are not collected as tests: " + ", ".join(missing_profiles)
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
    for name, spec in MEASURED_GAP_SIGNATURES.items():
        asserted = set(PROFILE_ASSERTED_FIELDS[name])
        if set(spec.frontend) != asserted or set(spec.epp) != asserted:
            raise AssertionError(
                f"{name} measured signature field keys "
                f"frontend={sorted(spec.frontend)} epp={sorted(spec.epp)} "
                f"!= asserted {sorted(asserted)}"
            )
    for name, envs in PROFILE_DEPRECATED_ENVS.items():
        unknown = [
            env
            for env in envs
            if env not in _POLICY_PARAMETER_ENVS and env not in _DIRECT_DEPRECATED_ENVS
        ]
        if unknown:
            raise AssertionError(
                f"{name} deprecated envs have no source Deprecated builder: "
                + ", ".join(unknown)
            )
    for name, marks in collected.items():
        spec = MEASURED_GAP_SIGNATURES.get(name)
        if spec is None:
            if marks:
                raise AssertionError(
                    f"{name} is an ordinary profile but collected {len(marks)} "
                    "xfail marker(s)"
                )
            continue
        if len(marks) != 1:
            raise AssertionError(
                f"{name} measured gap must collect exactly one xfail, "
                f"found {len(marks)}"
            )
        _assert_measured_xfail_marker(name, marks[0])
    for row in SETTING_DECISIONS:
        if row.status == STATUS_PENDING:
            continue
        if row.status == STATUS_IGNORED_COMPAT:
            _assert_ignored_compat_binding(row, item_list)
            continue
        if row.status == STATUS_NOT_FE_HONORED:
            _assert_not_fe_honored_binding(row, item_list)
            continue
        if row.name == REMOTE_INDEXER_ROW:
            _assert_remote_indexer_binding(row, item_list)
            continue
        if row.name == SESSION_TTL_ROW:
            _assert_session_ttl_binding(row, item_list)
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
            if row.status != STATUS_KNOWN_GAP:
                continue
            spec = MEASURED_GAP_SIGNATURES.get(profile)
            if spec is None:
                raise AssertionError(
                    f"{row.name} known_gap profile {profile} is not a "
                    "measured gap ID"
                )
            not_gap = [
                field
                for field in row.affected_fields
                if field not in spec.mismatch_fields
            ]
            if not_gap:
                raise AssertionError(
                    f"{row.name} affected_fields {not_gap} are not in "
                    f"{profile} measured mismatch_fields "
                    f"{sorted(spec.mismatch_fields)}"
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
    present: dict[str, Any] = {}
    omitted: list[OmittedDefaultNativeField] = []
    unexpected: list[str] = []
    for name in fields:
        if name in payload:
            present[name] = payload[name]
            continue
        if name in OMITTABLE_DEFAULT_NATIVE_FIELDS:
            evidence = OmittedDefaultNativeField(name)
            omitted.append(evidence)
            present[name] = evidence
            continue
        unexpected.append(name)
    if unexpected:
        raise AssertionError(
            "native default KvRouterConfig omitted sentinel fields "
            f"{unexpected}; allowed skip_serializing_if absences are "
            f"{sorted(OMITTABLE_DEFAULT_NATIVE_FIELDS)}; "
            f"recorded omitted={tuple(omitted)}; do not invent Rust defaults"
        )
    if omitted:
        present["_omitted_default_native_fields"] = tuple(omitted)
    return present


def _unwrap_option_event_raw(name: str, raw: str) -> str | None:
    if name not in OPTION_EVENT_FIELDS:
        if raw == "None" or raw.startswith("Some"):
            raise AssertionError(
                f"{name}={raw!r} uses a None/Some wrapper on a non-Option field"
            )
        return raw
    if raw == "None":
        return None
    match = _SOME_RE.fullmatch(raw)
    if match:
        inner = match.group(1)
        if inner == "" or inner.startswith("Some"):
            raise AssertionError(
                f"{name}={raw!r} is a nested or empty Some(...) tracing form"
            )
        if inner.startswith("(") or (inner.endswith(")") and "(" not in inner):
            raise AssertionError(f"{name}={raw!r} is an ambiguous Some(...) form")
        return inner
    if raw.startswith("Some"):
        raise AssertionError(f"{name}={raw!r} is a malformed Some(...) tracing form")
    return raw


def _decode_event_value(name: str, raw: str, expected: Any) -> Any:
    payload = _unwrap_option_event_raw(name, raw)
    if payload is None:
        return None
    if isinstance(expected, str):
        mapped = _ENUM_EVENT_VALUES.get(payload)
        if mapped is None:
            raise AssertionError(
                f"{name}={raw!r} is not a known enum display/serde tracing form"
            )
        return mapped
    if payload in {"true", "false"}:
        value: Any = payload == "true"
        if isinstance(expected, bool):
            return value
        raise AssertionError(f"{name}={raw!r} decoded bool, expected {expected!r}")
    try:
        if isinstance(expected, bool):
            raise AssertionError(f"{name}={raw!r} is not a bool")
        if isinstance(expected, int) and not isinstance(expected, bool):
            if any(token in payload for token in (".", "e", "E")):
                raise AssertionError(
                    f"{name}={raw!r} is not integer syntax for an integer field"
                )
            value = int(payload)
        else:
            value = float(payload)
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
        raise AssertionError(f"EPP log_path {log} is not this profile node {node_name}")
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


def assert_frontend_expected_fields(
    *,
    profile: str,
    frontend: Mapping[str, Any],
    expected: Mapping[str, Any],
    defaults: Mapping[str, Any],
) -> None:
    for name, wanted in expected.items():
        if name not in frontend:
            raise AssertionError(f"{profile} FE projection missing {name}")
        if not _values_equal(frontend[name], wanted):
            raise AssertionError(
                f"{profile} {name}: FE {frontend[name]!r} != {wanted!r}"
            )
        if name not in defaults:
            continue
        recorded = defaults[name]
        if isinstance(recorded, OmittedDefaultNativeField):
            continue
        if _values_equal(frontend[name], recorded):
            raise AssertionError(
                f"{profile} {name}={frontend[name]!r} equals the native default"
            )


def _projection_matches(got: Mapping[str, Any], wanted: Mapping[str, Any]) -> bool:
    if set(got) != set(wanted):
        return False
    return all(_values_equal(got[name], wanted[name]) for name in wanted)


def _projection_mismatch_detail(
    frontend: Mapping[str, Any],
    epp: Mapping[str, Any],
    mismatches: Mapping[str, tuple[Any, Any]],
) -> str:
    return f"frontend={dict(frontend)} epp={dict(epp)} mismatches={dict(mismatches)}"


def compare_effective_fields(
    *,
    profile: str,
    frontend: Mapping[str, Any],
    epp: Mapping[str, Any],
    expected: Mapping[str, Any],
    defaults: Mapping[str, Any],
) -> None:
    assert_frontend_expected_fields(
        profile=profile,
        frontend=frontend,
        expected=expected,
        defaults=defaults,
    )
    mismatches: dict[str, tuple[Any, Any]] = {}
    for name in expected:
        if name not in epp:
            raise AssertionError(f"{profile} EPP event missing {name}")
        if not _values_equal(frontend[name], epp[name]):
            mismatches[name] = (frontend[name], epp[name])
    LOGGER.info(
        "%s complete projections frontend=%r epp=%r mismatches=%r",
        profile,
        dict(frontend),
        dict(epp),
        mismatches,
    )
    if not mismatches:
        return
    spec = MEASURED_GAP_SIGNATURES.get(profile)
    if (
        spec is not None
        and set(mismatches) == spec.mismatch_fields
        and _projection_matches(frontend, spec.frontend)
        and _projection_matches(epp, spec.epp)
    ):
        raise spec.exception(
            f"{profile} measured gap "
            f"{_projection_mismatch_detail(frontend, epp, mismatches)}"
        )
    raise FrontendEppConfigMismatch(
        f"{profile} {_projection_mismatch_detail(frontend, epp, mismatches)}"
    )


def _expected_deprecation_messages(envs: Iterable[str]) -> tuple[str, ...]:
    from dynamo.common.configuration.groups.kv_router_args import _policy_parameter
    from dynamo.common.configuration.utils import Deprecated

    messages: list[str] = []
    unknown: list[str] = []
    for name in envs:
        parameter = _POLICY_PARAMETER_ENVS.get(name)
        if parameter is not None:
            messages.append(_policy_parameter(parameter).message(name))
            continue
        spec = _DIRECT_DEPRECATED_ENVS.get(name)
        if spec is None:
            unknown.append(name)
            continue
        replacement, remove_in = spec
        messages.append(Deprecated(replacement, remove_in=remove_in).message(name))
    if unknown:
        raise AssertionError(
            "deprecated envs have no authoritative Deprecated builder: "
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
    if name == "tracking_queue":
        expected = {
            "router_replica_sync": True,
            "router_track_output_blocks": True,
            "router_assume_kv_reuse": False,
            "router_track_prefill_tokens": False,
            "router_queue_threshold": 0.85,
            "router_queue_policy": "wspt",
        }
        return ConfigProfile(
            name,
            {
                "DYN_ROUTER_REPLICA_SYNC": "true",
                "DYN_ROUTER_TRACK_OUTPUT_BLOCKS": "true",
                "DYN_ROUTER_ASSUME_KV_REUSE": "false",
                "DYN_ROUTER_TRACK_PREFILL_TOKENS": "false",
                "DYN_ROUTER_QUEUE_THRESHOLD": "0.85",
                "DYN_ROUTER_QUEUE_POLICY": "wspt",
            },
            PROFILE_ASSERTED_FIELDS[name],
            expected,
            PROFILE_DEPRECATED_ENVS[name],
        )
    if name == "active_cache":
        expected = {
            "router_track_active_blocks": False,
            "router_predicted_ttl_secs": 45.0,
        }
        return ConfigProfile(
            name,
            {
                "DYN_ROUTER_TRACK_ACTIVE_BLOCKS": "false",
                "DYN_ROUTER_PREDICTED_TTL_SECS": "45",
            },
            PROFILE_ASSERTED_FIELDS[name],
            expected,
            PROFILE_DEPRECATED_ENVS[name],
        )
    if name == "load_aware_preset":
        expected = {
            "overlap_score_credit": 0.0,
            "use_kv_events": False,
        }
        return ConfigProfile(
            name,
            {"DYN_ROUTER_LOAD_AWARE": "true"},
            PROFILE_ASSERTED_FIELDS[name],
            expected,
            PROFILE_DEPRECATED_ENVS[name],
        )
    if name == "conditional_json":
        expected = {
            "conditional_disagg_enabled": True,
            "conditional_disagg_policy": "isl_or_load",
            "conditional_disagg_eff_isl_threshold": 1024,
            "conditional_disagg_eff_isl_ratio_threshold": 0.25,
            "conditional_disagg_prefill_busy_threshold": 0.8,
            "conditional_disagg_decode_busy_threshold": 0.7,
        }
        return ConfigProfile(
            name,
            {
                "DYN_ROUTER_CONDITIONAL_DISAGG": "true",
                "DYN_ROUTER_CONDITIONAL_DISAGG_CONFIG": _CONDITIONAL_JSON,
            },
            PROFILE_ASSERTED_FIELDS[name],
            expected,
            PROFILE_DEPRECATED_ENVS[name],
        )
    if name == "shared_cache_multiplier":
        expected = {"shared_cache_multiplier": 0.35}
        return ConfigProfile(
            name,
            {"DYN_SHARED_CACHE_MULTIPLIER": "0.35"},
            PROFILE_ASSERTED_FIELDS[name],
            expected,
            PROFILE_DEPRECATED_ENVS[name],
        )
    raise AssertionError(f"unknown config profile {name!r}")


class _RouterArgsWarningHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@dataclass(frozen=True)
class FrontendRouterSnapshot:
    namespace: Mapping[str, Any]
    config_attrs: Mapping[str, Any]
    router_mode: str
    native_router_mode: Any
    router_kwargs: Mapping[str, Any]
    kv_json: Mapping[str, Any]
    router_fields: tuple[str, ...]


def explicit_manifest_env_names() -> tuple[str, ...]:
    names: list[str] = []
    seen: set[str] = set()
    for row in SETTING_DECISIONS:
        for env in row.env:
            if env in seen:
                continue
            seen.add(env)
            names.append(env)
    return tuple(names)


def clear_known_manifest_router_envs(monkeypatch) -> None:
    for name in explicit_manifest_env_names():
        monkeypatch.delenv(name, raising=False)


def _add_argument_env_vars(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg != "env_var":
                continue
            if isinstance(keyword.value, ast.Constant) and isinstance(
                keyword.value.value, str
            ):
                names.add(keyword.value.value)
    return names


def _build_router_config_uses_explicit_kv_constructor() -> bool:
    tree = ast.parse(ROUTER_ARGS.read_text(encoding="utf-8"), filename=str(ROUTER_ARGS))
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name != "build_router_config":
            continue
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            func = child.func
            if not isinstance(func, ast.Name) or func.id != "KvRouterConfig":
                continue
            if child.args:
                continue
            if len(child.keywords) != 1 or child.keywords[0].arg is not None:
                continue
            value = child.keywords[0].value
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and value.func.attr == "kv_router_kwargs"
                and isinstance(value.func.value, ast.Name)
                and value.func.value.id == "config"
                and not value.args
                and not value.keywords
            ):
                return True
    return False


def _unique_scoped_rust_body(text: str, pattern: re.Pattern[str], label: str) -> str:
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise AssertionError(
            f"{label} is missing, duplicate, or ambiguous: found {len(matches)}"
        )
    start = matches[0].end() - 1
    if start < 0 or text[start] != "{":
        raise AssertionError(f"{label} is not followed by a scoped body")
    depth = 1
    index = start + 1
    while index < len(text) and depth:
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
        index += 1
    if depth:
        raise AssertionError(f"{label} has an unbalanced scoped body")
    return text[start + 1 : index - 1]


def assert_enforce_disagg_conversion_forces_false() -> None:
    if not RUST_PYO3_ENTRYPOINT.is_file():
        raise AssertionError(
            f"missing PyO3 RouterConfig source: {RUST_PYO3_ENTRYPOINT}"
        )
    text = _strip_rust_comments(RUST_PYO3_ENTRYPOINT.read_text(encoding="utf-8"))
    impl_body = _unique_scoped_rust_body(
        text,
        re.compile(r"impl\s+From<RouterConfig>\s+for\s+RsRouterConfig\s*\{"),
        "impl From<RouterConfig> for RsRouterConfig",
    )
    from_body = _unique_scoped_rust_body(
        impl_body,
        re.compile(
            r"fn\s+from\s*\(\s*rc\s*:\s*RouterConfig\s*\)\s*->\s*RsRouterConfig\s*\{"
        ),
        "fn from(rc: RouterConfig) -> RsRouterConfig",
    )
    struct_body = _unique_scoped_rust_body(
        from_body,
        re.compile(r"RsRouterConfig\s*\{"),
        "returned RsRouterConfig { ... }",
    )
    assigns = list(re.finditer(r"enforce_disagg\s*:\s*([^,}+]+)", struct_body))
    if len(assigns) != 1:
        raise AssertionError(
            "conversion enforce_disagg assignment is missing, duplicate, "
            f"or ambiguous: found {len(assigns)}"
        )
    if assigns[0].group(1).strip() != "false":
        raise AssertionError(
            "PyO3 RouterConfig conversion no longer forces enforce_disagg: false"
        )


def assert_ignored_inputs_excluded_from_router_fields() -> None:
    from dynamo.common.configuration.groups.router_args import _ROUTER_FIELDS

    leaked = [
        name
        for name in ("admission_control", "enforce_disagg")
        if name in _ROUTER_FIELDS
    ]
    if leaked:
        raise AssertionError(
            "_ROUTER_FIELDS now forwards ignored inputs: " + ", ".join(leaked)
        )


CONDITIONAL_SCALAR_RUST_READS = {
    "DYN_ROUTER_CONDITIONAL_DISAGG_POLICY": (
        'get_env("DYN_ROUTER_CONDITIONAL_DISAGG_POLICY")'
    ),
    "DYN_ROUTER_CONDITIONAL_DISAGG_EFF_ISL_THRESHOLD": (
        'parse_usize(&get_env,"DYN_ROUTER_CONDITIONAL_DISAGG_EFF_ISL_THRESHOLD")'
    ),
    "DYN_ROUTER_CONDITIONAL_DISAGG_EFF_ISL_RATIO_THRESHOLD": (
        'parse_f64(&get_env,"DYN_ROUTER_CONDITIONAL_DISAGG_EFF_ISL_RATIO_THRESHOLD")'
    ),
    "DYN_ROUTER_CONDITIONAL_DISAGG_PREFILL_BUSY_THRESHOLD": (
        'parse_f64(&get_env,"DYN_ROUTER_CONDITIONAL_DISAGG_PREFILL_BUSY_THRESHOLD")'
    ),
    "DYN_ROUTER_CONDITIONAL_DISAGG_DECODE_BUSY_THRESHOLD": (
        'parse_f64(&get_env,"DYN_ROUTER_CONDITIONAL_DISAGG_DECODE_BUSY_THRESHOLD")'
    ),
}


def _docstring_constants(tree: ast.AST) -> set[ast.AST]:
    docstrings: set[ast.AST] = set()
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        if not node.body:
            continue
        first = node.body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            docstrings.add(first.value)
    return docstrings


def _python_non_docstring_string_literals(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    skip = _docstring_constants(tree)
    literals: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        if node in skip:
            continue
        literals.append(node.value)
    return literals


def _count_token_in_python_source(path: Path, token: str) -> int:
    return sum(
        literal.count(token) for literal in _python_non_docstring_string_literals(path)
    )


def _frontend_python_source_paths() -> list[Path]:
    frontend_pkg = FRONTEND_ARGS.parent
    if not frontend_pkg.is_dir():
        raise AssertionError(f"missing frontend package: {frontend_pkg}")
    paths = sorted(frontend_pkg.rglob("*.py"))
    for extra in (ROUTER_ARGS, KV_ROUTER_ARGS):
        if extra not in paths:
            paths.append(extra)
    return paths


def _production_rust_source_paths() -> list[Path]:
    paths: list[Path] = []
    for tree in RUST_TREES:
        if not tree.exists():
            raise AssertionError(f"missing Rust source tree: {tree}")
        paths.extend(sorted(tree.rglob("*.rs")))
    if not RUST_PYO3_ENTRYPOINT.is_file():
        raise AssertionError(
            f"missing PyO3 RouterConfig source: {RUST_PYO3_ENTRYPOINT}"
        )
    paths.append(RUST_PYO3_ENTRYPOINT)
    return paths


def _stripped_production_rust_text(path: Path) -> str:
    return _drop_cfg_test_modules(
        _strip_rust_comments(path.read_text(encoding="utf-8"))
    )


def _count_token_in_stripped_rust(path: Path, token: str) -> int:
    return _stripped_production_rust_text(path).count(token)


def _rust_read_spelling_re(spelling: str) -> re.Pattern[str]:
    """Match one collapsed helper call, allowing Rust's trailing argument comma."""
    if not spelling.endswith(")"):
        raise AssertionError(f"read spelling {spelling!r} does not end with ')'")
    return re.compile(re.escape(spelling[:-1]) + r",?\)")


def _assert_conditional_scalar_source_sites() -> None:
    if set(CONDITIONAL_SCALAR_RUST_READS) != set(CONDITIONAL_SCALAR_ENV_NAMES):
        raise AssertionError(
            "conditional scalar read-spelling keys "
            f"{sorted(CONDITIONAL_SCALAR_RUST_READS)} != "
            f"{list(CONDITIONAL_SCALAR_ENV_NAMES)}"
        )
    rust_sites: dict[str, list[str]] = {
        name: [] for name in CONDITIONAL_SCALAR_ENV_NAMES
    }
    for path in _production_rust_source_paths():
        for name in CONDITIONAL_SCALAR_ENV_NAMES:
            count = _count_token_in_stripped_rust(path, name)
            if count:
                rust_sites[name].extend([str(path)] * count)
    python_sites: dict[str, list[str]] = {
        name: [] for name in CONDITIONAL_SCALAR_ENV_NAMES
    }
    for path in _frontend_python_source_paths():
        for name in CONDITIONAL_SCALAR_ENV_NAMES:
            count = _count_token_in_python_source(path, name)
            if count:
                python_sites[name].extend([str(path)] * count)
    for name in CONDITIONAL_SCALAR_ENV_NAMES:
        rust_hits = rust_sites[name]
        python_hits = python_sites[name]
        expected_rust = [str(RUST_CONFIG)]
        if rust_hits != expected_rust or python_hits:
            raise AssertionError(
                f"{name} source sites require review: rust={rust_hits} "
                f"expected rust={expected_rust} python={python_hits} "
                "expected python=[]"
            )
    collapsed = re.sub(r"\s+", "", _stripped_production_rust_text(RUST_CONFIG))
    missing_reads = [
        name
        for name, spelling in CONDITIONAL_SCALAR_RUST_READS.items()
        if _rust_read_spelling_re(spelling).search(collapsed) is None
    ]
    if missing_reads:
        raise AssertionError(
            "production parser no longer reads "
            + ", ".join(missing_reads)
            + " with get_env/parse_usize/parse_f64"
        )


def assert_conditional_scalar_envs_are_not_fe_consumers() -> None:
    _assert_conditional_scalar_source_sites()
    rust_names = _rust_env_literals(RUST_CONFIG)
    missing_rust = [
        name for name in CONDITIONAL_SCALAR_ENV_NAMES if name not in rust_names
    ]
    if missing_rust:
        raise AssertionError(
            "Rust kv_router_config_from_lookup no longer reads "
            + ", ".join(missing_rust)
        )
    fe_env_vars = (
        _add_argument_env_vars(KV_ROUTER_ARGS)
        | _add_argument_env_vars(ROUTER_ARGS)
        | _add_argument_env_vars(FRONTEND_ARGS)
    )
    leaked = [name for name in CONDITIONAL_SCALAR_ENV_NAMES if name in fe_env_vars]
    if leaked:
        raise AssertionError(
            "FE argparse now consumes conditional scalar envs: " + ", ".join(leaked)
        )
    required_fe = {
        "DYN_ROUTER_CONDITIONAL_DISAGG",
        "DYN_ROUTER_CONDITIONAL_DISAGG_CONFIG",
    }
    missing_fe = sorted(required_fe - fe_env_vars)
    if missing_fe:
        raise AssertionError(
            "FE no longer exposes conditional enable/JSON: " + ", ".join(missing_fe)
        )
    from dynamo.common.configuration.groups.kv_router_args import (
        _CONDITIONAL_DISAGG_CONFIG_FIELDS,
    )

    expected_fields = {
        "policy": "conditional_disagg_policy",
        "eff_isl_threshold": "conditional_disagg_eff_isl_threshold",
        "eff_isl_ratio_threshold": "conditional_disagg_eff_isl_ratio_threshold",
        "prefill_busy_threshold": "conditional_disagg_prefill_busy_threshold",
        "decode_busy_threshold": "conditional_disagg_decode_busy_threshold",
    }
    if dict(_CONDITIONAL_DISAGG_CONFIG_FIELDS) != expected_fields:
        raise AssertionError(
            "FE _CONDITIONAL_DISAGG_CONFIG_FIELDS changed: "
            f"{dict(_CONDITIONAL_DISAGG_CONFIG_FIELDS)}"
        )
    if not _build_router_config_uses_explicit_kv_constructor():
        raise AssertionError(
            "build_router_config no longer calls "
            "KvRouterConfig(**config.kv_router_kwargs())"
        )
    pyo3 = _strip_rust_comments(RUST_PYO3_ENTRYPOINT.read_text(encoding="utf-8"))
    for banned in (
        "kv_router_config_from_lookup",
        "try_kv_router_config_from_dynamo_env",
        "kv_router_config_from_dynamo_env",
    ):
        if banned in pyo3:
            raise AssertionError(f"PyO3 KvRouterConfig path now calls {banned}")


def _capture_router_args_warnings(build):
    from dynamo.common.configuration.groups.router_args import logger as router_logger

    handler = _RouterArgsWarningHandler()
    router_logger.addHandler(handler)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            snapshot = build()
    finally:
        router_logger.removeHandler(handler)
    return snapshot, handler.records


def _frontend_router_snapshot(
    extra_cli: tuple[str, ...] = (),
) -> FrontendRouterSnapshot:
    from dynamo.common.configuration.groups.router_args import (
        _ROUTER_FIELDS,
        build_router_config,
    )
    from dynamo.frontend.frontend_args import FrontendArgGroup, FrontendConfig
    from dynamo.llm import RouterMode

    parser = argparse.ArgumentParser(add_help=False)
    FrontendArgGroup().add_arguments(parser)
    namespace = parser.parse_args(list(WRAPPER_CLI) + list(extra_cli))
    config = FrontendConfig.from_cli_args(namespace)
    config.validate()
    router = build_router_config(config)
    if router is None or router.kv_router_config is None:
        raise AssertionError("build_router_config did not produce KvRouterConfig")
    if router.router_mode != RouterMode.KV:
        raise AssertionError(f"native router_mode {router.router_mode!r} is not KV")
    if config.router_mode != "kv":
        raise AssertionError(f"config.router_mode {config.router_mode!r} is not kv")
    return FrontendRouterSnapshot(
        dict(vars(namespace)),
        dict(vars(config)),
        config.router_mode,
        router.router_mode,
        dict(config.router_kwargs()),
        json.loads(router.kv_router_config.to_json()),
        tuple(_ROUTER_FIELDS),
    )


def _frontend_full_kv_json() -> dict[str, Any]:
    snapshot = _frontend_router_snapshot()
    return dict(snapshot.kv_json)


def _require_router_args_warning(
    records: list[logging.LogRecord],
    *,
    logger_name: str,
    message: str,
) -> None:
    matches = [
        record
        for record in records
        if record.name == logger_name and record.getMessage() == message
    ]
    if len(records) != 1 or len(matches) != 1:
        got = [(record.name, record.getMessage()) for record in records]
        raise AssertionError(
            f"expected exactly one {logger_name!r} warning {message!r}, got {got}"
        )


def _assert_ignored_options_absent(snapshot: FrontendRouterSnapshot) -> None:
    if "admission_control" in snapshot.namespace:
        raise AssertionError("admission_control was stored on the argparse namespace")
    if "admission_control" in snapshot.config_attrs:
        raise AssertionError("admission_control was stored on FrontendConfig")
    leaked = [
        name
        for name in ("admission_control", "enforce_disagg")
        if name in snapshot.router_kwargs
    ]
    if leaked:
        raise AssertionError(
            "ignored options entered router_kwargs(): " + ", ".join(leaked)
        )
    if "admission_control" in snapshot.router_fields:
        raise AssertionError("_ROUTER_FIELDS now includes admission_control")
    if "enforce_disagg" in snapshot.router_fields:
        raise AssertionError("_ROUTER_FIELDS now includes enforce_disagg")


def _assert_snapshots_match(
    baseline: FrontendRouterSnapshot, changed: FrontendRouterSnapshot
) -> None:
    if baseline.router_mode != changed.router_mode:
        raise AssertionError(
            f"router_mode changed {baseline.router_mode!r} -> {changed.router_mode!r}"
        )
    if baseline.native_router_mode != changed.native_router_mode:
        raise AssertionError(
            "native router_mode changed "
            f"{baseline.native_router_mode!r} -> {changed.native_router_mode!r}"
        )
    if dict(baseline.router_kwargs) != dict(changed.router_kwargs):
        raise AssertionError(
            "router_kwargs changed "
            f"{dict(baseline.router_kwargs)} -> {dict(changed.router_kwargs)}"
        )
    if dict(baseline.kv_json) != dict(changed.kv_json):
        raise AssertionError(
            "kv_router_config.to_json changed "
            f"{dict(baseline.kv_json)} -> {dict(changed.kv_json)}"
        )
    LOGGER.info(
        "ignored-compat snapshots baseline_json=%r changed_json=%r kwargs=%r",
        dict(baseline.kv_json),
        dict(changed.kv_json),
        dict(changed.router_kwargs),
    )


def _run_ignored_compat_case(
    monkeypatch,
    *,
    extra_cli: tuple[str, ...] = (),
    env_name: str | None = None,
    env_value: str | None = None,
    warning_message: str,
) -> None:
    from dynamo.common.configuration.groups.router_args import logger as router_logger

    assert_ignored_inputs_excluded_from_router_fields()
    assert_enforce_disagg_conversion_forces_false()
    clear_known_manifest_router_envs(monkeypatch)
    baseline, baseline_records = _capture_router_args_warnings(
        _frontend_router_snapshot
    )
    if baseline_records:
        got = [(record.name, record.getMessage()) for record in baseline_records]
        raise AssertionError(f"baseline parse emitted unexpected warnings {got}")
    _assert_ignored_options_absent(baseline)
    if env_name is not None:
        if env_value is None:
            raise AssertionError(f"{env_name} probe is missing a sentinel")
        monkeypatch.setenv(env_name, env_value)
    changed, changed_records = _capture_router_args_warnings(
        lambda: _frontend_router_snapshot(extra_cli)
    )
    _require_router_args_warning(
        changed_records,
        logger_name=router_logger.name,
        message=warning_message,
    )
    _assert_ignored_options_absent(changed)
    _assert_snapshots_match(baseline, changed)


def run_ignored_admission_control(monkeypatch, case_id: str) -> None:
    from dynamo.common.configuration.groups.router_args import (
        _ADMISSION_CONTROL_FLAG_REMOVAL_WARNING,
        _ADMISSION_CONTROL_REMOVAL_WARNING,
    )

    if case_id == "cli_token_capacity":
        _run_ignored_compat_case(
            monkeypatch,
            extra_cli=("--admission-control", "token-capacity"),
            warning_message=_ADMISSION_CONTROL_FLAG_REMOVAL_WARNING,
        )
        return
    if case_id == "cli_none":
        _run_ignored_compat_case(
            monkeypatch,
            extra_cli=("--admission-control", "none"),
            warning_message=_ADMISSION_CONTROL_FLAG_REMOVAL_WARNING,
        )
        return
    if case_id == "env_token_capacity":
        _run_ignored_compat_case(
            monkeypatch,
            env_name="DYN_ADMISSION_CONTROL",
            env_value="token-capacity",
            warning_message=_ADMISSION_CONTROL_REMOVAL_WARNING,
        )
        return
    if case_id == "env_none":
        _run_ignored_compat_case(
            monkeypatch,
            env_name="DYN_ADMISSION_CONTROL",
            env_value="none",
            warning_message=_ADMISSION_CONTROL_REMOVAL_WARNING,
        )
        return
    raise AssertionError(f"unknown admission_control case {case_id!r}")


def run_ignored_enforce_disagg(monkeypatch, case_id: str) -> None:
    from dynamo.common.configuration.groups.router_args import (
        _ENFORCE_DISAGG_DEPRECATION,
    )

    if case_id == "cli_positive":
        _run_ignored_compat_case(
            monkeypatch,
            extra_cli=("--enforce-disagg",),
            warning_message=_ENFORCE_DISAGG_DEPRECATION % "--enforce-disagg",
        )
        return
    if case_id == "cli_negative":
        _run_ignored_compat_case(
            monkeypatch,
            extra_cli=("--no-enforce-disagg",),
            warning_message=_ENFORCE_DISAGG_DEPRECATION % "--no-enforce-disagg",
        )
        return
    if case_id == "env_true":
        _run_ignored_compat_case(
            monkeypatch,
            env_name="DYN_ENFORCE_DISAGG",
            env_value="true",
            warning_message=_ENFORCE_DISAGG_DEPRECATION % "DYN_ENFORCE_DISAGG",
        )
        return
    if case_id == "env_false":
        _run_ignored_compat_case(
            monkeypatch,
            env_name="DYN_ENFORCE_DISAGG",
            env_value="false",
            warning_message=_ENFORCE_DISAGG_DEPRECATION % "DYN_ENFORCE_DISAGG",
        )
        return
    raise AssertionError(f"unknown enforce_disagg case {case_id!r}")


def run_conditional_scalar_not_fe_honored(
    monkeypatch, env_name: str, sentinel: str
) -> None:
    assert_conditional_scalar_envs_are_not_fe_consumers()
    if (env_name, sentinel) not in CONDITIONAL_SCALAR_PROBES:
        raise AssertionError(f"unknown conditional scalar probe {(env_name, sentinel)}")
    clear_known_manifest_router_envs(monkeypatch)
    monkeypatch.setenv("DYN_ROUTER_CONDITIONAL_DISAGG", "true")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        baseline = _frontend_full_kv_json()
    if baseline.get("conditional_disagg_enabled") is not True:
        raise AssertionError(
            "FE baseline conditional enable is not True: "
            f"{baseline.get('conditional_disagg_enabled')!r}"
        )
    monkeypatch.setenv(env_name, sentinel)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        changed = _frontend_full_kv_json()
    LOGGER.info(
        "conditional scalar %s=%s baseline=%r changed=%r",
        env_name,
        sentinel,
        baseline,
        changed,
    )
    if baseline != changed:
        raise AssertionError(
            f"{env_name}={sentinel!r} changed FE native JSON "
            f"{baseline} -> {changed}"
        )


def assert_session_inputs_stack_uses_ttl_300() -> None:
    if not INPUTS_PARITY.is_file():
        raise AssertionError(f"missing session inputs thin test: {INPUTS_PARITY}")
    tree = ast.parse(
        INPUTS_PARITY.read_text(encoding="utf-8"), filename=str(INPUTS_PARITY)
    )
    found = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name != "test_epp_frontend_inputs_parity":
            continue
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            func = child.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name != "EppFrontendRoutingParityStack":
                continue
            for keyword in child.keywords:
                if keyword.arg != "session_affinity_ttl_secs":
                    continue
                if (
                    isinstance(keyword.value, ast.Constant)
                    and keyword.value.value == SESSION_TTL_SECS
                ):
                    found = True
    if not found:
        raise AssertionError(
            "session_id stack no longer passes "
            f"session_affinity_ttl_secs={SESSION_TTL_SECS}"
        )


def _assert_indexer_test_modules_are_cfg_gated() -> None:
    if not KV_INDEXER_MOD.is_file():
        raise AssertionError(f"missing kv-router indexer module: {KV_INDEXER_MOD}")
    text = _strip_rust_comments(KV_INDEXER_MOD.read_text(encoding="utf-8"))
    if re.search(r"#\[cfg\(test\)\]\s*mod\s+tests;", text) is None:
        raise AssertionError(
            "indexer/mod.rs no longer gates tests.rs with #[cfg(test)]"
        )
    if re.search(r"#\[cfg\(test\)\]\s*mod\s+delegate_tests;", text) is None:
        raise AssertionError(
            "indexer/mod.rs no longer gates delegate_tests.rs with #[cfg(test)]"
        )


def _positional_excluded_rust_paths() -> frozenset[Path]:
    _assert_indexer_test_modules_are_cfg_gated()
    return frozenset({INDEXER_TESTS_RS, INDEXER_DELEGATE_TESTS_RS})


def _remove_unique_statement(text: str, statement: str, label: str) -> str:
    count = text.count(statement)
    if count != 1:
        raise AssertionError(f"{label} {statement!r} count {count} != 1")
    return text.replace(statement, "", 1)


def _positional_reference_hits(text: str) -> list[str]:
    return [match.group(0) for match in _POSITIONAL_REF_RE.finditer(text)]


def _assert_positional_lib_exports_only() -> None:
    if not KV_ROUTER_LIB.is_file():
        raise AssertionError(f"missing kv-router lib.rs: {KV_ROUTER_LIB}")
    text = _strip_rust_comments(KV_ROUTER_LIB.read_text(encoding="utf-8"))
    remainder = text
    for statement in POSITIONAL_LIB_EXPORTS:
        remainder = _remove_unique_statement(
            remainder, statement, "kv-router lib.rs export"
        )
    extra = _positional_reference_hits(remainder)
    if extra:
        raise AssertionError(
            "kv-router lib.rs has Positional references beyond the two "
            f"backcompat exports: {extra}"
        )


def _assert_positional_indexer_mod_declaration_only() -> None:
    if not KV_INDEXER_MOD.is_file():
        raise AssertionError(f"missing kv-router indexer module: {KV_INDEXER_MOD}")
    text = _strip_rust_comments(KV_INDEXER_MOD.read_text(encoding="utf-8"))
    remainder = _remove_unique_statement(
        text, POSITIONAL_INDEXER_MOD, "indexer/mod.rs module"
    )
    extra = _positional_reference_hits(remainder)
    if extra:
        raise AssertionError(
            "indexer/mod.rs has Positional references beyond "
            f"{POSITIONAL_INDEXER_MOD!r}: {extra}"
        )


def _assert_positional_factory_uses_radix() -> None:
    missing = [
        str(path)
        for path in (LLM_INDEXER_MOD, KV_INDEXER_RS, POSITIONAL_RS)
        if not path.is_file()
    ]
    if missing:
        raise AssertionError("positional factory sources missing: " + ", ".join(missing))
    factory = _stripped_production_rust_text(LLM_INDEXER_MOD)
    if _positional_reference_hits(factory) or POSITIONAL_ENV in factory:
        raise AssertionError(
            "FE/EPP indexer factory now references PositionalIndexer or "
            f"{POSITIONAL_ENV}"
        )
    if "ConcurrentRadixTreeCompressed" not in factory or "KvIndexer" not in factory:
        raise AssertionError(
            "FE/EPP indexer factory no longer chooses "
            "ConcurrentRadixTreeCompressed or KvIndexer"
        )
    kv_indexer = _stripped_production_rust_text(KV_INDEXER_RS)
    if _positional_reference_hits(kv_indexer) or POSITIONAL_ENV in kv_indexer:
        raise AssertionError("KvIndexer now references PositionalIndexer or the env")
    if "RadixTree::new" not in kv_indexer:
        raise AssertionError("KvIndexer no longer constructs RadixTree::new")
    positional = _stripped_production_rust_text(POSITIONAL_RS)
    if positional.count("SearchMode::from_env()") != 1:
        raise AssertionError(
            "positional.rs must contain exactly one production "
            "SearchMode::from_env() call"
        )
    new_body = _unique_scoped_rust_body(
        positional,
        re.compile(r"pub fn new\s*\(\s*jump_size\s*:\s*usize\s*\)\s*->\s*Self\s*\{"),
        "PositionalIndexer::new",
    )
    if "SearchMode::from_env()" not in new_body:
        raise AssertionError(
            "PositionalIndexer::new no longer calls SearchMode::from_env()"
        )


def _assert_positional_no_new_fe_consumer() -> None:
    excluded = _positional_excluded_rust_paths()
    rust_sites: list[str] = []
    for path in _production_rust_source_paths():
        if path in excluded or path == POSITIONAL_RS:
            continue
        text = _stripped_production_rust_text(path)
        if path == KV_ROUTER_LIB:
            for statement in POSITIONAL_LIB_EXPORTS:
                text = _remove_unique_statement(
                    text, statement, "kv-router lib.rs export"
                )
        elif path == KV_INDEXER_MOD:
            text = _remove_unique_statement(
                text, POSITIONAL_INDEXER_MOD, "indexer/mod.rs module"
            )
        hits = _positional_reference_hits(text)
        if POSITIONAL_ENV in text:
            hits.append(POSITIONAL_ENV)
        if hits:
            rust_sites.append(f"{path}:{hits}")
    if rust_sites:
        raise AssertionError(
            "production FE/EPP trees now reference Positional aliases: "
            + ", ".join(rust_sites)
        )
    python_sites: list[str] = []
    for path in _frontend_python_source_paths():
        for token in (
            POSITIONAL_ENV,
            "PositionalIndexer",
            "nested_map",
            "indexer.positional",
            "indexer::positional",
        ):
            if _count_token_in_python_source(path, token):
                python_sites.append(f"{path}:{token}")
    if python_sites:
        raise AssertionError(
            "FE Python sources now reference positional search: "
            + ", ".join(python_sites)
        )


def assert_positional_search_is_not_fe_consumer() -> None:
    _assert_positional_lib_exports_only()
    _assert_positional_indexer_mod_declaration_only()
    _assert_positional_factory_uses_radix()
    _assert_positional_no_new_fe_consumer()


def run_positional_search_not_fe_honored(monkeypatch) -> None:
    assert_positional_search_is_not_fe_consumer()
    clear_known_manifest_router_envs(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        baseline = _frontend_full_kv_json()
    monkeypatch.setenv(POSITIONAL_ENV, POSITIONAL_SENTINEL)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        changed = _frontend_full_kv_json()
    LOGGER.info(
        "positional search %s=%s baseline=%r changed=%r",
        POSITIONAL_ENV,
        POSITIONAL_SENTINEL,
        baseline,
        changed,
    )
    if baseline != changed:
        raise AssertionError(
            f"{POSITIONAL_ENV}={POSITIONAL_SENTINEL!r} changed FE native JSON "
            f"{baseline} -> {changed}"
        )


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
    assert_frontend_expected_fields(
        profile=profile.name,
        frontend=frontend,
        expected=profile.expected,
        defaults=defaults,
    )
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
