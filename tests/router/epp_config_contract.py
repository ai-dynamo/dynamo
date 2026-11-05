# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FE/EPP effective-config contract for issue #16592.

This is a staged partial configuration suite. Explicit pending rows and
their reasons are the unmeasured list. Discovered CLI/env/schema names
without a row, stale names, and false test bindings still fail. Do not
treat inventory or bindings as full-issue completion.

``load_aware_preset`` and ``conditional_json`` are measured strict gaps
(#16592): dedicated subclasses of ``FrontendEppConfigMismatch`` with
strict xfail. Setup, warning, FE sentinel, parse, and other mismatches
stay ordinary hard failures. Other profiles stay ordinary tests.

``admission_control`` and ``enforce_disagg`` are ignored-compatibility
parser contracts. The five conditional scalar envs and
``positional_search_mode`` are separate not-FE-honored rows.
``use_remote_indexer`` binds the existing first-selection query-counter
observer. ``session_affinity_ttl_secs`` binds TTL enable-presence only on
the existing session_id gap. ``overlap_refresh_after_secs`` is a
supported_candidate dual-process queue-construction INFO plus plain-chat
token-parity observer. ``router_tracking_hash`` and
``router_tracking_key_id`` bind the keyed-tracking runtime Item.
``router_tracking_key_file`` binds that runtime Item and the owned
frontend/epp size-control Items. ``router_approximate_cache_policy``
binds the owned core-metrics LRU and default-TTL Items. These four rows
are unverified candidates until fixed-SHA Linux execution.
``active_decode_blocks_threshold`` is a measured known gap: FE configured
0.25 vs EPP monitor None after the default-None control, on one owned
worker-monitor remote-load Item. ``enable_session_prefix_index`` is a
measured known gap: env-true/kv-false FE validation rejection vs EPP
accept-and-route after omitted/false/true and dual-invalid-YAML controls,
on one owned config-parity Item. ``shared_cache_type`` binds one ordinary
owned HiCache client-construction Item. ``shared_cache_multiplier`` is a
supported_candidate profile pending runtime evidence. ``kv_cache_block_size``
is not-FE-honored Dynamic worker-card geometry: independent CLI 8/64 did
not override card 32, and the card 16 control yielded 16.
``serve_indexer``, ``router_mode``, and both prefill-admission threshold
rows are source-backed ``unsupported_input`` on the stock Runtime EPP
profile. ``min_initial_workers``, ``session_affinity_mode``, and
``router_prefill_load_model`` are source-backed ``unsupported_capability``
(min-worker env parses, but stock EPP bypasses the N-worker gate). Prefill
abs/frac also bind three ordinary FE admin ``/busy_threshold`` boundary
Items. The four native active-sequence envs are ``supported_candidate``
live consumer profiles. The three policy selectors are
``unsupported_capability`` for the stock Runtime EPP named-provider
catalog (default still works; FE named-default is measured).

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
import hashlib
import json
import logging
import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from tests.utils.router_logs import completed_utf8_log_prefix

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
POSITIONAL_ITEM = f"tests/router/test_epp_frontend_config_parity.py::{POSITIONAL_TEST}"
OVERLAP_REFRESH_ROW = "overlap_refresh_after_secs"
OVERLAP_REFRESH_OBSERVER = "owned_fe_epp_queue_refresh_info+plain_chat_token_parity"
OVERLAP_REFRESH_TEST = "test_overlap_refresh_effective_config"
OVERLAP_REFRESH_ITEM = (
    f"tests/router/test_epp_frontend_config_parity.py::{OVERLAP_REFRESH_TEST}"
)
OVERLAP_REFRESH_ENV = "DYN_ROUTER_OVERLAP_REFRESH_AFTER_SECS"
OVERLAP_REFRESH_SENTINEL = "0"
OVERLAP_REFRESH_DEFAULT_SECS = 10
OVERLAP_REFRESH_DISABLED = (
    "Router queue overlap-score refresh disabled via "
    "DYN_ROUTER_OVERLAP_REFRESH_AFTER_SECS"
)
OVERLAP_REFRESH_ENABLED = "Router queue overlap-score refresh enabled after"
OVERLAP_REFRESH_TARGET = "dynamo_kv_router::scheduling::queue"
OVERLAP_REFRESH_LOG_FLUSH_S = 2.0
OVERLAP_REFRESH_RS = REPO_ROOT / "lib/kv-router/src/scheduling/overlap_refresh.rs"
TRACKING_HASH_ROW = "router_tracking_hash"
TRACKING_KEY_FILE_ROW = "router_tracking_key_file"
TRACKING_KEY_ID_ROW = "router_tracking_key_id"
APPROXIMATE_POLICY_ROW = "router_approximate_cache_policy"
TRACKING_RUNTIME_OBSERVER = (
    "fe_native_to_json+epp_init_event+owned_key_file_read+"
    "plain_chat_token_parity+fe_generation+epp_route"
)
TRACKING_KEY_FILE_OBSERVER = (
    "owned_fe_epp_key_file_read+size_diagnostic+keyed_tracking_runtime"
)
APPROXIMATE_POLICY_OBSERVER = (
    "owned_fe_epp_core_system_metrics+plain_chat_token_parity+default_ttl_control"
)
KEYED_TRACKING_PROFILE = "keyed_tracking"
KEYED_TRACKING_RUNTIME_TEST = "test_keyed_tracking_runtime_config"
KEYED_TRACKING_RUNTIME_ITEM = (
    "tests/router/test_epp_frontend_config_parity.py::"
    f"{KEYED_TRACKING_RUNTIME_TEST}[{KEYED_TRACKING_PROFILE}]"
)
KEYED_TRACKING_SIZE_TEST = "test_keyed_tracking_key_file_size"
KEYED_TRACKING_SIZE_ARMS = ("frontend", "epp")
KEYED_TRACKING_SIZE_ITEMS = tuple(
    "tests/router/test_epp_frontend_config_parity.py::"
    f"{KEYED_TRACKING_SIZE_TEST}[{arm}]"
    for arm in KEYED_TRACKING_SIZE_ARMS
)
APPROXIMATE_POLICY_TEST = "test_approximate_cache_policy_metrics"
APPROXIMATE_POLICY_NAMES = ("lru", "ttl")
APPROXIMATE_POLICY_ITEMS = tuple(
    "tests/router/test_epp_frontend_config_parity.py::"
    f"{APPROXIMATE_POLICY_TEST}[{name}]"
    for name in APPROXIMATE_POLICY_NAMES
)
_TRACKING_APPROXIMATE_BINDINGS = MappingProxyType(
    {
        TRACKING_HASH_ROW: MappingProxyType(
            {
                "observer": TRACKING_RUNTIME_OBSERVER,
                "flags": ("--router-tracking-hash",),
                "env": ("DYN_ROUTER_TRACKING_HASH",),
                "schema_fields": ("router_tracking_hash",),
                "affected_fields": ("router_tracking_hash",),
                "items": (KEYED_TRACKING_RUNTIME_ITEM,),
            }
        ),
        TRACKING_KEY_ID_ROW: MappingProxyType(
            {
                "observer": TRACKING_RUNTIME_OBSERVER,
                "flags": ("--router-tracking-key-id",),
                "env": ("DYN_ROUTER_TRACKING_KEY_ID",),
                "schema_fields": ("router_tracking_key_id",),
                "affected_fields": ("router_tracking_key_id",),
                "items": (KEYED_TRACKING_RUNTIME_ITEM,),
            }
        ),
        TRACKING_KEY_FILE_ROW: MappingProxyType(
            {
                "observer": TRACKING_KEY_FILE_OBSERVER,
                "flags": ("--router-tracking-key-file",),
                "env": ("DYN_ROUTER_TRACKING_KEY_FILE",),
                "schema_fields": ("router_tracking_key_file",),
                "affected_fields": ("router_tracking_key_file",),
                "items": (KEYED_TRACKING_RUNTIME_ITEM, *KEYED_TRACKING_SIZE_ITEMS),
            }
        ),
        APPROXIMATE_POLICY_ROW: MappingProxyType(
            {
                "observer": APPROXIMATE_POLICY_OBSERVER,
                "flags": ("--router-approximate-cache-policy",),
                "env": ("DYN_ROUTER_APPROXIMATE_CACHE_POLICY",),
                "schema_fields": ("router_approximate_cache_policy",),
                "affected_fields": ("router_approximate_cache_policy",),
                "items": APPROXIMATE_POLICY_ITEMS,
            }
        ),
    }
)
TRACKING_HASH_ALGORITHM = "keyed-xxh3-v1"
TRACKING_KEY_ID = "parity-public-test-key-v1"
TRACKING_KEY_PUBLIC_BYTES = bytes(range(32))
TRACKING_KEY_INVALID_BYTES = bytes(range(31))
TRACKING_KEY_SIZE = 32
TRACKING_KEY_SIZE_DIAGNOSTIC = (
    f"router tracking key file must contain exactly {TRACKING_KEY_SIZE} "
    "raw bytes; found 31"
)
TRACKING_KEY_SIZE_DEADLINE_S = 60.0
SESSION_PREFIX_INDEX_ROW = "enable_session_prefix_index"
SESSION_PREFIX_INDEX_ENV = "DYN_ENABLE_SESSION_PREFIX_INDEX"
SESSION_PREFIX_INDEX_DIAGNOSTIC = (
    "enable_session_prefix_index requires use_kv_events=true"
)
SESSION_PREFIX_INDEX_DEADLINE_S = 60.0
SESSION_PREFIX_INDEX_FIELDS = (SESSION_PREFIX_INDEX_ROW,)
SESSION_PREFIX_INDEX_OBSERVER = (
    "owned_fe_epp_session_prefix_config_validation+omitted_false_true_controls+"
    "yaml_dual_reject+env_true_kv_false_fe_reject_epp_accept"
)
SESSION_PREFIX_INDEX_TEST = "test_session_prefix_index_config_parity"
SESSION_PREFIX_INDEX_ITEM = (
    f"tests/router/test_epp_frontend_config_parity.py::{SESSION_PREFIX_INDEX_TEST}"
)
TRACKING_RUNTIME_FIELDS = ("router_tracking_hash", "router_tracking_key_id")
APPROXIMATE_CONFIGURED_METRIC = "router_approximate_cache_configured_policy"
APPROXIMATE_EFFECTIVE_METRIC = "router_approximate_cache_effective_policy"
APPROXIMATE_POLICY_LABELS = ("ttl", "lru", "disabled")
DECODE_THRESHOLD_ROW = "active_decode_blocks_threshold"
DECODE_THRESHOLD_OBSERVER = (
    "owned_fe_epp_worker_monitor_remote_load+plain_chat_token_parity+"
    "fe_generation+epp_route+default_none_control"
)
DECODE_THRESHOLD_TEST = "test_active_decode_threshold_consumer"
DECODE_THRESHOLD_ITEM = (
    f"tests/router/test_epp_frontend_config_parity.py::{DECODE_THRESHOLD_TEST}"
)
DECODE_THRESHOLD_ENV = "DYN_ACTIVE_DECODE_BLOCKS_THRESHOLD"
DECODE_THRESHOLD_PREFILL_ENVS = (
    "DYN_ACTIVE_PREFILL_TOKENS_THRESHOLD",
    "DYN_ACTIVE_PREFILL_TOKENS_THRESHOLD_FRAC",
)
DECODE_THRESHOLD_PROFILES = ("default", "configured")
DECODE_THRESHOLD_CONFIGURED = "0.25"
SERVE_INDEXER_ROW = "serve_indexer"
MIN_INITIAL_WORKERS_ROW = "min_initial_workers"
ROUTER_MODE_ROW = "router_mode"
SESSION_AFFINITY_MODE_ROW = "session_affinity_mode"
PREFILL_ABS_THRESHOLD_ROW = "active_prefill_tokens_threshold"
PREFILL_FRAC_THRESHOLD_ROW = "active_prefill_tokens_threshold_frac"
PREFILL_LOAD_MODEL_ROW = "router_prefill_load_model"
INPUT_STARTUP_SOURCE_OBSERVER = (
    "input_startup_source_capability_contract+fe_parser_native_controls"
)
PREFILL_ADMISSION_OBSERVER = (
    "fe_admin_get_busy_threshold+token_route_parity+"
    "input_startup_source_capability_contract"
)
INPUT_STARTUP_SOURCE_TEST = "test_input_startup_capability_decisions"
INPUT_STARTUP_SOURCE_ITEM = (
    f"tests/router/test_epp_frontend_config_parity.py::{INPUT_STARTUP_SOURCE_TEST}"
)
PREFILL_ADMISSION_TEST = "test_prefill_admission_input_boundaries"
PREFILL_ADMISSION_PROFILES = ("default", "absolute", "fractional")
PREFILL_ADMISSION_ITEMS = tuple(
    f"tests/router/test_epp_frontend_config_parity.py::"
    f"{PREFILL_ADMISSION_TEST}[{profile}]"
    for profile in PREFILL_ADMISSION_PROFILES
)
PREFILL_ABS_ENV = "DYN_ACTIVE_PREFILL_TOKENS_THRESHOLD"
PREFILL_FRAC_ENV = "DYN_ACTIVE_PREFILL_TOKENS_THRESHOLD_FRAC"
PREFILL_ABS_SENTINEL = "8192"
PREFILL_FRAC_SENTINEL = "0.75"
ABSOLUTE_PREFILL_THRESHOLD_WARNING = (
    "--active-prefill-tokens-threshold is deprecated and will be removed in v1.7; "
    "use --active-prefill-tokens-threshold-frac"
)
NON_KV_WRAPPER_CLI = ("--router-mode", "round-robin")
INPUT_STARTUP_SOURCE_ROWS = frozenset(
    {
        SERVE_INDEXER_ROW,
        MIN_INITIAL_WORKERS_ROW,
        ROUTER_MODE_ROW,
        SESSION_AFFINITY_MODE_ROW,
        PREFILL_LOAD_MODEL_ROW,
    }
)
UNSUPPORTED_INPUT_ROWS = frozenset(
    {
        SERVE_INDEXER_ROW,
        ROUTER_MODE_ROW,
        PREFILL_ABS_THRESHOLD_ROW,
        PREFILL_FRAC_THRESHOLD_ROW,
    }
)
ROUTER_PREFILL_POLICY_ROW = "router_prefill_policy"
ROUTER_DECODE_POLICY_ROW = "router_decode_policy"
WORKER_SELECTION_POLICY_ROW = "worker_selection_policy"
ACTIVE_REQUEST_EXPIRY_ROW = "active_request_expiry_secs"
ACTIVE_SEQUENCE_DIRECT_ZMQ_ROW = "active_sequence_direct_zmq"
ACTIVE_SEQUENCE_ZMQ_RCVHWM_ROW = "active_sequence_zmq_rcvhwm"
ZMQ_ENDPOINTS_PER_SUB_ROW = "zmq_endpoints_per_sub"
NATIVE_CONSUMER_TEST = "test_native_consumer_profile"
NAMED_POLICY_CATALOG_TEST = "test_named_policy_catalog"
NAMED_POLICY_CATALOG_ITEM = (
    f"tests/router/test_epp_frontend_config_parity.py::{NAMED_POLICY_CATALOG_TEST}"
)


def _native_consumer_helper_surface():
    """Collection-safe helper constants; avoid duplicating profile tables."""
    from tests.router.epp_native_consumers import (
        NATIVE_CONSUMER_OBSERVER,
        NATIVE_CONSUMER_PROFILES,
        NATIVE_CONSUMER_ROWS,
    )

    return NATIVE_CONSUMER_ROWS, NATIVE_CONSUMER_OBSERVER, NATIVE_CONSUMER_PROFILES


def _named_policy_helper_surface():
    from tests.router.epp_selector_catalog import (
        NAMED_POLICY_CATALOG_OBSERVER,
        NAMED_POLICY_CATALOG_ROWS,
    )

    return NAMED_POLICY_CATALOG_ROWS, NAMED_POLICY_CATALOG_OBSERVER


(
    NATIVE_CONSUMER_ROWS,
    NATIVE_CONSUMER_OBSERVER,
    _NATIVE_CONSUMER_PROFILES,
) = _native_consumer_helper_surface()
# Fixed independent wrapper/oracle profile set. Do not derive from helper
# keys: a silent helper deletion would otherwise shrink Items and row
# bindings together.
NATIVE_CONSUMER_PROFILE_NAMES = (
    "omitted",
    "legacy_runtime",
    "grouped",
    "expiry",
)
NATIVE_CONSUMER_PROFILE_ITEMS = MappingProxyType(
    {
        name: (
            "tests/router/test_epp_frontend_config_parity.py::"
            f"{NATIVE_CONSUMER_TEST}[{name}]"
        )
        for name in NATIVE_CONSUMER_PROFILE_NAMES
    }
)
# Exact profiles that actually consume each native field. Independent fixed
# tuples (not helper-derived). legacy_runtime does not claim pool sizing or
# direct-path HWM on the runtime subscriber sockets.
_NATIVE_CONSUMER_ROW_PROFILES = MappingProxyType(
    {
        ACTIVE_REQUEST_EXPIRY_ROW: (
            "omitted",
            "legacy_runtime",
            "grouped",
            "expiry",
        ),
        ACTIVE_SEQUENCE_DIRECT_ZMQ_ROW: (
            "omitted",
            "legacy_runtime",
            "grouped",
            "expiry",
        ),
        ACTIVE_SEQUENCE_ZMQ_RCVHWM_ROW: ("omitted", "grouped", "expiry"),
        ZMQ_ENDPOINTS_PER_SUB_ROW: ("omitted", "grouped", "expiry"),
    }
)
(
    NAMED_POLICY_CATALOG_ROWS,
    NAMED_POLICY_CATALOG_OBSERVER,
) = _named_policy_helper_surface()
UNSUPPORTED_CAPABILITY_ROWS = frozenset(
    {
        MIN_INITIAL_WORKERS_ROW,
        SESSION_AFFINITY_MODE_ROW,
        PREFILL_LOAD_MODEL_ROW,
        ROUTER_PREFILL_POLICY_ROW,
        ROUTER_DECODE_POLICY_ROW,
        WORKER_SELECTION_POLICY_ROW,
    }
)
RUST_EPP = REPO_ROOT / "deploy/inference-gateway/ext-proc/src/epp.rs"
RUST_WATCHER = REPO_ROOT / "lib/llm/src/discovery/watcher.rs"
RUST_MODEL_MANAGER = REPO_ROOT / "lib/llm/src/discovery/model_manager.rs"
RUST_PREFILL_ACTIVATION = (
    REPO_ROOT / "lib/llm/src/kv_router/prefill_router/activation.rs"
)
RUST_KV_ROUTER = REPO_ROOT / "lib/llm/src/kv_router.rs"
RUST_PREFILL_QUERY = REPO_ROOT / "lib/llm/src/kv_router/prefill_router/query.rs"
RUST_EMBEDDED = REPO_ROOT / "lib/llm/src/kv_router/embedded.rs"
RUST_INDEXER_INGRESS = REPO_ROOT / "lib/llm/src/kv_router/indexer/ingress.rs"
RUST_INDEXER_REMOTE = REPO_ROOT / "lib/llm/src/kv_router/indexer/remote.rs"
RUST_AIS_CALLBACK = REPO_ROOT / "lib/bindings/python/rust/llm/ais_callback.rs"
RUST_POLICY_CONFIG = REPO_ROOT / "lib/kv-router/src/scheduling/policy_config.rs"
RUST_QUEUE = REPO_ROOT / "lib/kv-router/src/scheduling/queue.rs"
# Named support-boundary source digests: sha256 of comment-stripped
# declaration+body for each decisive shared scope.
SOURCE_SCOPE_DIGESTS: dict[str, str] = {
    "epp_from_discovery": "fe9ffe7d2862b80c79b4d7150ceea4efd76b45463917adbd60bc77395ff197a5",
    "epp_route_decode": "e80ec535ef961ec3b966e10602d5c5adefd54b9ea7dfeab39ae2bd5f32e30d03",
    "managed_kv_router_for_with_policy": "480f353836bc5cbb248dfbfb7556a99dff7fbaa121a83d2101fc3c4793ecb8e9",
    "kv_chooser_for_with_policy_and_client": "f099fc885a7fc79d7811caed7c37d770ad9cc885229f3e4276a275427e8a0925",
    "prefill_router_new": "c6104e64711807f0ee9ce4bc589a110f2dc2f40956a2b8e87addbb9872f1c02b",
    "embedded_selection_start": "9187ac406b6dc9a4d6a6819e657032332136c3a18a5c9c49f69daf4115a4f100",
    "min_initial_workers_from_env": "73ba458edd8425a67b46dc97ef7d571f3bef9dd13641b9dee15ab7a050a4db2f",
    "ensure_served_indexer_service": "a6972ad24be7ebb2a1d7cc7ca0bc4b36cf4f073f7fbd2efe14ad3b84512aede9",
    "managed_kv_router_for_with_worker_role": "42c2a89cf5eb69d40f854fb3a21dc3cee5b8e5ce00db115a620955f21448c4cf",
    "kv_router_new_with_worker_role_and_scheduler_load": "271eeb24f45fc969465323b5dec871bef4fd7db3d3f2c0e740dd8e5244d91fc9",
    "prefill_router_new_with_selection_policy": "433ea1f40644175e4be7a39df1c48a0bfc8d01fa8f9d6ac28d1293908133b2ce",
    "prefill_build_binding": "25365899cbf3cbf36ecddd31cc5938ee015ca6a4435c69b5c04f9f15fcb19e31",
    "kv_router_config_from_lookup": "af0fecf97c0bd382e3b9c6230410850b1a0625256c08023a382ca36a0305dd34",
    "router_settings_apply": "1e8b69e3e9ebb2c7774eda7a732ba76a1b065996d343ec793e2e78a6dd4f025e",
    "reserve_prefill_worker": "2f3b89345bdd943696cf902c5462b7bb510546c8b43567915904362603c327be",
}
AIS_PERF_IDENTITY = MappingProxyType(
    {
        "backend": "vllm",
        "system": "h200_sxm",
        "backend_version": "current",
        "tp": 1,
        "model": "Qwen/Qwen3-0.6B",
        "worker_type": "aggregated",
    }
)
AIS_IDENTITY_ENV_FIELDS = (
    "backend",
    "system",
    "backend_version",
    "tp_size",
    "model_path",
    "moe_tp_size",
    "moe_ep_size",
    "attention_dp_size",
    "nextn",
    "nextn_accept_rates",
    "mtp_seed",
)
PREFILL_LOAD_MODEL_NONE_WIRE = "none"
PREFILL_LOAD_MODEL_AIS_WIRE = "aic"
DECODE_LOAD_EVENT = "processed active load update"
DECODE_LOAD_TARGET = "dynamo_llm::discovery::worker_monitor"
DECODE_CONSUMER_LOG = "info,dynamo_llm::discovery::worker_monitor=debug"
_DECODE_WORKER_ID_RE = re.compile(r"(?:^|\s)worker_id=(-?\d+)(?:\s|$)")
_DECODE_DP_RANK_RE = re.compile(r"(?:^|\s)dp_rank=(\d+)(?:\s|$)")
_DECODE_OBSERVATION_RE = re.compile(r"(?:^|\s)observation=(Remote|Scheduler)\b")
_DECODE_THRESHOLD_RE = re.compile(
    r"(?:^|\s)active_decode_blocks_threshold=(None|Some\(([^)]+)\))(?:\s|$)"
)
_DECODE_SKIP = object()
SHARED_CACHE_TYPE_ROW = "shared_cache_type"
SHARED_CACHE_TYPE_OBSERVER = (
    "fe_native_to_json+epp_init_event+owned_hicache_constructor+"
    "plain_chat_token_parity+fe_generation+epp_route+default_none_control"
)
SHARED_CACHE_TYPE_TEST = "test_shared_cache_type_consumer"
SHARED_CACHE_TYPE_ITEM = (
    f"tests/router/test_epp_frontend_config_parity.py::{SHARED_CACHE_TYPE_TEST}"
)
SHARED_CACHE_TYPE_ENV = "DYN_SHARED_CACHE_TYPE"
SHARED_CACHE_TYPE_PROFILES = ("none", "hicache")
SHARED_CACHE_TYPE_NONE = "none"
SHARED_CACHE_TYPE_HICACHE = "hicache"
SHARED_CACHE_TYPE_FIELDS = (SHARED_CACHE_TYPE_ROW,)
HICACHE_CONSTRUCTOR_EVENT = "Using HiCache shared KV cache"
HICACHE_CONSTRUCTOR_TARGET = "dynamo_llm::discovery::model_manager"
_SHARED_CACHE_TYPE_EVENT_VALUES = MappingProxyType(
    {
        SHARED_CACHE_TYPE_NONE: SHARED_CACHE_TYPE_NONE,
        SHARED_CACHE_TYPE_HICACHE: SHARED_CACHE_TYPE_HICACHE,
    }
)
_HICACHE_TARGET_RE = re.compile(
    r"(?:^|\s)" + re.escape(HICACHE_CONSTRUCTOR_TARGET) + r"(?::|\s|$)"
)
_HICACHE_COMPONENT_RE = re.compile(
    r'(?:^|\s|,)worker_component=("(?:\\.|[^"\\])*")(?:\s|,|$)'
)
_SHARED_CACHE_TYPE_BINDING = MappingProxyType(
    {
        "observer": SHARED_CACHE_TYPE_OBSERVER,
        "flags": ("--shared-cache-type",),
        "env": (SHARED_CACHE_TYPE_ENV,),
        "schema_fields": (SHARED_CACHE_TYPE_ROW,),
        "affected_fields": (SHARED_CACHE_TYPE_ROW,),
        "items": (SHARED_CACHE_TYPE_ITEM,),
    }
)
KV_CACHE_BLOCK_SIZE_ROW = "kv_cache_block_size"
KV_CACHE_BLOCK_SIZE_OBSERVER = (
    "dynamic_worker_card_cross_input+model_metric+provider_block_size"
)
KV_CACHE_BLOCK_SIZE_TEST = "test_dynamic_block_size_consumer"
KV_CACHE_BLOCK_SIZE_PROFILES = (
    "worker_card32_frontend_cli8",
    "worker_card32_frontend_cli64",
    "worker_card16_frontend_cli64",
)
KV_CACHE_BLOCK_SIZE_ITEMS = (
    "tests/router/test_epp_frontend_config_parity.py::"
    "test_dynamic_block_size_consumer[worker_card32_frontend_cli8]",
    "tests/router/test_epp_frontend_config_parity.py::"
    "test_dynamic_block_size_consumer[worker_card32_frontend_cli64]",
    "tests/router/test_epp_frontend_config_parity.py::"
    "test_dynamic_block_size_consumer[worker_card16_frontend_cli64]",
)
QUEUE_RS = REPO_ROOT / "lib/kv-router/src/scheduling/queue.rs"
INDEXER_BACKEND_RS = REPO_ROOT / "lib/kv-router/src/services/indexer/backend.rs"
WORKERS_RS = REPO_ROOT / "lib/kv-router/src/services/selection/core/workers.rs"
_DEFAULT_OVERLAP_REFRESH_RE = re.compile(
    r"pub const DEFAULT_OVERLAP_REFRESH_AFTER_SECS:\s*u64\s*=\s*(\d+)\s*;"
)
_READ_OVERLAP_REFRESH_RE = re.compile(
    r"pub fn read_overlap_refresh_after\s*\(\s*\)\s*->\s*Option<Duration>\s*\{"
)
_QUEUE_REFRESH_IF_RE = re.compile(
    r"let overlap_refresh_after = if overlap_scores_refresh\.is_some\(\)\s*\{"
)
_SUPPORTS_OVERLAP_REFRESH_RE = re.compile(
    r"pub fn supports_overlap_refresh\s*\(\s*&self\s*\)\s*->\s*bool\s*\{"
)
NOT_FE_HONORED_ROWS = frozenset(
    {NOT_FE_HONORED_ROW, POSITIONAL_ROW, KV_CACHE_BLOCK_SIZE_ROW}
)
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
    "public-xxh3-v1": "public-xxh3-v1",
    "keyed-xxh3-v1": "keyed-xxh3-v1",
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
        "router_tracking_hash",
        "router_tracking_key_id",
    }
)
OPTION_EVENT_FIELDS = frozenset(
    {
        "router_queue_threshold",
        "router_predicted_ttl_secs",
        "conditional_disagg_prefill_busy_threshold",
        "conditional_disagg_decode_busy_threshold",
        "shared_cache_multiplier",
        "router_tracking_key_id",
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
STATUS_UNSUPPORTED_INPUT = "unsupported_input"
STATUS_UNSUPPORTED_CAPABILITY = "unsupported_capability"
KNOWN_STATUSES = frozenset(
    {
        STATUS_PENDING,
        STATUS_SUPPORTED,
        STATUS_ALIAS,
        STATUS_KNOWN_GAP,
        STATUS_IGNORED_COMPAT,
        STATUS_NOT_FE_HONORED,
        STATUS_UNSUPPORTED_INPUT,
        STATUS_UNSUPPORTED_CAPABILITY,
    }
)
LOGGER = logging.getLogger(__name__)


class FrontendEppConfigMismatch(AssertionError):
    """FE and EPP both produced a value, and those values differ."""


class LoadAwarePresetConfigMismatch(FrontendEppConfigMismatch):
    """Measured FE load-aware preset vs EPP ignoring DYN_ROUTER_LOAD_AWARE."""


class ConditionalJsonConfigMismatch(FrontendEppConfigMismatch):
    """Measured FE conditional JSON vs EPP default scalar config."""


class ActiveDecodeThresholdConfigMismatch(FrontendEppConfigMismatch):
    """Measured FE configured 0.25 decode threshold vs EPP monitor None."""


class SessionPrefixIndexConfigMismatch(FrontendEppConfigMismatch):
    """FE rejected session-prefix validation while EPP accepted and routed.

    Scope is owned input/config validation only: env/YAML parse, the exact
    ``enable_session_prefix_index requires use_kv_events=true`` diagnostic,
    and plain-chat token/route acceptance. Not lineage, scoring, or affinity.
    """


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
        SERVE_INDEXER_ROW,
        ("--serve-indexer", "--no-serve-indexer"),
        ("DYN_SERVE_INDEXER",),
        ("serve_indexer",),
        STATUS_UNSUPPORTED_INPUT,
        INPUT_STARTUP_SOURCE_OBSERVER,
        (),
        (SERVE_INDEXER_ROW,),
        "Stock Runtime EPP has no FE CLI/env serve_indexer surface. Shared "
        "YAML router.serve_indexer and the native ensure_served_indexer_service "
        "factory exist at source; this row does not claim exercised YAML serve.",
    ),
    SettingDecision(
        KV_CACHE_BLOCK_SIZE_ROW,
        ("--kv-cache-block-size",),
        ("DYN_KV_CACHE_BLOCK_SIZE",),
        ("kv_cache_block_size",),
        STATUS_NOT_FE_HONORED,
        KV_CACHE_BLOCK_SIZE_OBSERVER,
        KV_CACHE_BLOCK_SIZE_PROFILES,
        (KV_CACHE_BLOCK_SIZE_ROW,),
        "The tested Dynamic engine consumes worker-card geometry; independent "
        "CLI 8/64 did not override card 32, and the card 16 control yielded 16. "
        "This is not CLI support or an all-engine statement. The live inputs "
        "were CLI cross-values; CLI/env identifier ownership shares the "
        "Frontend setting, and no independent env run is being claimed.",
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
        DECODE_THRESHOLD_ROW,
        ("--active-decode-blocks-threshold",),
        (DECODE_THRESHOLD_ENV,),
        ("active_decode_blocks_threshold",),
        STATUS_KNOWN_GAP,
        DECODE_THRESHOLD_OBSERVER,
        (),
        (DECODE_THRESHOLD_ROW,),
        "Measured consumer gap: FE configured 0.25 vs EPP monitor None with "
        "default None control. Effective remote-load consumer scope, not a "
        "proven admission-pressure boundary.",
    ),
    SettingDecision(
        PREFILL_ABS_THRESHOLD_ROW,
        ("--active-prefill-tokens-threshold",),
        (PREFILL_ABS_ENV,),
        ("active_prefill_tokens_threshold",),
        STATUS_UNSUPPORTED_INPUT,
        PREFILL_ADMISSION_OBSERVER,
        (),
        (PREFILL_ABS_THRESHOLD_ROW,),
        "FE attaches original LoadThresholdHandle from RouterConfig; stock "
        "EPP managed_kv_router builds Default empty handles. Absolute FE "
        "admin GET attachment is measured; saturation/rejection is not.",
    ),
    SettingDecision(
        PREFILL_FRAC_THRESHOLD_ROW,
        ("--active-prefill-tokens-threshold-frac",),
        (PREFILL_FRAC_ENV,),
        ("active_prefill_tokens_threshold_frac",),
        STATUS_UNSUPPORTED_INPUT,
        PREFILL_ADMISSION_OBSERVER,
        (),
        (PREFILL_FRAC_THRESHOLD_ROW,),
        "FE attaches original LoadThresholdHandle from RouterConfig; stock "
        "EPP managed_kv_router builds Default empty handles. Fractional FE "
        "admin GET attachment is measured; saturation/rejection is not.",
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
        SESSION_PREFIX_INDEX_ROW,
        ("--enable-session-prefix-index", "--no-enable-session-prefix-index"),
        (SESSION_PREFIX_INDEX_ENV,),
        (SESSION_PREFIX_INDEX_ROW,),
        STATUS_KNOWN_GAP,
        SESSION_PREFIX_INDEX_OBSERVER,
        (),
        (SESSION_PREFIX_INDEX_ROW,),
        "Measured env-true/kv-false gap: FE rejects "
        "enable_session_prefix_index requires use_kv_events=true while EPP "
        "accepts plain-chat tokens/routes. Policy YAML is consumed; the same "
        "invalid YAML rejects on both arms. Input/config validation only, not "
        "CLI matrix or lineage/scoring/affinity.",
    ),
    SettingDecision(
        MIN_INITIAL_WORKERS_ROW,
        ("--router-min-initial-workers",),
        ("DYN_ROUTER_MIN_INITIAL_WORKERS",),
        ("min_initial_workers",),
        STATUS_UNSUPPORTED_CAPABILITY,
        INPUT_STARTUP_SOURCE_OBSERVER,
        (),
        (MIN_INITIAL_WORKERS_ROW,),
        "Shared env parses non-negative usize (valid input), but stock Runtime "
        "EPP forces skip_initial_worker_wait and bypasses the N-worker gate "
        "while still waiting for nonempty runtime bootstrap. Not a "
        "KvRouterConfig field.",
    ),
    SettingDecision(
        ROUTER_MODE_ROW,
        ("--router-mode",),
        ("DYN_ROUTER_MODE",),
        ("router_mode",),
        STATUS_UNSUPPORTED_INPUT,
        INPUT_STARTUP_SOURCE_OBSERVER,
        (),
        (ROUTER_MODE_ROW,),
        "FE non-KV mode selector is unsupported on the stock Runtime EPP KV "
        "profile. Not a statement about standalone routers outside this "
        "profile. Not a KvRouterConfig to_json/init-event field.",
    ),
    SettingDecision(
        "router_approximate_cache_policy",
        ("--router-approximate-cache-policy",),
        ("DYN_ROUTER_APPROXIMATE_CACHE_POLICY",),
        ("router_approximate_cache_policy",),
        STATUS_SUPPORTED,
        APPROXIMATE_POLICY_OBSERVER,
        (),
        ("router_approximate_cache_policy",),
        "Unverified candidate until fixed-SHA Linux execution. Process-local; "
        "omitted from native to_json. Owned FE and EPP core gauges observe "
        "nondefault LRU and the omitted-env default TTL control.",
    ),
    SettingDecision(
        ROUTER_PREFILL_POLICY_ROW,
        ("--router-prefill-policy",),
        ("DYN_ROUTER_PREFILL_POLICY",),
        ("router_prefill_policy",),
        STATUS_UNSUPPORTED_CAPABILITY,
        NAMED_POLICY_CATALOG_OBSERVER,
        (),
        (ROUTER_PREFILL_POLICY_ROW,),
        "Stock Runtime EPP supports default selection; configured named "
        "instance selection is parsed but its provider is unavailable in "
        "this catalog. FE global named-default factory/selection is "
        "measured; prefill role resolution is source-backed and not "
        "independently disagg-executed.",
    ),
    SettingDecision(
        ROUTER_DECODE_POLICY_ROW,
        ("--router-decode-policy",),
        ("DYN_ROUTER_DECODE_POLICY",),
        ("router_decode_policy",),
        STATUS_UNSUPPORTED_CAPABILITY,
        NAMED_POLICY_CATALOG_OBSERVER,
        (),
        (ROUTER_DECODE_POLICY_ROW,),
        "Stock Runtime EPP supports default selection; configured named "
        "instance selection is parsed but its provider is unavailable in "
        "this catalog. FE global named-default factory/selection is "
        "measured; decode role resolution is source-backed and not "
        "independently disagg-executed.",
    ),
    SettingDecision(
        PREFILL_LOAD_MODEL_ROW,
        ("--router-prefill-load-model",),
        ("DYN_ROUTER_PREFILL_LOAD_MODEL",),
        ("router_prefill_load_model",),
        STATUS_UNSUPPORTED_CAPABILITY,
        INPUT_STARTUP_SOURCE_OBSERVER,
        (),
        (PREFILL_LOAD_MODEL_ROW,),
        "Enum/env/YAML parse and validation/recheck effects exist, but stock "
        "Runtime EPP decode/prefill factories pass estimator None and do not "
        "install a Ready AIS estimator. Not whole-enum ignored. FE AIS "
        "runtime is not claimed. Native wire alias aic is legitimate.",
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
        "300-second expiration sweep. Same-feature Hard/TTL300 evidence for "
        "session_affinity_mode; Soft remains unmeasured.",
    ),
    SettingDecision(
        SESSION_AFFINITY_MODE_ROW,
        ("--router-session-affinity-mode",),
        ("DYN_ROUTER_SESSION_AFFINITY_MODE",),
        ("session_affinity_mode",),
        STATUS_UNSUPPORTED_CAPABILITY,
        INPUT_STARTUP_SOURCE_OBSERVER,
        (),
        (SESSION_AFFINITY_MODE_ROW,),
        "Unsupported stock EPP session-affinity subconfiguration: no session "
        "context on decode/prefill selection and fixed prefill TTL None/Hard. "
        "Hard+TTL300 strict gap already binds SESSION_TTL_ITEM; Soft is not "
        "measured.",
    ),
    SettingDecision(
        "router_tracking_hash",
        ("--router-tracking-hash",),
        ("DYN_ROUTER_TRACKING_HASH",),
        ("router_tracking_hash",),
        STATUS_SUPPORTED,
        TRACKING_RUNTIME_OBSERVER,
        (),
        ("router_tracking_hash",),
        "Unverified candidate until fixed-SHA Linux execution. Keyed public "
        "test profile. FE native to_json and the owned EPP init event must "
        "expose keyed-xxh3-v1 after real model construction, query token "
        "parity, one FE generation, and one real EPP ext_proc route.",
    ),
    SettingDecision(
        "router_tracking_key_file",
        ("--router-tracking-key-file",),
        ("DYN_ROUTER_TRACKING_KEY_FILE",),
        ("router_tracking_key_file",),
        STATUS_SUPPORTED,
        TRACKING_KEY_FILE_OBSERVER,
        (),
        ("router_tracking_key_file",),
        "Unverified candidate until fixed-SHA Linux execution. Valid 32-byte "
        "public file is consumed during router construction. Independent "
        "frontend and epp 31-byte arms must emit the owned size diagnostic. "
        "The EPP init event does not expose the path.",
    ),
    SettingDecision(
        "router_tracking_key_id",
        ("--router-tracking-key-id",),
        ("DYN_ROUTER_TRACKING_KEY_ID",),
        ("router_tracking_key_id",),
        STATUS_SUPPORTED,
        TRACKING_RUNTIME_OBSERVER,
        (),
        ("router_tracking_key_id",),
        "Unverified candidate until fixed-SHA Linux execution. Keyed public "
        "test profile. FE native to_json and the owned EPP init event must "
        "expose parity-public-test-key-v1 after real model construction, "
        "query token parity, one FE generation, and one real EPP ext_proc "
        "route.",
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
        SHARED_CACHE_TYPE_ROW,
        ("--shared-cache-type",),
        (SHARED_CACHE_TYPE_ENV,),
        (SHARED_CACHE_TYPE_ROW,),
        STATUS_SUPPORTED,
        SHARED_CACHE_TYPE_OBSERVER,
        (),
        (SHARED_CACHE_TYPE_ROW,),
        "Omitted-env none and explicit hicache are observed on FE native JSON, "
        "the owned EPP init event, and HiCache client construction. This is "
        "not a remote cache-hit claim. Existing aliases stay owned, not "
        "independently executed.",
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
        ACTIVE_REQUEST_EXPIRY_ROW,
        (),
        ("DYN_ROUTER_ACTIVE_REQUEST_EXPIRY_SECS",),
        (),
        STATUS_SUPPORTED,
        NATIVE_CONSUMER_OBSERVER,
        _NATIVE_CONSUMER_ROW_PROFILES[ACTIVE_REQUEST_EXPIRY_ROW],
        (ACTIVE_REQUEST_EXPIRY_ROW,),
        "Unverified candidate until fixed-SHA Linux execution. Live FE/EPP "
        "native consumer profiles observe peer booking CLOCK expiry versus "
        "default hold, including the configured expiry=3 recover arm.",
    ),
    SettingDecision(
        ACTIVE_SEQUENCE_DIRECT_ZMQ_ROW,
        (),
        ("DYN_ROUTER_ACTIVE_SEQUENCE_DIRECT_ZMQ",),
        (),
        STATUS_SUPPORTED,
        NATIVE_CONSUMER_OBSERVER,
        _NATIVE_CONSUMER_ROW_PROFILES[ACTIVE_SEQUENCE_DIRECT_ZMQ_ROW],
        (ACTIVE_SEQUENCE_DIRECT_ZMQ_ROW,),
        "Unverified candidate until fixed-SHA Linux execution. Live FE/EPP "
        "native consumer profiles observe specialized direct ingress versus "
        "the legacy runtime subscriber path.",
    ),
    SettingDecision(
        ACTIVE_SEQUENCE_ZMQ_RCVHWM_ROW,
        (),
        ("DYN_ROUTER_ACTIVE_SEQUENCE_ZMQ_RCVHWM",),
        (),
        STATUS_SUPPORTED,
        NATIVE_CONSUMER_OBSERVER,
        _NATIVE_CONSUMER_ROW_PROFILES[ACTIVE_SEQUENCE_ZMQ_RCVHWM_ROW],
        (ACTIVE_SEQUENCE_ZMQ_RCVHWM_ROW,),
        "Unverified candidate until fixed-SHA Linux execution. Live FE/EPP "
        "native consumer profiles observe direct-path rcvhwm construction; "
        "legacy_runtime does not claim HWM on runtime subscriber sockets.",
    ),
    SettingDecision(
        OVERLAP_REFRESH_ROW,
        (),
        (OVERLAP_REFRESH_ENV,),
        (),
        STATUS_SUPPORTED,
        OVERLAP_REFRESH_OBSERVER,
        (),
        (OVERLAP_REFRESH_ROW,),
        "Rust env consumed at scheduler queue construction when the indexer "
        "supports overlap refresh. Owned FE and EPP queue INFO plus plain-chat "
        "token parity observe the shared nondefault disable sentinel.",
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
        WORKER_SELECTION_POLICY_ROW,
        (),
        ("DYN_ROUTER_WORKER_SELECTION_POLICY",),
        (),
        STATUS_UNSUPPORTED_CAPABILITY,
        NAMED_POLICY_CATALOG_OBSERVER,
        (),
        (WORKER_SELECTION_POLICY_ROW,),
        "Stock Runtime EPP supports default selection; configured named "
        "instance selection is parsed but its provider is unavailable in "
        "this catalog. FE global named-default factory/selection and exact "
        "UnknownType rejection are measured on the ordinary catalog Item.",
    ),
    SettingDecision(
        ZMQ_ENDPOINTS_PER_SUB_ROW,
        (),
        ("DYN_ROUTER_ZMQ_ENDPOINTS_PER_SUB",),
        (),
        STATUS_SUPPORTED,
        NATIVE_CONSUMER_OBSERVER,
        _NATIVE_CONSUMER_ROW_PROFILES[ZMQ_ENDPOINTS_PER_SUB_ROW],
        (ZMQ_ENDPOINTS_PER_SUB_ROW,),
        "Unverified candidate until fixed-SHA Linux execution. Live FE/EPP "
        "native consumer profiles observe Dedicated (omitted) and Grouped "
        "pool construction with dual-publisher payload consumption; "
        "legacy_runtime does not claim pool sizing.",
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
        RUST_EPP,
        RUST_WATCHER,
        RUST_MODEL_MANAGER,
        RUST_PREFILL_ACTIVATION,
        RUST_KV_ROUTER,
        RUST_PREFILL_QUERY,
        RUST_EMBEDDED,
        RUST_INDEXER_INGRESS,
        RUST_INDEXER_REMOTE,
        RUST_AIS_CALLBACK,
        RUST_POLICY_CONFIG,
        RUST_QUEUE,
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
    """Reject unknown statuses, incomplete pending rows, and duplicate owners."""
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
        if row.status == STATUS_PENDING:
            if not isinstance(row.reason, str) or not row.reason.strip():
                raise AssertionError(f"{row.name} pending reason is empty")
            if row.observer is not None:
                raise AssertionError(
                    f"{row.name} pending observer={row.observer!r} is not None"
                )
            if row.profiles != ():
                raise AssertionError(
                    f"{row.name} pending profiles {row.profiles} are not empty"
                )
            if row.affected_fields != ():
                raise AssertionError(
                    f"{row.name} pending affected_fields "
                    f"{row.affected_fields} are not empty"
                )
        if row.status in {STATUS_UNSUPPORTED_INPUT, STATUS_UNSUPPORTED_CAPABILITY}:
            if not isinstance(row.reason, str) or not row.reason.strip():
                raise AssertionError(f"{row.name} unsupported reason is empty")
            if row.profiles != ():
                raise AssertionError(
                    f"{row.name} unsupported profiles {row.profiles} are not empty"
                )
            if row.affected_fields != (row.name,):
                raise AssertionError(
                    f"{row.name} unsupported affected_fields "
                    f"{row.affected_fields} != {(row.name,)!r}"
                )
            if row.observer is None:
                raise AssertionError(f"{row.name} unsupported observer is None")
        if (
            row.status == STATUS_UNSUPPORTED_INPUT
            and row.name not in UNSUPPORTED_INPUT_ROWS
        ):
            raise AssertionError(
                f"{row.name} may not use {STATUS_UNSUPPORTED_INPUT}; only "
                f"{sorted(UNSUPPORTED_INPUT_ROWS)} may"
            )
        if (
            row.status == STATUS_UNSUPPORTED_CAPABILITY
            and row.name not in UNSUPPORTED_CAPABILITY_ROWS
        ):
            raise AssertionError(
                f"{row.name} may not use {STATUS_UNSUPPORTED_CAPABILITY}; only "
                f"{sorted(UNSUPPORTED_CAPABILITY_ROWS)} may"
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
        if row.name in _SEVEN_ROW_BINDINGS:
            wanted_observer = _SEVEN_ROW_BINDINGS[row.name]["observer"]
            if row.observer != wanted_observer:
                raise AssertionError(
                    f"{row.name} observer={row.observer!r} != {wanted_observer!r}"
                )
        elif row.observer in {
            INPUT_STARTUP_SOURCE_OBSERVER,
            PREFILL_ADMISSION_OBSERVER,
        }:
            raise AssertionError(
                f"{row.name} may not use {row.observer!r}; only closed seven-row "
                f"names {sorted(_SEVEN_ROW_BINDINGS)} may"
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
        if row.observer == OVERLAP_REFRESH_OBSERVER and row.name != OVERLAP_REFRESH_ROW:
            raise AssertionError(
                f"{row.name} may not use {OVERLAP_REFRESH_OBSERVER}; only "
                f"{OVERLAP_REFRESH_ROW} may"
            )
        if row.observer == TRACKING_RUNTIME_OBSERVER and row.name not in {
            TRACKING_HASH_ROW,
            TRACKING_KEY_ID_ROW,
        }:
            raise AssertionError(
                f"{row.name} may not use {TRACKING_RUNTIME_OBSERVER}; only "
                f"{TRACKING_HASH_ROW} and {TRACKING_KEY_ID_ROW} may"
            )
        if (
            row.observer == TRACKING_KEY_FILE_OBSERVER
            and row.name != TRACKING_KEY_FILE_ROW
        ):
            raise AssertionError(
                f"{row.name} may not use {TRACKING_KEY_FILE_OBSERVER}; only "
                f"{TRACKING_KEY_FILE_ROW} may"
            )
        if (
            row.observer == APPROXIMATE_POLICY_OBSERVER
            and row.name != APPROXIMATE_POLICY_ROW
        ):
            raise AssertionError(
                f"{row.name} may not use {APPROXIMATE_POLICY_OBSERVER}; only "
                f"{APPROXIMATE_POLICY_ROW} may"
            )
        if (
            row.observer == DECODE_THRESHOLD_OBSERVER
            and row.name != DECODE_THRESHOLD_ROW
        ):
            raise AssertionError(
                f"{row.name} may not use {DECODE_THRESHOLD_OBSERVER}; only "
                f"{DECODE_THRESHOLD_ROW} may"
            )
        if (
            row.observer == SESSION_PREFIX_INDEX_OBSERVER
            and row.name != SESSION_PREFIX_INDEX_ROW
        ):
            raise AssertionError(
                f"{row.name} may not use {SESSION_PREFIX_INDEX_OBSERVER}; only "
                f"{SESSION_PREFIX_INDEX_ROW} may"
            )
        if (
            row.observer == SHARED_CACHE_TYPE_OBSERVER
            and row.name != SHARED_CACHE_TYPE_ROW
        ):
            raise AssertionError(
                f"{row.name} may not use {SHARED_CACHE_TYPE_OBSERVER}; only "
                f"{SHARED_CACHE_TYPE_ROW} may"
            )
        if (
            row.observer == KV_CACHE_BLOCK_SIZE_OBSERVER
            and row.name != KV_CACHE_BLOCK_SIZE_ROW
        ):
            raise AssertionError(
                f"{row.name} may not use {KV_CACHE_BLOCK_SIZE_OBSERVER}; only "
                f"{KV_CACHE_BLOCK_SIZE_ROW} may"
            )
        if row.observer == NATIVE_CONSUMER_OBSERVER and row.name not in set(
            NATIVE_CONSUMER_ROWS
        ):
            raise AssertionError(
                f"{row.name} may not use {NATIVE_CONSUMER_OBSERVER}; only "
                f"{list(NATIVE_CONSUMER_ROWS)} may"
            )
        if row.observer == NAMED_POLICY_CATALOG_OBSERVER and row.name not in set(
            NAMED_POLICY_CATALOG_ROWS
        ):
            raise AssertionError(
                f"{row.name} may not use {NAMED_POLICY_CATALOG_OBSERVER}; only "
                f"{list(NAMED_POLICY_CATALOG_ROWS)} may"
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


def _require_collected_item(item: Any) -> Any:
    if isinstance(item, (str, bytes)) or not hasattr(item, "nodeid"):
        raise AssertionError(
            "manifest binding requires collected pytest Items, not "
            f"{type(item).__name__}"
        )
    if not callable(getattr(item, "iter_markers", None)):
        raise AssertionError(f"collected item {item.nodeid!r} has no iter_markers")
    return item


def _assert_strict_xfail_marker(label: str, mark: Any, exception: type) -> None:
    if mark.args:
        raise AssertionError(f"{label} xfail has positional condition {mark.args!r}")
    if "condition" in mark.kwargs:
        raise AssertionError(
            f"{label} xfail has condition={mark.kwargs['condition']!r}"
        )
    if mark.kwargs.get("strict") is not True:
        raise AssertionError(
            f"{label} xfail strict={mark.kwargs.get('strict')!r} is not True"
        )
    if mark.kwargs.get("raises") is not exception:
        raise AssertionError(
            f"{label} xfail raises={mark.kwargs.get('raises')!r} != {exception!r}"
        )
    if mark.kwargs.get("run", True) is not True:
        raise AssertionError(
            f"{label} xfail run={mark.kwargs.get('run')!r} is disabled"
        )


def _assert_measured_xfail_marker(name: str, mark: Any) -> None:
    spec = MEASURED_GAP_SIGNATURES[name]
    _assert_strict_xfail_marker(name, mark, spec.exception)


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


def _assert_kv_cache_block_size_not_fe_binding(
    row: SettingDecision, items: Iterable[Any]
) -> None:
    if row.name != KV_CACHE_BLOCK_SIZE_ROW:
        raise AssertionError(
            f"{row.name} is not the fixed {KV_CACHE_BLOCK_SIZE_ROW} contract"
        )
    if row.status != STATUS_NOT_FE_HONORED:
        raise AssertionError(
            f"{row.name} status {row.status!r} != {STATUS_NOT_FE_HONORED!r}"
        )
    if row.observer != KV_CACHE_BLOCK_SIZE_OBSERVER:
        raise AssertionError(
            f"{row.name} observer={row.observer!r} != "
            f"{KV_CACHE_BLOCK_SIZE_OBSERVER!r}"
        )
    if row.flags != ("--kv-cache-block-size",):
        raise AssertionError(
            f"{row.name} flags {row.flags} != ('--kv-cache-block-size',)"
        )
    if row.env != ("DYN_KV_CACHE_BLOCK_SIZE",):
        raise AssertionError(
            f"{row.name} env {row.env} != ('DYN_KV_CACHE_BLOCK_SIZE',)"
        )
    if row.schema_fields != (KV_CACHE_BLOCK_SIZE_ROW,):
        raise AssertionError(
            f"{row.name} schema_fields {row.schema_fields} != "
            f"{(KV_CACHE_BLOCK_SIZE_ROW,)!r}"
        )
    if row.affected_fields != (KV_CACHE_BLOCK_SIZE_ROW,):
        raise AssertionError(
            f"{row.name} affected_fields {row.affected_fields} != "
            f"{(KV_CACHE_BLOCK_SIZE_ROW,)!r}"
        )
    if row.profiles != KV_CACHE_BLOCK_SIZE_PROFILES:
        raise AssertionError(
            f"{row.name} profiles {row.profiles} != {KV_CACHE_BLOCK_SIZE_PROFILES}"
        )
    for item_id in KV_CACHE_BLOCK_SIZE_ITEMS:
        item = _require_unique_nodeid_item(items, item_id)
        _assert_ordinary_collected_item(item_id, item)


def _assert_not_fe_honored_binding(row: SettingDecision, items: Iterable[Any]) -> None:
    if row.name == NOT_FE_HONORED_ROW:
        _assert_conditional_scalar_not_fe_binding(row, items)
        return
    if row.name == POSITIONAL_ROW:
        _assert_positional_not_fe_binding(row, items)
        return
    if row.name == KV_CACHE_BLOCK_SIZE_ROW:
        _assert_kv_cache_block_size_not_fe_binding(row, items)
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
        raise AssertionError(
            f"{row.name} status {row.status!r} != {STATUS_SUPPORTED!r}"
        )
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

    _assert_strict_xfail_marker(SESSION_TTL_ROW, mark, SessionAffinityParityMismatch)


def _assert_session_ttl_binding(row: SettingDecision, items: Iterable[Any]) -> None:
    if row.name != SESSION_TTL_ROW:
        raise AssertionError(f"{row.name} is not the fixed {SESSION_TTL_ROW} contract")
    if row.status != STATUS_KNOWN_GAP:
        raise AssertionError(
            f"{row.name} status {row.status!r} != {STATUS_KNOWN_GAP!r}"
        )
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


def _assert_overlap_refresh_binding(row: SettingDecision, items: Iterable[Any]) -> None:
    if row.name != OVERLAP_REFRESH_ROW:
        raise AssertionError(
            f"{row.name} is not the fixed {OVERLAP_REFRESH_ROW} contract"
        )
    if row.status != STATUS_SUPPORTED:
        raise AssertionError(
            f"{row.name} status {row.status!r} != {STATUS_SUPPORTED!r}"
        )
    if row.observer != OVERLAP_REFRESH_OBSERVER:
        raise AssertionError(
            f"{row.name} observer={row.observer!r} != " f"{OVERLAP_REFRESH_OBSERVER!r}"
        )
    if row.flags != ():
        raise AssertionError(f"{row.name} flags {row.flags} != ()")
    if row.env != (OVERLAP_REFRESH_ENV,):
        raise AssertionError(f"{row.name} env {row.env} != {(OVERLAP_REFRESH_ENV,)!r}")
    if row.schema_fields != ():
        raise AssertionError(f"{row.name} schema_fields {row.schema_fields} != ()")
    if row.affected_fields != (OVERLAP_REFRESH_ROW,):
        raise AssertionError(
            f"{row.name} affected_fields {row.affected_fields} != "
            f"{(OVERLAP_REFRESH_ROW,)!r}"
        )
    if row.profiles != ():
        raise AssertionError(f"{row.name} profiles {row.profiles} != ()")
    item = _require_unique_nodeid_item(items, OVERLAP_REFRESH_ITEM)
    _assert_ordinary_collected_item(OVERLAP_REFRESH_ITEM, item)


def _assert_tracking_approximate_binding(
    row: SettingDecision, items: Iterable[Any]
) -> None:
    spec = _TRACKING_APPROXIMATE_BINDINGS.get(row.name)
    if spec is None:
        raise AssertionError(
            f"{row.name} is not a closed tracking/approximate contract"
        )
    if row.status != STATUS_SUPPORTED:
        raise AssertionError(
            f"{row.name} status {row.status!r} != {STATUS_SUPPORTED!r}"
        )
    for field in ("observer", "flags", "env", "schema_fields", "affected_fields"):
        got = getattr(row, field)
        wanted = spec[field]
        if got != wanted:
            raise AssertionError(f"{row.name} {field} {got!r} != {wanted!r}")
    if row.profiles != ():
        raise AssertionError(f"{row.name} profiles {row.profiles} != ()")
    for item_id in spec["items"]:
        item = _require_unique_nodeid_item(items, item_id)
        _assert_ordinary_collected_item(item_id, item)


def _assert_decode_threshold_xfail_marker(mark: Any) -> None:
    _assert_strict_xfail_marker(
        DECODE_THRESHOLD_ROW, mark, ActiveDecodeThresholdConfigMismatch
    )


def _assert_decode_threshold_binding(
    row: SettingDecision, items: Iterable[Any]
) -> None:
    if row.name != DECODE_THRESHOLD_ROW:
        raise AssertionError(
            f"{row.name} is not the fixed {DECODE_THRESHOLD_ROW} contract"
        )
    if row.status != STATUS_KNOWN_GAP:
        raise AssertionError(
            f"{row.name} status {row.status!r} != {STATUS_KNOWN_GAP!r}"
        )
    if row.observer != DECODE_THRESHOLD_OBSERVER:
        raise AssertionError(
            f"{row.name} observer={row.observer!r} != {DECODE_THRESHOLD_OBSERVER!r}"
        )
    if row.flags != ("--active-decode-blocks-threshold",):
        raise AssertionError(f"{row.name} flags {row.flags} are not the closed tuple")
    if row.env != (DECODE_THRESHOLD_ENV,):
        raise AssertionError(f"{row.name} env {row.env} != {(DECODE_THRESHOLD_ENV,)!r}")
    if row.schema_fields != (DECODE_THRESHOLD_ROW,):
        raise AssertionError(
            f"{row.name} schema_fields {row.schema_fields} != "
            f"{(DECODE_THRESHOLD_ROW,)!r}"
        )
    if row.affected_fields != (DECODE_THRESHOLD_ROW,):
        raise AssertionError(
            f"{row.name} affected_fields {row.affected_fields} != "
            f"{(DECODE_THRESHOLD_ROW,)!r}"
        )
    if row.profiles != ():
        raise AssertionError(f"{row.name} profiles {row.profiles} != ()")
    item = _require_unique_nodeid_item(items, DECODE_THRESHOLD_ITEM)
    skips = list(item.iter_markers(name="skip")) + list(
        item.iter_markers(name="skipif")
    )
    marks = list(item.iter_markers(name="xfail"))
    if skips or len(marks) != 1:
        raise AssertionError(
            f"{DECODE_THRESHOLD_ITEM} must collect exactly one xfail and no skip, "
            f"found xfail={len(marks)} skip={len(skips)}"
        )
    _assert_decode_threshold_xfail_marker(marks[0])


def _assert_session_prefix_index_binding(
    row: SettingDecision, items: Iterable[Any]
) -> None:
    if row.name != SESSION_PREFIX_INDEX_ROW:
        raise AssertionError(
            f"{row.name} is not the fixed {SESSION_PREFIX_INDEX_ROW} contract"
        )
    if row.status != STATUS_KNOWN_GAP:
        raise AssertionError(
            f"{row.name} status {row.status!r} != {STATUS_KNOWN_GAP!r}"
        )
    if row.observer != SESSION_PREFIX_INDEX_OBSERVER:
        raise AssertionError(
            f"{row.name} observer={row.observer!r} != "
            f"{SESSION_PREFIX_INDEX_OBSERVER!r}"
        )
    if row.flags != (
        "--enable-session-prefix-index",
        "--no-enable-session-prefix-index",
    ):
        raise AssertionError(f"{row.name} flags {row.flags} are not the closed tuple")
    if row.env != (SESSION_PREFIX_INDEX_ENV,):
        raise AssertionError(
            f"{row.name} env {row.env} != {(SESSION_PREFIX_INDEX_ENV,)!r}"
        )
    if row.schema_fields != (SESSION_PREFIX_INDEX_ROW,):
        raise AssertionError(
            f"{row.name} schema_fields {row.schema_fields} != "
            f"{(SESSION_PREFIX_INDEX_ROW,)!r}"
        )
    if row.affected_fields != (SESSION_PREFIX_INDEX_ROW,):
        raise AssertionError(
            f"{row.name} affected_fields {row.affected_fields} != "
            f"{(SESSION_PREFIX_INDEX_ROW,)!r}"
        )
    if row.profiles != ():
        raise AssertionError(f"{row.name} profiles {row.profiles} != ()")
    item = _require_unique_nodeid_item(items, SESSION_PREFIX_INDEX_ITEM)
    skips = list(item.iter_markers(name="skip")) + list(
        item.iter_markers(name="skipif")
    )
    marks = list(item.iter_markers(name="xfail"))
    if skips or len(marks) != 1:
        raise AssertionError(
            f"{SESSION_PREFIX_INDEX_ITEM} must collect exactly one xfail and "
            f"no skip, found xfail={len(marks)} skip={len(skips)}"
        )
    _assert_strict_xfail_marker(
        SESSION_PREFIX_INDEX_ROW, marks[0], SessionPrefixIndexConfigMismatch
    )


def _assert_shared_cache_type_binding(
    row: SettingDecision, items: Iterable[Any]
) -> None:
    if row.name != SHARED_CACHE_TYPE_ROW:
        raise AssertionError(
            f"{row.name} is not the fixed {SHARED_CACHE_TYPE_ROW} contract"
        )
    if row.status != STATUS_SUPPORTED:
        raise AssertionError(
            f"{row.name} status {row.status!r} != {STATUS_SUPPORTED!r}"
        )
    for field in ("observer", "flags", "env", "schema_fields", "affected_fields"):
        got = getattr(row, field)
        wanted = _SHARED_CACHE_TYPE_BINDING[field]
        if got != wanted:
            raise AssertionError(f"{row.name} {field} {got!r} != {wanted!r}")
    if row.profiles != ():
        raise AssertionError(f"{row.name} profiles {row.profiles} != ()")
    item = _require_unique_nodeid_item(items, SHARED_CACHE_TYPE_ITEM)
    _assert_ordinary_collected_item(SHARED_CACHE_TYPE_ITEM, item)


def _assert_native_consumer_surface_closed() -> None:
    if NATIVE_CONSUMER_ROWS != (
        ACTIVE_REQUEST_EXPIRY_ROW,
        ACTIVE_SEQUENCE_DIRECT_ZMQ_ROW,
        ACTIVE_SEQUENCE_ZMQ_RCVHWM_ROW,
        ZMQ_ENDPOINTS_PER_SUB_ROW,
    ):
        raise AssertionError(
            "NATIVE_CONSUMER_ROWS drifted from the four owned native setting ids"
        )
    if NATIVE_CONSUMER_PROFILE_NAMES != (
        "omitted",
        "legacy_runtime",
        "grouped",
        "expiry",
    ):
        raise AssertionError(
            "NATIVE_CONSUMER_PROFILE_NAMES drifted from the fixed four-profile "
            f"literal, got {NATIVE_CONSUMER_PROFILE_NAMES!r}"
        )
    helper_keys = set(_NATIVE_CONSUMER_PROFILES)
    expected_keys = set(NATIVE_CONSUMER_PROFILE_NAMES)
    if helper_keys != expected_keys:
        raise AssertionError(
            "helper NATIVE_CONSUMER_PROFILES keys "
            f"{sorted(helper_keys)} != fixed contract set {sorted(expected_keys)}"
        )
    if set(_NATIVE_CONSUMER_ROW_PROFILES) != set(NATIVE_CONSUMER_ROWS):
        raise AssertionError(
            "_NATIVE_CONSUMER_ROW_PROFILES keys must equal NATIVE_CONSUMER_ROWS"
        )
    expected_row_profiles = {
        ACTIVE_REQUEST_EXPIRY_ROW: (
            "omitted",
            "legacy_runtime",
            "grouped",
            "expiry",
        ),
        ACTIVE_SEQUENCE_DIRECT_ZMQ_ROW: (
            "omitted",
            "legacy_runtime",
            "grouped",
            "expiry",
        ),
        ACTIVE_SEQUENCE_ZMQ_RCVHWM_ROW: ("omitted", "grouped", "expiry"),
        ZMQ_ENDPOINTS_PER_SUB_ROW: ("omitted", "grouped", "expiry"),
    }
    for name, profiles in _NATIVE_CONSUMER_ROW_PROFILES.items():
        wanted = expected_row_profiles[name]
        if profiles != wanted:
            raise AssertionError(
                f"{name} row profile oracle {profiles!r} != fixed {wanted!r}"
            )
        unknown = [
            profile for profile in profiles if profile not in expected_keys
        ]
        if unknown:
            raise AssertionError(
                f"{name} binds unknown native profiles {unknown}; known "
                f"{list(NATIVE_CONSUMER_PROFILE_NAMES)}"
            )
        if name == ZMQ_ENDPOINTS_PER_SUB_ROW and "legacy_runtime" in profiles:
            raise AssertionError(
                "zmq_endpoints_per_sub must not claim pool sizing on legacy_runtime"
            )
        if name == ACTIVE_SEQUENCE_ZMQ_RCVHWM_ROW and "legacy_runtime" in profiles:
            raise AssertionError(
                "active_sequence_zmq_rcvhwm must not claim HWM on legacy_runtime"
            )


def _assert_native_consumer_binding(row: SettingDecision, items: Iterable[Any]) -> None:
    _assert_native_consumer_surface_closed()
    if row.name not in _NATIVE_CONSUMER_ROW_PROFILES:
        raise AssertionError(
            f"{row.name} is not a closed native-consumer contract; "
            f"expected one of {list(NATIVE_CONSUMER_ROWS)}"
        )
    if row.status != STATUS_SUPPORTED:
        raise AssertionError(
            f"{row.name} status {row.status!r} != {STATUS_SUPPORTED!r}"
        )
    if row.observer != NATIVE_CONSUMER_OBSERVER:
        raise AssertionError(
            f"{row.name} observer={row.observer!r} != {NATIVE_CONSUMER_OBSERVER!r}"
        )
    if row.flags != ():
        raise AssertionError(f"{row.name} flags {row.flags} != ()")
    wanted_env = {
        ACTIVE_REQUEST_EXPIRY_ROW: ("DYN_ROUTER_ACTIVE_REQUEST_EXPIRY_SECS",),
        ACTIVE_SEQUENCE_DIRECT_ZMQ_ROW: ("DYN_ROUTER_ACTIVE_SEQUENCE_DIRECT_ZMQ",),
        ACTIVE_SEQUENCE_ZMQ_RCVHWM_ROW: ("DYN_ROUTER_ACTIVE_SEQUENCE_ZMQ_RCVHWM",),
        ZMQ_ENDPOINTS_PER_SUB_ROW: ("DYN_ROUTER_ZMQ_ENDPOINTS_PER_SUB",),
    }[row.name]
    if row.env != wanted_env:
        raise AssertionError(f"{row.name} env {row.env} != {wanted_env!r}")
    if row.schema_fields != ():
        raise AssertionError(f"{row.name} schema_fields {row.schema_fields} != ()")
    if row.affected_fields != (row.name,):
        raise AssertionError(
            f"{row.name} affected_fields {row.affected_fields} != {(row.name,)!r}"
        )
    wanted_profiles = _NATIVE_CONSUMER_ROW_PROFILES[row.name]
    if row.profiles != wanted_profiles:
        raise AssertionError(
            f"{row.name} profiles {row.profiles} != {wanted_profiles!r}"
        )
    for profile in wanted_profiles:
        item_id = NATIVE_CONSUMER_PROFILE_ITEMS[profile]
        item = _require_unique_nodeid_item(items, item_id)
        _assert_ordinary_collected_item(item_id, item)


def _assert_named_policy_catalog_surface_closed() -> None:
    if NAMED_POLICY_CATALOG_ROWS != (
        ROUTER_PREFILL_POLICY_ROW,
        ROUTER_DECODE_POLICY_ROW,
        WORKER_SELECTION_POLICY_ROW,
    ):
        raise AssertionError(
            "NAMED_POLICY_CATALOG_ROWS drifted from the three owned selector ids"
        )


def _assert_named_policy_catalog_binding(
    row: SettingDecision, items: Iterable[Any]
) -> None:
    _assert_named_policy_catalog_surface_closed()
    if row.name not in set(NAMED_POLICY_CATALOG_ROWS):
        raise AssertionError(
            f"{row.name} is not a closed named-policy catalog contract; "
            f"expected one of {list(NAMED_POLICY_CATALOG_ROWS)}"
        )
    if row.status != STATUS_UNSUPPORTED_CAPABILITY:
        raise AssertionError(
            f"{row.name} status {row.status!r} != {STATUS_UNSUPPORTED_CAPABILITY!r}"
        )
    if row.observer != NAMED_POLICY_CATALOG_OBSERVER:
        raise AssertionError(
            f"{row.name} observer={row.observer!r} != "
            f"{NAMED_POLICY_CATALOG_OBSERVER!r}"
        )
    if row.profiles != ():
        raise AssertionError(f"{row.name} profiles {row.profiles} != ()")
    if row.affected_fields != (row.name,):
        raise AssertionError(
            f"{row.name} affected_fields {row.affected_fields} != {(row.name,)!r}"
        )
    wanted_flags = {
        ROUTER_PREFILL_POLICY_ROW: ("--router-prefill-policy",),
        ROUTER_DECODE_POLICY_ROW: ("--router-decode-policy",),
        WORKER_SELECTION_POLICY_ROW: (),
    }[row.name]
    wanted_env = {
        ROUTER_PREFILL_POLICY_ROW: ("DYN_ROUTER_PREFILL_POLICY",),
        ROUTER_DECODE_POLICY_ROW: ("DYN_ROUTER_DECODE_POLICY",),
        WORKER_SELECTION_POLICY_ROW: ("DYN_ROUTER_WORKER_SELECTION_POLICY",),
    }[row.name]
    wanted_schema = {
        ROUTER_PREFILL_POLICY_ROW: ("router_prefill_policy",),
        ROUTER_DECODE_POLICY_ROW: ("router_decode_policy",),
        WORKER_SELECTION_POLICY_ROW: (),
    }[row.name]
    if row.flags != wanted_flags:
        raise AssertionError(f"{row.name} flags {row.flags} != {wanted_flags!r}")
    if row.env != wanted_env:
        raise AssertionError(f"{row.name} env {row.env} != {wanted_env!r}")
    if row.schema_fields != wanted_schema:
        raise AssertionError(
            f"{row.name} schema_fields {row.schema_fields} != {wanted_schema!r}"
        )
    item = _require_unique_nodeid_item(items, NAMED_POLICY_CATALOG_ITEM)
    _assert_ordinary_collected_item(NAMED_POLICY_CATALOG_ITEM, item)


# Closed seven-row binding oracle: status/observer/ids/items only.
_SEVEN_ROW_BINDINGS: dict[str, dict[str, Any]] = {
    SERVE_INDEXER_ROW: {
        "status": STATUS_UNSUPPORTED_INPUT,
        "observer": INPUT_STARTUP_SOURCE_OBSERVER,
        "flags": ("--serve-indexer", "--no-serve-indexer"),
        "env": ("DYN_SERVE_INDEXER",),
        "schema_fields": ("serve_indexer",),
        "items": (INPUT_STARTUP_SOURCE_ITEM,),
        "require_session_ttl": False,
    },
    MIN_INITIAL_WORKERS_ROW: {
        "status": STATUS_UNSUPPORTED_CAPABILITY,
        "observer": INPUT_STARTUP_SOURCE_OBSERVER,
        "flags": ("--router-min-initial-workers",),
        "env": ("DYN_ROUTER_MIN_INITIAL_WORKERS",),
        "schema_fields": ("min_initial_workers",),
        "items": (INPUT_STARTUP_SOURCE_ITEM,),
        "require_session_ttl": False,
    },
    ROUTER_MODE_ROW: {
        "status": STATUS_UNSUPPORTED_INPUT,
        "observer": INPUT_STARTUP_SOURCE_OBSERVER,
        "flags": ("--router-mode",),
        "env": ("DYN_ROUTER_MODE",),
        "schema_fields": ("router_mode",),
        "items": (INPUT_STARTUP_SOURCE_ITEM,),
        "require_session_ttl": False,
    },
    SESSION_AFFINITY_MODE_ROW: {
        "status": STATUS_UNSUPPORTED_CAPABILITY,
        "observer": INPUT_STARTUP_SOURCE_OBSERVER,
        "flags": ("--router-session-affinity-mode",),
        "env": ("DYN_ROUTER_SESSION_AFFINITY_MODE",),
        "schema_fields": ("session_affinity_mode",),
        "items": (INPUT_STARTUP_SOURCE_ITEM,),
        "require_session_ttl": True,
    },
    PREFILL_LOAD_MODEL_ROW: {
        "status": STATUS_UNSUPPORTED_CAPABILITY,
        "observer": INPUT_STARTUP_SOURCE_OBSERVER,
        "flags": ("--router-prefill-load-model",),
        "env": ("DYN_ROUTER_PREFILL_LOAD_MODEL",),
        "schema_fields": ("router_prefill_load_model",),
        "items": (INPUT_STARTUP_SOURCE_ITEM,),
        "require_session_ttl": False,
    },
    PREFILL_ABS_THRESHOLD_ROW: {
        "status": STATUS_UNSUPPORTED_INPUT,
        "observer": PREFILL_ADMISSION_OBSERVER,
        "flags": ("--active-prefill-tokens-threshold",),
        "env": (PREFILL_ABS_ENV,),
        "schema_fields": ("active_prefill_tokens_threshold",),
        "items": (INPUT_STARTUP_SOURCE_ITEM, *PREFILL_ADMISSION_ITEMS),
        "require_session_ttl": False,
    },
    PREFILL_FRAC_THRESHOLD_ROW: {
        "status": STATUS_UNSUPPORTED_INPUT,
        "observer": PREFILL_ADMISSION_OBSERVER,
        "flags": ("--active-prefill-tokens-threshold-frac",),
        "env": (PREFILL_FRAC_ENV,),
        "schema_fields": ("active_prefill_tokens_threshold_frac",),
        "items": (INPUT_STARTUP_SOURCE_ITEM, *PREFILL_ADMISSION_ITEMS),
        "require_session_ttl": False,
    },
}


def _assert_seven_row_binding(row: SettingDecision, items: Iterable[Any]) -> None:
    spec = _SEVEN_ROW_BINDINGS.get(row.name)
    if spec is None:
        raise AssertionError(
            f"{row.name} is not in the closed seven-row binding oracle"
        )
    for field in ("status", "observer", "flags", "env", "schema_fields"):
        got = getattr(row, field)
        wanted = spec[field]
        if got != wanted:
            raise AssertionError(f"{row.name} {field} {got!r} != {wanted!r}")
    if row.profiles != ():
        raise AssertionError(f"{row.name} profiles {row.profiles} != ()")
    if row.affected_fields != (row.name,):
        raise AssertionError(
            f"{row.name} affected_fields {row.affected_fields} != {(row.name,)!r}"
        )
    for item_id in spec["items"]:
        item = _require_unique_nodeid_item(items, item_id)
        _assert_ordinary_collected_item(item_id, item)
    if spec["require_session_ttl"]:
        ttl_item = _require_unique_nodeid_item(items, SESSION_TTL_ITEM)
        skips = list(ttl_item.iter_markers(name="skip")) + list(
            ttl_item.iter_markers(name="skipif")
        )
        marks = list(ttl_item.iter_markers(name="xfail"))
        if skips or len(marks) != 1:
            raise AssertionError(
                f"{SESSION_TTL_ITEM} must collect exactly one xfail and no skip "
                f"as same-feature Hard/TTL300 evidence, found xfail={len(marks)} "
                f"skip={len(skips)}"
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
    bind only their exact existing collected Items. overlap_refresh_after_secs
    binds only its ordinary owned-process queue-INFO plus token-parity Item.
    router_tracking_hash and router_tracking_key_id bind only the ordinary
    keyed-tracking runtime Item. router_tracking_key_file binds that runtime
    Item and both ordinary frontend/epp size-control Items.
    router_approximate_cache_policy binds the ordinary LRU and omitted-env
    default-TTL core-metrics Items. active_decode_blocks_threshold binds
    only its owned worker-monitor remote-load Item with one unconditional
    strict xfail of ActiveDecodeThresholdConfigMismatch.
    enable_session_prefix_index binds only its owned config-parity Item with
    one unconditional strict xfail of SessionPrefixIndexConfigMismatch.
    shared_cache_type binds only its ordinary owned HiCache
    client-construction Item. kv_cache_block_size binds the three ordinary
    Dynamic worker-card cross-input Items. The five input/startup/capability
    source rows bind only ordinary test_input_startup_capability_decisions;
    session_affinity_mode additionally requires the existing SESSION_TTL_ITEM
    Hard/TTL300 strict marker. Prefill abs/frac unsupported_input rows bind
    that source Item plus the three ordinary
    test_prefill_admission_input_boundaries profiles. The four native
    active-sequence rows bind only their consuming
    test_native_consumer_profile[<profile>] Items. The three named-policy
    selector rows bind only the ordinary test_named_policy_catalog Item.
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
        if row.name == OVERLAP_REFRESH_ROW:
            _assert_overlap_refresh_binding(row, item_list)
            continue
        if row.name in _TRACKING_APPROXIMATE_BINDINGS:
            _assert_tracking_approximate_binding(row, item_list)
            continue
        if row.name == DECODE_THRESHOLD_ROW:
            _assert_decode_threshold_binding(row, item_list)
            continue
        if row.name == SESSION_PREFIX_INDEX_ROW:
            _assert_session_prefix_index_binding(row, item_list)
            continue
        if row.name == SHARED_CACHE_TYPE_ROW:
            _assert_shared_cache_type_binding(row, item_list)
            continue
        if row.name in _NATIVE_CONSUMER_ROW_PROFILES:
            _assert_native_consumer_binding(row, item_list)
            continue
        if row.name in set(NAMED_POLICY_CATALOG_ROWS):
            _assert_named_policy_catalog_binding(row, item_list)
            continue
        if row.name in _SEVEN_ROW_BINDINGS:
            _assert_seven_row_binding(row, item_list)
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
        if mapped is not None:
            return mapped
        if name == "router_tracking_key_id":
            text = payload
            if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
                text = text[1:-1]
            if text == expected:
                return text
        if name == SHARED_CACHE_TYPE_ROW:
            mapped_type = _SHARED_CACHE_TYPE_EVENT_VALUES.get(payload)
            if mapped_type is not None:
                return mapped_type
        raise AssertionError(
            f"{name}={raw!r} is not a known enum display/serde tracing form"
        )
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
    kv_router_kwargs: Mapping[str, Any]


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
    *,
    require_kv: bool = True,
    wrapper_cli: tuple[str, ...] | None = None,
) -> FrontendRouterSnapshot:
    from dynamo.common.configuration.groups.router_args import (
        _ROUTER_FIELDS,
        build_router_config,
    )
    from dynamo.frontend.frontend_args import FrontendArgGroup, FrontendConfig
    from dynamo.llm import RouterMode

    parser = argparse.ArgumentParser(add_help=False)
    FrontendArgGroup().add_arguments(parser)
    base_cli = WRAPPER_CLI if wrapper_cli is None else wrapper_cli
    namespace = parser.parse_args(list(base_cli) + list(extra_cli))
    config = FrontendConfig.from_cli_args(namespace)
    config.validate()
    router = build_router_config(config)
    if router is None:
        raise AssertionError("build_router_config returned None")
    # PyO3 RouterConfig::new uses unwrap_or_default for kv_router_config, so the
    # Python attribute is never None even for non-KV modes. Observe the actual
    # object/JSON; do not invent a None API.
    kv_json = json.loads(router.kv_router_config.to_json())
    kv_kwargs = dict(config.kv_router_kwargs())
    if require_kv:
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
        kv_json,
        tuple(_ROUTER_FIELDS),
        kv_kwargs,
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
    if dict(baseline.kv_router_kwargs) != dict(changed.kv_router_kwargs):
        raise AssertionError(
            "kv_router_kwargs changed "
            f"{dict(baseline.kv_router_kwargs)} -> "
            f"{dict(changed.kv_router_kwargs)}"
        )
    if dict(baseline.kv_json) != dict(changed.kv_json):
        raise AssertionError(
            "kv_router_config.to_json changed "
            f"{dict(baseline.kv_json)} -> {dict(changed.kv_json)}"
        )
    LOGGER.info(
        "ignored-compat snapshots baseline_json=%r changed_json=%r kwargs=%r "
        "kv_kwargs=%r",
        dict(baseline.kv_json),
        dict(changed.kv_json),
        dict(changed.router_kwargs),
        dict(changed.kv_router_kwargs),
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


def _unique_ast(nodes: list[Any], label: str) -> Any:
    if len(nodes) != 1:
        raise AssertionError(f"expected exactly one {label}, found {len(nodes)}")
    return nodes[0]


def _expr_eq(node: ast.AST | None, expr: str) -> bool:
    if node is None:
        return False
    wanted = ast.parse(expr, mode="eval").body
    return ast.dump(node, include_attributes=False) == ast.dump(
        wanted, include_attributes=False
    )


def _kw(call: ast.Call, name: str) -> ast.AST | None:
    return next((kw.value for kw in call.keywords if kw.arg == name), None)


def assert_session_inputs_stack_uses_ttl_300() -> None:
    """Finite Hard/TTL300 named-boundary AST check (not a dataflow proof)."""
    hard_env = (
        '{"DYN_ROUTER_SESSION_AFFINITY_TTL_SECS": str(self.session_affinity_ttl_secs), '
        '"DYN_ROUTER_SESSION_AFFINITY_MODE": "hard"}'
    )
    if not INPUTS_PARITY.is_file():
        raise AssertionError(f"missing session inputs thin test: {INPUTS_PARITY}")
    inputs = ast.parse(
        INPUTS_PARITY.read_text(encoding="utf-8"), filename=str(INPUTS_PARITY)
    )
    func = _unique_ast(
        [
            n
            for n in inputs.body
            if isinstance(n, ast.FunctionDef)
            and n.name == "test_epp_frontend_inputs_parity"
        ],
        "top-level test_epp_frontend_inputs_parity",
    )
    with_node = _unique_ast(
        [n for n in func.body if isinstance(n, ast.With)], "direct function.body With"
    )
    if len(with_node.items) != 1:
        raise AssertionError("session With must bind exactly one item")
    item = with_node.items[0]
    call = item.context_expr
    if not isinstance(call, ast.Call) or not _expr_eq(
        call.func, "EppFrontendRoutingParityStack"
    ):
        raise AssertionError("final With is not EppFrontendRoutingParityStack")
    if not _expr_eq(_kw(call, "session_affinity_ttl_secs"), str(SESSION_TTL_SECS)):
        raise AssertionError(
            f"final With lost session_affinity_ttl_secs={SESSION_TTL_SECS}"
        )
    if not isinstance(item.optional_vars, ast.Name):
        raise AssertionError("session With lost bound stack name")
    body = _unique_ast(with_node.body, "direct with-body statement")
    if not isinstance(body, ast.Expr) or not isinstance(body.value, ast.Call):
        raise AssertionError("with-body is not a call")
    if not _expr_eq(body.value.func, "run_frontend_epp_session_id_parity"):
        raise AssertionError("with-body is not run_frontend_epp_session_id_parity")
    if not body.value.args or not _expr_eq(body.value.args[0], item.optional_vars.id):
        raise AssertionError("session runner lost the bound stack variable")

    parity_path = REPO_ROOT / "tests/router/epp_parity.py"
    if not parity_path.is_file():
        raise AssertionError(f"missing routing parity source: {parity_path}")
    parity = ast.parse(
        parity_path.read_text(encoding="utf-8"), filename=str(parity_path)
    )
    cls = _unique_ast(
        [
            n
            for n in parity.body
            if isinstance(n, ast.ClassDef) and n.name == "EppFrontendRoutingParityStack"
        ],
        "top-level EppFrontendRoutingParityStack",
    )
    method = _unique_ast(
        [
            n
            for n in cls.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name == "open_consumer_pair"
        ],
        "open_consumer_pair",
    )
    try_node = _unique_ast(
        [n for n in method.body if isinstance(n, ast.Try)],
        "direct Try in open_consumer_pair.body",
    )
    ttl_if = _unique_ast(
        [
            n
            for n in try_node.body
            if isinstance(n, ast.If)
            and _expr_eq(n.test, "self.session_affinity_ttl_secs is not None")
        ],
        "direct TTL If in try.body",
    )
    session_assign = _unique_ast(
        [
            n
            for n in ttl_if.body
            if isinstance(n, ast.Assign)
            and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id == "session_env"
        ],
        "session_env assign in TTL If",
    )
    if not _expr_eq(session_assign.value, hard_env):
        raise AssertionError("TTL If lost Hard/TTL session_env dict")
    if not any(
        isinstance(n, ast.Expr)
        and _expr_eq(n.value, "frontend_env.update(session_env)")
        for n in ttl_if.body
    ):
        raise AssertionError("TTL If lost frontend_env.update(session_env)")
    fe_assign = _unique_ast(
        [
            n
            for n in try_node.body
            if isinstance(n, ast.Assign)
            and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id == "frontend"
            and isinstance(n.value, ast.Call)
            and _expr_eq(n.value.func, "FrontendRouterProcess")
        ],
        "direct FrontendRouterProcess assign in try.body",
    )
    if not isinstance(fe_assign.value, ast.Call):
        raise AssertionError("FrontendRouterProcess assign value is not a call")
    if not _expr_eq(_kw(fe_assign.value, "extra_env"), "frontend_env"):
        raise AssertionError("FrontendRouterProcess lost extra_env=frontend_env")
    if not _expr_eq(
        _kw(fe_assign.value, "session_affinity_ttl_secs"),
        "self.session_affinity_ttl_secs",
    ):
        raise AssertionError(
            "FrontendRouterProcess lost "
            "session_affinity_ttl_secs=self.session_affinity_ttl_secs"
        )
    epp_if = _unique_ast(
        [
            n
            for n in try_node.body
            if isinstance(n, ast.If) and _expr_eq(n.test, "include_epp")
        ],
        "direct if include_epp in try.body",
    )
    session_if = _unique_ast(
        [
            n
            for n in epp_if.body
            if isinstance(n, ast.If) and _expr_eq(n.test, "session_env is not None")
        ],
        "direct if session_env is not None in include_epp",
    )
    if not any(
        isinstance(n, ast.Expr) and _expr_eq(n.value, "epp_env.update(session_env)")
        for n in session_if.body
    ):
        raise AssertionError("session_env branch lost epp_env.update(session_env)")
    epp_assign = _unique_ast(
        [
            n
            for n in epp_if.body
            if isinstance(n, ast.Assign)
            and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id == "epp"
            and isinstance(n.value, ast.Call)
            and _expr_eq(n.value.func, "RuntimeEppProcess")
        ],
        "direct RuntimeEppProcess assign in include_epp",
    )
    if not isinstance(epp_assign.value, ast.Call) or not _expr_eq(
        _kw(epp_assign.value, "extra_env"), "epp_env"
    ):
        raise AssertionError("RuntimeEppProcess lost extra_env=epp_env")


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
        raise AssertionError(
            "positional factory sources missing: " + ", ".join(missing)
        )
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


def _collapsed_rust(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _assert_overlap_refresh_source_contract() -> None:
    missing = [
        str(path)
        for path in (OVERLAP_REFRESH_RS, QUEUE_RS, INDEXER_BACKEND_RS, WORKERS_RS)
        if not path.is_file()
    ]
    if missing:
        raise AssertionError("overlap refresh sources missing: " + ", ".join(missing))
    refresh = _stripped_production_rust_text(OVERLAP_REFRESH_RS)
    defaults = list(_DEFAULT_OVERLAP_REFRESH_RE.finditer(refresh))
    if len(defaults) != 1:
        raise AssertionError(
            "DEFAULT_OVERLAP_REFRESH_AFTER_SECS is missing, duplicate, or "
            f"ambiguous: found {len(defaults)}"
        )
    default_secs = int(defaults[0].group(1))
    if default_secs != OVERLAP_REFRESH_DEFAULT_SECS or default_secs <= 0:
        raise AssertionError(
            "DEFAULT_OVERLAP_REFRESH_AFTER_SECS="
            f"{default_secs} is not the documented positive "
            f"{OVERLAP_REFRESH_DEFAULT_SECS}s default; sentinel "
            f"{OVERLAP_REFRESH_SENTINEL} would no longer be nondefault"
        )
    read_body = _unique_scoped_rust_body(
        refresh,
        _READ_OVERLAP_REFRESH_RE,
        "read_overlap_refresh_after",
    )
    if f'"{OVERLAP_REFRESH_ENV}"' not in read_body:
        raise AssertionError(
            "read_overlap_refresh_after no longer consumes " f"{OVERLAP_REFRESH_ENV}"
        )
    if "ifparsed<=0.0{returnNone;}" not in _collapsed_rust(read_body):
        raise AssertionError(
            "read_overlap_refresh_after no longer maps parsed<=0 to None"
        )
    queue = _stripped_production_rust_text(QUEUE_RS)
    if queue.count(OVERLAP_REFRESH_DISABLED) != 1:
        raise AssertionError(
            "queue.rs must contain exactly one disabled overlap-refresh INFO"
        )
    if queue.count(OVERLAP_REFRESH_ENABLED) != 1:
        raise AssertionError(
            "queue.rs must contain exactly one enabled overlap-refresh INFO"
        )
    if_body = _unique_scoped_rust_body(
        queue,
        _QUEUE_REFRESH_IF_RE,
        "overlap_refresh_after constructor branch",
    )
    collapsed_if = _collapsed_rust(if_body)
    if "read_overlap_refresh_after()" not in if_body:
        raise AssertionError(
            "queue constructor no longer calls read_overlap_refresh_after()"
        )
    if (
        'None=>tracing::info!("' + _collapsed_rust(OVERLAP_REFRESH_DISABLED) + '")'
        not in collapsed_if
    ):
        raise AssertionError(
            "queue constructor no longer logs disabled only for the None "
            "read_overlap_refresh_after result"
        )
    if _collapsed_rust(OVERLAP_REFRESH_ENABLED) not in collapsed_if:
        raise AssertionError(
            "queue constructor no longer logs enabled for a Some refresh delay"
        )
    backend = _stripped_production_rust_text(INDEXER_BACKEND_RS)
    supports_body = _unique_scoped_rust_body(
        backend,
        _SUPPORTS_OVERLAP_REFRESH_RE,
        "Indexer::supports_overlap_refresh",
    )
    if "matches!(self,Self::Single{..}|Self::Concurrent{..})" not in _collapsed_rust(
        supports_body
    ):
        raise AssertionError(
            "supports_overlap_refresh no longer matches Single or Concurrent"
        )
    workers = _stripped_production_rust_text(WORKERS_RS)
    if "indexer.supports_overlap_refresh()" not in workers:
        raise AssertionError(
            "workers.rs no longer gates the refresher on supports_overlap_refresh"
        )
    if "TieredOverlapRefresher::new" not in workers:
        raise AssertionError("workers.rs no longer constructs TieredOverlapRefresher")


def _owned_process_log_text(
    process, *, label: str, node_name: str
) -> tuple[int, Path, str]:
    if process is None:
        raise AssertionError(f"EppFrontendParityStack.{label} was not saved")
    proc = process.proc
    if proc is None or proc.poll() is not None:
        raise AssertionError(f"owned {label} process is not live")
    log_path = process.log_path
    if not log_path:
        raise AssertionError(f"owned {label} process has no log_path")
    log = Path(log_path)
    if node_name not in log.as_posix():
        raise AssertionError(f"{label} log_path {log} is not this node {node_name}")
    if not log.is_file():
        raise AssertionError(f"{label} log is missing: {log}")
    data = log.read_bytes()
    text = completed_utf8_log_prefix(data, label=f"{label} log {log}")
    return proc.pid, log, text


def mark_owned_log_cursor(
    process, *, label: str, node_name: str, log_label: str
) -> tuple[int, Path, int]:
    if not log_label:
        raise AssertionError(f"{label} log_label must be nonempty")
    pid, log, text = _owned_process_log_text(process, label=label, node_name=node_name)
    if log_label not in log.as_posix():
        raise AssertionError(
            f"{label} log_path {log} is not this profile label {log_label}"
        )
    return pid, log, len(text.encode("utf-8"))


def _parse_owned_decode_load_threshold(line: str, *, worker_id: int, dp_rank: int):
    cleaned = ANSI_RE.sub("", line)
    if DECODE_LOAD_TARGET not in cleaned or DECODE_LOAD_EVENT not in cleaned:
        return _DECODE_SKIP
    observation = _DECODE_OBSERVATION_RE.search(cleaned)
    if observation is None:
        raise AssertionError(
            f"owned {DECODE_LOAD_EVENT} is missing observation: {cleaned}"
        )
    if observation.group(1) != "Remote":
        return _DECODE_SKIP
    worker = _DECODE_WORKER_ID_RE.search(cleaned)
    rank = _DECODE_DP_RANK_RE.search(cleaned)
    if worker is None or rank is None:
        raise AssertionError(
            f"owned {DECODE_LOAD_EVENT} is missing worker_id/dp_rank: {cleaned}"
        )
    if int(worker.group(1)) != worker_id or int(rank.group(1)) != dp_rank:
        return _DECODE_SKIP
    field = _DECODE_THRESHOLD_RE.search(cleaned)
    if field is None:
        raise AssertionError(
            f"owned {DECODE_LOAD_EVENT} is missing "
            f"{DECODE_THRESHOLD_ROW}: {cleaned}"
        )
    if field.group(1) == "None":
        return None
    try:
        return float(field.group(2))
    except ValueError as exc:
        raise AssertionError(
            f"owned {DECODE_LOAD_EVENT} {DECODE_THRESHOLD_ROW}="
            f"{field.group(1)!r} is not a typed Option: {cleaned}"
        ) from exc


def observe_owned_decode_load_threshold(
    process,
    *,
    label: str,
    node_name: str,
    pid: int,
    log: Path,
    offset: int,
    worker_id: int,
    dp_rank: int,
    deadline: float,
) -> float | None:
    while True:
        current_pid, current_log, text = _owned_process_log_text(
            process, label=label, node_name=node_name
        )
        if current_pid != pid or current_log != log:
            raise AssertionError(
                f"{label} owner moved from pid={pid} log={log} to "
                f"pid={current_pid} log={current_log}"
            )
        raw = text.encode("utf-8")
        if offset > len(raw):
            raise AssertionError(
                f"{label} pid={pid} log={log} shrank below cursor {offset}"
            )
        region = raw[offset:].decode("utf-8")
        values: list[float | None] = []
        for line in region.splitlines():
            parsed = _parse_owned_decode_load_threshold(
                line, worker_id=worker_id, dp_rank=dp_rank
            )
            if parsed is _DECODE_SKIP:
                continue
            if values and parsed != values[0]:
                raise AssertionError(
                    f"{label} pid={pid} log={log} conflicting "
                    f"{DECODE_THRESHOLD_ROW} values {values[0]!r} vs {parsed!r}"
                )
            values.append(parsed)
        if values:
            return values[0]
        if time.monotonic() >= deadline:
            raise AssertionError(
                f"{label} pid={pid} log={log} has no fresh Remote "
                f"{DECODE_LOAD_EVENT} for worker_id={worker_id} dp_rank={dp_rank}"
            )
        remaining = deadline - time.monotonic()
        time.sleep(min(0.05, max(remaining, 0.0)))


def owned_process_log_identity(
    process, *, label: str, node_name: str, log_label: str
) -> tuple[int, Path]:
    if not log_label:
        raise AssertionError(f"{label} log_label must be nonempty")
    pid, log, _text = _owned_process_log_text(process, label=label, node_name=node_name)
    if log_label not in log.as_posix():
        raise AssertionError(
            f"{label} log_path {log} is not this profile label {log_label}"
        )
    return pid, log


def observe_owned_hicache_constructor(
    process,
    *,
    label: str,
    node_name: str,
    log_label: str,
    pid: int,
    log: Path,
    component: str,
    deadline: float,
) -> None:
    if not log_label:
        raise AssertionError(f"{label} log_label must be nonempty")
    if not component:
        raise AssertionError(f"{label} worker_component must be nonempty")
    while True:
        current_pid, current_log, text = _owned_process_log_text(
            process, label=label, node_name=node_name
        )
        if current_pid != pid or current_log != log:
            raise AssertionError(
                f"{label} owner moved from pid={pid} log={log} to "
                f"pid={current_pid} log={current_log}"
            )
        if log_label not in current_log.as_posix():
            raise AssertionError(
                f"{label} log_path {current_log} is not this profile label "
                f"{log_label}"
            )
        matches: list[str] = []
        for line in text.splitlines():
            cleaned = ANSI_RE.sub("", line)
            if HICACHE_CONSTRUCTOR_EVENT not in cleaned:
                continue
            if _HICACHE_TARGET_RE.search(cleaned) is None:
                continue
            found = _HICACHE_COMPONENT_RE.search(cleaned)
            if found is None:
                raise AssertionError(
                    f"owned {HICACHE_CONSTRUCTOR_TARGET} "
                    f"{HICACHE_CONSTRUCTOR_EVENT} is missing quoted "
                    f"worker_component: {cleaned}"
                )
            try:
                decoded = json.loads(found.group(1))
            except json.JSONDecodeError as exc:
                raise AssertionError(
                    f"owned {HICACHE_CONSTRUCTOR_TARGET} "
                    f"{HICACHE_CONSTRUCTOR_EVENT} worker_component is not a "
                    f"quoted readable JSON string: {cleaned}"
                ) from exc
            if not isinstance(decoded, str) or decoded != component:
                raise AssertionError(
                    f"owned {HICACHE_CONSTRUCTOR_TARGET} "
                    f"{HICACHE_CONSTRUCTOR_EVENT} worker_component="
                    f"{decoded!r} != {component!r}: {cleaned}"
                )
            matches.append(cleaned)
        if len(matches) > 1:
            raise AssertionError(
                f"{label} pid={pid} log={log} has {len(matches)} "
                f"{HICACHE_CONSTRUCTOR_EVENT} records for "
                f"worker_component={component}"
            )
        if len(matches) == 1:
            return
        if time.monotonic() >= deadline:
            raise AssertionError(
                f"{label} pid={pid} log={log} has no "
                f"{HICACHE_CONSTRUCTOR_EVENT} for worker_component={component}"
            )
        remaining = deadline - time.monotonic()
        time.sleep(min(0.05, max(remaining, 0.0)))


def _overlap_refresh_log_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw in text.splitlines():
        cleaned = ANSI_RE.sub("", raw)
        if "overlap-score refresh" in cleaned:
            lines.append(cleaned)
    return lines


def _assert_exactly_one_disabled_refresh(
    text: str, *, label: str, pid: int, log: Path
) -> None:
    lines = _overlap_refresh_log_lines(text)
    if not lines:
        raise AssertionError(
            f"{label} pid={pid} log={log} has no overlap-score refresh "
            "queue-construction message"
        )
    if len(lines) != 1:
        raise AssertionError(
            f"{label} pid={pid} log={log} expected exactly one overlap-score "
            f"refresh message, found {len(lines)}: {lines}"
        )
    line = lines[0]
    if OVERLAP_REFRESH_ENABLED in line:
        raise AssertionError(
            f"{label} pid={pid} log={log} logged enabled overlap refresh: " f"{line}"
        )
    if OVERLAP_REFRESH_DISABLED not in line:
        raise AssertionError(
            f"{label} pid={pid} log={log} logged unknown refresh message: " f"{line}"
        )
    if OVERLAP_REFRESH_TARGET not in line:
        raise AssertionError(
            f"{label} pid={pid} log={log} refresh message is not from "
            f"{OVERLAP_REFRESH_TARGET}: {line}"
        )


def _require_owned_disabled_refresh(
    process, *, label: str, node_name: str
) -> tuple[int, Path]:
    deadline = time.monotonic() + OVERLAP_REFRESH_LOG_FLUSH_S
    while True:
        pid, log, text = _owned_process_log_text(
            process, label=label, node_name=node_name
        )
        if _overlap_refresh_log_lines(text):
            _assert_exactly_one_disabled_refresh(text, label=label, pid=pid, log=log)
            return pid, log
        if time.monotonic() >= deadline:
            raise AssertionError(
                f"{label} pid={pid} log={log} has no overlap-score refresh "
                "queue-construction message"
            )
        time.sleep(0.05)


def run_overlap_refresh_effective_config(
    request,
    tmp_path: Path,
    monkeypatch,
    model_name: str,
    stack_cls,
) -> None:
    from tests.router.common import (
        frontend_epp_token_parity_cases,
        run_frontend_epp_token_parity,
    )

    _assert_overlap_refresh_source_contract()
    clear_known_manifest_router_envs(monkeypatch)
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("RUST_LOG", "info")
    monkeypatch.setenv("DYN_LOG", "info")
    monkeypatch.setenv(OVERLAP_REFRESH_ENV, OVERLAP_REFRESH_SENTINEL)
    cases = frontend_epp_token_parity_cases(model_name)
    plain = [case for case in cases if case.name == "plain_chat"]
    if len(plain) != 1:
        raise AssertionError(
            f"expected exactly one plain_chat token case, found {len(plain)}"
        )
    with stack_cls(request, tmp_path, model_name) as stack:
        run_frontend_epp_token_parity(stack, plain, include_negative_control=False)
        frontend_pid, frontend_log = _require_owned_disabled_refresh(
            getattr(stack, "frontend_process", None),
            label="frontend_process",
            node_name=request.node.name,
        )
        epp_pid, epp_log = _require_owned_disabled_refresh(
            getattr(stack, "epp_process", None),
            label="epp_process",
            node_name=request.node.name,
        )
    LOGGER.info(
        "overlap refresh %s=%s default=%ss frontend_disabled=True "
        "epp_disabled=True frontend pid=%s log=%s epp pid=%s log=%s",
        OVERLAP_REFRESH_ENV,
        OVERLAP_REFRESH_SENTINEL,
        OVERLAP_REFRESH_DEFAULT_SECS,
        frontend_pid,
        frontend_log,
        epp_pid,
        epp_log,
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


def _plain_chat_token_case(model_name: str):
    from tests.router.common import frontend_epp_token_parity_cases

    cases = frontend_epp_token_parity_cases(model_name)
    plain = [case for case in cases if case.name == "plain_chat"]
    if len(plain) != 1:
        raise AssertionError(
            f"expected exactly one plain_chat token case, found {len(plain)}"
        )
    return plain


def _write_tracking_key_file(path: Path, payload: bytes) -> dict[str, Any]:
    path.write_bytes(payload)
    if path.read_bytes() != payload:
        raise AssertionError(
            f"tracking key file {path} was not written as public bytes"
        )
    return {
        "path": str(path),
        "length": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _keyed_tracking_consumer_env(key_path: Path) -> dict[str, str]:
    return {
        "DYN_ROUTER_TRACKING_HASH": TRACKING_HASH_ALGORITHM,
        "DYN_ROUTER_TRACKING_KEY_FILE": str(key_path),
        "DYN_ROUTER_TRACKING_KEY_ID": TRACKING_KEY_ID,
    }


def _approximate_consumer_env(policy_name: str) -> dict[str, str | None]:
    if policy_name not in APPROXIMATE_POLICY_NAMES:
        raise AssertionError(f"unknown approximate policy profile {policy_name!r}")
    env: dict[str, str | None] = {
        "DYN_ROUTER_USE_KV_EVENTS": "false",
        "DYN_USE_REMOTE_INDEXER": "false",
        "DYN_ROUTER_APPROXIMATE_CACHE_POLICY": None,
    }
    if policy_name == "lru":
        env["DYN_ROUTER_APPROXIMATE_CACHE_POLICY"] = "lru"
    return env


def decode_threshold_consumer_env(profile_name: str) -> dict[str, str | None]:
    if profile_name not in DECODE_THRESHOLD_PROFILES:
        raise AssertionError(f"unknown decode-threshold profile {profile_name!r}")
    env: dict[str, str | None] = {
        DECODE_THRESHOLD_ENV: None,
        "DYN_LOG": DECODE_CONSUMER_LOG,
        "RUST_LOG": DECODE_CONSUMER_LOG,
        "DYN_SDK_DISABLE_ANSI_LOGGING": "1",
    }
    for name in DECODE_THRESHOLD_PREFILL_ENVS:
        env[name] = None
    if profile_name == "configured":
        env[DECODE_THRESHOLD_ENV] = DECODE_THRESHOLD_CONFIGURED
    return env


def shared_cache_type_consumer_env(profile_name: str) -> dict[str, str | None]:
    if profile_name not in SHARED_CACHE_TYPE_PROFILES:
        raise AssertionError(f"unknown shared-cache-type profile {profile_name!r}")
    env: dict[str, str | None] = {
        SHARED_CACHE_TYPE_ENV: None,
        "DYN_MOONCAKE_KV_EVENTS_ENDPOINT": None,
        "DYN_LOG": "info",
        "RUST_LOG": "info",
        "DYN_SDK_DISABLE_ANSI_LOGGING": "1",
        "DYN_LOGGING_CONSOLE_FORMAT": "readable",
    }
    if profile_name == SHARED_CACHE_TYPE_HICACHE:
        env[SHARED_CACHE_TYPE_ENV] = SHARED_CACHE_TYPE_HICACHE
    return env


def session_prefix_index_consumer_env(
    *,
    session_prefix: str | None,
    use_kv_events: bool,
    policy_config: str | None = None,
) -> dict[str, str | None]:
    """Owned FE/EPP env for session-prefix input/config validation controls."""
    return {
        SESSION_PREFIX_INDEX_ENV: session_prefix,
        "DYN_ROUTER_USE_KV_EVENTS": "true" if use_kv_events else "false",
        "DYN_USE_KV_EVENTS": None,
        "DYN_USE_REMOTE_INDEXER": "false",
        "DYN_ROUTER_POLICY_CONFIG": policy_config,
        "DYN_LOG": "info",
        "RUST_LOG": "info",
        "DYN_SDK_DISABLE_ANSI_LOGGING": "1",
        "NO_COLOR": "1",
    }


def assert_frontend_session_prefix_index_false(*, profile: str) -> None:
    """Require FE config attrs and real kv_router_kwargs false after construction."""
    # KvRouterConfigBase is frontend-owned; keep collection-safe by importing
    # only inside this selected-test helper.
    from dynamo.common.configuration.groups.kv_router_args import KvRouterConfigBase

    default = KvRouterConfigBase.enable_session_prefix_index
    if default is not False:
        raise AssertionError(
            f"{profile} authoritative KvRouterConfigBase."
            f"{SESSION_PREFIX_INDEX_ROW}={default!r} is not False"
        )
    snapshot = _frontend_router_snapshot()
    attr = snapshot.config_attrs.get(SESSION_PREFIX_INDEX_ROW)
    if attr is not False:
        raise AssertionError(
            f"{profile} FE config attr {SESSION_PREFIX_INDEX_ROW}={attr!r} "
            "is not False"
        )
    kwargs_value = snapshot.kv_router_kwargs.get(SESSION_PREFIX_INDEX_ROW)
    if kwargs_value is not False:
        raise AssertionError(
            f"{profile} FE kv_router_kwargs {SESSION_PREFIX_INDEX_ROW}="
            f"{kwargs_value!r} is not False"
        )


def _core_metric_name(suffix: str) -> str:
    from dynamo.prometheus_names import name_prefix

    return f"{name_prefix.COMPONENT}_{suffix}"


def _assert_core_policy_gauges(
    content: str,
    *,
    role: str,
    pid: int,
    port: int,
    namespace: str,
    component: str,
    selected: str,
) -> None:
    from dynamo.prometheus_names import labels
    from tests.utils.prometheus import find_metric_samples

    if selected not in APPROXIMATE_POLICY_LABELS:
        raise AssertionError(f"{role} selected policy {selected!r} is unknown")
    if not namespace or not component:
        raise AssertionError(
            f"{role} owned mocker identity is missing "
            f"namespace={namespace!r} component={component!r}"
        )
    configured_name = _core_metric_name(APPROXIMATE_CONFIGURED_METRIC)
    effective_name = _core_metric_name(APPROXIMATE_EFFECTIVE_METRIC)
    namespace_label = namespace.replace("-", "_")
    identity = {
        labels.NAMESPACE: namespace_label,
        labels.COMPONENT: component,
    }
    for metric_name in (configured_name, effective_name):
        family = find_metric_samples(content, metric_name)
        if not family:
            raise AssertionError(
                f"{role} pid={pid} port={port} {metric_name} family is missing"
            )
        matched: dict[str, float] = {}
        for policy in APPROXIMATE_POLICY_LABELS:
            samples = find_metric_samples(
                content,
                metric_name,
                {**identity, "policy": policy},
            )
            if not samples:
                raise AssertionError(
                    f"{role} pid={pid} port={port} {metric_name} "
                    f"namespace={namespace_label!r} component={component!r} "
                    f"policy={policy!r} series is missing"
                )
            if len(samples) != 1:
                raise AssertionError(
                    f"{role} pid={pid} port={port} {metric_name} "
                    f"namespace={namespace_label!r} component={component!r} "
                    f"policy={policy!r} has {len(samples)} series"
                )
            matched[policy] = samples[0]
        if len(family) != len(matched):
            raise AssertionError(
                f"{role} pid={pid} port={port} {metric_name} has "
                f"{len(family)} family samples; {len(matched)} match "
                f"namespace={namespace_label!r} component={component!r}"
            )
        for policy, value in matched.items():
            wanted = 1.0 if policy == selected else 0.0
            if value != wanted:
                raise AssertionError(
                    f"{role} pid={pid} port={port} {metric_name} "
                    f"{policy}={value} != {wanted}; selected={selected}"
                )
        if matched[selected] != 1.0:
            raise AssertionError(
                f"{role} pid={pid} port={port} {metric_name} did not "
                f"observe selected {selected}"
            )


def _split_rust_call_args(arg_text: str) -> list[str]:
    """Split a known-call argument interior on zero-depth commas."""
    args: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(arg_text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth < 0:
                raise AssertionError("call-arg delimiter underflow while splitting")
        elif char == "," and depth == 0:
            part = arg_text[start:index].strip()
            if part:
                args.append(part)
            start = index + 1
    if depth != 0:
        raise AssertionError("call-arg delimiters unbalanced while splitting")
    tail = arg_text[start:].strip()
    if tail:
        args.append(tail)
    return args


def _require_unique_callee_args(
    scope: str,
    text: str,
    callee_prefix: re.Pattern[str],
    label: str,
    arity: int,
) -> list[str]:
    """Extract args for a unique callee prefix ending at '(' via balanced scan."""
    matches = list(callee_prefix.finditer(text))
    if len(matches) != 1:
        raise AssertionError(
            f"{label} in {scope} missing/duplicate: found {len(matches)}"
        )
    open_at = matches[0].end() - 1
    if open_at < 0 or text[open_at] != "(":
        raise AssertionError(f"{label} prefix must end at opening '('")
    depth = 1
    index = open_at + 1
    while index < len(text) and depth:
        char = text[index]
        if char == '"':
            raise AssertionError(
                f"{label} in {scope} has unsupported quoted form; review call extraction"
            )
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth < 0:
                raise AssertionError(f"{label} in {scope} delimiter underflow")
        index += 1
    if depth != 0:
        raise AssertionError(f"{label} in {scope} has unbalanced delimiters")
    args = _split_rust_call_args(text[open_at + 1 : index - 1])
    if len(args) != arity:
        raise AssertionError(
            f"{label} in {scope} arity {len(args)} != {arity}: {args!r}"
        )
    return args


def _fn_decl_and_body(text: str, pattern: re.Pattern[str], label: str) -> str:
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise AssertionError(
            f"{label} declaration missing/duplicate/ambiguous: found {len(matches)}"
        )
    match = matches[0]
    brace = match.end() - 1
    if brace < 0 or text[brace] != "{":
        raise AssertionError(f"{label} is not followed by a scoped body")
    depth = 1
    index = brace + 1
    while index < len(text) and depth:
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth < 0:
                raise AssertionError(f"{label} scoped body delimiter underflow")
        index += 1
    if depth:
        raise AssertionError(f"{label} has an unbalanced scoped body")
    return text[match.start() : index]


@dataclass(frozen=True)
class _SourceScope:
    key: str
    path: Path
    pattern: re.Pattern[str]
    label: str


# Fifteen decisive shared scopes (declaration+body digest). Readable checks
# below protect named support boundaries; not a formal proof of all future Rust.
_INPUT_STARTUP_SCOPES: tuple[_SourceScope, ...] = (
    _SourceScope(
        "epp_from_discovery",
        RUST_EPP,
        re.compile(
            r"pub(?:\(crate\))?\s+(?:async\s+)?fn\s+from_discovery\s*\([^;{]*?\)\s*"
            r"(?:->[^;{]+)?\s*\{"
        ),
        "EPP Router::from_discovery",
    ),
    _SourceScope(
        "epp_route_decode",
        RUST_EPP,
        re.compile(
            r"pub(?:\(crate\))?\s+(?:async\s+)?fn\s+route_decode\s*\([^;{]*?\)\s*"
            r"(?:->[^;{]+)?\s*\{"
        ),
        "EPP Router::route_decode",
    ),
    _SourceScope(
        "managed_kv_router_for_with_policy",
        RUST_MODEL_MANAGER,
        re.compile(
            r"pub\s+async\s+fn\s+managed_kv_router_for_with_policy\s*\([^;{]*?\)\s*"
            r"(?:->[^;{]+)?\s*\{"
        ),
        "ModelManager::managed_kv_router_for_with_policy",
    ),
    _SourceScope(
        "kv_chooser_for_with_policy_and_client",
        RUST_MODEL_MANAGER,
        re.compile(
            r"pub\s+async\s+fn\s+kv_chooser_for_with_policy_and_client\s*\([^;{]*?\)\s*"
            r"(?:->[^;{]+)?\s*\{"
        ),
        "ModelManager::kv_chooser_for_with_policy_and_client",
    ),
    _SourceScope(
        "prefill_router_new",
        RUST_PREFILL_ACTIVATION,
        re.compile(r"pub\s+fn\s+new\s*\([^;{]*?\)\s*(?:->[^;{]+)?\s*\{"),
        "PrefillRouter::new",
    ),
    _SourceScope(
        "embedded_selection_start",
        RUST_EMBEDDED,
        re.compile(
            r"pub(?:\(crate\))?\s+(?:async\s+)?fn\s+start\s*\([^;{]*?\)\s*"
            r"(?:->[^;{]+)?\s*\{"
        ),
        "EmbeddedSelection::start",
    ),
    _SourceScope(
        "min_initial_workers_from_env",
        RUST_CONFIG,
        re.compile(
            r"pub\s+fn\s+min_initial_workers_from_env\s*\(\s*\)\s*"
            r"->\s*anyhow::Result<usize>\s*\{"
        ),
        "min_initial_workers_from_env",
    ),
    _SourceScope(
        "ensure_served_indexer_service",
        RUST_INDEXER_REMOTE,
        re.compile(
            r"pub\s+async\s+fn\s+ensure_served_indexer_service\s*\([^;{]*?\)\s*"
            r"(?:->[^;{]+)?\s*\{"
        ),
        "ensure_served_indexer_service",
    ),
    _SourceScope(
        "managed_kv_router_for_with_worker_role",
        RUST_MODEL_MANAGER,
        re.compile(
            r"pub\s+async\s+fn\s+managed_kv_router_for_with_worker_role\s*\([^;{]*?\)\s*"
            r"->\s*anyhow::Result<crate::kv_router::ManagedKvRouter>\s*\{"
        ),
        "ModelManager::managed_kv_router_for_with_worker_role",
    ),
    _SourceScope(
        "kv_router_new_with_worker_role_and_scheduler_load",
        RUST_KV_ROUTER,
        re.compile(
            r"pub\(crate\)\s+async\s+fn\s+new_with_worker_role_and_scheduler_load\s*"
            r"\([^;{]*?\)\s*->\s*Result<Self>\s*\{"
        ),
        "KvRouter::new_with_worker_role_and_scheduler_load",
    ),
    _SourceScope(
        "prefill_router_new_with_selection_policy",
        RUST_PREFILL_ACTIVATION,
        re.compile(
            r"pub\(crate\)\s+fn\s+new_with_selection_policy\s*\([^;{]*?\)\s*"
            r"->\s*Arc<Self>\s*\{"
        ),
        "PrefillRouter::new_with_selection_policy",
    ),
    _SourceScope(
        "prefill_build_binding",
        RUST_PREFILL_ACTIVATION,
        re.compile(r"async\s+fn\s+build_binding\s*\([^;{]*?\)\s*(?:->[^;{]+)?\s*\{"),
        "PrefillRouter::build_binding",
    ),
    _SourceScope(
        "kv_router_config_from_lookup",
        RUST_CONFIG,
        re.compile(
            r"(?m)^fn\s+kv_router_config_from_lookup\s*\([^;{]*?\)\s*"
            r"->\s*Result<KvRouterConfig,\s*String>\s*\{"
        ),
        "kv_router_config_from_lookup",
    ),
    _SourceScope(
        "router_settings_apply",
        RUST_POLICY_CONFIG,
        re.compile(
            r"pub\(super\)\s+fn\s+apply\s*\(\s*&self\s*,\s*config\s*:\s*&mut\s+"
            r"super::config::KvRouterConfig\s*\)\s*->\s*Result<\(\),\s*String>\s*\{"
        ),
        "RouterSettings::apply",
    ),
    _SourceScope(
        "reserve_prefill_worker",
        RUST_PREFILL_QUERY,
        re.compile(
            r"pub\s+async\s+fn\s+reserve_prefill_worker\s*\([^;{]*?\)\s*"
            r"->\s*Result<PrefillReservation>\s*\{"
        ),
        "PrefillRouter::reserve_prefill_worker",
    ),
)


def _load_input_startup_scope_cache() -> tuple[dict[str, str], dict[Path, str]]:
    """Read/extract/hash each decisive scope once; fail closed on digests."""
    keys = [scope.key for scope in _INPUT_STARTUP_SCOPES]
    if len(set(keys)) != len(keys):
        raise AssertionError(f"duplicate _INPUT_STARTUP_SCOPES keys: {keys}")
    scope_key_set = set(keys)
    digest_key_set = set(SOURCE_SCOPE_DIGESTS)
    unused = digest_key_set - scope_key_set
    if unused:
        raise AssertionError(f"unused SOURCE_SCOPE_DIGESTS keys: {sorted(unused)}")
    if digest_key_set and digest_key_set != scope_key_set:
        raise AssertionError(
            "SOURCE_SCOPE_DIGESTS key set must equal scope keys; missing "
            f"{sorted(scope_key_set - digest_key_set)}"
        )
    file_text: dict[Path, str] = {}
    cached: dict[str, str] = {}
    for scope in _INPUT_STARTUP_SCOPES:
        if scope.path not in file_text:
            if not scope.path.is_file():
                raise AssertionError(
                    f"missing fingerprint source {scope.label}: {scope.path}"
                )
            file_text[scope.path] = _strip_rust_comments(
                scope.path.read_text(encoding="utf-8")
            )
        blob = _fn_decl_and_body(file_text[scope.path], scope.pattern, scope.label)
        cached[scope.key] = blob
        actual = hashlib.sha256(blob.encode("utf-8")).hexdigest()
        expected = SOURCE_SCOPE_DIGESTS.get(scope.key, "")
        if not re.fullmatch(r"[0-9a-f]{64}", expected or ""):
            raise AssertionError(
                f"SOURCE_SCOPE_DIGESTS[{scope.key!r}] unset; Root must paste "
                f"sha256 of comment-stripped declaration+body for {scope.label}: "
                f"actual={actual}"
            )
        if actual != expected:
            raise AssertionError(
                f"fingerprint mismatch for {scope.key}: scope={scope.label} "
                f"actual={actual} expected={expected}"
            )
    return cached, file_text


def _cached_stripped_file(path: Path, file_text: dict[Path, str]) -> str:
    if path not in file_text:
        if not path.is_file():
            raise AssertionError(f"missing source file: {path}")
        file_text[path] = _strip_rust_comments(path.read_text(encoding="utf-8"))
    return file_text[path]


def prefill_admission_consumer_env(profile_name: str) -> dict[str, str | None]:
    if profile_name not in PREFILL_ADMISSION_PROFILES:
        raise AssertionError(
            f"unknown prefill admission profile {profile_name!r}; "
            f"expected one of {PREFILL_ADMISSION_PROFILES}"
        )
    env: dict[str, str | None] = {
        PREFILL_ABS_ENV: None,
        PREFILL_FRAC_ENV: None,
    }
    if profile_name == "absolute":
        env[PREFILL_ABS_ENV] = PREFILL_ABS_SENTINEL
    elif profile_name == "fractional":
        env[PREFILL_FRAC_ENV] = PREFILL_FRAC_SENTINEL
    return env


def expected_prefill_admission_handles(
    profile_name: str,
) -> tuple[None, int | None, float | None]:
    """Return (decode, abs, frac) expected original FE admin handle values."""
    if profile_name == "default":
        return (None, None, None)
    if profile_name == "absolute":
        return (None, int(PREFILL_ABS_SENTINEL), None)
    if profile_name == "fractional":
        return (None, None, float(PREFILL_FRAC_SENTINEL))
    raise AssertionError(f"unknown prefill admission profile {profile_name!r}")


def clear_ais_identity_envs(monkeypatch) -> None:
    monkeypatch.delenv("DYN_AIS_PERF_CONFIG", raising=False)
    for field in AIS_IDENTITY_ENV_FIELDS:
        monkeypatch.delenv("DYN_AIS_" + field.upper(), raising=False)
        monkeypatch.delenv("DYN_AIC_" + field.upper(), raising=False)


def _require_attr(
    mapping: Mapping[str, Any], name: str, expected: Any, label: str
) -> None:
    if name not in mapping:
        raise AssertionError(f"{label} missing field {name!r}")
    if mapping[name] != expected:
        raise AssertionError(f"{label} {name}={mapping[name]!r} != {expected!r}")


def assert_input_startup_capability_source_contracts() -> None:
    """Named support-boundary checks for the seven input/startup rows.

    Digests and scoped call/field facts protect known factory and env/YAML
    boundaries. This is conservative coverage of those named scopes, not a
    formal proof against arbitrary future Rust.
    """
    assert_required_sources_exist()
    cached, file_text = _load_input_startup_scope_cache()
    none3 = ["None", "None", "None"]

    from_discovery = cached["epp_from_discovery"]
    if "skip_initial_worker_wait = true" not in from_discovery:
        raise AssertionError("EPP from_discovery lost skip_initial_worker_wait=true")
    if "wait_for(|m| !m.is_empty())" not in from_discovery and (
        "wait_for(|m|!m.is_empty())" not in from_discovery
    ):
        raise AssertionError("EPP from_discovery lost nonempty runtime bootstrap wait")
    prefill_args = _require_unique_callee_args(
        "from_discovery",
        from_discovery,
        re.compile(r"PrefillRouter::new\s*\("),
        "PrefillRouter::new call",
        12,
    )
    if prefill_args[5:8] != [
        "None",
        "None",
        "dynamo_llm::session_affinity::SessionAffinityMode::Hard",
    ]:
        raise AssertionError(
            "EPP PrefillRouter::new estimator/TTL/mode "
            f"{prefill_args[5:8]!r} != None/None/Hard"
        )
    managed_epp = _require_unique_callee_args(
        "from_discovery",
        from_discovery,
        re.compile(r"managed_kv_router_for_with_worker_role\s*\("),
        "managed_kv_router_for_with_worker_role call",
        8,
    )
    if managed_epp[3] != "None":
        raise AssertionError(
            f"EPP decode factory estimator arg {managed_epp[3]!r} != None"
        )

    find_args = _require_unique_callee_args(
        "route_decode",
        cached["epp_route_decode"],
        re.compile(r"find_best_match_details_with_policy_class\s*\("),
        "find_best_match_details_with_policy_class call",
        16,
    )
    if find_args[11:14] != none3:
        raise AssertionError(
            "route_decode session/expected/pin "
            f"{find_args[11:14]!r} != None,None,None"
        )

    token_struct = _unique_scoped_rust_body(
        _cached_stripped_file(RUST_EPP, file_text),
        re.compile(r"struct\s+TokenizeResult\s*\{"),
        "EPP TokenizeResult",
    )
    token_fields = tuple(
        re.findall(r"(?m)^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:", token_struct)
    )
    wanted_token_fields = (
        "tokens",
        "cache_namespace",
        "priority_jump",
        "strict_priority",
        "routing_constraints",
        "tokens_safe_to_inject",
    )
    if token_fields != wanted_token_fields:
        raise AssertionError(
            f"TokenizeResult fields {token_fields!r} != {wanted_token_fields!r}"
        )
    if "DYN_SERVE_INDEXER" in from_discovery or "DYN_ROUTER_MODE" in from_discovery:
        raise AssertionError("stock EPP from_discovery gained FE serve/mode env reads")

    role_body = cached["managed_kv_router_for_with_worker_role"]
    role_to_policy = _require_unique_callee_args(
        "managed_role",
        role_body,
        re.compile(r"managed_kv_router_for_with_policy\s*\("),
        "role→policy call",
        9,
    )
    if role_to_policy[4] != "prefill_load_estimator":
        raise AssertionError(
            f"role wrapper estimator forward {role_to_policy[4]!r} "
            "!= prefill_load_estimator"
        )

    policy_body = cached["managed_kv_router_for_with_policy"]
    if "LoadThresholdHandle::new(Default::default())" not in policy_body:
        raise AssertionError(
            "managed_kv_router_for_with_policy lost Default LoadThresholdHandle"
        )
    policy_to_chooser = _require_unique_callee_args(
        "managed_policy",
        policy_body,
        re.compile(r"kv_chooser_for_with_policy_and_client\s*\("),
        "policy→chooser call",
        11,
    )
    if policy_to_chooser[4] != "prefill_load_estimator":
        raise AssertionError(
            f"managed policy estimator forward {policy_to_chooser[4]!r} "
            "!= prefill_load_estimator"
        )

    chooser = cached["kv_chooser_for_with_policy_and_client"]
    chooser_to_new = _require_unique_callee_args(
        "kv_chooser",
        chooser,
        re.compile(r"KvRouter::new_with_worker_role_and_scheduler_load\s*\("),
        "chooser→KvRouter::new call",
        16,
    )
    if chooser_to_new[7] != "prefill_load_estimator":
        raise AssertionError(
            f"chooser estimator forward {chooser_to_new[7]!r} "
            "!= prefill_load_estimator"
        )

    pub_new = cached["prefill_router_new"]
    delegated = _require_unique_callee_args(
        "prefill_router_new",
        pub_new,
        re.compile(r"new_with_selection_policy\s*\("),
        "PrefillRouter::new→new_with_selection_policy",
        14,
    )
    if (
        delegated[6:9]
        != [
            "prefill_load_estimator",
            "session_affinity_ttl_secs",
            "session_affinity_mode",
        ]
        or delegated[11] != "load_thresholds"
    ):
        raise AssertionError(
            "PrefillRouter::new lost estimator/TTL/mode/handle forwards: "
            f"{delegated[6:9]!r}/{delegated[11]!r}"
        )

    selection = cached["prefill_router_new_with_selection_policy"]
    for required in (
        "prefill_load_estimator,",
        "session_affinity_mode,",
        "load_thresholds,",
    ):
        if required not in selection:
            raise AssertionError(
                f"new_with_selection_policy lost install site for {required!r}"
            )

    build_binding = cached["prefill_build_binding"]
    binding_chooser = _require_unique_callee_args(
        "build_binding",
        build_binding,
        re.compile(r"kv_chooser_for_with_policy_and_client\s*\("),
        "build_binding→chooser call",
        11,
    )
    if binding_chooser[4] != "context.prefill_load_estimator.clone()":
        raise AssertionError(
            f"build_binding estimator forward {binding_chooser[4]!r} "
            "!= context.prefill_load_estimator.clone()"
        )
    if "context.session_affinity_mode" not in build_binding:
        raise AssertionError("build_binding lost context.session_affinity_mode forward")

    kv_new = cached["kv_router_new_with_worker_role_and_scheduler_load"]
    if "min_initial_workers_from_env()" not in kv_new:
        raise AssertionError("KvRouter::new lost min_initial_workers_from_env call")
    if (
        "if min_initial_workers > 0 && !kv_router_config.skip_initial_worker_wait {"
        not in kv_new
    ):
        raise AssertionError("KvRouter::new lost min_initial_workers gate condition")
    if "prefill_load_estimator: prefill_load_estimator.clone()" not in kv_new:
        raise AssertionError(
            "KvRouter::new lost EmbeddedSelectionArgs.prefill_load_estimator forward"
        )

    embedded = cached["embedded_selection_start"]
    if "prefill_estimator: args.prefill_load_estimator" not in embedded:
        raise AssertionError(
            "EmbeddedSelection::start HostLoad no longer forwards prefill_estimator"
        )

    min_env = cached["min_initial_workers_from_env"]
    if "DYN_ROUTER_MIN_INITIAL_WORKERS" not in min_env:
        raise AssertionError("min_initial_workers_from_env lost env name")

    lookup = cached["kv_router_config_from_lookup"]
    if 'get_env("DYN_ROUTER_PREFILL_LOAD_MODEL")' not in lookup:
        raise AssertionError("kv_router_config_from_lookup lost AIS env read")
    for banned in (
        "DYN_SERVE_INDEXER",
        "DYN_ROUTER_MODE",
        "DYN_ROUTER_SESSION_AFFINITY_MODE",
    ):
        if banned in lookup:
            raise AssertionError(
                f"kv_router_config_from_lookup gained FE-only env {banned}"
            )
    if "config.apply_policy_config()?" not in lookup:
        raise AssertionError("kv_router_config_from_lookup lost final policy apply")
    config_text = _cached_stripped_file(RUST_CONFIG, file_text)
    try_env = _fn_decl_and_body(
        config_text,
        re.compile(
            r"pub\s+fn\s+try_kv_router_config_from_dynamo_env\s*\(\s*\)\s*"
            r"->\s*Result<KvRouterConfig,\s*String>\s*\{"
        ),
        "try_kv_router_config_from_dynamo_env",
    )
    if "kv_router_config_from_lookup(|key| env::var(key).ok())" not in try_env:
        raise AssertionError(
            "try_kv_router_config_from_dynamo_env lost lookup(|key| env::var) call"
        )
    if "requires router_track_prefill_tokens=true" not in config_text:
        raise AssertionError("AIS validation requiring prefill tracking is gone")

    apply_body = cached["router_settings_apply"]
    if (
        "serve_indexer" not in apply_body
        or "router_prefill_load_model" not in apply_body
    ):
        raise AssertionError("RouterSettings::apply lost serve_indexer/AIS whitelist")
    for banned in ("router_mode", "min_initial_workers", "session_affinity_mode"):
        if banned in apply_body:
            raise AssertionError(
                f"RouterSettings::apply unexpectedly allowlists {banned}"
            )
    if "unknown router setting:" not in apply_body:
        raise AssertionError("RouterSettings::apply lost unknown-field rejection")

    reserve = cached["reserve_prefill_worker"]
    sig = re.search(
        r"fn\s+reserve_prefill_worker\s*\((.*?)\)\s*->\s*Result<PrefillReservation>",
        reserve,
        re.DOTALL,
    )
    if sig is None:
        raise AssertionError("reserve_prefill_worker signature not found")
    reserve_params = _split_rust_call_args(sig.group(1))
    if reserve_params and reserve_params[0] in {"&self", "&mut self", "self"}:
        reserve_params = reserve_params[1:]
    if len(reserve_params) != 10:
        raise AssertionError(
            f"reserve_prefill_worker arity excluding self is {len(reserve_params)} != 10"
        )
    for param in reserve_params:
        name = param.split(":", 1)[0].strip().split()[-1]
        if "session" in name:
            raise AssertionError(
                f"reserve_prefill_worker gained session carrier param {name!r}"
            )
    admitted = _require_unique_callee_args(
        "reserve_prefill_worker",
        reserve,
        re.compile(r"find_best_match_details_with_policy_class_admitted\s*\("),
        "admitted find_best_match call",
        16,
    )
    if admitted[11:14] != none3:
        raise AssertionError(
            "reserve_prefill_worker session/expected/pin "
            f"{admitted[11:14]!r} != None,None,None"
        )

    served = cached["ensure_served_indexer_service"]
    if "bindings.insert(model_name.clone(), indexer)" not in served:
        raise AssertionError("ensure_served_indexer_service lost bindings.insert")
    if "ServedIndexerHandle" not in served:
        raise AssertionError("ensure_served_indexer_service lost ServedIndexerHandle")

    ingress = _fn_decl_and_body(
        _cached_stripped_file(RUST_INDEXER_INGRESS, file_text),
        re.compile(
            r"pub\(crate\)\s+async\s+fn\s+start\s*\(\s*args\s*:\s*"
            r"RuntimeIngressArgs<'_>\s*\)\s*->\s*Result<Arc<Self>>\s*\{"
        ),
        "RuntimeIngress::start",
    )
    if "if kv_router_config.serve_indexer {" not in ingress:
        raise AssertionError("RuntimeIngress::start lost serve_indexer branch")
    if "ensure_served_indexer_service(" not in ingress:
        raise AssertionError("RuntimeIngress::start lost ensure_served_indexer_service")

    prepare = _fn_decl_and_body(
        _cached_stripped_file(RUST_WATCHER, file_text),
        re.compile(
            r"async\s+fn\s+prepare_worker_set\s*\([^;{]*?\)\s*(?:->[^;{]+)?\s*\{"
        ),
        "prepare_worker_set",
    )
    if (
        "LoadThresholdHandle::new(router_config.load_threshold_config.clone())"
        not in prepare
    ):
        raise AssertionError(
            "prepare_worker_set lost original RouterConfig LoadThresholdHandle"
        )

    ais_text = _cached_stripped_file(RUST_AIS_CALLBACK, file_text)
    build_model = _fn_decl_and_body(
        ais_text,
        re.compile(
            r"(?m)^fn\s+build_model\s*\(\s*config\s*:\s*&serde_json::Value\s*\)\s*"
            r"->\s*PyResult<RustAisCallback>\s*\{"
        ),
        "ais_callback::build_model",
    )
    if "ForwardPassPerfModel::best_available(config)" not in build_model:
        raise AssertionError("build_model lost ForwardPassPerfModel::best_available")
    if "readiness != ForwardPassPerfReadiness::Ready" not in build_model:
        raise AssertionError("build_model lost Ready rejection")
    create_est = _fn_decl_and_body(
        ais_text,
        re.compile(
            r"pub\(super\)\s+fn\s+create_ais_prefill_load_estimator\s*\([^;{]*?\)\s*"
            r"(?:->[^;{]+)?\s*\{"
        ),
        "create_ais_prefill_load_estimator",
    )
    if "build_model(config)" not in create_est:
        raise AssertionError(
            "create_ais_prefill_load_estimator lost build_model factory call"
        )
    select_engine = _fn_decl_and_body(
        _cached_stripped_file(RUST_PYO3_ENTRYPOINT, file_text),
        re.compile(r"async\s+fn\s+select_engine\s*\([^;{]*?\)\s*(?:->[^;{]+)?\s*\{"),
        "select_engine",
    )
    if "create_ais_prefill_load_estimator" not in select_engine:
        raise AssertionError("select_engine lost create_ais_prefill_load_estimator")
    if "RsEngineConfig::Dynamic" not in select_engine:
        raise AssertionError("select_engine lost Dynamic AIS estimator supply path")

    for path, label in (
        (RUST_QUEUE, "queue::prefill_load_hint_for"),
        (RUST_KV_ROUTER, "KvRouter::prefill_load_hint_for"),
    ):
        hint = _fn_decl_and_body(
            _cached_stripped_file(path, file_text),
            re.compile(
                r"fn\s+prefill_load_hint_for\s*\([^;{]*?\)\s*(?:->[^;{]+)?\s*\{"
            ),
            label,
        )
        if "None => None," not in hint and "None => None" not in hint:
            raise AssertionError(f"{label} lost absent-estimator => no duration")

    assert_session_inputs_stack_uses_ttl_300()


def run_input_startup_capability_decisions(monkeypatch) -> None:
    """Ordinary FE parser→validate→native controls for the seven decisions."""
    from dynamo.common.configuration.groups.router_args import logger as router_logger
    from dynamo.llm import RouterMode

    assert_input_startup_capability_source_contracts()
    clear_known_manifest_router_envs(monkeypatch)
    clear_ais_identity_envs(monkeypatch)

    omitted = _frontend_router_snapshot()
    _require_attr(omitted.config_attrs, "serve_indexer", False, "default config")
    _require_attr(omitted.kv_router_kwargs, "serve_indexer", False, "default kwargs")
    _require_attr(omitted.kv_json, "serve_indexer", False, "default native json")

    false_snap = _frontend_router_snapshot(("--no-serve-indexer",))
    _require_attr(false_snap.kv_router_kwargs, "serve_indexer", False, "false kwargs")
    _require_attr(false_snap.kv_json, "serve_indexer", False, "false native json")
    true_snap = _frontend_router_snapshot(("--serve-indexer",))
    _require_attr(true_snap.kv_router_kwargs, "serve_indexer", True, "true kwargs")
    _require_attr(true_snap.kv_json, "serve_indexer", True, "true native json")

    min0 = _frontend_router_snapshot(("--router-min-initial-workers", "0"))
    _require_attr(min0.config_attrs, "min_initial_workers", 0, "min0 config")
    min2 = _frontend_router_snapshot(("--router-min-initial-workers", "2"))
    _require_attr(min2.config_attrs, "min_initial_workers", 2, "min2 config")

    kv_mode = _frontend_router_snapshot()
    if kv_mode.router_mode != "kv" or kv_mode.native_router_mode != RouterMode.KV:
        raise AssertionError(
            f"KV mode control failed: {kv_mode.router_mode!r}/{kv_mode.native_router_mode!r}"
        )
    rr_mode = _frontend_router_snapshot(
        require_kv=False, wrapper_cli=NON_KV_WRAPPER_CLI
    )
    if rr_mode.router_mode != "round-robin":
        raise AssertionError(f"round-robin mode got {rr_mode.router_mode!r}")
    if rr_mode.native_router_mode != RouterMode.RoundRobin:
        raise AssertionError(f"native round-robin got {rr_mode.native_router_mode!r}")
    if "session_affinity_mode" not in rr_mode.router_kwargs:
        raise AssertionError("non-KV snapshot lost router_kwargs session_affinity_mode")

    hard = _frontend_router_snapshot(("--router-session-affinity-mode", "hard"))
    _require_attr(hard.config_attrs, "session_affinity_mode", "hard", "hard config")
    _require_attr(hard.router_kwargs, "session_affinity_mode", "hard", "hard kwargs")
    soft = _frontend_router_snapshot(("--router-session-affinity-mode", "soft"))
    _require_attr(soft.config_attrs, "session_affinity_mode", "soft", "soft config")
    _require_attr(soft.router_kwargs, "session_affinity_mode", "soft", "soft kwargs")

    ais_json_arg = json.dumps(dict(AIS_PERF_IDENTITY), separators=(",", ":"))
    ais = _frontend_router_snapshot(
        (
            "--router-prefill-load-model",
            "ais",
            "--router-track-prefill-tokens",
            "--dyn-chat-processor",
            "dynamo",
            "--ais-perf-config",
            ais_json_arg,
        )
    )
    _require_attr(ais.config_attrs, "router_prefill_load_model", "ais", "AIS config")
    _require_attr(
        ais.kv_json,
        "router_prefill_load_model",
        PREFILL_LOAD_MODEL_AIS_WIRE,
        "AIS native json",
    )

    none_model = _frontend_router_snapshot(("--router-prefill-load-model", "none"))
    _require_attr(
        none_model.config_attrs, "router_prefill_load_model", "none", "none config"
    )
    _require_attr(
        none_model.kv_json,
        "router_prefill_load_model",
        PREFILL_LOAD_MODEL_NONE_WIRE,
        "none native json",
    )

    abs_snap, abs_records = _capture_router_args_warnings(
        lambda: _frontend_router_snapshot(
            ("--active-prefill-tokens-threshold", PREFILL_ABS_SENTINEL)
        )
    )
    _require_router_args_warning(
        abs_records,
        logger_name=router_logger.name,
        message=ABSOLUTE_PREFILL_THRESHOLD_WARNING,
    )
    _require_attr(
        abs_snap.config_attrs,
        "active_prefill_tokens_threshold",
        int(PREFILL_ABS_SENTINEL),
        "abs config",
    )
    _require_attr(
        abs_snap.router_kwargs,
        "active_prefill_tokens_threshold",
        int(PREFILL_ABS_SENTINEL),
        "abs kwargs",
    )
    frac_snap, frac_records = _capture_router_args_warnings(
        lambda: _frontend_router_snapshot(
            ("--active-prefill-tokens-threshold-frac", PREFILL_FRAC_SENTINEL)
        )
    )
    if frac_records:
        got = [(record.name, record.getMessage()) for record in frac_records]
        raise AssertionError(f"fractional threshold emitted unexpected warnings {got}")
    _require_attr(
        frac_snap.config_attrs,
        "active_prefill_tokens_threshold_frac",
        float(PREFILL_FRAC_SENTINEL),
        "frac config",
    )
    _require_attr(
        frac_snap.router_kwargs,
        "active_prefill_tokens_threshold_frac",
        float(PREFILL_FRAC_SENTINEL),
        "frac kwargs",
    )
