# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared KV router configuration ArgGroup.

Defines the shared KvRouterConfig parameters once so that both
``dynamo.frontend`` and ``dynamo.router`` can reuse them without duplication.
Field names on ``KvRouterConfigBase`` match the ``KvRouterConfig`` Python
constructor kwargs 1:1, so ``kv_router_kwargs()`` returns a dict that can be
unpacked directly into ``KvRouterConfig(**config.kv_router_kwargs())``.
"""

import argparse
import json
from functools import cached_property
from pathlib import Path
from typing import Any, Optional, get_type_hints

import yaml
from msgspec import ValidationError, convert

from dynamo.common.configuration.arg_group import ArgGroup
from dynamo.common.configuration.config_base import ConfigBase
from dynamo.common.configuration.utils import (
    ArgEnv,
    Deprecated,
    add_argument,
    add_negatable_bool_argument,
    add_removed_flags,
    reject_removed_env_vars,
)

# Shared router infrastructure settings accepted by the YAML `router` section.
_ROUTER_SETTINGS = (
    "host_cache_hit_weight",
    "disk_cache_hit_weight",
    "use_kv_events",
    "router_replica_sync",
    "router_track_active_blocks",
    "router_track_output_blocks",
    "router_assume_kv_reuse",
    "router_track_prefill_tokens",
    "router_tracking_hash",
    "router_tracking_key_file",
    "router_tracking_key_id",
    "router_prefill_load_model",
    "router_ttl_secs",
    "router_approximate_cache_policy",
    "router_event_threads",
    "use_remote_indexer",
    "serve_indexer",
    "enable_session_prefix_index",
    "shared_cache_type",
    "router_predicted_ttl_secs",
)

# Authoritative field list — used by kv_router_kwargs() to extract values.
_KV_ROUTER_FIELDS: tuple[str, ...] = (
    *_ROUTER_SETTINGS,
    "overlap_score_credit",
    "prefill_load_scale",
    "router_temperature",
    "router_policy_config",
    "conditional_disagg_enabled",
    "conditional_disagg_policy",
    "conditional_disagg_eff_isl_threshold",
    "conditional_disagg_eff_isl_ratio_threshold",
    "conditional_disagg_prefill_busy_threshold",
    "conditional_disagg_decode_busy_threshold",
)

CONDITIONAL_DISAGG_POLICY_CHOICES: tuple[str, ...] = (
    "isl_bounding",
    "prefill_load",
    "isl_or_load",
)
LOAD_AWARE_CONDITIONAL_DISAGG_POLICIES: frozenset[str] = frozenset(
    {"prefill_load", "isl_or_load"}
)

_CONDITIONAL_DISAGG_CONFIG_FIELDS: dict[str, str] = {
    "policy": "conditional_disagg_policy",
    "eff_isl_threshold": "conditional_disagg_eff_isl_threshold",
    "eff_isl_ratio_threshold": "conditional_disagg_eff_isl_ratio_threshold",
    "prefill_busy_threshold": "conditional_disagg_prefill_busy_threshold",
    "decode_busy_threshold": "conditional_disagg_decode_busy_threshold",
}

_POLICY_PARAMETERS = "worker_selection.<role>.parameters"
_POLICY_FILE = "--router-policy-config"

# Removed in v1.8: (flags, environment variables, the policy-file key that
# replaces them). A removed flag fails when parsed. A removed variable fails
# the startup of a process that used to read it, whatever its value; one that
# was only ignored would stop applying without a signal.
# TODO(v1.9): Remove this table and the two checks that read it.
_REMOVED: tuple[tuple[tuple[str, ...], tuple[str, ...], str], ...] = (
    (
        ("--load-aware", "--no-load-aware"),
        ("DYN_ROUTER_LOAD_AWARE",),
        f"overlap_score_credit: 0 and shared_cache_multiplier: 0 under "
        f"{_POLICY_PARAMETERS}, with the router tracking settings, in {_POLICY_FILE}",
    ),
    (
        ("--router-kv-overlap-score-credit-decay",),
        ("DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT_DECAY",),
        f"{_POLICY_PARAMETERS}.overlap_score_credit_decay in {_POLICY_FILE}",
    ),
    (
        ("--router-kv-overlap-score-weight", "--kv-overlap-score-weight"),
        ("DYN_ROUTER_KV_OVERLAP_SCORE_WEIGHT", "DYN_OVERLAP_SCORE_WEIGHT"),
        f"{_POLICY_PARAMETERS}.prefill_load_scale in {_POLICY_FILE}",
    ),
    (
        ("--router-decode-active-request-weight",),
        ("DYN_ROUTER_DECODE_ACTIVE_REQUEST_WEIGHT",),
        f"{_POLICY_PARAMETERS}.decode_active_request_weight in {_POLICY_FILE}",
    ),
    (
        ("--router-queue-threshold",),
        ("DYN_ROUTER_QUEUE_THRESHOLD",),
        f"policy_classes[].prefill_busy_threshold_frac in {_POLICY_FILE}, and set "
        "queue_policy there: the removed queue defaulted to fcfs, a class "
        "defaults to edf",
    ),
    (
        ("--router-queue-policy",),
        ("DYN_ROUTER_QUEUE_POLICY",),
        f"policy_classes[].queue_policy in {_POLICY_FILE}",
    ),
    (
        ("--router-prefill-policy",),
        ("DYN_ROUTER_PREFILL_POLICY",),
        f"worker_selection.prefill in {_POLICY_FILE}",
    ),
    (
        ("--router-decode-policy",),
        ("DYN_ROUTER_DECODE_POLICY",),
        f"worker_selection.decode in {_POLICY_FILE}",
    ),
    (
        ("--shared-cache-multiplier",),
        ("DYN_SHARED_CACHE_MULTIPLIER",),
        f"{_POLICY_PARAMETERS}.shared_cache_multiplier in {_POLICY_FILE}",
    ),
)
REMOVED_KV_ROUTER_FLAGS: dict[str, str] = {
    flag: replacement for flags, _, replacement in _REMOVED for flag in flags
}
REMOVED_KV_ROUTER_ENV_VARS: dict[str, str] = {
    name: replacement for _, names, replacement in _REMOVED for name in names
}
# A frontend read it in every router mode (a true value forced KV routing),
# so it fails one in any mode.
_LOAD_AWARE_ENV_VAR = "DYN_ROUTER_LOAD_AWARE"


# These three flags outlive the others deprecated in v1.6: the AISimulate DGD
# output sets them on the frontend it generates, and the planner's live retune
# reads a frontend's startup from them and cannot read a policy file.
def _policy_parameter(parameter: str) -> Deprecated:
    return Deprecated(
        f"{_POLICY_PARAMETERS}.{parameter} in {_POLICY_FILE}",
        remove_in="a later release",
    )


def _parse_conditional_disagg_config(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "--router-conditional-disagg-config must be a JSON object"
        ) from exc
    if not isinstance(parsed, dict):
        raise ValueError("--router-conditional-disagg-config must be a JSON object")

    unknown = sorted(set(parsed) - set(_CONDITIONAL_DISAGG_CONFIG_FIELDS))
    if unknown:
        raise ValueError(
            "--router-conditional-disagg-config has unknown field(s): "
            + ", ".join(unknown)
        )
    if "policy" in parsed and not isinstance(parsed["policy"], str):
        raise ValueError("--router-conditional-disagg-config policy must be a string")
    if "eff_isl_threshold" in parsed and (
        not isinstance(parsed["eff_isl_threshold"], int)
        or isinstance(parsed["eff_isl_threshold"], bool)
    ):
        raise ValueError(
            "--router-conditional-disagg-config eff_isl_threshold must be an integer"
        )
    if "eff_isl_ratio_threshold" in parsed and (
        parsed["eff_isl_ratio_threshold"] is None
        or not isinstance(parsed["eff_isl_ratio_threshold"], (int, float))
        or isinstance(parsed["eff_isl_ratio_threshold"], bool)
    ):
        raise ValueError(
            "--router-conditional-disagg-config eff_isl_ratio_threshold must be a number"
        )
    for field in ("prefill_busy_threshold", "decode_busy_threshold"):
        if parsed.get(field) is None:
            continue
        if not isinstance(parsed[field], (int, float)) or isinstance(
            parsed[field], bool
        ):
            raise ValueError(
                f"--router-conditional-disagg-config {field} must be a number"
            )
    return parsed


def _conditional_disagg_config_arg(value: str) -> dict[str, Any]:
    try:
        return _parse_conditional_disagg_config(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


class KvRouterConfigBase(ConfigBase):
    """Mixin carrying the shared KvRouterConfig fields."""

    overlap_score_credit: float
    prefill_load_scale: float
    host_cache_hit_weight: float
    disk_cache_hit_weight: float
    router_temperature: float
    use_kv_events: bool
    router_replica_sync: bool
    router_track_active_blocks: bool
    router_track_output_blocks: bool
    router_assume_kv_reuse: bool
    router_track_prefill_tokens: bool
    router_tracking_hash: str = "public-xxh3-v1"
    router_tracking_key_file: Optional[str] = None
    router_tracking_key_id: Optional[str] = None
    router_prefill_load_model: str
    router_ttl_secs: float
    router_approximate_cache_policy: str = "ttl"
    router_policy_config: Optional[str] = None
    router_event_threads: int
    use_remote_indexer: bool = False
    serve_indexer: bool = False
    enable_session_prefix_index: bool = False
    shared_cache_type: str = "none"
    conditional_disagg_enabled: bool = False
    conditional_disagg_config: Optional[dict[str, Any]] = None
    conditional_disagg_policy: str = "isl_bounding"
    conditional_disagg_eff_isl_threshold: int = 2048
    conditional_disagg_eff_isl_ratio_threshold: float = 0.7
    conditional_disagg_prefill_busy_threshold: Optional[float] = None
    conditional_disagg_decode_busy_threshold: Optional[float] = None
    router_predicted_ttl_secs: Optional[float] = None

    @cached_property
    def _router_settings(self) -> dict[str, Any]:
        if self.router_policy_config is None:
            return {}
        with Path(self.router_policy_config).open() as source:
            document = yaml.safe_load(source)
        if not isinstance(document, dict):
            raise ValueError("--router-policy-config must contain a YAML mapping")
        settings = document.get("router")
        if settings is None:
            return {}
        if not isinstance(settings, dict):
            raise ValueError("router must contain a YAML mapping")
        unknown = settings.keys() - set(_ROUTER_SETTINGS)
        if unknown:
            raise ValueError(f"unknown router setting(s): {sorted(unknown)}")
        types = get_type_hints(KvRouterConfigBase)
        try:
            return {
                name: convert(value, type=types[name])
                for name, value in settings.items()
            }
        except ValidationError as exc:
            raise ValueError(f"invalid router settings: {exc}") from exc

    def hint_policy_roles(self) -> list[str]:
        """Roles whose worker_selection entry lists hints in --router-policy-config.

        A role whose entry lists ``hints: []`` opts out of the defaults and runs none.
        """
        if self.router_policy_config is None:
            return []
        with Path(self.router_policy_config).open() as source:
            document = yaml.safe_load(source)
        if not isinstance(document, dict):
            raise ValueError("--router-policy-config must contain a YAML mapping")
        worker_selection = document.get("worker_selection")
        if worker_selection is None:
            return []
        if not isinstance(worker_selection, dict):
            raise ValueError("worker_selection must contain a YAML mapping")
        return [
            role
            for role, entry in worker_selection.items()
            if isinstance(entry, dict) and entry.get("hints")
        ]

    def apply_router_config(self, *, kv_routing: bool = True) -> None:
        """Apply the settings that depend on more than one input.

        ``kv_routing`` is False on a frontend in another router mode. It never
        read the removed scoring, selection and queue variables, so only
        DYN_ROUTER_LOAD_AWARE fails it.
        """
        removed = REMOVED_KV_ROUTER_ENV_VARS
        if not kv_routing:
            removed = {_LOAD_AWARE_ENV_VAR: removed[_LOAD_AWARE_ENV_VAR]}
        reject_removed_env_vars(removed, removed_in="v1.8")
        self.apply_conditional_disagg_config()
        # Explicit YAML settings win over the flags.
        for name, value in self._router_settings.items():
            setattr(self, name, value)

    def apply_conditional_disagg_config(self) -> None:
        if (
            self.conditional_disagg_config is not None
            and not self.conditional_disagg_enabled
        ):
            raise ValueError(
                "--router-conditional-disagg-config requires --router-conditional-disagg"
            )

        if self.conditional_disagg_config is not None:
            for key, value in self.conditional_disagg_config.items():
                setattr(self, _CONDITIONAL_DISAGG_CONFIG_FIELDS[key], value)

        if not self.conditional_disagg_enabled:
            return
        if self.conditional_disagg_policy not in LOAD_AWARE_CONDITIONAL_DISAGG_POLICIES:
            return
        # Load-aware conditional-disagg policies need a prefill busy threshold.
        if self.conditional_disagg_prefill_busy_threshold is None:
            raise ValueError(
                f"conditional_disagg policy={self.conditional_disagg_policy!r} "
                "needs prefill_busy_threshold; set it in "
                "--router-conditional-disagg-config"
            )

    def kv_router_kwargs(self) -> dict:
        """Return a dict suitable for ``KvRouterConfig(**kwargs)``."""
        self.apply_router_config()
        return {f: getattr(self, f) for f in _KV_ROUTER_FIELDS}


class KvRouterArgGroup(ArgGroup):
    """CLI arguments for the shared KvRouterConfig parameters.

    Args:
        env: Another process's env (see ``ArgEnv``).
    """

    def __init__(self, *, env: Optional[ArgEnv] = None) -> None:
        self.env = env

    def add_arguments(self, parser) -> None:
        g = parser.add_argument_group("KV Router Options")
        add_removed_flags(g, REMOVED_KV_ROUTER_FLAGS, removed_in="v1.8")

        add_argument(
            g,
            flag_name="--router-kv-overlap-score-credit",
            deprecated=_policy_parameter("overlap_score_credit"),
            env_var="DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT",
            default=1.0,
            help=(
                "KV Router: Credit multiplier for device-local prefix overlap. "
                "Must be finite and non-negative; values above 1.0 give device "
                "overlap extra credit, with adjusted prefill cost clamped at zero."
            ),
            arg_type=float,
            dest="overlap_score_credit",
            env=self.env,
        )
        add_argument(
            g,
            flag_name="--router-prefill-load-scale",
            deprecated=_policy_parameter("prefill_load_scale"),
            env_var="DYN_ROUTER_PREFILL_LOAD_SCALE",
            default=1.0,
            help=(
                "KV Router: Scale applied to adjusted prompt-side prefill load after "
                "overlap and lower-tier cache-hit credits are subtracted."
            ),
            arg_type=float,
            dest="prefill_load_scale",
            env=self.env,
        )
        add_argument(
            g,
            flag_name="--router-host-cache-hit-weight",
            env_var="DYN_ROUTER_HOST_CACHE_HIT_WEIGHT",
            default=0.75,
            help=(
                "KV Router: Credit multiplier for host-pinned (CPU offload) prefix overlap. "
                "Range: 0.0 to 1.0; higher values more strongly prefer workers holding the "
                "prefix in CPU-tier KV cache. Symmetric to --router-kv-overlap-score-credit "
                "but applied to host_pinned tier overlap."
            ),
            arg_type=float,
            dest="host_cache_hit_weight",
            env=self.env,
        )
        add_argument(
            g,
            flag_name="--router-disk-cache-hit-weight",
            env_var="DYN_ROUTER_DISK_CACHE_HIT_WEIGHT",
            default=0.25,
            help=(
                "KV Router: Credit multiplier for disk/lower-tier (e.g. NVMe-backed) prefix overlap. "
                "Range: 0.0 to 1.0. Same semantics as --router-host-cache-hit-weight applied to "
                "the disk tier."
            ),
            arg_type=float,
            dest="disk_cache_hit_weight",
            env=self.env,
        )
        add_argument(
            g,
            flag_name="--router-temperature",
            deprecated=_policy_parameter("router_temperature"),
            env_var="DYN_ROUTER_TEMPERATURE",
            default=0.0,
            help=(
                "KV Router: Temperature for normalized worker sampling via softmax. "
                "Higher values promote more randomness, and 0 falls back to deterministic."
            ),
            arg_type=float,
            env=self.env,
        )
        add_negatable_bool_argument(
            g,
            flag_name="--router-kv-events",
            env_var="DYN_ROUTER_USE_KV_EVENTS",
            default=True,
            help=(
                "KV Router: Enable/disable KV events. Use --router-kv-events to enable "
                "(default, router receives cache state events from workers) or --no-router-kv-events "
                "to disable (router predicts cache state based on routing decisions)."
            ),
            dest="use_kv_events",
            obsolete_flag="--kv-events",
            env=self.env,
        )
        add_negatable_bool_argument(
            g,
            flag_name="--router-replica-sync",
            env_var="DYN_ROUTER_REPLICA_SYNC",
            default=False,
            help=(
                "KV Router: Enable best-effort active-sequence synchronization through "
                "the Runtime event plane."
            ),
            env=self.env,
        )
        add_negatable_bool_argument(
            g,
            flag_name="--router-track-active-blocks",
            env_var="DYN_ROUTER_TRACK_ACTIVE_BLOCKS",
            default=True,
            dest="router_track_active_blocks",
            help=(
                "KV Router: Track active blocks (blocks being used for ongoing generation). "
                "By default, active blocks are tracked for load balancing."
            ),
            obsolete_flag="--track-active-blocks",
            env=self.env,
        )
        add_negatable_bool_argument(
            g,
            flag_name="--router-track-output-blocks",
            env_var="DYN_ROUTER_TRACK_OUTPUT_BLOCKS",
            default=False,
            dest="router_track_output_blocks",
            help=(
                "KV Router: Track output blocks during generation. When enabled, the router adds "
                "placeholder blocks as tokens are generated. With expected output sequence length, "
                "fractional decay applies to output blocks and the structurally exclusive prompt "
                "suffix; shared prompt blocks retain full weight."
            ),
            obsolete_flag="--track-output-blocks",
            env=self.env,
        )
        add_negatable_bool_argument(
            g,
            flag_name="--router-assume-kv-reuse",
            env_var="DYN_ROUTER_ASSUME_KV_REUSE",
            default=True,
            dest="router_assume_kv_reuse",
            help=(
                "KV Router: When tracking active blocks, assume KV cache reuse. "
                "Use --no-router-assume-kv-reuse to generate random hashes instead "
                "(when KV cache reuse is not expected)."
            ),
            obsolete_flag="--assume-kv-reuse",
            env=self.env,
        )
        add_negatable_bool_argument(
            g,
            flag_name="--router-track-prefill-tokens",
            env_var="DYN_ROUTER_TRACK_PREFILL_TOKENS",
            default=True,
            dest="router_track_prefill_tokens",
            help=(
                "KV Router: Include prompt-side prefill tokens in active load accounting. "
                "Use --no-router-track-prefill-tokens to ignore prompt tokens in router "
                "prefill-token load, queue pressure, and active_prefill_tokens metrics."
            ),
            env=self.env,
        )
        add_argument(
            g,
            flag_name="--router-tracking-hash",
            env_var="DYN_ROUTER_TRACKING_HASH",
            default="public-xxh3-v1",
            choices=["public-xxh3-v1", "keyed-xxh3-v1"],
            help="KV Router: Hash function for router-derived active-sequence identities.",
            env=self.env,
        )
        add_argument(
            g,
            flag_name="--router-tracking-key-file",
            env_var="DYN_ROUTER_TRACKING_KEY_FILE",
            default=None,
            help="KV Router: File containing the 32-byte provider tracking key.",
            env=self.env,
        )
        add_argument(
            g,
            flag_name="--router-tracking-key-id",
            env_var="DYN_ROUTER_TRACKING_KEY_ID",
            default=None,
            help="KV Router: Provider-managed tracking-key epoch identifier.",
            env=self.env,
        )
        add_argument(
            g,
            flag_name="--router-prefill-load-model",
            env_var="DYN_ROUTER_PREFILL_LOAD_MODEL",
            default="none",
            choices=["none", "ais"],
            arg_type=lambda value: "ais" if value == "aic" else value,
            help=(
                "[EXPERIMENTAL] KV Router: Prompt-side prefill load model. "
                "'none' keeps static prompt load accounting. "
                "'ais' decays the oldest active prefill request using AISimulate-predicted duration."
            ),
            env=self.env,
        )
        add_argument(
            g,
            flag_name="--router-ttl-secs",
            env_var="DYN_ROUTER_TTL_SECS",
            default=120.0,
            help=(
                "KV Router: Time-to-live in seconds for blocks when KV events are disabled. "
                "Only used when --no-router-kv-events is set."
            ),
            arg_type=float,
            env=self.env,
        )
        add_argument(
            g,
            flag_name="--router-approximate-cache-policy",
            env_var="DYN_ROUTER_APPROXIMATE_CACHE_POLICY",
            default="ttl",
            choices=["ttl", "lru"],
            help=(
                "[EXPERIMENTAL] KV Router: Retention policy for the local primary "
                "approximate indexer used with --no-router-kv-events. 'ttl' keeps "
                "the existing time-based behavior; 'lru' uses each worker rank's "
                "advertised KV capacity. Side indexers remain TTL-only."
            ),
            env=self.env,
        )
        add_argument(
            g,
            flag_name="--router-policy-config",
            env_var="DYN_ROUTER_POLICY_CONFIG",
            default=None,
            help=(
                "KV Router: Startup-only YAML configuration for shared router settings, "
                "policy-class queues, per-role worker-selection policies, and hint policies. "
                "On the frontend it needs no --router-mode: the file selects each "
                "role's policy, and any --router-mode other than kv is an error. "
                "When omitted, router queueing is disabled."
            ),
            arg_type=str,
            env=self.env,
        )
        add_negatable_bool_argument(
            g,
            flag_name="--router-conditional-disagg",
            env_var="DYN_ROUTER_CONDITIONAL_DISAGG",
            default=False,
            help=(
                "[EXPERIMENTAL] KV Router: Enable conditional disaggregation. "
                "When enabled, the router may bypass the remote prefill step "
                "and route requests directly to a decode worker. The configured "
                "policy determines whether local prefill and decode is preferable "
                "for each request."
            ),
            dest="conditional_disagg_enabled",
            env=self.env,
        )
        add_argument(
            g,
            flag_name="--router-conditional-disagg-config",
            env_var="DYN_ROUTER_CONDITIONAL_DISAGG_CONFIG",
            default=None,
            help=(
                "[EXPERIMENTAL] KV Router: JSON object for conditional "
                "disaggregation policy settings. Supported fields: policy, "
                "eff_isl_threshold, eff_isl_ratio_threshold, "
                "prefill_busy_threshold, decode_busy_threshold."
            ),
            arg_type=_conditional_disagg_config_arg,
            dest="conditional_disagg_config",
            metavar="JSON",
            env=self.env,
        )
        add_argument(
            g,
            flag_name="--router-event-threads",
            env_var="DYN_ROUTER_EVENT_THREADS",
            default=4,
            help=(
                "KV Router: Number of KV indexer worker threads. When > 1, uses a concurrent "
                "radix tree with a thread pool for higher throughput, including "
                "approximate routing when --no-router-kv-events is set."
            ),
            arg_type=int,
            env=self.env,
        )
        add_negatable_bool_argument(
            g,
            flag_name="--use-remote-indexer",
            env_var="DYN_USE_REMOTE_INDEXER",
            default=False,
            help=(
                "[EXPERIMENTAL] KV Router: Query a remote KV indexer served from the worker "
                "component via the request plane instead of maintaining a local radix tree."
            ),
            dest="use_remote_indexer",
            env=self.env,
        )
        add_negatable_bool_argument(
            g,
            flag_name="--enable-session-prefix-index",
            env_var="DYN_ENABLE_SESSION_PREFIX_INDEX",
            default=False,
            help=(
                "[EXPERIMENTAL] KV Router: Track per-session block lineage in a logical "
                "prefix index that outlives engine cache eviction. Nothing consumes the "
                "index for routing decisions yet."
            ),
            dest="enable_session_prefix_index",
            env=self.env,
        )
        add_argument(
            g,
            flag_name="--shared-cache-type",
            env_var="DYN_SHARED_CACHE_TYPE",
            default="none",
            help=(
                "[EXPERIMENTAL] KV Router: Type of external shared KV cache to query. "
                "'none' (default): disabled. "
                "'hicache': query Mooncake master directly for SGLang L3 (HiCache) state "
                "using SGLang-compatible Mooncake key derivation."
            ),
            arg_type=str,
            choices=["none", "hicache"],
            env=self.env,
        )
        add_argument(
            g,
            flag_name="--router-predicted-ttl-secs",
            env_var="DYN_ROUTER_PREDICTED_TTL_SECS",
            default=None,
            help=(
                "KV Router: Enable predict-on-route with this TTL in seconds for entries "
                "in the local side indexer. Requires KV events; omit to disable. "
                "Independent of --router-ttl-secs, which covers pure approximate mode."
            ),
            arg_type=float,
            env=self.env,
        )
