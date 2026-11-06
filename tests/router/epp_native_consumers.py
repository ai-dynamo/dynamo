# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared live native-consumer profiles for FE/EPP active-sequence env parity.

Owns the four native receiver profiles and the real peer-booking / CLOCK /
construction-event scenario. Stack lifecycle and inventory binding stay in a
later common/wrapper integration turn. Optional EPP and dynamo imports stay
function-local so ordinary collection does not require grpc_tools.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
import uuid
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from tests.router.epp_native_process import (
    NativeRouterFailure,
    NativeRouterOwner,
    _constraint_payload,
)

logger = logging.getLogger(__name__)

NATIVE_CONSUMER_ROWS: tuple[str, ...] = (
    "active_request_expiry_secs",
    "active_sequence_direct_zmq",
    "active_sequence_zmq_rcvhwm",
    "zmq_endpoints_per_sub",
)
NATIVE_CONSUMER_OBSERVER = "native_live_consumer_runner"

EXPIRY_ENV = "DYN_ROUTER_ACTIVE_REQUEST_EXPIRY_SECS"
DIRECT_ENV = "DYN_ROUTER_ACTIVE_SEQUENCE_DIRECT_ZMQ"
HWM_ENV = "DYN_ROUTER_ACTIVE_SEQUENCE_ZMQ_RCVHWM"
POOL_ENV = "DYN_ROUTER_ZMQ_ENDPOINTS_PER_SUB"
SERVING_LOAD_ENV = "DYN_SERVING_LOAD_PUBLISH"
BROKER_URL_ENV = "DYN_ZMQ_BROKER_URL"
BROKER_ENABLED_ENV = "DYN_ZMQ_BROKER_ENABLED"

PROVIDER_EXPIRY_SECS = 300
BACKLOG_TOKENS = 8192
PEER_BACKLOG_TOKENS = 16384
PREFERRED_WEIGHT = 0.85
CREDIT_ZERO = 0.0
CREDIT_FULL = 1.0
CLOCK_OBSERVE_S = 6.5
POLL_S = 0.05
READY_BUDGET_S = 45.0
HTTP_CALL_CAP_S = 2.0
ROUTE_CALL_CAP_S = 8.0
ACTIVE_SEQUENCES_TOPIC = "active_sequences_events"
GENERATE_ENDPOINT = "generate"
# Booking synthetics use base 1_000_000 + salt*10_000 for salt 0..1019.
# Accounting probe starts at 100_000_000 so ranges never cross (warm salt 600
# would otherwise land at 7_000_000).
ACCOUNTING_TOKEN_BASE = 100_000_000
SYNTHETIC_TOKEN_BASE = 1_000_000
SYNTHETIC_SALT_STRIDE = 10_000
SYNTHETIC_SALT_MAX = 1019
U32_EXCLUSIVE_END = 1 << 32

DEDICATED_EVENT = "ZMQ single-consumer stream connected"
GROUPED_EVENT = "Dynamic ZMQ SUB socket connected"
RUNTIME_EVENT = "Started consuming ZMQ endpoint stream"

PEER_CONTAINER_NAME = "kv-indexer-peer"
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
# Supports unquoted tokens and double-quoted values from tracing fields.
STRUCTURED_FIELD_RE = re.compile(
    r"""([A-Za-z_][A-Za-z0-9_]*)=(?:"((?:\\.|[^"\\])*)"|(\S+))"""
)


class WorkerSelectionPending(Exception):
    """Both arms observed, but the expected worker has not converged yet."""


@dataclass(frozen=True)
class NativeConsumerProfile:
    """One fixed receiver overlay for the shared native consumer scenario."""

    name: str
    expiry_secs: int | None
    direct_zmq: bool | None
    rcvhwm: int | None
    endpoints_per_sub: int | None


NATIVE_CONSUMER_PROFILES: Mapping[str, NativeConsumerProfile] = MappingProxyType(
    {
        "omitted": NativeConsumerProfile(
            name="omitted",
            expiry_secs=None,
            direct_zmq=None,
            rcvhwm=None,
            endpoints_per_sub=None,
        ),
        "legacy_runtime": NativeConsumerProfile(
            name="legacy_runtime",
            expiry_secs=300,
            direct_zmq=False,
            rcvhwm=37,
            endpoints_per_sub=1,
        ),
        "grouped": NativeConsumerProfile(
            name="grouped",
            expiry_secs=300,
            direct_zmq=True,
            rcvhwm=37,
            endpoints_per_sub=2,
        ),
        "expiry": NativeConsumerProfile(
            name="expiry",
            expiry_secs=3,
            direct_zmq=True,
            rcvhwm=37,
            endpoints_per_sub=2,
        ),
    }
)


def native_provider_env_overrides() -> dict[str, str | None]:
    """Hosting provider overlay: credit 1 (CACHE inputs) + expiry 300 + replica sync.

    Distinct from load-only peer/FE/EPP overlays (credit 0). serve_indexer stays
    true on the parity stack constructor; credit 1 declares WorkerInputs::CACHE.
    """
    overlay = _cleared_native_transport_env()
    overlay[EXPIRY_ENV] = str(PROVIDER_EXPIRY_SECS)
    overlay[SERVING_LOAD_ENV] = "false"
    overlay["DYN_ROUTER_REPLICA_SYNC"] = "true"
    overlay["DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT"] = "1"
    overlay["DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT_DECAY"] = "0"
    overlay["DYN_ROUTER_TEMPERATURE"] = "0"
    overlay["DYN_ROUTER_PREFILL_LOAD_SCALE"] = "1"
    overlay["DYN_ROUTER_TRACK_PREFILL_TOKENS"] = "true"
    overlay["DYN_ROUTER_TRACK_OUTPUT_BLOCKS"] = "false"
    overlay["DYN_ROUTER_PREFILL_LOAD_MODEL"] = "none"
    overlay["DYN_EVENT_PLANE"] = "zmq"
    return overlay


def native_provider_config_overrides() -> dict[str, Any]:
    """KvRouterConfig kwargs for the hosting provider only (credit 1 / CACHE)."""
    return {
        "router_replica_sync": True,
        "router_prefill_load_model": "none",
        "router_track_prefill_tokens": True,
        "router_track_output_blocks": False,
        "overlap_score_credit": CREDIT_FULL,
        "overlap_score_credit_decay": 0.0,
        "router_temperature": 0.0,
        "prefill_load_scale": 1.0,
    }


def native_peer_env_overrides() -> dict[str, str | None]:
    """Load-only peer sender overlay: credit 0, serve false via config, expiry 300."""
    overlay = _cleared_native_transport_env()
    overlay[EXPIRY_ENV] = str(PROVIDER_EXPIRY_SECS)
    overlay[SERVING_LOAD_ENV] = "false"
    overlay["DYN_ROUTER_REPLICA_SYNC"] = "true"
    overlay["DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT"] = "0"
    overlay["DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT_DECAY"] = "0"
    overlay["DYN_ROUTER_TEMPERATURE"] = "0"
    overlay["DYN_ROUTER_PREFILL_LOAD_SCALE"] = "1"
    overlay["DYN_ROUTER_TRACK_PREFILL_TOKENS"] = "true"
    overlay["DYN_ROUTER_TRACK_OUTPUT_BLOCKS"] = "false"
    overlay["DYN_ROUTER_PREFILL_LOAD_MODEL"] = "none"
    overlay["DYN_EVENT_PLANE"] = "zmq"
    return overlay


def native_peer_config_overrides() -> dict[str, Any]:
    """KvRouterConfig kwargs for the load-only peer (credit 0; not hosting)."""
    return {
        "router_replica_sync": True,
        "router_prefill_load_model": "none",
        "router_track_prefill_tokens": True,
        "router_track_output_blocks": False,
        "overlap_score_credit": CREDIT_ZERO,
        "overlap_score_credit_decay": 0.0,
        "router_temperature": 0.0,
        "prefill_load_scale": 1.0,
    }


def consumer_env_for_profile(profile_name: str) -> dict[str, str | None]:
    """FE/EPP child overlay for one profile. Does not touch the provider."""
    return _receiver_env(_require_profile(profile_name))


def run_native_consumer_profile(
    stack,
    *,
    profile_name: str,
    tmp_path: Path,
) -> dict[str, Any]:
    """Drive one shared live native-consumer profile on an entered routing stack.

    Requires a topology-enabled stack constructed with
    ``provider_env_overrides=native_provider_env_overrides()`` and
    ``provider_config_overrides=native_provider_config_overrides()``, plus the
    agreed ``open_consumer_pair`` / ``provider_free`` / ``epp_process`` contract.
    Query arms poll readiness; FE generation is used only at decisive states.
    """
    from dynamo.llm import RoutingConstraints
    from tests.router.epp_parity import ZONE_A_TAINT, ZONE_B_TAINT

    profile = _require_profile(profile_name)
    _require_hosting_provider_native_policy(stack)
    worker_a, worker_b = _require_zone_workers(stack)
    block_size = _require_positive_block_size(stack)
    cold = _cold_case(stack.model_name)
    prefer_a = RoutingConstraints(preferred_taints={ZONE_A_TAINT: PREFERRED_WEIGHT})
    # Native preferred_taints are f32; the round-tripped value drives multiplier/margins.
    native_weight = prefer_a.preferred_taints.get(ZONE_A_TAINT)
    if native_weight is None or abs(float(native_weight) - PREFERRED_WEIGHT) > 1e-5:
        raise AssertionError(
            f"{profile.name}: native preferred_taints weight is not near "
            f"{PREFERRED_WEIGHT}: {prefer_a.preferred_taints}"
        )
    native_weight_f32 = float(native_weight)
    preferred_multiplier = math.exp(-math.tanh(native_weight_f32))
    taints_a = set(stack.worker_configs[worker_a]["taints"])
    taints_b = set(stack.worker_configs[worker_b]["taints"])
    if ZONE_A_TAINT not in taints_a:
        raise AssertionError(
            f"{profile.name}: worker A missing {ZONE_A_TAINT}: {taints_a}"
        )
    if ZONE_A_TAINT in taints_b:
        raise AssertionError(
            f"{profile.name}: worker B unexpectedly carries {ZONE_A_TAINT}: {taints_b}"
        )
    mult_a = _preferred_multiplier_for_worker(prefer_a.preferred_taints, taints_a)
    mult_b = _preferred_multiplier_for_worker(prefer_a.preferred_taints, taints_b)
    if mult_a is None or abs(float(mult_a) - preferred_multiplier) > 1e-15:
        raise AssertionError(
            f"{profile.name}: preferred multiplier for A is {mult_a!r}, "
            f"expected {preferred_multiplier} from native weight {native_weight_f32}"
        )
    # Nonempty preferred map with no matching taint: Rust returns Some(1.0).
    if mult_b != 1.0:
        raise AssertionError(
            f"{profile.name}: preferred multiplier for B must be 1.0 "
            f"(nonempty map, no matching taints), got {mult_b!r}"
        )

    evidence: dict[str, Any] = {
        "profile": profile.name,
        "rows": NATIVE_CONSUMER_ROWS,
        "observer": NATIVE_CONSUMER_OBSERVER,
        "worker_a": worker_a,
        "worker_b": worker_b,
        "block_size": block_size,
        "native_preferred_weight": native_weight_f32,
        "preferred_multiplier": preferred_multiplier,
        "hosting_provider_credit": CREDIT_FULL,
        "load_only_receiver_peer_credit": CREDIT_ZERO,
        "query_vs_generation": (
            "FE query + EPP query for readiness/poll/hold; "
            "FE generation only at decisive baseline/loaded/restored states; "
            "EPP route is query-only (not backend generation)"
        ),
        "accounting_note": (
            "Owner-local/primary-mirror PotentialLoad reads use a synthetic "
            "cold accounting probe of equal length to the FE tokenizer probe; "
            "FE/EPP route payloads keep the real tokenizer vector. Provider "
            "hosting credit1 is not the receiver load-only scoring policy."
        ),
    }

    owned_bookings: list[dict[str, Any]] = []
    peer_stack = ExitStack()
    peer_holder: dict[str, Any] | None = None
    body_error: BaseException | None = None
    try:
        if profile.name == "expiry":
            dual, peer_holder = _run_dual_publisher_on_stable_receivers(
                stack,
                profile=profile,
                cold=cold,
                worker_a=worker_a,
                worker_b=worker_b,
                block_size=block_size,
                preferred_multiplier=preferred_multiplier,
                owned_bookings=owned_bookings,
                peer_stack=peer_stack,
                tmp_path=tmp_path,
                zone_a=ZONE_A_TAINT,
                zone_b=ZONE_B_TAINT,
            )
            evidence["dual_publisher_on_300"] = dual
            _close_peer(peer_stack, peer_holder)
            peer_holder = None
            peer_stack = ExitStack()
            evidence["expiry_clock"] = _run_expiry_clock_on_fresh_pair(
                stack,
                profile=profile,
                cold=cold,
                worker_a=worker_a,
                worker_b=worker_b,
                block_size=block_size,
                preferred_multiplier=preferred_multiplier,
                owned_bookings=owned_bookings,
                zone_a=ZONE_A_TAINT,
            )
        elif profile.endpoints_per_sub == 2:
            grouped, peer_holder = _run_grouped_profile(
                stack,
                profile=profile,
                cold=cold,
                worker_a=worker_a,
                worker_b=worker_b,
                block_size=block_size,
                preferred_multiplier=preferred_multiplier,
                owned_bookings=owned_bookings,
                peer_stack=peer_stack,
                tmp_path=tmp_path,
                zone_a=ZONE_A_TAINT,
                zone_b=ZONE_B_TAINT,
            )
            evidence["grouped"] = grouped
        else:
            evidence["stable"] = _run_stable_default_profile(
                stack,
                profile=profile,
                cold=cold,
                worker_a=worker_a,
                worker_b=worker_b,
                block_size=block_size,
                preferred_multiplier=preferred_multiplier,
                owned_bookings=owned_bookings,
                zone_a=ZONE_A_TAINT,
            )
        logger.info(
            "native consumer profile=%s evidence=%s",
            profile.name,
            json.dumps(evidence, sort_keys=True),
        )
        return evidence
    except BaseException as exc:
        body_error = exc
        raise
    finally:
        cleanup_errors: list[BaseException] = []
        try:
            _free_owned_bookings(stack, owned_bookings, timeout_s=HTTP_CALL_CAP_S)
        except BaseException as exc:  # noqa: BLE001 - collect all cleanup faults
            cleanup_errors.append(exc)
        try:
            stack.close_consumer_pair()
        except BaseException as exc:  # noqa: BLE001
            cleanup_errors.append(exc)
        try:
            _close_peer(peer_stack, peer_holder)
        except BaseException as exc:  # noqa: BLE001
            cleanup_errors.append(exc)
        if cleanup_errors:
            detail = "; ".join(f"{type(err).__name__}: {err}" for err in cleanup_errors)
            cleanup_error = RuntimeError(f"native consumer cleanup failed: {detail}")
            if body_error is not None:
                raise cleanup_error from body_error
            if len(cleanup_errors) == 1:
                raise cleanup_errors[0]
            raise BaseExceptionGroup("native consumer cleanup failures", cleanup_errors)


def _run_stable_default_profile(
    stack,
    *,
    profile: NativeConsumerProfile,
    cold: Mapping[str, Any],
    worker_a: int,
    worker_b: int,
    block_size: int,
    preferred_multiplier: float,
    owned_bookings: list[dict[str, Any]],
    zone_a: str,
) -> dict[str, Any]:
    from tests.router.helper import unique_request_id

    out: dict[str, Any] = {}
    stack.open_consumer_pair(
        include_epp=True,
        frontend_env_overrides=_receiver_env(profile),
        epp_env_overrides=_receiver_env(profile),
        epp_require_readiness=True,
    )
    deadline = time.monotonic() + READY_BUDGET_S
    probe_tokens = _tokenize(stack, cold, profile.name, deadline)
    out["probe_token_count"] = len(probe_tokens)
    accounting_tokens = _accounting_probe_tokens(len(probe_tokens))
    out["accounting_probe_n"] = len(accounting_tokens)
    out["accounting_cold_proof"] = _prove_accounting_probe_cold(
        stack,
        accounting_tokens,
        worker_a=worker_a,
        worker_b=worker_b,
        case_name=f"{profile.name}-accounting-cold",
        deadline=deadline,
    )
    out["warm"] = _warm_handshake(
        stack,
        cold=cold,
        probe_tokens=probe_tokens,
        accounting_tokens=accounting_tokens,
        worker_a=worker_a,
        worker_b=worker_b,
        block_size=block_size,
        preferred_multiplier=preferred_multiplier,
        owned_bookings=owned_bookings,
        sender="provider",
        book_worker=worker_a,
        required_taint=zone_a,
        case_name=f"{profile.name}-warm-primary",
        deadline=deadline,
    )
    baseline = _decisive_generation_state(
        stack,
        cold,
        expected_worker=worker_a,
        case_name=f"{profile.name}-baseline",
        shared_tokens=probe_tokens,
        deadline=deadline,
    )
    out["baseline"] = baseline
    stale_id = unique_request_id(f"native-stale-{profile.name}")
    admit = _admit_booking(
        stack,
        owner="provider",
        request_id=stale_id,
        tokens=_synthetic_tokens(BACKLOG_TOKENS, salt=2),
        required_taint=zone_a,
        expected_worker=worker_a,
        owned_bookings=owned_bookings,
        case_name=f"{profile.name}-stale",
        deadline=deadline,
    )
    out["admit"] = admit
    loads = _require_sender_booking_loads(
        stack,
        accounting_tokens,
        owner="provider",
        worker_a=worker_a,
        worker_b=worker_b,
        backlog_tokens=BACKLOG_TOKENS,
        booked_worker=worker_a,
        case_name=f"{profile.name}-sender-booking",
        timeout_s=_remaining(deadline, "sender-load"),
    )
    out["sender_loads_after_booking"] = loads
    _assert_strict_preferred_margin(
        load_a=loads["booked"],
        load_b=loads["idle"],
        accounting_isl=len(accounting_tokens),
        block_size=block_size,
        preferred_multiplier=preferred_multiplier,
        expect_b_wins=True,
        case_name=f"{profile.name}-margin-loaded",
    )
    clock_deadline = admit["end_monotonic"] + CLOCK_OBSERVE_S
    diverted = _wait_for_preferred_worker(
        stack,
        cold,
        expected_worker=worker_b,
        case_name=f"{profile.name}-divert-b",
        shared_tokens=probe_tokens,
        deadline=min(deadline, clock_deadline),
    )
    out["diverted"] = diverted
    if diverted["seen_monotonic"] > clock_deadline:
        raise AssertionError(
            f"{profile.name}: B witnessed after admit+{CLOCK_OBSERVE_S}s "
            f"({diverted['seen_monotonic']} > {clock_deadline})"
        )
    window_start = diverted["seen_monotonic"]
    hold_until = window_start + CLOCK_OBSERVE_S
    loaded = _decisive_generation_state(
        stack,
        cold,
        expected_worker=worker_b,
        case_name=f"{profile.name}-loaded-gen",
        shared_tokens=probe_tokens,
        deadline=min(deadline, clock_deadline),
    )
    out["loaded_generation"] = loaded
    held = _assert_preferred_holds(
        stack,
        cold,
        expected_worker=worker_b,
        case_name=f"{profile.name}-hold-b",
        shared_tokens=probe_tokens,
        window_start=window_start,
        hold_until=hold_until,
        overall_deadline=deadline,
    )
    out["held"] = held
    _free_booking(
        stack,
        owned_bookings,
        request_id=stale_id,
        timeout_s=_remaining(deadline, "explicit-free"),
    )
    restored_query = _wait_for_preferred_worker(
        stack,
        cold,
        expected_worker=worker_a,
        case_name=f"{profile.name}-after-free-a",
        shared_tokens=probe_tokens,
        deadline=deadline,
    )
    out["restored_query"] = restored_query
    restored = _decisive_generation_state(
        stack,
        cold,
        expected_worker=worker_a,
        case_name=f"{profile.name}-restored-gen",
        shared_tokens=probe_tokens,
        deadline=deadline,
    )
    out["restored_generation"] = restored
    owned_pubs = _wait_owned_event_channels(
        stack,
        owners={"provider": _provider_cr_name(stack)},
        case_name=f"{profile.name}-owned-channels",
        deadline=deadline,
    )
    out["owned_event_channels"] = owned_pubs
    scope_endpoints = _catalog_scope_event_channel_endpoints(
        stack, case_name=f"{profile.name}-scope-channels"
    )
    out["scope_event_channel_endpoints"] = sorted(scope_endpoints)
    out["construction"] = _observe_construction_events(
        stack,
        profile=profile,
        allowed_endpoints=scope_endpoints,
        case_name=f"{profile.name}-construction",
    )
    stack.close_consumer_pair()
    return out


def _run_grouped_profile(
    stack,
    *,
    profile: NativeConsumerProfile,
    cold: Mapping[str, Any],
    worker_a: int,
    worker_b: int,
    block_size: int,
    preferred_multiplier: float,
    owned_bookings: list[dict[str, Any]],
    peer_stack: ExitStack,
    tmp_path: Path,
    zone_a: str,
    zone_b: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    stack.open_consumer_pair(
        include_epp=True,
        frontend_env_overrides=_receiver_env(profile),
        epp_env_overrides=_receiver_env(profile),
        epp_require_readiness=True,
    )
    deadline = time.monotonic() + READY_BUDGET_S
    dual, peer_holder = _dual_publisher_sequence(
        stack,
        profile=profile,
        cold=cold,
        worker_a=worker_a,
        worker_b=worker_b,
        block_size=block_size,
        preferred_multiplier=preferred_multiplier,
        owned_bookings=owned_bookings,
        peer_stack=peer_stack,
        tmp_path=tmp_path,
        zone_a=zone_a,
        zone_b=zone_b,
        deadline=deadline,
        case_prefix=profile.name,
    )
    stack.close_consumer_pair()
    return dual, peer_holder


def _run_dual_publisher_on_stable_receivers(
    stack,
    *,
    profile: NativeConsumerProfile,
    cold: Mapping[str, Any],
    worker_a: int,
    worker_b: int,
    block_size: int,
    preferred_multiplier: float,
    owned_bookings: list[dict[str, Any]],
    peer_stack: ExitStack,
    tmp_path: Path,
    zone_a: str,
    zone_b: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Two-PUB proof on expiry=300 receivers so local CLOCK cannot fake peer B."""
    transport = NativeConsumerProfile(
        name=f"{profile.name}-transport300",
        expiry_secs=PROVIDER_EXPIRY_SECS,
        direct_zmq=profile.direct_zmq,
        rcvhwm=profile.rcvhwm,
        endpoints_per_sub=profile.endpoints_per_sub,
    )
    stack.open_consumer_pair(
        include_epp=True,
        frontend_env_overrides=_receiver_env(transport),
        epp_env_overrides=_receiver_env(transport),
        epp_require_readiness=True,
    )
    deadline = time.monotonic() + READY_BUDGET_S
    dual, peer_holder = _dual_publisher_sequence(
        stack,
        profile=transport,
        cold=cold,
        worker_a=worker_a,
        worker_b=worker_b,
        block_size=block_size,
        preferred_multiplier=preferred_multiplier,
        owned_bookings=owned_bookings,
        peer_stack=peer_stack,
        tmp_path=tmp_path,
        zone_a=zone_a,
        zone_b=zone_b,
        deadline=deadline,
        case_prefix=f"{profile.name}-pre",
    )
    stack.close_consumer_pair()
    return dual, peer_holder


def _run_expiry_clock_on_fresh_pair(
    stack,
    *,
    profile: NativeConsumerProfile,
    cold: Mapping[str, Any],
    worker_a: int,
    worker_b: int,
    block_size: int,
    preferred_multiplier: float,
    owned_bookings: list[dict[str, Any]],
    zone_a: str,
) -> dict[str, Any]:
    from tests.router.helper import unique_request_id

    out: dict[str, Any] = {}
    stack.open_consumer_pair(
        include_epp=True,
        frontend_env_overrides=_receiver_env(profile),
        epp_env_overrides=_receiver_env(profile),
        epp_require_readiness=True,
    )
    deadline = time.monotonic() + READY_BUDGET_S
    probe_tokens = _tokenize(stack, cold, profile.name, deadline)
    out["probe_token_count"] = len(probe_tokens)
    accounting_tokens = _accounting_probe_tokens(len(probe_tokens))
    out["accounting_probe_n"] = len(accounting_tokens)
    out["accounting_cold_proof"] = _prove_accounting_probe_cold(
        stack,
        accounting_tokens,
        worker_a=worker_a,
        worker_b=worker_b,
        case_name=f"{profile.name}-fresh-accounting-cold",
        deadline=deadline,
    )
    # Warm must Free before the configured 3s local CLOCK can fake restore.
    out["warm"] = _warm_handshake(
        stack,
        cold=cold,
        probe_tokens=probe_tokens,
        accounting_tokens=accounting_tokens,
        worker_a=worker_a,
        worker_b=worker_b,
        block_size=block_size,
        preferred_multiplier=preferred_multiplier,
        owned_bookings=owned_bookings,
        sender="provider",
        book_worker=worker_a,
        required_taint=zone_a,
        case_name=f"{profile.name}-fresh-warm",
        deadline=deadline,
    )
    baseline = _decisive_generation_state(
        stack,
        cold,
        expected_worker=worker_a,
        case_name=f"{profile.name}-fresh-baseline",
        shared_tokens=probe_tokens,
        deadline=deadline,
    )
    out["baseline"] = baseline
    stale_id = unique_request_id(f"native-expiry-stale-{profile.name}")
    admit = _admit_booking(
        stack,
        owner="provider",
        request_id=stale_id,
        tokens=_synthetic_tokens(BACKLOG_TOKENS, salt=11),
        required_taint=zone_a,
        expected_worker=worker_a,
        owned_bookings=owned_bookings,
        case_name=f"{profile.name}-clock-admit",
        deadline=deadline,
    )
    out["admit"] = admit
    clock_deadline = admit["end_monotonic"] + CLOCK_OBSERVE_S
    loads = _require_sender_booking_loads(
        stack,
        accounting_tokens,
        owner="provider",
        worker_a=worker_a,
        worker_b=worker_b,
        backlog_tokens=BACKLOG_TOKENS,
        booked_worker=worker_a,
        case_name=f"{profile.name}-clock-sender",
        timeout_s=_remaining(min(deadline, clock_deadline), "clock-sender"),
    )
    out["sender_loads_after_booking"] = loads
    diverted = _wait_for_preferred_worker(
        stack,
        cold,
        expected_worker=worker_b,
        case_name=f"{profile.name}-clock-divert-b",
        shared_tokens=probe_tokens,
        deadline=min(deadline, clock_deadline),
    )
    out["diverted"] = diverted
    if diverted["seen_monotonic"] > clock_deadline:
        raise AssertionError(
            f"{profile.name}: B after admit+{CLOCK_OBSERVE_S}s is too late for CLOCK"
        )
    loaded = _decisive_generation_state(
        stack,
        cold,
        expected_worker=worker_b,
        case_name=f"{profile.name}-clock-loaded-gen",
        shared_tokens=probe_tokens,
        deadline=min(deadline, clock_deadline),
    )
    out["loaded_generation"] = loaded
    recovered = _wait_for_preferred_worker(
        stack,
        cold,
        expected_worker=worker_a,
        case_name=f"{profile.name}-clock-recover-a",
        shared_tokens=probe_tokens,
        deadline=min(deadline, clock_deadline),
    )
    out["recovered"] = recovered
    if recovered["seen_monotonic"] > clock_deadline:
        raise AssertionError(
            f"{profile.name}: CLOCK recover {recovered['seen_monotonic']} > "
            f"admit+{CLOCK_OBSERVE_S}s ({clock_deadline})"
        )
    still = _require_sender_booking_loads(
        stack,
        accounting_tokens,
        owner="provider",
        worker_a=worker_a,
        worker_b=worker_b,
        backlog_tokens=BACKLOG_TOKENS,
        booked_worker=worker_a,
        case_name=f"{profile.name}-sender-still-booked",
        timeout_s=min(_remaining(deadline, "sender-still"), HTTP_CALL_CAP_S),
    )
    out["sender_loads_after_recover"] = still
    restored = _decisive_generation_state(
        stack,
        cold,
        expected_worker=worker_a,
        case_name=f"{profile.name}-clock-restored-gen",
        shared_tokens=probe_tokens,
        deadline=deadline,
    )
    out["restored_generation"] = restored
    owned_pubs = _wait_owned_event_channels(
        stack,
        owners={"provider": _provider_cr_name(stack)},
        case_name=f"{profile.name}-clock-channels",
        deadline=deadline,
    )
    out["owned_event_channels"] = owned_pubs
    scope_endpoints = _catalog_scope_event_channel_endpoints(
        stack, case_name=f"{profile.name}-clock-scope-channels"
    )
    out["scope_event_channel_endpoints"] = sorted(scope_endpoints)
    out["construction"] = _observe_construction_events(
        stack,
        profile=profile,
        allowed_endpoints=scope_endpoints,
        case_name=f"{profile.name}-clock-construction",
    )
    _free_booking(
        stack,
        owned_bookings,
        request_id=stale_id,
        timeout_s=_remaining(deadline, "clock-final-free"),
    )
    stack.close_consumer_pair()
    return out


def _dual_publisher_sequence(
    stack,
    *,
    profile: NativeConsumerProfile,
    cold: Mapping[str, Any],
    worker_a: int,
    worker_b: int,
    block_size: int,
    preferred_multiplier: float,
    owned_bookings: list[dict[str, Any]],
    peer_stack: ExitStack,
    tmp_path: Path,
    zone_a: str,
    zone_b: str,
    deadline: float,
    case_prefix: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from tests.router.helper import unique_request_id

    out: dict[str, Any] = {}
    probe_tokens = _tokenize(stack, cold, case_prefix, deadline)
    out["probe_token_count"] = len(probe_tokens)
    accounting_tokens = _accounting_probe_tokens(len(probe_tokens))
    out["accounting_probe_n"] = len(accounting_tokens)
    out["accounting_cold_proof"] = _prove_accounting_probe_cold(
        stack,
        accounting_tokens,
        worker_a=worker_a,
        worker_b=worker_b,
        case_name=f"{case_prefix}-accounting-cold",
        deadline=deadline,
    )
    out["warm_primary"] = _warm_handshake(
        stack,
        cold=cold,
        probe_tokens=probe_tokens,
        accounting_tokens=accounting_tokens,
        worker_a=worker_a,
        worker_b=worker_b,
        block_size=block_size,
        preferred_multiplier=preferred_multiplier,
        owned_bookings=owned_bookings,
        sender="provider",
        book_worker=worker_a,
        required_taint=zone_a,
        case_name=f"{case_prefix}-warm-primary",
        deadline=deadline,
    )
    peer_holder = _open_owned_peer_publisher(
        stack,
        tmp_path=tmp_path,
        exit_stack=peer_stack,
        deadline=deadline,
    )
    out["peer"] = {
        "container_name": peer_holder["container_name"],
        "metrics_port": peer_holder["metrics_port"],
        "cr_name": _peer_cr_name(stack),
    }
    out["warm_peer"] = _warm_handshake(
        stack,
        cold=cold,
        probe_tokens=probe_tokens,
        accounting_tokens=accounting_tokens,
        worker_a=worker_a,
        worker_b=worker_b,
        block_size=block_size,
        preferred_multiplier=preferred_multiplier,
        owned_bookings=owned_bookings,
        sender=peer_holder,
        book_worker=worker_a,
        required_taint=zone_a,
        case_name=f"{case_prefix}-warm-peer",
        deadline=deadline,
    )
    owned_pubs = _wait_owned_event_channels(
        stack,
        owners={
            "provider": _provider_cr_name(stack),
            "peer": _peer_cr_name(stack),
        },
        case_name=f"{case_prefix}-owned-channels",
        deadline=deadline,
    )
    if owned_pubs["provider"]["instance_id"] == owned_pubs["peer"]["instance_id"]:
        raise AssertionError(
            f"{case_prefix}: primary and peer share instance_id "
            f"{owned_pubs['provider']['instance_id']}"
        )
    if owned_pubs["provider"]["endpoint"] == owned_pubs["peer"]["endpoint"]:
        raise AssertionError(
            f"{case_prefix}: primary and peer share PUB endpoint "
            f"{owned_pubs['provider']['endpoint']!r}"
        )
    out["owned_event_channels"] = owned_pubs

    primary_id = unique_request_id(f"native-dual-a-{case_prefix}")
    admit_a = _admit_booking(
        stack,
        owner="provider",
        request_id=primary_id,
        tokens=_synthetic_tokens(BACKLOG_TOKENS, salt=3),
        required_taint=zone_a,
        expected_worker=worker_a,
        owned_bookings=owned_bookings,
        case_name=f"{case_prefix}-dual-a",
        deadline=deadline,
    )
    out["admit_primary"] = admit_a
    diverted = _wait_for_preferred_worker(
        stack,
        cold,
        expected_worker=worker_b,
        case_name=f"{case_prefix}-dual-divert-b",
        shared_tokens=probe_tokens,
        deadline=deadline,
    )
    out["diverted"] = diverted
    loaded = _decisive_generation_state(
        stack,
        cold,
        expected_worker=worker_b,
        case_name=f"{case_prefix}-dual-loaded-gen",
        shared_tokens=probe_tokens,
        deadline=deadline,
    )
    out["loaded_generation"] = loaded

    peer_id = unique_request_id(f"native-dual-b-{case_prefix}")
    admit_b = _admit_booking(
        stack,
        owner=peer_holder,
        request_id=peer_id,
        tokens=_synthetic_tokens(PEER_BACKLOG_TOKENS, salt=4),
        required_taint=zone_b,
        expected_worker=worker_b,
        owned_bookings=owned_bookings,
        case_name=f"{case_prefix}-dual-b",
        deadline=deadline,
    )
    out["admit_peer"] = admit_b
    # Dual peer-local: primary A + peer B both active (not single-booking idle/delta).
    peer_local = _require_dual_peer_local_loads(
        stack,
        accounting_tokens,
        owner=peer_holder,
        worker_a=worker_a,
        worker_b=worker_b,
        backlog_a=BACKLOG_TOKENS,
        backlog_b=PEER_BACKLOG_TOKENS,
        block_size=block_size,
        preferred_multiplier=preferred_multiplier,
        case_name=f"{case_prefix}-peer-local-dual",
        deadline=deadline,
    )
    out["peer_local_dual_loads"] = peer_local
    # Peer B backlog must be consumed: with both bookings live, preferred A wins.
    flipped = _wait_for_preferred_worker(
        stack,
        cold,
        expected_worker=worker_a,
        case_name=f"{case_prefix}-dual-flip-a",
        shared_tokens=probe_tokens,
        deadline=deadline,
    )
    out["flipped"] = flipped
    # Provider mirror of both bookings (not the peer sender snapshot): wait only
    # for the legitimate dual active_requests=1 state within remaining budget.
    both_rows = None
    last_both: list[Any] = []
    while time.monotonic() < deadline:
        timeout_s = deadline - time.monotonic()
        if timeout_s <= 0:
            break
        both_loads = stack.provider_potential_loads(
            accounting_tokens, timeout_s=timeout_s
        )
        last_both = both_loads
        rows = _dp0_rows(
            both_loads, worker_a, worker_b, f"{case_prefix}-provider-mirror"
        )
        if rows is None:
            time.sleep(POLL_S)
            continue
        if (
            rows[(worker_a, 0)]["active_requests"] == 1
            and rows[(worker_b, 0)]["active_requests"] == 1
        ):
            both_rows = rows
            break
        time.sleep(POLL_S)
    if both_rows is None:
        raise AssertionError(
            f"{case_prefix}: provider mirror never showed dual active_requests=1: "
            f"{last_both}"
        )
    out["provider_mirror_dual_loads"] = {
        "A": dict(both_rows[(worker_a, 0)]),
        "B": dict(both_rows[(worker_b, 0)]),
        "observation": "provider_mirror_not_peer_sender",
    }
    # Cold accounting ISL already subtracted inside potential_prefill by getter
    # only for cache hits; with proven cold probe, subtract accounting N.
    live_a = _live_backlog_from_potential(
        both_rows[(worker_a, 0)],
        accounting_isl=len(accounting_tokens),
        case_name=f"{case_prefix}-mirror",
        label="A",
    )
    live_b = _live_backlog_from_potential(
        both_rows[(worker_b, 0)],
        accounting_isl=len(accounting_tokens),
        case_name=f"{case_prefix}-mirror",
        label="B",
    )
    if live_a < BACKLOG_TOKENS or live_b < PEER_BACKLOG_TOKENS:
        raise AssertionError(
            f"{case_prefix}: live backlogs A={live_a} B={live_b} below booked "
            f"{BACKLOG_TOKENS}/{PEER_BACKLOG_TOKENS}; mirror={last_both}"
        )
    _assert_strict_preferred_margin(
        load_a=both_rows[(worker_a, 0)],
        load_b=both_rows[(worker_b, 0)],
        accounting_isl=len(accounting_tokens),
        block_size=block_size,
        preferred_multiplier=preferred_multiplier,
        expect_b_wins=False,
        case_name=f"{case_prefix}-dual-margin",
    )
    _free_booking(
        stack,
        owned_bookings,
        request_id=peer_id,
        timeout_s=_remaining(deadline, "dual-free-peer"),
    )
    _free_booking(
        stack,
        owned_bookings,
        request_id=primary_id,
        timeout_s=_remaining(deadline, "dual-free-primary"),
    )
    _wait_for_zero_active(
        stack,
        accounting_tokens,
        owner="provider",
        worker_a=worker_a,
        worker_b=worker_b,
        case_name=f"{case_prefix}-dual-zero-provider",
        deadline=deadline,
    )
    _wait_for_zero_active(
        stack,
        accounting_tokens,
        owner=peer_holder,
        worker_a=worker_a,
        worker_b=worker_b,
        case_name=f"{case_prefix}-dual-zero-peer",
        deadline=deadline,
    )
    restored = _wait_for_preferred_worker(
        stack,
        cold,
        expected_worker=worker_a,
        case_name=f"{case_prefix}-dual-restore-a",
        shared_tokens=probe_tokens,
        deadline=deadline,
    )
    out["restored"] = restored
    restored_gen = _decisive_generation_state(
        stack,
        cold,
        expected_worker=worker_a,
        case_name=f"{case_prefix}-dual-restored-gen",
        shared_tokens=probe_tokens,
        deadline=deadline,
    )
    out["restored_generation"] = restored_gen
    scope_endpoints = _catalog_scope_event_channel_endpoints(
        stack, case_name=f"{case_prefix}-scope-channels"
    )
    out["scope_event_channel_endpoints"] = sorted(scope_endpoints)
    out["construction"] = _observe_construction_events(
        stack,
        profile=profile,
        allowed_endpoints=scope_endpoints,
        case_name=f"{case_prefix}-construction",
    )
    return out, peer_holder


def _cleared_native_transport_env() -> dict[str, str | None]:
    return {
        EXPIRY_ENV: None,
        DIRECT_ENV: None,
        HWM_ENV: None,
        POOL_ENV: None,
        BROKER_URL_ENV: None,
        BROKER_ENABLED_ENV: None,
        SERVING_LOAD_ENV: None,
    }


def _receiver_env(profile: NativeConsumerProfile) -> dict[str, str | None]:
    overlay = _cleared_native_transport_env()
    overlay[SERVING_LOAD_ENV] = "false"
    overlay[BROKER_ENABLED_ENV] = None
    overlay[BROKER_URL_ENV] = None
    overlay["DYN_ROUTER_REPLICA_SYNC"] = "true"
    overlay["DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT"] = "0"
    overlay["DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT_DECAY"] = "0"
    overlay["DYN_ROUTER_TEMPERATURE"] = "0"
    overlay["DYN_ROUTER_PREFILL_LOAD_SCALE"] = "1"
    overlay["DYN_ROUTER_TRACK_PREFILL_TOKENS"] = "true"
    overlay["DYN_ROUTER_TRACK_OUTPUT_BLOCKS"] = "false"
    overlay["DYN_ROUTER_PREFILL_LOAD_MODEL"] = "none"
    overlay["DYN_EVENT_PLANE"] = "zmq"
    if profile.expiry_secs is None:
        overlay[EXPIRY_ENV] = None
    else:
        overlay[EXPIRY_ENV] = str(profile.expiry_secs)
    if profile.direct_zmq is None:
        overlay[DIRECT_ENV] = None
    else:
        overlay[DIRECT_ENV] = "true" if profile.direct_zmq else "false"
    if profile.rcvhwm is None:
        overlay[HWM_ENV] = None
    else:
        overlay[HWM_ENV] = str(profile.rcvhwm)
    if profile.endpoints_per_sub is None:
        overlay[POOL_ENV] = None
    else:
        overlay[POOL_ENV] = str(profile.endpoints_per_sub)
    return overlay


def _require_profile(profile_name: str) -> NativeConsumerProfile:
    profile = NATIVE_CONSUMER_PROFILES.get(profile_name)
    if profile is None:
        raise AssertionError(
            f"unknown native consumer profile {profile_name!r}; "
            f"expected one of {sorted(NATIVE_CONSUMER_PROFILES)}"
        )
    return profile


def _preferred_multiplier_for_worker(
    preferred_taints: Mapping[str, float], worker_taints: set[str]
) -> float | None:
    """Mirror protocols.rs preferred_taint_multiplier: exp(-tanh(sum matches)).

    Empty map → None. Nonempty map with no matching taints → 1.0 (Rust Some(1.0)).
    """
    if not preferred_taints:
        return None
    matched = [
        float(weight)
        for taint, weight in preferred_taints.items()
        if taint in worker_taints
    ]
    if not matched:
        return 1.0
    return math.exp(-math.tanh(sum(matched)))


def _remaining(deadline: float, step: str) -> float:
    left = deadline - time.monotonic()
    if left <= 0:
        raise AssertionError(f"budget exhausted at {step}")
    return left


def _require_hosting_provider_native_policy(stack) -> dict[str, float]:
    """Hosting provider gate: credit 1 / replica true / model none / expiry 300 env.

    stack.policy_parameters is the hosting provider policy, not FE/EPP/peer
    load-only credit-0 scoring. Do not treat these values as receiver policy.
    """
    parameters = stack.policy_parameters
    required = (
        "overlap_score_credit",
        "overlap_score_credit_decay",
        "prefill_load_scale",
        "router_temperature",
    )
    missing = [name for name in required if name not in parameters]
    if missing:
        raise AssertionError(
            f"native hosting provider policy_parameters missing {missing}: "
            f"{parameters}"
        )
    if parameters["overlap_score_credit"] != CREDIT_FULL:
        raise AssertionError(
            "native hosting provider overlap_score_credit must be 1.0 "
            f"(CACHE inputs for serve_indexer), got "
            f"{parameters['overlap_score_credit']}"
        )
    if parameters["overlap_score_credit_decay"] != 0.0:
        raise AssertionError(
            "native hosting provider overlap_score_credit_decay must be 0.0, "
            f"got {parameters['overlap_score_credit_decay']}"
        )
    if parameters["prefill_load_scale"] != 1.0:
        raise AssertionError(
            "native hosting provider prefill_load_scale must be 1.0, "
            f"got {parameters['prefill_load_scale']}"
        )
    if parameters["router_temperature"] != 0.0:
        raise AssertionError(
            "native hosting provider router_temperature must be 0.0, "
            f"got {parameters['router_temperature']}"
        )
    encoded = stack.provider_config_json
    if not isinstance(encoded, str) or not encoded:
        raise AssertionError("routing parity stack has no child provider config JSON")
    raw = json.loads(encoded)
    if "router_prefill_load_model" not in raw:
        raise AssertionError("native provider JSON missing router_prefill_load_model")
    if raw["router_prefill_load_model"] != "none":
        raise AssertionError(
            "native provider router_prefill_load_model must be exactly 'none', "
            f"got {raw['router_prefill_load_model']!r}"
        )
    if raw.get("router_track_prefill_tokens") is not True:
        raise AssertionError(
            "native provider must keep router_track_prefill_tokens=true: "
            f"{raw.get('router_track_prefill_tokens')!r}"
        )
    if raw.get("router_track_output_blocks") is not False:
        raise AssertionError(
            "native provider must keep router_track_output_blocks=false: "
            f"{raw.get('router_track_output_blocks')!r}"
        )
    if raw.get("router_replica_sync") is not True:
        raise AssertionError(
            "native provider must be constructed with router_replica_sync=true "
            "(pass provider_config_overrides=native_provider_config_overrides()); "
            f"got {raw.get('router_replica_sync')!r}"
        )
    if raw.get("overlap_score_credit") != CREDIT_FULL:
        raise AssertionError(
            "native hosting provider overlap_score_credit must be 1.0, "
            f"got {raw.get('overlap_score_credit')!r}"
        )
    return parameters


def _require_zone_workers(stack) -> tuple[int, int]:
    from tests.router.epp_parity import ZONE_A_TAINT, ZONE_B_TAINT

    if stack.enable_worker_topology is not True:
        raise AssertionError(
            "native consumer runner requires enable_worker_topology=True"
        )
    worker_a = stack.worker_ids["A"]
    worker_b = stack.worker_ids["B"]
    if stack.worker_configs[worker_a]["taints"] & {ZONE_A_TAINT} != {ZONE_A_TAINT}:
        raise AssertionError(
            f"worker A {worker_a} missing published {ZONE_A_TAINT}: "
            f"{stack.worker_configs[worker_a]}"
        )
    if stack.worker_configs[worker_b]["taints"] & {ZONE_B_TAINT} != {ZONE_B_TAINT}:
        raise AssertionError(
            f"worker B {worker_b} missing published {ZONE_B_TAINT}: "
            f"{stack.worker_configs[worker_b]}"
        )
    return worker_a, worker_b


def _require_positive_block_size(stack) -> int:
    size = stack.worker_block_size
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise AssertionError(
            f"stack worker_block_size {size!r} is not a positive instance size"
        )
    return size


def _cold_case(model_name: str) -> dict[str, Any]:
    return {
        "path": "/v1/completions",
        "payload": {
            "model": model_name,
            "prompt": (
                "native consumer cold routing stem " * 8 + ("unique cold branch " * 12)
            ),
            "stream": True,
            "max_tokens": 1,
        },
    }


def _with_nvext(payload: dict[str, Any], **fields: Any) -> dict[str, Any]:
    body = dict(payload)
    nvext = dict(body.get("nvext") or {})
    nvext.update(fields)
    body["nvext"] = nvext
    return body


def _preferred_payload(cold: Mapping[str, Any], *, salt: str) -> dict[str, Any]:
    from tests.router.epp_parity import ZONE_A_TAINT

    return _with_nvext(
        cold["payload"],
        cache_salt=salt,
        routing_constraints={"preferred_taints": {ZONE_A_TAINT: PREFERRED_WEIGHT}},
    )


def _synthetic_tokens(count: int, *, salt: int) -> list[int]:
    """Booking-only synthetic tokens; salt domain 0..1019 stays below accounting base."""
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise AssertionError(
            f"synthetic token count must be a positive int (bool excluded), "
            f"got {count!r}"
        )
    if (
        not isinstance(salt, int)
        or isinstance(salt, bool)
        or salt < 0
        or salt > SYNTHETIC_SALT_MAX
    ):
        raise AssertionError(
            f"synthetic salt must be int in 0..{SYNTHETIC_SALT_MAX} "
            f"(bool excluded), got {salt!r}"
        )
    base = SYNTHETIC_TOKEN_BASE + salt * SYNTHETIC_SALT_STRIDE
    end = base + count
    if end > ACCOUNTING_TOKEN_BASE:
        raise AssertionError(
            f"synthetic range [{base}, {end}) crosses accounting base "
            f"{ACCOUNTING_TOKEN_BASE} (count={count}, salt={salt})"
        )
    return [base + index for index in range(count)]


def _accounting_probe_tokens(count: int) -> list[int]:
    """Synthetic load-accounting probe: same N as FE tokenizer, never backend/booking."""
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise AssertionError(
            f"accounting probe length must be a positive int (bool excluded), "
            f"got {count!r}"
        )
    end = ACCOUNTING_TOKEN_BASE + count
    if end > U32_EXCLUSIVE_END:
        raise AssertionError(
            f"accounting range [{ACCOUNTING_TOKEN_BASE}, {end}) exceeds u32 "
            f"exclusive end {U32_EXCLUSIVE_END}"
        )
    return [ACCOUNTING_TOKEN_BASE + index for index in range(count)]


def _tokenize(stack, cold: Mapping[str, Any], label: str, deadline: float) -> list[int]:
    from tests.router.epp_parity import collect_frontend_query_token_ids
    from tests.router.helper import unique_request_id

    tokens = collect_frontend_query_token_ids(
        stack.frontend_url,
        cold["payload"],
        path=cold["path"],
        request_id=unique_request_id(f"fe-native-tok-{label}"),
        timeout_s=min(_remaining(deadline, "tokenize"), ROUTE_CALL_CAP_S),
    )
    if not tokens:
        raise AssertionError(f"{label}: cold tokenization returned no tokens")
    return tokens


def _prove_accounting_probe_cold(
    stack,
    accounting_tokens: list[int],
    *,
    worker_a: int,
    worker_b: int,
    case_name: str,
    deadline: float,
) -> dict[str, Any]:
    """Prove provider overlap for the accounting probe is fully cold before admit."""
    timeout_s = _remaining(deadline, f"{case_name}-cold-overlap")
    scores = stack.provider_overlap_scores(accounting_tokens, timeout_s=timeout_s)
    if time.monotonic() > deadline:
        raise AssertionError(f"{case_name}: cold overlap returned after deadline")
    if not isinstance(scores, Mapping):
        raise AssertionError(
            f"{case_name}: provider_overlap_scores is not a Mapping: {type(scores)}"
        )
    shared = scores.get("shared_cache")
    if not isinstance(shared, Mapping):
        raise AssertionError(
            f"{case_name}: shared_cache missing/malformed: {scores.get('shared_cache')!r}"
        )
    if shared.get("enabled") is not False:
        raise AssertionError(
            f"{case_name}: shared_cache.enabled must be False for native none: "
            f"{shared}"
        )
    workers = scores.get("workers")
    if not isinstance(workers, list):
        raise AssertionError(
            f"{case_name}: overlap workers missing/malformed: {workers!r}"
        )
    by_key: dict[tuple[int, int], Mapping[str, Any]] = {}
    for row in workers:
        if not isinstance(row, Mapping):
            raise AssertionError(
                f"{case_name}: overlap worker row not Mapping: {row!r}"
            )
        worker_id = row.get("worker_id")
        dp_rank = row.get("dp_rank")
        if (
            not isinstance(worker_id, int)
            or isinstance(worker_id, bool)
            or not isinstance(dp_rank, int)
            or isinstance(dp_rank, bool)
        ):
            raise AssertionError(
                f"{case_name}: overlap worker_id/dp_rank malformed: {row}"
            )
        key = (worker_id, dp_rank)
        if key in by_key:
            raise AssertionError(f"{case_name}: duplicate overlap row {key}")
        by_key[key] = row
    allowed = {(worker_a, 0), (worker_b, 0)}
    if set(by_key) != allowed:
        raise AssertionError(
            f"{case_name}: overlap workers must be exactly {sorted(allowed)}, "
            f"got {sorted(by_key)}"
        )
    cold_rows: dict[str, dict[str, Any]] = {}
    for label, worker_id in (("A", worker_a), ("B", worker_b)):
        row = by_key[(worker_id, 0)]
        for field in (
            "device_blocks",
            "host_pinned_blocks",
            "disk_blocks",
            "host_pinned_extension_blocks",
            "disk_extension_blocks",
        ):
            if field not in row:
                raise AssertionError(
                    f"{case_name}: overlap row missing {field} on {label}: {row}"
                )
            value = row[field]
            if not isinstance(value, int) or isinstance(value, bool) or value != 0:
                raise AssertionError(
                    f"{case_name}: {label} {field} must be exact 0 for cold proof, "
                    f"got {value!r}; row={row}"
                )
        if "shared_beyond_device_blocks" not in row:
            raise AssertionError(
                f"{case_name}: overlap row missing shared_beyond_device_blocks on "
                f"{label}: {row}"
            )
        if row["shared_beyond_device_blocks"] is not None:
            raise AssertionError(
                f"{case_name}: {label} shared_beyond_device_blocks must be None "
                f"with shared disabled, got {row['shared_beyond_device_blocks']!r}"
            )
        cold_rows[label] = {
            "worker_id": worker_id,
            "dp_rank": 0,
            "device_blocks": row["device_blocks"],
            "host_pinned_blocks": row["host_pinned_blocks"],
            "disk_blocks": row["disk_blocks"],
            "host_pinned_extension_blocks": row["host_pinned_extension_blocks"],
            "disk_extension_blocks": row["disk_extension_blocks"],
            "shared_beyond_device_blocks": row["shared_beyond_device_blocks"],
        }
    return {
        "accounting_probe_n": len(accounting_tokens),
        "shared_cache_enabled": False,
        "workers": cold_rows,
        "observation": "provider_overlap_cold_before_admit",
    }


def _dp0_rows(
    loads: list[Mapping[str, Any]],
    worker_a: int,
    worker_b: int,
    case_name: str,
) -> dict[tuple[int, int], Mapping[str, Any]] | None:
    """Validate present rows, then delegate incompleteness None/wait to common.

    ``common._dp0_worker_load_rows`` returns None for both missing allowed rows
    and extra/nonzero-DP keys. Raise here on malformed or disallowed keys so
    native waiters do not treat those hard errors as incomplete waits.
    Duplicates still fail inside common after this gate.
    """
    from tests.router.common import _dp0_worker_load_rows

    allowed = {(worker_a, 0), (worker_b, 0)}
    for row in loads:
        if not isinstance(row, Mapping):
            raise AssertionError(
                f"{case_name}: potential_load row is not a Mapping: {row!r}"
            )
        worker_id = row.get("worker_id")
        dp_rank = row.get("dp_rank")
        if (
            not isinstance(worker_id, int)
            or isinstance(worker_id, bool)
            or not isinstance(dp_rank, int)
            or isinstance(dp_rank, bool)
        ):
            raise AssertionError(
                f"{case_name}: worker_id/dp_rank must be exact ints "
                f"(bool excluded), got worker_id={worker_id!r} dp_rank={dp_rank!r}"
            )
        key = (worker_id, dp_rank)
        if key not in allowed:
            raise AssertionError(
                f"{case_name}: potential_load key {key} outside allowed "
                f"{sorted(allowed)}; loads={loads}"
            )
    return _dp0_worker_load_rows(loads, worker_a, worker_b, case_name)


def _owner_potential_loads(
    stack, owner: Any, tokens: list[int], *, timeout_s: float
) -> list[Mapping[str, Any]]:
    """Read potential loads from the actual booking owner (provider or peer)."""
    if owner == "provider":
        return stack.provider_potential_loads(tokens, timeout_s=timeout_s)
    peer_owner = owner.get("owner")
    if peer_owner is None:
        raise AssertionError("peer router owner was cleared before get_potential_loads")
    try:
        return peer_owner.call(
            "get_potential_loads",
            {"token_ids": list(tokens), "cache_namespace": None},
            timeout_s,
        )
    except NativeRouterFailure as exc:
        if exc.timed_out:
            raise AssertionError(
                f"owner get_potential_loads outlived the {timeout_s}s budget"
            ) from exc
        raise


def _wait_for_zero_active(
    stack,
    tokens: list[int],
    *,
    owner: Any = "provider",
    worker_a: int,
    worker_b: int,
    case_name: str,
    deadline: float,
) -> None:
    last: list[Any] = []
    while time.monotonic() < deadline:
        timeout_s = deadline - time.monotonic()
        if timeout_s <= 0:
            break
        loads = _owner_potential_loads(stack, owner, tokens, timeout_s=timeout_s)
        last = loads
        # Incomplete DP0 rows are a transient wait; duplicate/schema faults raise.
        rows = _dp0_rows(loads, worker_a, worker_b, case_name)
        if rows is not None and all(rows[key]["active_requests"] == 0 for key in rows):
            return
        time.sleep(POLL_S)
    raise AssertionError(
        f"{case_name}: owner active_requests did not return to zero: {last}"
    )


def _require_sender_booking_loads(
    stack,
    tokens: list[int],
    *,
    owner: Any,
    worker_a: int,
    worker_b: int,
    backlog_tokens: int,
    booked_worker: int,
    case_name: str,
    timeout_s: float,
) -> dict[str, Any]:
    """Strict single-booking sender snapshot: booked=1, other=0, backlog delta."""
    loads = _owner_potential_loads(stack, owner, tokens, timeout_s=timeout_s)
    rows = _dp0_rows(loads, worker_a, worker_b, case_name)
    if rows is None:
        raise AssertionError(f"{case_name}: incomplete potential_loads: {loads}")
    other = worker_b if booked_worker == worker_a else worker_a
    booked = rows[(booked_worker, 0)]
    idle = rows[(other, 0)]
    if booked["active_requests"] != 1:
        raise AssertionError(
            f"{case_name}: booked worker {booked_worker} active_requests="
            f"{booked['active_requests']} != 1; loads={loads}"
        )
    if idle["active_requests"] != 0:
        raise AssertionError(
            f"{case_name}: idle worker {other} active_requests="
            f"{idle['active_requests']} != 0; loads={loads}"
        )
    for key in (
        "potential_prefill_tokens",
        "potential_decode_blocks",
        "active_requests",
    ):
        if key not in booked or key not in idle:
            raise AssertionError(
                f"{case_name}: PotentialLoad missing {key}: loads={loads}"
            )
    delta = booked["potential_prefill_tokens"] - idle["potential_prefill_tokens"]
    if delta < backlog_tokens:
        raise AssertionError(
            f"{case_name}: potential_prefill delta {delta} < booked "
            f"{backlog_tokens}; loads={loads}"
        )
    return {
        "owner": "provider" if owner == "provider" else "peer",
        "booked_worker": booked_worker,
        "booked": dict(booked),
        "idle": dict(idle),
        "prefill_delta": delta,
        "loads": [dict(row) for row in loads],
    }


def _require_dual_peer_local_loads(
    stack,
    tokens: list[int],
    *,
    owner: Any,
    worker_a: int,
    worker_b: int,
    backlog_a: int,
    backlog_b: int,
    block_size: int,
    preferred_multiplier: float,
    case_name: str,
    deadline: float,
) -> dict[str, Any]:
    """Peer-local dual-booking snapshot after primary A + peer B admits.

    Expects exact DP0 A/B with active_requests=1/1 and separate live backlogs.
    Only waits for legitimate A:0→1 while B already holds 1; does not reuse the
    single-booking idle/delta helper. Not a primary-mirror substitute.
    """
    last: list[Any] = []
    while time.monotonic() < deadline:
        timeout_s = deadline - time.monotonic()
        if timeout_s <= 0:
            break
        loads = _owner_potential_loads(stack, owner, tokens, timeout_s=timeout_s)
        last = loads
        # Incomplete DP0 catalog rows may wait; hard schema/rank faults raise.
        rows = _dp0_rows(loads, worker_a, worker_b, case_name)
        if rows is None:
            time.sleep(POLL_S)
            continue
        row_a = rows[(worker_a, 0)]
        row_b = rows[(worker_b, 0)]
        for label, row in (("A", row_a), ("B", row_b)):
            for key in (
                "potential_prefill_tokens",
                "potential_decode_blocks",
                "active_requests",
            ):
                if key not in row:
                    raise AssertionError(
                        f"{case_name}: PotentialLoad missing {key} on {label}: "
                        f"{loads}"
                    )
            active = row["active_requests"]
            if not isinstance(active, int) or isinstance(active, bool) or active < 0:
                raise AssertionError(
                    f"{case_name}: {label} active_requests malformed: {active!r}; "
                    f"loads={loads}"
                )
            if active > 1:
                raise AssertionError(
                    f"{case_name}: {label} multi-booking active_requests={active}; "
                    f"loads={loads}"
                )
            prefill = row["potential_prefill_tokens"]
            if not isinstance(prefill, int) or isinstance(prefill, bool) or prefill < 0:
                raise AssertionError(
                    f"{case_name}: {label} potential_prefill_tokens malformed: "
                    f"{prefill!r}; loads={loads}"
                )
        active_a = row_a["active_requests"]
        active_b = row_b["active_requests"]
        # Peer just admitted B locally; B must already be 1. A may still be 0→1.
        if active_b != 1:
            raise AssertionError(
                f"{case_name}: peer-local B active_requests={active_b} != 1 after "
                f"peer B admit; loads={loads}"
            )
        if active_a == 0:
            time.sleep(POLL_S)
            continue
        if active_a != 1:
            raise AssertionError(
                f"{case_name}: peer-local A active_requests={active_a} not in "
                f"{{0,1}} wait path; loads={loads}"
            )
        live_a = _live_backlog_from_potential(
            row_a,
            accounting_isl=len(tokens),
            case_name=case_name,
            label="A",
        )
        live_b = _live_backlog_from_potential(
            row_b,
            accounting_isl=len(tokens),
            case_name=case_name,
            label="B",
        )
        if live_a < backlog_a or live_b < backlog_b:
            raise AssertionError(
                f"{case_name}: peer-local live backlogs A={live_a} B={live_b} "
                f"below booked {backlog_a}/{backlog_b}; loads={loads}"
            )
        _assert_strict_preferred_margin(
            load_a=row_a,
            load_b=row_b,
            accounting_isl=len(tokens),
            block_size=block_size,
            preferred_multiplier=preferred_multiplier,
            expect_b_wins=False,
            case_name=f"{case_name}-margin",
        )
        return {
            "owner": "peer",
            "observation": "peer_local_dual_not_provider_mirror",
            "A": dict(row_a),
            "B": dict(row_b),
            "live_prefill_a": live_a,
            "live_prefill_b": live_b,
            "loads": [dict(row) for row in loads],
        }
    raise AssertionError(
        f"{case_name}: peer-local dual A1/B1 never converged before deadline: "
        f"{last}"
    )


def _live_backlog_from_potential(
    row: Mapping[str, Any], *, accounting_isl: int, case_name: str, label: str
) -> int:
    """Active backlog from a cold-proven PotentialLoad: prefill - accounting ISL."""
    if accounting_isl <= 0:
        raise AssertionError(f"{case_name}: accounting_isl must be positive")
    if "potential_prefill_tokens" not in row:
        raise AssertionError(
            f"{case_name}: {label} missing potential_prefill_tokens: {row}"
        )
    prefill = row["potential_prefill_tokens"]
    if not isinstance(prefill, int) or isinstance(prefill, bool) or prefill < 0:
        raise AssertionError(
            f"{case_name}: {label} potential_prefill_tokens malformed: {prefill!r}"
        )
    live = prefill - accounting_isl
    if live < 0:
        raise AssertionError(
            f"{case_name}: {label} live backlog {live} negative after subtracting "
            f"cold accounting ISL {accounting_isl}; row={row}"
        )
    return live


def _assert_strict_preferred_margin(
    *,
    load_a: Mapping[str, Any],
    load_b: Mapping[str, Any],
    accounting_isl: int,
    block_size: int,
    preferred_multiplier: float,
    expect_b_wins: bool,
    case_name: str,
) -> None:
    """Receiver load-only relative cost using actual PotentialLoad rows.

    Builtin default scorer always adds ``potential_decode_blocks`` (decode_cost),
    even with track_output_blocks=false. Prefill uses cold-proven potential_prefill
    (active backlog + accounting ISL). Preferred taint multiplies the full logit.
    Provider hosting credit1 is not applied here (receiver policy is credit0).
    """
    if accounting_isl <= 0:
        raise AssertionError(f"{case_name}: accounting_isl must be positive")
    if (
        not isinstance(block_size, int)
        or isinstance(block_size, bool)
        or block_size <= 0
    ):
        raise AssertionError(f"{case_name}: block_size {block_size!r} invalid")
    for label, row in (("A", load_a), ("B", load_b)):
        if (
            "potential_prefill_tokens" not in row
            or "potential_decode_blocks" not in row
        ):
            raise AssertionError(
                f"{case_name}: {label} PotentialLoad missing prefill/decode: {row}"
            )
        prefill = row["potential_prefill_tokens"]
        decode = row["potential_decode_blocks"]
        if not isinstance(prefill, int) or isinstance(prefill, bool) or prefill < 0:
            raise AssertionError(
                f"{case_name}: {label} potential_prefill_tokens malformed: {prefill!r}"
            )
        if not isinstance(decode, int) or isinstance(decode, bool) or decode < 0:
            raise AssertionError(
                f"{case_name}: {label} potential_decode_blocks malformed: {decode!r}"
            )
    # scale=1.0; request_cost weight 0 under native overlays.
    logit_a = (float(load_a["potential_prefill_tokens"]) / float(block_size)) + float(
        load_a["potential_decode_blocks"]
    )
    logit_b = (float(load_b["potential_prefill_tokens"]) / float(block_size)) + float(
        load_b["potential_decode_blocks"]
    )
    cost_a = logit_a * preferred_multiplier
    cost_b = logit_b * 1.0
    if expect_b_wins:
        if not (cost_a > cost_b):
            raise AssertionError(
                f"{case_name}: expected B to win under preferred multiplier "
                f"{preferred_multiplier}, but cost_a={cost_a} cost_b={cost_b} "
                f"load_a={dict(load_a)} load_b={dict(load_b)} "
                f"accounting_isl={accounting_isl}"
            )
    else:
        if not (cost_a < cost_b):
            raise AssertionError(
                f"{case_name}: expected A to win under preferred multiplier "
                f"{preferred_multiplier}, but cost_a={cost_a} cost_b={cost_b} "
                f"load_a={dict(load_a)} load_b={dict(load_b)} "
                f"accounting_isl={accounting_isl}"
            )


def _route_timeout(deadline: float) -> float:
    left = deadline - time.monotonic()
    if left <= 0:
        raise AssertionError("observation deadline already exhausted")
    return min(left, ROUTE_CALL_CAP_S)


def _observe_query_routes(
    stack,
    cold: Mapping[str, Any],
    *,
    expected_worker: int,
    case_name: str,
    shared_tokens: list[int],
    deadline: float,
) -> dict[str, Any]:
    from tests.router.helper import unique_request_id

    salt = f"native-query-{case_name}-{uuid.uuid4()}"
    payload = _preferred_payload(cold, salt=salt)
    timeout_s = _route_timeout(deadline)
    frontend = stack.collect_frontend_route(
        payload,
        path=cold["path"],
        request_id=unique_request_id(f"fe-query-{case_name}"),
        timeout_s=timeout_s,
    )
    if time.monotonic() > deadline:
        raise AssertionError(f"{case_name}: FE query returned after deadline")
    timeout_s = _route_timeout(deadline)
    epp_route = stack.collect_epp_route(
        payload,
        path=cold["path"],
        request_id=unique_request_id(f"epp-query-{case_name}"),
        timeout_s=timeout_s,
    )
    if time.monotonic() > deadline:
        raise AssertionError(f"{case_name}: EPP query returned after deadline")
    if frontend.token_ids != shared_tokens:
        raise AssertionError(
            f"{case_name}: FE query tokens diverged from shared cold tokenization "
            f"({len(frontend.token_ids)} vs {len(shared_tokens)})"
        )
    if epp_route.token_ids != shared_tokens:
        raise AssertionError(
            f"{case_name}: EPP token_ids diverged from shared cold tokenization "
            f"({len(epp_route.token_ids)} vs {len(shared_tokens)})"
        )
    if frontend.dp_rank != 0 or epp_route.dp_rank != 0:
        raise AssertionError(
            f"{case_name}: expected dp_rank=0, got FE={frontend.dp_rank} "
            f"EPP={epp_route.dp_rank}"
        )
    catalog = set(stack.worker_ids.values())
    if frontend.worker_id not in catalog or epp_route.worker_id not in catalog:
        raise AssertionError(
            f"{case_name}: worker outside catalog FE={frontend.worker_id} "
            f"EPP={epp_route.worker_id} catalog={catalog}"
        )
    if (frontend.worker_id, epp_route.worker_id) != (
        expected_worker,
        expected_worker,
    ):
        raise WorkerSelectionPending(
            f"{case_name}: FE={frontend.worker_id} EPP={epp_route.worker_id} "
            f"expected={expected_worker}"
        )
    return {
        "frontend_worker": frontend.worker_id,
        "epp_worker": epp_route.worker_id,
        "frontend_dp_rank": frontend.dp_rank,
        "epp_dp_rank": epp_route.dp_rank,
        "cache_salt": salt,
        "token_count": len(shared_tokens),
        "seen_monotonic": time.monotonic(),
        "observation": "query",
    }


def _decisive_generation_state(
    stack,
    cold: Mapping[str, Any],
    *,
    expected_worker: int,
    case_name: str,
    shared_tokens: list[int],
    deadline: float,
) -> dict[str, Any]:
    """One real FE generation plus EPP query at a decisive state."""
    from tests.router.helper import unique_request_id

    salt = f"native-gen-{case_name}-{uuid.uuid4()}"
    payload = _preferred_payload(cold, salt=salt)
    timeout_s = _route_timeout(deadline)
    generation = stack.collect_frontend_generation(
        payload,
        path=cold["path"],
        request_id=unique_request_id(f"fe-gen-{case_name}"),
        timeout_s=timeout_s,
    )
    if time.monotonic() > deadline:
        raise AssertionError(f"{case_name}: FE generation returned after deadline")
    timeout_s = _route_timeout(deadline)
    epp_route = stack.collect_epp_route(
        payload,
        path=cold["path"],
        request_id=unique_request_id(f"epp-genstate-{case_name}"),
        timeout_s=timeout_s,
    )
    if time.monotonic() > deadline:
        raise AssertionError(f"{case_name}: EPP query returned after deadline")
    if epp_route.token_ids != shared_tokens:
        raise AssertionError(
            f"{case_name}: EPP tokens diverged from shared cold tokenization"
        )
    if generation.dp_rank != 0 or epp_route.dp_rank != 0:
        raise AssertionError(
            f"{case_name}: expected dp_rank=0, got FE gen={generation.dp_rank} "
            f"EPP={epp_route.dp_rank}"
        )
    if (generation.worker_id, epp_route.worker_id) != (
        expected_worker,
        expected_worker,
    ):
        raise AssertionError(
            f"{case_name}: decisive FE gen={generation.worker_id} "
            f"EPP={epp_route.worker_id}, expected={expected_worker}"
        )
    return {
        "frontend_worker": generation.worker_id,
        "epp_worker": epp_route.worker_id,
        "cache_salt": salt,
        "token_count": len(shared_tokens),
        "seen_monotonic": time.monotonic(),
        "observation": "frontend_generation+epp_query",
    }


def _wait_for_preferred_worker(
    stack,
    cold: Mapping[str, Any],
    *,
    expected_worker: int,
    case_name: str,
    shared_tokens: list[int],
    deadline: float,
) -> dict[str, Any]:
    last_pending: WorkerSelectionPending | None = None
    while time.monotonic() < deadline:
        try:
            return _observe_query_routes(
                stack,
                cold,
                expected_worker=expected_worker,
                case_name=case_name,
                shared_tokens=shared_tokens,
                deadline=deadline,
            )
        except WorkerSelectionPending as exc:
            last_pending = exc
            time.sleep(POLL_S)
    raise AssertionError(
        f"{case_name}: did not observe expected worker {expected_worker} "
        f"before deadline; last_pending={last_pending}"
    )


def _assert_preferred_holds(
    stack,
    cold: Mapping[str, Any],
    *,
    expected_worker: int,
    case_name: str,
    shared_tokens: list[int],
    window_start: float,
    hold_until: float,
    overall_deadline: float,
) -> dict[str, Any]:
    """Hold B from window_start through hold_until; final confirm uses overall_deadline."""
    samples: list[dict[str, Any]] = []
    while time.monotonic() < overall_deadline:
        sample = _observe_query_routes(
            stack,
            cold,
            expected_worker=expected_worker,
            case_name=case_name,
            shared_tokens=shared_tokens,
            deadline=overall_deadline,
        )
        samples.append(sample)
        if sample["seen_monotonic"] >= hold_until:
            break
        if time.monotonic() >= hold_until:
            break
        time.sleep(POLL_S)
    while time.monotonic() < hold_until and time.monotonic() < overall_deadline:
        time.sleep(POLL_S)
    if time.monotonic() > overall_deadline:
        raise AssertionError(
            f"{case_name}: overall deadline exhausted before final hold confirmation"
        )
    final = _observe_query_routes(
        stack,
        cold,
        expected_worker=expected_worker,
        case_name=f"{case_name}-final",
        shared_tokens=shared_tokens,
        deadline=overall_deadline,
    )
    samples.append(final)
    if final["seen_monotonic"] < hold_until:
        raise AssertionError(
            f"{case_name}: final B confirmation at {final['seen_monotonic']} "
            f"before hold_until={hold_until}"
        )
    first = samples[0]["seen_monotonic"]
    held_s = final["seen_monotonic"] - window_start
    if held_s < CLOCK_OBSERVE_S:
        raise AssertionError(
            f"{case_name}: held only {held_s}s from window_start={window_start}, "
            f"need >={CLOCK_OBSERVE_S}s (first_sample={first}, final={final['seen_monotonic']})"
        )
    return {
        "samples": len(samples),
        "window_start": window_start,
        "hold_until": hold_until,
        "first_sample_monotonic": first,
        "final_sample_monotonic": final["seen_monotonic"],
        "held_s": held_s,
        "last": final,
    }


def _register_booking(
    owned_bookings: list[dict[str, Any]],
    *,
    request_id: str,
    owner: Any,
) -> None:
    owned_bookings.append({"request_id": request_id, "owner": owner, "freed": False})


def _admit_booking(
    stack,
    *,
    owner: Any,
    request_id: str,
    tokens: list[int],
    required_taint: str,
    expected_worker: int,
    owned_bookings: list[dict[str, Any]],
    case_name: str,
    deadline: float,
) -> dict[str, Any]:
    from dynamo.llm import RoutingConstraints

    _register_booking(owned_bookings, request_id=request_id, owner=owner)
    begin = time.monotonic()
    timeout_s = _remaining(deadline, f"{case_name}-admit")
    constraints = RoutingConstraints(required_taints={required_taint})
    if owner == "provider":
        selected = stack.provider_best_worker(
            tokens,
            timeout_s=timeout_s,
            request_id=request_id,
            update_indexer=False,
            routing_constraints=constraints,
        )
    else:
        selected = _peer_best_worker(
            stack,
            owner,
            tokens,
            request_id=request_id,
            routing_constraints=constraints,
            timeout_s=timeout_s,
        )
    end = time.monotonic()
    if (selected[0], selected[1]) != (expected_worker, 0):
        raise AssertionError(
            f"{case_name}: booking returned {selected}, expected {(expected_worker, 0)}"
        )
    return {
        "request_id": request_id,
        "worker_id": selected[0],
        "dp_rank": selected[1],
        "begin_monotonic": begin,
        "end_monotonic": end,
        "token_count": len(tokens),
    }


def _free_booking(
    stack,
    owned_bookings: list[dict[str, Any]],
    *,
    request_id: str,
    timeout_s: float,
) -> None:
    for entry in owned_bookings:
        if entry["request_id"] != request_id or entry["freed"]:
            continue
        owner = entry["owner"]
        if owner == "provider":
            stack.provider_free(request_id, timeout_s=timeout_s)
        else:
            _peer_free(stack, owner, request_id, timeout_s=timeout_s)
        entry["freed"] = True
        return
    raise AssertionError(f"booking {request_id!r} was not registered for free")


def _free_owned_bookings(
    stack, owned_bookings: list[dict[str, Any]], *, timeout_s: float
) -> None:
    errors: list[BaseException] = []
    for entry in list(owned_bookings):
        if entry["freed"]:
            continue
        try:
            _free_booking(
                stack,
                owned_bookings,
                request_id=entry["request_id"],
                timeout_s=timeout_s,
            )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
    if errors:
        detail = "; ".join(f"{type(err).__name__}: {err}" for err in errors)
        raise RuntimeError(f"failed freeing owned bookings: {detail}")


_WARM_PREP_ATTEMPTS = 8


def _prepare_warm_dataplane(
    stack,
    *,
    cold: Mapping[str, Any],
    probe_tokens: list[int],
    accounting_tokens: list[int],
    worker_a: int,
    worker_b: int,
    block_size: int,
    preferred_multiplier: float,
    owned_bookings: list[dict[str, Any]],
    owner: Any,
    book_worker: int,
    divert_worker: int,
    required_taint: str,
    case_name: str,
    deadline: float,
) -> list[dict[str, Any]]:
    """Bounded fresh probes. Only WorkerSelectionPending repeats."""
    from tests.router.helper import unique_request_id

    attempts: list[dict[str, Any]] = []
    for attempt in range(1, _WARM_PREP_ATTEMPTS + 1):
        _remaining(deadline, f"{case_name}-prep-{attempt}")
        request_id = unique_request_id(f"{case_name}-prep-{attempt}")
        admit = _admit_booking(
            stack,
            owner=owner,
            request_id=request_id,
            tokens=_synthetic_tokens(
                BACKLOG_TOKENS, salt=hash(case_name) % 1000 + attempt
            ),
            required_taint=required_taint,
            expected_worker=book_worker,
            owned_bookings=owned_bookings,
            case_name=f"{case_name}-prep-{attempt}",
            deadline=deadline,
        )
        loads = _require_sender_booking_loads(
            stack,
            accounting_tokens,
            owner=owner,
            worker_a=worker_a,
            worker_b=worker_b,
            backlog_tokens=BACKLOG_TOKENS,
            booked_worker=book_worker,
            case_name=f"{case_name}-prep-{attempt}-loads",
            timeout_s=_remaining(deadline, f"{case_name}-prep-{attempt}-loads"),
        )
        if book_worker == worker_a:
            _assert_strict_preferred_margin(
                load_a=loads["booked"],
                load_b=loads["idle"],
                accounting_isl=len(accounting_tokens),
                block_size=block_size,
                preferred_multiplier=preferred_multiplier,
                expect_b_wins=True,
                case_name=f"{case_name}-prep-{attempt}-margin",
            )
        pending: str | None = None
        observed: dict[str, Any] | None = None
        try:
            observed = _observe_query_routes(
                stack,
                cold,
                expected_worker=divert_worker,
                case_name=f"{case_name}-prep-{attempt}",
                shared_tokens=probe_tokens,
                deadline=deadline,
            )
        except WorkerSelectionPending as exc:
            pending = f"{type(exc).__name__}: {exc}"
        _free_booking(
            stack,
            owned_bookings,
            request_id=request_id,
            timeout_s=_remaining(deadline, f"{case_name}-prep-{attempt}-free"),
        )
        _wait_for_zero_active(
            stack,
            accounting_tokens,
            owner=owner,
            worker_a=worker_a,
            worker_b=worker_b,
            case_name=f"{case_name}-prep-{attempt}-zero",
            deadline=deadline,
        )
        restored = _wait_for_preferred_worker(
            stack,
            cold,
            expected_worker=worker_a,
            case_name=f"{case_name}-prep-{attempt}-restore",
            shared_tokens=probe_tokens,
            deadline=deadline,
        )
        attempts.append(
            {
                "attempt": attempt,
                "request_id": request_id,
                "admit": admit,
                "loads": loads,
                "observation": observed,
                "pending": pending,
                "zero": "sender-active-zero",
                "restored": restored,
            }
        )
        if pending is None:
            return attempts
    raise AssertionError(
        f"{case_name}: data-plane preparation stayed pending for "
        f"{_WARM_PREP_ATTEMPTS} attempts within the original deadline: {attempts}"
    )


def _warm_handshake(
    stack,
    *,
    cold: Mapping[str, Any],
    probe_tokens: list[int],
    accounting_tokens: list[int],
    worker_a: int,
    worker_b: int,
    block_size: int,
    preferred_multiplier: float,
    owned_bookings: list[dict[str, Any]],
    sender: Any,
    book_worker: int,
    required_taint: str,
    case_name: str,
    deadline: float,
) -> dict[str, Any]:
    """Warm Add → both receivers flip → Free → both restore. Not SourceStarted."""
    from tests.router.helper import unique_request_id

    owner = "provider" if sender == "provider" else sender
    divert_worker = worker_b if book_worker == worker_a else worker_a
    preparation = _prepare_warm_dataplane(
        stack,
        cold=cold,
        probe_tokens=probe_tokens,
        accounting_tokens=accounting_tokens,
        worker_a=worker_a,
        worker_b=worker_b,
        block_size=block_size,
        preferred_multiplier=preferred_multiplier,
        owned_bookings=owned_bookings,
        owner=owner,
        book_worker=book_worker,
        divert_worker=divert_worker,
        required_taint=required_taint,
        case_name=case_name,
        deadline=deadline,
    )
    warm_id = unique_request_id(f"{case_name}-id")
    admit = _admit_booking(
        stack,
        owner=owner,
        request_id=warm_id,
        tokens=_synthetic_tokens(BACKLOG_TOKENS, salt=hash(case_name) % 1000 + 20),
        required_taint=required_taint,
        expected_worker=book_worker,
        owned_bookings=owned_bookings,
        case_name=case_name,
        deadline=deadline,
    )
    loads = _require_sender_booking_loads(
        stack,
        accounting_tokens,
        owner=owner,
        worker_a=worker_a,
        worker_b=worker_b,
        backlog_tokens=BACKLOG_TOKENS,
        booked_worker=book_worker,
        case_name=f"{case_name}-loads",
        timeout_s=_remaining(deadline, f"{case_name}-loads"),
    )
    if book_worker == worker_a:
        _assert_strict_preferred_margin(
            load_a=loads["booked"],
            load_b=loads["idle"],
            accounting_isl=len(accounting_tokens),
            block_size=block_size,
            preferred_multiplier=preferred_multiplier,
            expect_b_wins=True,
            case_name=f"{case_name}-margin",
        )
    diverted = _wait_for_preferred_worker(
        stack,
        cold,
        expected_worker=divert_worker,
        case_name=f"{case_name}-divert",
        shared_tokens=probe_tokens,
        deadline=deadline,
    )
    _free_booking(
        stack,
        owned_bookings,
        request_id=warm_id,
        timeout_s=_remaining(deadline, f"{case_name}-free"),
    )
    _wait_for_zero_active(
        stack,
        accounting_tokens,
        owner=owner,
        worker_a=worker_a,
        worker_b=worker_b,
        case_name=f"{case_name}-zero",
        deadline=deadline,
    )
    restored = _wait_for_preferred_worker(
        stack,
        cold,
        expected_worker=worker_a,
        case_name=f"{case_name}-restore",
        shared_tokens=probe_tokens,
        deadline=deadline,
    )
    return {
        "request_id": warm_id,
        "admit": admit,
        "loads": loads,
        "diverted": diverted,
        "restored": restored,
        "preparation": preparation,
    }


def _provider_cr_name(stack) -> str:
    from tests.router.epp_parity import PROVIDER_CONTAINER_NAME

    return f"{stack.identity.pod_name}-{PROVIDER_CONTAINER_NAME}"


def _peer_cr_name(stack) -> str:
    return f"{stack.identity.pod_name}-{PEER_CONTAINER_NAME}"


def _parse_event_channel(
    channel: Mapping[str, Any], *, case_name: str
) -> dict[str, Any]:
    if channel.get("type") != "EventChannel":
        raise AssertionError(
            f"{case_name}: expected type=EventChannel, got {channel.get('type')!r}"
        )
    topic = channel.get("topic")
    if topic != ACTIVE_SEQUENCES_TOPIC:
        raise AssertionError(
            f"{case_name}: expected topic={ACTIVE_SEQUENCES_TOPIC!r}, got {topic!r}"
        )
    instance_id = channel.get("instance_id")
    if not isinstance(instance_id, int) or isinstance(instance_id, bool):
        raise AssertionError(
            f"{case_name}: EventChannel instance_id malformed: {instance_id!r}"
        )
    scope = channel.get("scope")
    if not isinstance(scope, dict) or scope.get("kind") != "endpoint":
        raise AssertionError(
            f"{case_name}: EventChannel scope is not endpoint: {scope!r}"
        )
    endpoint = scope.get("endpoint")
    if not isinstance(endpoint, dict):
        raise AssertionError(f"{case_name}: EventChannel scope.endpoint missing")
    transport = channel.get("transport")
    if not isinstance(transport, dict) or transport.get("kind") != "Zmq":
        raise AssertionError(
            f"{case_name}: EventChannel transport.kind must be Zmq: {transport!r}"
        )
    config = transport.get("config")
    if not isinstance(config, dict):
        raise AssertionError(
            f"{case_name}: EventChannel transport.config missing: {transport!r}"
        )
    pub_endpoint = config.get("endpoint")
    if not isinstance(pub_endpoint, str) or not pub_endpoint:
        raise AssertionError(
            f"{case_name}: EventChannel transport.config.endpoint missing"
        )
    return {
        "instance_id": instance_id,
        "topic": topic,
        "endpoint": pub_endpoint,
        "scope": {
            "namespace": endpoint.get("namespace"),
            "component": endpoint.get("component"),
            "name": endpoint.get("name"),
        },
    }


def _catalog_owned_event_channel(
    stack, *, cr_name: str, case_name: str
) -> dict[str, Any] | None:
    if stack.kube_api is None or stack.namespace is None:
        raise AssertionError("routing parity stack has not started")
    listed = stack.kube_api.listed_metadata()
    matches: list[dict[str, Any]] = []
    for obj in listed:
        meta = obj.get("metadata") or {}
        name = meta.get("name")
        if name != cr_name:
            continue
        spec = obj.get("spec") or {}
        data = spec.get("data") if isinstance(spec, dict) else None
        if not isinstance(data, dict):
            continue
        channels = data.get("event_channels")
        if not isinstance(channels, dict):
            continue
        for channel in channels.values():
            if not isinstance(channel, dict):
                continue
            if channel.get("topic") != ACTIVE_SEQUENCES_TOPIC:
                continue
            parsed = _parse_event_channel(channel, case_name=case_name)
            if (
                parsed["scope"]["namespace"] != stack.namespace
                or parsed["scope"]["component"] != stack.component_name
                or parsed["scope"]["name"] != GENERATE_ENDPOINT
            ):
                raise AssertionError(
                    f"{case_name}: EventChannel scope mismatch for {cr_name}: "
                    f"{parsed['scope']}"
                )
            matches.append({**parsed, "cr_name": cr_name})
    if not matches:
        return None
    if len(matches) != 1:
        raise AssertionError(
            f"{case_name}: expected one active_sequences_events EventChannel on "
            f"{cr_name}, got {matches}"
        )
    return matches[0]


def _wait_owned_event_channels(
    stack,
    *,
    owners: Mapping[str, str],
    case_name: str,
    deadline: float,
) -> dict[str, dict[str, Any]]:
    """Wait only for missing owned rows; schema/conflict AssertionErrors fail now."""
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        found: dict[str, dict[str, Any]] = {}
        missing = False
        for label, cr_name in owners.items():
            row = _catalog_owned_event_channel(
                stack, cr_name=cr_name, case_name=f"{case_name}-{label}"
            )
            if row is None:
                last[label] = "missing"
                missing = True
                continue
            found[label] = row
        if missing:
            time.sleep(POLL_S)
            continue
        ids = [row["instance_id"] for row in found.values()]
        if len(set(ids)) != len(ids):
            raise AssertionError(f"{case_name}: duplicate owned instance_ids: {found}")
        return found
    raise AssertionError(
        f"{case_name}: owned EventChannels not ready before deadline: last={last}"
    )


def _catalog_scope_event_channel_endpoints(stack, *, case_name: str) -> set[str]:
    """All in-scope active_sequences_events PUB endpoints (provider/peer/mocker/FE/EPP)."""
    if stack.kube_api is None or stack.namespace is None:
        raise AssertionError("routing parity stack has not started")
    endpoints: set[str] = set()
    for obj in stack.kube_api.listed_metadata():
        spec = obj.get("spec") or {}
        data = spec.get("data") if isinstance(spec, dict) else None
        if not isinstance(data, dict):
            continue
        channels = data.get("event_channels")
        if not isinstance(channels, dict):
            continue
        for channel in channels.values():
            if not isinstance(channel, dict):
                continue
            if channel.get("topic") != ACTIVE_SEQUENCES_TOPIC:
                continue
            parsed = _parse_event_channel(channel, case_name=case_name)
            if (
                parsed["scope"]["namespace"] != stack.namespace
                or parsed["scope"]["component"] != stack.component_name
                or parsed["scope"]["name"] != GENERATE_ENDPOINT
            ):
                continue
            endpoints.add(parsed["endpoint"])
    if not endpoints:
        raise AssertionError(
            f"{case_name}: no in-scope active_sequences_events EventChannel endpoints"
        )
    return endpoints


def _owned_process_log_text(process, *, label: str, node_name: str | None) -> str:
    from tests.utils.router_logs import completed_utf8_log_prefix

    if process is None or process.proc is None or process.proc.poll() is not None:
        raise AssertionError(f"{label} process is not live")
    log_path = process.log_path
    if not log_path:
        raise AssertionError(f"{label} process has no log_path")
    path = Path(log_path)
    if node_name and node_name not in path.as_posix():
        raise AssertionError(f"{label} log_path {path} is not node {node_name}")
    if not path.is_file():
        raise AssertionError(f"{label} log is missing: {path}")
    return completed_utf8_log_prefix(path.read_bytes(), label=f"{label} log")


def _parse_structured_fields(line: str) -> dict[str, str]:
    cleaned = ANSI_RE.sub("", line)
    fields: dict[str, str] = {}
    for match in STRUCTURED_FIELD_RE.finditer(cleaned):
        key = match.group(1)
        quoted = match.group(2)
        bare = match.group(3)
        fields[key] = quoted if quoted is not None else bare
    return fields


def _matching_construction_lines(
    text: str,
    marker: str,
    *,
    allowed_endpoints: set[str],
    case_name: str,
) -> list[tuple[str, dict[str, str]]]:
    """Match constructor lines whose endpoint is in the scoped allowed PUB set.

    Endpoints outside the in-scope EventChannel set are unknown sources and fail.
    Primary/peer identity is proven separately via CR-bound payload flips.
    """
    matches: list[tuple[str, dict[str, str]]] = []
    for line in text.splitlines():
        if marker not in line:
            continue
        fields = _parse_structured_fields(line)
        if "topic" not in fields:
            raise AssertionError(
                f"{case_name}: {marker!r} line missing required topic field: {line!r}"
            )
        if fields["topic"] != ACTIVE_SEQUENCES_TOPIC:
            continue
        endpoint = fields.get("endpoint")
        if endpoint is None:
            raise AssertionError(
                f"{case_name}: {marker!r} line missing required endpoint: {line!r}"
            )
        if endpoint not in allowed_endpoints:
            raise AssertionError(
                f"{case_name}: {marker!r} endpoint {endpoint!r} is outside the "
                f"in-scope EventChannel set {sorted(allowed_endpoints)}; line={line!r}"
            )
        matches.append((line, fields))
    return matches


def _observe_construction_events(
    stack,
    *,
    profile: NativeConsumerProfile,
    allowed_endpoints: set[str],
    case_name: str,
) -> dict[str, Any]:
    node = stack.request.node.name
    frontend = stack.frontend_process
    epp = stack.epp_process
    if frontend is None:
        raise AssertionError(f"{case_name}: owned frontend_process was not saved")
    if epp is None:
        raise AssertionError(
            f"{case_name}: owned epp_process was not saved; integration must "
            "retain stack.epp_process from open_consumer_pair"
        )
    if not allowed_endpoints:
        raise AssertionError(f"{case_name}: allowed PUB endpoints are empty")
    fe_text = _owned_process_log_text(frontend, label="frontend", node_name=node)
    epp_text = _owned_process_log_text(epp, label="epp", node_name=node)

    expected_hwm = 1024 if profile.rcvhwm is None else profile.rcvhwm
    expected_pool = (
        1 if profile.endpoints_per_sub is None else profile.endpoints_per_sub
    )
    direct_default = True if profile.direct_zmq is None else profile.direct_zmq

    if not direct_default:
        fe_runtime = _matching_construction_lines(
            fe_text,
            RUNTIME_EVENT,
            allowed_endpoints=allowed_endpoints,
            case_name=f"{case_name}-fe",
        )
        epp_runtime = _matching_construction_lines(
            epp_text,
            RUNTIME_EVENT,
            allowed_endpoints=allowed_endpoints,
            case_name=f"{case_name}-epp",
        )
        if not fe_runtime or not epp_runtime:
            raise AssertionError(
                f"{case_name}: missing {RUNTIME_EVENT!r} for in-scope "
                f"active_sequences_events endpoints; fe={len(fe_runtime)} "
                f"epp={len(epp_runtime)}"
            )
        return {
            "branch": "runtime_subscriber",
            "frontend_events": [line for line, _ in fe_runtime],
            "epp_events": [line for line, _ in epp_runtime],
            "expected_hwm_not_required": True,
            "allowed_endpoints": sorted(allowed_endpoints),
        }

    if expected_pool == 1:
        marker = DEDICATED_EVENT
        branch = "direct_dedicated"
    else:
        marker = GROUPED_EVENT
        branch = "direct_grouped"
    fe_hits = _matching_construction_lines(
        fe_text,
        marker,
        allowed_endpoints=allowed_endpoints,
        case_name=f"{case_name}-fe",
    )
    epp_hits = _matching_construction_lines(
        epp_text,
        marker,
        allowed_endpoints=allowed_endpoints,
        case_name=f"{case_name}-epp",
    )
    if not fe_hits or not epp_hits:
        raise AssertionError(
            f"{case_name}: missing {marker!r} for in-scope topic/endpoints; "
            f"fe={len(fe_hits)} epp={len(epp_hits)}"
        )
    for label, hits in (("frontend", fe_hits), ("epp", epp_hits)):
        hwms: list[int] = []
        for line, fields in hits:
            raw = fields.get("rcvhwm")
            if raw is None:
                raise AssertionError(
                    f"{case_name}: {label} {marker!r} missing rcvhwm: {line!r}"
                )
            try:
                value = int(raw)
            except ValueError as exc:
                raise AssertionError(
                    f"{case_name}: {label} rcvhwm={raw!r} is not an int"
                ) from exc
            hwms.append(value)
        unique = set(hwms)
        if unique != {expected_hwm}:
            raise AssertionError(
                f"{case_name}: {label} rcvhwm values {sorted(unique)} != "
                f"{{{expected_hwm}}}"
            )
    return {
        "branch": branch,
        "frontend_events": [line for line, _ in fe_hits],
        "epp_events": [line for line, _ in epp_hits],
        "rcvhwm": expected_hwm,
        "endpoints_per_sub": expected_pool,
        "allowed_endpoints": sorted(allowed_endpoints),
        "note": (
            "Grouped proves an in-scope Dynamic constructor branch; dual PUB "
            "consumption is evidenced by owned primary/peer route flips, not "
            "requiring the first Dynamic endpoint to equal primary/peer, and "
            "not ceil(N/2)."
        ),
    }


def _open_owned_peer_publisher(
    stack,
    *,
    tmp_path: Path,
    exit_stack: ExitStack,
    deadline: float,
) -> dict[str, Any]:
    """Second spawn-owned native KvRouter peer on the same workers."""
    from tests.router.epp_parity import (
        KV_INDEXER_QUERY_ENDPOINT,
        KV_INDEXER_RECORD_ENDPOINT,
        loopback_process_env,
    )
    from tests.utils.constants import DynamoPortRange
    from tests.utils.port_utils import reserved_ports

    if stack.kube_api is None or stack.kubeconfig is None or stack.namespace is None:
        raise AssertionError("routing parity stack has not started")
    ports = exit_stack.enter_context(reserved_ports(1, DynamoPortRange.ROUTER.value))
    peer_metrics_port = ports[0]
    stack.kube_api.add_ready_container(
        identity=stack.identity,
        dynamo_namespace=stack.namespace,
        container_name=PEER_CONTAINER_NAME,
        http_port=peer_metrics_port,
    )
    peer_env = loopback_process_env(
        stack.kubeconfig, stack.identity, PEER_CONTAINER_NAME
    )
    # Load-only peer: credit 0 / serve false — do not spread hosting provider credit1.
    peer_env.update(native_peer_env_overrides())
    peer_env["DYN_SYSTEM_PORT"] = str(peer_metrics_port)
    config_kwargs = {
        "use_kv_events": stack.indexer_mode != "approximate",
        "serve_indexer": False,
        "use_remote_indexer": True,
        **native_peer_config_overrides(),
        "shared_cache_type": "none",
        "conditional_disagg_enabled": False,
        "router_tracking_hash": "public-xxh3-v1",
        "router_policy_config": None,
        "router_prefill_policy": None,
        "router_decode_policy": None,
        "router_queue_threshold": None,
    }
    construct_budget = _remaining(deadline, "peer-kvrouter")
    owner = NativeRouterOwner.spawn(
        role="peer",
        exit_stack=exit_stack,
        env=peer_env,
        config_kwargs=config_kwargs,
        namespace=stack.namespace,
        component=stack.component_name,
        block_size=stack.worker_block_size,
        num_workers=2,
        construct_timeout_s=construct_budget,
        worker_pids=[],
        query_endpoint=KV_INDEXER_QUERY_ENDPOINT,
        record_endpoint=KV_INDEXER_RECORD_ENDPOINT,
    )
    started = owner.call("initialize", {}, construct_budget)
    if not isinstance(started, dict) or not isinstance(started.get("config_json"), str):
        raise AssertionError("peer native child did not return its serialized config")
    resolved = json.loads(started["config_json"])
    if resolved.get("router_replica_sync") is not True:
        raise AssertionError(
            "peer publisher KvRouterConfig must keep router_replica_sync=true"
        )
    if resolved.get("router_prefill_load_model") != "none":
        raise AssertionError(
            "peer publisher router_prefill_load_model must be exactly 'none'"
        )
    if resolved.get("overlap_score_credit") != CREDIT_ZERO:
        raise AssertionError(
            "peer publisher overlap_score_credit must stay 0.0 (load-only), "
            f"got {resolved.get('overlap_score_credit')!r}"
        )
    if resolved.get("serve_indexer", False) is not False:
        raise AssertionError(
            "peer publisher serve_indexer must be false, "
            f"got {resolved.get('serve_indexer')!r}"
        )
    holder: dict[str, Any] = {
        "owner": owner,
        "config_json": started["config_json"],
        "metrics_port": peer_metrics_port,
        "container_name": PEER_CONTAINER_NAME,
        "tmp_path": str(tmp_path),
    }
    logger.info(
        "opened native peer publisher container=%s system_port=%s child_pid=%s",
        PEER_CONTAINER_NAME,
        peer_metrics_port,
        owner.pid,
    )
    return holder


def _close_peer(peer_stack: ExitStack, peer_holder: dict[str, Any] | None) -> None:
    peer_stack.close()
    if peer_holder is None:
        return
    owner = peer_holder.get("owner")
    if owner is not None and owner.alive:
        raise AssertionError(
            f"peer native owner pid={owner.pid} exitcode={owner.exitcode} "
            "still alive after close"
        )


def _peer_operation(
    peer: Mapping[str, Any],
    op: str,
    payload: dict[str, Any],
    timeout_s: float,
    label: str,
):
    owner = peer.get("owner")
    if owner is None:
        raise AssertionError(f"peer router owner was cleared before {label}")
    try:
        return owner.call(op, payload, timeout_s)
    except NativeRouterFailure as exc:
        if exc.timed_out:
            raise AssertionError(f"{label} outlived the {timeout_s}s budget") from exc
        raise


def _peer_best_worker(
    stack,
    peer: Mapping[str, Any],
    token_ids: list[int],
    *,
    request_id: str,
    routing_constraints,
    timeout_s: float,
):
    del stack
    mode = "omit"
    fields = None
    if routing_constraints is None:
        mode = "none"
    else:
        mode = "value"
        fields = _constraint_payload(routing_constraints)
    return _peer_operation(
        peer,
        "best_worker",
        {
            "token_ids": list(token_ids),
            "kwargs": {
                "request_id": request_id,
                "update_indexer": False,
            },
            "constraints_mode": mode,
            "constraints": fields,
        },
        timeout_s,
        "peer best_worker",
    )


def _peer_free(
    stack, peer: Mapping[str, Any], request_id: str, *, timeout_s: float
) -> None:
    del stack
    _peer_operation(
        peer,
        "free",
        {"request_id": request_id},
        timeout_s,
        "peer free",
    )
