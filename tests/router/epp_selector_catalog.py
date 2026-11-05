# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Named worker-selection catalog: FE provider evidence and stock EPP UnknownType.

Owned by the named-policy catalog slice. Callers supply an already-entered
``EppFrontendRoutingParityStack`` with ``enable_worker_topology=True``. Consumer
overlay kwargs on ``open_consumer_pair`` and retained ``epp_process`` are the
agreed integration API; this module does not launch processes itself.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import struct
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

# Exact inventory row identifiers plus the ordinary observer scope name.
NAMED_POLICY_CATALOG_ROWS: tuple[str, str, str] = (
    "router_prefill_policy",
    "router_decode_policy",
    "worker_selection_policy",
)
NAMED_POLICY_CATALOG_OBSERVER = (
    "named_policy_catalog+fe_admin_router_policy+epp_unknowntype"
)

CACHE_OFF_INSTANCE = "cache_off"
DEFAULT_COST_FN_TYPE = "dynamo-default-cost-fn"
PREFERRED_TAINT_INPUT_WEIGHT = 0.85
ARM_DEADLINE_S = 45.0
NAMED_POLICY_YAML = """\
worker_selection:
  aggregated: default
  instances:
    - name: cache_off
      type: dynamo-default-cost-fn
      parameters:
        overlap_score_credit: 0.0
        router_temperature: 0.0
"""
STOCK_EPP_UNKNOWN_TYPE_DIAGNOSTIC = (
    'unknown worker-selection policy type "dynamo-default-cost-fn"; '
    "linked policy types: <none>"
)

# Closed immutable scope specs: key, repo-relative path, declaration regex.
# Digests are independently reviewed sha256 of comment-stripped declaration+body.


@dataclass(frozen=True)
class NamedPolicyScopeSpec:
    key: str
    rel_path: str
    declaration_regex: str
    label: str


NAMED_POLICY_SCOPE_SPECS: tuple[NamedPolicyScopeSpec, ...] = (
    NamedPolicyScopeSpec(
        "ext_proc_bin_main",
        "deploy/inference-gateway/ext-proc/src/bin/dynamo-ext-proc.rs",
        r"(?:async\s+)?fn\s+main\s*\([^;{]*?\)\s*(?:->[^;{]+)?\s*\{",
        "dynamo-ext-proc main",
    ),
    NamedPolicyScopeSpec(
        "runner_run",
        "deploy/inference-gateway/ext-proc/src/runner.rs",
        r"pub\s+async\s+fn\s+run\s*\(\s*policy_registry\s*:\s*"
        r"Option<WorkerSelectionPolicyRegistry>\s*\)\s*->\s*Result<\(\)>\s*\{",
        "ext-proc runner::run",
    ),
    NamedPolicyScopeSpec(
        "router_plugin_registry_fallback",
        "lib/llm/src/kv_router/plugins.rs",
        r"pub\s+fn\s+router_plugin_registry\s*\(\s*\)\s*->\s*"
        r"RouterPluginRegistry\s*\{",
        "llm::kv_router::plugins::router_plugin_registry",
    ),
    NamedPolicyScopeSpec(
        "selected_worker_selection_policy_instances_from",
        "lib/kv-router/src/scheduling/config.rs",
        r"fn\s+selected_worker_selection_policy_instances_from\s*\([^;{]*?\)\s*"
        r"->\s*Result<WorkerSelectionPolicySelections,\s*"
        r"WorkerSelectionPolicyConfigError>\s*\{",
        "selected_worker_selection_policy_instances_from",
    ),
    NamedPolicyScopeSpec(
        "registry_resolve_selected",
        "lib/kv-router/src/plugins/registry.rs",
        r"fn\s+resolve_selected\s*\([^;{]*?\)\s*->\s*"
        r"Result<Option<WorkerSelectionPolicyFactory>,\s*"
        r"WorkerSelectionPolicyRegistryError>\s*\{",
        "WorkerSelectionPolicyRegistry::resolve_selected",
    ),
    NamedPolicyScopeSpec(
        "preferred_taint_multiplier",
        "lib/kv-router/src/protocols.rs",
        r"pub\s+fn\s+preferred_taint_multiplier\s*\([^;{]*?\)\s*"
        r"->\s*Option<f64>\s*\{",
        "RoutingConstraints::preferred_taint_multiplier",
    ),
)

NAMED_POLICY_SCOPE_DIGESTS: dict[str, str] = {
    "ext_proc_bin_main": (
        "00907d269c362a5389b8108d4df466181db0f6af21e0994be44870aa4a3e91c3"
    ),
    "runner_run": (
        "6d22a6da9425aa16130484abb1df00d07a93fe5e863e5502590ddce3d9bc047f"
    ),
    "router_plugin_registry_fallback": (
        "2c37fe0b1868665985b240f764fbae503b37ceb4db76c282b1d7ce810d1177f5"
    ),
    "selected_worker_selection_policy_instances_from": (
        "95279f9ecb0c3703fff5a9e6c446089e088417ea04fad8005abf2b5abc699289"
    ),
    "registry_resolve_selected": (
        "f8bb13389df5f8538a2495986730d8c87cd55e936d7fca4d138c53581faeddfe"
    ),
    "preferred_taint_multiplier": (
        "dff08a0e830e1f0dfa4d4e3e9127de26927bb85d5caec4e559ffd749079e21fe"
    ),
}


def assert_named_policy_source_contract() -> None:
    """Finite named-source ownership guards for the catalog support boundary.

    Reuses the shared comment-strip / scoped-body extractor. Digests are the
    reviewed sha256 hex of each unique declaration+body blob; empty or invalid
    values fail closed. Also reuses the established input-startup shared
    from_discovery / managed / prefill factory contract.
    """
    from tests.router.epp_config_contract import (
        REPO_ROOT,
        _fn_decl_and_body,
        _strip_rust_comments,
        assert_input_startup_capability_source_contracts,
    )

    assert_input_startup_capability_source_contracts()

    keys = [spec.key for spec in NAMED_POLICY_SCOPE_SPECS]
    if len(set(keys)) != len(keys):
        raise AssertionError(f"duplicate NAMED_POLICY_SCOPE_SPECS keys: {keys}")
    digest_keys = set(NAMED_POLICY_SCOPE_DIGESTS)
    scope_keys = set(keys)
    if digest_keys != scope_keys:
        raise AssertionError(
            "NAMED_POLICY_SCOPE_DIGESTS key set must equal scope specs; "
            f"missing={sorted(scope_keys - digest_keys)} "
            f"extra={sorted(digest_keys - scope_keys)}"
        )

    file_text: dict[Path, str] = {}
    for spec in NAMED_POLICY_SCOPE_SPECS:
        path = REPO_ROOT / spec.rel_path
        if not path.is_file():
            raise AssertionError(
                f"missing fingerprint source {spec.label}: {path}"
            )
        if path not in file_text:
            file_text[path] = _strip_rust_comments(path.read_text(encoding="utf-8"))
        blob = _fn_decl_and_body(
            file_text[path],
            re.compile(spec.declaration_regex),
            spec.label,
        )
        actual = hashlib.sha256(blob.encode("utf-8")).hexdigest()
        expected = NAMED_POLICY_SCOPE_DIGESTS[spec.key]
        if not re.fullmatch(r"[0-9a-f]{64}", expected or ""):
            raise AssertionError(
                f"NAMED_POLICY_SCOPE_DIGESTS[{spec.key!r}] is empty or not a "
                f"sha256 hex digest for {spec.label}: expected={expected!r} "
                f"actual={actual}"
            )
        if actual != expected:
            raise AssertionError(
                f"fingerprint mismatch for {spec.key}: scope={spec.label} "
                f"actual={actual} expected={expected}"
            )

    # Semantic tripwires on the same extracted scopes (not a whole-Rust parser).
    main_body = _fn_decl_and_body(
        file_text[REPO_ROOT / NAMED_POLICY_SCOPE_SPECS[0].rel_path],
        re.compile(NAMED_POLICY_SCOPE_SPECS[0].declaration_regex),
        NAMED_POLICY_SCOPE_SPECS[0].label,
    )
    if not re.search(r"dynamo_ext_proc\s*::\s*run\s*\(\s*None\s*\)", main_body):
        raise AssertionError(
            "stock Runtime EPP binary no longer calls run(None) as its sole "
            "linked-registry handoff"
        )

    run_body = _fn_decl_and_body(
        file_text[REPO_ROOT / "deploy/inference-gateway/ext-proc/src/runner.rs"],
        re.compile(NAMED_POLICY_SCOPE_SPECS[1].declaration_regex),
        NAMED_POLICY_SCOPE_SPECS[1].label,
    )
    if "linked worker-selection policies require DYN_EPP_MODE=standalone" not in run_body:
        raise AssertionError(
            "runner::run no longer rejects a nonempty linked registry outside "
            "standalone mode"
        )
    if not re.search(
        r"policy_registry\s*\.\s*unwrap_or_default\s*\(\s*\)", run_body
    ):
        raise AssertionError(
            "runner::run no longer falls back to an empty Default registry "
            "when the binary supplies None"
        )

    registry_body = _fn_decl_and_body(
        file_text[REPO_ROOT / "lib/llm/src/kv_router/plugins.rs"],
        re.compile(NAMED_POLICY_SCOPE_SPECS[2].declaration_regex),
        NAMED_POLICY_SCOPE_SPECS[2].label,
    )
    if "dynamo_custom_policy_builtin::default_registry" not in registry_body:
        raise AssertionError(
            "uninstalled host registry fallback no longer uses "
            "dynamo_custom_policy_builtin::default_registry"
        )

    selections_body = _fn_decl_and_body(
        file_text[REPO_ROOT / "lib/kv-router/src/scheduling/config.rs"],
        re.compile(NAMED_POLICY_SCOPE_SPECS[3].declaration_regex),
        NAMED_POLICY_SCOPE_SPECS[3].label,
    )
    for stage, field in (
        ("prefill", "router_prefill_policy"),
        ("decode", "router_decode_policy"),
    ):
        pattern = (
            rf"let\s+{stage}\s*=\s*normalized\(\s*self\.{field}\.as_deref\(\)\s*\)"
            rf"\s*\.or_else\(\s*\|\|\s*global\.clone\(\)\s*\)"
            rf"\s*\.or_else\("
        )
        if re.search(pattern, selections_body, flags=re.DOTALL) is None:
            raise AssertionError(
                f"{stage} selector precedence is no longer "
                f"explicit-stage > global env > YAML"
            )
    if not re.search(
        r"let\s+aggregated\s*=\s*global\.clone\(\)\s*\.or\(\s*yaml_aggregated\s*\)",
        selections_body,
    ):
        raise AssertionError(
            "aggregated selection no longer applies global env over YAML"
        )
    if 'name != "default"' not in selections_body:
        raise AssertionError(
            "reserved selection default is no longer stripped before custom "
            "instance resolution"
        )

    resolve_body = _fn_decl_and_body(
        file_text[REPO_ROOT / "lib/kv-router/src/plugins/registry.rs"],
        re.compile(NAMED_POLICY_SCOPE_SPECS[4].declaration_regex),
        NAMED_POLICY_SCOPE_SPECS[4].label,
    )
    if "UnknownType" not in resolve_body:
        raise AssertionError(
            "resolve_selected no longer maps a missing provider type to UnknownType"
        )
    if "UnknownInstance" not in resolve_body:
        raise AssertionError(
            "resolve_selected no longer maps a missing instance to UnknownInstance"
        )
    registry_text = file_text[REPO_ROOT / "lib/kv-router/src/plugins/registry.rs"]
    if (
        "unknown worker-selection policy type {name:?}; linked policy types: {available}"
        not in registry_text
    ):
        raise AssertionError(
            "WorkerSelectionPolicyRegistryError::UnknownType Display spelling drifted"
        )
    available_body = _fn_decl_and_body(
        registry_text,
        re.compile(r"fn\s+available_policy_types\s*\(\s*&self\s*\)\s*->\s*String\s*\{"),
        "available_policy_types",
    )
    if '"<none>"' not in available_body:
        raise AssertionError(
            "empty linked policy catalog no longer reports available=<none>"
        )

    taint_body = _fn_decl_and_body(
        file_text[REPO_ROOT / "lib/kv-router/src/protocols.rs"],
        re.compile(NAMED_POLICY_SCOPE_SPECS[5].declaration_regex),
        NAMED_POLICY_SCOPE_SPECS[5].label,
    )
    collapsed_taint = re.sub(r"\s+", "", taint_body)
    if ".tanh()" not in taint_body:
        raise AssertionError(
            "preferred_taint_multiplier no longer applies tanh to the summed bias"
        )
    if "(-bias).exp()" not in collapsed_taint:
        raise AssertionError(
            "preferred_taint_multiplier no longer returns exp(-bias) as the "
            "cost multiplier"
        )
    if "f64::from" not in taint_body:
        raise AssertionError(
            "preferred_taint_multiplier no longer widens matched f32 weights "
            "through f64::from"
        )

    # FE Python registrar + custom-policy feature / shipped types (finite paths).
    cargo = (REPO_ROOT / "lib/bindings/python/Cargo.toml").read_text(encoding="utf-8")
    if 'default = ["custom-policy"]' not in cargo and "default = ['custom-policy']" not in cargo:
        raise AssertionError(
            "Python bindings Cargo.toml default features no longer enable "
            "custom-policy"
        )
    bindings = _strip_rust_comments(
        (REPO_ROOT / "lib/bindings/python/rust/lib.rs").read_text(encoding="utf-8")
    )
    register_body = _fn_decl_and_body(
        bindings,
        re.compile(
            r"fn\s+register_core_with_router_plugins\s*\([^;{]*?\)\s*"
            r"->\s*PyResult<\(\)>\s*\{"
        ),
        "register_core_with_router_plugins",
    )
    if "dynamo_custom_policy_builtin::register" not in register_body:
        raise AssertionError(
            "Python custom-policy startup no longer registers shipped builtin "
            "providers"
        )
    if "install_router_plugin_registry" not in register_body:
        raise AssertionError(
            "Python custom-policy startup no longer installs the host registry"
        )
    default_params = (
        REPO_ROOT / "lib/router-plugins/builtin/src/default/parameters.rs"
    ).read_text(encoding="utf-8")
    if f'"{DEFAULT_COST_FN_TYPE}"' not in default_params:
        raise AssertionError(
            f"default provider no longer registers type {DEFAULT_COST_FN_TYPE!r}"
        )

    ws_path = REPO_ROOT / "lib/kv-router/src/plugins/worker_selection/config.rs"
    ws_text = _strip_rust_comments(ws_path.read_text(encoding="utf-8"))
    resolve_cfg = _fn_decl_and_body(
        ws_text,
        re.compile(
            r"pub\(crate\)\s+fn\s+resolve\s*\(\s*self\s*\)\s*->\s*"
            r"Result<WorkerSelectionConfig,\s*RouterPolicyConfigError>\s*\{"
        ),
        "RawWorkerSelectionConfig::resolve",
    )
    if "instance name 'default' is reserved" not in resolve_cfg:
        raise AssertionError(
            "worker_selection YAML no longer rejects reserved instance name default"
        )

    if NAMED_POLICY_CATALOG_ROWS != (
        "router_prefill_policy",
        "router_decode_policy",
        "worker_selection_policy",
    ):
        raise AssertionError(
            "NAMED_POLICY_CATALOG_ROWS drifted from the three owned setting ids"
        )


def run_named_policy_catalog(stack, *, tmp_path) -> dict[str, Any]:
    """Run the minimum global named-default provider catalog on a live stack.

    Requires an already-entered topology-enabled routing parity stack. Uses the
    agreed ``open_consumer_pair`` consumer overlays and retained ``epp_process``.
    Source contract runs before any live arm. Stack construction and
    ``open_consumer_pair`` bootstrap remain under existing stack APIs; each
    arm's 45s observation deadline starts immediately after a successful open.
    """
    from dynamo.llm import RoutingConstraints
    from tests.router.common import (
        _assert_catalog_valid_route,
        _assert_provider_snapshot_unchanged,
        _prepare_first_selection_snapshot,
        _provider_cache_snapshot,
        _require_cache_sensitive_policy,
        _with_nvext,
        assert_ordered_token_ids_match,
        assert_route_observations_differ,
        frontend_epp_first_selection_cases,
    )
    from tests.router.epp_parity import (
        READINESS_DEADLINE_S,
        ZONE_A_TAINT,
        ZONE_B_TAINT,
    )
    from tests.router.helper import unique_request_id
    from tests.router.router_process import observe_owned_initialization_diagnostic
    from tests.utils.constants import DynamoPortRange
    from tests.utils.port_utils import reserved_ports

    if ARM_DEADLINE_S != READINESS_DEADLINE_S:
        raise AssertionError(
            f"ARM_DEADLINE_S={ARM_DEADLINE_S} must match READINESS_DEADLINE_S="
            f"{READINESS_DEADLINE_S}"
        )

    assert_named_policy_source_contract()

    if stack.enable_worker_topology is not True:
        raise AssertionError(
            "named-policy catalog requires enable_worker_topology=True so "
            f"{ZONE_A_TAINT}/{ZONE_B_TAINT} are published"
        )
    if stack.provider_router is None:
        raise AssertionError("named-policy catalog requires a served provider KvRouter")
    if stack.namespace is None:
        raise AssertionError("named-policy catalog requires an owned stack namespace")

    policy = _require_cache_sensitive_policy(stack)
    cases = frontend_epp_first_selection_cases(stack.model_name)
    if len(cases) != 2 or {case.label for case in cases} != {"A", "B"}:
        raise AssertionError("named-policy catalog requires opposite A/B cases")
    case_a = next(case for case in cases if case.label == "A")
    worker_a = stack.worker_ids["A"]
    worker_b = stack.worker_ids["B"]
    taints_a = stack.worker_configs[worker_a]["taints"]
    taints_b = stack.worker_configs[worker_b]["taints"]
    if taints_a & {ZONE_A_TAINT} != {ZONE_A_TAINT}:
        raise AssertionError(
            f"worker A {worker_a} missing published {ZONE_A_TAINT}: "
            f"{stack.worker_configs[worker_a]}"
        )
    if ZONE_B_TAINT in taints_a:
        raise AssertionError(
            f"worker A {worker_a} must not publish {ZONE_B_TAINT}: "
            f"{stack.worker_configs[worker_a]}"
        )
    if taints_b & {ZONE_B_TAINT} != {ZONE_B_TAINT}:
        raise AssertionError(
            f"worker B {worker_b} missing published {ZONE_B_TAINT}: "
            f"{stack.worker_configs[worker_b]}"
        )
    if ZONE_A_TAINT in taints_b:
        raise AssertionError(
            f"worker B {worker_b} must not publish {ZONE_A_TAINT}: "
            f"{stack.worker_configs[worker_b]}"
        )

    prefer_b = RoutingConstraints(
        preferred_taints={ZONE_B_TAINT: PREFERRED_TAINT_INPUT_WEIGHT}
    )
    native_weight = prefer_b.preferred_taints.get(ZONE_B_TAINT)
    if native_weight is None:
        raise AssertionError(
            "native preferred_taints map lost zone-b after construction: "
            f"{prefer_b.preferred_taints}"
        )
    roundtripped = _f32_bits(float(native_weight))
    expected_input = _f32_bits(PREFERRED_TAINT_INPUT_WEIGHT)
    if roundtripped != expected_input:
        raise AssertionError(
            "native preferred_taints weight round-trip drifted from input 0.85 "
            f"f32: got={roundtripped!r} expected={expected_input!r} "
            f"map={prefer_b.preferred_taints}"
        )
    multiplier_b = _preferred_taint_multiplier_from_weights((roundtripped,))
    multiplier_a = 1.0  # zone-b preference does not match worker A's taints
    if not (0.0 < multiplier_b < 1.0):
        raise AssertionError(
            "source-backed preferred-taint multiplier for input 0.85 must be in "
            f"(0,1), got {multiplier_b}"
        )

    policy_path = Path(tmp_path) / "named-policy-catalog.yaml"
    policy_path.write_text(NAMED_POLICY_YAML, encoding="utf-8")
    if not policy_path.is_file():
        raise AssertionError(f"failed to write owned policy YAML at {policy_path}")

    # Provider warm/gate uses the shared readiness budget on the stack API.
    _prepare_first_selection_snapshot(stack, cases)
    tokens = list(stack.token_ids_by_label["A"])
    if not tokens:
        raise AssertionError("case A tokenization is empty")
    payload = _with_nvext(
        case_a.payload,
        routing_constraints={
            "preferred_taints": {ZONE_B_TAINT: PREFERRED_TAINT_INPUT_WEIGHT}
        },
    )
    path = case_a.path

    margin_deadline = time.monotonic() + ARM_DEADLINE_S
    margin = _gate_strict_default_cost_margin(
        stack,
        tokens,
        worker_a=worker_a,
        worker_b=worker_b,
        policy=policy,
        multiplier_a=multiplier_a,
        multiplier_b=multiplier_b,
        deadline=margin_deadline,
    )
    frozen_events = _provider_cache_snapshot(
        stack,
        timeout_s=_remaining(margin_deadline, label="margin-cache-snapshot"),
    )
    _require_deadline_alive(margin_deadline, label="margin-cache-snapshot")
    evidence: dict[str, Any] = {
        "policy_path": str(policy_path.resolve()),
        "worker_ids": dict(stack.worker_ids),
        "cost_margin": margin,
        "preferred_taint_input_weight": PREFERRED_TAINT_INPUT_WEIGHT,
        "preferred_taint_multiplier_b": multiplier_b,
        "observer": NAMED_POLICY_CATALOG_OBSERVER,
        "rows": list(NAMED_POLICY_CATALOG_ROWS),
    }

    default_env = _consumer_policy_env(policy_path, selection="default")
    named_env = _consumer_policy_env(policy_path, selection=CACHE_OFF_INSTANCE)
    remove_env = _consumer_policy_env(policy_path, selection=None)

    # 1) Explicit global default: FE + stock EPP query A; FE generation worker/rank.
    stack.open_consumer_pair(
        include_epp=True,
        frontend_env_overrides=default_env,
        epp_env_overrides=default_env,
        epp_require_readiness=True,
    )
    try:
        default_deadline = time.monotonic() + ARM_DEADLINE_S
        _require_quiescent_loads(
            stack,
            tokens,
            worker_a=worker_a,
            worker_b=worker_b,
            arm="default-pre",
            deadline=default_deadline,
        )
        default_fe = stack.collect_frontend_route(
            payload,
            path=path,
            request_id=unique_request_id("fe-named-default"),
            timeout_s=_remaining(default_deadline, label="default-fe-query"),
        )
        _require_deadline_alive(default_deadline, label="default-fe-query")
        default_epp = stack.collect_epp_route(
            payload,
            path=path,
            request_id=unique_request_id("epp-named-default"),
            timeout_s=_remaining(default_deadline, label="default-epp-query"),
        )
        _require_deadline_alive(default_deadline, label="default-epp-query")
        _assert_catalog_valid_route(default_fe, stack.worker_ids, "default-fe")
        _assert_catalog_valid_route(default_epp, stack.worker_ids, "default-epp")
        if default_fe.worker_id != worker_a or default_fe.dp_rank != 0:
            raise AssertionError(
                "default FE selection "
                f"{default_fe.worker_id}/{default_fe.dp_rank} != A {worker_a}/0"
            )
        if default_epp.worker_id != worker_a or default_epp.dp_rank != 0:
            raise AssertionError(
                "default EPP selection "
                f"{default_epp.worker_id}/{default_epp.dp_rank} != A {worker_a}/0"
            )
        assert_ordered_token_ids_match(
            default_fe.token_ids,
            tokens,
            left_label="default.fe.token_ids",
            right_label="prepare.token_ids",
            case_name="default-fe-tokens",
        )
        assert_ordered_token_ids_match(
            default_epp.token_ids,
            default_fe.token_ids,
            left_label="default.epp.token_ids",
            right_label="default.fe.token_ids",
            case_name="default-epp-tokens",
        )
        current_default = _provider_cache_snapshot(
            stack,
            timeout_s=_remaining(default_deadline, label="default-cache-snapshot"),
        )
        _require_deadline_alive(default_deadline, label="default-cache-snapshot")
        _assert_provider_snapshot_unchanged(
            frozen_events, current_default, "default-query"
        )
        # Generation proves completed DONE + typed worker/rank; input tokens are
        # asserted only on FE/EPP query surfaces above.
        default_gen = stack.collect_frontend_generation(
            payload,
            path=path,
            request_id=unique_request_id("fe-named-default-gen"),
            timeout_s=_remaining(default_deadline, label="default-fe-generation"),
        )
        _require_deadline_alive(default_deadline, label="default-fe-generation")
        if default_gen.worker_id != worker_a or default_gen.dp_rank != 0:
            raise AssertionError(
                "default FE generation "
                f"{default_gen.worker_id}/{default_gen.dp_rank} != A {worker_a}/0"
            )
        evidence["default"] = {
            "worker_id": default_fe.worker_id,
            "dp_rank": default_fe.dp_rank,
            "token_ids": list(default_fe.token_ids),
            "generation_worker_id": default_gen.worker_id,
            "generation_dp_rank": default_gen.dp_rank,
        }
        logger.info(
            "named-policy default selectedworker=%s rank=%s tokens=%s "
            "generation_worker=%s generation_rank=%s",
            default_fe.worker_id,
            default_fe.dp_rank,
            len(default_fe.token_ids),
            default_gen.worker_id,
            default_gen.dp_rank,
        )
    finally:
        stack.close_consumer_pair()

    # Generation on A can mutate provider cache; re-gate before named queries.
    remargin_deadline = time.monotonic() + ARM_DEADLINE_S
    _gate_strict_default_cost_margin(
        stack,
        tokens,
        worker_a=worker_a,
        worker_b=worker_b,
        policy=policy,
        multiplier_a=multiplier_a,
        multiplier_b=multiplier_b,
        deadline=remargin_deadline,
    )
    frozen_events = _provider_cache_snapshot(
        stack,
        timeout_s=_remaining(remargin_deadline, label="remargin-cache-snapshot"),
    )
    _require_deadline_alive(remargin_deadline, label="remargin-cache-snapshot")

    with reserved_ports(1, DynamoPortRange.FRONTEND) as (admin_port,):
        admin_env = {
            **named_env,
            "DYN_ENABLE_FRONTEND_ADMIN_API": "1",
            "DYN_FRONTEND_ADMIN_HOST": "127.0.0.1",
            "DYN_FRONTEND_ADMIN_PORT": str(admin_port),
        }
        # 2) Named FE: admin revision0/credit0 + query selects B.
        stack.open_consumer_pair(
            include_epp=False,
            frontend_env_overrides=admin_env,
            epp_require_readiness=True,
        )
        try:
            named_deadline = time.monotonic() + ARM_DEADLINE_S
            _require_quiescent_loads(
                stack,
                tokens,
                worker_a=worker_a,
                worker_b=worker_b,
                arm="named-pre",
                deadline=named_deadline,
            )
            admin = _read_owned_router_policy(
                admin_port,
                model=stack.model_name,
                namespace=stack.namespace,
                component=stack.component_name,
                endpoint="generate",
                expected_credit=0.0,
                expected_temperature=0.0,
                deadline=named_deadline,
            )
            named_fe = stack.collect_frontend_route(
                payload,
                path=path,
                request_id=unique_request_id("fe-named-cache-off"),
                timeout_s=_remaining(named_deadline, label="named-fe-query"),
            )
            _require_deadline_alive(named_deadline, label="named-fe-query")
            _assert_catalog_valid_route(named_fe, stack.worker_ids, "named-fe")
            if named_fe.worker_id != worker_b or named_fe.dp_rank != 0:
                raise AssertionError(
                    "named FE selection "
                    f"{named_fe.worker_id}/{named_fe.dp_rank} != B {worker_b}/0"
                )
            assert_ordered_token_ids_match(
                named_fe.token_ids,
                tokens,
                left_label="named.fe.token_ids",
                right_label="prepare.token_ids",
                case_name="named-fe-tokens",
            )
            current_named = _provider_cache_snapshot(
                stack,
                timeout_s=_remaining(named_deadline, label="named-cache-snapshot"),
            )
            _require_deadline_alive(named_deadline, label="named-cache-snapshot")
            _assert_provider_snapshot_unchanged(
                frozen_events, current_named, "named-query"
            )
            evidence["named_query"] = {
                "worker_id": named_fe.worker_id,
                "dp_rank": named_fe.dp_rank,
                "token_ids": list(named_fe.token_ids),
                "admin": admin,
            }
            logger.info(
                "named-policy named selectedworker=%s rank=%s tokens=%s "
                "admin=%s multiplier_b=%s",
                named_fe.worker_id,
                named_fe.dp_rank,
                len(named_fe.token_ids),
                admin,
                multiplier_b,
            )
        finally:
            stack.close_consumer_pair()

        # 3) Remove global selector; YAML instance remains; default A again.
        stack.open_consumer_pair(
            include_epp=False,
            frontend_env_overrides=remove_env,
            epp_require_readiness=True,
        )
        try:
            remove_deadline = time.monotonic() + ARM_DEADLINE_S
            _require_quiescent_loads(
                stack,
                tokens,
                worker_a=worker_a,
                worker_b=worker_b,
                arm="remove-pre",
                deadline=remove_deadline,
            )
            removed_fe = stack.collect_frontend_route(
                payload,
                path=path,
                request_id=unique_request_id("fe-named-remove-selector"),
                timeout_s=_remaining(remove_deadline, label="remove-fe-query"),
            )
            _require_deadline_alive(remove_deadline, label="remove-fe-query")
            _assert_catalog_valid_route(removed_fe, stack.worker_ids, "remove-fe")
            if removed_fe.worker_id != worker_a or removed_fe.dp_rank != 0:
                raise AssertionError(
                    "remove-selector FE selection "
                    f"{removed_fe.worker_id}/{removed_fe.dp_rank} != A {worker_a}/0"
                )
            assert_ordered_token_ids_match(
                removed_fe.token_ids,
                tokens,
                left_label="remove.fe.token_ids",
                right_label="prepare.token_ids",
                case_name="remove-fe-tokens",
            )
            current_remove = _provider_cache_snapshot(
                stack,
                timeout_s=_remaining(remove_deadline, label="remove-cache-snapshot"),
            )
            _require_deadline_alive(remove_deadline, label="remove-cache-snapshot")
            _assert_provider_snapshot_unchanged(
                frozen_events, current_remove, "remove-selector"
            )
            evidence["remove_selector"] = {
                "worker_id": removed_fe.worker_id,
                "dp_rank": removed_fe.dp_rank,
            }
        finally:
            stack.close_consumer_pair()

        # 4) Live comparator negative: default A vs named B, same query tokens.
        assert_ordered_token_ids_match(
            evidence["default"]["token_ids"],
            evidence["named_query"]["token_ids"],
            left_label="default.token_ids",
            right_label="named.token_ids",
            case_name="named-policy-live-tokens",
        )
        assert_route_observations_differ(
            _RouteView(
                worker_id=evidence["default"]["worker_id"],
                dp_rank=evidence["default"]["dp_rank"],
            ),
            _RouteView(
                worker_id=evidence["named_query"]["worker_id"],
                dp_rank=evidence["named_query"]["dp_rank"],
            ),
            left_label="frontend.default.A",
            right_label="frontend.named.B",
            case_name="named-policy-live-negative",
        )

        # 5) Named FE real generation on B last (warms B; after all query controls).
        stack.open_consumer_pair(
            include_epp=False,
            frontend_env_overrides=named_env,
            epp_require_readiness=True,
        )
        try:
            named_gen_deadline = time.monotonic() + ARM_DEADLINE_S
            _require_quiescent_loads(
                stack,
                tokens,
                worker_a=worker_a,
                worker_b=worker_b,
                arm="named-gen-pre",
                deadline=named_gen_deadline,
            )
            named_gen = stack.collect_frontend_generation(
                payload,
                path=path,
                request_id=unique_request_id("fe-named-cache-off-gen"),
                timeout_s=_remaining(
                    named_gen_deadline, label="named-fe-generation"
                ),
            )
            _require_deadline_alive(named_gen_deadline, label="named-fe-generation")
            if named_gen.worker_id != worker_b or named_gen.dp_rank != 0:
                raise AssertionError(
                    "named FE generation "
                    f"{named_gen.worker_id}/{named_gen.dp_rank} != B {worker_b}/0"
                )
            evidence["named_generation"] = {
                "worker_id": named_gen.worker_id,
                "dp_rank": named_gen.dp_rank,
            }
            logger.info(
                "named-policy generationB selectedworker=%s rank=%s",
                named_gen.worker_id,
                named_gen.dp_rank,
            )
        finally:
            stack.close_consumer_pair()

    # 6) Stock Runtime EPP rejects the same valid named instance/type.
    stack.open_consumer_pair(
        include_epp=True,
        frontend_env_overrides=named_env,
        epp_env_overrides=named_env,
        epp_require_readiness=False,
    )
    try:
        epp_deadline = time.monotonic() + ARM_DEADLINE_S
        epp_process = stack.epp_process
        if epp_process is None:
            raise AssertionError(
                "owned epp_process was not retained after open_consumer_pair"
            )
        if epp_process.proc is None:
            raise AssertionError("owned epp_process.proc was not started")
        pid = epp_process.proc.pid
        if not isinstance(pid, int) or pid <= 0:
            raise AssertionError(f"owned epp_process pid is invalid: {pid!r}")
        if set(stack.worker_ids) != {"A", "B"}:
            raise AssertionError(
                "owned catalog labels are not exactly A/B before EPP diagnostic: "
                f"{stack.worker_ids!r}"
            )
        if not stack.model_name:
            raise AssertionError(
                "owned model_name missing before interpreting EPP UnknownType"
            )
        if STOCK_EPP_UNKNOWN_TYPE_DIAGNOSTIC != (
            'unknown worker-selection policy type "dynamo-default-cost-fn"; '
            "linked policy types: <none>"
        ):
            raise AssertionError(
                "stock EPP UnknownType diagnostic constant drifted from source"
            )
        observe_s = _remaining(epp_deadline, label="epp-unknowntype-observe")
        pid, log = observe_owned_initialization_diagnostic(
            epp_process,
            diagnostic=STOCK_EPP_UNKNOWN_TYPE_DIAGNOSTIC,
            deadline_s=observe_s,
            node_name=stack.request.node.name,
            label="epp_process",
        )
        _require_deadline_alive(epp_deadline, label="epp-unknowntype-observe")
        evidence["epp_rejection"] = {
            "pid": pid,
            "log": str(log),
            "diagnostic": STOCK_EPP_UNKNOWN_TYPE_DIAGNOSTIC,
        }
        logger.info(
            "named-policy owneddiagnostic pid=%s log=%s diagnostic=%s",
            pid,
            log,
            STOCK_EPP_UNKNOWN_TYPE_DIAGNOSTIC,
        )
    finally:
        stack.close_consumer_pair()

    if stack.kube_api is not None and stack.kube_api.failures:
        raise AssertionError(
            "Loopback Kubernetes API recorded unexpected requests: "
            + "; ".join(stack.kube_api.failures)
        )
    return evidence


class _RouteView:
    """Minimal route observation for the shared (worker_id, dp_rank) comparator."""

    __slots__ = ("worker_id", "dp_rank")

    def __init__(self, *, worker_id: int, dp_rank: int) -> None:
        self.worker_id = worker_id
        self.dp_rank = dp_rank


def _consumer_policy_env(
    policy_path: Path, *, selection: str | None
) -> dict[str, str | None]:
    """Child-only overlay: YAML path plus global selector; stages stay cleared."""
    return {
        "DYN_ROUTER_POLICY_CONFIG": str(policy_path.resolve()),
        "DYN_ROUTER_WORKER_SELECTION_POLICY": selection,
        "DYN_ROUTER_PREFILL_POLICY": None,
        "DYN_ROUTER_DECODE_POLICY": None,
    }


def _f32_bits(value: float) -> float:
    """Round-trip through IEEE-754 binary32, matching Rust f32 storage."""
    return float(struct.unpack("=f", struct.pack("=f", float(value)))[0])


def _preferred_taint_multiplier_from_weights(weights: tuple[float, ...]) -> float:
    """Source-backed cost multiplier: exp(-tanh(sum(f32→f64 weights)))."""
    if not weights:
        raise AssertionError("preferred-taint multiplier requires at least one weight")
    bias = math.tanh(sum(float(weight) for weight in weights))
    return math.exp(-bias)


def _remaining(deadline: float, *, label: str) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise AssertionError(
            f"{label} exhausted the owned {ARM_DEADLINE_S}s observation deadline"
        )
    return remaining


def _require_deadline_alive(deadline: float, *, label: str) -> None:
    if time.monotonic() >= deadline:
        raise AssertionError(
            f"{label} completed after the owned {ARM_DEADLINE_S}s observation "
            "deadline expired"
        )


def _poll_sleep(deadline: float) -> None:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return
    time.sleep(min(0.05, remaining))


def _projected_default_cost(
    *,
    isl_tokens: int,
    block_size: int,
    device_blocks: float,
    potential_decode_blocks: float,
    active_requests: int,
    overlap_score_credit: float,
    prefill_load_scale: float,
    decode_active_request_weight: float,
    preferred_taint_multiplier: float,
) -> float:
    """Match default aggregated scoring for a quiescent available worker."""
    if isl_tokens <= 0:
        raise AssertionError(f"isl_tokens {isl_tokens} must be positive")
    if block_size <= 0:
        raise AssertionError(f"block_size {block_size} must be positive")
    if not math.isfinite(preferred_taint_multiplier) or preferred_taint_multiplier <= 0:
        raise AssertionError(
            f"preferred_taint_multiplier {preferred_taint_multiplier!r} must be "
            "finite and positive"
        )
    # active_prefill=0 when active_requests gated to 0; raw tokens collapse to ISL.
    raw_blocks = isl_tokens / float(block_size)
    credit = overlap_score_credit * float(device_blocks)
    prefill = max(raw_blocks - credit, 0.0)
    logit = (
        prefill_load_scale * prefill
        + float(potential_decode_blocks)
        + decode_active_request_weight * float(active_requests)
    )
    return logit * preferred_taint_multiplier


def _gate_strict_default_cost_margin(
    stack,
    tokens: list[int],
    *,
    worker_a: int,
    worker_b: int,
    policy: Mapping[str, float],
    multiplier_a: float,
    multiplier_b: float,
    deadline: float,
) -> dict[str, Any]:
    from tests.router.common import _dp0_worker_load_rows, _overlap_blocks

    block_size = stack.worker_block_size
    if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size <= 0:
        raise AssertionError(f"worker_block_size {block_size!r} is invalid")
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        remaining = _remaining(deadline, label="named-cost-margin-poll")
        scores = stack.provider_overlap_scores(tokens, timeout_s=remaining)
        _require_deadline_alive(deadline, label="named-cost-margin-overlap")
        remaining = _remaining(deadline, label="named-cost-margin-loads")
        loads = stack.provider_potential_loads(tokens, timeout_s=remaining)
        _require_deadline_alive(deadline, label="named-cost-margin-loads")
        blocks = _overlap_blocks(scores)
        rows = _dp0_worker_load_rows(loads, worker_a, worker_b, "named-cost-margin")
        last = {"blocks": blocks, "loads": loads, "rows": rows}
        if rows is None:
            _poll_sleep(deadline)
            continue
        load_a = rows[(worker_a, 0)]
        load_b = rows[(worker_b, 0)]
        blocks_a = float(blocks.get((worker_a, 0), 0))
        blocks_b = float(blocks.get((worker_b, 0), 0))
        if blocks_a <= blocks_b:
            _poll_sleep(deadline)
            continue
        if load_a["active_requests"] != 0 or load_b["active_requests"] != 0:
            _poll_sleep(deadline)
            continue
        prefill_a = load_a["potential_prefill_tokens"]
        prefill_b = load_b["potential_prefill_tokens"]
        # Scorer raw workload is the shared positive ISL when active_prefill is 0.
        if len(tokens) <= 0 or prefill_b <= 0:
            _poll_sleep(deadline)
            continue
        cost_a = _projected_default_cost(
            isl_tokens=len(tokens),
            block_size=block_size,
            device_blocks=blocks_a,
            potential_decode_blocks=float(load_a["potential_decode_blocks"]),
            active_requests=int(load_a["active_requests"]),
            overlap_score_credit=float(policy["overlap_score_credit"]),
            prefill_load_scale=float(policy["prefill_load_scale"]),
            decode_active_request_weight=float(
                policy.get("decode_active_request_weight", 0.0)
            ),
            preferred_taint_multiplier=multiplier_a,
        )
        cost_b = _projected_default_cost(
            isl_tokens=len(tokens),
            block_size=block_size,
            device_blocks=blocks_b,
            potential_decode_blocks=float(load_b["potential_decode_blocks"]),
            active_requests=int(load_b["active_requests"]),
            overlap_score_credit=float(policy["overlap_score_credit"]),
            prefill_load_scale=float(policy["prefill_load_scale"]),
            decode_active_request_weight=float(
                policy.get("decode_active_request_weight", 0.0)
            ),
            preferred_taint_multiplier=multiplier_b,
        )
        named_a = _projected_default_cost(
            isl_tokens=len(tokens),
            block_size=block_size,
            device_blocks=blocks_a,
            potential_decode_blocks=float(load_a["potential_decode_blocks"]),
            active_requests=int(load_a["active_requests"]),
            overlap_score_credit=0.0,
            prefill_load_scale=float(policy["prefill_load_scale"]),
            decode_active_request_weight=float(
                policy.get("decode_active_request_weight", 0.0)
            ),
            preferred_taint_multiplier=multiplier_a,
        )
        named_b = _projected_default_cost(
            isl_tokens=len(tokens),
            block_size=block_size,
            device_blocks=blocks_b,
            potential_decode_blocks=float(load_b["potential_decode_blocks"]),
            active_requests=int(load_b["active_requests"]),
            overlap_score_credit=0.0,
            prefill_load_scale=float(policy["prefill_load_scale"]),
            decode_active_request_weight=float(
                policy.get("decode_active_request_weight", 0.0)
            ),
            preferred_taint_multiplier=multiplier_b,
        )
        last["cost_a"] = cost_a
        last["cost_b"] = cost_b
        last["named_a"] = named_a
        last["named_b"] = named_b
        last["multiplier_b"] = multiplier_b
        if not (cost_a < cost_b):
            _poll_sleep(deadline)
            continue
        if not (named_b < named_a):
            raise AssertionError(
                "named credit0 projection does not make preferred-taint B win "
                f"under native multiplier {multiplier_b}: "
                f"named_a={named_a} named_b={named_b} state={last}"
            )
        logger.info(
            "named-policy cost margin default A < native_mult_B×B "
            "input_weight=%s multiplier_b=%s blocks=%s prefill=%s/%s "
            "costs=%s/%s named=%s/%s",
            PREFERRED_TAINT_INPUT_WEIGHT,
            multiplier_b,
            (blocks_a, blocks_b),
            prefill_a,
            prefill_b,
            cost_a,
            cost_b,
            named_a,
            named_b,
        )
        _require_deadline_alive(deadline, label="named-cost-margin-ready")
        return {
            "blocks_a": blocks_a,
            "blocks_b": blocks_b,
            "cost_a": cost_a,
            "cost_b": cost_b,
            "named_a": named_a,
            "named_b": named_b,
            "multiplier_b": multiplier_b,
            "budget_s": ARM_DEADLINE_S,
        }
    raise AssertionError(
        "strict default cost margin A < native_multiplier_B×B was not reached "
        f"from the actual prepared provider state before the readiness budget: "
        f"{last}"
    )


def _require_quiescent_loads(
    stack,
    tokens: list[int],
    *,
    worker_a: int,
    worker_b: int,
    arm: str,
    deadline: float,
) -> None:
    from tests.router.common import _dp0_worker_load_rows

    last: list[Any] = []
    while time.monotonic() < deadline:
        remaining = _remaining(deadline, label=f"{arm}-quiescent")
        loads = stack.provider_potential_loads(tokens, timeout_s=remaining)
        _require_deadline_alive(deadline, label=f"{arm}-quiescent-loads")
        last = loads
        rows = _dp0_worker_load_rows(loads, worker_a, worker_b, arm)
        if rows is None:
            _poll_sleep(deadline)
            continue
        if all(rows[key]["active_requests"] == 0 for key in rows):
            _require_deadline_alive(deadline, label=f"{arm}-quiescent-ready")
            return
        _poll_sleep(deadline)
    raise AssertionError(
        f"{arm}: workers were not quiescent (zero active_requests) before the "
        f"consumer query: loads={last}"
    )


def _read_owned_router_policy(
    admin_port: int,
    *,
    model: str,
    namespace: str,
    component: str,
    endpoint: str,
    expected_credit: float,
    expected_temperature: float,
    deadline: float,
) -> dict[str, Any]:
    import requests

    timeout_s = _remaining(deadline, label="admin-router-policy")
    with requests.Session() as session:
        session.trust_env = False
        response = session.get(
            f"http://127.0.0.1:{admin_port}/router_policy",
            params={"model": model},
            timeout=timeout_s,
        )
    _require_deadline_alive(deadline, label="admin-router-policy")
    if response.status_code != 200:
        raise AssertionError(
            f"/router_policy returned HTTP {response.status_code}: {response.text}"
        )
    body = response.json()
    routers = body.get("routers")
    if not isinstance(routers, list) or not routers:
        raise AssertionError(f"/router_policy missing routers list: {body!r}")
    matches: list[dict[str, Any]] = []
    for entry in routers:
        if not isinstance(entry, dict):
            raise AssertionError(f"/router_policy entry is not an object: {entry!r}")
        if (
            entry.get("model") == model
            and entry.get("namespace") == namespace
            and entry.get("component") == component
            and entry.get("endpoint") == endpoint
            and entry.get("worker_type") == "aggregated"
        ):
            matches.append(entry)
    if len(matches) != 1:
        raise AssertionError(
            "expected exactly one owned aggregated /router_policy target for "
            f"model={model!r} namespace={namespace!r} component={component!r} "
            f"endpoint={endpoint!r}; matches={matches!r} body={body!r}"
        )
    target = matches[0]
    try:
        router_id = uuid.UUID(str(target["router_id"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise AssertionError(
            f"/router_policy router_id is not a UUID: {target!r}"
        ) from exc
    configuration = target.get("configuration")
    if not isinstance(configuration, dict):
        raise AssertionError(
            f"owned /router_policy configuration must be a non-null object: {target!r}"
        )
    revision = configuration.get("revision")
    if not isinstance(revision, int) or isinstance(revision, bool) or revision != 0:
        raise AssertionError(
            f"owned /router_policy revision must be int 0: {configuration!r}"
        )
    parameters = configuration.get("parameters")
    if not isinstance(parameters, dict):
        raise AssertionError(
            f"owned /router_policy parameters missing: {configuration!r}"
        )
    for name, expected in (
        ("overlap_score_credit", expected_credit),
        ("router_temperature", expected_temperature),
        ("host_cache_hit_weight", 0.0),
        ("disk_cache_hit_weight", 0.0),
        ("shared_cache_multiplier", 0.0),
        ("overlap_score_credit_decay", 0.0),
        ("prefill_load_scale", 1.0),
    ):
        if name not in parameters:
            raise AssertionError(
                f"owned /router_policy parameters missing {name}: {parameters!r}"
            )
        got = parameters[name]
        if not isinstance(got, (int, float)) or isinstance(got, bool):
            raise AssertionError(
                f"owned /router_policy parameter {name} is not numeric: {got!r}"
            )
        if not math.isfinite(float(got)):
            raise AssertionError(
                f"owned /router_policy parameter {name} is not finite: {got!r}"
            )
        if abs(float(got) - float(expected)) > 1e-9:
            raise AssertionError(
                f"owned /router_policy {name}={got!r} != {expected!r}"
            )
    return {
        "router_id": str(router_id),
        "model": model,
        "namespace": namespace,
        "component": component,
        "endpoint": endpoint,
        "worker_type": "aggregated",
        "revision": 0,
        "overlap_score_credit": float(parameters["overlap_score_credit"]),
        "router_temperature": float(parameters["router_temperature"]),
    }
