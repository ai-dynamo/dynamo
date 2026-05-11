# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scenario Pydantic models + YAML loader with ``extends:`` inheritance.

Schema summary:
  - ScenarioSpec (top-level) — class, seed, ticks, interval_s, planner, fleet|mocker+overlay, load, events, assertions
  - PlannerSpec — mirrors key PlannerConfig fields for testbed
  - FleetSpec (α-class) — system, gpus_per_*, decode_power_floor_w, bias, noise
  - MockerSpec (γ-class) — trace_file, workload params, engine args
  - OverlaySpec (γ-class) — system, noise
  - LoadSpec — profile (constant|ramp|spike), tokens_per_sec
  - Event — union of all event types (bias_step, actuation_fault, node_down, prom_outage, …)
  - Assertion — structured or expression form

YAML loader performs:
  1. Single-level ``extends:`` merge (scalar override, dict recursive merge, list replace).
  2. Pydantic validation on merged dict.
  3. Load-time assertion field validation against TickSnapshot field set.
"""

from __future__ import annotations

import ast
import dataclasses
import math
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# System spec (loaded from systems/<sku>.yaml)
# ---------------------------------------------------------------------------

_SYSTEMS_DIR = Path(__file__).parent / "systems"


class SystemSpec(BaseModel):
    """Per-SKU hardware constants used by SyntheticFleet and FakeAICEstimator."""

    tdp_w: float
    sku_min_w: float
    sku_max_w: float
    decode_power_floor_w: float
    # FakeAIC estimator constants
    aic_ttft_ms: float
    aic_itl_ms: float
    aic_power_w_prefill: float
    aic_power_w_decode: float
    max_kv_tokens: int
    # Overlay model constants (γ-class)
    overlay_prefill_saturation_tokens: int = 8192
    overlay_decode_hbm_tokens: int = 200_000

    @classmethod
    def load(cls, system_name: str) -> "SystemSpec":
        path = _SYSTEMS_DIR / f"{system_name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"System spec not found: {path}")
        with path.open() as f:
            data = yaml.safe_load(f)
        return cls(**data)


# ---------------------------------------------------------------------------
# Sub-specs
# ---------------------------------------------------------------------------


class NoiseModel(BaseModel):
    model: Literal["gaussian", "uniform", "ar1"] = "gaussian"
    sigma: float = 0.0
    half_width: float = 0.0  # uniform
    rho: float = 0.9  # ar1
    # prev noise state is managed by SyntheticFleet, not stored here


class NoiseSpec(BaseModel):
    power_per_gpu: NoiseModel = Field(default_factory=lambda: NoiseModel(model="gaussian", sigma=0.07))
    ttft: NoiseModel = Field(default_factory=lambda: NoiseModel(model="gaussian", sigma=0.05))
    itl: NoiseModel = Field(default_factory=lambda: NoiseModel(model="gaussian", sigma=0.04))
    capacity: NoiseModel = Field(default_factory=lambda: NoiseModel(model="gaussian", sigma=0.03))


class BiasSpec(BaseModel):
    power_bias_prefill: float = 1.0
    power_bias_decode: float = 1.0
    ttft_bias: float = 1.0
    itl_bias: float = 1.0
    capacity_bias: float = 1.0


class FleetSpec(BaseModel):
    system: str = "h200_sxm"
    gpus_per_prefill_engine: int = 1
    gpus_per_decode_engine: int = 2
    decode_power_floor_w: Optional[float] = None  # overrides system spec if set
    bias: BiasSpec = Field(default_factory=BiasSpec)
    noise: NoiseSpec = Field(default_factory=NoiseSpec)


class MockerSpec(BaseModel):
    trace_file: Optional[str] = None
    synthetic_workload: bool = False
    arrival_rate: float = 200.0
    isl: int = 3000
    osl: int = 150
    trace_block_size: int = 512
    arrival_speedup_ratio: float = 1.0
    num_prefill_workers: int = 1
    num_decode_workers: int = 4
    router_mode: str = "kv_router"
    prefill_engine_args: dict[str, Any] = Field(default_factory=dict)
    decode_engine_args: dict[str, Any] = Field(default_factory=dict)


class OverlaySpec(BaseModel):
    system: str = "h200_sxm"
    bias: BiasSpec = Field(default_factory=BiasSpec)
    noise: NoiseSpec = Field(default_factory=NoiseSpec)


class PlannerSpec(BaseModel):
    mode: Literal["disagg", "agg"] = "disagg"
    ttft: float = 500.0  # ms
    itl: float = 50.0    # ms
    enable_power_awareness: bool = True
    enable_aic_optimizer: bool = True
    total_gpu_power_limit: Optional[int] = 4000
    power_agent_safe_default_watts: int = 500
    prefill_engine_gpu_power_limit: int = 500
    decode_engine_gpu_power_limit: int = 425
    aic_initial_c_power_prefill: float = 1.0
    aic_initial_c_power_decode: float = 1.0
    aic_initial_c_power_agg: float = 1.0
    aic_initial_c_ttft: float = 1.0
    aic_initial_c_itl: float = 1.0
    aic_reoptimize_interval: int = 300  # seconds (virtual); 5 ticks at 60s
    aic_drift_relative_threshold: float = 0.15
    aic_drift_consecutive_ticks: int = 3
    aic_max_consecutive_failures: int = 5
    min_endpoint: int = 1
    max_gpu_budget: int = 64
    admission_mode: Literal["off", "inherit", "autoset"] = "off"


class LoadSpec(BaseModel):
    profile: Literal["constant", "ramp", "spike", "sine"] = "constant"
    tokens_per_sec: float = 2000.0
    # ramp: start -> end over ramp_start_tick to ramp_end_tick
    ramp_start_tick: int = 0
    ramp_end_tick: int = 50
    ramp_from: float = 200.0
    ramp_to: float = 2000.0
    # spike: spike_tick, spike_duration_ticks, spike_tokens_per_sec
    spike_tick: int = 50
    spike_duration_ticks: int = 10
    spike_tokens_per_sec: float = 5000.0
    # sine: amplitude, period_ticks, offset_tps
    sine_amplitude: float = 500.0
    sine_period_ticks: int = 40
    sine_offset_tps: float = 2000.0


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


class BiasStepEvent(BaseModel):
    type: Literal["bias_step"]
    at_tick: int
    signal: str
    value: float
    auto_inject_window_cross: bool = False


class BiasRampEvent(BaseModel):
    type: Literal["bias_ramp"]
    start_tick: int
    end_tick: int
    signal: str
    from_: float = Field(alias="from")
    to: float

    model_config = {"populate_by_name": True}


class BiasSineEvent(BaseModel):
    type: Literal["bias_sine"]
    signal: str
    amplitude: float
    period_ticks: int
    offset: float = 0.0


class ActuationFaultEvent(BaseModel):
    type: Literal["actuation_fault"]
    at_tick: int
    duration_ticks: int
    mode: Literal["rbac_denied", "nvml_low", "nvml_high", "daemonset_absent"]
    auto_inject_window_cross: bool = False


class NodeDownEvent(BaseModel):
    type: Literal["node_down"]
    at_tick: int
    n_prefill_lost: int = 0
    n_decode_lost: int = 0


class NodeUpEvent(BaseModel):
    type: Literal["node_up"]
    at_tick: int
    n_prefill_restored: int = 0
    n_decode_restored: int = 0


class PromOutageEvent(BaseModel):
    type: Literal["prom_outage"]
    at_tick: int
    duration_ticks: int
    signals: list[str] = Field(default_factory=lambda: ["ttft", "itl", "power_p", "power_d", "capacity"])


class PromStaleEvent(BaseModel):
    type: Literal["prom_stale"]
    at_tick: int
    duration_ticks: int
    lag_ticks: int = 5


class PromWindowCrossEvent(BaseModel):
    type: Literal["prom_window_cross_event"]
    at_tick: int
    signal: str
    weight_old: float = 0.66


class BudgetChangeEvent(BaseModel):
    type: Literal["budget_change"]
    at_tick: int
    new_total_w: int


class FrontendPostFaultEvent(BaseModel):
    type: Literal["frontend_post_fault"]
    at_tick: int
    duration_ticks: int
    failing_fraction: float = 0.33


class MdcUnavailableEvent(BaseModel):
    type: Literal["mdc_unavailable"]
    at_tick: int
    duration_ticks: int


class AicFailureEvent(BaseModel):
    type: Literal["aic_failure"]
    at_tick: int
    mode: Literal["empty_pareto", "raises"]
    n_consecutive: int = 1


Event = Annotated[
    Union[
        BiasStepEvent,
        BiasRampEvent,
        BiasSineEvent,
        ActuationFaultEvent,
        NodeDownEvent,
        NodeUpEvent,
        PromOutageEvent,
        PromStaleEvent,
        PromWindowCrossEvent,
        BudgetChangeEvent,
        FrontendPostFaultEvent,
        MdcUnavailableEvent,
        AicFailureEvent,
    ],
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

# Valid assertion ``field:`` names — derived from TickSnapshot directly to
# guarantee the two stay in sync as fields are added/renamed.
def _load_tick_snapshot_fields() -> frozenset[str]:
    from dynamo.planner.tests.testbed.recorder import TICK_SNAPSHOT_FIELDS
    return frozenset(TICK_SNAPSHOT_FIELDS)


# Cached at first access; recorder is a sibling module so the cycle is fine.
_TICK_SNAPSHOT_FIELDS: Optional[frozenset[str]] = None


def _tick_snapshot_fields() -> frozenset[str]:
    global _TICK_SNAPSHOT_FIELDS
    if _TICK_SNAPSHOT_FIELDS is None:
        _TICK_SNAPSHOT_FIELDS = _load_tick_snapshot_fields()
    return _TICK_SNAPSHOT_FIELDS

_ASSERTION_OPS = {"<", "<=", "==", ">=", ">", "within", "!="}


class StructuredAssertion(BaseModel):
    field: Optional[str] = None
    op: Optional[str] = None
    value: Optional[float] = None
    tolerance: Optional[float] = None
    ref: Optional[str] = None
    description: Optional[str] = None
    # Exactly one of: at_tick (int), always (bool=True), eventually_by_tick (int).
    at_tick: Optional[int] = None
    always: Optional[bool] = None
    eventually_by_tick: Optional[int] = None
    # Counter form
    counter: Optional[str] = None  # counter name
    label: Optional[dict[str, str]] = None

    @model_validator(mode="after")
    def _validate_assertion(self) -> "StructuredAssertion":
        if self.op is not None and self.op not in _ASSERTION_OPS:
            raise ValueError(f"Unknown op: {self.op!r}. Must be one of {_ASSERTION_OPS}")
        if self.op == "within" and self.tolerance is None:
            raise ValueError("op='within' requires tolerance")

        # Exactly one predicate must be set, and `always` must be True if
        # present (bare ``always:`` in YAML parses to None which would
        # silently skip evaluation — that bit us hard once, never again).
        predicates = [
            self.at_tick is not None,
            self.always is True,
            self.eventually_by_tick is not None,
        ]
        if sum(predicates) == 0:
            raise ValueError(
                "StructuredAssertion requires exactly one of "
                "{at_tick: <int>, always: true, eventually_by_tick: <int>}. "
                "Bare `always:` in YAML parses to None and is rejected — "
                "write `always: true` explicitly."
            )
        if sum(predicates) > 1:
            raise ValueError(
                "StructuredAssertion: at_tick / always / eventually_by_tick are "
                "mutually exclusive — set exactly one."
            )

        # `field` / `op` are required for non-counter assertions.
        if self.counter is None and (self.field is None or self.op is None):
            raise ValueError(
                "StructuredAssertion requires `field` and `op` "
                "(or `counter` + `label` for counter-delta form)."
            )
        return self


class ExprAssertion(BaseModel):
    expr: str
    at_tick: Optional[int] = None
    always: Optional[bool] = None
    eventually_by_tick: Optional[int] = None
    description: Optional[str] = None


Assertion = Union[StructuredAssertion, ExprAssertion]


# ---------------------------------------------------------------------------
# Top-level scenario spec
# ---------------------------------------------------------------------------


class ScenarioSpec(BaseModel):
    name: str
    class_: Literal["alpha", "gamma"] = Field(alias="class", default="alpha")
    description: str = ""
    seed: int = 42
    ticks: int = 200
    interval_s: float = 60.0

    planner: PlannerSpec = Field(default_factory=PlannerSpec)
    fleet: Optional[FleetSpec] = None     # α-class
    mocker: Optional[MockerSpec] = None   # γ-class
    overlay: Optional[OverlaySpec] = None # γ-class
    load: LoadSpec = Field(default_factory=LoadSpec)
    events: list[dict[str, Any]] = Field(default_factory=list)
    assertions: list[dict[str, Any]] = Field(default_factory=list)

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def _validate_class_fields(self) -> "ScenarioSpec":
        if self.class_ == "alpha" and self.fleet is None:
            self.fleet = FleetSpec()
        if self.class_ == "gamma" and self.mocker is None:
            raise ValueError("gamma-class scenario requires a 'mocker:' block")
        if self.class_ == "gamma" and self.overlay is None:
            self.overlay = OverlaySpec()
        return self

    @model_validator(mode="after")
    def _validate_events_and_assertions(self) -> "ScenarioSpec":
        """Eagerly parse events/assertions so authoring errors surface at load.

        Without this, a typo in a YAML event ``type:`` only blows up when the
        runner actually consumes ``parsed_events()`` — which can be deep
        inside a tick loop, far from a useful stack trace.
        """
        # parsed_*() raises ValidationError on malformed entries.
        self.parsed_events()
        self.parsed_assertions()
        # Reference-name validation against TickSnapshot field set.
        errors = validate_assertion_fields(self.assertions)
        if errors:
            raise ValueError(
                "Scenario assertion validation failed:\n  " + "\n  ".join(errors)
            )
        return self

    @property
    def class_name(self) -> str:
        return self.class_

    def offered_load_at(self, tick: int) -> float:
        """Compute offered load (tok/s) for this tick according to load profile."""
        L = self.load
        if L.profile == "constant":
            return L.tokens_per_sec
        elif L.profile == "ramp":
            if tick <= L.ramp_start_tick:
                return L.ramp_from
            elif tick >= L.ramp_end_tick:
                return L.ramp_to
            t = (tick - L.ramp_start_tick) / max(1, L.ramp_end_tick - L.ramp_start_tick)
            return L.ramp_from + t * (L.ramp_to - L.ramp_from)
        elif L.profile == "spike":
            if L.spike_tick <= tick < L.spike_tick + L.spike_duration_ticks:
                return L.spike_tokens_per_sec
            return L.tokens_per_sec
        elif L.profile == "sine":
            return L.sine_offset_tps + L.sine_amplitude * math.sin(
                2 * math.pi * tick / max(1, L.sine_period_ticks)
            )
        return L.tokens_per_sec

    def parsed_events(self) -> list[Event]:
        """Parse raw event dicts into typed Event objects."""
        result = []
        for e in self.events:
            etype = e.get("type")
            type_map = {
                "bias_step": BiasStepEvent,
                "bias_ramp": BiasRampEvent,
                "bias_sine": BiasSineEvent,
                "actuation_fault": ActuationFaultEvent,
                "node_down": NodeDownEvent,
                "node_up": NodeUpEvent,
                "prom_outage": PromOutageEvent,
                "prom_stale": PromStaleEvent,
                "prom_window_cross_event": PromWindowCrossEvent,
                "budget_change": BudgetChangeEvent,
                "frontend_post_fault": FrontendPostFaultEvent,
                "mdc_unavailable": MdcUnavailableEvent,
                "aic_failure": AicFailureEvent,
            }
            cls = type_map.get(etype)
            if cls is None:
                raise ValueError(f"Unknown event type: {etype!r}")
            result.append(cls(**e))
        return result

    def parsed_assertions(self) -> list[Assertion]:
        """Parse raw assertion dicts into typed Assertion objects."""
        result = []
        for a in self.assertions:
            if "expr" in a:
                result.append(ExprAssertion(**a))
            else:
                result.append(StructuredAssertion(**a))
        return result


# ---------------------------------------------------------------------------
# YAML loader with ``extends:`` support
# ---------------------------------------------------------------------------

_SCENARIOS_DIR = Path(__file__).parent / "scenarios"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge: override keys win; missing keys inherit from base.

    Lists are REPLACED (not concatenated) — makes scenarios easy to read.
    """
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_scenario(path: Union[str, Path]) -> ScenarioSpec:
    """Load a scenario YAML file, resolving ``extends:`` inheritance.

    The ``extends:`` key must point to a path relative to the ``scenarios/``
    directory root.  Only a single level of inheritance is supported (the
    base template cannot itself ``extends:`` another file).
    """
    path = Path(path)
    with path.open() as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    extends = raw.pop("extends", None)
    if extends:
        base_path = _SCENARIOS_DIR / extends
        with base_path.open() as f:
            base_raw: dict[str, Any] = yaml.safe_load(f) or {}
        base_raw.pop("extends", None)  # base cannot chain
        raw = _deep_merge(base_raw, raw)

    # Rename Python-reserved alias: "class" → "class_" via Pydantic alias
    return ScenarioSpec.model_validate(raw)


def load_all_scenarios() -> list[tuple[str, ScenarioSpec]]:
    """Load all scenario YAML files from the scenarios/ directory.

    Returns list of (scenario_name, ScenarioSpec).  Skips _base/ templates.
    """
    results = []
    for yaml_path in sorted(_SCENARIOS_DIR.glob("*.yaml")):
        spec = load_scenario(yaml_path)
        results.append((spec.name, spec))
    return results


def validate_assertion_fields(assertions: list[dict[str, Any]]) -> list[str]:
    """Validate field / ref / expr references in assertions.

    Returns a list of error messages (empty list means all valid).
    """
    errors: list[str] = []
    known_fields = _tick_snapshot_fields()
    for i, a in enumerate(assertions):
        field_name = a.get("field")
        if field_name and field_name not in known_fields:
            errors.append(
                f"assertion[{i}]: field {field_name!r} not in TickSnapshot. "
                f"Known fields: {sorted(known_fields)}"
            )
        ref = a.get("ref")
        if ref:
            parts = ref.split(".")
            if parts[0] not in ("planner", "counters", "fleet", "overlay"):
                errors.append(
                    f"assertion[{i}]: ref {ref!r} must start with "
                    f"'planner.', 'counters.', 'fleet.', or 'overlay.'"
                )
        expr = a.get("expr")
        if expr:
            try:
                tree = ast.parse(expr, mode="eval")
                _validate_expr_node(tree, i, errors)
            except SyntaxError as e:
                errors.append(f"assertion[{i}]: expr syntax error: {e}")
    return errors


def _validate_expr_node(tree: ast.AST, idx: int, errors: list[str]) -> None:
    """Walk expression AST and reject unknown history/planner/counters references."""
    _ALLOWED_ROOTS = {"history", "planner", "counters", "abs", "min", "max"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in _ALLOWED_ROOTS:
            errors.append(
                f"assertion[{idx}]: expr references unknown name {node.id!r}; "
                f"allowed roots: {_ALLOWED_ROOTS}"
            )


class ScenarioLoadError(Exception):
    pass
