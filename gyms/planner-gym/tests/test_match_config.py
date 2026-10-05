# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure-Python contract tests for human-readable Arena match configs.

These tests intentionally exercise parsing, normalization, and matrix expansion
without importing the Dynamo-backed simulation runner or invoking AIPerf.
"""

from __future__ import annotations

import json
import os
import re
import textwrap
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pytest
from autoscaling_arena.match_config import MatchConfigError, load_match_config
from autoscaling_arena.substrates import SUBSTRATES

ARENA_ROOT = Path(__file__).resolve().parents[1]
ENDPOINT_CATALOG = ARENA_ROOT / "configs" / "endpoints.example.json"

SYNTHETIC_WORKLOADS = {
    "flat",
    "staircase",
    "square_wave",
    "flash_crowd",
    "diurnal",
    "decode_heavy",
    "shared_prefix",
}


def _write_yaml(tmp_path: Path, body: str, *, name: str = "match.yaml") -> Path:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body).lstrip())
    return path


def _field(value: Any, *names: str) -> Any:
    """Read a documented field while tolerating mapping-backed leaf configs."""
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    raise AssertionError(
        f"{type(value).__name__} exposes none of the expected fields: {names}"
    )


def _sequence(value: Any) -> list[Any]:
    assert not isinstance(value, (str, bytes))
    assert isinstance(value, Iterable)
    return list(value)


def _evaluation_names(config: Any) -> list[str]:
    evaluations = config.evaluations
    if not isinstance(evaluations, (str, bytes)) and hasattr(evaluations, "workloads"):
        evaluations = evaluations.workloads
    names = []
    for evaluation in _sequence(evaluations):
        names.append(
            str(
                _field(evaluation, "name", "workload_name", "workload")
                if not isinstance(evaluation, str)
                else evaluation
            )
        )
    return names


def _autoscaler_name(value: Any) -> str:
    return str(_field(value, "name", "autoscaler_name"))


def _profile_name(value: Any) -> str:
    return str(_field(value, "name"))


def _profile_triplet(value: Any) -> tuple[float | None, float | None, float | None]:
    return (
        _field(value, "ttft_ms"),
        _field(value, "itl_ms"),
        _field(value, "e2e_ms"),
    )


def _run_id(value: Any) -> str:
    return str(_field(value, "run_id", "id"))


def _run_autoscaler_name(value: Any) -> str:
    autoscaler = _field(value, "autoscaler_name", "autoscaler")
    return (
        _autoscaler_name(autoscaler) if not isinstance(autoscaler, str) else autoscaler
    )


def _run_workload_name(value: Any) -> str:
    workload = _field(value, "workload_name", "workload", "evaluation")
    if isinstance(workload, str):
        return workload
    return str(_field(workload, "name", "workload_name"))


def _run_profile(value: Any) -> Any:
    return _field(value, "slo_profile", "profile")


def _sim_yaml(
    *,
    name: str = "unit-sim",
    autoscalers: str = """
      - name: static-small
        type: static
        config:
          num_prefill: 2
          num_decode: 3
    """,
    evaluations: str = """
      workloads: [flat]
    """,
    slo_profiles: str = """
      - name: interactive
        ttft_ms: 300
        itl_ms: 50
    """,
    metrics: str = """
      rank_by: goodput_per_gpu
      include: [goodput_rps, goodput_per_gpu, gpu_hours]
    """,
    execution: str = """
      repetitions: 1
      max_runs: 100
    """,
    publish: str = """
      artifact_root: artifacts
      destinations:
        - type: console
    """,
) -> str:
    sections = [
        "schema_version: 1",
        f"name: {name}",
        "backend:",
        "  type: sim",
        "  substrate: gpt_oss",
        "  topology: disagg",
        "  autoscalers:",
        textwrap.indent(textwrap.dedent(autoscalers).strip(), "    "),
        "evaluations:",
        textwrap.indent(textwrap.dedent(evaluations).strip(), "  "),
        "slo_profiles:",
        textwrap.indent(textwrap.dedent(slo_profiles).strip(), "  "),
        "metrics:",
        textwrap.indent(textwrap.dedent(metrics).strip(), "  "),
        "execution:",
        textwrap.indent(textwrap.dedent(execution).strip(), "  "),
        "publish:",
        textwrap.indent(textwrap.dedent(publish).strip(), "  "),
    ]
    return "\n".join(sections) + "\n"


def _sim_yaml_with_backend(backend: str, **kwargs: Any) -> str:
    """Replace the preset shorthand with a focused sim-backend contract."""

    text = _sim_yaml(**kwargs)
    shorthand = "  substrate: gpt_oss\n  topology: disagg"
    assert shorthand in text
    replacement = textwrap.indent(textwrap.dedent(backend).strip(), "  ")
    return text.replace(shorthand, replacement, 1)


def _append_yaml_mapping(lines: list[str], key: str, body: str, *, indent: int) -> None:
    prefix = " " * indent
    normalized = textwrap.dedent(body).strip()
    if normalized == "{}":
        lines.append(f"{prefix}{key}: {{}}")
        return
    lines.append(f"{prefix}{key}:")
    lines.append(textwrap.indent(normalized, " " * (indent + 2)))


def _explicit_backend(
    *,
    topology: str = "disagg",
    gpu_budget: int | None = 64,
    model: str
    | None = """
      name: served-model
      ais_model_path: perf-db/model
    """,
    common: str = """
      system: h200_sxm
      backend: vllm
    """,
    roles: Mapping[str, str] | None = None,
    replay: str | None = None,
    include_engines: bool = True,
) -> str:
    lines = [f"topology: {topology}"]
    if gpu_budget is not None:
        lines.append(f"gpu_budget: {gpu_budget}")
    if model is not None:
        _append_yaml_mapping(lines, "model", model, indent=0)
    if include_engines:
        lines.append("engines:")
        _append_yaml_mapping(lines, "common", common, indent=2)
        selected_roles = roles or (
            {"prefill": "{}", "decode": "{}"}
            if topology == "disagg"
            else {"aggregate": "{}"}
        )
        for role, body in selected_roles.items():
            _append_yaml_mapping(lines, role, body, indent=2)
    if replay is not None:
        _append_yaml_mapping(lines, "replay", replay, indent=0)
    return "\n".join(lines)


def _real_yaml(
    endpoint_catalog: str,
    *,
    endpoint: str = "fast-endpoint",
    metrics: str = """
      rank_by: goodput_rps
      include: [goodput_rps, good_request_fraction, mean_ttft_ms]
    """,
    publish: str = """
      artifact_root: artifacts
      destinations:
        - type: console
    """,
) -> str:
    sections = [
        "schema_version: 1",
        "name: unit-real",
        "backend:",
        "  type: real",
        f"  endpoint_catalog: {endpoint_catalog}",
        "  autoscalers:",
        "    - name: endpoint-under-test",
        f"      endpoint: {endpoint}",
        "evaluations:",
        "  workloads: [flat]",
        "slo_profiles:",
        "  - name: interactive",
        "    ttft_ms: 300",
        "    itl_ms: 50",
        "metrics:",
        textwrap.indent(textwrap.dedent(metrics).strip(), "  "),
        "execution:",
        "  repetitions: 1",
        "  max_runs: 100",
        "publish:",
        textwrap.indent(textwrap.dedent(publish).strip(), "  "),
    ]
    return "\n".join(sections) + "\n"


@pytest.mark.parametrize(
    ("filename", "backend_type"),
    [
        ("match.sim.example.yaml", "sim"),
        ("match.real.example.yaml", "real"),
    ],
)
def test_committed_examples_load_and_expand(filename: str, backend_type: str):
    path = ARENA_ROOT / "configs" / filename
    assert path.exists(), f"committed example is missing: {path}"

    config = load_match_config(path)
    runs = list(config.iter_runs())

    assert config.backend.type == backend_type
    assert config.name
    assert _evaluation_names(config)
    assert _sequence(config.sla_profiles)
    assert config.metrics is not None
    assert config.publish is not None
    assert config.expected_runs == len(runs)
    assert runs


def test_golden_set_quickstart_loads_beside_trace(tmp_path: Path):
    source = ARENA_ROOT / "configs" / "match.golden-set.quickstart.yaml"
    config_path = tmp_path / "match.yaml"
    config_path.write_text(source.read_text())
    (tmp_path / "step-and-recovery.jsonl").write_text(
        json.dumps(
            {
                "timestamp": 0,
                "input_length": 16,
                "output_length": 4,
                "hash_ids": [1],
            }
        )
        + "\n"
    )

    config = load_match_config(config_path)
    runs = list(config.iter_runs())

    assert config.expected_runs == len(runs) == 2
    assert [_autoscaler_name(item) for item in config.backend.autoscalers] == [
        "planner",
        "keda",
    ]
    assert _evaluation_names(config) == ["step-and-recovery"]
    assert config.backend.replay.telemetry_sample_interval_s == 5.0


def test_real_example_runs_without_a_recorded_trace_or_dynamo_checkout():
    config = load_match_config(ARENA_ROOT / "configs" / "match.real.example.yaml")

    assert _evaluation_names(config) == ["flat"]
    assert config.expected_runs == 8


def test_external_trace_and_suite_expand_without_duplicates(tmp_path: Path):
    trace = tmp_path / "traces" / "private.jsonl"
    trace.parent.mkdir()
    trace.write_text(
        '{"timestamp": 0, "input_length": 8, "output_length": 2, "hash_ids": [1]}\n'
    )
    path = _write_yaml(
        tmp_path,
        _sim_yaml(
            evaluations="""
              workloads: [synthetic]
              suites: [synthetic]
              traces:
                - name: private-traffic
                  path: traces/private.jsonl
                  block_size: 64
                  presorted: true
            """
        ),
    )

    config = load_match_config(path)
    names = _evaluation_names(config)

    assert set(names) == SYNTHETIC_WORKLOADS | {"private-traffic"}
    assert len(names) == len(set(names)) == 8
    assert names == _evaluation_names(load_match_config(path))
    external = next(
        evaluation
        for evaluation in config.evaluations
        if evaluation.workload == "private-traffic"
    )
    assert external.trace_path == trace.resolve()
    assert external.trace_paths == (trace.resolve(),)
    assert external.trace_format == "mooncake"
    assert external.trace_block_size == 64
    assert external.trace_presorted is True
    run = next(
        item for item in config.iter_runs() if item.workload == "private-traffic"
    )
    assert run.trace_path == trace.resolve()
    assert run.trace_paths == (trace.resolve(),)
    assert run.trace_format == "mooncake"
    assert run.trace_block_size == 64
    assert run.trace_presorted is True


def test_exact_dynamo_trace_manifest_preserves_order_and_embedded_block_size(
    tmp_path: Path,
):
    traces_dir = tmp_path / "traces"
    traces_dir.mkdir()
    first = traces_dir / "recorded-01.jsonl.gz"
    second = traces_dir / "recorded-02.jsonl.gz"
    # Native traces are validated by Dynamo itself. Config loading deliberately
    # does not reinterpret them as Mooncake or rewrite their contents.
    first.write_bytes(b"native-dynamo-shard-one")
    second.write_bytes(b"native-dynamo-shard-two")
    path = _write_yaml(
        tmp_path,
        _sim_yaml(
            evaluations="""
              traces:
                - name: recorded-native
                  format: dynamo
                  paths:
                    - traces/recorded-02.jsonl.gz
                    - traces/recorded-01.jsonl.gz
            """
        ),
    )

    config = load_match_config(path)
    evaluation = config.evaluations[0]
    expected_paths = (second.resolve(), first.resolve())

    assert evaluation.workload == "recorded-native"
    assert evaluation.trace_format == "dynamo"
    assert evaluation.trace_paths == expected_paths
    assert evaluation.trace_path is None
    assert evaluation.trace_block_size is None
    assert evaluation.max_requests is None
    assert evaluation.arrival_speedup == 1.0

    run = next(config.iter_runs())
    assert run.trace_format == "dynamo"
    assert run.trace_paths == expected_paths
    assert run.trace_path is None
    assert run.trace_block_size is None

    resolved = config.to_dict()["evaluations"][0]
    assert resolved["trace_paths"] == [str(item) for item in expected_paths]
    assert resolved["trace_format"] == "dynamo"


def test_exact_dynamo_trace_allows_explicit_expected_block_size(tmp_path: Path):
    trace = tmp_path / "recorded.jsonl.gz"
    trace.write_bytes(b"native-dynamo-trace")
    path = _write_yaml(
        tmp_path,
        _sim_yaml(
            evaluations="""
              traces:
                - name: recorded-native
                  format: dynamo
                  paths: [recorded.jsonl.gz]
                  block_size: 64
            """
        ),
    )

    evaluation = load_match_config(path).evaluations[0]

    assert evaluation.trace_block_size == 64


def test_exact_dynamo_trace_is_rejected_for_real_backend(tmp_path: Path):
    trace = tmp_path / "recorded.jsonl.gz"
    trace.write_bytes(b"native-dynamo-trace")
    catalog_ref = os.path.relpath(ENDPOINT_CATALOG, tmp_path)
    body = _real_yaml(catalog_ref).replace(
        "  workloads: [flat]",
        "  traces:\n"
        "    - name: recorded-native\n"
        "      format: dynamo\n"
        "      paths: [recorded.jsonl.gz]",
    )
    path = _write_yaml(tmp_path, body)

    with pytest.raises(MatchConfigError, match="require backend.type 'sim'"):
        load_match_config(path)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ("max_requests: 100", "max_requests"),
        ("arrival_speedup: 2", "arrival_speedup"),
    ],
)
def test_exact_dynamo_trace_rejects_transforming_evaluation_options(
    tmp_path: Path, options: str, message: str
):
    trace = tmp_path / "recorded.jsonl.gz"
    trace.write_bytes(b"native-dynamo-trace")
    path = _write_yaml(
        tmp_path,
        _sim_yaml(
            evaluations=f"""
              traces:
                - name: recorded-native
                  format: dynamo
                  paths: [recorded.jsonl.gz]
              defaults:
{textwrap.indent(options, "                ")}
            """
        ),
    )

    with pytest.raises(MatchConfigError, match=message):
        load_match_config(path)


@pytest.mark.parametrize(
    ("trace_entry", "message"),
    [
        (
            """
            format: dynamo
            path: recorded.jsonl.gz
            """,
            "requires paths",
        ),
        (
            """
            format: dynamo
            paths: []
            """,
            "must not be empty",
        ),
        (
            """
            format: dynamo
            paths: [recorded.jsonl.gz]
            presorted: true
            """,
            "not supported for exact Dynamo traces",
        ),
        (
            """
            format: other
            path: recorded.jsonl.gz
            """,
            "expected 'mooncake' or 'dynamo'",
        ),
        (
            """
            format: mooncake
            paths: [recorded.jsonl.gz]
            """,
            "requires singular path",
        ),
    ],
)
def test_external_trace_manifest_rejects_ambiguous_shapes(
    tmp_path: Path, trace_entry: str, message: str
):
    trace = tmp_path / "recorded.jsonl.gz"
    trace.write_bytes(b"native-dynamo-trace")
    entry = textwrap.indent(textwrap.dedent(trace_entry).strip(), "                  ")
    path = _write_yaml(
        tmp_path,
        _sim_yaml(
            evaluations=f"""
              traces:
                - name: recorded-native
{entry}
            """
        ),
    )

    with pytest.raises(MatchConfigError, match=message):
        load_match_config(path)


def test_external_trace_must_exist(tmp_path: Path):
    path = _write_yaml(
        tmp_path,
        _sim_yaml(
            evaluations="""
              traces:
                - name: missing-traffic
                  path: traces/missing.jsonl
            """
        ),
    )

    with pytest.raises(MatchConfigError, match="trace file does not exist"):
        load_match_config(path)


def test_external_trace_schema_is_validated_at_config_load(tmp_path: Path):
    trace = tmp_path / "trace.jsonl"
    trace.write_text('{"timestamp": 0}\n')
    path = _write_yaml(
        tmp_path,
        _sim_yaml(
            evaluations="""
              traces:
                - name: recorded-traffic
                  path: trace.jsonl
            """
        ),
    )

    with pytest.raises(MatchConfigError, match="invalid Mooncake trace"):
        load_match_config(path)


def test_external_trace_name_cannot_shadow_builtin(tmp_path: Path):
    trace = tmp_path / "trace.jsonl"
    trace.write_text('{"timestamp": 0}\n')
    path = _write_yaml(
        tmp_path,
        _sim_yaml(
            evaluations="""
              traces:
                - name: flat
                  path: trace.jsonl
            """
        ),
    )

    with pytest.raises(MatchConfigError, match="conflicts with a built-in workload"):
        load_match_config(path)


@pytest.mark.parametrize("reserved_name", ["synthetic", "recorded", "all"])
def test_external_trace_name_cannot_shadow_suite(tmp_path: Path, reserved_name: str):
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        '{"timestamp": 0, "input_length": 8, "output_length": 2, "hash_ids": [1]}\n'
    )
    path = _write_yaml(
        tmp_path,
        _sim_yaml(
            evaluations=f"""
              traces:
                - name: {reserved_name}
                  path: trace.jsonl
            """
        ),
    )

    with pytest.raises(MatchConfigError, match="conflicts with"):
        load_match_config(path)


def test_external_trace_block_size_must_be_positive(tmp_path: Path):
    trace = tmp_path / "trace.jsonl"
    trace.write_text('{"timestamp": 0}\n')
    path = _write_yaml(
        tmp_path,
        _sim_yaml(
            evaluations="""
              traces:
                - name: private-traffic
                  path: trace.jsonl
                  block_size: 0
            """
        ),
    )

    with pytest.raises(MatchConfigError, match="block_size: must be > 0"):
        load_match_config(path)


def test_list_valued_slas_expand_as_cartesian_product(tmp_path: Path):
    path = _write_yaml(
        tmp_path,
        _sim_yaml(
            slo_profiles="""
              - name: latency-grid
                ttft_ms: [200, 400]
                itl_ms: [25, 50]
                e2e_ms: 5000
            """
        ),
    )

    config = load_match_config(path)
    profiles = _sequence(config.sla_profiles)

    assert len(profiles) == 4
    assert {_profile_triplet(profile) for profile in profiles} == {
        (200, 25, 5000),
        (200, 50, 5000),
        (400, 25, 5000),
        (400, 50, 5000),
    }
    assert len({_profile_name(profile) for profile in profiles}) == 4


def test_static_autoscaler_preserves_fixed_counts(tmp_path: Path):
    path = _write_yaml(tmp_path, _sim_yaml())

    config = load_match_config(path)
    static = next(
        autoscaler
        for autoscaler in _sequence(config.backend.autoscalers)
        if _field(autoscaler, "type") == "static"
    )
    static_config = _field(static, "config")

    assert _field(static_config, "num_prefill") == 2
    assert _field(static_config, "num_decode") == 3
    assert _field(static_config, "poll_interval_s") == 5.0


def test_dynamic_autoscalers_resolve_effective_defaults_and_overrides(
    tmp_path: Path,
):
    path = _write_yaml(
        tmp_path,
        _sim_yaml(
            autoscalers="""
              - name: reactive-defaults
                type: reactive
                start:
                  prefill: 1
                  decode: 1
              - name: keda-tuned
                type: keda
                start:
                  prefill: 1
                  decode: 1
                config:
                  poll_interval_s: 7
                  max_prefill: 12
                  max_decode: 6
                  tolerance: 0.05
            """
        ),
    )

    autoscalers = {
        autoscaler.name: autoscaler
        for autoscaler in _sequence(load_match_config(path).backend.autoscalers)
    }

    assert autoscalers["reactive-defaults"].config == {
        "min_prefill": 1,
        "max_prefill": 16,
        "min_decode": 1,
        "max_decode": 8,
        "prefill_queue_up": 4,
        "prefill_queue_down": 1,
        "decode_kv_up": 0.8,
        "decode_kv_down": 0.3,
        "poll_interval_s": 15.0,
        "step": 1,
    }
    assert autoscalers["keda-tuned"].config == {
        "poll_interval_s": 7.0,
        "queue_threshold": 5.0,
        "kv_threshold": 0.9,
        "min_prefill": 1,
        "max_prefill": 12,
        "min_decode": 1,
        "max_decode": 6,
        "tolerance": 0.05,
        "scale_up_stabilization_s": 0.0,
        "scale_down_stabilization_s": 300.0,
    }


def test_preset_shorthand_resolves_model_and_disagg_engines(tmp_path: Path):
    config = load_match_config(_write_yaml(tmp_path, _sim_yaml()))
    backend = config.backend

    assert backend.substrate == "gpt_oss"
    assert backend.model.name == "openai/gpt-oss-120b"
    assert backend.model.ais_model_path == "openai/gpt-oss-120b"
    assert backend.engines.aggregate is None

    prefill = backend.engines.prefill
    decode = backend.engines.decode
    assert prefill is not None and decode is not None
    assert prefill == decode
    assert prefill.system == "h200_sxm"
    assert prefill.backend == "vllm"
    assert prefill.backend_version == "0.19.0"
    assert prefill.tp_size == 1
    assert prefill.attention_dp_size == 1
    assert prefill.num_gpus == 1
    assert prefill.extra_args == {}
    assert json.loads(SUBSTRATES["gpt_oss"].engine_args())["engine_type"] == "vllm"


def test_preset_legacy_replay_model_name_preserves_ais_model_path(tmp_path: Path):
    body = _sim_yaml().replace(
        "  topology: disagg",
        "  topology: disagg\n  replay:\n    model_name: request-model-alias",
        1,
    )

    backend = load_match_config(_write_yaml(tmp_path, body)).backend

    assert backend.model.name == "request-model-alias"
    assert backend.model.ais_model_path == "openai/gpt-oss-120b"


@pytest.mark.parametrize(
    ("replay", "expected"),
    [
        (None, 5.0),
        ("telemetry_sample_interval_s: 30", 30.0),
        ("telemetry_sample_interval_s: null", None),
    ],
)
def test_replay_telemetry_sample_interval_is_configurable(
    tmp_path: Path, replay: str | None, expected: float | None
):
    backend_body = _explicit_backend(replay=replay)

    config = load_match_config(
        _write_yaml(tmp_path, _sim_yaml_with_backend(backend_body))
    )

    assert config.backend.replay.telemetry_sample_interval_s == expected


@pytest.mark.parametrize("value", ["0", "-1", ".inf", "not-a-number"])
def test_replay_telemetry_sample_interval_must_be_positive_and_finite(
    tmp_path: Path, value: str
):
    backend_body = _explicit_backend(replay=f"telemetry_sample_interval_s: {value}")

    with pytest.raises(
        MatchConfigError,
        match="backend.replay.telemetry_sample_interval_s",
    ):
        load_match_config(_write_yaml(tmp_path, _sim_yaml_with_backend(backend_body)))


def test_explicit_disagg_resolves_asymmetric_roles_and_deep_merges_extra_args(
    tmp_path: Path,
):
    backend_body = _explicit_backend(
        gpu_budget=64,
        common="""
          system: h200_sxm
          backend: vllm
          backend_version: "0.19.0"
          tp_size: 2
          attention_dp_size: 1
          extra_args:
            num_gpu_blocks: 100
            common_only: retained
            transport:
              common_only: true
              nested:
                keep: common
                replace: common
        """,
        roles={
            "prefill": """
              tp_size: 4
              moe_tp_size: 2
              moe_ep_size: 2
              extra_args:
                transport:
                  nested:
                    replace: prefill
                    role_only: true
            """,
            "decode": """
              backend: sglang
              tp_size: 2
              attention_dp_size: 2
              moe_tp_size: 1
              moe_ep_size: 4
              extra_args:
                num_gpu_blocks: 200
            """,
        },
    )
    config = load_match_config(
        _write_yaml(tmp_path, _sim_yaml_with_backend(backend_body))
    )

    assert config.backend.substrate is None
    assert config.backend.model.name == "served-model"
    assert config.backend.model.ais_model_path == "perf-db/model"

    prefill = config.backend.engines.prefill
    decode = config.backend.engines.decode
    assert prefill is not None and decode is not None
    assert config.backend.engines.aggregate is None

    assert prefill.system == "h200_sxm"
    assert prefill.backend == "vllm"
    assert prefill.backend_version == "0.19.0"
    assert prefill.tp_size == 4
    assert prefill.attention_dp_size == 1
    assert prefill.num_gpus == 4
    assert prefill.extra_args == {
        "num_gpu_blocks": 100,
        "common_only": "retained",
        "transport": {
            "common_only": True,
            "nested": {
                "keep": "common",
                "replace": "prefill",
                "role_only": True,
            },
        },
    }

    assert decode.system == "h200_sxm"
    assert decode.backend == "sglang"
    assert decode.backend_version == "0.19.0"
    assert decode.tp_size == 2
    assert decode.attention_dp_size == 2
    assert decode.num_gpus == 4
    assert decode.extra_args == {
        "num_gpu_blocks": 200,
        "common_only": "retained",
        "transport": {
            "common_only": True,
            "nested": {"keep": "common", "replace": "common"},
        },
    }


def test_engine_runtime_common_and_role_layers_merge(tmp_path: Path):
    backend_body = _explicit_backend(
        common="""
          system: h200_sxm
          backend: vllm
          runtime:
            cold_start_delay_s: 30
            kv_transfer_bandwidth_gbps: 40
            kv_bytes_per_token: 262144
        """,
        roles={
            "prefill": """
              runtime:
                cold_start_delay_s: 45
            """,
            "decode": """
              runtime:
                kv_transfer_bandwidth_gbps: 80
            """,
        },
    )

    backend = load_match_config(
        _write_yaml(tmp_path, _sim_yaml_with_backend(backend_body))
    ).backend
    prefill = backend.engines.prefill
    decode = backend.engines.decode
    assert prefill is not None and decode is not None

    assert prefill.runtime.cold_start_delay_s == 45.0
    assert prefill.runtime.kv_transfer_bandwidth_gbps == 40.0
    assert prefill.runtime.kv_bytes_per_token == 262144
    assert decode.runtime.cold_start_delay_s == 30.0
    assert decode.runtime.kv_transfer_bandwidth_gbps == 80.0
    assert decode.runtime.kv_bytes_per_token == 262144


@pytest.mark.parametrize(
    ("runtime", "message_parts"),
    [
        ("cold_start_delay_s: -1", ("cold_start_delay_s", ">= 0")),
        (
            "kv_transfer_bandwidth_gbps: 0\nkv_bytes_per_token: 1",
            ("kv_transfer_bandwidth_gbps", "> 0"),
        ),
        (
            "kv_transfer_bandwidth_gbps: 1\nkv_bytes_per_token: 0",
            ("kv_bytes_per_token", "> 0"),
        ),
        (
            "kv_transfer_bandwidth_gbps: 1",
            (
                "kv_transfer_bandwidth_gbps",
                "kv_bytes_per_token",
                "together",
            ),
        ),
        (
            "kv_bytes_per_token: 1024",
            (
                "kv_transfer_bandwidth_gbps",
                "kv_bytes_per_token",
                "together",
            ),
        ),
        ("mystery_delay_s: 1", ("mystery_delay_s", "unknown")),
    ],
    ids=[
        "negative-cold-start",
        "zero-transfer-bandwidth",
        "zero-kv-bytes",
        "bandwidth-without-kv-bytes",
        "kv-bytes-without-bandwidth",
        "unknown-runtime-field",
    ],
)
def test_engine_runtime_rejects_invalid_values(
    tmp_path: Path, runtime: str, message_parts: tuple[str, ...]
):
    backend_body = _explicit_backend(
        common=f"""
          system: h200_sxm
          backend: vllm
          runtime:
{textwrap.indent(runtime, "            ")}
        """
    )
    path = _write_yaml(tmp_path, _sim_yaml_with_backend(backend_body))

    _assert_config_error(path, *message_parts)


def test_explicit_aggregate_engine_resolves_without_disagg_roles(tmp_path: Path):
    backend_body = _explicit_backend(
        topology="agg",
        gpu_budget=12,
        model="name: aggregate-model",
        common="""
          system: b200_sxm
          backend: sglang
          tp_size: 2
          extra_args:
            block_size: 64
        """,
        roles={
            "aggregate": """
              backend_version: "0.5.10"
              attention_dp_size: 2
              extra_args:
                num_gpu_blocks: 4096
            """
        },
    )
    body = _sim_yaml_with_backend(
        backend_body,
        autoscalers="""
          - name: static-aggregate
            type: static
            config:
              num_decode: 3
        """,
    )

    backend = load_match_config(_write_yaml(tmp_path, body)).backend

    assert backend.substrate is None
    assert backend.model.name == "aggregate-model"
    assert backend.model.ais_model_path == "aggregate-model"
    assert backend.engines.prefill is None
    assert backend.engines.decode is None
    aggregate = backend.engines.aggregate
    assert aggregate is not None
    assert aggregate.system == "b200_sxm"
    assert aggregate.backend == "sglang"
    assert aggregate.backend_version == "0.5.10"
    assert aggregate.tp_size == 2
    assert aggregate.attention_dp_size == 2
    assert aggregate.num_gpus == 4
    assert aggregate.extra_args == {
        "block_size": 64,
        "num_gpu_blocks": 4096,
    }


def test_explicit_model_name_and_ais_lookup_path_are_distinct(tmp_path: Path):
    backend_body = _explicit_backend(
        model="""
          name: request-facing-model
          ais_model_path: org/perf-database-model
        """
    )

    model = load_match_config(
        _write_yaml(tmp_path, _sim_yaml_with_backend(backend_body))
    ).backend.model

    assert model.name == "request-facing-model"
    assert model.ais_model_path == "org/perf-database-model"


@pytest.mark.parametrize(
    ("backend_body", "message_parts"),
    [
        (
            "topology: disagg\ngpu_budget: 8",
            ("either substrate", "model and engines"),
        ),
        (
            _explicit_backend(model=None),
            ("backend.model", "required"),
        ),
        (
            _explicit_backend(include_engines=False),
            ("backend.engines", "required"),
        ),
        (
            """
              topology: disagg
              substrate: gpt_oss
              gpu_budget: 32
              model:
                name: ambiguous
              engines:
                common:
                  system: h200_sxm
                  backend: vllm
            """,
            ("either", "do not combine"),
        ),
        (
            _explicit_backend(gpu_budget=None),
            ("gpu_budget", "required", "explicit"),
        ),
    ],
    ids=[
        "no-preset-or-explicit-config",
        "missing-model",
        "missing-engines",
        "preset-and-explicit-are-ambiguous",
        "explicit-needs-gpu-budget",
    ],
)
def test_sim_model_engine_selection_rejects_missing_or_ambiguous_config(
    tmp_path: Path, backend_body: str, message_parts: tuple[str, ...]
):
    path = _write_yaml(tmp_path, _sim_yaml_with_backend(backend_body))

    _assert_config_error(path, *message_parts)


@pytest.mark.parametrize(
    ("backend_body", "message_parts"),
    [
        (
            _explicit_backend(
                topology="disagg",
                roles={"aggregate": "{}"},
            ),
            ("aggregate", "disagg"),
        ),
        (
            _explicit_backend(
                topology="agg",
                roles={"prefill": "{}", "decode": "{}"},
            ),
            ("prefill/decode", "agg"),
        ),
    ],
    ids=["aggregate-role-in-disagg", "disagg-roles-in-agg"],
)
def test_engine_roles_must_match_topology(
    tmp_path: Path, backend_body: str, message_parts: tuple[str, ...]
):
    path = _write_yaml(tmp_path, _sim_yaml_with_backend(backend_body))

    _assert_config_error(path, *message_parts)


@pytest.mark.parametrize("field", ["backend", "ais_backend"])
def test_engine_backend_names_are_strict(tmp_path: Path, field: str):
    common = (
        """
          system: h200_sxm
          backend: typo-runtime
        """
        if field == "backend"
        else """
          system: h200_sxm
          backend: vllm
          ais_backend: typo-runtime
        """
    )
    backend_body = _explicit_backend(common=common)
    path = _write_yaml(tmp_path, _sim_yaml_with_backend(backend_body))

    _assert_config_error(path, field, "sglang", "trtllm", "vllm")


@pytest.mark.parametrize(
    "reserved_name",
    [
        "ais_backend",
        "ais_system",
        "ais_model_path",
        "ais_tp_size",
        "ais_backend_version",
        "ais_moe_tp_size",
        "ais_moe_ep_size",
        "ais_attention_dp_size",
        "engine_type",
        "dp_size",
        "worker_type",
        "is_prefill",
        "is_decode",
        "startup_time",
        "kv_transfer_bandwidth",
        "kv_bytes_per_token",
    ],
)
def test_engine_extra_args_reject_runner_owned_fields(
    tmp_path: Path, reserved_name: str
):
    backend_body = _explicit_backend(
        common=f"""
          system: h200_sxm
          backend: vllm
          extra_args:
            {reserved_name}: forbidden
        """
    )
    path = _write_yaml(tmp_path, _sim_yaml_with_backend(backend_body))

    _assert_config_error(path, "extra_args", "runner-owned", reserved_name)


@pytest.mark.parametrize(
    ("moe_fields", "message_parts"),
    [
        (
            "moe_tp_size: 2",
            ("moe_tp_size", "moe_ep_size", "together"),
        ),
        (
            "moe_tp_size: 2\nmoe_ep_size: 1",
            ("world size", "4"),
        ),
    ],
    ids=["moe-pair-is-atomic", "moe-product-matches-world-size"],
)
def test_moe_parallelism_must_match_engine_world_size(
    tmp_path: Path, moe_fields: str, message_parts: tuple[str, ...]
):
    backend_body = _explicit_backend(
        common="\n".join(
            [
                "system: h200_sxm",
                "backend: vllm",
                "tp_size: 2",
                "attention_dp_size: 2",
                moe_fields,
            ]
        )
    )
    path = _write_yaml(tmp_path, _sim_yaml_with_backend(backend_body))

    _assert_config_error(path, *message_parts)


def test_asymmetric_start_fleet_uses_role_weighted_gpu_budget(tmp_path: Path):
    roles = {
        "prefill": "tp_size: 2",
        "decode": "tp_size: 4",
    }
    autoscalers = """
      - name: role-weighted-static
        type: static
        config:
          num_prefill: 2
          num_decode: 3
    """
    valid = _sim_yaml_with_backend(
        _explicit_backend(gpu_budget=16, roles=roles),
        autoscalers=autoscalers,
    )
    invalid = _sim_yaml_with_backend(
        _explicit_backend(gpu_budget=15, roles=roles),
        autoscalers=autoscalers,
    )

    config = load_match_config(_write_yaml(tmp_path, valid, name="valid.yaml"))
    assert config.backend.engines.prefill.num_gpus == 2
    assert config.backend.engines.decode.num_gpus == 4
    _assert_config_error(
        _write_yaml(tmp_path, invalid, name="invalid.yaml"),
        "starting fleet",
        "16 gpus",
        "gpu_budget=15",
    )


def test_asymmetric_dynamic_maximum_uses_role_weighted_gpu_budget(
    tmp_path: Path,
):
    roles = {
        "prefill": "tp_size: 2",
        "decode": "tp_size: 4",
    }
    autoscalers = """
      - name: role-weighted-reactive
        type: reactive
        start:
          prefill: 1
          decode: 1
        config:
          max_prefill: 3
          max_decode: 2
    """
    valid = _sim_yaml_with_backend(
        _explicit_backend(gpu_budget=14, roles=roles),
        autoscalers=autoscalers,
    )
    invalid = _sim_yaml_with_backend(
        _explicit_backend(gpu_budget=13, roles=roles),
        autoscalers=autoscalers,
    )

    load_match_config(_write_yaml(tmp_path, valid, name="valid.yaml"))
    _assert_config_error(
        _write_yaml(tmp_path, invalid, name="invalid.yaml"),
        "maximum fleet",
        "14 gpus",
        "gpu_budget=13",
    )


def test_explicit_model_rejects_legacy_replay_model_name(tmp_path: Path):
    backend_body = _explicit_backend(
        replay="model_name: legacy-model-name",
    )
    path = _write_yaml(tmp_path, _sim_yaml_with_backend(backend_body))

    _assert_config_error(
        path,
        "replay.model_name",
        "cannot be combined",
        "backend.model.name",
    )


@pytest.mark.parametrize(
    ("filename", "planner_body"),
    [
        (
            "reusable.yaml",
            """
              optimization_target: sla
              ttft_ms: 250
            """,
        ),
        (
            "reusable.json",
            '{"optimization_target": "sla", "ttft_ms": 250}',
        ),
    ],
)
def test_planner_config_path_loads_yaml_or_json_relative_to_match_file(
    tmp_path: Path,
    monkeypatch,
    filename: str,
    planner_body: str,
):
    config_dir = tmp_path / "config"
    planner_path = _write_yaml(
        config_dir / "planner",
        planner_body,
        name=filename,
    )
    backend_body = _explicit_backend() + f"\nplanner_config: planner/{filename}"
    match_path = _write_yaml(
        config_dir,
        _sim_yaml_with_backend(backend_body),
    )
    unrelated_cwd = tmp_path / "elsewhere"
    unrelated_cwd.mkdir()
    monkeypatch.chdir(unrelated_cwd)

    backend = load_match_config(match_path).backend

    assert backend.planner_config == {
        "optimization_target": "sla",
        "ttft_ms": 250,
    }
    assert backend.planner_config_path == planner_path.resolve()


def test_planner_config_still_accepts_an_inline_mapping(tmp_path: Path):
    backend_body = (
        _explicit_backend()
        + """
planner_config:
  optimization_target: sla
  ttft_ms: 250
"""
    )

    backend = load_match_config(
        _write_yaml(tmp_path, _sim_yaml_with_backend(backend_body))
    ).backend

    assert backend.planner_config == {
        "optimization_target": "sla",
        "ttft_ms": 250,
    }
    assert backend.planner_config_path is None


def test_planner_config_path_must_exist(tmp_path: Path):
    config_dir = tmp_path / "config"
    backend_body = _explicit_backend() + "\nplanner_config: planner/does-not-exist.yaml"
    path = _write_yaml(
        config_dir,
        _sim_yaml_with_backend(backend_body),
    )

    _assert_config_error(path, "planner_config", "cannot read", "does-not-exist")


@pytest.mark.parametrize(
    ("planner_body", "message_parts"),
    [
        ("[not, a, mapping]", ("planner_config", "mapping")),
        (
            "max_gpu_budget: 999",
            ("planner_config", "runner-owned", "max_gpu_budget"),
        ),
    ],
    ids=["non-mapping", "reserved-runner-owned-field"],
)
def test_planner_config_file_rejects_invalid_contents(
    tmp_path: Path,
    planner_body: str,
    message_parts: tuple[str, ...],
):
    config_dir = tmp_path / "config"
    _write_yaml(
        config_dir / "planner",
        planner_body,
        name="invalid.yaml",
    )
    backend_body = _explicit_backend() + "\nplanner_config: planner/invalid.yaml"
    path = _write_yaml(
        config_dir,
        _sim_yaml_with_backend(backend_body),
    )

    _assert_config_error(path, *message_parts)


def test_publish_destination_cannot_overwrite_planner_config_file(
    tmp_path: Path,
):
    config_dir = tmp_path / "config"
    _write_yaml(
        config_dir / "planner",
        "optimization_target: sla",
        name="reusable.yaml",
    )
    backend_body = _explicit_backend() + "\nplanner_config: planner/reusable.yaml"
    path = _write_yaml(
        config_dir,
        _sim_yaml_with_backend(
            backend_body,
            publish="""
              artifact_root: artifacts
              destinations:
                - type: json
                  path: planner/reusable.yaml
            """,
        ),
    )

    _assert_config_error(
        path,
        "publish destination",
        "must not overwrite",
        "input configuration file",
    )


def test_real_paths_resolve_relative_to_config_not_cwd(tmp_path: Path, monkeypatch):
    config_dir = tmp_path / "config"
    catalog_ref = os.path.relpath(ENDPOINT_CATALOG, config_dir)
    path = _write_yaml(
        config_dir,
        _real_yaml(
            catalog_ref,
            publish="""
              artifact_root: ../artifacts
              destinations:
                - type: json
                  path: ../published/results.json
            """,
        ),
    )
    unrelated_cwd = tmp_path / "elsewhere"
    unrelated_cwd.mkdir()
    monkeypatch.chdir(unrelated_cwd)

    config = load_match_config(path)

    assert Path(config.backend.endpoint_catalog) == ENDPOINT_CATALOG.resolve()
    assert Path(config.publish.artifact_root) == (tmp_path / "artifacts").resolve()
    json_destination = next(
        destination
        for destination in _sequence(config.publish.destinations)
        if _field(destination, "type") == "json"
    )
    assert (
        Path(_field(json_destination, "path"))
        == (tmp_path / "published" / "results.json").resolve()
    )


def test_matrix_cardinality_and_run_ids_are_deterministic_and_safe(tmp_path: Path):
    path = _write_yaml(
        tmp_path,
        _sim_yaml(
            name="matrix ids",
            autoscalers="""
              - name: static-small
                type: static
                config:
                  num_prefill: 1
                  num_decode: 1
              - name: static-large
                type: static
                config:
                  num_prefill: 4
                  num_decode: 4
            """,
            evaluations="""
              workloads: [flat, staircase]
            """,
            slo_profiles="""
              - name: latency
                ttft_ms: [250, 500]
                itl_ms: 50
            """,
            execution="""
              repetitions: 2
              max_runs: 16
            """,
        ),
    )

    config = load_match_config(path)
    runs = list(config.iter_runs())
    run_ids = [_run_id(run) for run in runs]

    assert config.expected_runs == 2 * 2 * 2 * 2 == len(runs)
    assert len(run_ids) == len(set(run_ids))
    assert run_ids == [_run_id(run) for run in config.iter_runs()]
    assert all(re.fullmatch(r"[A-Za-z0-9._-]+", run_id) for run_id in run_ids)
    assert {_run_autoscaler_name(run) for run in runs} == {
        "static-small",
        "static-large",
    }
    assert {_run_workload_name(run) for run in runs} == {"flat", "staircase"}
    assert len({_profile_name(_run_profile(run)) for run in runs}) == 2
    assert len({_field(run, "repetition") for run in runs}) == 2


def _assert_config_error(path: Path, *message_parts: str) -> None:
    with pytest.raises(MatchConfigError) as exc_info:
        load_match_config(path)
    message = str(exc_info.value).lower()
    for part in message_parts:
        assert part.lower() in message


def test_duplicate_yaml_key_is_rejected(tmp_path: Path):
    path = _write_yaml(
        tmp_path,
        _sim_yaml().replace(
            "name: unit-sim",
            "name: first-name\nname: second-name",
            1,
        ),
    )

    _assert_config_error(path, "duplicate", "name")


def test_unknown_top_level_key_is_rejected(tmp_path: Path):
    path = _write_yaml(
        tmp_path,
        _sim_yaml().replace(
            "name: unit-sim",
            "name: unit-sim\nmystery_option: true",
            1,
        ),
    )

    _assert_config_error(path, "mystery_option")


def test_real_backend_rejects_sim_only_metric(tmp_path: Path):
    catalog_ref = os.path.relpath(ENDPOINT_CATALOG, tmp_path)
    path = _write_yaml(
        tmp_path,
        _real_yaml(
            catalog_ref,
            metrics="""
              rank_by: goodput_rps
              include: [goodput_rps, gpu_hours]
            """,
        ),
    )

    _assert_config_error(path, "gpu_hours", "real")


def test_real_backend_requires_selected_endpoint_to_exist(tmp_path: Path):
    catalog_ref = os.path.relpath(ENDPOINT_CATALOG, tmp_path)
    path = _write_yaml(
        tmp_path,
        _real_yaml(catalog_ref, endpoint="does-not-exist"),
    )

    _assert_config_error(path, "does-not-exist", "endpoint")


def test_static_autoscaler_requires_both_fixed_counts(tmp_path: Path):
    path = _write_yaml(
        tmp_path,
        _sim_yaml(
            autoscalers="""
              - name: incomplete-static
                type: static
                config:
                  num_prefill: 2
            """
        ),
    )

    _assert_config_error(path, "num_decode")


def test_max_runs_rejects_accidental_matrix_explosion(tmp_path: Path):
    path = _write_yaml(
        tmp_path,
        _sim_yaml(
            evaluations="""
              workloads: [flat, staircase]
            """,
            slo_profiles="""
              - name: latency-grid
                ttft_ms: [200, 400]
                itl_ms: [25, 50]
            """,
            execution="""
              repetitions: 2
              max_runs: 15
            """,
        ),
    )

    # 1 autoscaler x 2 workloads x 4 SLA points x 2 repetitions = 16.
    _assert_config_error(path, "max_runs", "16")


def test_dynamic_autoscaler_maximum_must_fit_gpu_budget(tmp_path: Path):
    path = _write_yaml(
        tmp_path,
        _sim_yaml(
            autoscalers="""
              - name: keda-too-wide
                type: keda
                start:
                  prefill: 1
                  decode: 1
                config:
                  max_prefill: 8
                  max_decode: 8
            """
        ).replace(
            "  topology: disagg",
            "  topology: disagg\n  gpu_budget: 8",
            1,
        ),
    )

    _assert_config_error(path, "maximum fleet", "gpu_budget")


@pytest.mark.parametrize(
    ("start", "field", "default_max"),
    [
        ({"prefill": 1, "decode": 9}, "start.decode", "8"),
        ({"prefill": 17, "decode": 1}, "start.prefill", "16"),
    ],
)
def test_dynamic_start_cannot_exceed_omitted_adapter_default_maximum(
    tmp_path: Path,
    start: dict[str, int],
    field: str,
    default_max: str,
):
    path = _write_yaml(
        tmp_path,
        _sim_yaml(
            autoscalers=f"""
              - name: reactive-too-large
                type: reactive
                start:
                  prefill: {start["prefill"]}
                  decode: {start["decode"]}
            """
        ),
    )

    _assert_config_error(path, field, default_max)


@pytest.mark.parametrize(
    "autoscaler",
    [
        """
          - name: static-large
            type: static
            config:
              num_prefill: 17
              num_decode: 9
        """,
        """
          - name: planner-large-start
            type: planner
            start:
              prefill: 17
              decode: 9
        """,
    ],
)
def test_static_and_planner_starts_are_not_bound_by_rival_adapter_defaults(
    tmp_path: Path, autoscaler: str
):
    config = load_match_config(_write_yaml(tmp_path, _sim_yaml(autoscalers=autoscaler)))

    start = config.backend.autoscalers[0].start
    assert (start.prefill, start.decode) == (17, 9)


def test_html_publish_destination_resolves_relative_path_and_overwrite(
    tmp_path: Path,
):
    config_dir = tmp_path / "config"
    config = load_match_config(
        _write_yaml(
            config_dir,
            _sim_yaml(
                publish="""
                  artifact_root: artifacts
                  destinations:
                    - type: html
                      path: ../published/arena-report.html
                      overwrite: true
                """
            ),
        )
    )

    destinations = _sequence(config.publish.destinations)
    assert len(destinations) == 1
    destination = destinations[0]
    assert _field(destination, "type") == "html"
    assert (
        Path(_field(destination, "path"))
        == (tmp_path / "published" / "arena-report.html").resolve()
    )
    assert _field(destination, "overwrite") is True


def test_publish_destinations_reject_duplicate_path_across_json_and_html(
    tmp_path: Path,
):
    path = _write_yaml(
        tmp_path,
        _sim_yaml(
            publish="""
              artifact_root: artifacts
              destinations:
                - type: json
                  path: published/results
                - type: html
                  path: published/results
            """
        ),
    )

    _assert_config_error(
        path,
        "duplicate output destination",
        "published/results",
        "json",
    )


@pytest.mark.parametrize(
    "topology, override, expected_error",
    [
        ("disagg", "decode_kv_up: 0.2", "decode_kv_down must be <= decode_kv_up"),
        ("disagg", "decode_kv_down: 0.9", "decode_kv_down must be <= decode_kv_up"),
        (
            "disagg",
            "prefill_queue_up: 0",
            "prefill_queue_down must be <= prefill_queue_up",
        ),
        (
            "disagg",
            "prefill_queue_down: 5",
            "prefill_queue_down must be <= prefill_queue_up",
        ),
        ("agg", "agg_queue_up: 0", "agg_queue_down must be <= agg_queue_up"),
        ("agg", "agg_queue_down: 5", "agg_queue_down must be <= agg_queue_up"),
    ],
)
def test_reactive_threshold_order_includes_defaults(
    tmp_path, topology, override, expected_error
):
    body = _sim_yaml(
        autoscalers=f"- name: reactive\n  type: reactive\n  config: {{{override}}}"
    )
    body = body.replace("topology: disagg", f"topology: {topology}")
    path = _write_yaml(tmp_path, body)
    with pytest.raises(MatchConfigError, match=expected_error):
        load_match_config(path)


@pytest.mark.parametrize(
    "flag",
    [
        "--isl-block-size",
        "--prompt-input-tokens-block-size",
        "--synthetic-input-tokens-block-size",
    ],
)
@pytest.mark.parametrize("equals", [False, True])
def test_real_config_rejects_block_size_overrides(tmp_path, flag, equals):
    body = (ARENA_ROOT / "configs/match.real.example.yaml").read_text()
    body = body.replace(
        "endpoint_catalog: endpoints.example.yaml",
        f"endpoint_catalog: {ARENA_ROOT / 'configs/endpoints.example.yaml'}",
    )
    arguments = [f"{flag}=16"] if equals else [flag, "16"]
    body = body.replace("extra_args: []", f"extra_args: {json.dumps(arguments)}")
    with pytest.raises(MatchConfigError, match="managed by the Match Config runner"):
        load_match_config(_write_yaml(tmp_path, body))
