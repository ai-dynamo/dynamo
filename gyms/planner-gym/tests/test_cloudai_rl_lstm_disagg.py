# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CloudAI RL agg LSTM and disagg policies: encoders, networks, Match Config and wiring.

Pure-Python Match Config / factory tests run everywhere; the encoder, network
and adapter tests need torch plus the optional Dynamo types (and the Git LFS
checkpoints) and skip cleanly without them.
"""

from __future__ import annotations

import asyncio
import math
import sys
import textwrap
import types
from pathlib import Path
from typing import Any

import pytest
from autoscaling_arena import match_config, match_runner
from autoscaling_arena.match_config import (
    MatchConfigError,
    SimBackendConfig,
    load_match_config,
)

ARENA_ROOT = Path(__file__).resolve().parents[1]
PLANNER_CONFIG = ARENA_ROOT / "configs" / "planner.sim.example.yaml"
CHECKPOINTS = ARENA_ROOT / "checkpoints"
# Only the leaderboard exports (v3 + LSTM in each mode) are bundled.
AGG_LSTM = {"v3": CHECKPOINTS / "cql_autoscaler_best_v3_lstm.pt"}
DISAGG = {"disagg_v3_lstm": CHECKPOINTS / "cql_autoscaler_best_disagg_v3_lstm.pt"}


def _real(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 1024


def _write_config(tmp_path: Path, *, autoscalers: str, topology: str) -> Path:
    roles = (
        "aggregate: {}" if topology == "agg" else "prefill: {}\n            decode: {}"
    )
    entries = textwrap.indent(textwrap.dedent(autoscalers).strip(), "    ")
    body = textwrap.dedent(
        f"""
        schema_version: 1
        name: cloudai-rl-contract
        backend:
          type: sim
          topology: {topology}
          gpu_budget: 32
          model:
            name: openai/gpt-oss-120b
            ais_model_path: openai/gpt-oss-120b
          engines:
            common:
              system: h200_sxm
              backend: vllm
              backend_version: "0.24.0"
              tp_size: 1
              runtime: {{cold_start_delay_s: 30}}
            {roles}
          planner_config: {PLANNER_CONFIG}
          replay:
            telemetry_sample_interval_s: 5
          autoscalers:
        @@AUTOSCALERS@@
        evaluations:
          workloads: [flat]
          defaults:
            seed: 0
            max_requests: 50
            arrival_speedup: 1.0
        slo_profiles:
          - name: relaxed
            ttft_ms: 2000
            itl_ms: 50
        metrics:
          rank_by: goodput_per_gpu
          include: [goodput_per_gpu]
        execution:
          repetitions: 1
          fail_fast: false
        publish:
          artifact_root: out
          destinations:
            - type: console
        """
    ).lstrip()
    path = tmp_path / "match.yaml"
    path.write_text(body.replace("@@AUTOSCALERS@@", entries))
    return path


def _autoscalers(path: Path):
    backend = load_match_config(path).backend
    assert isinstance(backend, SimBackendConfig)
    return backend.autoscalers


# --------------------------------------------------------------------------- #
# Match Config
# --------------------------------------------------------------------------- #


def test_cloudai_rl_lstm_requires_checkpoint_and_shares_v3_defaults(tmp_path: Path):
    with pytest.raises(MatchConfigError, match="checkpoint_path: required"):
        load_match_config(
            _write_config(
                tmp_path,
                topology="agg",
                autoscalers="""
                - name: lstm
                  type: cloudai_rl_lstm
                  start: {decode: 1}
                """,
            )
        )
    (lstm,) = _autoscalers(
        _write_config(
            tmp_path,
            topology="agg",
            autoscalers=f"""
            - name: lstm
              type: cloudai_rl_lstm
              start: {{decode: 1}}
              config: {{checkpoint_path: {AGG_LSTM['v3']}}}
            """,
        )
    )
    assert lstm.config == {
        "checkpoint_path": str(AGG_LSTM["v3"]),
        "poll_interval_s": 5.0,
        "slo_ttft_ms": 2000.0,
        "slo_itl_ms": 50.0,
        "min_replicas": 1,
        "max_replicas": 8,
        "hidden_dim": 256,
        "num_blocks": 4,
        "cold_start_s": 30.0,
    }


def test_cloudai_rl_lstm_is_agg_only(tmp_path: Path):
    with pytest.raises(MatchConfigError, match="agg topology only"):
        load_match_config(
            _write_config(
                tmp_path,
                topology="disagg",
                autoscalers=f"""
                - name: lstm
                  type: cloudai_rl_lstm
                  start: {{prefill: 1, decode: 1}}
                  config: {{checkpoint_path: {AGG_LSTM['v3']}}}
                """,
            )
        )


def test_cloudai_rl_disagg_defaults_resolve_engines_and_pool_bounds(tmp_path: Path):
    (rl,) = _autoscalers(
        _write_config(
            tmp_path,
            topology="disagg",
            autoscalers=f"""
            - name: rl
              type: cloudai_rl_disagg
              start: {{prefill: 1, decode: 1}}
              config: {{checkpoint_path: {DISAGG['disagg_v3_lstm']}}}
            """,
        )
    )
    assert rl.config == {
        "checkpoint_path": str(DISAGG["disagg_v3_lstm"]),
        "poll_interval_s": 5.0,
        "slo_ttft_ms": 2000.0,
        "slo_itl_ms": 50.0,
        "min_prefill": 1,
        "max_prefill": 16,
        "min_decode": 1,
        "max_decode": 8,
        "smoothing_alpha": 0.4,
        "hidden_dim": 256,
        "num_blocks": 4,
        "cold_start_s": 30.0,
        "prefill_gpus_per_worker": 1,
        "decode_gpus_per_worker": 1,
    }


@pytest.mark.parametrize(
    ("autoscaler", "message_parts"),
    [
        (
            f"""
            - name: bad
              type: cloudai_rl_disagg
              start: {{prefill: 1, decode: 1}}
              config: {{checkpoint_path: {DISAGG['disagg_v3_lstm']}, max_replicas: 8}}
            """,
            ("unknown fields", "max_replicas"),
        ),
        (
            f"""
            - name: bad
              type: cloudai_rl_disagg
              start: {{prefill: 17, decode: 1}}
              config: {{checkpoint_path: {DISAGG['disagg_v3_lstm']}}}
            """,
            ("start.prefill", "[1, 16]"),
        ),
        (
            f"""
            - name: bad
              type: cloudai_rl_disagg
              start: {{prefill: 1, decode: 1}}
              config: {{checkpoint_path: {DISAGG['disagg_v3_lstm']}, max_prefill: 30}}
            """,
            ("maximum fleet requires", "gpu_budget=32"),
        ),
        (
            """
            - name: bad
              type: cloudai_rl_disagg
              start: {prefill: 1, decode: 1}
            """,
            ("checkpoint_path", "required"),
        ),
    ],
)
def test_cloudai_rl_disagg_config_errors(
    tmp_path: Path, autoscaler: str, message_parts
):
    with pytest.raises(MatchConfigError) as excinfo:
        load_match_config(
            _write_config(tmp_path, topology="disagg", autoscalers=autoscaler)
        )
    for part in message_parts:
        assert part in str(excinfo.value)


def test_cloudai_rl_disagg_is_disagg_only(tmp_path: Path):
    with pytest.raises(MatchConfigError, match="disagg topology only"):
        load_match_config(
            _write_config(
                tmp_path,
                topology="agg",
                autoscalers=f"""
                - name: rl
                  type: cloudai_rl_disagg
                  start: {{decode: 1}}
                  config: {{checkpoint_path: {DISAGG['disagg_v3_lstm']}}}
                """,
            )
        )


def test_committed_rl_leaderboard_configs_expand_seven_entries(
    monkeypatch: pytest.MonkeyPatch,
):
    if not (ARENA_ROOT / "data" / "golden-set" / "astra-gpt-oss").is_dir():
        # The Golden Set traces are gitignored (data/README.md). Without them,
        # check the configs' structure and skip only the trace-file checks.
        monkeypatch.setattr(
            match_config, "_require_trace_file", lambda source, *, path: None
        )
        monkeypatch.setattr(
            match_config, "validate_mooncake_trace", lambda *args, **kwargs: None
        )
    for mode, expected_type in (
        ("agg", "cloudai_rl_lstm"),
        ("disagg", "cloudai_rl_disagg"),
    ):
        config = load_match_config(
            ARENA_ROOT / "configs" / f"match.sim.{mode}.all-datasets.yaml"
        )
        backend = config.backend
        assert isinstance(backend, SimBackendConfig)
        types_by_name = {entry.name: entry.type for entry in backend.autoscalers}
        assert types_by_name["cloudai-rl"] == expected_type
        assert len(backend.autoscalers) == 7
        assert config.expected_runs == len(list(config.iter_runs())) == 294
        html = [d.path.name for d in config.publish.destinations if d.type == "html"]
        assert html == [f"report.{mode}.html"]


# --------------------------------------------------------------------------- #
# Sim factory
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("topology", "autoscaler_type", "class_name"),
    [
        ("agg", "cloudai_rl_lstm", "CloudAIRLAutoscaleLSTM"),
        ("disagg", "cloudai_rl_disagg", "CloudAIRLDisaggAutoscaler"),
    ],
)
def test_build_sim_factory_telemetry_rl_types_pass_decision_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    topology: str,
    autoscaler_type: str,
    class_name: str,
):
    checkpoint = AGG_LSTM["v3"] if topology == "agg" else DISAGG["disagg_v3_lstm"]
    start = "{decode: 1}" if topology == "agg" else "{prefill: 1, decode: 1}"
    (rl,) = _autoscalers(
        _write_config(
            tmp_path,
            topology=topology,
            autoscalers=f"""
            - name: rl
              type: {autoscaler_type}
              start: {start}
              config: {{checkpoint_path: {checkpoint}}}
            """,
        )
    )

    class Fake:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    fake_adapters = types.ModuleType("autoscaling_arena.adapters")
    setattr(fake_adapters, class_name, Fake)
    monkeypatch.setitem(sys.modules, "autoscaling_arena.adapters", fake_adapters)
    log = tmp_path / "rl-decisions.jsonl"
    instance = match_runner._build_sim_factory(rl, topology=topology, decision_log=log)(
        object(), "caps"
    )
    assert isinstance(instance, Fake)
    assert instance.kwargs == {
        **rl.config,
        "mode": topology,
        "capabilities": "caps",
        "decision_log": log,
    }


# --------------------------------------------------------------------------- #
# Encoders, networks and adapters (optional torch + Dynamo types)
# --------------------------------------------------------------------------- #


def _modules():
    pytest.importorskip("torch", reason="the CQL policies require torch")
    pytest.importorskip(
        "dynamo.planner.core.types", reason="requires the optional Dynamo runtime"
    )
    from autoscaling_arena.adapters import cloudai_rl_disagg as disagg
    from autoscaling_arena.adapters import cloudai_rl_lstm as lstm
    from autoscaling_arena.adapters import cql_common as common

    return common, lstm, disagg


def _agg_sample(
    at_s: float,
    *,
    arriving=0,
    completed=0,
    isl=0.0,
    osl=0.0,
    active=(0,),
    starting=(),
    waiting=0,
    kv=0.0,
):
    start = max(0.0, at_s - 5.0)
    return {
        "kind": "baseline" if at_s == 0 else "periodic",
        "interval_start_ms": start * 1000,
        "sampled_at_ms": at_s * 1000,
        "traffic": {
            "duration_s": at_s - start,
            "arriving_requests": arriving,
            "completed_requests": completed,
            "avg_isl": isl,
            "avg_osl": osl,
            "avg_ttft_ms": 500.0,
            "avg_itl_ms": 20.0,
            "ttft_count": completed,
            "itl_count": completed,
        },
        "decode_scheduler_metrics": [
            {
                "worker_id": w,
                "waiting_requests": waiting,
                "running_requests": 1,
                "active_cache_usage": kv,
            }
            for w in active
        ],
        "router_pending_decode_requests": 0,
        "active_decode_ids": list(active),
        "starting_decode_ids": list(starting),
        "draining_decode_ids": [],
    }


def _disagg_sample(
    at_s: float,
    *,
    arriving=20,
    completed=2,
    prefill=(0,),
    decode=(0,),
    starting_prefill=(),
    p_waiting=0,
    d_kv=0.1,
):
    start = max(0.0, at_s - 5.0)
    sample = {
        "kind": "baseline" if at_s == 0 else "periodic",
        "interval_start_ms": start * 1000,
        "sampled_at_ms": at_s * 1000,
        "traffic": {
            "duration_s": at_s - start,
            "arriving_requests": arriving,
            "completed_requests": completed,
            "avg_isl": 1024.0,
            "avg_osl": 128.0,
            "avg_ttft_ms": 2000.0,
            "avg_itl_ms": 50.0,
            "ttft_count": completed,
            "itl_count": completed,
        },
    }
    for pool, ids, starting in (
        ("prefill", prefill, starting_prefill),
        ("decode", decode, ()),
    ):
        sample.update(
            {
                f"active_{pool}_ids": list(ids),
                f"starting_{pool}_ids": list(starting),
                f"draining_{pool}_ids": [],
                f"router_pending_{pool}_requests": 0,
                f"{pool}_scheduler_metrics": [
                    {
                        "worker_id": w,
                        "waiting_requests": p_waiting if pool == "prefill" else 0,
                        "running_requests": 0,
                        "active_cache_usage": d_kv,
                    }
                    for w in ids
                ],
            }
        )
    return sample


def test_history_window_left_pads_and_lstm_ignores_padding():
    common, _, _ = _modules()
    import torch

    window = common.history_window([[1.0, 1.0], [2.0, 1.0]], 4)
    assert window.tolist() == [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [2.0, 1.0]]
    torch.manual_seed(0)
    network = common.LSTMQNetwork(
        state_dim=12, hidden_dim=16, num_blocks=0, lstm_hidden_dim=4
    ).eval()
    actual = torch.randn(2, 3, 12)
    actual[:, :, -1] = 1.0
    padded = torch.cat((torch.zeros(2, 7, 12), actual), dim=1)
    torch.testing.assert_close(network(actual), network(padded), rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError, match="valid current observation"):
        network(torch.zeros(1, 10, 12))
    branching = common.BranchingLSTMQNetwork(
        state_dim=20, hidden_dim=8, num_blocks=0, lstm_hidden_dim=4
    ).eval()
    assert tuple(branching(torch.ones(1, 10, 20)).shape) == (1, 2, 7)
    assert (
        "lstm.weight_ih_l0" in branching.state_dict()
        and "encoder.lstm.weight_ih_l0" in branching.state_dict()
    )


def _v3_sample(
    at_s: float,
    *,
    kind="periodic",
    arriving=0,
    completed=0,
    isl=0.0,
    osl=0.0,
    ttft_ms=0.0,
    itl_ms=0.0,
    ttft_count=0,
    itl_count=0,
    active=(0,),
    starting=(),
    draining=(),
    waiting=0,
    running=0,
    kv=0.0,
    pending=0,
):
    start = max(0.0, at_s - 5.0)
    return {
        "kind": kind,
        "interval_start_ms": start * 1000,
        "sampled_at_ms": at_s * 1000,
        "traffic": {
            "duration_s": at_s - start,
            "arriving_requests": arriving,
            "completed_requests": completed,
            "avg_isl": isl,
            "avg_osl": osl,
            "avg_ttft_ms": ttft_ms,
            "avg_itl_ms": itl_ms,
            "ttft_count": ttft_count,
            "itl_count": itl_count,
        },
        "decode_scheduler_metrics": [
            {
                "worker_id": w,
                "waiting_requests": waiting,
                "running_requests": running,
                "active_cache_usage": kv,
            }
            for w in [*active, *draining]
        ],
        "router_pending_decode_requests": pending,
        "active_decode_ids": list(active),
        "starting_decode_ids": list(starting),
        "draining_decode_ids": list(draining),
    }


def _log(value: float, scale: float) -> float:
    return math.log1p(value) / math.log1p(scale)


def test_v3_encoder_reproduces_the_training_feature_formulas():
    _, lstm, _ = _modules()
    encoder = lstm.V3StateEncoder(
        ttft_ms=2000.0, itl_ms=50.0, max_replicas=8, cold_start_s=30.0
    )
    first = encoder.observe(_v3_sample(0.0, kind="baseline", waiting=15))
    assert len(first) == len(lstm.V3_SMOOTHED_FEATURES) == 24
    assert first[3] == pytest.approx(_log(15, 50)) and first[6] == pytest.approx(1 / 8)
    assert first[9] == pytest.approx(1 / 8) and first[19] == 1.0 and first[23] == 0.0
    second = encoder.observe(
        _v3_sample(
            5.0,
            arriving=37,
            completed=3,
            isl=7257.0,
            osl=3.0,
            ttft_ms=1633.9,
            itl_ms=209.6,
            ttft_count=3,
            itl_count=3,
            starting=(1,),
            waiting=20,
            running=14,
            kv=0.084,
        )
    )
    alpha = 1 - 0.6**1.0
    assert second[0] == pytest.approx(
        _log(alpha * 37 / 5.0, 20)
    )  # smoothed arrival rate
    assert second[1] == pytest.approx(_log(7257.0, 4096)) and second[
        2
    ] == pytest.approx(_log(3.0, 1024))
    assert second[7] == pytest.approx(1 / 8) and second[9] == pytest.approx(2 / 8)
    assert (
        second[10] == second[11] == pytest.approx(25 / 30)
    )  # starting worker requested at the previous sample
    assert second[12] == 2.0 and second[13] == 2.0  # trends clamp at +2
    assert second[14] == pytest.approx(math.log1p(1633.9 / 2000.0)) and second[
        15
    ] == pytest.approx(math.log1p(209.6 / 50.0))
    assert second[16:20] == [1.0, 1.0, 1.0, 1.0] and second[21] == second[
        22
    ] == pytest.approx(_log(3, 50))
    third = encoder.observe(_v3_sample(10.0, starting=(1,)))
    assert third[10] == pytest.approx(20 / 30) and third[13] == 0.0 and third[18] == 0.0
    assert third[23] == pytest.approx(_log(5.0, 60))
    # Current-interval variant: no smoothing, zero shape when nothing completed, three extra fields.
    raw = lstm.RawV3Encoder(
        ttft_ms=2000.0, itl_ms=50.0, max_replicas=8, cold_start_s=30.0
    )
    raw.observe(_v3_sample(0.0, kind="baseline"))
    current = raw.observe(
        _v3_sample(5.0, arriving=100, completed=2, isl=4096.0, osl=128.0)
    )
    assert (
        len(current) == 27
        and current[0] == pytest.approx(_log(20.0, 20))
        and current[1] == 1.0
    )
    assert (
        current[24] == 0.0 and current[25] == 1.0 and current[26] == 1.0
    )  # trend invalid (first shape), 5 s, valid
    missing = raw.observe(_v3_sample(10.0, arriving=0, completed=0))
    assert missing[:3] == [0.0, 0.0, 0.0] and missing[18] == 0.0 and missing[23] > 0.0


def test_v3_encoder_rejects_inconsistent_telemetry():
    _, lstm, _ = _modules()
    encoder = lstm.V3StateEncoder()
    encoder.observe(_v3_sample(0.0, kind="baseline"))
    with pytest.raises(ValueError, match="advance in time"):
        encoder.observe(_v3_sample(0.0, kind="baseline"))
    bad_interval = _v3_sample(5.0)
    bad_interval["traffic"]["duration_s"] = 4.0
    with pytest.raises(ValueError, match="Inconsistent telemetry interval"):
        encoder.observe(bad_interval)
    with pytest.raises(ValueError, match="Overlapping worker lifecycle"):
        encoder.observe(_v3_sample(5.0, active=(0,), starting=(0,)))


def test_raw_v2_encoder_uses_current_values_and_float32_rounding():
    _, lstm, _ = _modules()
    encoder = lstm.RawV2Encoder(max_kv_tokens=100, observation_max_replicas=32)
    encoder.observe(_agg_sample(0.0))
    first = encoder.observe(
        _agg_sample(5.0, arriving=1000, completed=1, isl=4096.0, starting=(1,))
    )
    assert first[:2] == [1.0, 1.0]  # 200 rps / 200, 4096 / 4096
    assert first[4] == pytest.approx(
        2 / 32
    )  # active + starting over the v2 normalization
    assert first[7] == 0.5 and first[8] == 1.0 and first[-1] == 1.0
    assert first[-2] == 1.0  # interval / 5 s
    second = encoder.observe(_agg_sample(10.0, arriving=0, completed=0))
    assert second[:2] == [0.0, 0.0] and second[8] == 0.0 and second[9] == 0.0
    assert len(first) == len(lstm.V2_FEATURES) == 12
    assert len(lstm.V3_FEATURES) == 27


def test_disagg_encoder_contracts_and_masks():
    _, _, disagg = _modules()
    assert [
        len(disagg.feature_names(v, lstm))
        for v in ("v2", "v3")
        for lstm in (False, True)
    ] == [18, 20, 35, 37]
    assert disagg.schema_name("v3", True) == "disagg_v3_current_37"
    assert disagg.masks([1, 8], disagg.DEFAULT_SETTINGS) == [
        [False, False, False, True, True, True, True],
        [True, True, True, True, False, False, False],
    ]
    assert disagg.targets([3, 5], [3, 3], disagg.DEFAULT_SETTINGS) == [3, 5]
    with pytest.raises(ValueError, match="clamping is forbidden"):
        disagg.targets([1, 8], [0, 6], disagg.DEFAULT_SETTINGS)
    v2, v3, raw = (
        disagg.DisaggStateEncoder(version="v2"),
        disagg.DisaggStateEncoder(version="v3"),
        disagg.DisaggStateEncoder(version="v2", lstm=True),
    )
    first = _disagg_sample(5.0, starting_prefill=(1,))
    a2, a3, ar = (
        v2.observe(first, [1, 1]),
        v3.observe(first, [1, 1]),
        raw.observe(first, [1, 1]),
    )
    assert a3[:18] == a2
    assert a2[6] == pytest.approx(1 / 16) and a2[8] == pytest.approx(
        1 / 16
    )  # P requested, P starting
    assert a3[20] == pytest.approx(
        1.0
    )  # prefill next startup ETA: observed starting now
    assert a3[27] == pytest.approx(math.log1p(1.0))  # TTFT at the SLO
    assert ar[0] == pytest.approx(math.log1p(20 / 5) / math.log1p(20)) and ar[-2:] == [
        1.0,
        1.0,
    ]
    second = _disagg_sample(10.0, completed=0)
    b3, br = v3.observe(second, [1, 1]), raw.observe(second, [1, 1])
    assert b3[1:3] == a3[1:3] and br[1:3] == [0.0, 0.0] and b3[26] == 0.0
    with pytest.raises(ValueError, match="advance in time"):
        v3.observe(second, [1, 1])


def test_disagg_adapter_follows_requested_targets_and_rejects_runtime_drift(
    tmp_path: Path,
):
    _, _, disagg = _modules()
    if not all(_real(path) for path in DISAGG.values()):
        pytest.skip("disagg CQL checkpoints are Git LFS pointers; run `git lfs pull`")
    dynamo_types = pytest.importorskip("dynamo.planner.core.types")
    for name, path in DISAGG.items():
        rl = disagg.CloudAIRLDisaggAutoscaler(
            str(path), decision_log=tmp_path / f"{name}.jsonl"
        )
        assert rl.model == name and rl.consumes_replay_telemetry
        assert rl.requested is None
        rl.on_telemetry(_disagg_sample(0.0, arriving=0, completed=0))
        assert rl.requested == [1, 1]

        def tick(at_s, prefill_ready, decode_ready):
            counts = dynamo_types.WorkerCounts(
                ready_num_prefill=prefill_ready,
                ready_num_decode=decode_ready,
                expected_num_prefill=prefill_ready,
                expected_num_decode=decode_ready,
            )
            return dynamo_types.ScheduledTick(
                at_s=at_s, need_worker_states=True
            ), dynamo_types.TickInput(now_s=at_s, worker_counts=counts)

        effects = asyncio.run(rl.tick(*tick(0.0, 1, 1)))
        target = [effects.scale_to.num_prefill, effects.scale_to.num_decode]
        assert 1 <= target[0] <= 16 and 1 <= target[1] <= 8
        assert rl.requested == target
        assert effects.next_tick.at_s == pytest.approx(5.0)
        # The next sample must show the requested fleet (active + starting);
        # anything else means the runtime ignored the target.
        drifted = _disagg_sample(5.0, prefill=(0,), decode=(0,))
        drifted["starting_prefill_ids"] = list(range(1, target[0] + 2))
        rl.on_telemetry(drifted)
        with pytest.raises(ValueError, match="Runtime changed requested targets"):
            asyncio.run(rl.tick(*tick(5.0, 1, 1)))
        asyncio.run(rl.shutdown())
    with pytest.raises(ValueError, match="settings mismatch"):
        disagg.CloudAIRLDisaggAutoscaler(str(DISAGG["disagg_v3_lstm"]), max_prefill=8)
    with pytest.raises(ValueError, match="disagg topology only"):
        disagg.CloudAIRLDisaggAutoscaler(str(DISAGG["disagg_v3_lstm"]), mode="agg")


def test_agg_lstm_adapter_decides_over_bounded_history(tmp_path: Path):
    _, lstm, _ = _modules()
    if not all(_real(path) for path in AGG_LSTM.values()):
        pytest.skip("agg LSTM checkpoints are Git LFS pointers; run `git lfs pull`")
    dynamo_types = pytest.importorskip("dynamo.planner.core.types")
    for version, path in AGG_LSTM.items():
        rl = lstm.CloudAIRLAutoscaleLSTM(
            str(path), decision_log=tmp_path / f"{version}.jsonl"
        )
        assert rl.version == f"{version}_lstm_current" and rl.history_length == 10
        for step in range(12):
            at = 5.0 * step
            rl.on_telemetry(
                _agg_sample(
                    at,
                    arriving=30 if step else 0,
                    completed=3 if step else 0,
                    isl=2000.0,
                    osl=100.0,
                    waiting=5,
                )
            )
            counts = dynamo_types.WorkerCounts(
                ready_num_decode=1, expected_num_decode=1
            )
            effects = asyncio.run(
                rl.tick(
                    dynamo_types.ScheduledTick(at_s=at, need_worker_states=True),
                    dynamo_types.TickInput(now_s=at, worker_counts=counts),
                )
            )
            assert 1 <= effects.scale_to.num_decode <= 8
        assert len(rl._history) == 10 and rl.decisions == 12
        asyncio.run(rl.shutdown())
    with pytest.raises(ValueError, match="max_kv_tokens"):
        lstm.CloudAIRLAutoscaleLSTM(
            str(AGG_LSTM["v3"]),
            capabilities=dynamo_types.WorkerCapabilities(
                decode=dynamo_types.EngineCapabilities(max_kv_tokens=1)
            ),
        )
    with pytest.raises(ValueError, match="Expected an agg LSTM checkpoint"):
        lstm.CloudAIRLAutoscaleLSTM(str(DISAGG["disagg_v3_lstm"]))
