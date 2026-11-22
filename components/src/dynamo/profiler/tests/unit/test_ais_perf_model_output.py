# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the profiler writes for AIS: mocker worker arguments and the Planner value."""

import copy
import functools
import json
import logging
import re
import sys
from pathlib import Path

import pytest
import yaml

# These need the native bindings.
try:
    from dynamo.mocker.args import parse_args
    from dynamo.mocker.config import build_mocker_engine_args
    from dynamo.planner.config.aic_interpolation_spec import AICInterpolationSpec
    from dynamo.planner.config.parallelization import PickedParallelConfig
    from dynamo.planner.config.planner_config import (
        PlannerConfig,
        PlannerPreDeploymentSweepMode,
    )
    from dynamo.profiler import __main__ as profiler_main
    from dynamo.profiler.utils import dgd_generation
    from dynamo.profiler.utils.dgdr_v1beta1_types import (
        DynamoGraphDeploymentRequestSpec,
        FeaturesSpec,
        MockerSpec,
    )
except ImportError:
    pytest.skip("dynamo.llm bindings not available", allow_module_level=True)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
]

MODEL = "Qwen/Qwen3-32B"
SYSTEM = "h200_sxm"
DENSE = PickedParallelConfig(tp=2)
MOE = PickedParallelConfig(tp=1, dp=8, moe_tp=1, moe_ep=8)
WORKERS = ("prefill", "decode")


def _interpolation(backend="vllm", pick=DENSE, model=MODEL, system=SYSTEM):
    return AICInterpolationSpec(
        hf_id=model,
        system=system,
        backend=backend,
        isl=1000,
        osl=100,
        sweep_max_context_length=4096,
        prefill_interpolation_granularity=8,
        decode_interpolation_granularity=4,
        prefill_pick=pick,
        decode_pick=pick,
    )


def _worker_arguments():
    return ["--model-path", MODEL, "--model-name", MODEL]


def _perf_model(arguments):
    """Return the config a worker's ``--ais-perf-model`` value holds."""
    return json.loads(arguments[arguments.index("--ais-perf-model") + 1])


def _request(planner=None, model=MODEL):
    return DynamoGraphDeploymentRequestSpec(
        model=model,
        features=FeaturesSpec(planner=planner, mocker=MockerSpec(enabled=True)),
    )


def _planner(mode, ais_perf_model=None):
    return PlannerConfig(
        enable_throughput_scaling=True,
        enable_load_scaling=False,
        mode=mode,
        optimization_target="sla",
        pre_deployment_sweeping_mode=PlannerPreDeploymentSweepMode.Rapid,
        ais_perf_model=ais_perf_model,
    )


def _config(**keys):
    """Return a perf model config of the deployment the tests select, with ``keys``."""
    return {"model": MODEL, "system": SYSTEM, "backend": "vllm", **keys}


def _planner_value(request, prefill_pick, decode_pick):
    return dgd_generation.build_ais_perf_model_spec(
        request,
        best_prefill_pick=prefill_pick,
        best_decode_pick=decode_pick,
        resolved_backend="vllm",
        system=SYSTEM,
    )


def _generated_workers(request, interpolation):
    """Return the argument list of each generated mocker worker."""
    config = dgd_generation.generate_mocker_config(request, aic_spec=interpolation)
    return {
        component["name"]: component["podTemplate"]["spec"]["containers"][0]["args"]
        for component in config["spec"]["components"]
        if component["name"] in WORKERS
    }


@pytest.fixture
def database_version(monkeypatch):
    monkeypatch.setattr(
        dgd_generation, "get_latest_database_version", lambda **kwargs: "0.24.0"
    )


@pytest.fixture(scope="module")
def installed_database_version():
    """Return the version lookup of the installed timing data, with a cache.

    One lookup reads every installed data set and takes about a second.
    """
    return functools.cache(dgd_generation.get_latest_database_version)


@pytest.mark.parametrize(
    "backend,engine_type",
    [
        pytest.param("sglang", ["--engine-type", "sglang"], id="sglang"),
        # The mocker has no trtllm engine type; the worker keeps its own.
        pytest.param("trtllm", [], id="trtllm"),
    ],
)
def test_engine_type_is_written_for_an_engine_the_mocker_has(
    database_version, backend, engine_type
):
    out = dgd_generation._inject_mocker_ais_args(
        _worker_arguments(), _interpolation(backend), DENSE
    )

    assert _perf_model(out)["backend"] == backend
    assert out[out.index("--ais-perf-model") + 2 :] == engine_type


@pytest.mark.parametrize("worker", WORKERS)
@pytest.mark.parametrize(
    "backend,pick,with_version",
    [
        pytest.param("vllm", DENSE, True, id="vllm"),
        pytest.param("sglang", DENSE, True, id="sglang"),
        pytest.param("trtllm", DENSE, True, id="trtllm"),
        pytest.param("vllm", MOE, True, id="moe"),
        pytest.param("vllm", DENSE, False, id="no-database-version"),
    ],
)
@pytest.mark.usefixtures("no_ais_environment")
def test_old_and_new_arguments_give_same_engine_args(
    monkeypatch, installed_database_version, backend, pick, with_version, worker
):
    # The version comes from the installed timing data, which the mocker
    # checks the config against.
    monkeypatch.setattr(
        dgd_generation,
        "get_latest_database_version",
        installed_database_version if with_version else lambda **kwargs: None,
    )
    interpolation = _interpolation(backend, pick)
    start = _worker_arguments() + ["--disaggregation-mode", worker]
    start += ["--num-gpu-blocks-override", "1024"]
    old = dgd_generation._inject_mocker_ais_args_old_form(
        list(start), interpolation, pick
    )
    new = dgd_generation._inject_mocker_ais_args(list(start), interpolation, pick)
    assert ("backend_version" in _perf_model(new)) is with_version

    with pytest.warns(FutureWarning, match=r"use --ais-perf-model VALUE\.$"):
        old_engine_args = build_mocker_engine_args(parse_args(old))
    new_engine_args = build_mocker_engine_args(parse_args(new))

    assert new_engine_args == old_engine_args
    timing = new_engine_args["engine"]["timing_model"]["config"]
    assert (timing["backend"], timing["worker_type"]) == (backend, worker)


def test_profiler_built_planner_value_is_a_list(database_version):
    prefill_pick = PickedParallelConfig(tp=1)
    base = _config(backend_version="0.24.0")

    disagg = _planner_value(_request(_planner("disagg")), prefill_pick, MOE)
    agg = _planner_value(_request(_planner("agg")), prefill_pick, DENSE)

    # A key that would only repeat a default is not written: pp of 1 and the
    # MoE sizes of a dense pick.
    assert disagg == [
        {**base, "worker_type": "prefill", "tp": 1, "attention_dp": 1},
        {
            **base,
            "worker_type": "decode",
            "tp": 1,
            "attention_dp": 8,
            "moe_tp_size": 1,
            "moe_ep_size": 8,
        },
    ]
    assert agg == [{**base, "worker_type": "aggregated", "tp": 2, "attention_dp": 1}]


@pytest.mark.parametrize(
    "pick",
    [
        pytest.param(DENSE, id="dense"),
        pytest.param(MOE, id="moe"),
        pytest.param(PickedParallelConfig(tp=2, pp=2), id="two-pipeline-stages"),
    ],
)
def test_planner_entry_is_the_mocker_config_with_a_worker_type(database_version, pick):
    workers = _generated_workers(_request(), _interpolation(pick=pick))
    planner_value = _planner_value(_request(_planner("disagg")), pick, pick)

    assert planner_value == [
        {**_perf_model(workers[worker]), "worker_type": worker} for worker in WORKERS
    ]


def test_profiler_built_value_round_trips(database_version):
    request = _request(_planner("disagg"))
    prefill_pick = PickedParallelConfig(tp=1)
    value = _planner_value(request, prefill_pick, DENSE)
    planner_config = dgd_generation._build_planner_config(
        request, prefill_pick, DENSE, ais_perf_model=value
    )

    read_back = PlannerConfig.model_validate_json(planner_config.model_dump_json())

    assert read_back.ais_perf_model == value


def test_user_written_new_form_is_kept_as_written():
    written = [
        _config(worker_type="prefill", tp=2),
        _config(
            worker_type="decode",
            tp=2,
            estimator_config={"correction": {"enabled": False}},
        ),
    ]
    planner = _planner("disagg", copy.deepcopy(written))

    value = _planner_value(_request(planner), DENSE, DENSE)

    assert value == written
    assert value is not planner.ais_perf_model


def test_user_written_new_form_conflict_text():
    planner = _planner("decode", _config(tp=4))

    text = (
        "ais_perf_model entry for decode: tp=4 conflicts with selected deployment tp=2"
    )
    with pytest.raises(ValueError, match=f"^{re.escape(text)}$"):
        _planner_value(_request(planner), DENSE, DENSE)


def test_user_written_missing_field_is_compared_as_default():
    planner = _planner("decode", _config())

    with pytest.raises(
        ValueError, match="^ais_perf_model entry for decode: tp=1 conflicts"
    ):
        _planner_value(_request(planner), DENSE, DENSE)


def test_model_comes_from_model_path_or_request_model(database_version):
    interpolation = _interpolation()

    served = dgd_generation._inject_mocker_ais_args(
        ["--model-path", "served/model"], interpolation, DENSE
    )
    requested = dgd_generation._inject_mocker_ais_args([], interpolation, DENSE)

    assert _perf_model(served)["model"] == "served/model"
    assert _perf_model(requested)["model"] == interpolation.hf_id


@pytest.fixture
def profiling_runs(monkeypatch):
    """Replace the profiling run that ``main()`` starts; return the requests it got."""
    requests = []

    async def run_profile(dgdr, ops):
        requests.append(dgdr)

    monkeypatch.setattr(profiler_main, "run_profile", run_profile)
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield requests
    # main() configures the root logger and adds a log file handler to it.
    for handler in root.handlers[:]:
        if handler not in handlers:
            root.removeHandler(handler)
            handler.close()
    root.setLevel(level)


def test_profiler_main_warns_once_for_old_form(monkeypatch, tmp_path, profiling_runs):
    old_form = {"roles": {"decode": {"model": "m", "system": "s", "backend": "vllm"}}}
    config = {
        "model": MODEL,
        "features": {
            "planner": {
                "mode": "decode",
                "optimization_target": "sla",
                "ais_perf_model": old_form,
            }
        },
    }
    argv = ["profiler", "--config", json.dumps(config), "--output-dir", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", argv)

    text = (
        "ais_perf_model.roles is deprecated and will be removed in v1.9; use "
        "ais_perf_model with one perf model config or a list of configs "
        "(one per worker_type)."
    )
    with pytest.warns(FutureWarning, match=f"^{re.escape(text)}$") as caught:
        profiler_main.main()

    assert len(caught) == 1
    # The warning points at the line of main() that starts the profiling run.
    assert caught[0].filename == profiler_main.__file__
    assert len(profiling_runs) == 1


def _argument_lists(node):
    """Yield every list of a decoded YAML document that sets ``--ais-perf-model``."""
    if isinstance(node, dict):
        for value in node.values():
            yield from _argument_lists(value)
    elif isinstance(node, list):
        if "--ais-perf-model" in node:
            yield node
        for value in node:
            yield from _argument_lists(value)


def test_arguments_for_operator_fixture_input(monkeypatch):
    # The database version the operator fixtures were recorded with.
    monkeypatch.setattr(
        dgd_generation, "get_latest_database_version", lambda **kwargs: "0.30.0"
    )
    model = "Qwen/Qwen3-0.6B"
    pick = PickedParallelConfig(tp=1)
    workers = _generated_workers(
        _request(model=model), _interpolation(pick=pick, model=model, system="h100_sxm")
    )
    values = {
        arguments[arguments.index("--ais-perf-model") + 1]
        for arguments in workers.values()
    }
    assert len(values) == 1
    value = values.pop()

    # The operator's expected profiler output for the same input.
    repository_root = Path(__file__).resolve().parents[6]
    folder = repository_root / "deploy/operator/internal/controller/testdata/dgdr"
    for name, groups in {"deployment": 6, "grove": 4, "lws": 7}.items():
        text = (folder / name / "output.yaml").read_text(encoding="utf-8")
        written = [
            arguments[arguments.index("--ais-perf-model") + 1]
            for arguments in _argument_lists(list(yaml.safe_load_all(text)))
        ]
        assert written == [value] * groups, name
