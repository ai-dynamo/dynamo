# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the profiler writes for AIS: mocker worker arguments and the Planner value."""

import copy
import json
import logging
import re
import sys
import warnings
from pathlib import Path

import pytest
import yaml

from dynamo.common.ais_perf_model import ENV_VAR

# These need the native bindings; the import above does not.
try:
    from dynamo.mocker.args import parse_args
    from dynamo.mocker.config import build_mocker_engine_args
    from dynamo.planner.config.aic_interpolation_spec import AICInterpolationSpec
    from dynamo.planner.config.parallelization import PickedParallelConfig
    from dynamo.planner.config.planner_config import (
        AISPerfModelSpec,
        PlannerConfig,
        PlannerPreDeploymentSweepMode,
    )
    from dynamo.planner.core.perf_model import ais_adapter
    from dynamo.planner.core.perf_model.ais_adapter import PlannerEnginePerfModel
    from dynamo.planner.core.types import EngineCapabilities
    from dynamo.profiler import __main__ as profiler_main
    from dynamo.profiler.utils import dgd_generation
    from dynamo.profiler.utils.dgdr_v1beta1_types import (
        DynamoGraphDeploymentRequestSpec,
        FeaturesSpec,
        MockerSpec,
        OverridesSpec,
    )
    from dynamo.profiler.utils.profile_common import ProfilerOperationalConfig
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
NEW_FORM = (
    "ais_perf_model with one perf model config or a list of configs "
    "(one per worker_type)"
)


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


def _request(planner=None, override=None, model=MODEL):
    return DynamoGraphDeploymentRequestSpec(
        model=model,
        features=FeaturesSpec(planner=planner, mocker=MockerSpec(enabled=True)),
        overrides=OverridesSpec(dgd=override) if override is not None else None,
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


def _workers_with(inject, interpolation=None):
    """Return the worker argument lists with ``inject`` applied to each."""
    interpolation = interpolation or _interpolation()
    return {
        name: inject(list(arguments), interpolation, interpolation.decode_pick)
        for name, arguments in _generated_workers(_request(), None).items()
    }


@pytest.fixture
def database_version(monkeypatch):
    monkeypatch.setattr(
        dgd_generation, "get_latest_database_version", lambda **kwargs: "0.24.0"
    )


@pytest.mark.parametrize(
    "backend,pick,added",
    [
        pytest.param(
            "vllm",
            DENSE,
            ["--ais-perf-model"]
            + ["--ais-backend", "vllm", "--ais-backend-version", "0.24.0"]
            + ["--ais-system", "h200_sxm", "--ais-tp-size", "2"]
            + ["--ais-attention-dp-size", "1", "--engine-type", "vllm"],
            id="vllm",
        ),
        pytest.param(
            "sglang",
            DENSE,
            ["--ais-perf-model"]
            + ["--ais-backend", "sglang", "--ais-backend-version", "0.24.0"]
            + ["--ais-system", "h200_sxm", "--ais-tp-size", "2"]
            + ["--ais-attention-dp-size", "1", "--engine-type", "sglang"],
            id="sglang",
        ),
        pytest.param(
            "vllm",
            MOE,
            ["--ais-perf-model"]
            + ["--ais-backend", "vllm", "--ais-backend-version", "0.24.0"]
            + ["--ais-system", "h200_sxm", "--ais-tp-size", "1"]
            + ["--ais-moe-tp-size", "1", "--ais-moe-ep-size", "8"]
            + ["--ais-attention-dp-size", "8", "--engine-type", "vllm"],
            id="moe",
        ),
        pytest.param(
            "trtllm",
            DENSE,
            ["--ais-perf-model"]
            + ["--ais-backend", "trtllm", "--ais-backend-version", "0.24.0"]
            + ["--ais-system", "h200_sxm", "--ais-tp-size", "2"]
            + ["--ais-attention-dp-size", "1"],
            id="trtllm",
        ),
    ],
)
def test_old_argument_list_is_unchanged(database_version, backend, pick, added):
    out = dgd_generation._inject_mocker_ais_args_old_form(
        _worker_arguments(), _interpolation(backend, pick), pick
    )

    assert out == _worker_arguments() + added


@pytest.mark.parametrize("lookup", [None, lambda **kwargs: None])
def test_old_argument_list_without_database_version_is_unchanged(monkeypatch, lookup):
    monkeypatch.setattr(dgd_generation, "get_latest_database_version", lookup)

    out = dgd_generation._inject_mocker_ais_args_old_form(
        _worker_arguments(), _interpolation(), DENSE
    )

    assert out == _worker_arguments() + [
        "--ais-perf-model",
        "--ais-backend",
        "vllm",
        "--ais-system",
        "h200_sxm",
        "--ais-tp-size",
        "2",
        "--ais-attention-dp-size",
        "1",
        "--engine-type",
        "vllm",
    ]


def test_mocker_arguments_are_one_inline_json(database_version):
    out = dgd_generation._inject_mocker_ais_args(
        _worker_arguments(), _interpolation(), DENSE
    )

    assert out == _worker_arguments() + [
        "--ais-perf-model",
        '{"attention_dp":1,"backend":"vllm","backend_version":"0.24.0",'
        '"model":"Qwen/Qwen3-32B","system":"h200_sxm","tp":2}',
        "--engine-type",
        "vllm",
    ]


def test_moe_pick_writes_moe_sizes(database_version):
    moe = _perf_model(
        dgd_generation._inject_mocker_ais_args(
            _worker_arguments(), _interpolation(pick=MOE), MOE
        )
    )
    dense = _perf_model(
        dgd_generation._inject_mocker_ais_args(
            _worker_arguments(), _interpolation(), DENSE
        )
    )

    assert (moe["tp"], moe["attention_dp"]) == (1, 8)
    assert (moe["moe_tp_size"], moe["moe_ep_size"]) == (1, 8)
    assert "moe_tp_size" not in dense
    assert "moe_ep_size" not in dense


def test_trtllm_writes_backend_and_keeps_engine_type(database_version):
    out = dgd_generation._inject_mocker_ais_args(
        _worker_arguments(), _interpolation("trtllm"), DENSE
    )

    assert _perf_model(out)["backend"] == "trtllm"
    # The mocker has no trtllm engine type; the worker keeps its own.
    assert "--engine-type" not in out


@pytest.mark.parametrize("lookup", [None, lambda **kwargs: None])
def test_no_database_version_omits_key(monkeypatch, lookup):
    monkeypatch.setattr(dgd_generation, "get_latest_database_version", lookup)

    out = dgd_generation._inject_mocker_ais_args(
        _worker_arguments(), _interpolation(), DENSE
    )

    assert _perf_model(out) == {
        "attention_dp": 1,
        "backend": "vllm",
        "model": MODEL,
        "system": SYSTEM,
        "tp": 2,
    }


def test_injection_twice_gives_one_pair(database_version):
    once = dgd_generation._inject_mocker_ais_args(
        _worker_arguments(), _interpolation(), DENSE
    )

    twice = dgd_generation._inject_mocker_ais_args(list(once), _interpolation(), DENSE)

    assert twice == once
    assert twice.count("--ais-perf-model") == 1


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
def test_old_and_new_arguments_give_same_engine_args(
    monkeypatch, backend, pick, with_version, worker
):
    # A value set in the shell that runs the tests would reach the parser.
    monkeypatch.delenv(ENV_VAR, raising=False)
    if not with_version:
        monkeypatch.setattr(
            dgd_generation, "get_latest_database_version", lambda **kwargs: None
        )
    # Otherwise the version is looked up in the installed timing data, which
    # the mocker checks the config against.
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
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        new_engine_args = build_mocker_engine_args(parse_args(new))

    assert new_engine_args == old_engine_args
    timing = new_engine_args["engine"]["timing_model"]["config"]
    assert (timing["backend"], timing["worker_type"]) == (backend, worker)


def _generated_entry(worker_type, pick):
    moe = (pick.moe_tp, pick.moe_ep) != (1, 1)
    return {
        "model": MODEL,
        "system": SYSTEM,
        "backend": "vllm",
        "backend_version": "0.24.0",
        "worker_type": worker_type,
        "tp": pick.tp,
        "pp": 1,
        "attention_dp": pick.dp,
        "moe_tp_size": pick.moe_tp if moe else None,
        "moe_ep_size": pick.moe_ep if moe else None,
    }


def test_profiler_built_planner_value_is_a_list(database_version):
    prefill_pick = PickedParallelConfig(tp=1)

    disagg = _planner_value(_request(_planner("disagg")), prefill_pick, MOE)
    agg = _planner_value(_request(_planner("agg")), prefill_pick, DENSE)

    assert disagg == [
        {
            "model": MODEL,
            "system": SYSTEM,
            "backend": "vllm",
            "backend_version": "0.24.0",
            "worker_type": "prefill",
            "tp": 1,
            "pp": 1,
            "attention_dp": 1,
            "moe_tp_size": None,
            "moe_ep_size": None,
        },
        {
            "model": MODEL,
            "system": SYSTEM,
            "backend": "vllm",
            "backend_version": "0.24.0",
            "worker_type": "decode",
            "tp": 1,
            "pp": 1,
            "attention_dp": 8,
            "moe_tp_size": 1,
            "moe_ep_size": 8,
        },
    ]
    assert agg == [_generated_entry("aggregated", DENSE)]


class _ReadyEngine:
    """Stands in for the native engine model, which these tests do not use."""

    @classmethod
    def best_available(cls, **kwargs):
        return cls()

    def diagnostics(self):
        return {"readiness": "ready"}


def _adapter_config(planner_config, worker_type):
    limits = EngineCapabilities(
        max_num_batched_tokens=1024, max_num_seqs=128, max_kv_tokens=100_000
    )
    return PlannerEnginePerfModel(
        worker_type=worker_type, config=planner_config, capabilities=limits
    )._build_ais_config()


def test_profiler_built_value_round_trips_and_equals_old_form(
    database_version, monkeypatch
):
    monkeypatch.setattr(ais_adapter, "AISCoreEnginePerfModel", _ReadyEngine)
    request = _request(_planner("disagg"))
    prefill_pick = PickedParallelConfig(tp=1)
    value = _planner_value(request, prefill_pick, DENSE)
    planner_config = dgd_generation._build_planner_config(
        request, prefill_pick, DENSE, ais_perf_model=value
    )

    read_back = PlannerConfig.model_validate_json(planner_config.model_dump_json())

    assert read_back.ais_perf_model == value
    old_form = planner_config.model_copy(
        update={
            "ais_perf_model": AISPerfModelSpec(
                roles={entry["worker_type"]: entry for entry in value}
            )
        }
    )
    for worker_type in WORKERS:
        assert _adapter_config(read_back, worker_type) == _adapter_config(
            old_form, worker_type
        )


def _written_entry(**keys):
    return {"model": MODEL, "system": SYSTEM, "backend": "vllm", **keys}


@pytest.mark.parametrize(
    "mode,written",
    [
        pytest.param(
            "disagg",
            [
                _written_entry(worker_type="prefill", tp=2),
                _written_entry(
                    worker_type="decode",
                    tp=2,
                    estimator_config={"correction": {"enabled": False}},
                ),
            ],
            id="list",
        ),
        pytest.param("decode", _written_entry(tp=2), id="one-config"),
    ],
)
def test_user_written_new_form_is_kept_as_written(mode, written):
    planner = _planner(mode, copy.deepcopy(written))

    value = _planner_value(_request(planner), DENSE, DENSE)

    assert value == written
    assert value is not planner.ais_perf_model


def test_user_written_new_form_conflict_text():
    planner = _planner("decode", _written_entry(tp=4))

    text = (
        "ais_perf_model entry for decode: tp=4 conflicts with selected deployment tp=2"
    )
    with pytest.raises(ValueError, match=f"^{re.escape(text)}$"):
        _planner_value(_request(planner), DENSE, DENSE)


def test_user_written_from_entry_is_not_opened():
    written = {"from": "/missing.yaml"}
    planner = _planner("decode", dict(written))

    assert _planner_value(_request(planner), DENSE, DENSE) == written


def test_user_written_missing_field_is_compared_as_default():
    planner = _planner("decode", _written_entry())

    text = (
        "ais_perf_model entry for decode: tp=1 conflicts with selected deployment tp=2"
    )
    with pytest.raises(ValueError, match=f"^{re.escape(text)}$"):
        _planner_value(_request(planner), DENSE, DENSE)


def test_model_comes_from_model_path_or_request_model(database_version):
    interpolation = _interpolation()

    served = dgd_generation._inject_mocker_ais_args(
        ["--model-path", "served/model"], interpolation, DENSE
    )
    requested = dgd_generation._inject_mocker_ais_args([], interpolation, DENSE)

    assert _perf_model(served)["model"] == "served/model"
    assert _perf_model(requested)["model"] == interpolation.hf_id


def _override(shape, arguments):
    """Return a deployment override that gives ``arguments`` to a worker."""
    if shape.startswith("v1alpha1"):
        service = "VllmDecodeWorker" if shape == "v1alpha1-other-name" else "decode"
        override = {
            "apiVersion": "nvidia.com/v1alpha1",
            "kind": "DynamoGraphDeployment",
            "spec": {
                "services": {
                    service: {"extraPodSpec": {"mainContainer": {"args": arguments}}}
                }
            },
        }
        if shape == "v1alpha1-untyped":
            del override["apiVersion"], override["kind"]
        return override
    container = {"name": "main", "args": arguments}
    if shape == "v1beta1-append":
        container["$patch"] = {"args": "append"}
    return {
        "apiVersion": "nvidia.com/v1beta1",
        "kind": "DynamoGraphDeployment",
        "spec": {
            "components": [
                {
                    "name": "prefill",
                    "podTemplate": {"spec": {"containers": [container]}},
                }
            ]
        },
    }


OVERRIDE_SHAPES = [
    "v1alpha1",
    "v1alpha1-untyped",
    "v1alpha1-other-name",
    "v1beta1-replace",
    "v1beta1-append",
]
PERF_MODEL_VALUE = json.dumps({"model": MODEL, "system": SYSTEM, "backend": "vllm"})


@pytest.mark.parametrize("shape", OVERRIDE_SHAPES)
@pytest.mark.parametrize(
    "arguments,names",
    [
        pytest.param(["--ais-perf-config", "x"], "--ais-perf-config", id="old-name"),
        pytest.param(
            ["--speedup-ratio", "2.0", "--ais-perf-model"],
            "--ais-perf-model without a value",
            id="perf-model-last",
        ),
        pytest.param(
            ["--ais-perf-model", "--x"],
            "--ais-perf-model without a value",
            id="perf-model-before-flag",
        ),
        pytest.param(["--aic-system", "s"], "--aic-system", id="aic-name"),
        pytest.param(["--ais-system=s"], "--ais-system", id="equals-sign"),
        pytest.param(["--ais-sys", "s"], "--ais-sys", id="short-system"),
        pytest.param(["--ais-backend-v", "1"], "--ais-backend-v", id="short-version"),
        pytest.param(["--ais-tp", "2"], "--ais-tp", id="short-tp"),
        pytest.param(
            ["--extra-engine-args", "f"], "--extra-engine-args", id="engine-args"
        ),
        pytest.param(["--model-path", "m"], "--model-path", id="model-path"),
        pytest.param(["--model-p", "m"], "--model-p", id="short-model-path"),
        pytest.param(
            ["--ais-system", "s", "--model-path", "m", "--ais-system=t"],
            "--ais-system and --model-path",
            id="two-names",
        ),
    ],
)
def test_override_with_trigger_keeps_old_arguments(
    database_version, caplog, arguments, names, shape
):
    request = _request(override=_override(shape, arguments))

    with caplog.at_level(logging.WARNING):
        workers = _generated_workers(request, _interpolation())

    assert workers == _workers_with(dgd_generation._inject_mocker_ais_args_old_form)
    assert [record.getMessage() for record in caplog.records] == [
        f"overrides.dgd sets {names}: the generated mocker workers keep the old AIS "
        "arguments (--ais-perf-model without a value plus --ais-* flags), which are "
        "deprecated and will be removed in v1.9; to end this, write --ais-perf-model "
        "VALUE in the override and remove its old --ais-* arguments"
    ]


@pytest.mark.parametrize(
    "override",
    [
        pytest.param(None, id="no-override"),
        pytest.param(
            {
                "apiVersion": "nvidia.com/v1beta1",
                "kind": "DynamoGraphDeployment",
                "metadata": {"name": "renamed"},
            },
            id="metadata-only",
        ),
        *(
            pytest.param(_override(shape, arguments), id=f"{shape}:{arguments[0]}")
            for shape in ("v1alpha1", "v1beta1-replace")
            for arguments in (
                ["--ais-nextn", "2"],
                ["--aic-nextn", "2"],
                ["--ais-nextn-accept-rates", "0.5"],
                ["--ais-mtp-seed", "1"],
                ["--planner-profile-data", "x"],
                ["--ais-perf-model", PERF_MODEL_VALUE],
                [f"--ais-perf-model={PERF_MODEL_VALUE}"],
                ["--ais-perf-m", PERF_MODEL_VALUE],
                ["--model-name", "m"],
            )
        ),
        *(
            pytest.param(_override(shape, arguments), id=f"{shape}:{kind}")
            for shape in ("v1alpha1", "v1beta1-replace")
            for kind, arguments in (
                ("arguments-as-text", "--ais-system s"),
                ("argument-that-is-a-number", ["--ais-tp-size", 2]),
            )
        ),
    ],
)
def test_override_without_trigger_keeps_new_arguments(
    database_version, caplog, override
):
    with caplog.at_level(logging.WARNING):
        workers = _generated_workers(_request(override=override), _interpolation())

    assert workers == _workers_with(dgd_generation._inject_mocker_ais_args)
    assert caplog.records == []


def test_override_is_not_read_without_ais_workers(caplog):
    override = _override("v1alpha1", ["--ais-system", "s"])

    with caplog.at_level(logging.WARNING):
        workers = _generated_workers(_request(override=override), None)

    assert workers == _generated_workers(_request(), None)
    for arguments in workers.values():
        assert not [token for token in arguments if token.startswith("--ais-")]
    assert caplog.records == []


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


OLD_ROLES = {"roles": {"decode": {"model": "m", "system": "s", "backend": "vllm"}}}
OLD_DATASET_PROFILES = {
    "dataset_profiles": {
        "decode": {"root": "/not-mounted-yet/dataset", "manifest": "profile.json"}
    }
}


def _old_form_planner(ais_perf_model):
    return PlannerConfig(
        mode="decode", optimization_target="sla", ais_perf_model=ais_perf_model
    )


@pytest.mark.parametrize(
    "ais_perf_model,text",
    [
        pytest.param(
            OLD_ROLES,
            "ais_perf_model.roles is deprecated and will be removed in v1.9; "
            f"use {NEW_FORM}.",
            id="roles",
        ),
        pytest.param(
            OLD_DATASET_PROFILES,
            "ais_perf_model.dataset_profiles is deprecated and will be removed in "
            f"v1.9; use {NEW_FORM}.",
            id="dataset-profiles",
        ),
    ],
)
def test_profiler_main_warns_for_old_form(
    monkeypatch, tmp_path, profiling_runs, ais_perf_model, text
):
    request = _request(_old_form_planner(ais_perf_model))
    options = ProfilerOperationalConfig(output_dir=str(tmp_path))
    monkeypatch.setattr(profiler_main, "_parse_args", lambda: (request, options))

    with pytest.warns(FutureWarning, match=re.escape(text)) as caught:
        profiler_main.main()

    assert [str(warning.message) for warning in caught] == [text]
    assert profiling_runs == [request]


def test_profiler_main_warns_once_for_old_form(monkeypatch, tmp_path, profiling_runs):
    config = {
        "model": MODEL,
        "features": {
            "planner": {
                "mode": "decode",
                "optimization_target": "sla",
                "ais_perf_model": OLD_ROLES,
            }
        },
    }
    argv = ["profiler", "--config", json.dumps(config), "--output-dir", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", argv)

    text = (
        "ais_perf_model.roles is deprecated and will be removed in v1.9; "
        f"use {NEW_FORM}."
    )
    with pytest.warns(FutureWarning, match=f"^{re.escape(text)}$") as caught:
        profiler_main.main()

    assert len(caught) == 1
    # The warning points at the line of main() that starts the profiling run.
    assert caught[0].filename == profiler_main.__file__
    assert len(profiling_runs) == 1


@pytest.mark.parametrize(
    "with_planner,written",
    [
        pytest.param(False, None, id="no-planner"),
        pytest.param(True, None, id="no-perf-model"),
        pytest.param(True, _written_entry(tp=2), id="one-config"),
        pytest.param(True, [_written_entry(worker_type="decode", tp=2)], id="list"),
    ],
)
def test_profiler_main_is_silent_for_new_form(
    monkeypatch, tmp_path, profiling_runs, with_planner, written
):
    # The planner is built here: reading a written value needs the AISimulate
    # loader, and a missing loader must fail this test, not the whole file.
    planner = _planner("decode", written) if with_planner else None
    request = _request(planner)
    options = ProfilerOperationalConfig(output_dir=str(tmp_path))
    monkeypatch.setattr(profiler_main, "_parse_args", lambda: (request, options))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        profiler_main.main()

    assert profiling_runs == [request]


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


def test_arguments_for_operator_fixture_input(database_version):
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
