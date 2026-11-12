# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from dynamo.common import ais_perf_model
from dynamo.common.configuration.utils import Deprecated

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]

NEW_FLAG = "--ais-perf-model"
ENGINE_ARGS_FLAG = "--extra-engine-args"


def _config(**keys: Any) -> dict[str, Any]:
    """One perf model config; ``keys`` add to it or replace its values."""
    return {"model": "Qwen/Qwen3-0.6B", "system": "h200_sxm", "backend": "vllm", **keys}


def _yaml(config: dict[str, Any]) -> str:
    # JSON is YAML, so a config file needs no YAML writer here.
    return json.dumps(config)


def _names() -> dict[str, str]:
    return {"engine_args_name": ENGINE_ARGS_FLAG, "perf_model_name": NEW_FLAG}


def _run_child(script: str, **env: str) -> str:
    """Run ``script`` in a new interpreter that sees the same code as this one."""
    child_env = dict(
        os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p), **env
    )
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        env=child_env,
        capture_output=True,
        text=True,
        timeout=50,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return proc.stdout


def _report(stacklevel: int) -> None:
    ais_perf_model.warn_old_inputs(["--ais-backend"], NEW_FLAG, stacklevel=stacklevel)


def _validate(stacklevel: int) -> None:
    _report(stacklevel)


class _RecordingLoader:
    """Stands in for the loader and keeps the arguments of every resolve call."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def resolve(self, entry: Any, **keywords: Any) -> dict[str, Any]:
        self.calls.append(keywords)
        return dict(entry)


@pytest.fixture
def dataset_workspaces(monkeypatch: pytest.MonkeyPatch):
    """Give the test its own, empty set of dataset folders and remove them after."""
    workspaces: dict[int, Any] = {}
    monkeypatch.setattr(ais_perf_model, "_dataset_workspaces", workspaces)
    yield workspaces
    for folder in workspaces.values():
        folder.cleanup()


def test_old_inputs_message_one_two_three_names() -> None:
    assert (
        ais_perf_model.old_inputs_message(["--ais-backend"], NEW_FLAG)
        == "--ais-backend is deprecated and will be removed in v1.9; "
        "use --ais-perf-model."
    )
    assert (
        ais_perf_model.old_inputs_message(["--ais-backend", "--ais-system"], NEW_FLAG)
        == "--ais-backend and --ais-system are deprecated and will be removed in "
        "v1.9; use --ais-perf-model."
    )
    assert (
        ais_perf_model.old_inputs_message(
            ["--ais-backend", "--ais-system", "DYN_AIS_MODEL_PATH"], NEW_FLAG
        )
        == "--ais-backend, --ais-system and DYN_AIS_MODEL_PATH are deprecated and "
        "will be removed in v1.9; use --ais-perf-model."
    )
    assert ais_perf_model.join_names(["a"]) == "a"
    assert ais_perf_model.join_names(["a", "b"]) == "a and b"
    assert ais_perf_model.join_names(("a", "b", "c", "d")) == "a, b, c and d"
    with pytest.raises(ValueError, match="^old_names is empty$"):
        ais_perf_model.old_inputs_message([], NEW_FLAG)


def test_old_inputs_message_matches_existing_helper_for_one_name() -> None:
    existing = Deprecated(NEW_FLAG, remove_in=ais_perf_model.REMOVE_IN)

    assert ais_perf_model.old_inputs_message(
        ["--ais-perf-config"], NEW_FLAG
    ) == existing.message("--ais-perf-config")


def test_warn_old_inputs_is_future_warning_at_caller() -> None:
    with pytest.warns(FutureWarning) as records:
        ais_perf_model.warn_old_inputs(["--ais-perf-config"], NEW_FLAG, stacklevel=1)

    assert len(records) == 1
    assert records[0].category is FutureWarning
    assert str(records[0].message) == (
        "--ais-perf-config is deprecated and will be removed in v1.9; "
        "use --ais-perf-model."
    )
    assert records[0].filename == __file__


def test_warn_old_inputs_stacklevel_counts_callers() -> None:
    with pytest.warns(FutureWarning) as reporting_line:
        _validate(stacklevel=1)
    with pytest.warns(FutureWarning) as callers_line:
        _validate(stacklevel=2)

    # Each helper is one line of code directly under its ``def`` line.
    assert reporting_line[0].lineno == _report.__code__.co_firstlineno + 1
    assert callers_line[0].lineno == _validate.__code__.co_firstlineno + 1
    assert callers_line[0].filename == __file__


def test_warn_old_inputs_warns_on_every_call() -> None:
    """Nothing is remembered: one warning per process needs one call per process."""
    with pytest.warns(FutureWarning) as one_after_the_other:
        ais_perf_model.warn_old_inputs(["--ais-backend"], NEW_FLAG, stacklevel=1)
        ais_perf_model.warn_old_inputs(["--ais-system"], NEW_FLAG, stacklevel=1)
    with pytest.warns(FutureWarning) as together:
        ais_perf_model.warn_old_inputs(
            ["--ais-backend", "--ais-system"], NEW_FLAG, stacklevel=1
        )

    assert [str(record.message) for record in one_after_the_other] == [
        "--ais-backend is deprecated and will be removed in v1.9; "
        "use --ais-perf-model.",
        "--ais-system is deprecated and will be removed in v1.9; "
        "use --ais-perf-model.",
    ]
    assert [str(record.message) for record in together] == [
        "--ais-backend and --ais-system are deprecated and will be removed in "
        "v1.9; use --ais-perf-model."
    ]


def test_timing_model_shape_and_copy() -> None:
    config = _config(estimator_config={"correction": {"enabled": False}})
    as_given = copy.deepcopy(config)

    result = ais_perf_model.timing_model(config)

    assert result == {"type": "external", "provider": "ais", "config": as_given}
    result["config"]["estimator_config"]["correction"]["enabled"] = True
    result["config"]["model"] = "changed"
    assert config == as_given


def test_set_timing_model_none_text_and_mapping() -> None:
    config = _config(worker_type="decode")
    timing = {"type": "external", "provider": "ais", "config": _config()}
    timing["config"]["worker_type"] = "decode"
    written = {"engine": {"block_size": 32}, "num_workers": 2}
    as_given = copy.deepcopy(written)

    assert ais_perf_model.set_timing_model(None, config, **_names()) == {
        "engine": {"timing_model": timing}
    }
    assert ais_perf_model.set_timing_model(json.dumps(written), config, **_names()) == {
        "engine": {"block_size": 32, "timing_model": timing},
        "num_workers": 2,
    }
    assert ais_perf_model.set_timing_model({"num_workers": 2}, config, **_names()) == {
        "engine": {"timing_model": timing},
        "num_workers": 2,
    }
    result = ais_perf_model.set_timing_model(written, config, **_names())
    assert result == {
        "engine": {"block_size": 32, "timing_model": timing},
        "num_workers": 2,
    }
    assert written == as_given
    assert config == _config(worker_type="decode")

    assert ais_perf_model.engine_args_as_written(None, **_names()) == {}
    copied = ais_perf_model.engine_args_as_written(written, **_names())
    copied["engine"]["block_size"] = 64
    assert written == as_given


@pytest.mark.parametrize(
    "engine_args",
    [
        {"engine": {"timing_model": {"type": "polynomial"}}},
        {"engine": {"timing_model": None}},
        '{"engine": {"timing_model": {"type": "external"}}}',
    ],
)
def test_set_timing_model_refuses_written_timing_model(engine_args: Any) -> None:
    message = (
        "prefill_engine_args already sets engine.timing_model; it cannot be "
        "combined with ais_perf_model"
    )
    names = {
        "engine_args_name": "prefill_engine_args",
        "perf_model_name": "ais_perf_model",
    }

    with pytest.raises(ValueError) as from_set:
        ais_perf_model.set_timing_model(engine_args, _config(), **names)
    with pytest.raises(ValueError) as from_read:
        ais_perf_model.engine_args_as_written(engine_args, **names)

    assert str(from_set.value) == message
    assert str(from_read.value) == message


@pytest.mark.parametrize(
    "engine_args",
    ["[1, 2]", "3", "null", "not json", '{"engine": 3}', {"engine": ["a"]}, 3],
)
def test_engine_args_as_written_rejects_non_object(engine_args: Any) -> None:
    with pytest.raises(ValueError) as error:
        ais_perf_model.engine_args_as_written(engine_args, **_names())

    assert str(error.value) == "--extra-engine-args must be a JSON object"


def test_env_value_empty_is_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ais_perf_model.ENV_VAR, raising=False)
    assert ais_perf_model.env_value() is None

    monkeypatch.setenv(ais_perf_model.ENV_VAR, "")
    assert ais_perf_model.env_value() is None

    monkeypatch.setenv(ais_perf_model.ENV_VAR, "cfg.yaml")
    assert ais_perf_model.env_value() == "cfg.yaml"

    monkeypatch.setenv(ais_perf_model.WORKER_TYPE_ENV_VAR, "prefill")
    assert ais_perf_model.env_value(ais_perf_model.WORKER_TYPE_ENV_VAR) == "prefill"
    assert ais_perf_model.ENV_VAR == "DYN_AIS_PERF_MODEL"
    assert ais_perf_model.WORKER_TYPE_ENV_VAR == "DYN_ROUTER_PREFILL_LOAD_WORKER_TYPE"


@pytest.mark.timeout(60)  # one interpreter start, about 1s
def test_import_does_not_load_aisimulate() -> None:
    output = _run_child(
        """
        import sys

        import dynamo.common.ais_perf_model as module

        print(module.__file__)
        print(sorted(name for name in sys.modules if name.startswith("aisimulate")))
        """
    )

    module_file, loaded = output.strip().splitlines()
    # The child must test the same code as this process.
    assert module_file == ais_perf_model.__file__
    assert loaded == "[]"


@pytest.mark.timeout(60)  # one interpreter start, about 1s
def test_functions_without_a_value_work_when_aisimulate_is_missing() -> None:
    output = _run_child(
        """
        import json
        import sys
        import warnings

        # A None entry makes every import of that name fail in this interpreter.
        sys.modules["aisimulate_core"] = None
        sys.modules["aisimulate"] = None

        from dynamo.common import ais_perf_model

        names = {"engine_args_name": "--extra-engine-args", "perf_model_name": "x"}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ais_perf_model.warn_old_inputs(["--ais-backend"], "x", stacklevel=1)
        import_errors = {}
        for name, call in {
            "load_document": lambda: ais_perf_model.load_document({"model": "m"}),
            "resolve_entry": lambda: ais_perf_model.resolve_entry({"model": "m"}),
            "router_config": lambda: ais_perf_model.router_config({"model": "m"}),
            "uses_relative_paths": lambda: ais_perf_model.uses_relative_paths({}),
        }.items():
            try:
                call()
            except ImportError as error:
                import_errors[name] = error.name
        print(
            json.dumps(
                {
                    "env_value": ais_perf_model.env_value("DYN_TEST_UNSET_NAME"),
                    "timing_model": ais_perf_model.timing_model({"model": "m"}),
                    "engine_args_as_written": ais_perf_model.engine_args_as_written(
                        '{"a": 1}', **names
                    ),
                    "set_timing_model": ais_perf_model.set_timing_model(
                        None, {"model": "m"}, **names
                    ),
                    "join_names": ais_perf_model.join_names(["a", "b"]),
                    "old_inputs_message": ais_perf_model.old_inputs_message(["a"], "x"),
                    "warnings": [str(warning.message) for warning in caught],
                    "import_errors": import_errors,
                }
            )
        )
        """
    )

    timing = {"type": "external", "provider": "ais", "config": {"model": "m"}}
    assert json.loads(output) == {
        "env_value": None,
        "timing_model": timing,
        "engine_args_as_written": {"a": 1},
        "set_timing_model": {"engine": {"timing_model": timing}},
        "join_names": "a and b",
        "old_inputs_message": "a is deprecated and will be removed in v1.9; use x.",
        "warnings": ["--ais-backend is deprecated and will be removed in v1.9; use x."],
        # The functions that read a value fail with the import error itself.
        "import_errors": {
            "load_document": "aisimulate_core",
            "resolve_entry": "aisimulate_core",
            "router_config": "aisimulate_core",
            "uses_relative_paths": "aisimulate_core",
        },
    }


def test_dataset_workspace_only_for_dataset_entries(
    monkeypatch: pytest.MonkeyPatch, dataset_workspaces: dict[int, Any]
) -> None:
    loader = _RecordingLoader()
    monkeypatch.setattr(ais_perf_model, "_loader", lambda: loader)
    dataset = {"from": {"dataset": {"root": "/data", "manifest": "manifest.json"}}}

    ais_perf_model.resolve_entry(_config())
    ais_perf_model.resolve_entry({"from": "template.yaml"})
    ais_perf_model.resolve_entry({"from": {"state_snapshot": {"path": "/snapshot"}}})
    assert [call["workspace"] for call in loader.calls] == [None, None, None]
    assert dataset_workspaces == {}

    ais_perf_model.resolve_entry(
        dataset, worker_type="decode", fill={"nextn": 2}, defaults={"tp": 1}
    )
    ais_perf_model.resolve_entry(dataset)
    first, second = loader.calls[3], loader.calls[4]
    assert os.path.basename(first["workspace"]).startswith("dynamo-ais-perf-model-")
    assert os.path.isdir(first["workspace"])
    assert second["workspace"] == first["workspace"]
    assert list(dataset_workspaces) == [os.getpid()]
    assert first == {
        "worker_type": "decode",
        "fill": {"nextn": 2},
        "defaults": {"tp": 1},
        "workspace": first["workspace"],
    }


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
@pytest.mark.timeout(60)  # one interpreter start and one fork, about 1s
def test_forked_child_gets_its_own_dataset_workspace(tmp_path: Path) -> None:
    output = _run_child(
        """
        import os

        from dynamo.common import ais_perf_model

        parent_folder = ais_perf_model._dataset_workspace()
        read_end, write_end = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.write(write_end, ais_perf_model._dataset_workspace().encode())
            os._exit(0)
        os.waitpid(pid, 0)
        child_folder = os.read(read_end, 4096).decode()
        print(child_folder != parent_folder)
        print(os.path.isdir(parent_folder))
        print(ais_perf_model._dataset_workspace() == parent_folder)
        """,
        # The child of the fork leaves its folder behind; keep it under tmp_path.
        TMPDIR=str(tmp_path),
    )

    assert output.split() == ["True", "True", "True"]


def test_load_document_accepts_file_inline_json_mapping_list(tmp_path: Path) -> None:
    prefill = _config(worker_type="prefill")
    decode = _config(worker_type="decode")
    path = tmp_path / "cfg.yaml"
    path.write_text(_yaml(prefill))

    from_file = ais_perf_model.load_document(str(path))
    assert (from_file.bare, from_file.source) == (True, str(path))
    assert from_file.worker_types == ("prefill",)
    assert from_file.to_data() == prefill

    from_json = ais_perf_model.load_document(json.dumps([prefill, decode]))
    assert (from_json.bare, from_json.source) == (False, None)
    assert from_json.worker_types == ("prefill", "decode")
    assert from_json.to_data() == [prefill, decode]

    from_mapping = ais_perf_model.load_document(_config())
    assert (from_mapping.bare, from_mapping.source) == (True, None)
    assert from_mapping.worker_types == ()
    assert from_mapping.to_data() == _config()

    from_list = ais_perf_model.load_document([decode, prefill])
    assert (from_list.bare, from_list.source) == (False, None)
    assert from_list.worker_types == ("decode", "prefill")
    assert from_list.to_data() == [decode, prefill]
    assert from_list.select("prefill") == prefill

    # A template is not opened when the value is read.
    unread = ais_perf_model.load_document({"from": str(tmp_path / "missing.yaml")})
    assert unread.to_data() == {"from": str(tmp_path / "missing.yaml")}


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("", "ais_perf_model: the value is empty; write a file path or inline JSON"),
        ({}, "ais_perf_model: the config is empty; write at least one key"),
        ([], "ais_perf_model: the list is empty; write at least one config"),
        (
            3,
            "ais_perf_model: expected one config (a mapping) or a list of configs, "
            "got int 3",
        ),
        (
            _config(worker_type="router"),
            'ais_perf_model: worker_type is "router"; use one of: prefill, decode, '
            "aggregated",
        ),
        (
            [_config(worker_type="prefill"), _config(worker_type="prefill")],
            "ais_perf_model: entries 1 and 2 both have worker_type 'prefill'; a "
            "worker type may appear once",
        ),
        (
            [_config()],
            "ais_perf_model: entry 1: worker_type is missing; every config in a "
            "list needs one of: prefill, decode, aggregated",
        ),
        (
            "missing.yaml",
            "cannot read {folder}/missing.yaml: [Errno 2] No such file or "
            "directory: '{folder}/missing.yaml'",
        ),
    ],
)
def test_load_document_errors_are_value_errors(
    value: Any, message: str, tmp_path: Path
) -> None:
    with pytest.raises(ValueError) as error:
        ais_perf_model.load_document(value, base_dir=tmp_path)

    assert str(error.value) == message.format(folder=tmp_path)


@pytest.mark.parametrize(
    ("value", "worker_type", "expected"),
    [
        # One config without worker_type.
        (_config(), "prefill", _config()),
        (_config(), "aggregated", _config()),
        (
            _config(),
            None,
            "ais_perf_model: the config has no worker_type; write worker_type or "
            "set --router-prefill-load-worker-type",
        ),
        # One config written prefill or aggregated: returned as written, also
        # when another type is asked for.
        (_config(worker_type="prefill"), "prefill", _config(worker_type="prefill")),
        (_config(worker_type="prefill"), "aggregated", _config(worker_type="prefill")),
        (_config(worker_type="prefill"), None, _config(worker_type="prefill")),
        (_config(worker_type="aggregated"), None, _config(worker_type="aggregated")),
        # One config written decode.
        (_config(worker_type="decode"), "prefill", _config(worker_type="decode")),
        (
            _config(worker_type="decode"),
            None,
            "ais_perf_model: worker_type is 'decode'; the router needs prefill or "
            "aggregated",
        ),
        # A list, worker type given.
        (
            [_config(worker_type="decode"), _config(worker_type="prefill", tp=2)],
            "prefill",
            _config(worker_type="prefill", tp=2),
        ),
        (
            [_config(worker_type="decode")],
            "prefill",
            "ais_perf_model: no entry for worker type 'prefill'; it has: decode",
        ),
        # A list, worker type not given.
        (
            [_config(worker_type="decode"), _config(worker_type="prefill", tp=2)],
            None,
            _config(worker_type="prefill", tp=2),
        ),
        (
            [_config(worker_type="aggregated", tp=4), _config(worker_type="decode")],
            None,
            _config(worker_type="aggregated", tp=4),
        ),
        (
            [_config(worker_type="aggregated"), _config(worker_type="prefill")],
            None,
            "ais_perf_model: it has entries for prefill and aggregated; set "
            "--router-prefill-load-worker-type to choose one",
        ),
        (
            [_config(worker_type="decode")],
            None,
            "ais_perf_model: no entry for worker type prefill or aggregated; it "
            "has: decode",
        ),
        # A worker type no router runs as.
        (
            _config(),
            "decode",
            "worker_type must be 'prefill' or 'aggregated', got 'decode'",
        ),
        (
            [_config(worker_type="prefill")],
            "router",
            "worker_type must be 'prefill' or 'aggregated', got 'router'",
        ),
    ],
)
def test_select_router_entry_table(
    value: Any, worker_type: str | None, expected: dict[str, Any] | str
) -> None:
    document = ais_perf_model.load_document(value)

    if isinstance(expected, str):
        with pytest.raises(ValueError) as error:
            ais_perf_model.select_router_entry(document, worker_type)
        assert str(error.value) == expected
    else:
        assert ais_perf_model.select_router_entry(document, worker_type) == expected


def test_select_router_entry_names_the_file_in_errors(tmp_path: Path) -> None:
    one = tmp_path / "one.yaml"
    one.write_text(_yaml(_config()))
    several = tmp_path / "several.yaml"
    several.write_text(_yaml([_config(worker_type="decode")]))

    with pytest.raises(ValueError) as no_type:
        ais_perf_model.select_router_entry(ais_perf_model.load_document(str(one)))
    with pytest.raises(ValueError) as no_entry:
        ais_perf_model.select_router_entry(ais_perf_model.load_document(str(several)))

    assert str(no_type.value) == (
        f"{one}: the config has no worker_type; write worker_type or set "
        "--router-prefill-load-worker-type"
    )
    assert str(no_entry.value) == (
        f"{several}: no entry for worker type prefill or aggregated; it has: decode"
    )


def test_select_then_resolve_reports_loader_mismatch() -> None:
    document = ais_perf_model.load_document(_config(worker_type="decode"))
    entry = ais_perf_model.select_router_entry(document, "prefill")

    with pytest.raises(ValueError) as error:
        ais_perf_model.resolve_entry(entry, worker_type="prefill")

    assert str(error.value) == (
        "ais_perf_model entry: worker_type is 'decode' but worker type 'prefill' "
        "was requested"
    )


def test_resolve_entry_fill_adds_and_conflicts() -> None:
    entry = ais_perf_model.load_document(_config()).select("decode")
    written = ais_perf_model.load_document(_config(nextn=2)).select("decode")
    fill = {"nextn": 3}

    assert ais_perf_model.resolve_entry(
        entry, worker_type="decode", fill=fill
    ) == _config(worker_type="decode", nextn=3)
    assert ais_perf_model.resolve_entry(
        written, worker_type="decode", fill={"nextn": 2}
    ) == _config(worker_type="decode", nextn=2)
    with pytest.raises(ValueError) as error:
        ais_perf_model.resolve_entry(written, worker_type="decode", fill=fill)

    assert str(error.value) == (
        "ais_perf_model entry for worker type 'decode': nextn is 2 but the caller "
        "requires 3; write the same value or remove nextn"
    )
    assert entry == _config(worker_type="decode")
    assert fill == {"nextn": 3}


def test_resolve_entry_twice_is_equal(tmp_path: Path) -> None:
    template = tmp_path / "template.yaml"
    template.write_text(_yaml(_config(tp=2, backend_version="1.0")))
    plain = _config(estimator_config={"correction": {"enabled": False}})
    from_template = {"from": str(template), "tp": 4}

    first = ais_perf_model.resolve_entry(plain, worker_type="aggregated")
    assert first == {**plain, "worker_type": "aggregated"}
    assert ais_perf_model.resolve_entry(plain, worker_type="aggregated") == first

    first = ais_perf_model.resolve_entry(from_template, worker_type="prefill")
    assert first == _config(tp=4, backend_version="1.0", worker_type="prefill")
    assert ais_perf_model.resolve_entry(from_template, worker_type="prefill") == first
    assert from_template == {"from": str(template), "tp": 4}


def test_router_config_equals_forward_pass_config_dict(tmp_path: Path) -> None:
    # Imported here: the tests that read no value run without AISimulate installed.
    from aisimulate_core.sdk import ForwardPassPerfModelConfig

    entry = _config(worker_type="prefill", tp=2, backend_version="1.0")
    expected = ForwardPassPerfModelConfig(**entry).to_dict()
    (tmp_path / "cfg.yaml").write_text(_yaml([_config(worker_type="decode"), entry]))
    untyped = {key: value for key, value in entry.items() if key != "worker_type"}

    assert ais_perf_model.router_config(entry) == expected
    assert ais_perf_model.router_config(entry, "prefill") == expected
    assert ais_perf_model.router_config(untyped, "prefill") == expected
    assert ais_perf_model.router_config("cfg.yaml", base_dir=tmp_path) == expected
    assert expected["worker_type"] == "prefill"
    assert expected["tp"] == 2


def test_router_config_reports_unknown_key_as_value_error() -> None:
    with pytest.raises(ValueError) as error:
        ais_perf_model.router_config(_config(tp_size=2), "prefill")

    assert type(error.value) is ValueError
    assert str(error.value).startswith("ais_perf_model: ")
    assert "tp_size" in str(error.value)
    assert type(error.value.__cause__) is TypeError
    assert str(error.value) == f"ais_perf_model: {error.value.__cause__}"


@pytest.mark.parametrize(
    ("change", "missing"),
    [
        pytest.param({"model": ""}, "model", id="empty model"),
        pytest.param({"system": ""}, "system", id="empty system"),
        pytest.param({"backend": ""}, "backend", id="empty backend"),
        pytest.param({"model": False}, "model", id="false model"),
        pytest.param({"model": None}, "model", id="null model"),
        pytest.param(
            {"backend": "", "model": ""}, "backend, model", id="two empty values"
        ),
    ],
)
def test_router_config_requires_identity_values(
    change: dict[str, Any], missing: str
) -> None:
    with pytest.raises(ValueError) as error:
        ais_perf_model.router_config(_config(**change), "prefill")

    assert type(error.value) is ValueError
    assert str(error.value) == f"ais_perf_model requires {missing}"


def test_router_config_requires_a_written_model() -> None:
    entry = {key: value for key, value in _config().items() if key != "model"}

    with pytest.raises(ValueError) as error:
        ais_perf_model.router_config(entry, "aggregated")

    assert str(error.value) == "ais_perf_model requires model"


def test_normalized_config_checks_a_resolved_config() -> None:
    # Imported here: the tests that read no value run without AISimulate installed.
    from aisimulate_core.sdk import ForwardPassPerfModelConfig

    resolved = _config(worker_type="decode", tp=2)

    assert (
        ais_perf_model.normalized_config(resolved)
        == ForwardPassPerfModelConfig(**resolved).to_dict()
    )
    assert resolved == _config(worker_type="decode", tp=2)
    with pytest.raises(ValueError, match="^ais_perf_model requires worker_type$"):
        ais_perf_model.normalized_config(_config())
    with pytest.raises(ValueError, match="^ais_perf_model: .*not_a_field"):
        ais_perf_model.normalized_config(_config(worker_type="decode", not_a_field=1))


def test_uses_relative_paths(tmp_path: Path) -> None:
    dataset = {"root": "dataset", "manifest": "manifest.json"}
    interpolation = {"fpm_interpolation": {"fpm_parquet_path": "fpm.parquet"}}
    absolute = _config(
        systems_paths=[str(tmp_path / "sys")],
        estimator_config={
            "fpm_interpolation": {"fpm_parquet_path": str(tmp_path / "fpm.parquet")}
        },
    )

    assert ais_perf_model.uses_relative_paths(_config(systems_paths=["sys"]))
    assert ais_perf_model.uses_relative_paths({"from": "template.yaml"})
    assert ais_perf_model.uses_relative_paths({"from": {"dataset": dataset}})
    assert ais_perf_model.uses_relative_paths(_config(estimator_config=interpolation))
    assert ais_perf_model.uses_relative_paths(
        [_config(worker_type="decode"), {"from": "x", "worker_type": "prefill"}]
    )

    assert not ais_perf_model.uses_relative_paths(_config())
    assert not ais_perf_model.uses_relative_paths(absolute)
    assert not ais_perf_model.uses_relative_paths(
        {"from": str(tmp_path / "template.yaml"), "worker_type": "prefill"}
    )


def test_relative_file_value_starts_at_base_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "conf"
    other = tmp_path / "other"
    folder.mkdir()
    other.mkdir()
    (folder / "cfg.yaml").write_text(_yaml(_config(worker_type="prefill")))
    (other / "cfg.yaml").write_text(_yaml(_config(worker_type="decode")))
    monkeypatch.chdir(other)

    with_base_dir = ais_perf_model.load_document("cfg.yaml", base_dir=folder)
    without = ais_perf_model.load_document("cfg.yaml")

    assert with_base_dir.source == str(folder / "cfg.yaml")
    assert with_base_dir.worker_types == ("prefill",)
    assert without.source == str(other / "cfg.yaml")
    assert without.worker_types == ("decode",)


def test_relative_paths_inside_file_start_at_file_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conf = tmp_path / "conf"
    other = tmp_path / "other"
    conf.mkdir()
    other.mkdir()
    (conf / "cfg.yaml").write_text(_yaml(_config(systems_paths=["sys"])))
    monkeypatch.chdir(other)

    by_path = ais_perf_model.load_document(str(conf / "cfg.yaml"))
    by_base_dir = ais_perf_model.load_document("cfg.yaml", base_dir=conf)
    inline = ais_perf_model.load_document(_config(systems_paths=["sys"]))

    assert by_path.to_data()["systems_paths"] == [str(conf / "sys")]
    assert by_base_dir.to_data()["systems_paths"] == [str(conf / "sys")]
    # Parsed data has no file: its relative paths start at the working directory.
    assert inline.to_data()["systems_paths"] == [str(other / "sys")]
