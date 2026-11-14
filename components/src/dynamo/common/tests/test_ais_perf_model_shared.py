# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from dynamo.common import ais_perf_model
from tests.utils.ais_perf_model import IDENTITY

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]

NEW_FLAG = "--ais-perf-model"
ENGINE_ARGS_FLAG = "--extra-engine-args"


def _config(**keys: Any) -> dict[str, Any]:
    """One perf model config; ``keys`` add to it or replace its values."""
    return {**IDENTITY, **keys}


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
    workspaces: dict[int, str] = {}
    monkeypatch.setattr(ais_perf_model, "_dataset_workspaces", workspaces)
    yield workspaces
    for folder in workspaces.values():
        shutil.rmtree(folder, ignore_errors=True)


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
        import importlib.abc
        import json
        import sys

        class WithoutAISimulate(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in {"aisimulate", "aisimulate_core"}:
                    raise ModuleNotFoundError(
                        f"No module named {fullname!r}", name=fullname
                    )

        sys.meta_path.insert(0, WithoutAISimulate())

        from dynamo.common import ais_perf_model

        names = {"engine_args_name": "--extra-engine-args", "perf_model_name": "x"}
        config = {"model": "m", "system": "s", "backend": "b", "worker_type": "decode"}
        import_errors = {}
        for name, call in {
            "load_document": lambda: ais_perf_model.load_document(config),
            "resolve_entry": lambda: ais_perf_model.resolve_entry(config),
            "router_perf_config": lambda: ais_perf_model.router_perf_config(config),
            "engine_perf_config": lambda: ais_perf_model.engine_perf_config(
                config, "decode"
            ),
            "normalized_config": lambda: ais_perf_model.normalized_config(config),
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
        # The functions that read a value fail with the import error itself.
        "import_errors": {
            "load_document": "aisimulate_core",
            "resolve_entry": "aisimulate_core",
            "router_perf_config": "aisimulate_core",
            "engine_perf_config": "aisimulate_core",
            "normalized_config": "aisimulate_core",
            "uses_relative_paths": "aisimulate_core",
        },
    }


@pytest.mark.timeout(60)  # one interpreter start, about 1s
def test_aisimulate_build_without_the_loader_is_an_import_error(
    tmp_path: Path,
) -> None:
    # An aisimulate_core package that has no ais_perf_model module.
    (tmp_path / "aisimulate_core").mkdir()
    (tmp_path / "aisimulate_core" / "__init__.py").write_text("")

    output = _run_child(
        """
        import os
        import sys

        sys.path.insert(0, os.environ["PACKAGES_WITHOUT_LOADER"])

        from dynamo.common import ais_perf_model

        try:
            ais_perf_model.load_document({"model": "m"})
        except ImportError as error:
            print(type(error).__name__, error.name)
        """,
        PACKAGES_WITHOUT_LOADER=str(tmp_path),
    )

    assert output.split() == ["ModuleNotFoundError", "aisimulate_core.ais_perf_model"]


def test_dataset_workspace_only_for_dataset_entries(
    monkeypatch: pytest.MonkeyPatch, dataset_workspaces: dict[int, str]
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
def test_each_process_removes_only_its_own_dataset_workspace(tmp_path: Path) -> None:
    output = _run_child(
        """
        import os
        import sys

        from dynamo.common import ais_perf_model

        parent_folder = ais_perf_model._dataset_workspace()
        read_end, write_end = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.write(write_end, ais_perf_model._dataset_workspace().encode())
            # A normal exit: the exit hooks inherited from the parent run here.
            sys.exit(0)
        os.close(write_end)
        _, status = os.waitpid(pid, 0)
        child_folder = os.read(read_end, 4096).decode()
        print(parent_folder)
        print(os.waitstatus_to_exitcode(status))
        print(child_folder != parent_folder)
        print(os.path.isdir(parent_folder))
        print(os.path.exists(child_folder))
        print(ais_perf_model._dataset_workspace() == parent_folder)
        """,
        # The folders are made under tmp_path, so the checks below see them.
        TMPDIR=str(tmp_path),
    )

    parent_folder, *answers = output.split()
    assert answers == ["0", "True", "True", "False", "True"]
    assert Path(parent_folder).parent == tmp_path
    # The interpreter that made the folder has exited.
    assert not os.path.exists(parent_folder)


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


# The loader raises these errors and their texts are AISimulate's. Dynamo
# relies on the type: a wrong value is a ValueError.
@pytest.mark.parametrize(
    "value",
    [
        pytest.param("", id="empty text"),
        pytest.param({}, id="empty config"),
        pytest.param([], id="empty list"),
        pytest.param(3, id="number"),
        pytest.param(_config(worker_type="router"), id="unknown worker type"),
        pytest.param(
            [_config(worker_type="prefill"), _config(worker_type="prefill")],
            id="worker type twice",
        ),
        pytest.param([_config()], id="list entry without worker type"),
        pytest.param("missing.yaml", id="missing file"),
    ],
)
def test_load_document_errors_are_value_errors(value: Any, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ais_perf_model.load_document(value, base_dir=tmp_path)


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


def test_select_router_entry_needs_an_entry_for_the_given_worker_type() -> None:
    document = ais_perf_model.load_document([_config(worker_type="decode")])

    # The loader reports the missing entry, in its own words.
    with pytest.raises(ValueError):
        ais_perf_model.select_router_entry(document, "prefill")


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

    with pytest.raises(ValueError):
        ais_perf_model.resolve_entry(entry, worker_type="prefill")


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
    with pytest.raises(ValueError):
        ais_perf_model.resolve_entry(written, worker_type="decode", fill=fill)

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


def test_engine_perf_config_is_the_entry_of_the_worker_type(tmp_path: Path) -> None:
    prefill = _config(worker_type="prefill", tp=2)
    decode = _config(worker_type="decode")
    path = tmp_path / "cfg.yaml"
    path.write_text(_yaml([decode, prefill]))

    # The config as resolved: no key is added but a missing worker_type.
    assert ais_perf_model.engine_perf_config([decode, prefill], "prefill") == prefill
    assert ais_perf_model.engine_perf_config(str(path), "decode") == decode
    assert ais_perf_model.engine_perf_config(_config(), "decode") == decode


def test_engine_perf_config_fill_adds_and_conflicts() -> None:
    fill = {"nextn": 3}

    filled = ais_perf_model.engine_perf_config(_config(), "decode", fill=fill)

    assert filled == _config(worker_type="decode", nextn=3)
    # A written value that differs from the required one.
    with pytest.raises(ValueError):
        ais_perf_model.engine_perf_config(_config(nextn=2), "decode", fill=fill)
    assert fill == {"nextn": 3}


@pytest.mark.parametrize(
    "value",
    [
        pytest.param([_config(worker_type="prefill")], id="list without the type"),
        pytest.param(_config(worker_type="prefill"), id="config of another type"),
    ],
)
def test_engine_perf_config_needs_an_entry_for_the_worker_type(value: Any) -> None:
    with pytest.raises(ValueError):
        ais_perf_model.engine_perf_config(value, "decode")


def test_router_perf_config_reports_unknown_key_as_value_error() -> None:
    with pytest.raises(ValueError) as error:
        ais_perf_model.router_perf_config(_config(tp_size=2), "prefill")

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
def test_router_perf_config_requires_identity_values(
    change: dict[str, Any], missing: str
) -> None:
    with pytest.raises(ValueError) as error:
        ais_perf_model.router_perf_config(_config(**change), "prefill")

    assert type(error.value) is ValueError
    assert str(error.value) == f"ais_perf_model requires {missing}"


def test_router_perf_config_requires_a_written_model() -> None:
    entry = {key: value for key, value in _config().items() if key != "model"}

    with pytest.raises(ValueError) as error:
        ais_perf_model.router_perf_config(entry, "aggregated")

    assert str(error.value) == "ais_perf_model requires model"


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
    (conf / "sys").mkdir(parents=True)
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
    # The two calls the components make hand a file path on as it is given,
    # so the same holds for the config an engine or a router gets.
    engine = ais_perf_model.engine_perf_config(str(conf / "cfg.yaml"), "decode")
    router = ais_perf_model.router_perf_config(str(conf / "cfg.yaml"), "prefill")
    assert engine["systems_paths"] == [str(conf / "sys")]
    assert router["systems_paths"] == [str(conf / "sys")]
