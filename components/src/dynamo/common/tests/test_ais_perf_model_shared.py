# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import types
from pathlib import Path
from typing import Any

import pytest

from dynamo.common import ais_perf_model
from tests.utils.ais_perf_model import IDENTITY

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]

NAMES = {
    "engine_args_name": "--extra-engine-args",
    "perf_model_name": "--ais-perf-model",
}


def _config(**keys: Any) -> dict[str, Any]:
    """One perf model config; ``keys`` add to it or replace its values."""
    return {**IDENTITY, **keys}


PREFILL = _config(worker_type="prefill")
DECODE = _config(worker_type="decode")
AGGREGATED = _config(worker_type="aggregated")


def _write(path: Path, value: Any) -> str:
    """Write ``value`` as the config file ``path`` and return the path as text."""
    # JSON is YAML, so a config file needs no YAML writer here.
    path.write_text(json.dumps(value))
    return str(path)


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


def test_set_timing_model_none_text_and_mapping() -> None:
    config = _config(worker_type="decode", estimator_config={"correction": {}})
    timing = {"type": "external", "provider": "ais", "config": copy.deepcopy(config)}
    written = {"engine": {"block_size": 32}, "num_workers": 2}

    from_none = ais_perf_model.set_timing_model(None, config, **NAMES)
    from_text = ais_perf_model.set_timing_model('{"num_workers": 2}', config, **NAMES)
    result = ais_perf_model.set_timing_model(written, config, **NAMES)

    assert from_none == {"engine": {"timing_model": timing}}
    assert from_text == {"engine": {"timing_model": timing}, "num_workers": 2}
    assert result == {**written, "engine": {"block_size": 32, "timing_model": timing}}
    # The result is a copy: a change to it reaches neither input.
    result["engine"]["block_size"] = 64
    result["engine"]["timing_model"]["config"]["estimator_config"].clear()
    assert written == {"engine": {"block_size": 32}, "num_workers": 2}
    assert config == timing["config"]


@pytest.mark.parametrize(
    "engine_args",
    [
        {"engine": {"timing_model": None}},
        '{"engine": {"timing_model": {"type": "polynomial"}}}',
    ],
)
def test_set_timing_model_refuses_written_timing_model(engine_args: Any) -> None:
    names = {"engine_args_name": "decode_args", "perf_model_name": "ais_perf_model"}

    with pytest.raises(ValueError) as from_set:
        ais_perf_model.set_timing_model(engine_args, _config(), **names)
    with pytest.raises(ValueError) as from_read:
        ais_perf_model.engine_args_as_written(engine_args, **names)

    assert str(from_set.value) == str(from_read.value)
    # The documentation quotes this sentence with the names of each component.
    assert str(from_read.value) == (
        "decode_args already sets engine.timing_model; it cannot be combined with "
        "ais_perf_model"
    )


@pytest.mark.parametrize(
    "engine_args", ["[1, 2]", "null", "not json", {"engine": ["a"]}, 3]
)
def test_engine_args_as_written_rejects_non_object(engine_args: Any) -> None:
    with pytest.raises(ValueError, match="^--extra-engine-args must be a JSON object"):
        ais_perf_model.engine_args_as_written(engine_args, **NAMES)


def test_env_value_empty_is_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DYN_AIS_PERF_MODEL", raising=False)
    assert ais_perf_model.env_value() is None
    monkeypatch.setenv("DYN_AIS_PERF_MODEL", "")
    assert ais_perf_model.env_value() is None
    monkeypatch.setenv("DYN_AIS_PERF_MODEL", "cfg.yaml")
    assert ais_perf_model.env_value() == "cfg.yaml"

    monkeypatch.setenv("DYN_ROUTER_PREFILL_LOAD_WORKER_TYPE", "prefill")
    assert ais_perf_model.env_value(ais_perf_model.ROUTER_WORKER_TYPE_ENV) == "prefill"


@pytest.mark.timeout(60)  # one interpreter start, about 1s
def test_functions_without_a_value_work_when_aisimulate_is_missing() -> None:
    output = _run_child(
        """
        import importlib.abc
        import json
        import sys

        tried = []

        class WithoutAISimulate(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in {"aisim", "aisim_core"}:
                    tried.append(fullname)
                    raise ModuleNotFoundError(fullname, name=fullname)

        sys.meta_path.insert(0, WithoutAISimulate())

        from dynamo.common import ais_perf_model

        config = {"model": "m", "system": "s", "backend": "b", "worker_type": "decode"}
        names = {"engine_args_name": "a", "perf_model_name": "b"}
        result = {
            "tried_by_import": list(tried),
            "env_value": ais_perf_model.env_value("DYN_TEST_UNSET_NAME"),
            "set_timing_model": ais_perf_model.set_timing_model(None, {}, **names),
        }
        for call in (ais_perf_model.load_document, ais_perf_model.normalized_config):
            try:
                call(config)
            except ImportError as error:
                result[call.__name__] = error.name
        print(json.dumps(result))
        """
    )

    timing = {"type": "external", "provider": "ais", "config": {}}
    assert json.loads(output) == {
        # Importing the module does not ask for AISimulate.
        "tried_by_import": [],
        "env_value": None,
        "set_timing_model": {"engine": {"timing_model": timing}},
        # The functions that read a value fail with the import error itself.
        "load_document": "aisim_core",
        "normalized_config": "aisim_core",
    }


def test_aisimulate_build_without_the_loader_is_an_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An aisim_core that has no ais_perf_model module.
    monkeypatch.setitem(sys.modules, "aisim_core", types.ModuleType("aisim_core"))
    monkeypatch.delitem(sys.modules, "aisim_core.ais_perf_model", raising=False)

    with pytest.raises(ImportError) as error:
        ais_perf_model.load_document(_config())

    assert error.value.name == "aisim_core.ais_perf_model"


def test_dataset_workspace_only_for_dataset_entries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict[str, Any]] = []
    loader = types.SimpleNamespace(resolve=lambda entry, **keys: calls.append(keys))
    monkeypatch.setattr(ais_perf_model, "_loader", lambda: loader)
    monkeypatch.setattr(ais_perf_model, "_dataset_workspaces", {})
    # The folder is made under tmp_path, so the checks below see it.
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    dataset = {"from": {"dataset": {"root": "/data", "manifest": "manifest.json"}}}
    handed_on = {"worker_type": "decode", "fill": {"nextn": 2}, "defaults": {"tp": 1}}

    ais_perf_model.resolve_entry(_config())
    ais_perf_model.resolve_entry({"from": "template.yaml"})
    ais_perf_model.resolve_entry({"from": {"state_snapshot": {"path": "/snapshot"}}})
    assert [call["workspace"] for call in calls] == [None, None, None]
    assert list(tmp_path.iterdir()) == []

    ais_perf_model.resolve_entry(dataset, **handed_on)
    ais_perf_model.resolve_entry(dataset)
    (folder,) = tmp_path.iterdir()
    assert folder.name.startswith("dynamo-ais-perf-model-")
    assert calls[3] == {**handed_on, "workspace": str(folder)}
    # One folder for the process: the next dataset entry gets the same one.
    assert calls[4]["workspace"] == str(folder)


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
    path = _write(tmp_path / "cfg.yaml", PREFILL)
    both = [PREFILL, DECODE]
    # A template is not opened when the value is read.
    unread = {"from": str(tmp_path / "missing.yaml")}

    for value, expected in [
        (path, (True, path, ("prefill",), PREFILL)),
        (json.dumps(both), (False, None, ("prefill", "decode"), both)),
        (_config(), (True, None, (), _config())),
        ([DECODE, PREFILL], (False, None, ("decode", "prefill"), [DECODE, PREFILL])),
        (unread, (True, None, (), unread)),
    ]:
        doc = ais_perf_model.load_document(value)
        assert (doc.bare, doc.source, doc.worker_types, doc.to_data()) == expected


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
        pytest.param([PREFILL, PREFILL], id="worker type twice"),
        pytest.param([_config()], id="list entry without worker type"),
        pytest.param("missing.yaml", id="missing file"),
    ],
)
def test_load_document_errors_are_value_errors(value: Any, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ais_perf_model.load_document(value, base_dir=tmp_path)


# An error is given as a pattern. The first one is the complete sentence, as
# the documentation quotes it.
@pytest.mark.parametrize(
    ("value", "worker_type", "expected"),
    [
        # One config without worker_type.
        (_config(), "prefill", _config()),
        (
            _config(),
            None,
            "^ais_perf_model: the config has no worker_type; write worker_type or "
            "set --router-prefill-load-worker-type$",
        ),
        # One config with a worker_type: returned as written, also when
        # another type is asked for.
        (PREFILL, None, PREFILL),
        (AGGREGATED, None, AGGREGATED),
        (DECODE, "prefill", DECODE),
        (DECODE, None, "^ais_perf_model: worker_type is 'decode'; the router needs"),
        # A list, worker type given.
        ([DECODE, PREFILL], "prefill", PREFILL),
        # A list, worker type not given.
        ([DECODE, PREFILL], None, PREFILL),
        ([AGGREGATED, DECODE], None, AGGREGATED),
        ([AGGREGATED, PREFILL], None, "^ais_perf_model: it has entries for prefill"),
        ([DECODE], None, "^ais_perf_model: no entry .* aggregated; it has: decode$"),
        # A worker type no router runs as.
        (_config(), "decode", "^worker_type must be 'prefill' or 'aggregated', got"),
    ],
)
def test_select_router_entry_table(
    value: Any, worker_type: str | None, expected: dict[str, Any] | str
) -> None:
    document = ais_perf_model.load_document(value)

    if isinstance(expected, str):
        with pytest.raises(ValueError, match=expected):
            ais_perf_model.select_router_entry(document, worker_type)
    else:
        assert ais_perf_model.select_router_entry(document, worker_type) == expected


def test_select_router_entry_needs_an_entry_for_the_given_worker_type() -> None:
    document = ais_perf_model.load_document([DECODE])

    # The loader reports the missing entry, in its own words.
    with pytest.raises(ValueError):
        ais_perf_model.select_router_entry(document, "prefill")


def test_select_router_entry_names_the_file_in_errors(tmp_path: Path) -> None:
    path = _write(tmp_path / "one.yaml", _config())

    with pytest.raises(ValueError, match=f"^{re.escape(path)}: the config has no"):
        ais_perf_model.select_router_entry(ais_perf_model.load_document(path))
    several = _write(tmp_path / "several.yaml", [DECODE])
    with pytest.raises(ValueError, match=f"^{re.escape(several)}: no entry for"):
        ais_perf_model.select_router_entry(ais_perf_model.load_document(several))


def test_select_then_resolve_reports_loader_mismatch() -> None:
    document = ais_perf_model.load_document(DECODE)
    entry = ais_perf_model.select_router_entry(document, "prefill")

    with pytest.raises(ValueError):
        ais_perf_model.resolve_entry(entry, worker_type="prefill")


def test_resolve_entry_twice_is_equal(tmp_path: Path) -> None:
    template = _write(tmp_path / "template.yaml", _config(tp=2, backend_version="1.0"))
    entry = {"from": template, "tp": 4}

    first = ais_perf_model.resolve_entry(entry, worker_type="prefill")

    # The keys of the entry replace those of the template.
    assert first == _config(tp=4, backend_version="1.0", worker_type="prefill")
    assert ais_perf_model.resolve_entry(entry, worker_type="prefill") == first
    assert entry == {"from": template, "tp": 4}


def test_engine_perf_config_is_the_entry_of_the_worker_type(tmp_path: Path) -> None:
    path = _write(tmp_path / "cfg.yaml", [DECODE, PREFILL])

    # The config as resolved: no key is added but a missing worker_type.
    assert ais_perf_model.engine_perf_config([DECODE, PREFILL], "prefill") == PREFILL
    assert ais_perf_model.engine_perf_config(path, "decode") == DECODE
    assert ais_perf_model.engine_perf_config(_config(), "decode") == DECODE


def test_engine_perf_config_fill_adds_and_conflicts() -> None:
    fill = {"nextn": 3}
    filled = _config(worker_type="decode", nextn=3)

    assert ais_perf_model.engine_perf_config(_config(), "decode", fill=fill) == filled
    # A written value must equal the required one.
    written = ais_perf_model.engine_perf_config(_config(nextn=3), "decode", fill=fill)
    assert written == filled
    with pytest.raises(ValueError):
        ais_perf_model.engine_perf_config(_config(nextn=2), "decode", fill=fill)
    assert fill == {"nextn": 3}


@pytest.mark.parametrize("value", [[PREFILL], PREFILL])
def test_engine_perf_config_needs_an_entry_for_the_worker_type(value: Any) -> None:
    with pytest.raises(ValueError):
        ais_perf_model.engine_perf_config(value, "decode")


def test_router_perf_config_reports_unknown_key_as_value_error() -> None:
    with pytest.raises(ValueError, match="^ais_perf_model: .*tp_size") as error:
        ais_perf_model.router_perf_config(_config(tp_size=2), "prefill")

    assert type(error.value.__cause__) is TypeError


# The documentation quotes "ais_perf_model requires model" and says that a
# null value is the same as a key left out.
@pytest.mark.parametrize(
    ("change", "missing"),
    [
        ({"system": ""}, "system"),
        ({"model": None}, "model"),
        ({"backend": "", "model": ""}, "backend, model"),
    ],
)
def test_router_perf_config_requires_identity_values(
    change: dict[str, Any], missing: str
) -> None:
    with pytest.raises(ValueError, match=f"^ais_perf_model requires {missing}$"):
        ais_perf_model.router_perf_config(_config(**change), "prefill")


def test_router_perf_config_requires_a_written_model() -> None:
    entry = {key: value for key, value in _config().items() if key != "model"}

    with pytest.raises(ValueError, match="^ais_perf_model requires model$"):
        ais_perf_model.router_perf_config(entry, "aggregated")


@pytest.mark.parametrize(
    ("value", "relative"),
    [
        (_config(systems_paths=["sys"]), True),
        ({"from": "template.yaml"}, True),
        ({"from": {"dataset": {"root": "data", "manifest": "manifest.json"}}}, True),
        ([AGGREGATED, {"from": "x", "worker_type": "prefill"}], True),
        (
            _config(estimator_config={"fpm_interpolation": {"fpm_parquet_path": "f"}}),
            True,
        ),
        (_config(), False),
        (_config(systems_paths=["/abs/sys"]), False),
        ({"from": "/abs/template.yaml", "worker_type": "prefill"}, False),
    ],
)
def test_uses_relative_paths(value: Any, relative: bool) -> None:
    assert ais_perf_model.has_relative_paths(value) is relative


def test_relative_file_value_starts_at_base_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conf, other = tmp_path / "conf", tmp_path / "other"
    for folder, config in ((conf, PREFILL), (other, DECODE)):
        folder.mkdir()
        _write(folder / "cfg.yaml", config)
    monkeypatch.chdir(other)

    with_base_dir = ais_perf_model.load_document("cfg.yaml", base_dir=conf)
    without = ais_perf_model.load_document("cfg.yaml")
    router = ais_perf_model.router_perf_config("cfg.yaml", base_dir=conf)

    assert with_base_dir.source == str(conf / "cfg.yaml")
    assert without.source == str(other / "cfg.yaml")
    assert without.worker_types == ("decode",)
    # router_perf_config hands base_dir on: the file it read is the prefill one.
    assert router["worker_type"] == "prefill"


def test_relative_paths_inside_file_start_at_file_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conf, other = tmp_path / "conf", tmp_path / "other"
    (conf / "sys").mkdir(parents=True)
    other.mkdir()
    path = _write(conf / "cfg.yaml", _config(systems_paths=["sys"]))
    monkeypatch.chdir(other)

    # The two calls the components make hand a file path on as it is given.
    engine = ais_perf_model.engine_perf_config(path, "decode")
    router = ais_perf_model.router_perf_config(path, "prefill")
    inline = ais_perf_model.load_document(_config(systems_paths=["sys"]))

    assert engine["systems_paths"] == [str(conf / "sys")]
    assert router["systems_paths"] == [str(conf / "sys")]
    # Parsed data has no file: its relative paths start at the working directory.
    assert inline.to_data()["systems_paths"] == [str(other / "sys")]
