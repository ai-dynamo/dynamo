# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for applying the GMS graph-archive policy to a dynamo.vllm engine."""

import importlib.metadata
import importlib.util
import os
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

# GMS is an optional dependency of dynamo.vllm, and a test image can have GMS
# without vLLM.
pytest.importorskip("gpu_memory_service.integrations.vllm.graph_archive")
pytest.importorskip("vllm.engine.arg_utils")

from gpu_memory_service.integrations.vllm.graph_archive import (  # noqa: E402
    FOUNDRY_FORMAT,
    ArchiveMode,
    GraphArchive,
)
from vllm.engine.arg_utils import AsyncEngineArgs  # noqa: E402

from dynamo.vllm import gms_graph_archive  # noqa: E402

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]

FOUNDRY_ENV = ("FOUNDRY_VLLM_CONFIG", "FOUNDRY_MODE", "LD_PRELOAD", "VLLM_PLUGINS")


@pytest.fixture
def hook(monkeypatch, tmp_path):
    """A stand-in for Foundry's hook library and a clean Foundry environment."""
    library = tmp_path / "libcuda_hook.so"
    library.write_bytes(b"")
    monkeypatch.setattr(
        gms_graph_archive, "_foundry_hook_library", lambda: str(library)
    )
    for name in FOUNDRY_ENV:
        # Set first, so the variable is unset again after the test even if the
        # code under test set it.
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    return str(library)


def _engine_args(tmp_path, read_only, **loader_options) -> AsyncEngineArgs:
    if read_only:
        loader_options["gms_read_only"] = True
    return AsyncEngineArgs(
        model=str(tmp_path),  # local: vLLM resolves a model name on the hub
        load_format="gms",
        num_gpu_blocks_override=64,
        tensor_parallel_size=2,
        disable_custom_all_reduce=True,
        model_loader_extra_config=loader_options,
    )


def _start(tmp_path, read_only, settings=None, **loader_options):
    """Plan one engine; returns the start context and the engine arguments."""
    settings_file = None
    if settings is not None:
        settings_file = tmp_path / "foundry.toml"
        settings_file.write_text(settings)
    engine_args = _engine_args(tmp_path, read_only, **loader_options)
    start = gms_graph_archive.start_graph_archive(
        str(tmp_path / "archive"), settings_file, engine_args
    )
    return start, engine_args


def _save_all_ranks(workspace: Path) -> None:
    """What Foundry's SAVE leaves in the workspace of a TP 2 engine."""
    for rank in range(2):
        rank_dir = workspace / FOUNDRY_FORMAT.rank_dir.format(rank=rank)
        rank_dir.mkdir()
        (rank_dir / FOUNDRY_FORMAT.complete_marker).write_text("{}")


def test_first_read_only_engine_saves_with_foundry(hook, tmp_path, monkeypatch):
    monkeypatch.setenv("LD_PRELOAD", "/lib/other.so")
    start, engine_args = _start(tmp_path, True, 'region_size = "8GB"\n')

    assert start.archive.mode == ArchiveMode.SAVE
    options = engine_args.model_loader_extra_config
    assert options["gms_graph_archive"] == "save"
    assert options["gms_reconstruction_mode"] == "load_diff"
    assert options["gms_fixed_va_windows"] == {
        tag: list(window)
        for tag, window in gms_graph_archive.DEFAULT_FIXED_VA_WINDOWS.items()
    }
    # Nothing in this process's environment until the engine starts.
    assert os.environ.get("FOUNDRY_VLLM_CONFIG") is None
    assert os.environ.get("LD_PRELOAD") == "/lib/other.so"

    with start:
        settings_file = Path(os.environ.get("FOUNDRY_VLLM_CONFIG"))
        assert tomllib.loads(settings_file.read_text()) == {
            "mode": "save",
            "workspace_root": str(start.archive.workspace),
            "region_size": "8GB",
        }
        assert os.environ.get("FOUNDRY_MODE") == "save"
        assert os.environ.get("LD_PRELOAD") == f"{hook}:/lib/other.so"
        _save_all_ranks(start.archive.workspace)

    # The engine started: the save is committed, and nothing this process
    # starts later runs with Foundry's hook.
    assert (start.archive.path / "manifest.json").is_file()
    assert os.environ.get("FOUNDRY_VLLM_CONFIG") is None
    assert os.environ.get("FOUNDRY_MODE") is None
    assert os.environ.get("LD_PRELOAD") == "/lib/other.so"
    assert not settings_file.exists()


def test_failed_engine_start_is_reported_and_reraised(hook, tmp_path):
    start, _ = _start(tmp_path, read_only=True)

    with pytest.raises(RuntimeError, match="Engine core initialization failed"):
        with start:
            settings_file = Path(os.environ.get("FOUNDRY_VLLM_CONFIG"))
            raise RuntimeError("Engine core initialization failed")

    assert os.environ.get("FOUNDRY_VLLM_CONFIG") is None
    assert os.environ.get("LD_PRELOAD") is None
    assert not settings_file.exists()
    # The failed save was dropped, so the next read-only engine saves.
    retry, _ = _start(tmp_path, read_only=True)
    assert retry.archive.mode == ArchiveMode.SAVE


def test_writer_captures_natively_without_foundry(hook, tmp_path):
    start, engine_args = _start(tmp_path, read_only=False)

    assert start.archive.mode == ArchiveMode.NATIVE
    # The writer still publishes the load diff that read-only engines attach with.
    assert engine_args.model_loader_extra_config == {
        "gms_reconstruction_mode": "load_diff"
    }
    with start:
        assert os.environ.get("FOUNDRY_VLLM_CONFIG") is None
        assert os.environ.get("LD_PRELOAD") is None


def test_start_without_the_hook_library_is_refused(tmp_path):
    archive = GraphArchive(tmp_path, {"engine.model": "m"}, [0])
    assert archive.plan(read_only=True) == ArchiveMode.SAVE
    try:
        with pytest.raises(RuntimeError, match="Foundry's hook library"):
            with gms_graph_archive.GraphArchiveStart(archive):
                pass
        assert os.environ.get("FOUNDRY_VLLM_CONFIG") is None
    finally:
        archive.abort("test finished")


def test_replacement_loads_what_the_first_shadow_saved(hook, tmp_path):
    saver, _ = _start(tmp_path, read_only=True)
    with saver:
        _save_all_ranks(saver.archive.workspace)

    # Engine 0 restarted after a failover: it may write, but an archive exists.
    loader, engine_args = _start(tmp_path, read_only=False)

    assert loader.archive.mode == ArchiveMode.LOAD
    assert engine_args.model_loader_extra_config["gms_graph_archive"] == "load"
    with loader:
        foundry = tomllib.loads(Path(os.environ.get("FOUNDRY_VLLM_CONFIG")).read_text())
        assert foundry["workspace_root"] == str(saver.archive.path)
        assert os.environ.get("FOUNDRY_MODE") == "load"


def test_without_the_flag_nothing_changes(tmp_path):
    engine_args = _engine_args(tmp_path, read_only=True)

    with gms_graph_archive.start_graph_archive(None, None, engine_args):
        assert engine_args.model_loader_extra_config == {"gms_read_only": True}


@pytest.mark.parametrize(
    "settings, env, message",
    [
        ('mode = "load"\n', {}, "must not set \\['mode'\\]"),
        ("region_size = \n", {}, "--gms-graph-archive-config .*foundry.toml"),
        ("", {"FOUNDRY_VLLM_CONFIG": "/x.toml"}, "FOUNDRY_VLLM_CONFIG is set"),
        ("", {"VLLM_PLUGINS": "other"}, "needs the foundry plugin"),
    ],
)
def test_refuses_foundry_settings_it_cannot_honour(
    hook, tmp_path, monkeypatch, settings, env, message
):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=message):
        _start(tmp_path, True, settings)


def test_refuses_a_settings_file_that_does_not_exist(hook, tmp_path):
    engine_args = _engine_args(tmp_path, read_only=True)
    with pytest.raises(ValueError, match="--gms-graph-archive-config .*missing.toml"):
        gms_graph_archive.start_graph_archive(
            str(tmp_path / "archive"), tmp_path / "missing.toml", engine_args
        )


@pytest.mark.parametrize(
    "installed, message",
    [
        ({}, "requires the foundry package"),
        ({"package": True}, "requires Foundry's vLLM plugin"),
        ({"package": True, "plugin": True}, "hook library .* is missing"),
    ],
)
def test_refuses_an_image_without_foundrys_vllm_plugin(
    tmp_path, monkeypatch, installed, message
):
    spec = SimpleNamespace(origin=str(tmp_path / "foundry" / "__init__.py"))
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: spec if installed.get("package") else None,
    )
    plugins = ["foundry"] if installed.get("plugin") else ["lora_filesystem_resolver"]
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda group: SimpleNamespace(names=plugins),
    )

    with pytest.raises(ValueError, match=message):
        gms_graph_archive._foundry_hook_library()


@pytest.mark.parametrize(
    "settings, windows, message",
    [
        # Foundry's default region (0x600000000000 + 64 GB) and a window inside it.
        ("", {"weights": ["0x600000000000", 1 << 30]}, "overlaps"),
        ("base_addr = 0x420000000000\n", None, "Foundry region .* overlaps"),
        ('base_addr = "0x420000000000"\n', None, "Foundry region .* overlaps"),
        ('region_size = "48TB"\nbase_addr = 0x480000000000\n', None, "overlaps"),
        (
            "",
            {"weights": [0x10000000000, 2 << 40], "kv_cache": [0x11000000000, 1 << 30]},
            "weights.* overlaps .*kv_cache",
        ),
        # A window for one tag keeps the other tag's default.
        ("", {"weights": ["0x4a0000000000", 1 << 30]}, "weights.* overlaps .*kv_cache"),
        ("", {"kvcache": [1 << 46, 1 << 30]}, "has no tag \\['kvcache'\\]"),
        ("", {"kv_cache": [1 << 46, 0]}, "positive size"),
        ("", {"kv_cache": None}, "must be \\[base, size\\]"),
        ("", {"kv_cache": [1 << 46]}, "must be \\[base, size\\]"),
        ("base_addr = true\n", None, "cannot read Foundry address"),
    ],
)
def test_refuses_windows_it_cannot_use(hook, tmp_path, settings, windows, message):
    loader_options = {}
    if windows is not None:
        loader_options["gms_fixed_va_windows"] = windows
    with pytest.raises(ValueError, match=message):
        _start(tmp_path, True, settings, **loader_options)
    assert not (tmp_path / "archive").exists()


def test_default_windows_fit_below_foundrys_region():
    windows = gms_graph_archive.DEFAULT_FIXED_VA_WINDOWS
    gms_graph_archive.check_fixed_va_windows(windows, {})
    gms_graph_archive.check_fixed_va_windows(
        windows, {"base_addr": 0x600000000000, "region_size": "512GB"}
    )
    for base, size in windows.values():
        assert base + size <= 0x600000000000


def test_changing_a_window_changes_the_archive(hook, tmp_path):
    default, _ = _start(tmp_path, read_only=True)
    moved, engine_args = _start(
        tmp_path,
        read_only=True,
        gms_fixed_va_windows={"kv_cache": ["0x4c0000000000", 256 << 30]},
    )

    assert moved.archive.key != default.archive.key
    assert moved.archive.mode == ArchiveMode.SAVE  # not "another engine is saving"
    assert engine_args.model_loader_extra_config["gms_fixed_va_windows"] == {
        "weights": list(gms_graph_archive.DEFAULT_FIXED_VA_WINDOWS["weights"]),
        "kv_cache": [0x4C0000000000, 256 << 30],
    }
