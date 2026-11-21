# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the graph-archive policy: who saves, who loads, and what is set aside."""

import dataclasses
import enum
import fcntl
import json
import logging
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
from _deps import HAS_GMS

if not HAS_GMS:
    pytest.skip(
        "gpu_memory_service package is not available in this test image",
        allow_module_level=True,
    )

from gpu_memory_service.common.locks import GrantedLockType
from gpu_memory_service.integrations.vllm import graph_archive
from gpu_memory_service.integrations.vllm.graph_archive import (
    FOUNDRY_FORMAT,
    LOADER_OPTION,
    MAX_FAILED_STARTS,
    ArchiveMode,
    GraphArchive,
    GraphArchiveError,
    archive_key,
    check_read_only_attach,
    engine_identity,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
]

IDENTITY = MappingProxyType({"engine.model": "m", "engine.tensor_parallel_size": 2})
RANKS = (0, 1)
DIED_LOADING = "an engine died while loading this archive"
DIED_SAVING = "an engine died while saving this archive"

_PLAN_AND_DIE = """
import json, os, sys
from gpu_memory_service.integrations.vllm.graph_archive import GraphArchive
root, identity, ranks, read_only = json.loads(sys.argv[1])
GraphArchive(root, identity, ranks).plan(read_only=read_only)
os._exit(0)  # no commit() or abort(): the kernel drops the engine's flocks
"""


@pytest.fixture
def engine(tmp_path):
    """Make one engine's GraphArchive on tmp_path. What a test leaves planned
    is released after it, so no flock outlives the test."""
    engines = []

    def make(identity=IDENTITY, ranks=RANKS, root=tmp_path):
        archive = GraphArchive(root, identity, ranks)
        engines.append(archive)
        return archive

    yield make
    for archive in engines:
        archive.abort("test finished")


@pytest.fixture
def warnings():
    """Warnings the policy logs, whatever the session did to log propagation."""
    records = []
    handler = logging.Handler(level=logging.WARNING)
    handler.emit = records.append
    graph_archive.logger.addHandler(handler)
    yield records
    graph_archive.logger.removeHandler(handler)


def _engine_dies(root: Path, read_only: bool = True) -> None:
    """Run an engine in its own process that plans, then dies."""
    args = json.dumps([str(root), dict(IDENTITY), list(RANKS), read_only])
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, sys.path)))
    subprocess.run(
        [sys.executable, "-c", _PLAN_AND_DIE, args], env=env, check=True, timeout=50
    )


def _save_ranks(workspace: Path, ranks) -> None:
    """What the backend's SAVE leaves in a workspace."""
    for rank in ranks:
        rank_dir = workspace / FOUNDRY_FORMAT.rank_dir.format(rank=rank)
        rank_dir.mkdir()
        (rank_dir / "graph_0.cugraph").write_bytes(b"g")
        (rank_dir / FOUNDRY_FORMAT.complete_marker).write_text("{}")


def _refuse(workspace: Path, rank: int, record: str, reason: str) -> None:
    """What a backend rank that refused the archive leaves behind."""
    rank_dir = workspace / FOUNDRY_FORMAT.rank_dir.format(rank=rank)
    rank_dir.mkdir(exist_ok=True)
    (rank_dir / record).write_text(
        json.dumps({"rank": f"rank_{rank}", "reason": reason})
    )


def _commit_save(engine) -> GraphArchive:
    saver = engine()
    assert saver.plan(read_only=True) == ArchiveMode.SAVE
    _save_ranks(saver.workspace, RANKS)
    saver.commit()
    return saver


def _fail_load(engine, reason: str = "engine start failed") -> None:
    loader = engine()
    assert loader.plan(read_only=True) == ArchiveMode.LOAD
    loader.abort(reason)


def _fail_save(engine, reason: str = "engine start failed") -> None:
    saver = engine()
    assert saver.plan(read_only=True) == ArchiveMode.SAVE
    saver.abort(reason)


def _set_aside_records(root: Path) -> list[dict]:
    return [
        json.loads((d / "reason.json").read_text())
        for d in sorted((root / ".quarantine").iterdir())
    ]


def test_first_read_only_engine_saves_and_later_engines_load(tmp_path, engine):
    saver = engine()
    assert saver.plan(read_only=True) == ArchiveMode.SAVE
    assert saver.workspace.parent == tmp_path / ".staging"
    assert not saver.path.exists()  # invisible until every rank is done

    _save_ranks(saver.workspace, RANKS)
    saver.commit()

    assert (saver.path / "manifest.json").is_file()
    assert not list((tmp_path / ".staging").iterdir())
    for read_only in (True, False):
        loader = engine()
        assert loader.plan(read_only=read_only) == ArchiveMode.LOAD
        assert loader.workspace == saver.path
        loader.commit()
    assert not list((tmp_path / ".loading").iterdir())


def test_engine_that_may_write_the_weights_never_saves(tmp_path, engine):
    archive = engine()
    assert archive.plan(read_only=False) == ArchiveMode.NATIVE
    assert archive.workspace is None
    assert not list((tmp_path / ".staging").iterdir())


def test_only_one_engine_saves_at_a_time(engine):
    first, second = engine(), engine()
    assert first.plan(read_only=True) == ArchiveMode.SAVE
    assert second.plan(read_only=True) == ArchiveMode.NATIVE

    # The second engine's plan left the live save alone.
    _save_ranks(first.workspace, RANKS)
    first.commit()
    assert engine().plan(read_only=True) == ArchiveMode.LOAD


def test_incomplete_save_is_set_aside_and_not_retried(tmp_path, engine):
    saver = engine()
    saver.plan(read_only=True)
    _save_ranks(saver.workspace, [0])
    saver.commit()

    assert not saver.path.exists()
    (record,) = _set_aside_records(tmp_path)
    assert record["reason"] == "SAVE left ranks [1] incomplete"
    assert engine().plan(read_only=True) == ArchiveMode.NATIVE


def test_failed_load_without_a_refusal_keeps_the_archive(tmp_path, engine):
    _commit_save(engine)
    _fail_load(engine, "ValueError: free memory below utilization")

    assert not _set_aside_records(tmp_path)
    assert not list((tmp_path / ".loading").iterdir())
    for read_only in (True, False):
        assert engine().plan(read_only=read_only) == ArchiveMode.LOAD


def test_load_refusal_sets_the_archive_aside_with_each_ranks_reason(tmp_path, engine):
    saver = _commit_save(engine)
    for rank in RANKS:
        _refuse(saver.path, rank, FOUNDRY_FORMAT.load_refusal, "identity differs")
    _fail_load(engine, "RuntimeError: Engine core initialization failed")

    (record,) = _set_aside_records(tmp_path)
    assert record == {
        "reason": "LOAD refused: rank_0: identity differs; rank_1: identity differs",
        "refusals": {"rank_0": "identity differs", "rank_1": "identity differs"},
        "engine_error": "RuntimeError: Engine core initialization failed",
    }
    for read_only in (True, False):
        assert engine().plan(read_only=read_only) == ArchiveMode.NATIVE
    assert len(_set_aside_records(tmp_path)) == 1


def test_unparsed_refusal_record_is_kept_verbatim(tmp_path, engine):
    saver = _commit_save(engine)
    (saver.path / "rank_1" / FOUNDRY_FORMAT.load_refusal).write_text("not json\n")
    _fail_load(engine)

    assert _set_aside_records(tmp_path)[0]["refusals"] == {"rank_1": "not json"}


def test_archive_set_aside_under_a_live_loader_stays_in_place(tmp_path, engine):
    saver = _commit_save(engine)
    refused, live = engine(), engine()
    assert refused.plan(read_only=True) == ArchiveMode.LOAD
    assert live.plan(read_only=True) == ArchiveMode.LOAD
    _refuse(saver.path, 0, FOUNDRY_FORMAT.load_refusal, "identity differs")
    refused.abort("RuntimeError: Engine core initialization failed")

    # The engine still loading reads the same files, and its own start is not
    # charged to the archive.
    assert (live.workspace / "rank_1" / "graph_0.cugraph").is_file()
    live.commit()
    assert len(_set_aside_records(tmp_path)) == 1
    assert engine().plan(read_only=True) == ArchiveMode.NATIVE


def test_failed_save_without_a_refusal_is_retried(tmp_path, engine):
    saver = engine()
    saver.plan(read_only=True)
    _save_ranks(saver.workspace, [0])
    saver.abort("RuntimeError: NCCL error")

    assert not _set_aside_records(tmp_path)
    assert not list((tmp_path / ".staging").iterdir())
    assert engine().plan(read_only=True) == ArchiveMode.SAVE


def test_save_refusal_sets_the_save_aside_and_is_not_retried(tmp_path, engine):
    saver = engine()
    saver.plan(read_only=True)
    _refuse(saver.workspace, 0, FOUNDRY_FORMAT.save_refusal, "break allocated")
    saver.abort("RuntimeError: Engine core initialization failed")

    (record,) = _set_aside_records(tmp_path)
    assert record["reason"] == "SAVE refused: rank_0: break allocated"
    assert record["refusals"] == {"rank_0": "break allocated"}
    assert not list((tmp_path / ".staging").iterdir())
    assert engine().plan(read_only=True) == ArchiveMode.NATIVE


def test_consecutive_failed_starts_set_the_archive_aside(tmp_path, engine):
    _commit_save(engine)
    for attempt in range(MAX_FAILED_STARTS):
        assert not _set_aside_records(tmp_path)
        _fail_load(engine, f"failure {attempt}")

    (record,) = _set_aside_records(tmp_path)
    assert record["reason"] == (
        f"{MAX_FAILED_STARTS} starts in a row failed; "
        f"last: LOAD failed: failure {MAX_FAILED_STARTS - 1}"
    )
    assert record["failed_starts"] == [
        f"LOAD failed: failure {i}" for i in range(MAX_FAILED_STARTS)
    ]
    for read_only in (True, False):
        assert engine().plan(read_only=read_only) == ArchiveMode.NATIVE


def test_consecutive_failed_saves_stop_saving(tmp_path, engine):
    for _ in range(MAX_FAILED_STARTS):
        _fail_save(engine)

    assert len(_set_aside_records(tmp_path)) == 1
    assert engine().plan(read_only=True) == ArchiveMode.NATIVE


def test_a_good_start_resets_the_failure_count(tmp_path, engine):
    for _ in range(MAX_FAILED_STARTS - 1):
        _fail_save(engine)
    _commit_save(engine)
    for _ in range(MAX_FAILED_STARTS - 1):
        _fail_load(engine)
    loader = engine()
    assert loader.plan(read_only=True) == ArchiveMode.LOAD
    loader.commit()
    for _ in range(MAX_FAILED_STARTS - 1):
        _fail_load(engine)

    assert not _set_aside_records(tmp_path)


def test_engine_that_may_write_captures_natively_after_its_load_failed(
    tmp_path, engine
):
    """An engine that may write the weights and is granted the write lock (no
    weights published, e.g. after a GMS server restart) fails in load mode. It
    is the only engine that can publish, so its restart must not load again.
    Only that restart: the archive stays good for every later engine."""
    _commit_save(engine)
    writer = engine()
    assert writer.plan(read_only=False) == ArchiveMode.LOAD
    writer.abort("RuntimeError: Engine core initialization failed")

    assert engine().plan(read_only=False) == ArchiveMode.NATIVE
    assert not _set_aside_records(tmp_path)
    for read_only in (True, False):
        loader = engine()
        assert loader.plan(read_only=read_only) == ArchiveMode.LOAD
        loader.commit()


@pytest.mark.timeout(60)
def test_engine_that_died_while_saving_counts_as_a_failed_start(tmp_path, engine):
    _engine_dies(tmp_path)

    assert engine().plan(read_only=True) == ArchiveMode.SAVE
    assert not _set_aside_records(tmp_path)
    assert len(list((tmp_path / ".staging").iterdir())) == 1  # the dead save is gone


@pytest.mark.timeout(180)
def test_engines_that_keep_dying_while_saving_stop_saving(tmp_path, engine):
    for _ in range(MAX_FAILED_STARTS):
        _engine_dies(tmp_path)

    assert engine().plan(read_only=True) == ArchiveMode.NATIVE
    (record,) = _set_aside_records(tmp_path)
    assert record["reason"] == (
        f"{MAX_FAILED_STARTS} starts in a row failed; last: {DIED_SAVING}"
    )


@pytest.mark.timeout(60)
def test_engine_that_died_while_loading_counts_as_one_failed_start(tmp_path, engine):
    _commit_save(engine)
    _engine_dies(tmp_path)
    for _ in range(MAX_FAILED_STARTS - 1):
        _fail_load(engine)

    (record,) = _set_aside_records(tmp_path)
    assert record["failed_starts"] == [DIED_LOADING] + [
        "LOAD failed: engine start failed"
    ] * (MAX_FAILED_STARTS - 1)


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    ("died_read_only", "next_writer_mode"),
    [(False, ArchiveMode.NATIVE), (True, ArchiveMode.LOAD)],
)
def test_engine_that_may_write_and_died_while_loading_captures_natively_next(
    tmp_path, engine, died_read_only, next_writer_mode
):
    """A may-write engine killed in load mode never reaches abort(), and may
    have held the write lock all the same. Whoever finds its marker applies
    the rule abort() applies; a dead read-only engine does not trigger it."""
    _commit_save(engine)
    _engine_dies(tmp_path, read_only=died_read_only)

    reader = engine()
    assert reader.plan(read_only=True) == ArchiveMode.LOAD
    reader.commit()
    assert engine().plan(read_only=False) == next_writer_mode
    assert engine().plan(read_only=False) == ArchiveMode.LOAD
    assert not _set_aside_records(tmp_path)


@pytest.mark.timeout(60)
def test_loader_that_died_before_a_good_start_is_not_counted_after_it(tmp_path, engine):
    """Deaths are found late, by the next engine to look. One that happened
    before a start succeeded is not charged to the failures before that start
    or to the starts after it."""
    _commit_save(engine)
    for _ in range(MAX_FAILED_STARTS - 1):
        _fail_load(engine)
    loader = engine()
    assert loader.plan(read_only=True) == ArchiveMode.LOAD
    _engine_dies(tmp_path)
    loader.commit()

    assert not list((tmp_path / ".loading").iterdir())
    for _ in range(MAX_FAILED_STARTS - 1):
        _fail_load(engine)
    assert not _set_aside_records(tmp_path)
    assert engine().plan(read_only=True) == ArchiveMode.LOAD


@pytest.mark.timeout(60)
def test_refusal_left_by_a_loader_that_died_before_a_good_start_is_kept(
    tmp_path, engine
):
    """Leaving such a death out of the count does not drop its evidence."""
    saver = _commit_save(engine)
    loader = engine()
    assert loader.plan(read_only=True) == ArchiveMode.LOAD
    _engine_dies(tmp_path)
    _refuse(saver.path, 0, FOUNDRY_FORMAT.load_refusal, "identity differs")
    loader.commit()

    (record,) = _set_aside_records(tmp_path)
    assert record["engine_error"] == DIED_LOADING
    assert engine().plan(read_only=True) == ArchiveMode.NATIVE


def test_marker_that_cannot_be_removed_is_not_left_locked(
    tmp_path, engine, monkeypatch
):
    """An engine that fails to remove a dead loader's marker must not keep the
    flock it took to check it: the marker would look live to every later
    engine, and the death would never be counted."""
    key = _commit_save(engine).key
    marker = tmp_path / ".loading" / f"{key}.deadbeef"
    marker.touch()  # what a dead loader leaves: a marker nobody holds
    real_unlink = Path.unlink

    def unlink(path, missing_ok=False):
        if path == marker:
            raise PermissionError(f"injected: {path}")
        real_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", unlink)
    assert engine().plan(read_only=True) == ArchiveMode.NATIVE
    monkeypatch.setattr(Path, "unlink", real_unlink)

    assert engine().plan(read_only=True) == ArchiveMode.LOAD
    assert not marker.exists()
    failures = json.loads((tmp_path / ".failures" / f"{key}.json").read_text())
    assert failures == [DIED_LOADING]


def test_lock_and_marker_files_are_for_their_owner_only(tmp_path, engine):
    """Whoever can open a lock file can hold its flock and stall every start,
    so the mode is not left to the umask."""
    previous = os.umask(0)
    try:
        _commit_save(engine)
        assert engine().plan(read_only=False) == ArchiveMode.LOAD
    finally:
        os.umask(previous)

    files = [p for sub in (".locks", ".loading") for p in (tmp_path / sub).iterdir()]
    assert len(files) == 3  # the state lock, the save lock and the load marker
    assert {stat.S_IMODE(p.stat().st_mode) for p in files} == {0o600}


@pytest.mark.timeout(60)
@pytest.mark.parametrize("mode", [ArchiveMode.LOAD, ArchiveMode.SAVE])
def test_refusal_left_by_an_engine_that_died_sets_the_archive_aside(
    tmp_path, engine, mode
):
    if mode == ArchiveMode.LOAD:
        workspace = _commit_save(engine).path
        record, died = FOUNDRY_FORMAT.load_refusal, DIED_LOADING
    else:
        record, died = FOUNDRY_FORMAT.save_refusal, DIED_SAVING
    _engine_dies(tmp_path)
    if mode == ArchiveMode.SAVE:
        (workspace,) = (tmp_path / ".staging").iterdir()
    _refuse(workspace, 0, record, "identity differs")

    assert engine().plan(read_only=True) == ArchiveMode.NATIVE
    (set_aside,) = _set_aside_records(tmp_path)
    assert set_aside["refusals"] == {"rank_0": "identity differs"}
    assert set_aside["engine_error"] == died


def test_live_loader_is_not_mistaken_for_a_dead_one(tmp_path, engine):
    _commit_save(engine)
    first, second = engine(), engine()
    assert first.plan(read_only=True) == ArchiveMode.LOAD
    assert second.plan(read_only=True) == ArchiveMode.LOAD
    first.commit()
    second.commit()

    assert not (tmp_path / ".failures" / f"{first.key}.json").exists()
    assert not _set_aside_records(tmp_path)


@pytest.mark.parametrize(
    ("damage", "reason"),
    [
        pytest.param(
            lambda archive: (archive / "manifest.json").write_text(
                json.dumps({"format": 0})
            ),
            "manifest does not match this engine",
            id="other-manifest",
        ),
        pytest.param(
            lambda archive: (archive / "manifest.json").write_text("{"),
            "unreadable manifest",
            id="unreadable-manifest",
        ),
        pytest.param(
            lambda archive: (
                archive / "rank_1" / FOUNDRY_FORMAT.complete_marker
            ).unlink(),
            "ranks [1] are incomplete",
            id="incomplete-rank",
        ),
    ],
)
def test_committed_archive_that_cannot_be_loaded_is_set_aside(
    tmp_path, engine, damage, reason
):
    damage(_commit_save(engine).path)

    assert engine().plan(read_only=True) == ArchiveMode.NATIVE
    (record,) = _set_aside_records(tmp_path)
    assert record["reason"].startswith(reason)


def test_unusable_directory_falls_back_to_native_capture(tmp_path, engine, warnings):
    not_a_directory = tmp_path / "file"
    not_a_directory.write_text("")
    archive = engine(root=not_a_directory)

    assert archive.plan(read_only=True) == ArchiveMode.NATIVE
    assert "archive directory unusable" in warnings[0].getMessage()
    archive.commit()
    archive.abort("engine start failed")


def test_state_lock_held_by_a_stuck_engine_does_not_block_a_start(
    tmp_path, engine, monkeypatch, warnings
):
    archive = engine()
    (tmp_path / ".locks").mkdir()
    stuck = os.open(
        tmp_path / ".locks" / f"{archive.key}.state", os.O_RDWR | os.O_CREAT
    )
    fcntl.flock(stuck, fcntl.LOCK_EX)
    monkeypatch.setattr(graph_archive, "STATE_LOCK_TIMEOUT_S", 0)
    try:
        assert archive.plan(read_only=True) == ArchiveMode.NATIVE
    finally:
        os.close(stuck)

    assert "is held by another engine" in warnings[0].getMessage()


def test_commit_does_not_fail_a_started_engine(engine):
    saver = engine()
    saver.plan(read_only=True)
    saver.workspace.rmdir()  # e.g. the volume was cleaned under the engine

    saver.commit()

    assert not saver.path.exists()
    assert engine().plan(read_only=True) == ArchiveMode.SAVE


def test_abort_does_not_raise_over_the_start_failure(tmp_path, engine, warnings):
    saver = engine()
    saver.plan(read_only=True)
    shutil.rmtree(tmp_path / ".failures")
    (tmp_path / ".failures").write_text("")  # the count cannot be written

    saver.abort("RuntimeError: Engine core initialization failed")

    assert "not set aside" in warnings[0].getMessage()


def test_commit_or_abort_ends_the_engines_use(tmp_path, engine):
    saver = _commit_save(engine)
    saver.abort("a later error of the caller")
    saver.commit()
    loader = engine()
    assert loader.plan(read_only=True) == ArchiveMode.LOAD
    loader.commit()
    loader.abort("a later error of the caller")
    loader.abort("a later error of the caller")

    assert not (tmp_path / ".failures" / f"{loader.key}.json").exists()
    assert engine().plan(read_only=True) == ArchiveMode.LOAD


def test_archives_for_different_identities_or_ranks_do_not_meet(engine):
    saver = _commit_save(engine)
    other = engine(identity={**IDENTITY, "engine.max_num_seqs": 8})
    fewer_ranks = engine(ranks=(0,))

    assert len({saver.key, other.key, fewer_ranks.key}) == 3
    assert other.plan(read_only=True) == ArchiveMode.SAVE
    assert fewer_ranks.plan(read_only=True) == ArchiveMode.SAVE


class _Mode(enum.Enum):
    NONE = 0


@dataclasses.dataclass
class _Compilation:
    mode: _Mode = _Mode.NONE
    capture_sizes: tuple = (1, 2)
    hook: object = dataclasses.field(default_factory=object)


def test_key_is_the_same_for_equal_identities():
    identity = {
        "model": Path("/models/m"),
        "compilation": _Compilation(),
        "backends": set("hgfedcba"),
    }
    same = {
        "backends": sorted("abcdefgh"),
        "compilation": {"mode": 0, "capture_sizes": [1, 2], "hook": "object"},
        "model": "/models/m",
    }
    other = {**identity, "compilation": _Compilation(capture_sizes=(1,))}

    assert archive_key(identity, RANKS) == archive_key(same, RANKS)
    assert archive_key(identity, RANKS) != archive_key(other, RANKS)


def _engine_args(**overrides) -> SimpleNamespace:
    """Engine arguments with every field the identity reads."""
    fields = dict.fromkeys(graph_archive._ENGINE_ARGS)
    fields.update(
        model=Path("/models/m"),
        max_num_seqs=8,
        tensor_parallel_size=2,
        compilation_config=_Compilation(),
        model_loader_extra_config={},
        master_port=29500,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _engine_key(engine_args) -> str:
    identity = engine_identity(engine_args, {"region_size": "8GB"}, ["pytest"])
    return archive_key(identity, RANKS)


def test_key_ignores_what_differs_between_the_engines_of_a_pod():
    base = _engine_args()
    other_engine = _engine_args(
        master_port=base.master_port + 100,
        model_loader_extra_config={
            "gms_read_only": True,
            "gms_ro_connect_timeout_ms": 1000,
            LOADER_OPTION: "load",
        },
        compilation_config=_Compilation(),  # a different, equal object
    )

    assert _engine_key(base) == _engine_key(other_engine)
    assert _engine_key(base) != _engine_key(_engine_args(max_num_seqs=16))
    assert _engine_key(base) != _engine_key(_engine_args(moe_backend="triton"))
    assert _engine_key(base) != _engine_key(
        _engine_args(model_loader_extra_config={"gms_reconstruction_mode": "load_diff"})
    )
    assert _engine_key(base) != _engine_key(
        _engine_args(compilation_config=_Compilation(capture_sizes=(1,)))
    )


def test_key_covers_backend_settings_and_versions():
    args = _engine_args()
    identity = engine_identity(
        args, {"region_size": "8GB"}, ["pytest", "no-such-package"]
    )
    other = engine_identity(
        args, {"region_size": "16GB"}, ["pytest", "no-such-package"]
    )

    assert identity["version.no-such-package"] is None
    assert identity["version.pytest"] == pytest.__version__
    assert archive_key(identity, RANKS) != archive_key(other, RANKS)


def test_read_only_attach_is_required_only_in_archive_mode():
    check_read_only_attach({}, GrantedLockType.RW)
    check_read_only_attach({LOADER_OPTION: "load"}, GrantedLockType.RO)
    with pytest.raises(GraphArchiveError, match="read-only weights attach"):
        check_read_only_attach({LOADER_OPTION: "load"}, GrantedLockType.RW)
