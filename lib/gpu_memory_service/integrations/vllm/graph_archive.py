# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decide whether a GMS vLLM engine saves, loads or natively captures its CUDA graphs.

A graph-persistence backend (Foundry) can save an engine's CUDA graphs to a
directory and rebuild them in a later engine without running the model. This
module owns the policy around that directory, not the archive format:

- Only an engine that attaches its weights read-only saves, so graphs are saved
  by the kind of engine that loads them. The first read-only engine for an
  identity saves; an engine that may write the weights never does.
- A save becomes visible only once every rank has finished: it is written to a
  staging directory and renamed into place. Every later engine loads it.
- An archive the backend refused, at save or at load, is set aside, and later
  engines capture natively. A start that fails for another reason keeps the
  archive, so an unrelated failure does not discard a good one, and a failed
  save is dropped and saved again. After ``MAX_FAILED_STARTS`` such failures in
  a row the archive is set aside anyway, so a crash loop cannot retry it
  forever.
- An engine that may write the weights loads like any other, because it
  attaches read-only whenever weights are published. If that start fails, or
  the engine dies in it, its next start captures natively: it may have been
  granted the write lock (nothing published yet, for example after the GMS
  server restarted), and a writer cannot use graphs saved by a reader.

Engines in different containers coordinate with ``flock`` on files in the
directory, which the kernel releases when a process dies. The directory must
be writable and on a filesystem whose ``flock`` those containers share (an
``emptyDir`` is), and the engines must run as the same user, because only
their owner can open the lock files; otherwise every engine captures natively.
A set-aside lasts as long as the directory. To start over, delete ``<key>/``
and ``.quarantine/<key>.*``.
"""

from __future__ import annotations

import contextlib
import dataclasses
import enum
import fcntl
import hashlib
import importlib.metadata
import json
import logging
import os
import shutil
import time
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from gpu_memory_service.common.locks import GrantedLockType

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1
MANIFEST = "manifest.json"
QUARANTINE_REASON = "reason.json"

# Consecutive failed starts in save or load mode, none refused by the backend,
# after which the archive is set aside. A start can fail for reasons unrelated
# to the archive (the next start may succeed), but a deterministic failure must
# not make every restart pay for a doomed save or load. The count lives in the
# archive directory, so it lasts as long as the directory (the pod, for an
# emptyDir) and is reset by any engine that starts in save or load mode.
MAX_FAILED_STARTS = 3

# The state lock is held for a few directory operations. An engine that cannot
# get it in this time (the holder is stopped, or the filesystem hangs) captures
# natively instead of waiting.
STATE_LOCK_TIMEOUT_S = 30.0
_STATE_LOCK_POLL_S = 0.05

# Lock and marker files hold no data. Whoever can open one can hold its flock
# and stall every engine's start, so only their owner may.
_LOCK_FILE_MODE = 0o600
# Ends the name of the load marker of an engine that may write the weights.
_WRITER_SUFFIX = ".writer"

# model_loader_extra_config key telling the GMS worker that its graphs are
# saved to or loaded from an archive ("save" or "load").
LOADER_OPTION = "gms_graph_archive"

# Loader options that differ between the engines of one deployment and do not
# change which graphs are captured.
_PER_ENGINE_LOADER_OPTIONS = frozenset(
    {"gms_read_only", "gms_ro_connect_timeout_ms", LOADER_OPTION}
)

# Engine arguments that decide which graphs exist and what is allocated around
# them. An allow-list, because the engines of one deployment must get the same
# key and many other arguments differ between them (ports, per-process ids).
# An argument missing here only matters in a directory that outlives a
# configuration change: the backend then refuses the old archive at load.
_ENGINE_ARGS = (
    "model",
    "revision",
    "code_revision",
    "hf_config_path",
    "hf_overrides",
    "trust_remote_code",
    "model_impl",
    "load_format",
    "dtype",
    "quantization",
    "kv_cache_dtype",
    "kv_cache_dtype_skip_layers",
    "max_model_len",
    "block_size",
    "num_gpu_blocks_override",
    "gpu_memory_utilization",
    "kv_cache_memory_bytes",
    "max_num_seqs",
    "max_num_batched_tokens",
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "data_parallel_size",
    "enable_expert_parallel",
    "enable_eplb",
    "enable_dbo",
    "all2all_backend",
    "disable_custom_all_reduce",
    "enforce_eager",
    "cudagraph_capture_sizes",
    "max_cudagraph_capture_size",
    "optimization_level",
    "performance_mode",
    "compilation_config",
    "kernel_config",
    "attention_config",
    "attention_backend",
    "moe_backend",
    "linear_backend",
    "sparse_indexer_topk_backend",
    "speculative_config",
    "spec_method",
    "spec_model",
    "spec_tokens",
    "enable_lora",
    "max_loras",
    "max_lora_rank",
)


class ArchiveMode(str, enum.Enum):
    NATIVE = "native"
    SAVE = "save"
    LOAD = "load"


class GraphArchiveError(RuntimeError):
    """The archive cannot be used by this engine."""


@dataclasses.dataclass(frozen=True)
class ArchiveFormat:
    """The backend's per-rank layout inside an archive."""

    rank_dir: str
    # Written by SAVE into a rank directory after that rank's last capture.
    complete_marker: str
    # Written into a rank directory by a rank that refused to load the archive,
    # or refused to save it: the only evidence that the archive itself is bad.
    load_refusal: str
    save_refusal: str

    def rank_complete(self, archive: Path, rank: int) -> bool:
        return (
            archive / self.rank_dir.format(rank=rank) / self.complete_marker
        ).is_file()

    def refusals(self, archive: Path, record: str) -> dict[str, str]:
        """Each refusing rank directory's reason."""
        refusals = {}
        for path in sorted(archive.glob(f"*/{record}")):
            text = path.read_text()
            try:
                parsed = json.loads(text)
            except ValueError:
                parsed = None
            if isinstance(parsed, dict) and "reason" in parsed:
                text = str(parsed["reason"])
            refusals[path.parent.name] = text.strip()
        return refusals


# Foundry's vLLM integration: one workspace per global rank, closed by the
# allocation watermark that SAVE records after the last capture.
FOUNDRY_FORMAT = ArchiveFormat(
    rank_dir="rank_{rank}",
    complete_marker="final_alloc_offset.json",
    load_refusal="load_refused.json",
    save_refusal="save_refused.json",
)


def _plain(value: object) -> object:
    """A JSON value that is equal across processes for equal inputs."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, enum.Enum):
        return _plain(value.value)
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, Mapping):
        return {
            str(k): _plain(v)
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_plain(v) for v in value), key=json.dumps)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: _plain(getattr(value, f.name)) for f in dataclasses.fields(value)
        }
    # A default repr carries the object's address, which differs per process.
    return type(value).__qualname__


def archive_key(identity: Mapping[str, object], ranks: Sequence[int]) -> str:
    """The directory name of the archive that ``ranks`` save for ``identity``.

    The key only names the directory: the backend's own identity check at load
    is what refuses an archive that does not fit the engine.
    """
    canonical = json.dumps(
        {"identity": _plain(identity), "ranks": list(ranks)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:32]


def engine_identity(
    engine_args: Any,
    backend_settings: Mapping[str, object],
    packages: Iterable[str],
) -> dict[str, object]:
    """What an engine's archive is keyed by: engine arguments, loader options,
    backend settings and package versions.

    Call it before the engine config is created: vLLM resolves some arguments
    in place when it builds the config.
    """
    identity: dict[str, object] = {
        f"engine.{name}": _plain(getattr(engine_args, name)) for name in _ENGINE_ARGS
    }
    loader_options = engine_args.model_loader_extra_config or {}
    identity["engine.model_loader_extra_config"] = _plain(
        {k: v for k, v in loader_options.items() if k not in _PER_ENGINE_LOADER_OPTIONS}
    )
    identity["backend"] = _plain(backend_settings)
    for package in sorted(packages):
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = None
        identity[f"version.{package}"] = version
    return identity


def check_read_only_attach(loader_options: Mapping, granted: GrantedLockType) -> None:
    """Refuse an archive engine that was granted the weights write lock.

    A writer allocates differently from a reader, so graphs saved by one do not
    fit the other. This happens only when the weights are not published yet,
    for example after the GMS server restarted. The engine's next start
    captures natively (see ``GraphArchive.abort``).
    """
    mode = loader_options.get(LOADER_OPTION)
    if mode is not None and granted != GrantedLockType.RO:
        raise GraphArchiveError(
            f"[GMS] graph archive {mode} needs a read-only weights attach, but this "
            f"engine was granted {granted.value}: no published weights to attach"
        )


@contextlib.contextmanager
def _flocked(path: Path, timeout_s: float) -> Iterator[None]:
    """Hold an exclusive flock on ``path``; TimeoutError if another process
    keeps it for ``timeout_s``."""
    fd = os.open(path, os.O_RDWR | os.O_CREAT, _LOCK_FILE_MODE)
    try:
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"{path} is held by another engine") from None
                time.sleep(_STATE_LOCK_POLL_S)
        yield
    finally:
        os.close(fd)


def _try_flock(path: Path, create: bool = True) -> int | None:
    """An fd holding an exclusive flock on ``path``, or None if another process
    holds it (or ``path`` is gone and ``create`` is False)."""
    try:
        fd = os.open(path, os.O_RDWR | (os.O_CREAT if create else 0), _LOCK_FILE_MODE)
    except FileNotFoundError:
        return None
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    except OSError:
        os.close(fd)
        raise
    return fd


class GraphArchive:
    """One engine's use of an archive directory.

    ``plan()`` once before the engine starts, then ``commit()`` after it started
    or ``abort()`` if it failed to start. Whichever comes first ends this
    engine's use; later calls do nothing. Layout under ``root``::

        <key>/manifest.json, <key>/<rank dirs>   committed archive
        .staging/<key>.<token>/                  a SAVE in progress
        .quarantine/<key>.<token>/reason.json    the archive was set aside
        .loading/<key>.<token>[.writer]          flock held while an engine LOADs
        .failures/<key>.json                     failed starts since the last good one
        .failures/<key>.writer                   a may-write engine failed to load
        .locks/<key>.save                        flock held for a whole SAVE
        .locks/<key>.state                       flock around every state change

    A committed archive is never moved, because another engine may be loading
    it: once set aside it stays in place and no engine plans to load it.
    """

    def __init__(
        self,
        root: str | os.PathLike,
        identity: Mapping[str, object],
        ranks: Sequence[int],
    ):
        if not ranks:
            raise ValueError("a graph archive needs at least one rank")
        self.root = Path(root)
        self.identity = _plain(identity)
        self.ranks = list(ranks)
        self.key = archive_key(self.identity, self.ranks)
        self.mode: ArchiveMode | None = None
        self.workspace: Path | None = None
        self._format = FOUNDRY_FORMAT
        self._read_only = False
        # True from a plan() that chose SAVE or LOAD until commit() or abort().
        self._open = False
        self._save_fd: int | None = None
        self._load_fd: int | None = None
        self._load_marker: Path | None = None

    @property
    def path(self) -> Path:
        return self.root / self.key

    def _state_lock(self) -> contextlib.AbstractContextManager[None]:
        return _flocked(
            self.root / ".locks" / f"{self.key}.state", STATE_LOCK_TIMEOUT_S
        )

    def _failures_path(self) -> Path:
        return self.root / ".failures" / f"{self.key}.json"

    def _writer_failed_path(self) -> Path:
        return self.root / ".failures" / f"{self.key}.writer"

    def _is_set_aside(self) -> bool:
        return any((self.root / ".quarantine").glob(f"{self.key}.*"))

    def _set_aside(
        self, reason: str, staging: Path | None = None, **details: object
    ) -> None:
        """Record that no engine may use this key's archive, and why.

        A SAVE's ``staging`` directory is moved next to the record; a committed
        archive stays where it is.
        """
        target = self.root / ".quarantine" / f"{self.key}.{uuid.uuid4().hex[:8]}"
        if staging is None:
            target.mkdir()
        else:
            os.rename(staging, target)
        record = {"reason": reason, **details}
        (target / QUARANTINE_REASON).write_text(json.dumps(record, indent=1))
        self._failures_path().unlink(missing_ok=True)
        logger.warning("[GMS] Graph archive set aside at %s: %s", target, reason)

    def _refused(
        self, mode: ArchiveMode, error: str, staging: Path | None = None
    ) -> bool:
        """Set the archive (or a SAVE's ``staging``) aside if the backend
        refused it; True if it did."""
        if mode == ArchiveMode.LOAD:
            refusals = self._format.refusals(self.path, self._format.load_refusal)
        elif staging is not None:
            refusals = self._format.refusals(staging, self._format.save_refusal)
        else:
            refusals = {}
        if not refusals:
            return False
        summary = "; ".join(f"{rank}: {why}" for rank, why in refusals.items())
        self._set_aside(
            f"{mode.name} refused: {summary}",
            staging,
            refusals=refusals,
            engine_error=error,
        )
        return True

    def _failed_start(self, reason: str, staging: Path | None = None) -> None:
        """Count a failed start the backend did not refuse.

        The archive is kept (a SAVE's partial ``staging`` is dropped, so the
        next engine saves again) until ``MAX_FAILED_STARTS`` in a row.
        """
        path = self._failures_path()
        failures = json.loads(path.read_text()) if path.exists() else []
        if not isinstance(failures, list):
            raise ValueError(f"{path} does not hold a list of failed starts")
        failures.append(reason)
        if len(failures) >= MAX_FAILED_STARTS:
            self._set_aside(
                f"{len(failures)} starts in a row failed; last: {reason}",
                staging,
                failed_starts=failures,
            )
            return
        if staging is not None:
            shutil.rmtree(staging)
        tmp = path.with_suffix(f".{uuid.uuid4().hex[:8]}")
        tmp.write_text(json.dumps(failures))
        os.replace(tmp, path)
        logger.warning(
            "[GMS] Graph archive %s kept after a failed start (%d of %d before it "
            "is set aside): %s",
            self.key,
            len(failures),
            MAX_FAILED_STARTS,
            reason,
        )

    def _failed(
        self,
        mode: ArchiveMode,
        error: str,
        counted_as: str | None,
        staging: Path | None = None,
    ) -> None:
        """A start in ``mode`` failed with ``error``, or an engine died in it.

        ``counted_as`` None: a refusal still sets the archive aside, but the
        failure is not counted.
        """
        if mode == ArchiveMode.LOAD:
            if self._is_set_aside() or not self.path.exists():
                return
        elif staging is not None and not staging.exists():
            staging = None
        if not self._refused(mode, error, staging) and counted_as is not None:
            self._failed_start(counted_as, staging)

    def _writer_failed(self) -> None:
        """An engine that may write the weights failed or died in LOAD. It may
        have been granted the write lock: its next start captures natively."""
        if self.path.exists() and not self._is_set_aside():
            self._writer_failed_path().touch()

    def _committed_problem(self) -> str | None:
        """Why the committed archive cannot be loaded, or None if it can."""
        try:
            manifest = json.loads((self.path / MANIFEST).read_text())
        except (OSError, ValueError) as e:
            return f"unreadable manifest: {e}"
        if manifest != self._manifest():
            return "manifest does not match this engine"
        missing = [
            r for r in self.ranks if not self._format.rank_complete(self.path, r)
        ]
        return f"ranks {missing} are incomplete" if missing else None

    def _reap_dead_loaders(self, count: bool = True) -> None:
        """Handle what an engine that died mid-LOAD left behind, as a failed
        start of that engine; ``count`` False leaves it out of the count."""
        for marker in sorted((self.root / ".loading").glob(f"{self.key}.*")):
            fd = _try_flock(marker, create=False)
            if fd is None:
                continue
            try:
                marker.unlink()
            finally:
                # A marker left locked would look like a live loader to every
                # later engine for as long as this one runs.
                os.close(fd)
            died = "an engine died while loading this archive"
            self._failed(ArchiveMode.LOAD, died, died if count else None)
            if marker.name.endswith(_WRITER_SUFFIX):
                self._writer_failed()

    def _reap_dead_engines(self) -> None:
        """Handle what an engine that died mid-SAVE or mid-LOAD left behind, as
        a failed start of that engine."""
        self._reap_dead_loaders()
        staging = sorted((self.root / ".staging").glob(f"{self.key}.*"))
        if staging:
            fd = _try_flock(self.root / ".locks" / f"{self.key}.save")
            if fd is not None:
                died = "an engine died while saving this archive"
                try:
                    for directory in staging:
                        self._failed(ArchiveMode.SAVE, died, died, directory)
                finally:
                    os.close(fd)

    def plan(self, read_only: bool) -> ArchiveMode:
        """Choose this engine's mode and claim what it needs.

        ``read_only``: the engine is configured to attach its weights read-only.
        """
        if self.mode is not None:
            raise RuntimeError("plan() was already called")
        self._read_only = read_only
        level = logging.INFO
        try:
            reason = self._plan()
        except (OSError, ValueError) as e:
            # The archive only saves time: a directory problem must not stop an
            # engine that can capture natively.
            self._release()
            self.mode, self.workspace = ArchiveMode.NATIVE, None
            reason, level = f"archive directory unusable: {e}", logging.WARNING
        self._open = self.mode != ArchiveMode.NATIVE
        logger.log(
            level,
            "[GMS] Graph archive %s: %s (%s), workspace %s",
            self.key,
            self.mode.value,
            reason,
            self.workspace,
        )
        return self.mode

    def _plan(self) -> str:
        for sub in (".staging", ".quarantine", ".loading", ".failures", ".locks"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        with self._state_lock():
            self._reap_dead_engines()
            self.mode = ArchiveMode.NATIVE
            if self._is_set_aside():
                return "archive was set aside"
            if not self.path.exists():
                if not self._read_only:
                    return "engine may write the weights"
                return self._start_save()
            problem = self._committed_problem()
            if problem is not None:
                self._set_aside(problem)
                return "archive was set aside"
            if not self._read_only and self._writer_failed_path().exists():
                self._writer_failed_path().unlink()
                return "an engine that may write the weights failed to load"
            self._start_load()
            return "archive found"

    def _start_load(self) -> None:
        # The name tells whoever finds the marker of a dead engine whether
        # that engine may write the weights.
        name = f"{self.key}.{uuid.uuid4().hex[:8]}"
        if not self._read_only:
            name += _WRITER_SUFFIX
        marker = self.root / ".loading" / name
        fd = os.open(marker, os.O_RDWR | os.O_CREAT | os.O_EXCL, _LOCK_FILE_MODE)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
        except OSError:
            os.close(fd)
            marker.unlink(missing_ok=True)
            raise
        self._load_fd, self._load_marker = fd, marker
        self.mode, self.workspace = ArchiveMode.LOAD, self.path

    def _start_save(self) -> str:
        fd = _try_flock(self.root / ".locks" / f"{self.key}.save")
        if fd is None:
            return "another engine is saving"
        self._save_fd = fd
        staging = self.root / ".staging" / f"{self.key}.{uuid.uuid4().hex[:8]}"
        staging.mkdir()
        self.mode, self.workspace = ArchiveMode.SAVE, staging
        return "first read-only engine"

    def commit(self) -> None:
        """The engine started: publish a SAVE, release a LOAD.

        Never raises: the engine's own graphs are fine, so an incomplete SAVE
        or a directory error only leaves no usable archive.
        """
        if not self._open:
            return
        try:
            with self._state_lock():
                if self.mode == ArchiveMode.SAVE:
                    self._publish()
                else:
                    # An engine that died before this start succeeded is not
                    # a failure in a row after it.
                    self._reap_dead_loaders(count=False)
                    self._failures_path().unlink(missing_ok=True)
        except OSError as e:
            logger.warning("[GMS] Graph archive %s not committed: %s", self.key, e)
        finally:
            self._release()

    def _publish(self) -> None:
        staging = self.workspace
        missing = [r for r in self.ranks if not self._format.rank_complete(staging, r)]
        if missing:
            self._set_aside(f"SAVE left ranks {missing} incomplete", staging)
            return
        (staging / MANIFEST).write_text(json.dumps(self._manifest()))
        os.rename(staging, self.path)
        self._failures_path().unlink(missing_ok=True)
        logger.info("[GMS] Graph archive %s committed at %s", self.key, self.path)

    def abort(self, reason: str) -> None:
        """The engine failed to start: set the archive aside if the backend
        refused it, else count the failure (see ``MAX_FAILED_STARTS``).

        Runs while the start failure propagates, so a directory error is logged
        rather than raised over it.
        """
        if not self._open:
            return
        try:
            with self._state_lock():
                if self.mode == ArchiveMode.SAVE:
                    self._failed(
                        self.mode, reason, f"SAVE failed: {reason}", self.workspace
                    )
                    return
                self._failed(self.mode, reason, f"LOAD failed: {reason}")
                if not self._read_only:
                    self._writer_failed()
        except (OSError, ValueError) as e:
            logger.warning("[GMS] Graph archive %s not set aside: %s", self.key, e)
        finally:
            self._release()

    def _manifest(self) -> dict[str, object]:
        return {
            "format": FORMAT_VERSION,
            "key": self.key,
            "identity": self.identity,
            "ranks": self.ranks,
        }

    def _release(self) -> None:
        """Drop the SAVE lock and the LOAD marker this engine holds."""
        self._open = False
        if self._save_fd is not None:
            os.close(self._save_fd)
            self._save_fd = None
        if self._load_fd is None or self._load_marker is None:
            return
        try:
            # Under the state lock, so a reaper never sees the marker unlocked.
            with self._state_lock():
                self._load_marker.unlink(missing_ok=True)
        except OSError as e:
            logger.warning("[GMS] Graph archive %s load marker kept: %s", self.key, e)
        finally:
            os.close(self._load_fd)
            self._load_fd = self._load_marker = None
