# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cut a retune window file from the frontends' live request-trace segments.

The frontends append ``dynamo.request.trace.v1`` rows to rotating
``<segments_root>/<pod>/req.NNNNNN.jsonl.gz`` segments on a shared RWX claim,
one gzip member per flush (``lib/llm/src/telemetry/jsonl_gz.rs``).
``cut_window`` turns the ``request_end`` rows that arrived in
``[start_ms, end_ms)`` and completed by ``as_of_ms`` into the bare-row window
file a DGDR Run replays. Selection, censoring, de-duplication, the single
block-size rule and ``row_counts`` are those of
``retune.window.read_trace_window``, and the rows are the same bytes; only
the output is gzip-compressed and the memory is bounded.

* **Members.** Each gzip member is decoded on its own and checked against its
  CRC32 and ISIZE before any of its lines is read. A member that fails (cut
  short by a segment still being written, a zero-filled hole a lagging RWX
  client shows, a flipped bit) is dropped whole and the reader resynchronises
  on the next gzip header, so one run of bad bytes may hold many members.
  Runs of bad bytes above ``MAX_DROPPED_MEMBER_RATIO`` of the members, or
  bad bytes above that ratio of the gzip bytes, fail the cut
  (``trace_corrupt``); a run that reaches the end of a writer's newest
  segment is its in-flight flush (or a crashed writer's last one) and is
  counted but does not count toward either ratio.
* **Pass 1** parses every row as the replay reader does (a row it would
  reject, in a member whose CRC is good, fails the cut as ``trace_corrupt``)
  and keeps only columns for each selected row (~45 B: arrival, completion,
  a 128-bit digest of ``request_id``, where its line is, block size, flags).
  Duplicates are removed by digest, keeping the first in file order,
  whenever the columns outgrow ``max_rows`` by a slack, so ``max_rows``
  bounds memory and is checked on de-duplicated rows.
* **Pass 2** re-reads only the members that hold kept rows, in output order
  (arrival, completion, then ``request_id`` within equal pairs), re-parses
  each line with its hashes and streams ``_window_record`` rows to gzip,
  compressed and written by a thread. Decoded lines wait in a bounded cache
  that evicts the member needed furthest in the future. A member or row
  that no longer matches pass 1 fails the cut (the segment changed).
* **Atomic output.** The file is written to ``.tmp-<window_end>-<nonce>``
  next to ``out_path``, fsynced, made 0600 and hard-linked into place; an
  existing ``out_path`` is reused (another process cut the same window).
* **Child process.** ``cut_window`` runs the cut in a ``spawn`` child, like
  the replay sweep's workers, and kills it at the deadline or on cancel, so
  a hung RWX read or a runaway cut never holds the sweep thread.
"""

from __future__ import annotations

import errno
import gzip
import hashlib
import json
import logging
import math
import multiprocessing
import os
import queue
import re
import secrets
import stat
import struct
import sys
import threading
import time
import zlib
from array import array
from collections import OrderedDict
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from multiprocessing.connection import Connection, wait
from operator import attrgetter
from pathlib import Path
from typing import Any, BinaryIO, Literal, Optional, Protocol, Union, get_args

import numpy as np

from dynamo.planner.offline.trace_data import _reject_duplicate_copies
from dynamo.planner.retune.live.contracts import WindowCutResult
from dynamo.planner.retune.window import (
    _dumps,
    _new_counts,
    _parse_request_end,
    _request_end_event,
    _RequestRow,
    _window_record,
)

logger = logging.getLogger(__name__)

StrPath = Union[str, "os.PathLike[str]"]


class _ByteSink(Protocol):
    def write(self, data: bytes, /) -> Any:
        ...


WindowCutFailure = Literal[
    "window_cut_timeout",  # the deadline passed; the child was killed
    "window_too_large",  # more than max_rows de-duplicated rows
    "trace_corrupt",  # too many bad gzip members, or a malformed row
    "window_cut_failed",  # I/O error, mixed block sizes, child died, ...
    "cancelled",  # the caller's cancel event was set; the child was killed
]
WINDOW_CUT_FAILURES: tuple[str, ...] = get_args(WindowCutFailure)

# Runs of bad bytes (other than in-flight tails) over all members, or their
# bytes over all gzip bytes, above which the cut fails as trace_corrupt. A run
# counts once however many members it wiped out (a zero-filled hole); its
# bytes measure how much of the trace it took.
MAX_DROPPED_MEMBER_RATIO = 0.01
# Window files are written fast rather than small: on hash-chain rows level
# 1 is ~5x faster than level 6 for ~8% more bytes, and the cut budget pays
# for compression.
OUTPUT_COMPRESSLEVEL = 1
# Temporary window files: ``.tmp-<window_end_ms>-<nonce>`` next to out_path.
TEMP_PREFIX = ".tmp-"

_TEMP_NAME = re.compile(r"^\.tmp-(\d+)-([0-9A-Za-z]+)$")
_SEGMENT_INDEX = re.compile(r"\.(\d+)\.jsonl(?:\.gz)?$")
_GZIP_MAGIC = b"\x1f\x8b\x08"  # ID1, ID2, CM=deflate
_GZIP_WBITS = 16 + zlib.MAX_WBITS  # one gzip member, header and trailer
_READ_CHUNK = 1 << 20
_DECODE_STEP = 1 << 16
_COMPACT_AT = 1 << 20  # consumed reader bytes before the buffer is compacted
_PLAIN_BLOCK = 1 << 20  # bytes of whole lines per plain-JSONL pseudo-member
# A trace file the reconciler deleted under the walk (ESTALE: on NFS).
_VANISHED = frozenset({errno.ENOENT, errno.ESTALE})
# Pass-1 columns may hold this fraction of max_rows (at least the minimum)
# beyond max_rows before duplicates are removed.
_DEDUP_SLACK_RATIO = 0.25
_DEDUP_SLACK_MIN = 1024
_PASS2_CACHE_BYTES = 256 << 20
_PASS2_BLOCK_ROWS = 1 << 16
_LINE_OVERHEAD = 64  # bytes object and dict slot per cached line
_OPEN_FILES = 32
_WRITE_BUFFER = 1 << 20
_WRITE_QUEUE = 4  # batches waiting for the writer thread
_FLAG_AGENT_CONTEXT = 1
_FLAG_NO_TOTAL_TIME = 2
_POLL_INTERVAL_S = 0.25
_EXIT_GRACE_S = 1.0
_JOIN_TIMEOUT_S = 5.0
_CHILD_NICENESS = 10
_CHILD_SWITCH_INTERVAL_S = 0.0005
_MAX_DROP_EXAMPLES = 5


class WindowCutError(Exception):
    """A window cut failed; ``reason`` is one of ``WINDOW_CUT_FAILURES``."""

    def __init__(self, reason: WindowCutFailure, message: str) -> None:
        if reason not in WINDOW_CUT_FAILURES:
            raise ValueError(f"unknown window cut failure reason {reason!r}")
        super().__init__(message)
        self.reason: WindowCutFailure = reason
        self.message = message

    def __reduce__(self) -> tuple[Any, ...]:
        return (type(self), (self.reason, self.message))


def temp_file_name(window_end_ms: float, nonce: Optional[str] = None) -> str:
    """The name of a window's temporary file (``.tmp-<window_end_ms>-<nonce>``)."""

    if nonce is None:
        nonce = secrets.token_hex(8)
    elif not re.fullmatch(r"[0-9A-Za-z]+", nonce):
        raise ValueError(f"nonce must be alphanumeric, got {nonce!r}")
    return f"{TEMP_PREFIX}{int(round(window_end_ms))}-{nonce}"


def temp_file_window_end(name: str) -> Optional[int]:
    """The window end of a ``temp_file_name`` name; ``None`` for any other name."""

    match = _TEMP_NAME.match(name)
    return int(match.group(1)) if match else None


def segment_index(name: str) -> Optional[int]:
    """The ``NNNNNN`` of a ``<prefix>.NNNNNN.jsonl[.gz]`` segment name, else ``None``."""

    match = _SEGMENT_INDEX.search(name)
    return int(match.group(1)) if match else None


# ------------------------------------------------------------- entry points


def cut_window(
    segments_root: StrPath,
    start_ms: float,
    end_ms: float,
    as_of_ms: float,
    out_path: StrPath,
    *,
    max_rows: int,
    deadline_s: float,
    cancel: Optional[threading.Event] = None,
) -> WindowCutResult:
    """Cut a window file in a spawned child process (``contracts.WindowCutter``).

    ``deadline_s`` is absolute wall-clock time (``time.time()`` seconds), e.g.
    ``min(now + cut_budget_seconds, D_k)``. The child is killed at the
    deadline (``window_cut_timeout``) or as soon as ``cancel`` is set
    (``cancelled``; any object with ``is_set()`` works, so a caller can pass
    one that watches several events). Every failure raises ``WindowCutError``;
    invalid arguments raise ``ValueError`` before a child is started.
    """

    _validate(start_ms, end_ms, as_of_ms, out_path, max_rows)
    if not math.isfinite(deadline_s):
        raise ValueError("deadline_s must be finite")
    deadline = time.monotonic() + (deadline_s - time.time())
    out_path = os.fspath(out_path)
    temp_path = os.path.join(
        os.path.dirname(os.path.abspath(out_path)), temp_file_name(end_ms)
    )
    started = time.monotonic()
    result = _run_in_child(
        cut_window_inprocess,
        (os.fspath(segments_root), start_ms, end_ms, as_of_ms, out_path),
        {"max_rows": max_rows, "temp_path": temp_path},
        deadline=deadline,
        cancel=cancel,
        cleanup=(temp_path,),
        name=f"retune-window-cut-{int(round(end_ms))}",
    )
    logger.info(
        "retune window [%s, %s) as of %s: %d requests in %s (%s) in %.1fs; "
        "dropped %d gzip members (%d bytes)",
        start_ms,
        end_ms,
        as_of_ms,
        result.num_requests,
        result.path,
        "reused" if result.reused else "written",
        time.monotonic() - started,
        result.dropped_members,
        result.dropped_bytes,
    )
    return result


def cut_window_inprocess(
    segments_root: StrPath,
    start_ms: float,
    end_ms: float,
    as_of_ms: float,
    out_path: StrPath,
    *,
    max_rows: int,
    temp_path: Optional[StrPath] = None,
) -> WindowCutResult:
    """Cut a window file in this process; the body of ``cut_window``'s child.

    An empty window writes no file (``path=None``). ``temp_path`` names the
    temporary file (it must be in ``out_path``'s directory); by default a
    fresh ``temp_file_name`` there. Failures raise ``WindowCutError``.
    """

    _validate(start_ms, end_ms, as_of_ms, out_path, max_rows)
    out_path = os.fspath(out_path)
    directory = os.path.dirname(os.path.abspath(out_path))
    if temp_path is None:
        temp_path = os.path.join(directory, temp_file_name(end_ms))
    temp_path = os.fspath(temp_path)
    if os.path.dirname(os.path.abspath(temp_path)) != directory:
        # os.link needs one filesystem; the reconciler looks for it there.
        raise ValueError(f"temp_path {temp_path!r} is not next to {out_path!r}")
    try:
        files = _segment_files(os.fspath(segments_root))
        selection = _Selection(start_ms, end_ms, as_of_ms, max_rows)
        items = _trace_items(files)
        try:
            selection.scan(items)
        finally:
            items.close()
        selection.finish()

        def result(path: Optional[str], reused: bool = False) -> WindowCutResult:
            return WindowCutResult(
                path=path,
                num_requests=selection.num_rows,
                trace_block_size=selection.block_size,
                row_counts=dict(selection.counts),
                dropped_members=selection.dropped_members,
                dropped_bytes=selection.dropped_bytes,
                reused=reused,
            )

        if selection.num_rows == 0:
            return result(None)
        if os.path.lexists(out_path):
            # Another process (an earlier leader, an adopted Request's cut)
            # already published this window; DGDR Requests name it by path.
            return result(out_path, reused=True)
        paths = [path for path, _ in files]
        created = _publish(
            out_path, temp_path, lambda sink: _write_rows(selection, paths, sink)
        )
        return result(out_path, reused=not created)
    except OSError as error:
        raise WindowCutError(
            "window_cut_failed", f"{type(error).__name__}: {error}"
        ) from error


def _validate(
    start_ms: float,
    end_ms: float,
    as_of_ms: float,
    out_path: StrPath,
    max_rows: int,
) -> None:
    for name, value in (
        ("start_ms", start_ms),
        ("end_ms", end_ms),
        ("as_of_ms", as_of_ms),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number, got {value!r}")
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if not start_ms < end_ms:
        raise ValueError("start_ms must be before end_ms")
    if as_of_ms < end_ms:
        raise ValueError("as_of_ms must not be before end_ms")
    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 1:
        raise ValueError(f"max_rows must be a positive integer, got {max_rows!r}")
    if not os.fspath(out_path).endswith(".gz"):
        raise ValueError(f"window file must be gzip (.gz): {os.fspath(out_path)!r}")


# ------------------------------------------------------------ child process


def _run_in_child(
    function: Callable[..., Any],
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    *,
    deadline: float,
    cancel: Optional[threading.Event],
    cleanup: Sequence[str] = (),
    name: str = "retune-window-cut",
) -> Any:
    """``function(*args, **kwargs)`` in a ``spawn`` child; its return value.

    ``deadline`` is ``time.monotonic()`` time. The child is killed at the
    deadline, on cancel, or once it has answered; ``cleanup`` paths are
    removed unless it answered with a result (a killed child leaves its
    temporary file behind). A ``WindowCutError`` raised by ``function`` is
    re-raised with its reason; any other failure, the child's death included,
    is ``window_cut_failed``.
    """

    _check_interrupted(deadline, cancel)
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_child_main,
        args=(sender, function, tuple(args), dict(kwargs)),
        name=name,
        daemon=True,
    )
    outcome: Optional[tuple[Any, ...]] = None
    try:
        try:
            process.start()
        except Exception as error:
            raise WindowCutError(
                "window_cut_failed", f"cannot start the window cut process: {error}"
            ) from error
        finally:
            sender.close()
        while outcome is None:
            _check_interrupted(deadline, cancel)
            timeout = max(0.0, min(deadline - time.monotonic(), _POLL_INTERVAL_S))
            if not wait([receiver, process.sentinel], timeout):
                continue
            try:
                outcome = receiver.recv()
            except (EOFError, OSError):
                process.join(_JOIN_TIMEOUT_S)
                raise WindowCutError(
                    "window_cut_failed",
                    f"window cut process exited with code {process.exitcode} "
                    "before returning a result",
                ) from None
    finally:
        _stop(process, grace_s=_EXIT_GRACE_S if outcome is not None else 0.0)
        receiver.close()
        if outcome is None or outcome[0] != "ok":
            for path in cleanup:
                _unlink_quietly(path)
    assert outcome is not None  # the loop only ends with one
    if outcome[0] == "ok":
        return outcome[1]
    _, reason, message = outcome
    raise WindowCutError(reason, message)


def _check_interrupted(deadline: float, cancel: Optional[threading.Event]) -> None:
    if cancel is not None and cancel.is_set():
        raise WindowCutError("cancelled", "window cut cancelled")
    if time.monotonic() >= deadline:
        raise WindowCutError("window_cut_timeout", "window cut exceeded its deadline")


def _stop(process: multiprocessing.process.BaseProcess, *, grace_s: float) -> None:
    """Let ``process`` exit for ``grace_s``, then kill it and reap it.

    A child blocked in uninterruptible I/O (a hard-mounted RWX volume) does
    not die on SIGKILL until the I/O returns; it is left behind rather than
    holding the sweep thread. SIGKILL takes effect before it returns to user
    space, so it cannot publish a window afterwards.
    """

    if process.pid is None:
        return
    if grace_s > 0:
        process.join(grace_s)
    if process.is_alive():
        process.kill()
    process.join(_JOIN_TIMEOUT_S)
    if process.is_alive():
        logger.error(
            "retune window cut process %s (pid %s) did not exit after SIGKILL; "
            "leaving it behind",
            process.name,
            process.pid,
        )


def _child_main(
    conn: Connection,
    function: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    """Child entry point: answer ``("ok", value)`` or ``("error", reason, message)``."""

    try:
        os.nice(_CHILD_NICENESS)
    except OSError:
        pass
    # The writer thread takes the GIL back after every zlib block; at the
    # default 5 ms interval it would wait behind parsing (pass 2 ~30% slower).
    sys.setswitchinterval(_CHILD_SWITCH_INTERVAL_S)
    try:
        outcome: tuple[Any, ...] = ("ok", function(*args, **kwargs))
    except WindowCutError as error:
        outcome = ("error", error.reason, error.message)
    except BaseException as error:  # noqa: BLE001 - every failure is reported
        outcome = ("error", "window_cut_failed", f"{type(error).__name__}: {error}")
    try:
        conn.send(outcome)
    except Exception as error:
        conn.send(("error", "window_cut_failed", f"unsendable result: {error}"))
    finally:
        conn.close()


# ------------------------------------------------------------ gzip members


@dataclass(frozen=True)
class _Member:
    """A gzip member that decoded and matched its CRC32 and ISIZE."""

    offset: int
    end: int
    payload: bytes


@dataclass(frozen=True)
class _Garbage:
    """A maximal run of bytes ``[offset, end)`` that holds no valid member."""

    offset: int
    end: int
    at_eof: bool


class _GzipMemberReader:
    """Iterate a gzip file as validated members and runs of bad bytes.

    Every byte from ``offset`` to the end of the file is in exactly one item,
    in file order. A member is yielded only after zlib reached its end and
    its trailer's CRC32 and ISIZE match the decompressed bytes (zlib checks
    them too; the explicit check does not depend on how zlib was built).
    Anything else - a member cut short by the end of the file, a corrupt
    one, zeros - starts a ``_Garbage`` run that ends at the next gzip header
    that decodes, or at the end of the file. Memory holds one member's
    compressed and decompressed bytes plus a read chunk.
    """

    def __init__(self, source: BinaryIO, offset: int = 0) -> None:
        if offset:
            source.seek(offset)
        self._source = source
        self._buf = bytearray()
        self._pos = 0  # current position in _buf
        self._base = offset  # file offset of _buf[0]
        self._eof = False
        self._pending: Optional[_Member] = None

    def __iter__(self) -> _GzipMemberReader:
        return self

    def __next__(self) -> Union[_Member, _Garbage]:
        if self._pending is not None:
            pending, self._pending = self._pending, None
            return pending
        if self._pos >= len(self._buf) and not self._fill():
            raise StopIteration
        start = self._offset()
        member = self._decode()
        if member is not None:
            return member
        while self._skip_to_magic():
            member = self._decode()
            if member is not None:
                self._pending = member
                return _Garbage(start, member.offset, at_eof=False)
        return _Garbage(start, self._offset(), at_eof=True)

    def _offset(self) -> int:
        return self._base + self._pos

    def _fill(self) -> bool:
        if self._eof:
            return False
        chunk = self._source.read(_READ_CHUNK)
        if not chunk:
            self._eof = True
            return False
        self._buf += chunk
        return True

    def _consume(self, count: int) -> None:
        self._pos += count
        if self._pos >= _COMPACT_AT and 2 * self._pos >= len(self._buf):
            del self._buf[: self._pos]
            self._base += self._pos
            self._pos = 0

    def _decode(self) -> Optional[_Member]:
        """Decode and consume the member at the current position.

        Returns ``None``, consuming nothing, when no valid member starts here.
        """

        decoder = zlib.decompressobj(wbits=_GZIP_WBITS)
        parts: list[bytes] = []
        index = self._pos
        while True:
            if index >= len(self._buf) and not self._fill():
                return None  # cut short by the end of the file
            piece = self._buf[index : index + _DECODE_STEP]
            try:
                parts.append(decoder.decompress(piece))
            except zlib.error:
                return None
            if decoder.eof:
                end = index + len(piece) - len(decoder.unused_data)
                payload = b"".join(parts)
                if not _trailer_matches(self._buf[end - 8 : end], payload):
                    return None
                member = _Member(self._offset(), self._base + end, payload)
                self._consume(end - self._pos)
                return member
            index += len(piece)

    def _skip_to_magic(self) -> bool:
        """Consume at least one byte, then up to the next gzip magic.

        ``False`` (everything consumed) when no magic is left in the file.
        """

        self._consume(1)
        keep = len(_GZIP_MAGIC) - 1  # a magic may straddle two reads
        while True:
            found = self._buf.find(_GZIP_MAGIC, self._pos)
            if found >= 0:
                self._consume(found - self._pos)
                return True
            self._consume(max(0, len(self._buf) - keep - self._pos))
            if not self._fill():
                self._consume(len(self._buf) - self._pos)
                return False


def _trailer_matches(trailer: Union[bytes, bytearray], payload: bytes) -> bool:
    crc, size = struct.unpack("<II", trailer)
    return crc == zlib.crc32(payload) and size == len(payload) & 0xFFFFFFFF


def _decode_exact_member(data: bytes) -> Optional[bytes]:
    """The payload of ``data`` when it is exactly one valid gzip member."""

    decoder = zlib.decompressobj(wbits=_GZIP_WBITS)
    try:
        payload = decoder.decompress(data)
    except zlib.error:
        return None
    if not decoder.eof or decoder.unused_data or len(data) < 8:
        return None
    return payload if _trailer_matches(data[-8:], payload) else None


# ---------------------------------------------------------------- pass 1


def _segment_files(root: str) -> list[tuple[str, bool]]:
    """Trace files under ``root`` in replay-reader order, and which are newest.

    The files and their order are those of ``resolve_trace_files([root])``:
    sorted ``*.jsonl`` and ``*.jsonl.gz`` regular files, and a file present
    both compressed and not is an error. A directory or file under ``root``
    that vanishes during the walk (ENOENT, or ESTALE on NFS: the reconciler
    deletes a dead pod's segments) is skipped; any other error listing or
    stat-ing one (EACCES, EIO, ETIMEDOUT on a soft mount, ...) fails the cut
    as ``window_cut_failed`` rather than silently losing a pod's rows, as
    does any error on ``root`` itself. Each file comes with whether it is the
    newest segment (highest ``NNNNNN``) of its directory, the one its writer
    may still be appending to.
    """

    try:
        root_mode = os.stat(root).st_mode
    except OSError as error:
        if error.errno in _VANISHED:
            raise WindowCutError(
                "window_cut_failed", f"trace segments root {root} does not exist"
            ) from error
        raise _unreadable(root, error) from error
    if stat.S_ISDIR(root_mode):

        def walk_error(error: OSError) -> None:
            if error.filename != root and error.errno in _VANISHED:
                logger.info(
                    "retune window cut: trace directory %s vanished (%s); "
                    "skipping it",
                    error.filename,
                    error,
                )
                return
            raise _unreadable(error.filename, error) from error

        found: list[Path] = []
        for directory, _, names in os.walk(root, onerror=walk_error):
            for name in names:
                if name.endswith(".jsonl") or name.endswith(".jsonl.gz"):
                    candidate = os.path.join(directory, name)
                    if _is_regular_file(candidate):
                        found.append(Path(candidate))
        try:
            paths = _reject_duplicate_copies(sorted(found))
        except ValueError as error:
            raise WindowCutError("window_cut_failed", str(error)) from error
    else:
        paths = [Path(root)]
    newest: dict[Path, tuple[int, str]] = {}
    for path in paths:
        index = segment_index(path.name)
        key = (-1 if index is None else index, path.name)
        if path.parent not in newest or key > newest[path.parent]:
            newest[path.parent] = key
    return [(str(path), newest[path.parent][1] == path.name) for path in paths]


def _is_regular_file(path: str) -> bool:
    """Whether ``path`` is a regular file (following symlinks, like ``is_file``).

    ``False`` when it vanished; any other stat error fails the cut.
    """

    try:
        return stat.S_ISREG(os.stat(path).st_mode)
    except OSError as error:
        if error.errno in _VANISHED:
            return False
        raise _unreadable(path, error) from error


def _unreadable(path: Any, error: OSError) -> WindowCutError:
    return WindowCutError(
        "window_cut_failed",
        f"cannot read trace segments at {path}: {type(error).__name__}: {error}",
    )


@dataclass(frozen=True)
class _PlainBlock:
    """Whole lines ``[offset, end)`` of a plain JSONL file.

    ``dropped`` counts the bytes of an unterminated last line that does not
    parse (still being written); they are not in ``payload``.
    """

    offset: int
    end: int
    payload: bytes
    dropped: int = 0


# (file index, path, newest, item); an OSError item means the file vanished.
_TraceItem = tuple[int, str, bool, Union[_Member, _Garbage, _PlainBlock, OSError]]


def _trace_items(
    files: Sequence[tuple[str, bool]],
) -> Generator[_TraceItem, None, None]:
    """Every member, bad byte run and plain block of ``files``, in file order.

    A file that vanishes (ENOENT, or ESTALE on NFS) before or while it is
    read yields its error and the walk goes on; other I/O errors raise.
    """

    for file_index, (path, newest) in enumerate(files):
        try:
            with open(path, "rb") as source:
                items: Iterator[Union[_Member, _Garbage, _PlainBlock]] = (
                    _GzipMemberReader(source)
                    if path.endswith(".gz")
                    else _plain_blocks(source)
                )
                for item in items:
                    yield file_index, path, newest, item
        except OSError as error:
            if error.errno not in _VANISHED:
                raise
            yield file_index, path, newest, error


def _plain_blocks(source: BinaryIO) -> Iterator[_PlainBlock]:
    """A plain JSONL file as blocks of about ``_PLAIN_BLOCK`` bytes of lines.

    Reading stops at the first block that does not end in a newline: its
    last line was the end of the file when read. A writer may finish that
    line before the next read, which would then start inside it.
    """

    offset = 0
    while True:
        lines = source.readlines(_PLAIN_BLOCK)
        if not lines:
            return
        block = b"".join(lines)
        if block.endswith(b"\n"):
            yield _PlainBlock(offset, offset + len(block), block)
            offset += len(block)
            continue
        # Only the file's last line can be unterminated. One that does not
        # parse is still being written; one that does is a row, as the
        # replay reader reads it.
        payload, dropped = block, 0
        cut = block.rfind(b"\n") + 1
        tail = block[cut:]
        if tail.strip() and not _is_json(tail):
            payload, dropped = block[:cut], len(tail)
        yield _PlainBlock(offset, offset + len(payload), payload, dropped)
        return


class _Selection:
    """Pass 1: the selected rows of every trace file, as columns.

    Row ``i`` is ``arrival[i]``, ``completion[i]``, ``rid[16*i:16*i+16]``
    (BLAKE2b of ``request_id``), line ``line[i]`` of member
    ``members[member[i]]``, ``block[i]`` and ``flags[i]``. Rows are in file
    order, the order the replay reader sees them, so de-duplication keeps the
    same (first) row it does.
    """

    def __init__(
        self, start_ms: float, end_ms: float, as_of_ms: float, max_rows: int
    ) -> None:
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.as_of_ms = as_of_ms
        self.max_rows = max_rows
        self._dedup_at = max_rows + max(
            _DEDUP_SLACK_MIN, int(max_rows * _DEDUP_SLACK_RATIO)
        )
        self.counts = _new_counts()
        # (file index, offset, end, is gzip) of each member with a selected
        # row; a plain JSONL file is cut into pseudo-members of whole lines.
        self.members: list[tuple[int, int, int, bool]] = []
        self.arrival = array("q")
        self.completion = array("q")
        self.rid = bytearray()
        self.member = array("I")
        self.line = array("I")
        self.block = array("q")
        self.flags = array("B")
        self.duplicates = 0
        self.valid_members = 0
        # Runs of bad bytes (``_Garbage``), each counted once however many
        # members it held, and their bytes (plus plain-JSONL partial tails).
        self.dropped_members = 0
        self.dropped_bytes = 0
        # The runs, and their bytes, that count toward MAX_DROPPED_MEMBER_RATIO.
        self.corrupt_members = 0
        self.corrupt_bytes = 0
        # Bytes of every gzip member and run of bad bytes.
        self.gzip_bytes = 0
        self.block_size: Optional[int] = None
        self._drop_examples: list[str] = []

    @property
    def num_rows(self) -> int:
        return len(self.arrival)

    def digest(self, row: int) -> bytes:
        return bytes(self.rid[16 * row : 16 * row + 16])

    def scan(self, items: Iterable[_TraceItem]) -> None:
        """Consume ``_trace_items`` (in file order)."""

        for file_index, path, newest, item in items:
            if isinstance(item, _Member):
                self.valid_members += 1
                self.gzip_bytes += item.end - item.offset
                self._scan_payload(
                    item.payload, file_index, path, item.offset, item.end, True
                )
            elif isinstance(item, _PlainBlock):
                self.dropped_bytes += item.dropped
                if item.payload:
                    self._scan_payload(
                        item.payload, file_index, path, item.offset, item.end, False
                    )
            elif isinstance(item, _Garbage):
                self._dropped(path, item, newest=newest)
            else:
                # The reconciler deletes only segments far older than any
                # window being cut; a row taken from this one fails pass 2.
                logger.warning(
                    "retune window cut: trace file %s vanished (%s); skipping it",
                    path,
                    item,
                )

    def finish(self) -> None:
        """De-duplicate, then apply the corruption, size and block-size rules."""

        self._dedup()
        self.counts["duplicate"] = self.duplicates
        total = self.valid_members + self.dropped_members
        # A run counts as one member: its bytes catch a hole that wiped out
        # many members.
        if (
            self.corrupt_members > MAX_DROPPED_MEMBER_RATIO * total
            or self.corrupt_bytes > MAX_DROPPED_MEMBER_RATIO * self.gzip_bytes
        ):
            raise WindowCutError(
                "trace_corrupt",
                f"{self.corrupt_members} of {total} gzip members of the trace "
                f"failed validation ({self.corrupt_bytes} of {self.gzip_bytes} "
                f"bytes; limit {MAX_DROPPED_MEMBER_RATIO:.0%} of either); "
                f"dropped byte runs: {', '.join(self._drop_examples)}",
            )
        if self.num_rows > self.max_rows:
            raise _too_large(self.max_rows)
        if not self.num_rows:
            return
        blocks = np.unique(np.frombuffer(self.block, dtype=self.block.typecode))
        if len(blocks) > 1:
            raise WindowCutError(
                "window_cut_failed",
                "trace window mixes trace_block_size values "
                f"{blocks.tolist()}; replay needs one block size",
            )
        self.block_size = int(blocks[0])
        flags: np.ndarray = np.frombuffer(self.flags, dtype=self.flags.typecode)
        self.counts["agent_context_stripped"] = int(
            np.count_nonzero(flags & _FLAG_AGENT_CONTEXT)
        )
        self.counts["censor_unavailable"] = int(
            np.count_nonzero(flags & _FLAG_NO_TOTAL_TIME)
        )

    def _dropped(self, path: str, item: _Garbage, *, newest: bool) -> None:
        size = item.end - item.offset
        self.dropped_members += 1
        self.dropped_bytes += size
        self.gzip_bytes += size
        if item.at_eof and newest:
            # The flush its writer is appending, or a crashed writer's last
            # one: at most one batch, written after the cut began.
            logger.debug(
                "retune window cut: %d trailing bytes of %s hold no complete "
                "gzip member (in-flight write)",
                size,
                path,
            )
            return
        self.corrupt_members += 1
        self.corrupt_bytes += size
        if len(self._drop_examples) < _MAX_DROP_EXAMPLES:
            self._drop_examples.append(f"{path}@{item.offset}+{size}")
        logger.warning(
            "retune window cut: dropped %d bytes at %s offset %d that hold no "
            "valid gzip member",
            size,
            path,
            item.offset,
        )

    def _scan_payload(
        self,
        payload: bytes,
        file_index: int,
        path: str,
        offset: int,
        end: int,
        is_gzip: bool,
    ) -> None:
        lines = payload.split(b"\n")
        if not lines[-1]:
            lines.pop()
        member_id = -1
        start_ms, end_ms, as_of_ms = self.start_ms, self.end_ms, self.as_of_ms
        counts = self.counts
        for index, raw in enumerate(lines):
            if not raw.strip():
                continue
            location = f"{path}@{offset}:{index + 1}"
            event = _event(raw, location, "trace_corrupt")
            if event is None:
                continue
            row = _request_row(event, location, "trace_corrupt")
            if row is None:
                counts["missing_output_tokens"] += 1
                continue
            if not start_ms <= row.arrival_ms < end_ms:
                continue
            if row.completion_ms > as_of_ms:
                counts["censored"] += 1
                continue
            if member_id < 0:
                member_id = len(self.members)
                self.members.append((file_index, offset, end, is_gzip))
            self._append(row, member_id, index, location)

    def _append(
        self, row: _RequestRow, member_id: int, index: int, location: str
    ) -> None:
        try:
            self.arrival.append(row.arrival_ms)
            self.completion.append(row.completion_ms)
            self.block.append(row.trace_block_size)
        except OverflowError as error:
            raise WindowCutError(
                "trace_corrupt", f"trace row value out of range at {location}"
            ) from error
        self.rid += _digest(row.request_id)
        self.member.append(member_id)
        self.line.append(index)
        self.flags.append(
            (_FLAG_AGENT_CONTEXT if row.had_agent_context else 0)
            | (_FLAG_NO_TOTAL_TIME if row.total_time_ms is None else 0)
        )
        if len(self.arrival) > self._dedup_at:
            # Duplicates only shrink the window, and rows already kept stay
            # kept: past max_rows after de-duplication, the window is too large.
            self._dedup()
            if len(self.arrival) > self.max_rows:
                raise _too_large(self.max_rows)

    def _dedup(self) -> None:
        count = len(self.arrival)
        if count < 2:
            return
        keep = _first_occurrences(self.rid, count)
        kept = int(np.count_nonzero(keep))
        if kept == count:
            return
        self.duplicates += count - kept
        self.arrival = _compress(self.arrival, keep)
        self.completion = _compress(self.completion, keep)
        self.member = _compress(self.member, keep)
        self.line = _compress(self.line, keep)
        self.block = _compress(self.block, keep)
        self.flags = _compress(self.flags, keep)
        self.rid = bytearray(
            np.frombuffer(self.rid, dtype=np.uint8).reshape(count, 16)[keep].tobytes()
        )


def _first_occurrences(rid: bytearray, count: int) -> np.ndarray:
    """Mask of the rows whose 128-bit digest has not occurred before them."""

    words: np.ndarray = np.frombuffer(rid, dtype=np.uint64).reshape(count, 2)
    high, low = words[:, 0], words[:, 1]
    order = np.lexsort((np.arange(count), low, high))
    high, low = high[order], low[order]
    repeat: np.ndarray = np.zeros(count, dtype=bool)
    repeat[1:] = (high[1:] == high[:-1]) & (low[1:] == low[:-1])
    keep: np.ndarray = np.ones(count, dtype=bool)
    keep[order[repeat]] = False
    return keep


def _compress(column: array, keep: np.ndarray) -> array:
    kept = array(column.typecode)
    kept.frombytes(np.frombuffer(column, dtype=column.typecode)[keep].tobytes())
    return kept


def _too_large(max_rows: int) -> WindowCutError:
    return WindowCutError(
        "window_too_large",
        f"the window has more than {max_rows} requests (max_window_rows)",
    )


def _digest(request_id: str) -> bytes:
    # UTF-8 with surrogatepass is injective on str, lone surrogates included.
    return hashlib.blake2b(
        request_id.encode("utf-8", "surrogatepass"), digest_size=16
    ).digest()


def _is_json(raw: bytes) -> bool:
    try:
        json.loads(raw)
    except ValueError:
        return False
    return True


def _event(
    raw: bytes, location: str, failure: WindowCutFailure
) -> Optional[dict[str, Any]]:
    """The ``request_end`` event of a trace line, as the replay reader reads it."""

    try:
        record = json.loads(raw)
    except ValueError as error:  # JSONDecodeError, UnicodeDecodeError
        raise WindowCutError(failure, f"invalid JSON at {location}: {error}") from error
    if not isinstance(record, dict):
        raise WindowCutError(failure, f"trace row at {location} must be a JSON object")
    try:
        return _request_end_event(record, location)
    except ValueError as error:
        raise WindowCutError(failure, str(error)) from error


def _request_row(
    event: dict[str, Any], location: str, failure: WindowCutFailure
) -> Optional[_RequestRow]:
    try:
        return _parse_request_end(event, location, need_hashes=True)
    except ValueError as error:
        raise WindowCutError(failure, str(error)) from error


# ---------------------------------------------------------------- pass 2


def _publish(out_path: str, temp_path: str, write: Callable[[_ByteSink], None]) -> bool:
    """Write ``out_path`` atomically through ``temp_path``.

    Returns ``False`` when ``out_path`` already existed: it is kept and the
    new file is discarded. ``temp_path`` is removed in every case.
    """

    directory = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(directory, exist_ok=True)
    try:
        descriptor = os.open(
            temp_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        with open(descriptor, "wb") as raw:
            # No name and no mtime in the header: equal cuts are equal bytes.
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw,
                compresslevel=OUTPUT_COMPRESSLEVEL,
                mtime=0,
            ) as sink:
                write(sink)
            raw.flush()
            os.fsync(raw.fileno())
        # Only the owner reads the window. The Run Job reads it as the planner
        # image user (uid 1000), which the Planner also runs as by default.
        os.chmod(temp_path, 0o600)
        try:
            os.link(temp_path, out_path)
            created = True
        except FileExistsError:
            created = False
    finally:
        _unlink_quietly(temp_path)
    _fsync_directory(directory)
    return created


def _fsync_directory(directory: str) -> None:
    try:
        descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError as error:
        logger.debug("retune window cut: cannot open %s to fsync: %s", directory, error)
        return
    try:
        os.fsync(descriptor)
    except OSError as error:
        # Some network filesystems reject fsync on a directory; their LINK
        # is synchronous at the server anyway.
        logger.debug("retune window cut: fsync of %s failed: %s", directory, error)
    finally:
        os.close(descriptor)


def _unlink_quietly(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except OSError as error:
        logger.warning("retune window cut: cannot remove %s: %s", path, error)


def _write_rows(selection: _Selection, paths: Sequence[str], sink: _ByteSink) -> None:
    """Pass 2: write the kept rows to ``sink`` in window-file order.

    The order is ``read_trace_window``'s: (arrival, completion, request_id).
    The columns order rows by the first two; rows with an equal pair are
    collected and ordered by their ``request_id`` before they are written.
    """

    count = selection.num_rows
    arrival: np.ndarray = np.frombuffer(
        selection.arrival, dtype=selection.arrival.typecode
    )
    completion: np.ndarray = np.frombuffer(
        selection.completion, dtype=selection.completion.typecode
    )
    members: np.ndarray = np.frombuffer(
        selection.member, dtype=selection.member.typecode
    )
    lines: np.ndarray = np.frombuffer(selection.line, dtype=selection.line.typecode)
    order = np.lexsort((completion, arrival))
    position: np.ndarray = np.empty(count, dtype=np.int64)
    position[order] = np.arange(count, dtype=np.int64)
    # Each member's rows, in the order they are written.
    by_member = np.lexsort((position, members))
    bounds = np.asarray(
        np.searchsorted(members[by_member], np.arange(len(selection.members) + 1))
    )
    cache = _MemberLines(
        paths,
        selection.members,
        bounds,
        position[by_member],
        lines[by_member],
        capacity=_PASS2_CACHE_BYTES,
    )
    del position, by_member
    group: list[_RequestRow] = []
    group_key: Optional[tuple[int, int]] = None
    writer = _LineWriter(sink)
    try:
        for first in range(0, count, _PASS2_BLOCK_ROWS):
            block = order[first : first + _PASS2_BLOCK_ROWS]
            for row, arrival_ms, completion_ms, member_id, line_index in zip(
                block.tolist(),
                arrival[block].tolist(),
                completion[block].tolist(),
                members[block].tolist(),
                lines[block].tolist(),
            ):
                if (arrival_ms, completion_ms) != group_key:
                    _write_group(group, writer)
                    group = []
                    group_key = (arrival_ms, completion_ms)
                raw = cache.take(member_id, line_index)
                location = cache.location(member_id, line_index)
                event = _event(raw, location, "window_cut_failed")
                parsed = (
                    None
                    if event is None
                    else _request_row(event, location, "window_cut_failed")
                )
                if (
                    parsed is None
                    or parsed.arrival_ms != arrival_ms
                    or parsed.completion_ms != completion_ms
                    or _digest(parsed.request_id) != selection.digest(row)
                ):
                    raise _changed(f"the row at {location} differs")
                group.append(parsed)
        _write_group(group, writer)
        writer.close()
    except BaseException:
        writer.abort()
        raise
    finally:
        cache.close()


def _write_group(group: list[_RequestRow], writer: _LineWriter) -> None:
    if len(group) > 1:
        group.sort(key=attrgetter("request_id"))
    for row in group:
        writer.write(_dumps(_window_record(row)))


def _changed(detail: str) -> WindowCutError:
    return WindowCutError(
        "window_cut_failed", f"trace segment changed between passes: {detail}"
    )


class _LineWriter:
    """Batches window lines and writes them to the gzip sink in a thread.

    zlib and file writes release the GIL, so compressing and writing one
    batch overlaps parsing the next; at most ``_WRITE_QUEUE`` batches wait.
    ``close`` writes what is left and re-raises the thread's error;
    ``abort`` stops it without writing more.
    """

    def __init__(self, sink: _ByteSink) -> None:
        self._sink = sink
        self._parts: list[str] = []
        self._size = 0
        self._queue: queue.Queue[Optional[bytes]] = queue.Queue(maxsize=_WRITE_QUEUE)
        self._error: Optional[BaseException] = None
        self._aborted = False
        self._stopped = False
        self._thread = threading.Thread(
            target=self._run, name="retune-window-write", daemon=True
        )
        self._thread.start()

    def write(self, line: str) -> None:
        self._parts.append(line)
        self._parts.append("\n")
        self._size += len(line) + 1
        if self._size >= _WRITE_BUFFER:
            self._send()

    def close(self) -> None:
        self._send()
        self._stop()
        if self._error is not None:
            raise self._error

    def abort(self) -> None:
        self._aborted = True
        self._stop()

    def _send(self) -> None:
        if self._error is not None:
            raise self._error
        if self._parts:
            self._queue.put("".join(self._parts).encode("utf-8"))
            self._parts = []
            self._size = 0

    def _stop(self) -> None:
        if not self._stopped:
            self._stopped = True
            self._queue.put(None)
            self._thread.join()

    def _run(self) -> None:
        while True:
            data = self._queue.get()
            if data is None:
                return
            if self._error is not None or self._aborted:
                continue
            try:
                self._sink.write(data)
            except BaseException as error:  # noqa: BLE001 - re-raised by close()
                self._error = error


class _MemberLines:
    """The lines pass 2 still has to write, decoded member by member.

    Rows are taken in output order. A member is read and decoded when one of
    its rows is first needed and keeps only its lines still to be written; it
    is dropped when its last one is taken. Above ``capacity`` bytes, the
    members whose next row is written last are evicted (Belady) and decoded
    again when needed, so memory stays bounded whatever the request durations.

    ``bounds[m]:bounds[m+1]`` indexes member ``m``'s rows in ``positions``
    (their output positions, ascending) and ``lines`` (their line indices).
    """

    def __init__(
        self,
        paths: Sequence[str],
        members: Sequence[tuple[int, int, int, bool]],
        bounds: np.ndarray,
        positions: np.ndarray,
        lines: np.ndarray,
        *,
        capacity: int,
    ) -> None:
        self._paths = paths
        self._members = members
        self._cursor: np.ndarray = bounds[:-1].astype(np.int64)
        self._ends = bounds[1:]
        self._positions = positions
        self._lines = lines
        self._capacity = capacity
        self._loaded: dict[int, dict[int, bytes]] = {}
        self._sizes: dict[int, int] = {}
        self._bytes = 0
        self._files: OrderedDict[int, BinaryIO] = OrderedDict()
        self.loads = 0

    def location(self, member_id: int, line_index: int) -> str:
        file_index, offset, _, _ = self._members[member_id]
        return f"{self._paths[file_index]}@{offset}:{line_index + 1}"

    def take(self, member_id: int, line_index: int) -> bytes:
        lines = self._loaded.get(member_id)
        if lines is None:
            lines = self._load(member_id)
        raw = lines.pop(line_index, None)
        if raw is None:
            raise WindowCutError(
                "window_cut_failed",
                f"window cut lost track of the row at "
                f"{self.location(member_id, line_index)}",
            )
        self._cursor[member_id] += 1
        size = len(raw) + _LINE_OVERHEAD
        self._sizes[member_id] -= size
        self._bytes -= size
        if not lines:
            del self._loaded[member_id]
            del self._sizes[member_id]
        elif self._bytes > self._capacity:
            self._evict()
        return raw

    def close(self) -> None:
        for source in self._files.values():
            source.close()
        self._files.clear()

    def _load(self, member_id: int) -> dict[int, bytes]:
        file_index, offset, end, is_gzip = self._members[member_id]
        data = self._read(file_index, offset, end)
        payload = _decode_exact_member(data) if is_gzip else data
        if payload is None:
            raise _changed(
                f"the gzip member at {self._paths[file_index]}@{offset} "
                "no longer decodes"
            )
        pieces = payload.split(b"\n")
        wanted = self._lines[self._cursor[member_id] : self._ends[member_id]].tolist()
        if max(wanted) >= len(pieces):
            raise _changed(
                f"the member at {self._paths[file_index]}@{offset} has fewer lines"
            )
        lines = {index: pieces[index] for index in wanted}
        size = sum(map(len, lines.values())) + _LINE_OVERHEAD * len(lines)
        self._loaded[member_id] = lines
        self._sizes[member_id] = size
        self._bytes += size
        self.loads += 1
        return lines

    def _read(self, file_index: int, offset: int, end: int) -> bytes:
        source = self._files.pop(file_index, None)
        if source is None:
            if len(self._files) >= _OPEN_FILES:
                _, oldest = self._files.popitem(last=False)
                oldest.close()
            source = open(self._paths[file_index], "rb")
        self._files[file_index] = source
        source.seek(offset)
        data = source.read(end - offset)
        if len(data) != end - offset:
            raise _changed(f"{self._paths[file_index]} is shorter than when scanned")
        return data

    def _evict(self) -> None:
        target = self._capacity * 3 // 4
        furthest_first = sorted(
            self._loaded,
            key=lambda member_id: int(self._positions[self._cursor[member_id]]),
            reverse=True,
        )
        for member_id in furthest_first:
            if self._bytes <= target:
                return
            del self._loaded[member_id]
            self._bytes -= self._sizes.pop(member_id)
