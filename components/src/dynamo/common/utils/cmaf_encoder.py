# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CMAF streaming encoder: frames in, fragmented-MP4 segments out.

One long-lived ffmpeg per request, raw RGB in on stdin, fragmented MP4 out on
stdout. One muxer instance sees every frame, so ``mfhd.sequence_number`` and
``tfdt.baseMediaDecodeTime`` are continuous by construction and the only
bitstream work done here is reading box headers to find segment boundaries.
No box is modified. See DEP 0017.

The encoder is the one the batch path (``video_utils.encode_to_video_bytes``)
uses: software VP9 (``libvpx-vp9``) by default, hardware AV1 (``av1_vaapi``)
when ``DYN_XPU_FFMPEG_PATH`` is set; see
:mod:`dynamo.common.utils.ffmpeg_command`. Encoding starts after generation has
finished: the whole clip is available before the first frame is pushed, and
what streams is the encode and the delivery.

The wire vocabulary (tags, chunking, metadata) is in
:mod:`dynamo.common.utils.cmaf_video`.
"""

import asyncio
import contextlib
import logging
import os
import signal
from typing import AsyncIterator, Iterator, List, Optional, Tuple

import numpy as np

from dynamo.common.utils.cmaf_video import (
    CMAF_INIT_TAG,
    DEFAULT_CMAF_GOP_FRAMES,
    segment_tag,
)
from dynamo.common.utils.ffmpeg_command import (
    ENV_XPU_FFMPEG_PATH,
    HW_VIDEO_ENCODER,
    SW_VIDEO_ENCODER,
    build_ffmpeg_command,
    hw_ffmpeg_path,
    validate_raw_rgb_frames,
    xpu_video_device,
)

logger = logging.getLogger(__name__)

# The software ffmpeg binary. imageio, which the batch path uses, returns
# nothing until encoding completes, so it cannot hand back fragments; this path
# drives the CLI directly. The default is the LGPL build the runtime images
# ship. Not used when DYN_XPU_FFMPEG_PATH selects the hardware encoder.
ENV_FFMPEG_PATH = "DYN_FFMPEG_PATH"
DEFAULT_FFMPEG_PATH = "/usr/local/bin/ffmpeg"

# The default (software) encoder.
CMAF_VIDEO_ENCODER = SW_VIDEO_ENCODER

# Used when the init segment cannot be parsed, so the client still gets a codec
# string it can hand to MediaSource.isTypeSupported(). VP9 profile 0, 8-bit for
# the software encoder; AV1 Main profile, level 3.1, 8-bit for the hardware one.
CMAF_FALLBACK_VIDEO_CODEC = "vp09.00.10.08"
CMAF_FALLBACK_AV1_CODEC = "av01.0.05M.08"

# How long to wait on one read from, or one write to, a live ffmpeg before
# giving up on it.
CMAF_IO_TIMEOUT_S = 120.0

# How long teardown waits for a killed ffmpeg to be reaped and to release its
# pipes before it stops waiting.
CMAF_REAP_TIMEOUT_S = 5.0

MP4_BOX_HEADER_SIZE = 8

# Bytes of VisualSampleEntry body that precede the codec configuration box.
_VISUAL_SAMPLE_ENTRY_BODY = 78


def cmaf_ffmpeg_path() -> str:
    """Resolve the ffmpeg binary for the CMAF path (env: ``DYN_FFMPEG_PATH``).

    Raises:
        RuntimeError: If the binary is not an executable file.
    """
    path = (os.environ.get(ENV_FFMPEG_PATH) or "").strip() or DEFAULT_FFMPEG_PATH
    if not os.path.isfile(path) or not os.access(path, os.X_OK):
        raise RuntimeError(
            f"CMAF streaming needs an ffmpeg binary, but {path!r} is not an "
            f"executable file. Point {ENV_FFMPEG_PATH} at an ffmpeg carrying "
            f"{CMAF_VIDEO_ENCODER}, or set {ENV_XPU_FFMPEG_PATH} to encode "
            f"{HW_VIDEO_ENCODER} in hardware."
        )
    return path


def resolve_cmaf_encoder() -> Tuple[str, Optional[str]]:
    """Resolve the ffmpeg binary and the VA-API device for a CMAF session.

    Returns ``(ffmpeg, device)``, where ``device`` is the DRM render node for
    hardware AV1, or ``None`` for software VP9. The switch is the batch path's:
    ``DYN_XPU_FFMPEG_PATH``.

    Raises:
        RuntimeError: If the selected binary is not an executable file. There
            is no fallback to the other encoder.
    """
    hw = hw_ffmpeg_path()
    if hw is not None:
        return hw, xpu_video_device()
    return cmaf_ffmpeg_path(), None


def build_cmaf_ffmpeg_command(
    ffmpeg: str,
    *,
    width: int,
    height: int,
    fps: int,
    gop: int,
    hw_device: Optional[str] = None,
) -> List[str]:
    """Build the ffmpeg argv for one raw-RGB-in, fragmented-MP4-out session.

    Args:
        ffmpeg: Binary to invoke.
        width: Frame width, for the rawvideo input.
        height: Frame height, for the rawvideo input.
        fps: Input frame rate.
        gop: Keyframe interval in frames. Every fragment must start with a
            keyframe, so this is the segment length rather than an encoder
            heuristic.
        hw_device: DRM render node. Its presence selects the hardware encoder.
    """
    return build_ffmpeg_command(
        ffmpeg,
        width=width,
        height=height,
        fps=fps,
        output="pipe:1",
        hw_device=hw_device,
        gop=gop,
        streaming=True,
    )


def validate_cmaf_frames(frames) -> None:
    """Validate the encoder input contract.

    Raises ``ValueError`` unless ``frames`` is an ``np.ndarray`` of shape
    ``(T, H, W, 3)`` with dtype ``uint8``.
    """
    validate_raw_rgb_frames(frames, what="CMAF encoder")


def _discard_outcome(future: asyncio.Future) -> None:
    """Mark a future's exception as seen, so it is not logged as never retrieved.

    For the wait on ffmpeg's stdin closing: a pipe that broke because ffmpeg
    died is reported through the exit status, not through this wait.
    """
    if not future.cancelled():
        future.exception()


# ---------------------------------------------------------------------------
# MP4 box reading
# ---------------------------------------------------------------------------


def _box_header(buf, pos: int, limit: int) -> Optional[Tuple[int, int, bytes]]:
    """Parse the box header at ``pos``.

    Returns ``(header_size, box_size, box_type)``, or ``None`` when fewer than
    the header's bytes are available yet.

    Raises:
        ValueError: If the box declares a size smaller than its own header.
    """
    if limit - pos < MP4_BOX_HEADER_SIZE:
        return None
    size = int.from_bytes(buf[pos : pos + 4], "big")
    btype = bytes(buf[pos + 4 : pos + 8])
    header = MP4_BOX_HEADER_SIZE
    if size == 1:
        if limit - pos < 16:
            return None
        size = int.from_bytes(buf[pos + 8 : pos + 16], "big")
        header = 16
    if size == 0:
        # "Extends to end of stream" -- legal only for the last top-level box,
        # and never completable from a growing buffer.
        return None
    if size < header:
        raise ValueError(f"invalid MP4 box size {size} for {btype!r} at offset {pos}")
    return header, size, btype


def _iter_child_boxes(
    buf: bytes, start: int, end: int
) -> Iterator[Tuple[bytes, int, int]]:
    """Yield ``(box_type, payload_start, box_end)`` for boxes in ``[start, end)``."""
    pos = start
    while pos < end:
        header = _box_header(buf, pos, end)
        if header is None:
            return
        header_size, box_size, btype = header
        box_end = min(pos + box_size, end)
        yield btype, pos + header_size, box_end
        pos += box_size


def _find_child(
    buf: bytes, start: int, end: int, btype: bytes
) -> Optional[Tuple[int, int]]:
    """Return ``(payload_start, box_end)`` of the first ``btype`` box, or ``None``."""
    for found, payload_start, box_end in _iter_child_boxes(buf, start, end):
        if found == btype:
            return payload_start, box_end
    return None


def _av1_codec_string(cfg: bytes) -> Optional[str]:
    """RFC 6381 ``av01.P.LLT.DD`` from an ``av1C`` payload."""
    if len(cfg) < 3:
        return None
    profile = (cfg[1] >> 5) & 0x07
    level = cfg[1] & 0x1F
    tier = "H" if (cfg[2] >> 7) & 0x01 else "M"
    high_bitdepth = (cfg[2] >> 6) & 0x01
    twelve_bit = (cfg[2] >> 5) & 0x01
    depth = 12 if twelve_bit else (10 if high_bitdepth else 8)
    return f"av01.{profile}.{level:02d}{tier}.{depth:02d}"


def _vp9_codec_string(cfg: bytes) -> Optional[str]:
    """RFC 6381 ``vp09.PP.LL.DD`` from a ``vpcC`` payload (a FullBox)."""
    if len(cfg) < 7:
        return None
    profile, level = cfg[4], cfg[5]
    depth = cfg[6] >> 4
    return f"vp09.{profile:02d}.{level:02d}.{depth:02d}"


def codec_string_from_init(init: bytes) -> Optional[str]:
    """Read the RFC 6381 codec string out of a CMAF init segment.

    It cannot be a constant: it names the codec the deployment resolved, the
    level depends on the resolution and frame rate of the encode, and the
    client passes the string to ``MediaSource.isTypeSupported()`` before
    appending anything.

    Returns ``None`` if the sample entry cannot be located.
    """
    try:
        scope = _find_child(init, 0, len(init), b"moov")
        for name in (b"trak", b"mdia", b"minf", b"stbl", b"stsd"):
            if scope is None:
                return None
            scope = _find_child(init, scope[0], scope[1], name)
        if scope is None:
            return None

        # stsd is a FullBox: 4 bytes of version/flags, then a 4-byte entry count.
        entries_start = scope[0] + 8
        entry = _box_header(init, entries_start, scope[1])
        if entry is None:
            return None
        header_size, box_size, _ = entry
        body = entries_start + header_size + _VISUAL_SAMPLE_ENTRY_BODY
        entry_end = entries_start + box_size

        for btype, payload_start, box_end in _iter_child_boxes(init, body, entry_end):
            if btype == b"vpcC":
                return _vp9_codec_string(init[payload_start:box_end])
            if btype == b"av1C":
                return _av1_codec_string(init[payload_start:box_end])
    except (ValueError, IndexError) as err:
        logger.warning("Could not parse CMAF init segment: %s", err)
    return None


class FragmentedMp4Cutter:
    """Cut a growing fragmented-MP4 byte stream into CMAF segments.

    Feed bytes as they arrive; each call returns the segments that just became
    complete, as ``(kind, payload)`` with ``kind`` in ``{"init", "segment"}``.
    The init segment is ``ftyp`` through the end of ``moov``; each later segment
    is one ``moof`` plus its ``mdat``. A boundary landing mid-read costs only a
    wait for the rest of the box.

    Byte ranges are emitted verbatim -- no box is modified, because the muxer
    owns the timeline.
    """

    def __init__(self) -> None:
        """Start empty, ahead of the init segment."""
        self._buf = bytearray()
        self._pos = 0  # offset of the next unparsed box header
        self._init_done = False
        self._in_fragment = False
        # Set by flush(): bytes of a box or fragment the stream ended inside.
        self.incomplete_bytes = 0

    def feed(self, data: bytes) -> List[Tuple[str, bytes]]:
        """Add bytes and return whatever segments they completed."""
        self._buf += data
        return self._cut()

    def flush(self) -> List[Tuple[str, bytes]]:
        """End the stream and return the one segment only its end can close.

        That is a final ``mdat`` whose size field is 0, "to end of stream".
        Anything else still buffered is a box or a fragment the stream was cut
        off inside. It is dropped rather than emitted -- a client cannot use a
        partial fragment -- and counted in :attr:`incomplete_bytes`, so the
        caller can report an output that ended early.
        """
        out = self._cut()
        if self._buf:
            if self._in_fragment and self._at_open_ended_mdat():
                out.append(("segment", bytes(self._buf)))
            else:
                self.incomplete_bytes = len(self._buf)
            self._reset_fragment()
        return out

    def _at_open_ended_mdat(self) -> bool:
        """Whether the box being waited for is an ``mdat`` of size 0."""
        header = bytes(self._buf[self._pos : self._pos + MP4_BOX_HEADER_SIZE])
        return header == b"\x00\x00\x00\x00mdat"

    def _reset_fragment(self) -> None:
        """Drop the buffered bytes and leave fragment state."""
        self._buf.clear()
        self._pos = 0
        self._in_fragment = False

    def _take(self, end: int, kind: str) -> Tuple[str, bytes]:
        """Cut the first ``end`` buffered bytes out as one segment."""
        payload = bytes(self._buf[:end])
        del self._buf[:end]
        self._pos = 0
        return kind, payload

    def _cut(self) -> List[Tuple[str, bytes]]:
        """Walk the complete top-level boxes and return the segments they close."""
        out: List[Tuple[str, bytes]] = []
        while True:
            header = _box_header(self._buf, self._pos, len(self._buf))
            if header is None:
                break
            _, box_size, btype = header
            end = self._pos + box_size
            if len(self._buf) < end:
                break  # box still arriving

            if not self._init_done:
                self._pos = end
                if btype == b"moov":
                    out.append(self._take(end, "init"))
                    self._init_done = True
                continue

            if btype == b"moof":
                if self._in_fragment:
                    # The previous fragment never got its mdat. Emitting its
                    # moof alone would hand the client a fragment with no
                    # media in it, which the next fragment cannot repair.
                    raise ValueError(
                        "invalid CMAF stream: a moof follows a moof with no mdat "
                        "in between"
                    )
                self._pos = end
                self._in_fragment = True
                continue

            if not self._in_fragment:
                # Not part of any fragment (the mfra the muxer writes after the
                # last one, say): the SourceBuffer wants fragments only.
                logger.debug("Skipping top-level %r box outside a fragment", btype)
                del self._buf[:end]
                self._pos = 0
                continue

            if btype == b"mdat":
                out.append(self._take(end, "segment"))
                self._in_fragment = False
                continue

            self._pos = end  # some other box inside the fragment
        return out


class StreamingCmafEncoder:
    """One long-lived ffmpeg session producing a CMAF stream for one request.

    Frames go in a chunk at a time; completed fragments come out as
    ``(tag, payload)`` pairs using the tags in :mod:`dynamo.common.utils.cmaf_video`.

    Three pipes are pumped concurrently: writing stdin while ffmpeg blocks on a
    full stdout pipe would deadlock, so stdout and stderr are drained by
    background tasks for the life of the process.

    A fragmented muxer cannot finalize fragment *k* until it sees the first frame
    of *k+1* or end of input, so the last fragment only appears once
    :meth:`finish` closes stdin.

    The encoder is resolved in :meth:`start`: hardware AV1 when
    ``DYN_XPU_FFMPEG_PATH`` is set, software VP9 otherwise. There is no
    fallback: if ffmpeg fails, the session raises.

    ffmpeg runs in its own process group and teardown kills the group, so a
    ``DYN_FFMPEG_PATH`` that is a wrapper script does not leave the real
    encoder behind, even if the wrapper exits first. No wait on the process is
    unbounded.

    Usage::

        enc = StreamingCmafEncoder(fps, width, height)
        await enc.start()
        try:
            for chunk in iter_cmaf_chunks(frames, enc.gop_frames):
                async for tag, payload in enc.push(chunk):
                    ...
            async for tag, payload in enc.finish():
                ...
        finally:
            await enc.aclose()
    """

    def __init__(
        self,
        fps: int,
        width: int,
        height: int,
        *,
        gop_frames: Optional[int] = None,
        io_timeout_s: float = CMAF_IO_TIMEOUT_S,
    ) -> None:
        """Record the session geometry; ``start()`` launches ffmpeg."""
        self.fps = int(fps)
        self.width = int(width)
        self.height = int(height)
        self.gop_frames = int(
            gop_frames if gop_frames is not None else DEFAULT_CMAF_GOP_FRAMES
        )
        self._io_timeout_s = io_timeout_s
        # Set by start(): the DRM render node when encoding in hardware.
        self._hw_device: Optional[str] = None
        self._encoder = CMAF_VIDEO_ENCODER

        self._proc: Optional[asyncio.subprocess.Process] = None
        self._cutter = FragmentedMp4Cutter()
        self._out_q: asyncio.Queue = asyncio.Queue()
        self._stdout_task: Optional[asyncio.Task] = None
        self._stderr_task: Optional[asyncio.Task] = None
        self._stderr = bytearray()
        self._stdout_eof = False
        # Set when the stdout reader stopped on an error instead of at end of
        # stream; the queue's end marker alone cannot tell the two apart.
        self._stdout_error: Optional[BaseException] = None
        self._stdin_closed = False
        self._segments = 0
        self._frames_in = 0
        self._init_segment: Optional[bytes] = None

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        """Spawn ffmpeg and start draining its output pipes."""
        if self._proc is not None:
            raise RuntimeError("StreamingCmafEncoder.start() called twice")

        ffmpeg, self._hw_device = resolve_cmaf_encoder()
        self._encoder = HW_VIDEO_ENCODER if self._hw_device else CMAF_VIDEO_ENCODER
        cmd = build_cmaf_ffmpeg_command(
            ffmpeg,
            width=self.width,
            height=self.height,
            fps=self.fps,
            gop=self.gop_frames,
            hw_device=self._hw_device,
        )
        logger.info("CMAF encode session: %s", " ".join(cmd))
        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            # Its own process group, so teardown can kill ffmpeg together with
            # anything a wrapper script started.
            start_new_session=True,
        )
        self._stdout_task = asyncio.create_task(self._pump_stdout())
        self._stderr_task = asyncio.create_task(self._pump_stderr())

    async def aclose(self) -> None:
        """Tear the session down without waiting for a clean end of stream.

        Safe to call on a client disconnect or from an exception path, and safe
        to call twice. Always reaps the process.
        """
        if self._proc is None:
            return
        await self._reap()
        for task in (self._stdout_task, self._stderr_task):
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
        self._stdout_task = self._stderr_task = None

    async def __aenter__(self) -> "StreamingCmafEncoder":
        """Start the session."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        """Tear the session down."""
        await self.aclose()

    # -- streaming ---------------------------------------------------------

    async def push(self, frames: np.ndarray) -> AsyncIterator[Tuple[str, bytes]]:
        """Encode one chunk of frames, yielding the fragments already finished.

        Yields nothing for a chunk whose fragment is not finished yet, which is
        normal: the muxer needs the next chunk's keyframe first.

        Raises:
            ValueError: If ``frames`` is not ``(T, H, W, 3) uint8``, or its
                geometry does not match the session's.
            RuntimeError: If ffmpeg has exited or stops accepting input.
        """
        proc = self._require_started()
        validate_cmaf_frames(frames)
        _, height, width, _ = frames.shape
        if (width, height) != (self.width, self.height):
            raise ValueError(
                f"chunk geometry {width}x{height} does not match the CMAF session's "
                f"{self.width}x{self.height}"
            )
        if proc.returncode is not None:
            raise await self._failure()

        assert proc.stdin is not None
        try:
            proc.stdin.write(np.ascontiguousarray(frames).tobytes())
            await asyncio.wait_for(proc.stdin.drain(), timeout=self._io_timeout_s)
        except (BrokenPipeError, ConnectionResetError):
            raise await self._failure() from None
        except asyncio.TimeoutError:
            await self.aclose()
            raise RuntimeError(
                f"ffmpeg {self._encoder} accepted no input for "
                f"{self._io_timeout_s:.0f}s while encoding the CMAF stream"
            ) from None
        self._frames_in += len(frames)

        for item in self._drain_ready():
            yield item
        if self._stdout_eof:
            # ffmpeg closed stdout while we still have frames to give it.
            raise await self._failure()

    async def finish(self) -> AsyncIterator[Tuple[str, bytes]]:
        """Close the input and yield the fragments that remain, as they complete.

        Raises:
            RuntimeError: If ffmpeg exits non-zero, or exits zero without having
                written the whole stream. Its stderr is included.
        """
        proc = self._require_started()
        await self._close_stdin()

        while not self._stdout_eof:
            try:
                item = await asyncio.wait_for(
                    self._out_q.get(), timeout=self._io_timeout_s
                )
            except asyncio.TimeoutError:
                await self.aclose()
                raise RuntimeError(
                    f"ffmpeg {self._encoder} produced no output for "
                    f"{self._io_timeout_s:.0f}s while draining the CMAF stream"
                ) from None
            if item is None:
                self._stdout_eof = True
                break
            for kind, payload in self._cutter.feed(item):
                yield self._tag(kind, payload)

        if self._stdout_error is not None:
            # Not end of stream: what ffmpeg wrote after the read error is lost.
            raise await self._failure()

        for kind, payload in self._cutter.flush():
            yield self._tag(kind, payload)

        try:
            returncode = await asyncio.wait_for(proc.wait(), timeout=self._io_timeout_s)
        except asyncio.TimeoutError:
            await self.aclose()
            raise RuntimeError(
                f"ffmpeg {self._encoder} did not exit within "
                f"{self._io_timeout_s:.0f}s of ending the CMAF stream"
            ) from None
        await self._collect_stderr()
        if returncode != 0:
            raise RuntimeError(
                f"ffmpeg {self._encoder} CMAF encode failed (exit {returncode}): "
                f"{self._stderr_text()}"
            )
        missing = self._missing_output()
        if missing:
            # A zero exit status is not enough: the client would be told the
            # stream is complete.
            raise RuntimeError(
                f"ffmpeg {self._encoder} CMAF encode exited 0 but {missing}: "
                f"{self._stderr_text()}"
            )
        logger.info(
            "CMAF session done: %d frame(s) in, %d segment(s) out",
            self._frames_in,
            self._segments,
        )

    # -- introspection -----------------------------------------------------

    @property
    def init_segment(self) -> Optional[bytes]:
        """The init segment, once it has been drained."""
        return self._init_segment

    def codec_string(self) -> str:
        """RFC 6381 codec string for this stream.

        Read from the init segment, so it is only exact once that has been
        drained; before that, and if the parse fails, it falls back to the
        encoder's typical configuration.
        """
        if self._init_segment is not None:
            parsed = codec_string_from_init(self._init_segment)
            if parsed:
                return parsed
        if self._hw_device:
            return CMAF_FALLBACK_AV1_CODEC
        return CMAF_FALLBACK_VIDEO_CODEC

    # -- internals ---------------------------------------------------------

    def _require_started(self) -> asyncio.subprocess.Process:
        """The ffmpeg process, or a RuntimeError if ``start()`` has not run."""
        if self._proc is None:
            raise RuntimeError("StreamingCmafEncoder.start() has not been called")
        return self._proc

    def _tag(self, kind: str, payload: bytes) -> Tuple[str, bytes]:
        """Turn a cutter ``(kind, payload)`` into its wire tag, numbering segments."""
        if kind == "init":
            self._init_segment = payload
            return CMAF_INIT_TAG, payload
        self._segments += 1
        return segment_tag(self._segments), payload

    def _missing_output(self) -> Optional[str]:
        """What a finished stream lacks, or ``None`` if it is whole."""
        if self._cutter.incomplete_bytes:
            return (
                f"its output ends inside a box, {self._cutter.incomplete_bytes} "
                "byte(s) short of a complete fragment"
            )
        if not self._frames_in:
            return None
        if self._init_segment is None:
            return "wrote no init segment"
        if not self._segments:
            return f"wrote no media segment for {self._frames_in} frame(s)"
        return None

    def _drain_ready(self) -> List[Tuple[str, bytes]]:
        """Cut whatever stdout bytes have already arrived, without waiting."""
        data = bytearray()
        while True:
            try:
                item = self._out_q.get_nowait()
            except asyncio.QueueEmpty:
                break
            if item is None:
                self._stdout_eof = True
                break
            data += item
        if not data:
            return []
        return [self._tag(kind, payload) for kind, payload in self._cutter.feed(data)]

    async def _pump_stdout(self) -> None:
        """Move stdout into a queue so a full pipe can never block our writes."""
        proc = self._proc
        assert proc is not None and proc.stdout is not None
        try:
            while True:
                data = await proc.stdout.read(65536)
                if not data:
                    break
                await self._out_q.put(data)
        except asyncio.CancelledError:
            raise
        except Exception as err:  # noqa: BLE001 - raised by push() and finish()
            self._stdout_error = err
            logger.warning("CMAF stdout reader failed: %s", err)
        finally:
            self._out_q.put_nowait(None)

    async def _pump_stderr(self) -> None:
        """Keep the tail of ffmpeg's stderr for error reports."""
        proc = self._proc
        assert proc is not None and proc.stderr is not None
        with contextlib.suppress(asyncio.CancelledError, Exception):
            while True:
                data = await proc.stderr.read(4096)
                if not data:
                    break
                # Keep the tail: ffmpeg's useful diagnostics are the last lines.
                self._stderr += data
                if len(self._stderr) > 65536:
                    del self._stderr[:-65536]

    def _stderr_text(self) -> str:
        """The captured stderr tail as text."""
        return bytes(self._stderr).decode("utf-8", errors="replace").strip()

    async def _close_stdin(self) -> None:
        """Close ffmpeg's stdin once, signalling end of input."""
        proc = self._require_started()
        if self._stdin_closed or proc.stdin is None:
            return
        self._stdin_closed = True
        proc.stdin.close()
        # Closing flushes what is still buffered, so this waits on ffmpeg. The
        # wait is bounded with asyncio.wait(), which leaves the waiter running
        # on timeout: wait_for() would cancel the future asyncio completes when
        # the pipe closes, and the event loop then fails completing it.
        closed = asyncio.ensure_future(proc.stdin.wait_closed())
        closed.add_done_callback(_discard_outcome)
        done, _ = await asyncio.wait({closed}, timeout=self._io_timeout_s)
        if not done:
            await self.aclose()
            raise RuntimeError(
                f"ffmpeg {self._encoder} accepted no input for "
                f"{self._io_timeout_s:.0f}s while ending the CMAF stream"
            )

    async def _reap(self) -> None:
        """Kill what is left of the session and wait for it, never indefinitely."""
        proc = self._require_started()
        # Nothing will read what is still queued for stdin; drop it so the pipe
        # closes now instead of when the buffer drains.
        self._stdin_closed = True
        stdin = proc.stdin
        if stdin is not None and not stdin.transport.is_closing():
            with contextlib.suppress(Exception):
                stdin.transport.abort()
        self._kill_group(proc)
        try:
            # asyncio's Process.wait() returns once the process has exited *and*
            # its pipes have closed, so a process that escaped the kill and
            # still holds one would block it forever.
            await asyncio.wait_for(proc.wait(), timeout=CMAF_REAP_TIMEOUT_S)
        except asyncio.TimeoutError:
            logger.warning(
                "CMAF encode session: ffmpeg (pid %d) did not release its pipes "
                "within %.0fs of being killed",
                proc.pid,
                CMAF_REAP_TIMEOUT_S,
            )
        except Exception:  # noqa: BLE001 - teardown must not raise
            pass

    @staticmethod
    def _kill_group(proc: asyncio.subprocess.Process) -> None:
        """SIGKILL whatever is left of the session's process group.

        The group, not just the pid: a wrapper script's child would otherwise
        outlive it, still encoding and still holding the pipes. That includes
        a wrapper that has already exited and left its child behind.
        """
        if proc.returncode is not None:
            # The leader has been reaped. The kernel keeps its pid reserved
            # while the group still has a member, so a group by that id is
            # still this session's -- unless the group emptied and the number
            # was handed to a new process, which is not ours to signal.
            try:
                os.kill(proc.pid, 0)
            except ProcessLookupError:
                pass  # no process holds the number: the group id is still ours
            except OSError:
                return  # a process we may not signal holds it
            else:
                return  # a new process holds it
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGKILL)

    async def _collect_stderr(self) -> None:
        """Let the stderr reader finish, so an error report carries ffmpeg's last lines."""
        task = self._stderr_task
        if task is None or task.done():
            return
        # wait_for cancels the reader if a surviving process still holds stderr.
        with contextlib.suppress(Exception):
            await asyncio.wait_for(task, timeout=CMAF_REAP_TIMEOUT_S)

    async def _failure(self) -> RuntimeError:
        """Reap the process and build the error describing why it died."""
        proc = self._require_started()
        await self._reap()
        await self._collect_stderr()
        status = f"exit {proc.returncode}"
        if self._stdout_error is not None:
            status += f", reading its output failed: {self._stdout_error!r}"
        return RuntimeError(
            f"ffmpeg {self._encoder} CMAF encode failed ({status}) "
            f"after {self._frames_in} frame(s) and {self._segments} segment(s): "
            f"{self._stderr_text()}"
        )
