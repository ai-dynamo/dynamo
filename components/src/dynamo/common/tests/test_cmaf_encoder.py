# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for dynamo.common.utils.cmaf_encoder.

Two groups. The first never starts a process: the streaming path only *reads*
MP4 boxes, so segment framing, codec detection and command construction are
testable on synthetic bytes. The second drives a real ffmpeg and checks what a
player depends on -- decodable fragments, one continuous timeline, a partial
final segment, and fragments arriving while input is still being written. It
skips when no ffmpeg with libvpx-vp9 is available.

Every test runs with software encoding unless it selects hardware itself
(``conftest.py``). The hardware group at the end drives a real VA-API encode
and runs only where ``DYN_XPU_FFMPEG_PATH`` is set.
"""

import asyncio
import os
import shutil
import subprocess
from typing import Optional

import numpy as np
import pytest

from dynamo.common.utils import cmaf_encoder
from dynamo.common.utils.cmaf_encoder import StreamingCmafEncoder
from dynamo.common.utils.cmaf_video import (
    CMAF_INIT_TAG,
    CMAF_SEGMENT_TAG_PREFIX,
    iter_cmaf_chunks,
)
from dynamo.common.utils.ffmpeg_command import (
    ENV_XPU_FFMPEG_PATH,
    ENV_XPU_VIDEO_DEVICE,
    FRAGMENTED_MOVFLAGS,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


# ---------------------------------------------------------------------------
# Synthetic MP4 helpers
# ---------------------------------------------------------------------------


def _box(btype: bytes, payload: bytes = b"") -> bytes:
    """Serialize one MP4 box."""
    size = len(payload) + cmaf_encoder.MP4_BOX_HEADER_SIZE
    return size.to_bytes(4, "big") + btype + payload


# vpcC is a FullBox: version/flags, then profile 0, level 10, 8-bit in the
# high nibble.
_VPCC = bytes([1, 0, 0, 0, 0, 10, 0x80])

# av1C: marker/version, then profile 0 and level index 5 (3.1), then Main
# tier, 8-bit, 4:2:0.
_AV1C = bytes([0x81, 0x05, 0x0C, 0x00])


def _init_segment(sample_entry: bytes, config_type: bytes, config: bytes) -> bytes:
    """Minimal ftyp+moov nesting one codec configuration box where ffmpeg puts it."""
    entry = _box(
        sample_entry,
        b"\x00" * cmaf_encoder._VISUAL_SAMPLE_ENTRY_BODY + _box(config_type, config),
    )
    stsd = _box(b"stsd", b"\x00" * 8 + entry)  # version/flags + entry count
    moov = _box(
        b"moov", _box(b"trak", _box(b"mdia", _box(b"minf", _box(b"stbl", stsd))))
    )
    return _box(b"ftyp", b"isom") + moov


def _fragments(count: int) -> bytes:
    """`count` media fragments, each a moof followed by its mdat."""
    return b"".join(
        _box(b"moof", b"moof-%d" % i) + _box(b"mdat", bytes([i]) * 16)
        for i in range(count)
    )


def _executable_stub(tmp_path, body: str = "exit 0") -> str:
    """Write an executable shell script standing in for ffmpeg."""
    path = tmp_path / "ffmpeg"
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(0o755)
    return str(path)


def _replaying_stub(tmp_path, output: bytes, status: int = 0) -> str:
    """A stand-in ffmpeg that reads all its input, writes ``output``, and exits."""
    recorded = tmp_path / "output.bin"
    recorded.write_bytes(output)
    return _executable_stub(
        tmp_path, f'cat > /dev/null\ncat "{recorded}"\nexit {status}'
    )


# ---------------------------------------------------------------------------
# No process started
# ---------------------------------------------------------------------------


class TestCodecStringFromInit:
    """The codec string is read from the encode that actually ran."""

    def test_reads_vp9_profile_level_and_depth(self):
        """The VP9 codec string comes from ``vpcC``."""
        init = _init_segment(b"vp09", b"vpcC", _VPCC)
        assert cmaf_encoder.codec_string_from_init(init) == "vp09.00.10.08"

    def test_reads_av1_profile_level_tier_and_depth(self):
        """The AV1 codec string comes from ``av1C``."""
        init = _init_segment(b"av01", b"av1C", _AV1C)
        assert cmaf_encoder.codec_string_from_init(init) == "av01.0.05M.08"

    def test_reads_av1_high_tier_and_ten_bit(self):
        """Tier and bit depth are read, not assumed."""
        init = _init_segment(b"av01", b"av1C", bytes([0x81, 0x0D, 0xCC, 0x00]))
        assert cmaf_encoder.codec_string_from_init(init) == "av01.0.13H.10"

    def test_returns_none_for_a_codec_this_encoder_does_not_produce(self):
        """An unknown configuration box gives ``None`` rather than a guess."""
        init = _init_segment(b"avc1", b"avcC", bytes([1, 100, 0, 31]))
        assert cmaf_encoder.codec_string_from_init(init) is None

    def test_returns_none_when_the_sample_entry_is_missing(self):
        """No sample entry gives ``None`` rather than a guess."""
        assert cmaf_encoder.codec_string_from_init(_box(b"ftyp", b"isom")) is None

    def test_truncated_init_does_not_raise(self):
        """A truncated init segment gives ``None``, not an exception."""
        init = _init_segment(b"vp09", b"vpcC", _VPCC)
        assert cmaf_encoder.codec_string_from_init(init[:40]) is None

    def test_encoder_fallback_covers_an_unreadable_init(self):
        """The client cannot call isTypeSupported() without some codec string."""
        enc = StreamingCmafEncoder(16, 64, 64)
        assert enc.codec_string() == cmaf_encoder.CMAF_FALLBACK_VIDEO_CODEC


class TestFragmentedMp4Cutter:
    """ftyp+moov is the init segment; every later moof+mdat is one segment."""

    def test_splits_init_and_segments_verbatim(self):
        """Init and each fragment come out as the exact bytes that went in."""
        init, body = _init_segment(b"vp09", b"vpcC", _VPCC), _fragments(3)
        out = cmaf_encoder.FragmentedMp4Cutter().feed(init + body)

        assert [kind for kind, _ in out] == ["init", "segment", "segment", "segment"]
        # Byte-exact: the muxer owns the timeline, nothing is rewritten.
        assert out[0][1] == init
        assert b"".join(payload for _, payload in out[1:]) == body

    def test_a_boundary_mid_read_costs_a_wait_not_a_segment(self):
        """A box split across reads completes on the next feed, not cut short."""
        init, body = _init_segment(b"vp09", b"vpcC", _VPCC), _fragments(2)
        cutter = cmaf_encoder.FragmentedMp4Cutter()
        out = []
        for byte in init + body:
            out += cutter.feed(bytes([byte]))

        assert [kind for kind, _ in out] == ["init", "segment", "segment"]
        assert out[0][1] == init
        assert b"".join(payload for _, payload in out[1:]) == body

    def test_skips_a_top_level_box_outside_a_fragment(self):
        """The mfra ffmpeg writes after the last fragment is not playable data."""
        init, body = _init_segment(b"vp09", b"vpcC", _VPCC), _fragments(1)
        out = cmaf_encoder.FragmentedMp4Cutter().feed(init + body + _box(b"mfra", b"x"))

        assert [kind for kind, _ in out] == ["init", "segment"]

    def test_flush_closes_an_mdat_that_runs_to_end_of_stream(self):
        """A final mdat of size 0 has no end until the stream does."""
        init = _init_segment(b"vp09", b"vpcC", _VPCC)
        tail = _box(b"moof", b"moof-0") + b"\x00\x00\x00\x00mdat" + b"\x07" * 16
        cutter = cmaf_encoder.FragmentedMp4Cutter()

        assert [kind for kind, _ in cutter.feed(init + tail)] == ["init"]
        assert cutter.flush() == [("segment", tail)]
        assert cutter.incomplete_bytes == 0

    @pytest.mark.parametrize(
        "lost",
        [4, 24, 34],
        ids=["mdat cut short", "moof with no mdat", "moof cut short"],
    )
    def test_flush_drops_a_fragment_the_stream_ended_inside(self, lost):
        """A truncated fragment is counted, never emitted as a segment."""
        init, body = _init_segment(b"vp09", b"vpcC", _VPCC), _fragments(2)
        fragment = len(body) // 2
        cutter = cmaf_encoder.FragmentedMp4Cutter()

        whole = cutter.feed(init + body[:-lost])
        assert [kind for kind, _ in whole] == ["init", "segment"]

        assert cutter.flush() == []
        assert cutter.incomplete_bytes == fragment - lost

    def test_rejects_a_moof_that_follows_a_moof(self):
        """A fragment with no mdat is a framing error, not a segment."""
        init, moof = _init_segment(b"vp09", b"vpcC", _VPCC), _box(b"moof", b"moof-0")
        with pytest.raises(ValueError, match="no mdat"):
            cmaf_encoder.FragmentedMp4Cutter().feed(init + moof + moof)

    def test_rejects_an_impossible_box_size(self):
        """A box size smaller than its header is a ValueError."""
        bad = (4).to_bytes(4, "big") + b"moov" + b"\x00" * 8
        with pytest.raises(ValueError, match="invalid MP4 box size"):
            cmaf_encoder.FragmentedMp4Cutter().feed(bad)


class TestCmafFfmpegPath:
    """Streaming needs a real binary: imageio cannot hand back fragments."""

    def test_the_env_var_names_the_binary(self, monkeypatch, tmp_path):
        """``DYN_FFMPEG_PATH`` is used when it names an executable."""
        exe = _executable_stub(tmp_path)
        monkeypatch.setenv(cmaf_encoder.ENV_FFMPEG_PATH, exe)
        assert cmaf_encoder.cmaf_ffmpeg_path() == exe

    def test_a_missing_binary_is_a_hard_error(self, monkeypatch, tmp_path):
        """A path that is not an executable file raises, naming the variable."""
        monkeypatch.setenv(cmaf_encoder.ENV_FFMPEG_PATH, str(tmp_path / "nope"))
        with pytest.raises(RuntimeError, match=cmaf_encoder.ENV_FFMPEG_PATH):
            cmaf_encoder.cmaf_ffmpeg_path()

    def test_a_non_executable_file_is_a_hard_error(self, monkeypatch, tmp_path):
        """A regular file without the execute bit is refused."""
        path = tmp_path / "ffmpeg"
        path.write_text("not a program")
        path.chmod(0o644)
        monkeypatch.setenv(cmaf_encoder.ENV_FFMPEG_PATH, str(path))
        with pytest.raises(RuntimeError, match="not an executable file"):
            cmaf_encoder.cmaf_ffmpeg_path()

    async def test_start_fails_before_spawning_anything(self, monkeypatch, tmp_path):
        """A session with no usable binary fails at start(), not mid-stream."""
        monkeypatch.setenv(cmaf_encoder.ENV_FFMPEG_PATH, str(tmp_path / "nope"))
        enc = StreamingCmafEncoder(16, 64, 64)
        with pytest.raises(RuntimeError, match="not an executable file"):
            await enc.start()
        await enc.aclose()  # nothing to reap; must not raise


class TestCmafEncoderSelection:
    """The session uses the batch path's switch: ``DYN_XPU_FFMPEG_PATH``."""

    def test_software_by_default(self, monkeypatch, tmp_path):
        exe = _executable_stub(tmp_path)
        monkeypatch.setenv(cmaf_encoder.ENV_FFMPEG_PATH, exe)
        assert cmaf_encoder.resolve_cmaf_encoder() == (exe, None)

    def test_hardware_when_declared_and_the_software_binary_is_not_needed(
        self, monkeypatch, tmp_path
    ):
        exe = _executable_stub(tmp_path)
        monkeypatch.setenv(ENV_XPU_FFMPEG_PATH, exe)
        monkeypatch.setenv(ENV_XPU_VIDEO_DEVICE, "/dev/dri/renderD129")
        monkeypatch.setenv(cmaf_encoder.ENV_FFMPEG_PATH, str(tmp_path / "nope"))
        assert cmaf_encoder.resolve_cmaf_encoder() == (exe, "/dev/dri/renderD129")

    def test_an_unusable_hardware_path_does_not_fall_back_to_software(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv(cmaf_encoder.ENV_FFMPEG_PATH, _executable_stub(tmp_path))
        monkeypatch.setenv(ENV_XPU_FFMPEG_PATH, str(tmp_path / "nope"))
        with pytest.raises(RuntimeError, match=ENV_XPU_FFMPEG_PATH):
            cmaf_encoder.resolve_cmaf_encoder()

    def test_hardware_command_line(self):
        """VA-API upload and av1_vaapi, with the same fragmenting and GOP."""
        cmd = cmaf_encoder.build_cmaf_ffmpeg_command(
            "/opt/ffmpeg/bin/ffmpeg",
            width=832,
            height=480,
            fps=16,
            gop=8,
            hw_device="/dev/dri/renderD129",
        )
        pairs = set(zip(cmd, cmd[1:]))

        assert ("-vaapi_device", "/dev/dri/renderD129") in pairs
        assert ("-vf", "format=nv12,hwupload") in pairs
        assert ("-c:v", "av1_vaapi") in pairs
        assert ("-async_depth", "1") in pairs
        assert ("-g", "8") in pairs and ("-keyint_min", "8") in pairs
        assert ("-movflags", FRAGMENTED_MOVFLAGS) in pairs
        assert cmd[-3:] == ["-f", "mp4", "pipe:1"]
        assert "-deadline" not in cmd, "libvpx options do not apply to av1_vaapi"

    async def test_the_session_runs_the_declared_hardware_ffmpeg(
        self, monkeypatch, tmp_path
    ):
        """start() resolves the encoder; errors and the fallback codec follow it."""
        argv_file = tmp_path / "argv"
        stream = tmp_path / "stream.bin"
        stream.write_bytes(_init_segment(b"av01", b"av1C", _AV1C) + _fragments(1))
        exe = _executable_stub(
            tmp_path,
            f'printf \'%s\\n\' "$@" > "{argv_file}"\ncat > /dev/null\ncat "{stream}"',
        )
        monkeypatch.setenv(ENV_XPU_FFMPEG_PATH, exe)
        monkeypatch.setenv(ENV_XPU_VIDEO_DEVICE, "/dev/dri/renderD129")
        enc = StreamingCmafEncoder(16, 128, 96)
        # Before start() nothing is resolved yet.
        assert enc.codec_string() == cmaf_encoder.CMAF_FALLBACK_VIDEO_CODEC
        await enc.start()
        # Resolved, and no init segment read yet: the AV1 fallback applies.
        assert enc.codec_string() == cmaf_encoder.CMAF_FALLBACK_AV1_CODEC
        try:
            async for _ in enc.push(np.zeros((8, 96, 128, 3), dtype=np.uint8)):
                pass
            async for _ in enc.finish():
                pass
        finally:
            await enc.aclose()

        argv = argv_file.read_text().splitlines()
        pairs = set(zip(argv, argv[1:]))
        assert ("-vaapi_device", "/dev/dri/renderD129") in pairs
        assert ("-c:v", "av1_vaapi") in pairs

    async def test_a_failing_hardware_encode_names_the_hardware_encoder(
        self, monkeypatch, tmp_path
    ):
        exe = _executable_stub(tmp_path, 'echo "boom: no VA display" >&2\nexit 3')
        monkeypatch.setenv(ENV_XPU_FFMPEG_PATH, exe)
        enc = StreamingCmafEncoder(16, 128, 96)
        await enc.start()
        try:
            with pytest.raises(
                RuntimeError, match=r"av1_vaapi.*exit 3\).*boom: no VA display"
            ):
                async for _ in enc.push(np.zeros((8, 96, 128, 3), dtype=np.uint8)):
                    pass
                async for _ in enc.finish():
                    pass
        finally:
            await enc.aclose()


class TestCmafCommandLine:
    """The argv that makes ffmpeg emit keyframe-aligned fragmented MP4."""

    @staticmethod
    def _cmd(gop: int = 8):
        return cmaf_encoder.build_cmaf_ffmpeg_command(
            "/usr/local/bin/ffmpeg", width=832, height=480, fps=16, gop=gop
        )

    @staticmethod
    def _pairs(cmd):
        """Adjacent argv pairs, for flag/value assertions."""
        return set(zip(cmd, cmd[1:]))

    def test_raw_rgb_in_fragmented_mp4_out(self):
        """Raw RGB on stdin, software VP9, fragmented MP4 on stdout."""
        cmd = self._cmd()
        pairs = self._pairs(cmd)

        assert cmd[0] == "/usr/local/bin/ffmpeg"
        assert ("-f", "rawvideo") in pairs and ("-pix_fmt", "rgb24") in pairs
        assert ("-s", "832x480") in pairs and ("-r", "16") in pairs
        assert ("-i", "-") in pairs
        assert ("-c:v", "libvpx-vp9") in pairs
        assert ("-movflags", FRAGMENTED_MOVFLAGS) in pairs
        assert cmd[-3:] == ["-f", "mp4", "pipe:1"]
        # +faststart rewrites the file afterwards and needs a seekable output.
        assert not any("faststart" in arg for arg in cmd)

    def test_the_gop_pins_a_keyframe_at_every_segment_start(self):
        """The GOP is pinned with ``-g`` and ``-keyint_min``."""
        pairs = self._pairs(self._cmd(gop=16))
        assert ("-g", "16") in pairs
        assert ("-keyint_min", "16") in pairs

    def test_low_latency_and_browser_playable_settings(self):
        """No lookahead, and 4:2:0 so the stream is VP9 profile 0."""
        pairs = self._pairs(self._cmd())
        assert ("-deadline", "realtime") in pairs
        assert ("-lag-in-frames", "0") in pairs
        assert ("-auto-alt-ref", "0") in pairs
        # MSE support for vp09.01 (4:4:4) is not dependable.
        assert ("-pix_fmt", "yuv420p") in pairs


class TestValidateCmafFrames:
    """The encoder accepts exactly ``(T, H, W, 3) uint8``."""

    def test_accepts_the_contract(self):
        cmaf_encoder.validate_cmaf_frames(np.zeros((2, 8, 8, 3), dtype=np.uint8))

    @pytest.mark.parametrize(
        "frames, match",
        [
            ([np.zeros((8, 8, 3), dtype=np.uint8)], "np.ndarray"),
            (np.zeros((8, 8, 3), dtype=np.uint8), "shape"),
            (np.zeros((2, 8, 8, 4), dtype=np.uint8), "shape"),
            (np.zeros((2, 8, 8, 3), dtype=np.float32), "uint8"),
        ],
    )
    def test_rejects_everything_else(self, frames, match):
        with pytest.raises(ValueError, match=match):
            cmaf_encoder.validate_cmaf_frames(frames)


# ---------------------------------------------------------------------------
# Failure and cancellation, against stand-in processes
# ---------------------------------------------------------------------------


@pytest.mark.timeout(60)
class TestEncoderFailure:
    """A failed encode raises with ffmpeg's own words; it never ends quietly."""

    async def test_a_failing_ffmpeg_raises_with_its_exit_code_and_stderr(
        self, monkeypatch, tmp_path
    ):
        """Non-zero exit becomes a RuntimeError carrying stderr."""
        exe = _executable_stub(tmp_path, 'echo "boom: no such encoder" >&2\nexit 3')
        monkeypatch.setenv(cmaf_encoder.ENV_FFMPEG_PATH, exe)
        enc = StreamingCmafEncoder(16, 64, 64)
        await enc.start()
        try:
            with pytest.raises(RuntimeError, match=r"exit 3\).*boom: no such encoder"):
                async for _ in enc.push(np.zeros((8, 64, 64, 3), dtype=np.uint8)):
                    pass
                async for _ in enc.finish():
                    pass
        finally:
            await enc.aclose()

    async def test_an_ffmpeg_that_stops_reading_times_out(self, monkeypatch, tmp_path):
        """A process that never drains stdin is killed after the I/O timeout."""
        exe = _executable_stub(tmp_path, "exec sleep 60")
        monkeypatch.setenv(cmaf_encoder.ENV_FFMPEG_PATH, exe)
        enc = StreamingCmafEncoder(16, 256, 256, io_timeout_s=0.5)
        await enc.start()
        try:
            # 1.5 MB, well past any pipe buffer, so the write cannot complete.
            with pytest.raises(RuntimeError, match="accepted no input"):
                async for _ in enc.push(np.zeros((8, 256, 256, 3), dtype=np.uint8)):
                    pass
            assert enc._proc.returncode is not None
        finally:
            await enc.aclose()

    async def test_an_ffmpeg_that_produces_nothing_times_out(
        self, monkeypatch, tmp_path
    ):
        """A process that reads but never answers is killed while draining."""
        exe = _executable_stub(tmp_path, "cat > /dev/null\nexec sleep 60")
        monkeypatch.setenv(cmaf_encoder.ENV_FFMPEG_PATH, exe)
        enc = StreamingCmafEncoder(16, 64, 64, io_timeout_s=0.5)
        await enc.start()
        try:
            async for _ in enc.push(np.zeros((8, 64, 64, 3), dtype=np.uint8)):
                pass
            with pytest.raises(RuntimeError, match="produced no output"):
                async for _ in enc.finish():
                    pass
            assert enc._proc.returncode is not None
        finally:
            await enc.aclose()

    async def test_an_ffmpeg_that_stalls_at_end_of_input_times_out(
        self, monkeypatch, tmp_path
    ):
        """Input still buffered when the stream ends is not waited on forever.

        Nor does the timeout leave the event loop an error to log: the pipe
        closing afterwards must find its waiter intact.
        """
        exe = _executable_stub(tmp_path, "exec sleep 60")
        monkeypatch.setenv(cmaf_encoder.ENV_FFMPEG_PATH, exe)
        loop_errors = []
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(lambda _, context: loop_errors.append(context))
        enc = StreamingCmafEncoder(16, 64, 64, io_timeout_s=0.5)
        await enc.start()
        try:
            # 98 kB: more than a pipe holds, little enough that the rest sits in
            # the writer's buffer without blocking push().
            async for _ in enc.push(np.zeros((8, 64, 64, 3), dtype=np.uint8)):
                pass
            with pytest.raises(RuntimeError, match="while ending the CMAF stream"):
                async for _ in enc.finish():
                    pass
            assert enc._proc.returncode is not None
            await asyncio.sleep(0)  # one loop turn: callbacks of the teardown run
            assert loop_errors == []
        finally:
            loop.set_exception_handler(None)
            await enc.aclose()

    async def test_a_chunk_with_the_wrong_geometry_is_refused(
        self, monkeypatch, tmp_path
    ):
        """Every chunk of a session must match the geometry ffmpeg was told."""
        exe = _executable_stub(tmp_path, "cat > /dev/null")
        monkeypatch.setenv(cmaf_encoder.ENV_FFMPEG_PATH, exe)
        async with StreamingCmafEncoder(16, 64, 64) as enc:
            with pytest.raises(ValueError, match="does not match the CMAF session"):
                async for _ in enc.push(np.zeros((8, 32, 64, 3), dtype=np.uint8)):
                    pass


@pytest.mark.timeout(60)
class TestIncompleteOutput:
    """Exit status 0 is not enough: the stream has to be whole."""

    @staticmethod
    async def _session(exe, monkeypatch, *, push: bool = True):
        """Run one session against ``exe``; return the tags it yielded and its error."""
        monkeypatch.setenv(cmaf_encoder.ENV_FFMPEG_PATH, exe)
        enc = StreamingCmafEncoder(16, 64, 64)
        tags, error = [], None
        await enc.start()
        try:
            if push:
                async for tag, _ in enc.push(np.zeros((8, 64, 64, 3), dtype=np.uint8)):
                    tags.append(tag)
            async for tag, _ in enc.finish():
                tags.append(tag)
        except RuntimeError as err:
            error = str(err)
        finally:
            await enc.aclose()
        return tags, error

    async def test_a_whole_stream_ends_cleanly(self, monkeypatch, tmp_path):
        """The control: init plus complete fragments, exit 0, no error."""
        output = _init_segment(b"vp09", b"vpcC", _VPCC) + _fragments(2)
        tags, error = await self._session(
            _replaying_stub(tmp_path, output), monkeypatch
        )

        assert error is None
        assert tags == [CMAF_INIT_TAG, "cmaf:segment:1", "cmaf:segment:2"]

    async def test_no_output_at_all_is_a_failure(self, monkeypatch, tmp_path):
        """Frames went in and nothing came out."""
        tags, error = await self._session(_replaying_stub(tmp_path, b""), monkeypatch)

        assert tags == []
        assert "exited 0 but wrote no init segment" in error

    async def test_an_init_segment_alone_is_a_failure(self, monkeypatch, tmp_path):
        """The client must not be told that a stream with no media is complete."""
        output = _init_segment(b"vp09", b"vpcC", _VPCC)
        tags, error = await self._session(
            _replaying_stub(tmp_path, output), monkeypatch
        )

        assert tags == [CMAF_INIT_TAG]
        assert "exited 0 but wrote no media segment for 8 frame(s)" in error

    async def test_a_truncated_final_fragment_is_a_failure_and_is_not_delivered(
        self, monkeypatch, tmp_path
    ):
        """The complete fragments go out; the partial one does not."""
        output = _init_segment(b"vp09", b"vpcC", _VPCC) + _fragments(2)[:-4]
        tags, error = await self._session(
            _replaying_stub(tmp_path, output), monkeypatch
        )

        assert tags == [CMAF_INIT_TAG, "cmaf:segment:1"]
        assert "exited 0 but its output ends inside a box" in error

    async def test_a_non_zero_exit_is_reported_ahead_of_a_truncated_stream(
        self, monkeypatch, tmp_path
    ):
        """ffmpeg's own status and stderr say more than "the output is short"."""
        output = _init_segment(b"vp09", b"vpcC", _VPCC) + _fragments(2)[:-4]
        tags, error = await self._session(
            _replaying_stub(tmp_path, output, status=3), monkeypatch
        )

        assert tags == [CMAF_INIT_TAG, "cmaf:segment:1"]
        assert "(exit 3)" in error

    @pytest.mark.parametrize("push", [True, False], ids=["in push", "in finish"])
    async def test_a_stdout_read_error_is_not_an_end_of_stream(
        self, monkeypatch, tmp_path, push
    ):
        """A reader that stopped on an error must not pass for a finished stream."""
        output = _init_segment(b"vp09", b"vpcC", _VPCC) + _fragments(1)
        real_start = StreamingCmafEncoder.start

        async def start_with_a_broken_stdout(self):
            await real_start(self)
            self._proc.stdout.set_exception(OSError("pipe went away"))
            await asyncio.sleep(0)  # one loop turn: the reader hits the error

        monkeypatch.setattr(StreamingCmafEncoder, "start", start_with_a_broken_stdout)
        _, error = await self._session(
            _replaying_stub(tmp_path, output), monkeypatch, push=push
        )

        assert "reading its output failed" in error
        assert "pipe went away" in error


@pytest.mark.timeout(60)
class TestEncoderCancellation:
    """A client that goes away must not leave an ffmpeg behind."""

    async def test_aclose_mid_stream_reaps_the_process(self, monkeypatch, tmp_path):
        """aclose() kills a live process, stops the pump tasks, and is repeatable."""
        exe = _executable_stub(tmp_path, "cat > /dev/null\nexec sleep 60")
        monkeypatch.setenv(cmaf_encoder.ENV_FFMPEG_PATH, exe)
        enc = StreamingCmafEncoder(16, 64, 64)
        await enc.start()
        async for _ in enc.push(np.zeros((8, 64, 64, 3), dtype=np.uint8)):
            pass
        proc = enc._proc
        assert proc.returncode is None

        await enc.aclose()

        assert proc.returncode is not None
        assert enc._stdout_task is None and enc._stderr_task is None
        await enc.aclose()  # second call is a no-op

    async def test_a_cancelled_consumer_reaps_the_process(self, monkeypatch, tmp_path):
        """Cancelling the task that drives the session still reaps ffmpeg."""
        exe = _executable_stub(tmp_path, "cat > /dev/null\nexec sleep 60")
        monkeypatch.setenv(cmaf_encoder.ENV_FFMPEG_PATH, exe)
        started = asyncio.Event()
        holder = {}

        async def session():
            async with StreamingCmafEncoder(16, 64, 64) as enc:
                holder["proc"] = enc._proc
                async for _ in enc.push(np.zeros((8, 64, 64, 3), dtype=np.uint8)):
                    pass
                started.set()
                async for _ in enc.finish():  # blocks: the stub never answers
                    pass

        task = asyncio.create_task(session())
        await asyncio.wait_for(started.wait(), timeout=10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert holder["proc"].returncode is not None

    async def test_aclose_reaps_a_wrapper_script_and_its_child(
        self, monkeypatch, tmp_path
    ):
        """DYN_FFMPEG_PATH may be a wrapper: its child must die with it.

        The child inherits the session's stdout and stderr, so killing only
        the wrapper would leave it running and the pipes open.
        """
        pid_file = tmp_path / "child.pid"
        exe = _executable_stub(tmp_path, f'sleep 60 &\necho $! > "{pid_file}"\nwait')
        monkeypatch.setenv(cmaf_encoder.ENV_FFMPEG_PATH, exe)
        enc = StreamingCmafEncoder(16, 64, 64)
        await enc.start()
        try:
            child = await _read_pid(pid_file)
            assert _alive(child)

            await asyncio.wait_for(enc.aclose(), timeout=30)

            assert enc._proc.returncode is not None
            for _ in range(100):
                if not _alive(child):
                    break
                await asyncio.sleep(0.05)
            assert not _alive(child)
        finally:
            await enc.aclose()

    async def test_aclose_reaps_the_child_of_a_wrapper_that_already_exited(
        self, monkeypatch, tmp_path
    ):
        """A wrapper that exits first leaves its child in the session's group.

        The wrapper's exit status says nothing about the child, which still
        holds the pipes: the group has to be killed all the same.
        """
        pid_file = tmp_path / "child.pid"
        exe = _executable_stub(tmp_path, f'sleep 60 &\necho $! > "{pid_file}"\nexit 0')
        monkeypatch.setenv(cmaf_encoder.ENV_FFMPEG_PATH, exe)
        enc = StreamingCmafEncoder(16, 64, 64)
        await enc.start()
        try:
            child = await _read_pid(pid_file)
            assert await _until(lambda: enc._proc.returncode is not None)
            assert _alive(child)

            await asyncio.wait_for(enc.aclose(), timeout=30)

            assert await _until(lambda: not _alive(child))
            await enc.aclose()  # second call is a no-op
        finally:
            await enc.aclose()

    async def test_a_recycled_leader_pid_is_not_signalled(self, monkeypatch, tmp_path):
        """Once the group is empty its id can be reused, by a process that is not ours."""
        monkeypatch.setenv(cmaf_encoder.ENV_FFMPEG_PATH, _executable_stub(tmp_path))
        enc = StreamingCmafEncoder(16, 64, 64)
        await enc.start()
        try:
            assert await _until(lambda: enc._proc.returncode is not None)
            signalled = []
            # A process answers at the leader's old pid: the number was reused.
            monkeypatch.setattr(cmaf_encoder.os, "kill", lambda pid, sig: None)
            monkeypatch.setattr(
                cmaf_encoder.os, "killpg", lambda pgid, sig: signalled.append(pgid)
            )

            await enc.aclose()

            assert signalled == []
        finally:
            await enc.aclose()


async def _until(condition, *, tries: int = 200) -> bool:
    """Poll ``condition`` until it holds; False if it never does."""
    for _ in range(tries):
        if condition():
            return True
        await asyncio.sleep(0.05)
    return False


async def _read_pid(pid_file) -> int:
    """The pid a stub wrote to ``pid_file``, once it has written it."""
    for _ in range(200):
        text = pid_file.read_text().strip() if pid_file.exists() else ""
        if text:
            return int(text)
        await asyncio.sleep(0.05)
    raise AssertionError(f"the stub never wrote {pid_file}")


def _alive(pid: int) -> bool:
    """Whether ``pid`` is running; a zombie awaiting its parent does not count."""
    try:
        with open(f"/proc/{pid}/stat") as stat:
            return stat.read().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        if os.path.isdir("/proc/self"):
            return False
    try:  # no procfs
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


# ---------------------------------------------------------------------------
# Real ffmpeg
# ---------------------------------------------------------------------------


def _find_ffmpeg() -> Optional[str]:
    """An ffmpeg that carries libvpx-vp9, or ``None``."""
    candidates = (
        os.environ.get(cmaf_encoder.ENV_FFMPEG_PATH),
        cmaf_encoder.DEFAULT_FFMPEG_PATH,
        os.environ.get("IMAGEIO_FFMPEG_EXE"),
        shutil.which("ffmpeg"),
    )
    for candidate in candidates:
        if not candidate or not os.access(candidate, os.X_OK):
            continue
        try:
            encoders = subprocess.run(
                [candidate, "-hide_banner", "-encoders"],
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        if cmaf_encoder.CMAF_VIDEO_ENCODER in encoders:
            return candidate
    return None


@pytest.fixture(scope="module")
def ffmpeg_exe() -> str:
    exe = _find_ffmpeg()
    if exe is None:
        pytest.skip(f"no ffmpeg with {cmaf_encoder.CMAF_VIDEO_ENCODER} available")
    return exe


@pytest.fixture
def real_ffmpeg(ffmpeg_exe, monkeypatch) -> str:
    """Point the encoder at the ffmpeg found for this run."""
    monkeypatch.setenv(cmaf_encoder.ENV_FFMPEG_PATH, ffmpeg_exe)
    return ffmpeg_exe


def _clip(n: int, width: int = 96, height: int = 64) -> np.ndarray:
    """Deterministic ``(n, height, width, 3) uint8`` clip: gradients plus motion."""
    yy, xx = np.mgrid[0:height, 0:width]
    out = np.empty((n, height, width, 3), dtype=np.uint8)
    for t in range(n):
        out[t, ..., 0] = xx * 255 // (width - 1)
        out[t, ..., 1] = yy * 255 // (height - 1)
        out[t, ..., 2] = (4 * t) % 256
        x0 = (3 * t) % (width - 16)
        out[t, 24:40, x0 : x0 + 16] = 255
    return out


async def _run_session(frames: np.ndarray, *, fps: int = 16, gop: int = 8, pace=0.0):
    """Encode ``frames``; return the encoder and ``(tag, payload, phase)`` items.

    ``phase`` is ``"push"`` for a fragment delivered while input was still being
    written and ``"finish"`` for one delivered after the input was closed.
    """
    _, height, width, _ = frames.shape
    enc = StreamingCmafEncoder(fps, width, height, gop_frames=gop)
    await enc.start()
    items = []
    try:
        for chunk in iter_cmaf_chunks(frames, gop):
            async for tag, payload in enc.push(chunk):
                items.append((tag, payload, "push"))
            if pace:
                await asyncio.sleep(pace)
        async for tag, payload in enc.finish():
            items.append((tag, payload, "finish"))
    finally:
        await enc.aclose()
    return enc, items


def _decode(ffmpeg: str, stream: bytes, tmp_path, width: int, height: int):
    """Decode a complete stream with ffmpeg; return frames and the stream line."""
    path = tmp_path / "stream.mp4"
    path.write_bytes(stream)
    proc = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-i",
            str(path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        capture_output=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")[-2000:]
    frames = np.frombuffer(proc.stdout, dtype=np.uint8).reshape(-1, height, width, 3)
    info = next(
        (
            line
            for line in proc.stderr.decode(errors="replace").splitlines()
            if "Video:" in line
        ),
        "",
    )
    return frames, info


def _child(buf: bytes, start: int, end: int, *path: bytes):
    """``(payload_start, box_end)`` of the box at ``path`` below ``[start, end)``."""
    scope = (start, end)
    for name in path:
        scope = cmaf_encoder._find_child(buf, scope[0], scope[1], name)
        assert scope is not None, f"no {name!r} box"
    return scope


def _fragment_timing(segment: bytes):
    """``(sequence_number, base_media_decode_time, sample_count)`` of one fragment."""
    moof = _child(segment, 0, len(segment), b"moof")
    mfhd = _child(segment, *moof, b"mfhd")
    sequence = int.from_bytes(segment[mfhd[0] + 4 : mfhd[0] + 8], "big")
    traf = _child(segment, *moof, b"traf")
    tfdt = _child(segment, *traf, b"tfdt")
    version = segment[tfdt[0]]
    width = 8 if version == 1 else 4
    decode_time = int.from_bytes(segment[tfdt[0] + 4 : tfdt[0] + 4 + width], "big")
    trun = _child(segment, *traf, b"trun")
    samples = int.from_bytes(segment[trun[0] + 4 : trun[0] + 8], "big")
    return sequence, decode_time, samples


def _timescale(init: bytes) -> int:
    """Media timescale from the init segment's ``mdhd``."""
    mdhd = _child(init, 0, len(init), b"moov", b"trak", b"mdia", b"mdhd")
    version = init[mdhd[0]]
    offset = mdhd[0] + 4 + (16 if version == 1 else 8)
    return int.from_bytes(init[offset : offset + 4], "big")


@pytest.mark.timeout(180)
class TestStreamingCmafEncoderReal:
    """What a player depends on, checked against a real encode."""

    async def test_stream_is_one_init_then_numbered_segments(self, real_ffmpeg):
        """One init segment first, then media segments numbered from one."""
        _, items = await _run_session(_clip(20))
        tags = [tag for tag, _, _ in items]

        assert tags == [CMAF_INIT_TAG] + [
            f"{CMAF_SEGMENT_TAG_PREFIX}{i}" for i in (1, 2, 3)
        ]
        init = items[0][1]
        assert init[4:8] == b"ftyp" and b"moov" in init
        for _, payload, _ in items[1:]:
            assert payload[4:8] == b"moof", "a media segment starts with its moof"

    async def test_fragments_are_playable_and_carry_every_frame(
        self, real_ffmpeg, tmp_path
    ):
        """Init plus segments, concatenated, decode to the frames that went in."""
        source = _clip(20)
        _, items = await _run_session(source)
        stream = b"".join(payload for _, payload, _ in items)

        decoded, info = _decode(real_ffmpeg, stream, tmp_path, 96, 64)

        assert "vp9" in info and "yuv420p" in info, info
        assert decoded.shape == source.shape
        mse = np.mean((decoded.astype(np.float64) - source.astype(np.float64)) ** 2)
        psnr = 10 * np.log10(255.0**2 / mse)
        assert psnr >= 25.0, f"decoded picture does not match the input: {psnr:.1f} dB"

    async def test_each_segment_is_decodable_once_the_init_is_known(
        self, real_ffmpeg, tmp_path
    ):
        """init + segment k alone decodes: every segment starts with a keyframe."""
        _, items = await _run_session(_clip(20))
        init = items[0][1]

        for index, (_, payload, _) in enumerate(items[1:], start=1):
            decoded, _ = _decode(real_ffmpeg, init + payload, tmp_path, 96, 64)
            assert len(decoded) == (8, 8, 4)[index - 1], f"segment {index}"

    async def test_the_timeline_is_continuous_across_segments(self, real_ffmpeg):
        """Sequence numbers count up by one and decode times never restart."""
        fps, gop = 16, 8
        _, items = await _run_session(_clip(20), fps=fps, gop=gop)
        timescale = _timescale(items[0][1])
        timing = [_fragment_timing(payload) for _, payload, _ in items[1:]]

        sequences = [seq for seq, _, _ in timing]
        assert sequences == list(range(sequences[0], sequences[0] + len(timing)))

        frames_before = 0
        for index, (_, decode_time, samples) in enumerate(timing, start=1):
            # tfdt is in timescale units: frames already delivered, over fps.
            assert decode_time * fps == frames_before * timescale, f"segment {index}"
            frames_before += samples
        assert frames_before == 20

    async def test_the_final_segment_carries_the_remainder(self, real_ffmpeg):
        """17 frames at 8 per segment end in a single-frame segment."""
        _, items = await _run_session(_clip(17))
        samples = [_fragment_timing(payload)[2] for _, payload, _ in items[1:]]
        assert samples == [8, 8, 1]

    async def test_fragments_arrive_while_input_is_still_being_written(
        self, real_ffmpeg
    ):
        """A segment is delivered before the session is finished.

        Frames are paced to stand in for an encode that outlasts a single
        write. A fragment is complete once the next segment's keyframe has
        been written, so an early one must surface from push(), not only from
        finish() after the input is closed.
        """
        _, items = await _run_session(_clip(48), pace=0.3)
        early = [tag for tag, _, phase in items if phase == "push"]

        assert CMAF_INIT_TAG in early, "the init segment waited for end of input"
        assert any(
            tag.startswith(CMAF_SEGMENT_TAG_PREFIX) for tag in early
        ), "no media segment was delivered before the input ended"
        assert len([t for t, _, _ in items if t != CMAF_INIT_TAG]) == 6

    async def test_the_codec_string_describes_the_stream(self, real_ffmpeg):
        """The codec string is read from the init segment the encode produced."""
        enc, items = await _run_session(_clip(8))
        codec = enc.codec_string()

        assert codec == cmaf_encoder.codec_string_from_init(items[0][1])
        # VP9 profile 0, 8-bit: the 4:2:0 stream every MSE implementation plays.
        assert codec.startswith("vp09.00.") and codec.endswith(".08"), codec


# ---------------------------------------------------------------------------
# Real hardware (VA-API AV1)
# ---------------------------------------------------------------------------

# Hardware encoders have a minimum frame size (128x96 on an Arc Pro B60).
_HW_WIDTH, _HW_HEIGHT = 160, 96


@pytest.mark.xpu_1
@pytest.mark.timeout(180)
class TestStreamingCmafEncoderHardwareReal:
    """The same player-facing properties, on a real VA-API AV1 encode.

    Runs only where ``DYN_XPU_FFMPEG_PATH`` is set; see ``hw_video_encoder``.
    """

    async def test_fragments_are_playable_and_carry_every_frame(
        self, hw_video_encoder, tmp_path
    ):
        ffmpeg, _ = hw_video_encoder
        source = _clip(20, _HW_WIDTH, _HW_HEIGHT)
        _, items = await _run_session(source)
        tags = [tag for tag, _, _ in items]
        stream = b"".join(payload for _, payload, _ in items)

        assert tags == [CMAF_INIT_TAG] + [
            f"{CMAF_SEGMENT_TAG_PREFIX}{i}" for i in (1, 2, 3)
        ]
        decoded, info = _decode(ffmpeg, stream, tmp_path, _HW_WIDTH, _HW_HEIGHT)
        assert "av1" in info and "yuv420p" in info, info
        assert decoded.shape == source.shape
        mse = np.mean((decoded.astype(np.float64) - source.astype(np.float64)) ** 2)
        psnr = 10 * np.log10(255.0**2 / mse)
        assert psnr >= 25.0, f"decoded picture does not match the input: {psnr:.1f} dB"

    async def test_each_segment_is_decodable_once_the_init_is_known(
        self, hw_video_encoder, tmp_path
    ):
        ffmpeg, _ = hw_video_encoder
        _, items = await _run_session(_clip(20, _HW_WIDTH, _HW_HEIGHT))
        init = items[0][1]

        for index, (_, payload, _) in enumerate(items[1:], start=1):
            decoded, _ = _decode(
                ffmpeg, init + payload, tmp_path, _HW_WIDTH, _HW_HEIGHT
            )
            assert len(decoded) == (8, 8, 4)[index - 1], f"segment {index}"

    async def test_the_timeline_is_continuous_and_the_tail_is_partial(
        self, hw_video_encoder
    ):
        fps = 16
        _, items = await _run_session(_clip(17, _HW_WIDTH, _HW_HEIGHT), fps=fps)
        timescale = _timescale(items[0][1])
        timing = [_fragment_timing(payload) for _, payload, _ in items[1:]]

        sequences = [seq for seq, _, _ in timing]
        assert sequences == list(range(sequences[0], sequences[0] + len(timing)))
        assert [samples for _, _, samples in timing] == [8, 8, 1]
        frames_before = 0
        for index, (_, decode_time, samples) in enumerate(timing, start=1):
            assert decode_time * fps == frames_before * timescale, f"segment {index}"
            frames_before += samples

    async def test_fragments_arrive_while_input_is_still_being_written(
        self, hw_video_encoder
    ):
        _, items = await _run_session(_clip(48, _HW_WIDTH, _HW_HEIGHT), pace=0.3)
        early = [tag for tag, _, phase in items if phase == "push"]

        assert CMAF_INIT_TAG in early, "the init segment waited for end of input"
        assert any(
            tag.startswith(CMAF_SEGMENT_TAG_PREFIX) for tag in early
        ), "no media segment was delivered before the input ended"
        assert len([t for t, _, _ in items if t != CMAF_INIT_TAG]) == 6

    async def test_the_codec_string_describes_the_stream(self, hw_video_encoder):
        enc, items = await _run_session(_clip(8, _HW_WIDTH, _HW_HEIGHT))
        codec = enc.codec_string()

        assert codec == cmaf_encoder.codec_string_from_init(items[0][1])
        # AV1 Main profile, 8-bit.
        assert codec.startswith("av01.0.") and codec.endswith(".08"), codec
