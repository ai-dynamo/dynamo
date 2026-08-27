# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU integration test for NVDEC decode (needs a GPU + PyNvVideoCodec).

Decodes the committed H.264/H.265 fixtures through the real ``nvdec_decoder``
and asserts the frame contract, mirroring the hardware validation done on
gpu-ts. Skips cleanly where NVDEC is unavailable, so it is a no-op on CPU lanes
and images without PyNvVideoCodec.

The clips are committed fixtures rather than synthesized at run time on purpose:
generating them needs an H.264/H.265 *encoder*, which the codec-compliant image
deliberately does not ship (the in-tree ffmpeg is VP9-only, and imageio-ffmpeg
is installed with --no-binary). Synthesizing here made the test unable to run in
the very image it is meant to validate.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.post_merge,
    pytest.mark.gpu_1,
    pytest.mark.vllm,
]


_MEDIA_DIR = (
    Path(__file__).resolve().parents[6] / "lib" / "llm" / "tests" / "data" / "media"
)

# Committed fixtures, re-encoded from the VP9 240p_10.mp4 clip.
_FIXTURES = {"h264": "240p_10_h264.mp4", "hevc": "240p_10_h265.mp4"}
_WORKER_FLAG = "--nvdec-worker"
_WORKER_SKIP = 77
_WORKER_TIMEOUT_S = 60


def _readable_output(output: str | bytes | None) -> str:
    if isinstance(output, bytes):
        return output.decode(errors="replace")
    return output or ""


def _decode_and_assert(codec: str) -> int:
    """Decode one fixture; local imports keep parent pytest free of CUDA/NVDEC."""
    import numpy as np

    from dynamo.common.multimodal import nvdec_decoder as nd

    if not nd.nvdec_available():
        print(
            "PyNvVideoCodec/NVDEC not available (needs the video capability)",
            file=sys.stderr,
        )
        return _WORKER_SKIP

    path = _MEDIA_DIR / _FIXTURES[codec]
    if not path.is_file():
        print(f"fixture not available: {path}", file=sys.stderr)
        return _WORKER_SKIP
    data = path.read_bytes()
    # Guard against an unresolved Git LFS pointer masquerading as the clip.
    if data[:7] == b"version":
        print(f"fixture is an unresolved LFS pointer: {path}", file=sys.stderr)
        return _WORKER_SKIP

    # Sanity: the probe classifies the clip as an NVDEC-routed codec.
    assert nd.probe_video_codec(data) in nd.HW_ROUTED_CODECS

    frames, metadata = nd.decode_video_nvdec(data, num_frames=8)

    assert frames.ndim == 4 and frames.shape[-1] == 3  # (T, H, W, 3)
    assert frames.dtype == np.uint8
    assert frames.flags["C_CONTIGUOUS"]
    assert frames.shape[0] == 8
    assert frames[:, :, :, :].max() > 0  # real pixels, not a black clip
    assert metadata["total_num_frames"] >= 8
    assert len(metadata["frames_indices"]) == 8
    return 0


def _run_worker() -> int:
    for codec in sorted(_FIXTURES):
        print(f"Validating NVDEC {codec}", file=sys.stderr, flush=True)
        status = _decode_and_assert(codec)
        if status != 0:
            return status
    return 0


@pytest.mark.timeout(_WORKER_TIMEOUT_S + 15)
def test_nvdec_decodes_real_clips():
    try:
        result = subprocess.run(
            [sys.executable, __file__, _WORKER_FLAG],
            capture_output=True,
            text=True,
            timeout=_WORKER_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            f"NVDEC worker timed out after {_WORKER_TIMEOUT_S}s\n"
            f"--- stdout ---\n{_readable_output(exc.stdout)}\n"
            f"--- stderr ---\n{_readable_output(exc.stderr)}"
        )

    if result.returncode == _WORKER_SKIP:
        pytest.skip(result.stderr.strip() or "NVDEC worker unavailable")
    if result.returncode != 0:
        pytest.fail(
            f"NVDEC worker failed (rc={result.returncode})\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )


if __name__ == "__main__":
    if sys.argv[1:] != [_WORKER_FLAG]:
        sys.exit(f"usage: {sys.argv[0]} {_WORKER_FLAG}")
    sys.exit(_run_worker())
