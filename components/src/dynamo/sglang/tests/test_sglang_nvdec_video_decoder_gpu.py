# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU test: SGLang's NVDEC decoder returns the frames that were encoded.

Decodes the committed triangle fixtures through ``NvdecVideoDecoder`` at the
indices SGLang's ``preprocess_video`` requests for a 10-frame clip, and compares
every frame with the raw frame that ``make_triangle_fixture.py`` generated.

The H.265 fixture is HEVC Range Extensions, intra only. Through the E/PD serve
test on an L4, the model said that this clip has no video (DYN-5446), while the
prompt still had the full count of video tokens. A comparison with ground truth
tells a decoder fault (wrong pixels) apart from a model-side difference (right
pixels). Equality with a second NVDEC decode of the same clip cannot detect a
fault that both decodes share.

The decode runs in a subprocess so the pytest parent stays free of CUDA, as in
``common/tests/multimodal/test_nvdec_decoder_gpu.py``.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.post_merge,
    pytest.mark.gpu_1,
    pytest.mark.sglang,
    pytest.mark.multimodal,
]

_MEDIA_DIR = (
    Path(__file__).resolve().parents[5] / "lib" / "llm" / "tests" / "data" / "media"
)
_FIXTURES = {"h264": "triangle_240p_10_h264.mp4", "hevc": "triangle_240p_10_h265.mp4"}
# np.unique(np.linspace(0, 9, 4)) -- preprocess_video's pick for this clip.
_INDICES = [0, 3, 6, 9]
# The fixture docstring measures 1.59 to 1.68 mean absolute difference against
# the source; a blank, stale, or wrong frame differs far more than this.
_MAX_MEAN_ABS_DIFF = 8.0
_WORKER_FLAG = "--nvdec-worker"
_WORKER_SKIP = 77
_WORKER_TIMEOUT_S = 60


def _readable_output(output: str | bytes | None) -> str:
    if isinstance(output, bytes):
        return output.decode(errors="replace")
    return output or ""


def _load_frame_generator():
    """``make_triangle_fixture._frame``, loaded by path from the fixture dir."""
    spec = importlib.util.spec_from_file_location(
        "make_triangle_fixture", _MEDIA_DIR / "make_triangle_fixture.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._frame


def _decode_and_compare(codec: str) -> int:
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

    from dynamo.sglang.request_handlers.multimodal.nvdec_video_decoder import (
        NvdecVideoDecoder,
    )

    make_frame = _load_frame_generator()
    with NvdecVideoDecoder(data) as decoder:
        assert len(decoder) == 10
        out = decoder.get_frames_as_tensor(_INDICES)

    assert tuple(out.shape) == (len(_INDICES), 240, 320, 3), out.shape
    frames = out.numpy()
    for pos, index in enumerate(_INDICES):
        expected = make_frame(index).astype(np.int16)
        diff = float(np.abs(frames[pos].astype(np.int16) - expected).mean())
        print(f"{codec} frame {index}: mean abs diff {diff:.2f}", file=sys.stderr)
        assert diff < _MAX_MEAN_ABS_DIFF, (
            f"{codec} frame {index} does not match the encoded source "
            f"(mean abs diff {diff:.2f} >= {_MAX_MEAN_ABS_DIFF})"
        )
    # The triangle moves, so the first and last sampled frames must differ.
    assert not np.array_equal(frames[0], frames[-1]), f"{codec}: frames repeat"
    return 0


def _run_worker() -> int:
    for codec in sorted(_FIXTURES):
        print(f"Validating SGLang NVDEC {codec}", file=sys.stderr, flush=True)
        status = _decode_and_compare(codec)
        if status != 0:
            return status
    return 0


@pytest.mark.timeout(_WORKER_TIMEOUT_S + 15)
def test_nvdec_video_decoder_frames_match_source():
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
    # Keep the per-frame differences in the report, so an L4 run shows them.
    print(result.stderr)


if __name__ == "__main__":
    if sys.argv[1:] != [_WORKER_FLAG]:
        sys.exit(f"usage: {sys.argv[0]} {_WORKER_FLAG}")
    sys.exit(_run_worker())
