# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the rendered launcher example without launching an inference engine."""

import json
import shlex
import subprocess
from pathlib import Path

import pytest

try:
    import torch
    from transformers import Qwen3VLVideoProcessor
except ImportError:
    pytest.skip(
        "Native Qwen3-VL processor unavailable in this backend image",
        allow_module_level=True,
    )

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.vllm,
    pytest.mark.multimodal,
    pytest.mark.timeout(30),
]


@pytest.fixture(params=["default", "environment", "cli"])
def video_example(request, monkeypatch):
    model = "test/Selected-VLM"
    monkeypatch.setenv("DYN_MODEL_NAME", model)
    args = []
    if request.param == "default":
        monkeypatch.delenv("DYN_MODEL_NAME")
        # Only compare the printed name; no model weights are loaded.
        model = "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
    elif request.param == "cli":
        model = "test/CLI-VLM"
        args = ["--model", model]
    script = (
        Path(__file__).parents[2] / "examples/backends/vllm/launch/agg_multimodal.sh"
    )
    # Stop at the footer, before any engine starts. Keep real Bash argument
    # parsing and heredoc expansion; never run the launcher's process-group trap.
    result = subprocess.run(
        [
            "bash",
            "-c",
            """
python() { echo "Unexpected engine launch" >&2; return 1; }
source() {
    builtin source "$@"
    if [[ "$1" == */launch_utils.sh ]]; then
        print_curl_footer() { cat; exit 0; }
        dynamo_exit_trap() { :; }
    fi
}
source "$@"
""",
            "example-test",
            str(script),
            *args,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert not result.stderr, result.stderr
    tokens = shlex.split(result.stdout, comments=True)
    bodies = [
        json.loads(tokens[i + 1]) for i, token in enumerate(tokens) if token == "-d"
    ]
    assert len(bodies) == 2
    assert all(body["model"] == model for body in bodies)
    return bodies[1]


@pytest.mark.parametrize("height,width", [(720, 1280), (1280, 720)])
def test_video_example_fits_default_context_with_native_processor(
    video_example, height, width
):
    """Native Qwen3-VL resizing must leave room for text and the completion."""
    num_frames = video_example["media_io_kwargs"]["video"]["num_frames"]
    assert num_frames == 4
    processor = Qwen3VLVideoProcessor()
    processed = processor(
        videos=[torch.zeros((num_frames, 3, height, width), dtype=torch.uint8)],
        do_sample_frames=False,
        return_tensors="pt",
        **video_example["mm_processor_kwargs"]["videos_kwargs"],
    )
    grid = processed["video_grid_thw"][0]
    visual_tokens = int(grid.prod()) // processor.merge_size**2
    assert 0 < visual_tokens <= 512
    # Reserve 256 tokens for chat-template text, timestamps and special tokens.
    assert visual_tokens + 256 + video_example["max_tokens"] <= 4096
    assert processed["pixel_values_videos"].numel() > 0
