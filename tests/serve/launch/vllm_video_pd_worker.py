# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the native worker with fail-closed video P/D regression guards."""

import argparse
import runpy
from contextlib import ExitStack
from unittest.mock import patch

import torch
from vllm.renderers import BaseRenderer

from dynamo.common.backend.health_check import is_probe
from dynamo.common.constants import DisaggregationMode
from dynamo.common.multimodal.video_loader import VideoLoader
from dynamo.vllm.multimodal_utils.request_processor import (
    VllmMultimodalRequestProcessor,
)


def require_video_inputs(prepared) -> None:
    """Reject text-only, metadata-only, and raw-reload decode paths."""
    prompt = prepared.pre_rendered_prompt
    if prompt is None or prompt.get("type") != "multimodal":
        raise RuntimeError("Video P/D must submit native processed inputs on decode")
    items = prompt["mm_kwargs"].get("video", [])
    if len(items) != 1 or items[0] is None:
        raise RuntimeError("Video P/D must preserve the video item")
    field = items[0].get("pixel_values_videos")
    if field is None or not torch.is_tensor(field.data) or field.data.numel() == 0:
        raise RuntimeError("Video P/D must retain real pixels for recomputation")


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--disaggregation-mode")
    args, _ = parser.parse_known_args()
    original_prepare = VllmMultimodalRequestProcessor.prepare_input

    async def prepare(self, request, request_id, context, mode):
        prepared = await original_prepare(self, request, request_id, context, mode)
        if mode == DisaggregationMode.DECODE and not is_probe(request):
            require_video_inputs(prepared)
        return prepared

    with ExitStack() as stack:
        if args.disaggregation_mode == "decode":
            stack.enter_context(
                patch.object(
                    VllmMultimodalRequestProcessor, "prepare_input", new=prepare
                )
            )
            stack.enter_context(
                patch.object(
                    VideoLoader,
                    "load_video_batch",
                    side_effect=RuntimeError("Decode must not reload video"),
                )
            )
            stack.enter_context(
                patch.object(
                    BaseRenderer,
                    "_process_multimodal",
                    side_effect=RuntimeError("Decode must not reprocess video"),
                )
            )
        runpy.run_module("dynamo.vllm", run_name="__main__")


if __name__ == "__main__":
    main()
